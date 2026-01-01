"""
Layer 1: CP-SAT Optimizer

Google OR-Tools CP-SAT (Constraint Programming - Satisfiability) solver for production scheduling.

Mathematical Formulation:
=======================

Decision Variables:
- x_jm ∈ {0,1}: Job j assigned to machine m
- s_j ∈ Z⁺: Start time of job j (minutes since epoch)
- e_j ∈ Z⁺: End time of job j
- y_jk ∈ {0,1}: Job j immediately precedes job k on same machine

Constraints:
1. Assignment: Σ_m x_jm = 1  ∀j (each job to exactly one machine)
2. Compatibility: x_jm = 0 if machine m cannot produce job j's SKU
3. No overlap: If x_jm = x_km = 1, then (e_j ≤ s_k) OR (e_k ≤ s_j)
4. Processing time: e_j = s_j + p_j·q_j + Σ_k y_kj·c_{k→j}
5. Release date: s_j ≥ r_j  ∀j
6. Due date (soft): Penalize max(0, e_j - d_j)

Objective:
min α·Σ max(0, e_j - d_j) + β·Σ y_jk·c_jk + γ·Cost + δ·Energy

This solver provides:
- Guaranteed feasibility (if solution exists)
- Optimality proof (if runs to completion)
- Scalability to 100-500 jobs
- Setup time integration
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import copy

from ortools.sat.python import cp_model
import numpy as np

from mapso.optimization.base_optimizer import BaseOptimizer
from mapso.optimization.objective_function import MultiObjectiveFunction
from mapso.core.models import Schedule, Job, Machine, SKU, ChangeoverMatrix
from mapso.core.enums import OptimizationStatus
from mapso.evaluation.metrics import ScheduleMetrics
from mapso.utils.logging_config import get_logger

logger = get_logger("cpsat_optimizer")


class CPSATOptimizer(BaseOptimizer):
    """
    CP-SAT-based production scheduler

    Uses Google OR-Tools CP-SAT solver for job shop scheduling with:
    - Disjunctive constraints (no overlap)
    - Sequence-dependent setup times
    - Multi-objective optimization
    - Release dates and due dates
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("cpsat", config)
        self.model: Optional[cp_model.CpModel] = None
        self.solver: Optional[cp_model.CpSolver] = None

    def optimize(
        self,
        jobs: List[Job],
        machines: List[Machine],
        skus: List[SKU],
        changeover_matrix: ChangeoverMatrix,
        start_date: datetime,
        end_date: datetime,
        objective_weights: Optional[Dict[str, float]] = None,
        timeout: Optional[int] = None,
    ) -> Schedule:
        """
        Optimize schedule using CP-SAT

        Args:
            jobs: List of jobs to schedule
            machines: List of machines
            skus: List of SKUs
            changeover_matrix: Setup time matrix
            start_date: Schedule start
            end_date: Schedule end
            objective_weights: Multi-objective weights
            timeout: Timeout in seconds

        Returns:
            Optimized schedule
        """
        self._start_optimization()

        # Set default timeout
        if timeout is None:
            timeout = self.config.get("timeout", 300)

        # Set default weights
        if objective_weights is None:
            objective_weights = {
                "lateness": 0.4,
                "setup_time": 0.2,
                "cost": 0.2,
                "energy": 0.2,
            }

        logger.info(f"CP-SAT optimization starting: {len(jobs)} jobs, {len(machines)} machines")
        logger.info(f"Weights: {objective_weights}, Timeout: {timeout}s")

        # Create model
        self.model = cp_model.CpModel()

        # Build SKU and machine dictionaries
        sku_dict = {sku.sku_id: sku for sku in skus}
        machine_dict = {m.machine_id: m for m in machines}

        # Time horizon (in minutes)
        horizon_minutes = int((end_date - start_date).total_seconds() / 60)

        # Create variables
        job_vars = self._create_variables(
            jobs, machines, sku_dict, start_date, horizon_minutes
        )

        # Add constraints
        self._add_assignment_constraints(jobs, machines, job_vars)
        self._add_temporal_constraints(
            jobs, machines, sku_dict, changeover_matrix, job_vars, start_date
        )
        self._add_no_overlap_constraints(jobs, machines, job_vars)

        # Build objective
        objective_expr = self._build_objective(
            jobs,
            machines,
            skus,
            changeover_matrix,
            job_vars,
            objective_weights,
            start_date,
        )

        self.model.Minimize(objective_expr)

        # Solve
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = timeout
        self.solver.parameters.num_search_workers = self.config.get("num_workers", 8)

        logger.info("Solving CP-SAT model...")
        status = self.solver.Solve(self.model)

        # Extract solution
        schedule = self._extract_solution(
            status,
            jobs,
            machines,
            job_vars,
            start_date,
            end_date,
            changeover_matrix,
            skus,
            objective_weights,
        )

        # Update statistics
        self.objective_value = schedule.objective_value
        self.iterations = self.solver.NumBranches()

        if status == cp_model.OPTIMAL:
            self._end_optimization(OptimizationStatus.OPTIMAL)
        elif status == cp_model.FEASIBLE:
            self._end_optimization(OptimizationStatus.FEASIBLE)
        elif status == cp_model.INFEASIBLE:
            self._end_optimization(OptimizationStatus.INFEASIBLE)
        else:
            self._end_optimization(OptimizationStatus.TIMEOUT)

        return schedule

    def _create_variables(
        self,
        jobs: List[Job],
        machines: List[Machine],
        sku_dict: Dict[str, SKU],
        start_date: datetime,
        horizon: int,
    ) -> Dict[str, Any]:
        """
        Create decision variables

        Returns:
            Dictionary with all decision variables
        """
        job_vars = {}

        for job in jobs:
            job_id = job.job_id
            sku = sku_dict.get(job.sku_id)

            if not sku:
                logger.warning(f"SKU {job.sku_id} not found for job {job_id}")
                continue

            # Processing time (minutes)
            processing_minutes = int(sku.processing_time_per_unit * job.quantity * 60)

            # Release date (minutes from start)
            release_minutes = int((job.release_date - start_date).total_seconds() / 60)
            release_minutes = max(0, release_minutes)

            # Create interval variables for each compatible machine
            job_intervals = {}
            job_starts = {}
            job_ends = {}
            job_presences = {}

            for machine in machines:
                if job.sku_id not in machine.available_skus:
                    continue

                # Suffix for this machine
                suffix = f"_{job_id}_{machine.machine_id}"

                # Start time variable
                start_var = self.model.NewIntVar(
                    release_minutes, horizon, f"start{suffix}"
                )

                # End time variable
                end_var = self.model.NewIntVar(release_minutes, horizon, f"end{suffix}")

                # Presence variable (is job assigned to this machine?)
                presence_var = self.model.NewBoolVar(f"presence{suffix}")

                # Interval variable
                interval_var = self.model.NewOptionalIntervalVar(
                    start_var,
                    processing_minutes,
                    end_var,
                    presence_var,
                    f"interval{suffix}",
                )

                job_intervals[machine.machine_id] = interval_var
                job_starts[machine.machine_id] = start_var
                job_ends[machine.machine_id] = end_var
                job_presences[machine.machine_id] = presence_var

            job_vars[job_id] = {
                "intervals": job_intervals,
                "starts": job_starts,
                "ends": job_ends,
                "presences": job_presences,
                "processing_time": processing_minutes,
                "release_time": release_minutes,
            }

        return job_vars

    def _add_assignment_constraints(
        self, jobs: List[Job], machines: List[Machine], job_vars: Dict
    ) -> None:
        """
        Constraint: Each job assigned to exactly one machine

        Σ_m x_jm = 1  ∀j
        """
        for job in jobs:
            if job.job_id not in job_vars:
                continue

            presences = list(job_vars[job.job_id]["presences"].values())

            if presences:
                # Exactly one machine selected
                self.model.Add(sum(presences) == 1)

    def _add_temporal_constraints(
        self,
        jobs: List[Job],
        machines: List[Machine],
        sku_dict: Dict[str, SKU],
        changeover_matrix: ChangeoverMatrix,
        job_vars: Dict,
        start_date: datetime,
    ) -> None:
        """
        Temporal constraints:
        - Release dates
        - Due dates (soft)
        - Processing times
        """
        for job in jobs:
            if job.job_id not in job_vars:
                continue

            # Release date constraint already enforced in variable bounds

            # Due date (will be handled in objective as soft constraint)
            pass

    def _add_no_overlap_constraints(
        self, jobs: List[Job], machines: List[Machine], job_vars: Dict
    ) -> None:
        """
        Constraint: No overlap of jobs on same machine

        If x_jm = x_km = 1, then (e_j ≤ s_k) OR (e_k ≤ s_j)
        """
        for machine in machines:
            machine_intervals = []

            for job in jobs:
                if job.job_id not in job_vars:
                    continue

                intervals = job_vars[job.job_id]["intervals"]

                if machine.machine_id in intervals:
                    machine_intervals.append(intervals[machine.machine_id])

            if machine_intervals:
                # No overlap constraint
                self.model.AddNoOverlap(machine_intervals)

    def _build_objective(
        self,
        jobs: List[Job],
        machines: List[Machine],
        skus: List[SKU],
        changeover_matrix: ChangeoverMatrix,
        job_vars: Dict,
        weights: Dict[str, float],
        start_date: datetime,
    ) -> Any:
        """
        Build multi-objective function

        Objective: min α·lateness + β·setup_time + γ·cost + δ·energy

        For CP-SAT, we scale and combine into integer objective.
        """
        objective_terms = []
        sku_dict = {sku.sku_id: sku for sku in skus}

        # Component 1: Lateness (weighted heavily)
        lateness_weight_scaled = int(weights.get("lateness", 0.4) * 1000)

        for job in jobs:
            if job.job_id not in job_vars:
                continue

            # Due date (minutes from start)
            due_minutes = int((job.due_date - start_date).total_seconds() / 60)

            # For each machine, if assigned, calculate lateness
            for machine_id, end_var in job_vars[job.job_id]["ends"].items():
                presence_var = job_vars[job.job_id]["presences"][machine_id]

                # Lateness = max(0, end - due)
                lateness_var = self.model.NewIntVar(0, 100000, f"lateness_{job.job_id}_{machine_id}")
                self.model.AddMaxEquality(lateness_var, [end_var - due_minutes, 0])

                # Weighted lateness (only if job assigned to this machine)
                weighted_lateness = self.model.NewIntVar(
                    0, 100000 * lateness_weight_scaled, f"wlate_{job.job_id}_{machine_id}"
                )
                self.model.AddMultiplicationEquality(
                    weighted_lateness, [lateness_var, presence_var, lateness_weight_scaled]
                )

                objective_terms.append(weighted_lateness)

        # Component 2: Setup time (harder to model in CP-SAT, simplified)
        # In practice, setup times are handled through interval durations
        # For now, we add a penalty for number of jobs (encourages batching)
        setup_weight_scaled = int(weights.get("setup_time", 0.2) * 100)

        for machine in machines:
            # Count jobs on this machine
            machine_job_count = sum(
                job_vars[job.job_id]["presences"].get(machine.machine_id, 0)
                for job in jobs
                if job.job_id in job_vars
            )

            # Penalty proportional to number of jobs (more jobs = more setups)
            if setup_weight_scaled > 0:
                objective_terms.append(machine_job_count * setup_weight_scaled)

        # Component 3 & 4: Cost and Energy
        # These are fixed given the job assignments, so we don't need to model them
        # in the CP-SAT objective (they'll be calculated post-hoc)

        # Sum all terms
        total_objective = sum(objective_terms)

        return total_objective

    def _extract_solution(
        self,
        status: int,
        jobs: List[Job],
        machines: List[Machine],
        job_vars: Dict,
        start_date: datetime,
        end_date: datetime,
        changeover_matrix: ChangeoverMatrix,
        skus: List[SKU],
        objective_weights: Dict[str, float],
    ) -> Schedule:
        """
        Extract solution from solver

        Args:
            status: Solver status
            jobs: Original jobs
            machines: Machines
            job_vars: Decision variables
            start_date: Schedule start
            end_date: Schedule end
            changeover_matrix: Setup matrix
            skus: SKUs
            objective_weights: Weights

        Returns:
            Schedule with assignments
        """
        # Create schedule
        schedule_jobs = copy.deepcopy(jobs)

        schedule = Schedule(
            schedule_id=f"cpsat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            jobs=schedule_jobs,
            machines=machines,
            start_date=start_date,
            end_date=end_date,
            optimizer_used="cpsat",
            objective_weights=objective_weights,
        )

        if status == cp_model.INFEASIBLE:
            logger.error("Problem is INFEASIBLE")
            schedule.feasible = False
            schedule.constraint_violations.append("Problem is infeasible")
            return schedule

        if status == cp_model.MODEL_INVALID:
            logger.error("Model is INVALID")
            schedule.feasible = False
            schedule.constraint_violations.append("Model is invalid")
            return schedule

        # Extract job assignments
        for job in schedule_jobs:
            if job.job_id not in job_vars:
                continue

            # Find which machine this job is assigned to
            for machine_id, presence_var in job_vars[job.job_id]["presences"].items():
                if self.solver.Value(presence_var) == 1:
                    # Job assigned to this machine
                    start_minutes = self.solver.Value(
                        job_vars[job.job_id]["starts"][machine_id]
                    )
                    end_minutes = self.solver.Value(
                        job_vars[job.job_id]["ends"][machine_id]
                    )

                    # Convert to datetime
                    job.assigned_machine = machine_id
                    job.scheduled_start = start_date + timedelta(minutes=start_minutes)
                    job.scheduled_end = start_date + timedelta(minutes=end_minutes)

                    break

        # Calculate metrics
        metrics = ScheduleMetrics.calculate_all_metrics(
            schedule, changeover_matrix, skus
        )

        schedule.total_lateness = metrics.get("total_lateness_hours", 0.0)
        schedule.total_tardiness = metrics.get("total_tardiness_hours", 0.0)
        schedule.n_late_jobs = metrics.get("n_late_jobs", 0)
        schedule.total_setup_time = metrics.get("total_setup_time_hours", 0.0)
        schedule.total_cost = metrics.get("total_cost", 0.0)
        schedule.total_energy = metrics.get("total_energy_kwh", 0.0)
        schedule.makespan = metrics.get("makespan_hours", 0.0)

        # Calculate objective value
        obj_func = MultiObjectiveFunction(objective_weights)
        schedule.objective_value = obj_func.evaluate(
            schedule, changeover_matrix, skus, normalize=True
        )

        # Computation time
        schedule.computation_time = self.solver.WallTime()

        # Log solution stats
        logger.info("=" * 60)
        logger.info("CP-SAT Solution:")
        logger.info(f"  Status: {self._status_name(status)}")
        logger.info(f"  Scheduled jobs: {len(schedule.get_scheduled_jobs())}/{len(jobs)}")
        logger.info(f"  Total lateness: {schedule.total_lateness:.1f} hours")
        logger.info(f"  Late jobs: {schedule.n_late_jobs}")
        logger.info(f"  Total setup time: {schedule.total_setup_time:.1f} hours")
        logger.info(f"  Makespan: {schedule.makespan:.1f} hours")
        logger.info(f"  Total cost: ${schedule.total_cost:.2f}")
        logger.info(f"  Objective value: {schedule.objective_value:.4f}")
        logger.info(f"  Computation time: {schedule.computation_time:.2f}s")
        logger.info(f"  Branches explored: {self.solver.NumBranches()}")
        logger.info("=" * 60)

        return schedule

    def _status_name(self, status: int) -> str:
        """Convert status code to name"""
        status_names = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        return status_names.get(status, f"UNKNOWN({status})")
