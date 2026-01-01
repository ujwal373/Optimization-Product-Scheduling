"""
Optimization Orchestrator Agent

Central coordinator that:
1. Implements baseline heuristics (FIFO, EDD, SPT, Priority)
2. Coordinates between specialized agents
3. Manages optimization layer selection
4. Validates schedules through agent pipeline

This is the "brain" that orchestrates the multi-agent scheduling system.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import copy

from mapso.agents.base import AbstractAgent
from mapso.core.models import Schedule, Job, Machine, ChangeoverMatrix
from mapso.core.enums import HeuristicType, SolverType
from mapso.utils.logging_config import get_logger

logger = get_logger("orchestrator")


class OrchestratorAgent(AbstractAgent):
    """
    Orchestrator agent for multi-agent scheduling

    Responsibilities:
    - Generate baseline schedules using heuristics
    - Coordinate agent communication
    - Manage optimization workflow
    - Validate schedules through agent pipeline
    """

    def __init__(
        self,
        agent_id: str = "orchestrator",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id, config)
        self.agents: List[AbstractAgent] = []
        self.changeover_matrix: Optional[ChangeoverMatrix] = None

    def register_agent(self, agent: AbstractAgent) -> None:
        """
        Register a specialized agent

        Args:
            agent: Agent to register
        """
        self.agents.append(agent)
        self.logger.info(f"Registered agent: {agent.agent_id}")

    def set_changeover_matrix(self, changeover_matrix: ChangeoverMatrix) -> None:
        """Set changeover matrix for setup time calculations"""
        self.changeover_matrix = changeover_matrix

    def process(self, schedule: Schedule) -> Schedule:
        """
        Process schedule through agent pipeline

        Args:
            schedule: Input schedule

        Returns:
            Processed schedule
        """
        current_schedule = schedule

        # Pass through each agent
        for agent in self.agents:
            if agent.is_enabled():
                self.logger.info(f"Processing with {agent.agent_id}")
                current_schedule = agent.process(current_schedule)

        return current_schedule

    def validate(self, schedule: Schedule) -> tuple[bool, List[str]]:
        """
        Validate schedule through all agents

        Args:
            schedule: Schedule to validate

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        all_violations = []
        all_valid = True

        for agent in self.agents:
            if agent.is_enabled():
                is_valid, violations = agent.validate(schedule)

                if not is_valid:
                    all_valid = False
                    all_violations.extend(
                        [f"[{agent.agent_id}] {v}" for v in violations]
                    )

        return all_valid, all_violations

    def score(self, schedule: Schedule) -> float:
        """
        Aggregate score from all agents

        Args:
            schedule: Schedule to score

        Returns:
            Weighted sum of agent scores
        """
        total_score = 0.0

        for agent in self.agents:
            if agent.is_enabled():
                agent_score = agent.score(schedule)
                total_score += agent_score

        return total_score

    # ==================== Baseline Heuristics ====================

    def create_baseline_schedule(
        self,
        jobs: List[Job],
        machines: List[Machine],
        start_date: datetime,
        end_date: datetime,
        heuristic: HeuristicType = HeuristicType.EDD,
    ) -> Schedule:
        """
        Create baseline schedule using heuristic

        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            start_date: Schedule start date
            end_date: Schedule end date
            heuristic: Heuristic to use

        Returns:
            Baseline schedule
        """
        self.logger.info(f"Creating baseline schedule with {heuristic.value} heuristic")

        # Create schedule object
        schedule = Schedule(
            schedule_id=f"baseline_{heuristic.value}",
            jobs=copy.deepcopy(jobs),
            machines=machines,
            start_date=start_date,
            end_date=end_date,
            optimizer_used=f"baseline_{heuristic.value}",
        )

        # Apply heuristic
        if heuristic == HeuristicType.FIFO:
            schedule = self._schedule_fifo(schedule)
        elif heuristic == HeuristicType.EDD:
            schedule = self._schedule_edd(schedule)
        elif heuristic == HeuristicType.SPT:
            schedule = self._schedule_spt(schedule)
        elif heuristic == HeuristicType.PRIORITY:
            schedule = self._schedule_priority(schedule)
        elif heuristic == HeuristicType.RANDOM:
            schedule = self._schedule_random(schedule)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")

        self.logger.info(f"Baseline schedule created: {len(schedule.get_scheduled_jobs())} jobs scheduled")
        return schedule

    def _schedule_fifo(self, schedule: Schedule) -> Schedule:
        """
        First In First Out (FIFO) - Schedule by release date

        Jobs are scheduled in order of release date (earliest first).
        """
        # Sort jobs by release date
        sorted_jobs = sorted(schedule.jobs, key=lambda j: j.release_date)

        # Assign to machines
        schedule = self._assign_jobs_to_machines(schedule, sorted_jobs)

        return schedule

    def _schedule_edd(self, schedule: Schedule) -> Schedule:
        """
        Earliest Due Date (EDD) - Schedule by due date

        Jobs with earlier due dates are scheduled first.
        This minimizes maximum lateness.
        """
        # Sort jobs by due date
        sorted_jobs = sorted(schedule.jobs, key=lambda j: j.due_date)

        # Assign to machines
        schedule = self._assign_jobs_to_machines(schedule, sorted_jobs)

        return schedule

    def _schedule_spt(self, schedule: Schedule) -> Schedule:
        """
        Shortest Processing Time (SPT) - Schedule shortest jobs first

        Minimizes average completion time and average flow time.
        """
        # Sort jobs by processing time (estimated)
        sorted_jobs = sorted(
            schedule.jobs,
            key=lambda j: j.quantity * 0.1,  # Simplified: assume 0.1 hour/unit
        )

        # Assign to machines
        schedule = self._assign_jobs_to_machines(schedule, sorted_jobs)

        return schedule

    def _schedule_priority(self, schedule: Schedule) -> Schedule:
        """
        Priority-based scheduling

        Jobs with higher priority (lower number) scheduled first.
        """
        # Sort jobs by priority (ascending - 1 is highest priority)
        sorted_jobs = sorted(schedule.jobs, key=lambda j: (j.priority, j.due_date))

        # Assign to machines
        schedule = self._assign_jobs_to_machines(schedule, sorted_jobs)

        return schedule

    def _schedule_random(self, schedule: Schedule) -> Schedule:
        """
        Random scheduling (for baseline comparison)
        """
        import random

        sorted_jobs = schedule.jobs.copy()
        random.shuffle(sorted_jobs)

        # Assign to machines
        schedule = self._assign_jobs_to_machines(schedule, sorted_jobs)

        return schedule

    def _assign_jobs_to_machines(
        self, schedule: Schedule, ordered_jobs: List[Job]
    ) -> Schedule:
        """
        Assign jobs to machines in given order

        Uses greedy assignment:
        - Find first available machine that can produce the SKU
        - Schedule at earliest available time
        - Account for setup times if changeover matrix is available

        Args:
            schedule: Schedule object
            ordered_jobs: Jobs in desired scheduling order

        Returns:
            Schedule with jobs assigned
        """
        # Track latest end time for each machine
        machine_end_times = {m.machine_id: schedule.start_date for m in schedule.machines}
        machine_last_sku = {}  # Track last SKU on each machine for setup times

        for job in ordered_jobs:
            # Find candidate machines
            candidate_machines = [
                m for m in schedule.machines if job.sku_id in m.available_skus
            ]

            if not candidate_machines:
                # No machine can produce this SKU
                self.logger.warning(f"Job {job.job_id}: No compatible machine for SKU {job.sku_id}")
                continue

            # Find machine with earliest available time
            best_machine = None
            best_start_time = None
            best_end_time = None

            for machine in candidate_machines:
                # Start time: max of (release date, machine available time)
                start_time = max(job.release_date, machine_end_times[machine.machine_id])

                # Add setup time if different SKU
                setup_time_hours = 0.0
                if self.changeover_matrix and machine.machine_id in machine_last_sku:
                    last_sku = machine_last_sku[machine.machine_id]
                    if last_sku != job.sku_id:
                        setup_time_minutes = self.changeover_matrix.get_setup_time(
                            last_sku, job.sku_id
                        )
                        setup_time_hours = setup_time_minutes / 60
                        start_time += timedelta(hours=setup_time_hours)

                # Estimated processing time (simplified: 0.1 hour per unit)
                # In reality, this should use SKU processing time
                processing_hours = job.quantity * 0.1
                end_time = start_time + timedelta(hours=processing_hours)

                # Select machine with earliest completion
                if best_machine is None or end_time < best_end_time:
                    best_machine = machine
                    best_start_time = start_time
                    best_end_time = end_time

            # Assign job to best machine
            if best_machine:
                job.assigned_machine = best_machine.machine_id
                job.scheduled_start = best_start_time
                job.scheduled_end = best_end_time

                # Update machine availability
                machine_end_times[best_machine.machine_id] = best_end_time
                machine_last_sku[best_machine.machine_id] = job.sku_id

                self.logger.debug(
                    f"Job {job.job_id} → Machine {best_machine.machine_id} "
                    f"[{best_start_time.strftime('%Y-%m-%d %H:%M')} - "
                    f"{best_end_time.strftime('%Y-%m-%d %H:%M')}]"
                )

        return schedule

    def compare_heuristics(
        self,
        jobs: List[Job],
        machines: List[Machine],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Schedule]:
        """
        Compare all baseline heuristics

        Args:
            jobs: List of jobs
            machines: List of machines
            start_date: Schedule start
            end_date: Schedule end

        Returns:
            Dictionary of {heuristic_name: schedule}
        """
        heuristics = [
            HeuristicType.FIFO,
            HeuristicType.EDD,
            HeuristicType.SPT,
            HeuristicType.PRIORITY,
        ]

        schedules = {}

        for heuristic in heuristics:
            schedule = self.create_baseline_schedule(
                jobs, machines, start_date, end_date, heuristic
            )
            schedules[heuristic.value] = schedule

        return schedules
