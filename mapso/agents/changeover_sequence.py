"""
Changeover/Sequence Agent

Manages sequence-dependent setup times:
- Calculates total changeover time
- Suggests better job sequences (greedy nearest-neighbor)
- Tracks setup costs
- Minimizes total setup time

Uses Traveling Salesman Problem (TSP) heuristics for sequencing.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from mapso.agents.base import AbstractAgent
from mapso.core.models import Schedule, Job, ChangeoverMatrix
from mapso.utils.logging_config import get_logger

logger = get_logger("changeover_agent")


class ChangeoverSequenceAgent(AbstractAgent):
    """
    Optimizes job sequencing to minimize changeover times

    Uses greedy nearest-neighbor heuristic:
    - Start with first job
    - Always pick next job with minimum setup time
    - Continue until all jobs scheduled

    This is a 2-approximation for metric TSP.
    """

    def __init__(
        self,
        agent_id: str = "changeover_sequence",
        changeover_matrix: Optional[ChangeoverMatrix] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id, config)
        self.changeover_matrix = changeover_matrix

    def set_changeover_matrix(self, changeover_matrix: ChangeoverMatrix) -> None:
        """Set changeover matrix"""
        self.changeover_matrix = changeover_matrix

    def process(self, schedule: Schedule) -> Schedule:
        """
        Process schedule and suggest better sequences

        Args:
            schedule: Input schedule

        Returns:
            Schedule (potentially resequenced)
        """
        if self.changeover_matrix is None:
            self.logger.warning("No changeover matrix set, skipping processing")
            return schedule

        # Calculate current setup time
        total_setup = self._calculate_total_setup_time(schedule)
        self.logger.info(f"Current total setup time: {total_setup:.1f} minutes")

        # Optionally resequence (if enabled in config)
        if self.config.get("auto_resequence", False):
            schedule = self._resequence_all_machines(schedule)
            new_setup = self._calculate_total_setup_time(schedule)
            self.logger.info(f"After resequencing: {new_setup:.1f} minutes")

        return schedule

    def validate(self, schedule: Schedule) -> Tuple[bool, List[str]]:
        """
        Validate that setup times are accounted for

        This is a soft constraint - schedules are valid even with high setup times,
        but we report it as a warning.

        Returns:
            (True, warnings)
        """
        warnings = []

        if self.changeover_matrix is None:
            return True, ["No changeover matrix available"]

        # Check if schedule has excessive setup time
        total_setup = self._calculate_total_setup_time(schedule)
        avg_setup = self._calculate_average_setup_time()

        setup_by_machine = self._calculate_setup_by_machine(schedule)

        for machine_id, setup_time in setup_by_machine.items():
            # Warn if any changeover is > 2x average
            machine_jobs = schedule.get_jobs_by_machine(machine_id)
            if len(machine_jobs) < 2:
                continue

            machine_jobs.sort(key=lambda j: j.scheduled_start or float("inf"))

            for i in range(len(machine_jobs) - 1):
                from_sku = machine_jobs[i].sku_id
                to_sku = machine_jobs[i + 1].sku_id

                setup = self.changeover_matrix.get_setup_time(from_sku, to_sku)

                if setup > avg_setup * 2:
                    warnings.append(
                        f"High setup time on {machine_id}: {from_sku} → {to_sku} "
                        f"({setup:.0f} min, avg: {avg_setup:.0f} min)"
                    )

        return True, warnings

    def score(self, schedule: Schedule) -> float:
        """
        Score based on total setup time

        Lower is better (minimization).

        Returns:
            Total setup time in hours
        """
        if self.changeover_matrix is None:
            return 0.0

        total_setup_minutes = self._calculate_total_setup_time(schedule)
        return total_setup_minutes / 60  # Convert to hours

    # ==================== Setup Time Calculations ====================

    def _calculate_total_setup_time(self, schedule: Schedule) -> float:
        """
        Calculate total setup time across all machines

        Returns:
            Total setup time in minutes
        """
        if self.changeover_matrix is None:
            return 0.0

        total_setup = 0.0

        for machine in schedule.machines:
            machine_setup = self._calculate_machine_setup_time(
                schedule, machine.machine_id
            )
            total_setup += machine_setup

        return total_setup

    def _calculate_machine_setup_time(
        self, schedule: Schedule, machine_id: str
    ) -> float:
        """
        Calculate setup time for one machine

        Args:
            schedule: Schedule
            machine_id: Machine ID

        Returns:
            Setup time in minutes
        """
        machine_jobs = schedule.get_jobs_by_machine(machine_id)

        if len(machine_jobs) < 2:
            return 0.0

        # Sort by start time
        machine_jobs.sort(key=lambda j: j.scheduled_start or float("inf"))

        total_setup = 0.0

        for i in range(len(machine_jobs) - 1):
            from_sku = machine_jobs[i].sku_id
            to_sku = machine_jobs[i + 1].sku_id

            setup_time = self.changeover_matrix.get_setup_time(from_sku, to_sku)
            total_setup += setup_time

        return total_setup

    def _calculate_setup_by_machine(self, schedule: Schedule) -> Dict[str, float]:
        """
        Calculate setup time for each machine

        Returns:
            Dictionary of {machine_id: setup_time_minutes}
        """
        setup_by_machine = {}

        for machine in schedule.machines:
            setup_time = self._calculate_machine_setup_time(schedule, machine.machine_id)
            setup_by_machine[machine.machine_id] = setup_time

        return setup_by_machine

    def _calculate_average_setup_time(self) -> float:
        """Calculate average setup time from changeover matrix"""
        if self.changeover_matrix is None:
            return 0.0

        return self.changeover_matrix.get_average_setup_time()

    # ==================== Sequence Optimization ====================

    def optimize_sequence_greedy(self, jobs: List[Job]) -> List[Job]:
        """
        Optimize job sequence using greedy nearest-neighbor

        Greedy Nearest-Neighbor Algorithm:
        1. Start with first job
        2. Repeatedly select unscheduled job with minimum setup time from current job
        3. Continue until all jobs scheduled

        This is O(n²) and provides 2-approximation for metric TSP.

        Args:
            jobs: List of jobs to sequence

        Returns:
            Reordered list of jobs
        """
        if len(jobs) <= 1 or self.changeover_matrix is None:
            return jobs

        # Create mapping from SKU to jobs
        remaining_jobs = set(jobs)
        sequence = []

        # Start with first job (arbitrary choice)
        current_job = next(iter(remaining_jobs))
        sequence.append(current_job)
        remaining_jobs.remove(current_job)

        # Greedily select next job with minimum setup time
        while remaining_jobs:
            current_sku = current_job.sku_id
            best_job = None
            best_setup = float("inf")

            for job in remaining_jobs:
                setup_time = self.changeover_matrix.get_setup_time(
                    current_sku, job.sku_id
                )

                if setup_time < best_setup:
                    best_setup = setup_time
                    best_job = job

            if best_job:
                sequence.append(best_job)
                remaining_jobs.remove(best_job)
                current_job = best_job
            else:
                break

        return sequence

    def _resequence_all_machines(self, schedule: Schedule) -> Schedule:
        """
        Resequence jobs on all machines to minimize setup time

        Args:
            schedule: Input schedule

        Returns:
            Schedule with resequenced jobs
        """
        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)

            if len(machine_jobs) < 2:
                continue

            # Optimize sequence
            optimized_sequence = self.optimize_sequence_greedy(machine_jobs)

            # Update job timings (preserve total duration, resequence)
            # This is simplified - real implementation would recalculate all timings
            if optimized_sequence:
                start_time = min(
                    j.scheduled_start for j in machine_jobs if j.scheduled_start
                )

                current_time = start_time

                for job in optimized_sequence:
                    if job.scheduled_start and job.scheduled_end:
                        duration = job.scheduled_end - job.scheduled_start
                        job.scheduled_start = current_time
                        job.scheduled_end = current_time + duration
                        current_time = job.scheduled_end

                        # Add setup time for next job
                        if job != optimized_sequence[-1]:
                            next_job = optimized_sequence[
                                optimized_sequence.index(job) + 1
                            ]
                            setup_minutes = self.changeover_matrix.get_setup_time(
                                job.sku_id, next_job.sku_id
                            )
                            from datetime import timedelta
                            current_time += timedelta(minutes=setup_minutes)

        return schedule

    def suggest_resequencing(
        self, schedule: Schedule, machine_id: str
    ) -> List[Job]:
        """
        Suggest better job sequence for a machine

        Args:
            schedule: Current schedule
            machine_id: Machine to optimize

        Returns:
            Suggested job sequence
        """
        machine_jobs = schedule.get_jobs_by_machine(machine_id)

        if len(machine_jobs) < 2:
            return machine_jobs

        # Current setup time
        current_setup = self._calculate_machine_setup_time(schedule, machine_id)

        # Optimized sequence
        optimized_jobs = self.optimize_sequence_greedy(machine_jobs)

        # Calculate optimized setup time
        optimized_setup = 0.0
        for i in range(len(optimized_jobs) - 1):
            setup = self.changeover_matrix.get_setup_time(
                optimized_jobs[i].sku_id, optimized_jobs[i + 1].sku_id
            )
            optimized_setup += setup

        if optimized_setup < current_setup:
            self.logger.info(
                f"Machine {machine_id}: Resequencing saves "
                f"{current_setup - optimized_setup:.1f} minutes "
                f"({(1 - optimized_setup/current_setup)*100:.1f}% reduction)"
            )

        return optimized_jobs

    # ==================== Analysis Methods ====================

    def analyze_changeover_pattern(self, schedule: Schedule) -> Dict[str, Any]:
        """
        Analyze changeover patterns in schedule

        Returns:
            Dictionary with analysis results
        """
        if self.changeover_matrix is None:
            return {}

        changeovers = []

        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)

            if len(machine_jobs) < 2:
                continue

            machine_jobs.sort(key=lambda j: j.scheduled_start or float("inf"))

            for i in range(len(machine_jobs) - 1):
                from_sku = machine_jobs[i].sku_id
                to_sku = machine_jobs[i + 1].sku_id

                setup_time = self.changeover_matrix.get_setup_time(from_sku, to_sku)
                setup_cost = self.changeover_matrix.get_setup_cost(from_sku, to_sku)

                changeovers.append(
                    {
                        "machine": machine.machine_id,
                        "from_sku": from_sku,
                        "to_sku": to_sku,
                        "setup_time_minutes": setup_time,
                        "setup_cost": setup_cost,
                    }
                )

        if not changeovers:
            return {"n_changeovers": 0}

        setup_times = [c["setup_time_minutes"] for c in changeovers]
        setup_costs = [c["setup_cost"] for c in changeovers]

        return {
            "n_changeovers": len(changeovers),
            "total_setup_time_minutes": sum(setup_times),
            "total_setup_cost": sum(setup_costs),
            "avg_setup_time_minutes": np.mean(setup_times),
            "max_setup_time_minutes": max(setup_times),
            "min_setup_time_minutes": min(setup_times),
            "std_setup_time_minutes": np.std(setup_times),
            "changeovers": changeovers,
        }
