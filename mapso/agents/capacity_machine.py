"""
Capacity & Machine Agent

Hard constraint validator that ensures:
- Machine capacity not exceeded
- Jobs assigned to compatible machines
- Shift availability respected
- Maintenance windows avoided
- No job overlaps on same machine

This agent REJECTS infeasible schedules.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from mapso.agents.base import AbstractAgent
from mapso.core.models import Schedule, Job, Machine
from mapso.utils.logging_config import get_logger

logger = get_logger("capacity_agent")


class CapacityMachineAgent(AbstractAgent):
    """
    Validates and enforces machine capacity constraints

    Hard Constraints:
    1. Job-Machine Compatibility: Jobs only on compatible machines
    2. Machine Capacity: Total load ≤ capacity
    3. Shift Availability: Jobs only during shifts
    4. Maintenance Windows: No jobs during maintenance
    5. No Overlaps: Jobs don't overlap on same machine
    """

    def __init__(
        self,
        agent_id: str = "capacity_machine",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id, config)
        self.tolerance_hours = config.get("tolerance_hours", 0.1) if config else 0.1

    def process(self, schedule: Schedule) -> Schedule:
        """
        Process schedule and mark feasibility

        Args:
            schedule: Input schedule

        Returns:
            Schedule with feasibility status updated
        """
        is_valid, violations = self.validate(schedule)

        schedule.feasible = is_valid
        schedule.constraint_violations.extend(violations)

        if not is_valid:
            self.logger.warning(f"Schedule {schedule.schedule_id} is INFEASIBLE")
            for violation in violations:
                self.logger.warning(f"  - {violation}")

        return schedule

    def validate(self, schedule: Schedule) -> Tuple[bool, List[str]]:
        """
        Validate all capacity and machine constraints

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # 1. Check job-machine compatibility
        compat_violations = self._check_compatibility(schedule)
        violations.extend(compat_violations)

        # 2. Check machine capacity
        capacity_violations = self._check_capacity(schedule)
        violations.extend(capacity_violations)

        # 3. Check shift availability
        shift_violations = self._check_shift_availability(schedule)
        violations.extend(shift_violations)

        # 4. Check maintenance windows
        maintenance_violations = self._check_maintenance_windows(schedule)
        violations.extend(maintenance_violations)

        # 5. Check no overlaps
        overlap_violations = self._check_no_overlaps(schedule)
        violations.extend(overlap_violations)

        is_valid = len(violations) == 0

        return is_valid, violations

    def score(self, schedule: Schedule) -> float:
        """
        Score based on capacity utilization

        Lower score = better (more balanced utilization)

        Returns:
            Standard deviation of machine utilizations
        """
        utilizations = []

        for machine in schedule.machines:
            utilization = schedule.calculate_utilization(machine.machine_id)
            utilizations.append(utilization)

        if not utilizations:
            return 0.0

        # Return std dev of utilizations (lower = more balanced)
        import numpy as np
        return float(np.std(utilizations))

    # ==================== Validation Methods ====================

    def _check_compatibility(self, schedule: Schedule) -> List[str]:
        """Check that jobs are assigned to compatible machines"""
        violations = []

        machine_dict = {m.machine_id: m for m in schedule.machines}

        for job in schedule.get_scheduled_jobs():
            if job.assigned_machine not in machine_dict:
                violations.append(
                    f"Job {job.job_id}: Assigned to non-existent machine {job.assigned_machine}"
                )
                continue

            machine = machine_dict[job.assigned_machine]

            if job.sku_id not in machine.available_skus:
                violations.append(
                    f"Job {job.job_id}: SKU {job.sku_id} cannot be produced on "
                    f"machine {machine.machine_id}"
                )

        return violations

    def _check_capacity(self, schedule: Schedule) -> List[str]:
        """Check that machine capacity is not exceeded"""
        violations = []

        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)

            if not machine_jobs:
                continue

            # Calculate total required capacity (simplified)
            total_hours = sum(
                (job.scheduled_end - job.scheduled_start).total_seconds() / 3600
                for job in machine_jobs
                if job.scheduled_start and job.scheduled_end
            )

            # Available capacity
            schedule_duration = (schedule.end_date - schedule.start_date).total_seconds() / 3600
            available_capacity = machine.capacity_per_hour * schedule_duration

            # Check if capacity exceeded (with small tolerance)
            if total_hours > available_capacity * 1.01:  # 1% tolerance
                violations.append(
                    f"Machine {machine.machine_id}: Capacity exceeded "
                    f"(required: {total_hours:.1f}h, available: {available_capacity:.1f}h)"
                )

        return violations

    def _check_shift_availability(self, schedule: Schedule) -> List[str]:
        """Check that jobs only execute during shift times"""
        violations = []

        machine_dict = {m.machine_id: m for m in schedule.machines}

        for job in schedule.get_scheduled_jobs():
            if not job.scheduled_start or not job.scheduled_end:
                continue

            machine = machine_dict.get(job.assigned_machine)
            if not machine:
                continue

            # Check if job time overlaps with any shift
            if not self._is_within_shifts(
                job.scheduled_start,
                job.scheduled_end,
                machine.shift_calendar,
            ):
                violations.append(
                    f"Job {job.job_id}: Scheduled outside shift hours on "
                    f"machine {machine.machine_id} "
                    f"({job.scheduled_start.strftime('%Y-%m-%d %H:%M')} - "
                    f"{job.scheduled_end.strftime('%Y-%m-%d %H:%M')})"
                )

        return violations

    def _is_within_shifts(
        self, start: datetime, end: datetime, shift_calendar
    ) -> bool:
        """
        Check if time window is covered by shifts

        This is a simplified check - assumes if any shift is active on that day,
        the time is valid. Real implementation would check exact times.

        Args:
            start: Start time
            end: End time
            shift_calendar: Machine shift calendar

        Returns:
            True if within shifts
        """
        # Simplified: check if day of week has shifts
        day_of_week = start.weekday()

        for shift in shift_calendar.shifts:
            if day_of_week in shift.days_of_week:
                return True

        return False

    def _check_maintenance_windows(self, schedule: Schedule) -> List[str]:
        """Check that jobs don't overlap with maintenance"""
        violations = []

        machine_dict = {m.machine_id: m for m in schedule.machines}

        for job in schedule.get_scheduled_jobs():
            if not job.scheduled_start or not job.scheduled_end:
                continue

            machine = machine_dict.get(job.assigned_machine)
            if not machine:
                continue

            # Check if job overlaps with maintenance
            for maint_start, maint_end in machine.maintenance_windows:
                if self._times_overlap(
                    job.scheduled_start, job.scheduled_end, maint_start, maint_end
                ):
                    violations.append(
                        f"Job {job.job_id}: Overlaps with maintenance window on "
                        f"machine {machine.machine_id} "
                        f"({maint_start.strftime('%Y-%m-%d %H:%M')} - "
                        f"{maint_end.strftime('%Y-%m-%d %H:%M')})"
                    )

        return violations

    def _check_no_overlaps(self, schedule: Schedule) -> List[str]:
        """Check that jobs on same machine don't overlap"""
        violations = []

        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)

            if len(machine_jobs) < 2:
                continue

            # Sort by start time
            sorted_jobs = sorted(
                [j for j in machine_jobs if j.scheduled_start],
                key=lambda j: j.scheduled_start,
            )

            # Check consecutive jobs
            for i in range(len(sorted_jobs) - 1):
                job1 = sorted_jobs[i]
                job2 = sorted_jobs[i + 1]

                if not job1.scheduled_end or not job2.scheduled_start:
                    continue

                # Check overlap (with small tolerance for numerical errors)
                if job1.scheduled_end > job2.scheduled_start + timedelta(
                    hours=self.tolerance_hours
                ):
                    violations.append(
                        f"Jobs {job1.job_id} and {job2.job_id} overlap on "
                        f"machine {machine.machine_id} "
                        f"({job1.job_id} ends at {job1.scheduled_end.strftime('%H:%M')}, "
                        f"{job2.job_id} starts at {job2.scheduled_start.strftime('%H:%M')})"
                    )

        return violations

    def _times_overlap(
        self, start1: datetime, end1: datetime, start2: datetime, end2: datetime
    ) -> bool:
        """Check if two time intervals overlap"""
        return start1 < end2 and start2 < end1

    # ==================== Utility Methods ====================

    def compute_utilization(self, schedule: Schedule) -> Dict[str, float]:
        """
        Compute utilization for each machine

        Returns:
            Dictionary of {machine_id: utilization_rate}
        """
        utilization = {}

        for machine in schedule.machines:
            util_rate = schedule.calculate_utilization(machine.machine_id)
            utilization[machine.machine_id] = util_rate

        return utilization

    def identify_bottlenecks(self, schedule: Schedule) -> List[str]:
        """
        Identify bottleneck machines (high utilization)

        Returns:
            List of machine IDs with utilization > 90%
        """
        bottlenecks = []
        utilization = self.compute_utilization(schedule)

        for machine_id, util_rate in utilization.items():
            if util_rate > 0.9:
                bottlenecks.append(machine_id)
                self.logger.info(
                    f"Bottleneck identified: Machine {machine_id} "
                    f"(utilization: {util_rate:.1%})"
                )

        return bottlenecks

    def suggest_rebalancing(self, schedule: Schedule) -> List[str]:
        """
        Suggest load rebalancing actions

        Returns:
            List of suggestions
        """
        suggestions = []
        utilization = self.compute_utilization(schedule)

        # Find underutilized and overutilized machines
        underutilized = [m for m, u in utilization.items() if u < 0.5]
        overutilized = [m for m, u in utilization.items() if u > 0.9]

        if underutilized and overutilized:
            suggestions.append(
                f"Consider moving jobs from {', '.join(overutilized)} "
                f"to {', '.join(underutilized)}"
            )

        return suggestions
