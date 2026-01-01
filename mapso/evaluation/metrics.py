"""
Evaluation metrics for production schedules

Computes comprehensive metrics to evaluate schedule quality:
- Lateness and tardiness
- Makespan
- Setup/changeover time
- Cost (production, overtime, setup)
- Energy consumption
- Machine utilization
- Feasibility

Mathematical formulations:

1. Total Lateness: L = Σ max(0, C_j - d_j) where C_j = completion time, d_j = due date
2. Tardiness: T = Σ max(0, C_j - d_j) (same as lateness, but often refers to count)
3. Makespan: M = max(C_j) - min(S_j) where S_j = start time
4. Setup Time: ST = Σ c_{ij} where c_{ij} = changeover time from job i to j
5. Utilization: U = (Production Time) / (Available Time)
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

from mapso.core.models import Schedule, Job, Machine, ChangeoverMatrix
from mapso.utils.logging_config import get_logger

logger = get_logger("metrics")


class ScheduleMetrics:
    """
    Computes comprehensive metrics for production schedules

    All time-based metrics are in hours unless specified.
    """

    @staticmethod
    def calculate_lateness(schedule: Schedule) -> Dict[str, float]:
        """
        Calculate lateness metrics

        Lateness = completion_time - due_date (can be negative if early)
        Tardiness = max(0, lateness) (only positive lateness)

        Returns:
            Dictionary with lateness metrics
        """
        scheduled_jobs = schedule.get_scheduled_jobs()

        if not scheduled_jobs:
            return {
                "total_lateness_hours": 0.0,
                "total_tardiness_hours": 0.0,
                "n_late_jobs": 0,
                "n_on_time_jobs": 0,
                "max_lateness_hours": 0.0,
                "avg_lateness_hours": 0.0,
                "on_time_rate": 0.0,
            }

        lateness_values = []
        tardiness_values = []
        n_late = 0

        for job in scheduled_jobs:
            lateness = job.lateness()
            tardiness = job.tardiness()

            lateness_values.append(lateness)
            tardiness_values.append(tardiness)

            if tardiness > 0:
                n_late += 1

        return {
            "total_lateness_hours": sum(lateness_values),
            "total_tardiness_hours": sum(tardiness_values),
            "n_late_jobs": n_late,
            "n_on_time_jobs": len(scheduled_jobs) - n_late,
            "max_lateness_hours": max(lateness_values) if lateness_values else 0.0,
            "avg_lateness_hours": np.mean(lateness_values) if lateness_values else 0.0,
            "on_time_rate": 1 - (n_late / len(scheduled_jobs)) if scheduled_jobs else 0.0,
        }

    @staticmethod
    def calculate_makespan(schedule: Schedule) -> Dict[str, float]:
        """
        Calculate makespan (total schedule duration)

        Makespan = max(completion_time) - min(start_time)

        Returns:
            Dictionary with makespan metrics
        """
        scheduled_jobs = schedule.get_scheduled_jobs()

        if not scheduled_jobs:
            return {"makespan_hours": 0.0, "start_time": None, "end_time": None}

        start_times = [job.scheduled_start for job in scheduled_jobs if job.scheduled_start]
        end_times = [job.scheduled_end for job in scheduled_jobs if job.scheduled_end]

        if not start_times or not end_times:
            return {"makespan_hours": 0.0, "start_time": None, "end_time": None}

        earliest_start = min(start_times)
        latest_end = max(end_times)

        makespan_hours = (latest_end - earliest_start).total_seconds() / 3600

        return {
            "makespan_hours": makespan_hours,
            "start_time": earliest_start,
            "end_time": latest_end,
        }

    @staticmethod
    def calculate_setup_time(
        schedule: Schedule, changeover_matrix: ChangeoverMatrix
    ) -> Dict[str, float]:
        """
        Calculate total changeover/setup time

        Computes setup time between consecutive jobs on each machine.

        Args:
            schedule: Schedule to evaluate
            changeover_matrix: Changeover time matrix

        Returns:
            Dictionary with setup time metrics
        """
        total_setup_minutes = 0.0
        setup_count = 0
        setup_by_machine = {}

        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)

            if len(machine_jobs) < 2:
                setup_by_machine[machine.machine_id] = 0.0
                continue

            # Sort jobs by start time
            machine_jobs.sort(key=lambda j: j.scheduled_start or datetime.max)

            machine_setup = 0.0
            for i in range(len(machine_jobs) - 1):
                if (
                    machine_jobs[i].scheduled_end is not None
                    and machine_jobs[i + 1].scheduled_start is not None
                ):
                    # Get changeover time
                    from_sku = machine_jobs[i].sku_id
                    to_sku = machine_jobs[i + 1].sku_id

                    setup_time = changeover_matrix.get_setup_time(from_sku, to_sku)
                    machine_setup += setup_time
                    setup_count += 1

            setup_by_machine[machine.machine_id] = machine_setup
            total_setup_minutes += machine_setup

        return {
            "total_setup_time_hours": total_setup_minutes / 60,
            "total_setup_time_minutes": total_setup_minutes,
            "n_changeovers": setup_count,
            "avg_setup_time_minutes": total_setup_minutes / setup_count if setup_count > 0 else 0.0,
            "setup_by_machine": {k: v / 60 for k, v in setup_by_machine.items()},  # Convert to hours
        }

    @staticmethod
    def calculate_utilization(schedule: Schedule) -> Dict[str, float]:
        """
        Calculate machine utilization rates

        Utilization = (Production Time) / (Available Time)

        Returns:
            Dictionary with utilization metrics
        """
        utilization_by_machine = {}
        production_times = []
        available_times = []

        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)

            if not machine_jobs:
                utilization_by_machine[machine.machine_id] = 0.0
                continue

            # Total production time
            total_production = sum(
                (job.scheduled_end - job.scheduled_start).total_seconds() / 3600
                for job in machine_jobs
                if job.scheduled_start and job.scheduled_end
            )

            # Available time (schedule duration)
            schedule_duration = (schedule.end_date - schedule.start_date).total_seconds() / 3600

            utilization = total_production / schedule_duration if schedule_duration > 0 else 0.0
            utilization_by_machine[machine.machine_id] = utilization

            production_times.append(total_production)
            available_times.append(schedule_duration)

        avg_utilization = np.mean(list(utilization_by_machine.values())) if utilization_by_machine else 0.0

        return {
            "avg_utilization": avg_utilization,
            "utilization_by_machine": utilization_by_machine,
            "total_production_hours": sum(production_times),
            "total_available_hours": sum(available_times),
        }

    @staticmethod
    def calculate_cost(
        schedule: Schedule,
        changeover_matrix: ChangeoverMatrix,
        skus: List,
        labor_rate_per_hour: float = 50.0,
    ) -> Dict[str, float]:
        """
        Calculate total cost

        Cost components:
        1. Production cost (based on SKU cost per unit)
        2. Setup cost (from changeover matrix)
        3. Overtime cost (if beyond regular shifts)
        4. Late penalty (for late jobs)

        Args:
            schedule: Schedule to evaluate
            changeover_matrix: Changeover cost matrix
            skus: List of SKUs
            labor_rate_per_hour: Labor cost per hour

        Returns:
            Dictionary with cost metrics
        """
        sku_dict = {sku.sku_id: sku for sku in skus}

        # 1. Production cost
        production_cost = 0.0
        for job in schedule.get_scheduled_jobs():
            if job.sku_id in sku_dict:
                sku = sku_dict[job.sku_id]
                production_cost += sku.cost_per_unit * job.quantity

        # 2. Setup cost
        setup_cost = 0.0
        setup_count = 0

        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)

            if len(machine_jobs) < 2:
                continue

            machine_jobs.sort(key=lambda j: j.scheduled_start or datetime.max)

            for i in range(len(machine_jobs) - 1):
                from_sku = machine_jobs[i].sku_id
                to_sku = machine_jobs[i + 1].sku_id

                setup_cost += changeover_matrix.get_setup_cost(from_sku, to_sku)
                setup_count += 1

        # 3. Overtime cost (simplified: assume jobs beyond 8 hours/day are overtime)
        # This is a simplified calculation - real implementation would use shift calendars
        overtime_cost = 0.0

        # 4. Late penalty
        late_penalty = 0.0
        lateness_metrics = ScheduleMetrics.calculate_lateness(schedule)
        late_penalty = lateness_metrics["total_tardiness_hours"] * 100.0  # $100/hour penalty

        total_cost = production_cost + setup_cost + overtime_cost + late_penalty

        return {
            "total_cost": total_cost,
            "production_cost": production_cost,
            "setup_cost": setup_cost,
            "overtime_cost": overtime_cost,
            "late_penalty": late_penalty,
        }

    @staticmethod
    def calculate_energy(
        schedule: Schedule, skus: List, machines: List[Machine]
    ) -> Dict[str, float]:
        """
        Calculate energy consumption

        Energy = Σ (energy_per_unit * quantity * machine_efficiency)

        Args:
            schedule: Schedule to evaluate
            skus: List of SKUs
            machines: List of machines

        Returns:
            Dictionary with energy metrics
        """
        sku_dict = {sku.sku_id: sku for sku in skus}
        machine_dict = {m.machine_id: m for m in machines}

        total_energy = 0.0
        energy_by_machine = {}

        for machine in schedule.machines:
            machine_jobs = schedule.get_jobs_by_machine(machine.machine_id)
            machine_energy = 0.0

            for job in machine_jobs:
                if job.sku_id in sku_dict:
                    sku = sku_dict[job.sku_id]
                    efficiency = machine_dict[machine.machine_id].energy_efficiency

                    # Energy = energy_per_unit * quantity / efficiency
                    energy = (sku.energy_per_unit * job.quantity) / efficiency
                    machine_energy += energy

            energy_by_machine[machine.machine_id] = machine_energy
            total_energy += machine_energy

        return {
            "total_energy_kwh": total_energy,
            "energy_by_machine": energy_by_machine,
        }

    @staticmethod
    def calculate_all_metrics(
        schedule: Schedule,
        changeover_matrix: ChangeoverMatrix,
        skus: List,
        machines: Optional[List[Machine]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate all metrics for a schedule

        Args:
            schedule: Schedule to evaluate
            changeover_matrix: Changeover matrix
            skus: List of SKUs
            machines: List of machines (uses schedule.machines if None)

        Returns:
            Dictionary with all metrics
        """
        if machines is None:
            machines = schedule.machines

        metrics = {}

        # Lateness metrics
        metrics.update(ScheduleMetrics.calculate_lateness(schedule))

        # Makespan metrics
        metrics.update(ScheduleMetrics.calculate_makespan(schedule))

        # Setup time metrics
        setup_metrics = ScheduleMetrics.calculate_setup_time(schedule, changeover_matrix)
        metrics.update(setup_metrics)

        # Utilization metrics
        util_metrics = ScheduleMetrics.calculate_utilization(schedule)
        metrics.update(util_metrics)

        # Cost metrics
        cost_metrics = ScheduleMetrics.calculate_cost(schedule, changeover_matrix, skus)
        metrics.update(cost_metrics)

        # Energy metrics
        energy_metrics = ScheduleMetrics.calculate_energy(schedule, skus, machines)
        metrics.update(energy_metrics)

        # Feasibility
        metrics["feasible"] = schedule.is_feasible()
        metrics["n_constraint_violations"] = len(schedule.constraint_violations)

        # Job statistics
        metrics["n_jobs"] = len(schedule.jobs)
        metrics["n_scheduled_jobs"] = len(schedule.get_scheduled_jobs())
        metrics["n_unscheduled_jobs"] = len(schedule.get_unscheduled_jobs())

        return metrics


class MetricsComparator:
    """Compare metrics between different schedules"""

    @staticmethod
    def compare_schedules(
        schedules: Dict[str, Schedule],
        changeover_matrix: ChangeoverMatrix,
        skus: List,
        machines: List[Machine],
    ) -> pd.DataFrame:
        """
        Compare multiple schedules

        Args:
            schedules: Dictionary of {name: schedule}
            changeover_matrix: Changeover matrix
            skus: List of SKUs
            machines: List of machines

        Returns:
            DataFrame with comparison
        """
        import pandas as pd

        comparison_data = []

        for name, schedule in schedules.items():
            metrics = ScheduleMetrics.calculate_all_metrics(
                schedule, changeover_matrix, skus, machines
            )
            metrics["schedule_name"] = name
            metrics["optimizer"] = schedule.optimizer_used
            metrics["computation_time"] = schedule.computation_time

            comparison_data.append(metrics)

        df = pd.DataFrame(comparison_data)

        # Reorder columns
        priority_cols = [
            "schedule_name",
            "optimizer",
            "feasible",
            "total_lateness_hours",
            "total_tardiness_hours",
            "n_late_jobs",
            "on_time_rate",
            "makespan_hours",
            "total_setup_time_hours",
            "total_cost",
            "total_energy_kwh",
            "avg_utilization",
            "computation_time",
        ]

        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[priority_cols + other_cols]

        return df

    @staticmethod
    def calculate_improvement(baseline_metrics: Dict, optimized_metrics: Dict) -> Dict[str, float]:
        """
        Calculate improvement percentages

        Args:
            baseline_metrics: Metrics from baseline schedule
            optimized_metrics: Metrics from optimized schedule

        Returns:
            Dictionary of improvement percentages
        """
        improvements = {}

        improvement_metrics = [
            "total_lateness_hours",
            "total_tardiness_hours",
            "n_late_jobs",
            "makespan_hours",
            "total_setup_time_hours",
            "total_cost",
            "total_energy_kwh",
        ]

        for metric in improvement_metrics:
            if metric in baseline_metrics and metric in optimized_metrics:
                baseline_val = baseline_metrics[metric]
                optimized_val = optimized_metrics[metric]

                if baseline_val > 0:
                    improvement_pct = ((baseline_val - optimized_val) / baseline_val) * 100
                    improvements[f"{metric}_improvement_pct"] = improvement_pct

        # Utilization improvement (higher is better)
        if "avg_utilization" in baseline_metrics and "avg_utilization" in optimized_metrics:
            baseline_util = baseline_metrics["avg_utilization"]
            optimized_util = optimized_metrics["avg_utilization"]

            if baseline_util > 0:
                util_improvement = ((optimized_util - baseline_util) / baseline_util) * 100
                improvements["utilization_improvement_pct"] = util_improvement

        return improvements
