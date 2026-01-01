"""
Run optimization example

Demonstrates the complete MAPSO workflow:
1. Generate synthetic dataset
2. Create baseline schedules (FIFO, EDD, SPT, Priority)
3. Run CP-SAT optimization
4. Compare results
5. Display metrics
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import pandas as pd

from mapso.data.generator import DataGenerator
from mapso.agents.orchestrator import OrchestratorAgent
from mapso.agents.capacity_machine import CapacityMachineAgent
from mapso.agents.changeover_sequence import ChangeoverSequenceAgent
from mapso.optimization.layer1_cpsat import CPSATOptimizer
from mapso.evaluation.metrics import ScheduleMetrics, MetricsComparator
from mapso.core.enums import HeuristicType


def main():
    print("=" * 80)
    print("MAPSO - Multi-Agent Production Scheduling Optimizer")
    print("=" * 80)
    print()

    # Step 1: Generate synthetic dataset
    print("[1/5] Generating synthetic dataset...")
    generator = DataGenerator(random_seed=42)

    dataset = generator.generate_complete_dataset(
        n_skus=20,
        n_machines=5,
        n_jobs=50,  # Start with smaller problem
        historical_days=365,
        horizon_days=14,
    )

    skus = dataset["skus"]
    machines = dataset["machines"]
    changeover_matrix = dataset["changeover_matrix"]
    jobs = dataset["jobs"]

    print(f"  Generated: {len(skus)} SKUs, {len(machines)} machines, {len(jobs)} jobs")
    print()

    # Define time horizon
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=14)

    # Step 2: Create baseline schedules
    print("[2/5] Creating baseline schedules...")
    orchestrator = OrchestratorAgent()
    orchestrator.set_changeover_matrix(changeover_matrix)

    baseline_schedules = {}

    for heuristic in [HeuristicType.FIFO, HeuristicType.EDD, HeuristicType.SPT]:
        schedule = orchestrator.create_baseline_schedule(
            jobs, machines, start_date, end_date, heuristic
        )
        baseline_schedules[heuristic.value] = schedule
        print(f"  {heuristic.value}: {len(schedule.get_scheduled_jobs())} jobs scheduled")

    print()

    # Step 3: Run CP-SAT optimization
    print("[3/5] Running CP-SAT optimization...")
    cpsat_optimizer = CPSATOptimizer(config={"timeout": 60, "num_workers": 4})

    cpsat_schedule = cpsat_optimizer.optimize(
        jobs=jobs,
        machines=machines,
        skus=skus,
        changeover_matrix=changeover_matrix,
        start_date=start_date,
        end_date=end_date,
        objective_weights={
            "lateness": 0.4,
            "setup_time": 0.2,
            "cost": 0.2,
            "energy": 0.2,
        },
        timeout=60,
    )

    print(f"  CP-SAT: {len(cpsat_schedule.get_scheduled_jobs())} jobs scheduled")
    print()

    # Step 4: Calculate metrics for all schedules
    print("[4/5] Calculating metrics...")
    all_schedules = {**baseline_schedules, "cpsat": cpsat_schedule}

    metrics_data = []
    for name, schedule in all_schedules.items():
        metrics = ScheduleMetrics.calculate_all_metrics(
            schedule, changeover_matrix, skus, machines
        )
        metrics["schedule_name"] = name
        metrics["optimizer"] = schedule.optimizer_used
        metrics_data.append(metrics)

    metrics_df = pd.DataFrame(metrics_data)
    print()

    # Step 5: Display comparison
    print("[5/5] Results Comparison")
    print("=" * 80)
    print()

    # Select key metrics to display
    display_metrics = [
        "schedule_name",
        "n_scheduled_jobs",
        "total_lateness_hours",
        "n_late_jobs",
        "on_time_rate",
        "makespan_hours",
        "total_setup_time_hours",
        "avg_utilization",
        "total_cost",
    ]

    print(metrics_df[display_metrics].to_string(index=False))
    print()

    # Calculate improvements
    best_baseline = "edd"  # EDD usually performs well
    if best_baseline in baseline_schedules and "cpsat" in all_schedules:
        baseline_metrics = ScheduleMetrics.calculate_all_metrics(
            baseline_schedules[best_baseline], changeover_matrix, skus, machines
        )
        cpsat_metrics = ScheduleMetrics.calculate_all_metrics(
            cpsat_schedule, changeover_matrix, skus, machines
        )

        improvements = MetricsComparator.calculate_improvement(
            baseline_metrics, cpsat_metrics
        )

        print("Improvements (CP-SAT vs EDD Baseline):")
        print("-" * 80)
        for metric, improvement in improvements.items():
            print(f"  {metric}: {improvement:+.1f}%")
        print()

    print("=" * 80)
    print("Optimization complete!")
    print()
    print("Next steps:")
    print("  - Save dataset: dataset_generator.save_dataset(dataset, 'data/synthetic/sample')")
    print("  - Visualize schedule: Use Streamlit dashboard (coming soon)")
    print("  - Try different optimizers: Simulated Annealing, weight tuning")
    print("=" * 80)


if __name__ == "__main__":
    main()
