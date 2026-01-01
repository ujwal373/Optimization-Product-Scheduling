"""
Synthetic Data Generator for MAPSO

Generates realistic production scheduling datasets with:
- SKU catalog with diverse characteristics
- Machine pool with different capabilities
- Sequence-dependent changeover matrices
- Stochastic order book
- Historical demand with seasonality

The generated data follows realistic distributions observed in manufacturing environments.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import asdict

from mapso.core.models import (
    SKU,
    Machine,
    Job,
    ChangeoverMatrix,
    Shift,
    ShiftCalendar,
    DemandForecast,
    ShiftType,
)
from mapso.constants import (
    DEFAULT_N_SKUS,
    DEFAULT_N_MACHINES,
    DEFAULT_N_JOBS,
    DEFAULT_HISTORICAL_DAYS,
    DEFAULT_RANDOM_SEED,
    CHANGEOVER_LOW_MIN,
    CHANGEOVER_LOW_MAX,
    CHANGEOVER_MEDIUM_MIN,
    CHANGEOVER_MEDIUM_MAX,
    CHANGEOVER_HIGH_MIN,
    CHANGEOVER_HIGH_MAX,
)


class DataGenerator:
    """
    Generates synthetic production scheduling datasets

    Attributes:
        random_seed: Random seed for reproducibility
        rng: NumPy random number generator
    """

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED):
        """
        Initialize data generator

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def generate_skus(
        self,
        n_skus: int = DEFAULT_N_SKUS,
        product_families: Optional[List[str]] = None,
    ) -> List[SKU]:
        """
        Generate SKU catalog

        Processing times follow log-normal distribution (realistic for manufacturing).
        SKUs are grouped into product families for changeover modeling.

        Args:
            n_skus: Number of SKUs to generate
            product_families: List of product family names (auto-generated if None)

        Returns:
            List of SKU objects
        """
        if product_families is None:
            # Create 4-6 product families
            n_families = self.rng.randint(4, 7)
            product_families = [f"Family_{chr(65+i)}" for i in range(n_families)]

        skus = []

        for i in range(n_skus):
            # Log-normal distribution for processing time (0.1 to 10 hours/unit)
            # Mean: 1 hour, Std: 2 hours
            processing_time = self.rng.lognormal(mean=0.0, sigma=1.0)
            processing_time = max(0.1, min(10.0, processing_time))

            # Batch sizes: typically 10-500 units
            batch_size = int(self.rng.lognormal(mean=3.5, sigma=1.0))
            batch_size = max(10, min(500, batch_size))

            # Priority: uniform distribution 1-5
            priority = self.rng.randint(1, 6)

            # Cost per unit: $10-$500, correlated with processing time
            cost_per_unit = 50 + processing_time * self.rng.uniform(20, 80)

            # Energy per unit: 0.5-10 kWh, correlated with processing time
            energy_per_unit = processing_time * self.rng.uniform(0.5, 2.0)

            # Assign to product family
            family = product_families[i % len(product_families)]

            sku = SKU(
                sku_id=f"SKU_{i+1:03d}",
                name=f"Product {chr(65 + (i % 26))}{i//26 + 1}",
                processing_time_per_unit=round(processing_time, 2),
                batch_size=batch_size,
                priority=priority,
                cost_per_unit=round(cost_per_unit, 2),
                energy_per_unit=round(energy_per_unit, 2),
                product_family=family,
            )
            skus.append(sku)

        return skus

    def generate_machines(
        self,
        n_machines: int = DEFAULT_N_MACHINES,
        skus: Optional[List[SKU]] = None,
    ) -> List[Machine]:
        """
        Generate machine pool

        Machine capabilities:
        - 70% multi-SKU machines (can produce multiple SKUs)
        - 30% specialist machines (limited SKU set)

        Shifts:
        - 40% single-shift (8 hours)
        - 40% two-shift (16 hours)
        - 20% three-shift (24/7)

        Args:
            n_machines: Number of machines to generate
            skus: List of SKUs (auto-generated if None)

        Returns:
            List of Machine objects
        """
        if skus is None:
            skus = self.generate_skus()

        sku_ids = [sku.sku_id for sku in skus]
        machines = []

        for i in range(n_machines):
            # Machine capacity: 50-200 units/hour
            capacity = self.rng.uniform(50, 200)

            # Determine if multi-SKU or specialist
            is_specialist = self.rng.random() < 0.3

            if is_specialist:
                # Specialist: can produce 20-40% of SKUs
                n_available = max(1, int(len(sku_ids) * self.rng.uniform(0.2, 0.4)))
                available_skus = self.rng.choice(sku_ids, size=n_available, replace=False).tolist()
            else:
                # Multi-SKU: can produce 60-100% of SKUs
                n_available = max(1, int(len(sku_ids) * self.rng.uniform(0.6, 1.0)))
                available_skus = self.rng.choice(sku_ids, size=n_available, replace=False).tolist()

            # Generate shift calendar
            shift_calendar = self._generate_shift_calendar(i)

            # Maintenance windows: 5-10% downtime
            maintenance_windows = self._generate_maintenance_windows()

            # Energy efficiency: 0.7-1.0 (newer machines more efficient)
            energy_efficiency = self.rng.uniform(0.7, 1.0)

            machine = Machine(
                machine_id=f"M{i+1:02d}",
                name=f"Machine {chr(65 + (i % 26))}{i//26 + 1}",
                capacity_per_hour=round(capacity, 1),
                available_skus=available_skus,
                shift_calendar=shift_calendar,
                maintenance_windows=maintenance_windows,
                energy_efficiency=round(energy_efficiency, 2),
            )
            machines.append(machine)

        return machines

    def _generate_shift_calendar(self, machine_index: int) -> ShiftCalendar:
        """Generate shift calendar for a machine"""
        # Distribution: 40% 1-shift, 40% 2-shift, 20% 3-shift
        rand = self.rng.random()

        if rand < 0.4:
            # Single shift (8 hours): 06:00-14:00
            shifts = [
                Shift(
                    shift_id="shift_1",
                    shift_type=ShiftType.MORNING,
                    start_time="06:00",
                    end_time="14:00",
                    days_of_week=list(range(5)),  # Mon-Fri
                    overtime_multiplier=1.0,
                    energy_cost_multiplier=1.0,
                )
            ]
        elif rand < 0.8:
            # Two shifts (16 hours)
            shifts = [
                Shift(
                    shift_id="shift_1",
                    shift_type=ShiftType.MORNING,
                    start_time="06:00",
                    end_time="14:00",
                    days_of_week=list(range(5)),
                    overtime_multiplier=1.0,
                    energy_cost_multiplier=1.0,
                ),
                Shift(
                    shift_id="shift_2",
                    shift_type=ShiftType.AFTERNOON,
                    start_time="14:00",
                    end_time="22:00",
                    days_of_week=list(range(5)),
                    overtime_multiplier=1.2,
                    energy_cost_multiplier=1.3,  # Peak hours
                ),
            ]
        else:
            # Three shifts (24/7)
            shifts = [
                Shift(
                    shift_id="shift_1",
                    shift_type=ShiftType.MORNING,
                    start_time="06:00",
                    end_time="14:00",
                    days_of_week=list(range(7)),  # All days
                    overtime_multiplier=1.0,
                    energy_cost_multiplier=1.0,
                ),
                Shift(
                    shift_id="shift_2",
                    shift_type=ShiftType.AFTERNOON,
                    start_time="14:00",
                    end_time="22:00",
                    days_of_week=list(range(7)),
                    overtime_multiplier=1.2,
                    energy_cost_multiplier=1.3,
                ),
                Shift(
                    shift_id="shift_3",
                    shift_type=ShiftType.NIGHT,
                    start_time="22:00",
                    end_time="06:00",
                    days_of_week=list(range(7)),
                    overtime_multiplier=1.5,
                    energy_cost_multiplier=0.7,  # Off-peak
                ),
            ]

        return ShiftCalendar(shifts=shifts)

    def _generate_maintenance_windows(
        self, start_date: Optional[datetime] = None, days: int = 30
    ) -> List[Tuple[datetime, datetime]]:
        """Generate maintenance windows for a machine"""
        if start_date is None:
            start_date = datetime.now()

        windows = []

        # 1-3 maintenance windows per month
        n_windows = self.rng.randint(1, 4)

        for _ in range(n_windows):
            # Random day within period
            day_offset = self.rng.randint(0, days)
            maint_start = start_date + timedelta(days=day_offset, hours=self.rng.randint(0, 20))

            # Maintenance duration: 2-8 hours
            duration = self.rng.uniform(2, 8)
            maint_end = maint_start + timedelta(hours=duration)

            windows.append((maint_start, maint_end))

        return windows

    def generate_changeover_matrix(self, skus: List[SKU]) -> ChangeoverMatrix:
        """
        Generate sequence-dependent changeover matrix

        Setup times depend on product family similarity:
        - Same family: Low setup (5-30 min)
        - Different family: Medium setup (30-120 min)
        - Some pairs: High setup (120-480 min)

        Matrix is asymmetric: time(A→B) ≠ time(B→A)

        Args:
            skus: List of SKUs

        Returns:
            ChangeoverMatrix object
        """
        n_skus = len(skus)
        sku_ids = [sku.sku_id for sku in skus]

        # Initialize matrices
        setup_times = np.zeros((n_skus, n_skus))
        setup_costs = np.zeros((n_skus, n_skus))

        for i in range(n_skus):
            for j in range(n_skus):
                if i == j:
                    # No setup time from SKU to itself
                    setup_times[i, j] = 0
                    setup_costs[i, j] = 0
                else:
                    # Determine setup time based on product family
                    same_family = skus[i].product_family == skus[j].product_family

                    if same_family:
                        # Low setup time
                        setup_time = self.rng.uniform(CHANGEOVER_LOW_MIN, CHANGEOVER_LOW_MAX)
                    else:
                        # Check if major changeover (10% probability)
                        if self.rng.random() < 0.1:
                            # High setup time
                            setup_time = self.rng.uniform(
                                CHANGEOVER_HIGH_MIN, CHANGEOVER_HIGH_MAX
                            )
                        else:
                            # Medium setup time
                            setup_time = self.rng.uniform(
                                CHANGEOVER_MEDIUM_MIN, CHANGEOVER_MEDIUM_MAX
                            )

                    setup_times[i, j] = setup_time

                    # Setup cost: $50-500, correlated with time
                    setup_cost = 50 + setup_time * self.rng.uniform(0.5, 2.0)
                    setup_costs[i, j] = setup_cost

        return ChangeoverMatrix(
            skus=sku_ids, setup_times=setup_times, setup_costs=setup_costs
        )

    def generate_orders(
        self,
        n_jobs: int = DEFAULT_N_JOBS,
        skus: Optional[List[SKU]] = None,
        start_date: Optional[datetime] = None,
        horizon_days: int = 30,
    ) -> List[Job]:
        """
        Generate order book (jobs)

        Orders arrive following a Poisson process with exponential inter-arrival times.
        Quantities follow log-normal distribution.
        Due dates: release_date + Uniform(1, horizon_days) days

        Args:
            n_jobs: Number of jobs to generate
            skus: List of SKUs (auto-generated if None)
            start_date: Start date for orders
            horizon_days: Planning horizon in days

        Returns:
            List of Job objects
        """
        if skus is None:
            skus = self.generate_skus()

        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        jobs = []

        # Poisson arrival: mean inter-arrival time
        mean_inter_arrival_hours = (horizon_days * 24) / n_jobs

        current_time = start_date

        for i in range(n_jobs):
            # Exponential inter-arrival time
            inter_arrival = self.rng.exponential(mean_inter_arrival_hours)
            release_date = current_time + timedelta(hours=inter_arrival)

            # Random SKU (weighted by priority - higher priority SKUs more frequent)
            sku_weights = np.array([1.0 / sku.priority for sku in skus])
            sku_weights /= sku_weights.sum()
            sku = self.rng.choice(skus, p=sku_weights)

            # Quantity: log-normal distribution, typical batch size as mean
            quantity = int(self.rng.lognormal(mean=np.log(sku.batch_size), sigma=0.5))
            quantity = max(1, quantity)

            # Due date: release + uniform(1, horizon_days) days
            days_to_due = self.rng.uniform(1, horizon_days)
            due_date = release_date + timedelta(days=days_to_due)

            # Customer ID
            customer_id = f"CUST_{self.rng.randint(1, 51):03d}"

            job = Job(
                job_id=f"JOB_{i+1:04d}",
                sku_id=sku.sku_id,
                quantity=quantity,
                due_date=due_date,
                release_date=release_date,
                priority=sku.priority,
                customer_id=customer_id,
            )
            jobs.append(job)

            current_time = release_date

        return jobs

    def generate_historical_demand(
        self,
        skus: List[SKU],
        days: int = DEFAULT_HISTORICAL_DAYS,
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate historical demand time series

        Demand components:
        - Trend: Linear growth/decline
        - Weekly seasonality: Higher Mon-Fri, lower weekends
        - Monthly seasonality: End-of-month peaks
        - Random noise: Normal(0, 15%)
        - Special events: 20% demand spikes (5% probability)

        Args:
            skus: List of SKUs
            days: Number of historical days
            start_date: Start date

        Returns:
            DataFrame with columns: date, sku_id, demand
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)

        records = []

        for sku in skus:
            # Base demand: 50-500 units/day
            base_demand = self.rng.uniform(50, 500)

            # Trend: ±2% per month
            trend_rate = self.rng.uniform(-0.02, 0.02)

            for day in range(days):
                date = start_date + timedelta(days=day)

                # Trend component
                trend = base_demand * (1 + trend_rate * (day / 30))

                # Weekly seasonality
                day_of_week = date.weekday()
                if day_of_week < 5:  # Mon-Fri
                    weekly_factor = 1.2
                else:  # Weekend
                    weekly_factor = 0.6

                # Monthly seasonality (end-of-month peak)
                day_of_month = date.day
                if day_of_month > 25:
                    monthly_factor = 1.3
                elif day_of_month < 5:
                    monthly_factor = 0.9
                else:
                    monthly_factor = 1.0

                # Random noise: ±15%
                noise = self.rng.normal(1.0, 0.15)

                # Special events: 5% probability of 20% spike
                if self.rng.random() < 0.05:
                    event_factor = 1.2
                else:
                    event_factor = 1.0

                # Combine all factors
                demand = trend * weekly_factor * monthly_factor * noise * event_factor
                demand = max(0, int(demand))

                records.append(
                    {"date": date, "sku_id": sku.sku_id, "demand": demand, "sku_name": sku.name}
                )

        df = pd.DataFrame(records)
        return df

    def generate_complete_dataset(
        self,
        n_skus: int = DEFAULT_N_SKUS,
        n_machines: int = DEFAULT_N_MACHINES,
        n_jobs: int = DEFAULT_N_JOBS,
        historical_days: int = DEFAULT_HISTORICAL_DAYS,
        horizon_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate complete dataset for scheduling problem

        Args:
            n_skus: Number of SKUs
            n_machines: Number of machines
            n_jobs: Number of jobs
            historical_days: Days of historical demand
            horizon_days: Planning horizon in days

        Returns:
            Dictionary containing all dataset components
        """
        # Generate SKUs
        skus = self.generate_skus(n_skus=n_skus)

        # Generate machines
        machines = self.generate_machines(n_machines=n_machines, skus=skus)

        # Generate changeover matrix
        changeover_matrix = self.generate_changeover_matrix(skus)

        # Generate jobs
        jobs = self.generate_orders(n_jobs=n_jobs, skus=skus, horizon_days=horizon_days)

        # Generate historical demand
        historical_demand = self.generate_historical_demand(skus, days=historical_days)

        return {
            "skus": skus,
            "machines": machines,
            "changeover_matrix": changeover_matrix,
            "jobs": jobs,
            "historical_demand": historical_demand,
            "metadata": {
                "n_skus": n_skus,
                "n_machines": n_machines,
                "n_jobs": n_jobs,
                "historical_days": historical_days,
                "horizon_days": horizon_days,
                "random_seed": self.random_seed,
                "generated_at": datetime.now().isoformat(),
            },
        }

    def save_dataset(self, dataset: Dict[str, Any], output_path: str) -> None:
        """
        Save dataset to files

        Args:
            dataset: Dataset dictionary from generate_complete_dataset()
            output_path: Directory path to save files
        """
        import os
        import json

        os.makedirs(output_path, exist_ok=True)

        # Save SKUs
        skus_data = [asdict(sku) for sku in dataset["skus"]]
        with open(os.path.join(output_path, "skus.json"), "w") as f:
            json.dump(skus_data, f, indent=2)

        # Save machines (without shift calendar objects - need custom serialization)
        machines_data = []
        for machine in dataset["machines"]:
            machine_dict = {
                "machine_id": machine.machine_id,
                "name": machine.name,
                "capacity_per_hour": machine.capacity_per_hour,
                "available_skus": machine.available_skus,
                "energy_efficiency": machine.energy_efficiency,
                "n_shifts": len(machine.shift_calendar.shifts),
                "weekly_capacity_hours": machine.shift_calendar.get_weekly_capacity_hours(),
            }
            machines_data.append(machine_dict)

        with open(os.path.join(output_path, "machines.json"), "w") as f:
            json.dump(machines_data, f, indent=2)

        # Save changeover matrix
        changeover_data = {
            "skus": dataset["changeover_matrix"].skus,
            "setup_times": dataset["changeover_matrix"].setup_times.tolist(),
            "setup_costs": dataset["changeover_matrix"].setup_costs.tolist(),
        }
        with open(os.path.join(output_path, "changeover_matrix.json"), "w") as f:
            json.dump(changeover_data, f, indent=2)

        # Save jobs
        jobs_data = []
        for job in dataset["jobs"]:
            job_dict = {
                "job_id": job.job_id,
                "sku_id": job.sku_id,
                "quantity": job.quantity,
                "due_date": job.due_date.isoformat(),
                "release_date": job.release_date.isoformat(),
                "priority": job.priority,
                "customer_id": job.customer_id,
            }
            jobs_data.append(job_dict)

        with open(os.path.join(output_path, "jobs.json"), "w") as f:
            json.dump(jobs_data, f, indent=2)

        # Save historical demand as CSV
        dataset["historical_demand"].to_csv(
            os.path.join(output_path, "historical_demand.csv"), index=False
        )

        # Save metadata
        with open(os.path.join(output_path, "metadata.json"), "w") as f:
            json.dump(dataset["metadata"], f, indent=2)

        print(f"Dataset saved to {output_path}")
        print(f"  - {len(dataset['skus'])} SKUs")
        print(f"  - {len(dataset['machines'])} Machines")
        print(f"  - {len(dataset['jobs'])} Jobs")
        print(f"  - {len(dataset['historical_demand'])} Historical demand records")
