"""
Core domain models for MAPSO

Defines the fundamental data structures used throughout the scheduling system:
- SKU: Stock Keeping Unit (product types)
- Machine: Production machines/lines
- Job: Production orders
- Schedule: Complete production schedules
- ChangeoverMatrix: Sequence-dependent setup times
- Shift/ShiftCalendar: Factory shift patterns
- DemandForecast: Forecasted demand

All models use dataclasses for clean, type-safe data structures with automatic
initialization, repr, and comparison methods.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import uuid


class JobStatus(Enum):
    """Status of a production job"""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


class MachineState(Enum):
    """State of a production machine"""

    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    DOWN = "down"


class ShiftType(Enum):
    """Type of work shift"""

    MORNING = "morning"  # e.g., 6am-2pm
    AFTERNOON = "afternoon"  # e.g., 2pm-10pm
    NIGHT = "night"  # e.g., 10pm-6am
    FULL_DAY = "full_day"  # 24/7


@dataclass
class SKU:
    """
    Stock Keeping Unit - represents a product type

    Attributes:
        sku_id: Unique identifier
        name: Product name
        processing_time_per_unit: Time to produce one unit (hours)
        batch_size: Typical batch size for production
        priority: Priority level (1=highest, 5=lowest)
        cost_per_unit: Production cost per unit ($)
        energy_per_unit: Energy consumption per unit (kWh)
        product_family: Product family/category (for changeover grouping)
    """

    sku_id: str
    name: str
    processing_time_per_unit: float
    batch_size: int
    priority: int = 3
    cost_per_unit: float = 0.0
    energy_per_unit: float = 0.0
    product_family: Optional[str] = None

    def __post_init__(self):
        """Validate SKU attributes"""
        if self.processing_time_per_unit <= 0:
            raise ValueError("Processing time must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not (1 <= self.priority <= 5):
            raise ValueError("Priority must be between 1 and 5")

    def total_processing_time(self, quantity: int) -> float:
        """
        Calculate total processing time for a given quantity

        Args:
            quantity: Number of units to produce

        Returns:
            Total processing time in hours
        """
        return self.processing_time_per_unit * quantity

    def total_cost(self, quantity: int) -> float:
        """Calculate total production cost for quantity"""
        return self.cost_per_unit * quantity

    def total_energy(self, quantity: int) -> float:
        """Calculate total energy consumption for quantity"""
        return self.energy_per_unit * quantity


@dataclass
class Shift:
    """
    Represents a work shift

    Attributes:
        shift_id: Unique identifier
        shift_type: Type of shift
        start_time: Start time (HH:MM format)
        end_time: End time (HH:MM format)
        days_of_week: Days when shift operates (0=Monday, 6=Sunday)
        overtime_multiplier: Cost multiplier for this shift
        energy_cost_multiplier: Energy cost multiplier (peak/off-peak)
    """

    shift_id: str
    shift_type: ShiftType
    start_time: str  # "HH:MM"
    end_time: str  # "HH:MM"
    days_of_week: List[int] = field(default_factory=lambda: list(range(5)))  # Mon-Fri
    overtime_multiplier: float = 1.0
    energy_cost_multiplier: float = 1.0

    def __post_init__(self):
        """Validate shift attributes"""
        # Validate time format
        try:
            time.fromisoformat(self.start_time)
            time.fromisoformat(self.end_time)
        except ValueError:
            raise ValueError("Invalid time format. Use HH:MM")

        # Validate days of week
        if not all(0 <= day <= 6 for day in self.days_of_week):
            raise ValueError("Days of week must be between 0 (Monday) and 6 (Sunday)")

    def duration_hours(self) -> float:
        """Calculate shift duration in hours"""
        start = time.fromisoformat(self.start_time)
        end = time.fromisoformat(self.end_time)

        start_minutes = start.hour * 60 + start.minute
        end_minutes = end.hour * 60 + end.minute

        # Handle overnight shifts
        if end_minutes < start_minutes:
            end_minutes += 24 * 60

        return (end_minutes - start_minutes) / 60


@dataclass
class ShiftCalendar:
    """
    Collection of shifts for a machine

    Attributes:
        shifts: List of shifts
        timezone: Timezone for shift times
    """

    shifts: List[Shift]
    timezone: str = "UTC"

    def get_available_hours_per_day(self, day_of_week: int) -> float:
        """
        Calculate available hours for a specific day of week

        Args:
            day_of_week: 0=Monday, 6=Sunday

        Returns:
            Total available hours
        """
        total_hours = 0.0
        for shift in self.shifts:
            if day_of_week in shift.days_of_week:
                total_hours += shift.duration_hours()
        return total_hours

    def get_weekly_capacity_hours(self) -> float:
        """Calculate total weekly capacity in hours"""
        return sum(self.get_available_hours_per_day(day) for day in range(7))


@dataclass
class Machine:
    """
    Production machine or line

    Attributes:
        machine_id: Unique identifier
        name: Machine name
        capacity_per_hour: Production capacity (units/hour)
        available_skus: List of SKU IDs this machine can produce
        shift_calendar: Operating shifts
        maintenance_windows: Scheduled maintenance periods
        energy_efficiency: Energy efficiency factor (0-1, 1=most efficient)
        state: Current state of machine
    """

    machine_id: str
    name: str
    capacity_per_hour: float
    available_skus: List[str]
    shift_calendar: ShiftCalendar
    maintenance_windows: List[Tuple[datetime, datetime]] = field(default_factory=list)
    energy_efficiency: float = 1.0
    state: MachineState = MachineState.AVAILABLE

    def __post_init__(self):
        """Validate machine attributes"""
        if self.capacity_per_hour <= 0:
            raise ValueError("Capacity must be positive")
        if not (0 < self.energy_efficiency <= 1):
            raise ValueError("Energy efficiency must be between 0 and 1")

    def can_produce(self, sku_id: str) -> bool:
        """Check if machine can produce a specific SKU"""
        return sku_id in self.available_skus

    def is_available(self, start: datetime, end: datetime) -> bool:
        """
        Check if machine is available during a time window

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            True if available (not in maintenance)
        """
        for maint_start, maint_end in self.maintenance_windows:
            # Check for overlap
            if start < maint_end and end > maint_start:
                return False
        return True


@dataclass
class ChangeoverMatrix:
    """
    Sequence-dependent setup times between SKUs

    Represents the time and cost required to change production from one SKU to another.
    Matrix is typically asymmetric (A→B ≠ B→A).

    Attributes:
        skus: List of SKU IDs
        setup_times: Setup time matrix (minutes) [from_sku, to_sku]
        setup_costs: Setup cost matrix ($) [from_sku, to_sku]
    """

    skus: List[str]
    setup_times: np.ndarray  # shape: (n_skus, n_skus)
    setup_costs: np.ndarray  # shape: (n_skus, n_skus)

    def __post_init__(self):
        """Validate changeover matrix"""
        n_skus = len(self.skus)

        if self.setup_times.shape != (n_skus, n_skus):
            raise ValueError(f"Setup times shape must be ({n_skus}, {n_skus})")
        if self.setup_costs.shape != (n_skus, n_skus):
            raise ValueError(f"Setup costs shape must be ({n_skus}, {n_skus})")

        # Zero diagonal (no setup time from SKU to itself)
        np.fill_diagonal(self.setup_times, 0)
        np.fill_diagonal(self.setup_costs, 0)

    def get_setup_time(self, from_sku: str, to_sku: str) -> float:
        """
        Get setup time from one SKU to another

        Args:
            from_sku: Source SKU ID
            to_sku: Target SKU ID

        Returns:
            Setup time in minutes
        """
        try:
            i = self.skus.index(from_sku)
            j = self.skus.index(to_sku)
            return float(self.setup_times[i, j])
        except ValueError:
            raise ValueError(f"SKU not found in changeover matrix: {from_sku} or {to_sku}")

    def get_setup_cost(self, from_sku: str, to_sku: str) -> float:
        """Get setup cost from one SKU to another"""
        try:
            i = self.skus.index(from_sku)
            j = self.skus.index(to_sku)
            return float(self.setup_costs[i, j])
        except ValueError:
            raise ValueError(f"SKU not found in changeover matrix: {from_sku} or {to_sku}")

    def get_average_setup_time(self) -> float:
        """Calculate average non-zero setup time"""
        non_zero = self.setup_times[self.setup_times > 0]
        return float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0


@dataclass
class Job:
    """
    Production job (order)

    Attributes:
        job_id: Unique identifier
        sku_id: SKU to produce
        quantity: Number of units
        due_date: Due date
        release_date: Earliest start date
        priority: Job priority (1-5)
        customer_id: Customer identifier
        status: Current status
        assigned_machine: Machine ID (after scheduling)
        scheduled_start: Scheduled start time
        scheduled_end: Scheduled end time
        actual_start: Actual start time
        actual_end: Actual completion time
    """

    job_id: str
    sku_id: str
    quantity: int
    due_date: datetime
    release_date: datetime
    priority: int = 3
    customer_id: str = "default"
    status: JobStatus = JobStatus.PENDING

    # Scheduling results
    assigned_machine: Optional[str] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None

    # Actual execution
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None

    def __post_init__(self):
        """Validate job attributes"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.release_date > self.due_date:
            raise ValueError("Release date must be before due date")
        if not (1 <= self.priority <= 5):
            raise ValueError("Priority must be between 1 and 5")

    def is_scheduled(self) -> bool:
        """Check if job has been scheduled"""
        return self.assigned_machine is not None and self.scheduled_start is not None

    def lateness(self) -> float:
        """
        Calculate lateness in hours

        Returns:
            Lateness (positive if late, negative if early, 0 if on time)
        """
        if self.scheduled_end is None:
            return 0.0

        delta = self.scheduled_end - self.due_date
        return delta.total_seconds() / 3600  # Convert to hours

    def tardiness(self) -> float:
        """
        Calculate tardiness in hours (only positive lateness)

        Returns:
            Tardiness (0 if on time or early)
        """
        return max(0.0, self.lateness())

    def is_late(self) -> bool:
        """Check if job is late"""
        return self.tardiness() > 0


@dataclass
class Schedule:
    """
    Complete production schedule

    Attributes:
        schedule_id: Unique identifier
        jobs: List of jobs in schedule
        machines: List of machines
        start_date: Schedule start date
        end_date: Schedule end date

        # Computed metrics
        total_lateness: Total lateness (hours)
        total_tardiness: Total tardiness (hours)
        total_setup_time: Total changeover time (hours)
        total_cost: Total cost ($)
        total_energy: Total energy consumption (kWh)
        makespan: Total schedule duration (hours)
        feasible: Whether schedule satisfies all constraints

        # Metadata
        optimizer_used: Name of optimizer that created this schedule
        computation_time: Time taken to generate schedule (seconds)
        objective_weights: Weights used in objective function
        objective_value: Final objective function value
    """

    schedule_id: str
    jobs: List[Job]
    machines: List[Machine]
    start_date: datetime
    end_date: datetime

    # Computed metrics (updated after scheduling)
    total_lateness: float = 0.0
    total_tardiness: float = 0.0
    n_late_jobs: int = 0
    total_setup_time: float = 0.0
    total_cost: float = 0.0
    total_energy: float = 0.0
    makespan: float = 0.0
    feasible: bool = True
    constraint_violations: List[str] = field(default_factory=list)

    # Metadata
    optimizer_used: str = "none"
    computation_time: float = 0.0
    objective_weights: Dict[str, float] = field(default_factory=dict)
    objective_value: float = float("inf")
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize schedule ID if not provided"""
        if not self.schedule_id:
            self.schedule_id = str(uuid.uuid4())

    def get_jobs_by_machine(self, machine_id: str) -> List[Job]:
        """Get all jobs assigned to a specific machine"""
        return [job for job in self.jobs if job.assigned_machine == machine_id]

    def get_scheduled_jobs(self) -> List[Job]:
        """Get all jobs that have been scheduled"""
        return [job for job in self.jobs if job.is_scheduled()]

    def get_unscheduled_jobs(self) -> List[Job]:
        """Get all jobs that haven't been scheduled"""
        return [job for job in self.jobs if not job.is_scheduled()]

    def calculate_utilization(self, machine_id: str) -> float:
        """
        Calculate utilization rate for a machine

        Args:
            machine_id: Machine ID

        Returns:
            Utilization rate (0-1)
        """
        machine_jobs = self.get_jobs_by_machine(machine_id)
        if not machine_jobs:
            return 0.0

        # Total production time
        total_production = sum(
            (job.scheduled_end - job.scheduled_start).total_seconds() / 3600
            for job in machine_jobs
            if job.scheduled_start and job.scheduled_end
        )

        # Available time
        schedule_duration = (self.end_date - self.start_date).total_seconds() / 3600

        return total_production / schedule_duration if schedule_duration > 0 else 0.0

    def is_feasible(self) -> bool:
        """Check if schedule is feasible (all constraints satisfied)"""
        return self.feasible and len(self.constraint_violations) == 0

    def summary(self) -> Dict[str, any]:
        """Get summary statistics for schedule"""
        scheduled_jobs = self.get_scheduled_jobs()
        late_jobs = [job for job in scheduled_jobs if job.is_late()]

        return {
            "schedule_id": self.schedule_id,
            "total_jobs": len(self.jobs),
            "scheduled_jobs": len(scheduled_jobs),
            "unscheduled_jobs": len(self.get_unscheduled_jobs()),
            "late_jobs": len(late_jobs),
            "on_time_rate": 1 - (len(late_jobs) / len(scheduled_jobs))
            if scheduled_jobs
            else 0.0,
            "total_lateness_hours": self.total_lateness,
            "total_tardiness_hours": self.total_tardiness,
            "makespan_hours": self.makespan,
            "total_setup_time_hours": self.total_setup_time,
            "total_cost": self.total_cost,
            "total_energy_kwh": self.total_energy,
            "feasible": self.feasible,
            "optimizer": self.optimizer_used,
            "computation_time_seconds": self.computation_time,
            "objective_value": self.objective_value,
        }


@dataclass
class DemandForecast:
    """
    Demand forecast for a SKU

    Attributes:
        sku_id: SKU identifier
        forecast_date: Date of forecast
        predicted_quantity: Predicted demand quantity
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: Forecasting method used
    """

    sku_id: str
    forecast_date: datetime
    predicted_quantity: float
    lower_bound: float
    upper_bound: float
    confidence: float = 0.95
    method: str = "unknown"

    def __post_init__(self):
        """Validate forecast attributes"""
        if self.predicted_quantity < 0:
            raise ValueError("Predicted quantity cannot be negative")
        if not (0 < self.confidence < 1):
            raise ValueError("Confidence must be between 0 and 1")
        if self.lower_bound > self.predicted_quantity:
            raise ValueError("Lower bound must be <= predicted quantity")
        if self.upper_bound < self.predicted_quantity:
            raise ValueError("Upper bound must be >= predicted quantity")

    def uncertainty_range(self) -> float:
        """Calculate uncertainty range"""
        return self.upper_bound - self.lower_bound

    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV)"""
        if self.predicted_quantity == 0:
            return float("inf")
        return self.uncertainty_range() / self.predicted_quantity
