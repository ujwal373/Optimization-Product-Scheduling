"""
Constraint definitions for production scheduling

Defines constraint types and validation logic used by agents and optimizers.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable, Any
from datetime import datetime
from mapso.core.enums import ConstraintType, ValidationResult


@dataclass
class Constraint:
    """
    Base constraint definition

    Attributes:
        constraint_id: Unique identifier
        constraint_type: Type of constraint
        name: Human-readable name
        description: Detailed description
        hard: Whether constraint is hard (must satisfy) or soft (prefer to satisfy)
        penalty_weight: Weight for soft constraint violations
        validator: Function to validate constraint
    """

    constraint_id: str
    constraint_type: ConstraintType
    name: str
    description: str
    hard: bool = True
    penalty_weight: float = 1.0
    validator: Optional[Callable] = None

    def validate(self, schedule: Any) -> tuple[ValidationResult, str]:
        """
        Validate constraint against a schedule

        Args:
            schedule: Schedule to validate

        Returns:
            Tuple of (ValidationResult, message)
        """
        if self.validator is None:
            return (ValidationResult.UNKNOWN, "No validator function provided")

        try:
            is_valid, message = self.validator(schedule)
            result = ValidationResult.VALID if is_valid else ValidationResult.INVALID
            return (result, message)
        except Exception as e:
            return (ValidationResult.ERROR, f"Validation error: {str(e)}")


# Predefined constraint templates


@dataclass
class AssignmentConstraint(Constraint):
    """Each job must be assigned to exactly one machine"""

    def __init__(self):
        super().__init__(
            constraint_id="assign_001",
            constraint_type=ConstraintType.ASSIGNMENT,
            name="Job Assignment",
            description="Each job must be assigned to exactly one machine that can produce its SKU",
            hard=True,
        )


@dataclass
class CapacityConstraint(Constraint):
    """Machine capacity must not be exceeded"""

    def __init__(self, machine_id: str, max_capacity: float):
        super().__init__(
            constraint_id=f"capacity_{machine_id}",
            constraint_type=ConstraintType.CAPACITY,
            name=f"Capacity - {machine_id}",
            description=f"Machine {machine_id} capacity must not exceed {max_capacity}",
            hard=True,
        )
        self.machine_id = machine_id
        self.max_capacity = max_capacity


@dataclass
class TemporalConstraint(Constraint):
    """Time-related constraints"""

    def __init__(
        self,
        constraint_id: str,
        name: str,
        description: str,
        earliest: Optional[datetime] = None,
        latest: Optional[datetime] = None,
    ):
        super().__init__(
            constraint_id=constraint_id,
            constraint_type=ConstraintType.TEMPORAL,
            name=name,
            description=description,
            hard=True,
        )
        self.earliest = earliest
        self.latest = latest


@dataclass
class NoOverlapConstraint(Constraint):
    """Jobs on same machine must not overlap"""

    def __init__(self, machine_id: str):
        super().__init__(
            constraint_id=f"no_overlap_{machine_id}",
            constraint_type=ConstraintType.TEMPORAL,
            name=f"No Overlap - {machine_id}",
            description=f"Jobs on machine {machine_id} must not overlap in time",
            hard=True,
        )
        self.machine_id = machine_id


@dataclass
class ReleaseDateConstraint(Constraint):
    """Job cannot start before release date"""

    def __init__(self, job_id: str, release_date: datetime):
        super().__init__(
            constraint_id=f"release_{job_id}",
            constraint_type=ConstraintType.TEMPORAL,
            name=f"Release Date - {job_id}",
            description=f"Job {job_id} cannot start before {release_date}",
            hard=True,
        )
        self.job_id = job_id
        self.release_date = release_date


@dataclass
class DueDateConstraint(Constraint):
    """Job should finish by due date (soft constraint)"""

    def __init__(self, job_id: str, due_date: datetime, penalty_per_hour: float = 100.0):
        super().__init__(
            constraint_id=f"due_{job_id}",
            constraint_type=ConstraintType.TEMPORAL,
            name=f"Due Date - {job_id}",
            description=f"Job {job_id} should finish by {due_date}",
            hard=False,
            penalty_weight=penalty_per_hour,
        )
        self.job_id = job_id
        self.due_date = due_date


@dataclass
class ShiftConstraint(Constraint):
    """Job must execute within shift windows"""

    def __init__(self, machine_id: str):
        super().__init__(
            constraint_id=f"shift_{machine_id}",
            constraint_type=ConstraintType.SHIFT,
            name=f"Shift Availability - {machine_id}",
            description=f"Jobs on machine {machine_id} must execute during available shifts",
            hard=True,
        )
        self.machine_id = machine_id


@dataclass
class ChangeoverConstraint(Constraint):
    """Sequence-dependent setup times must be accounted for"""

    def __init__(self):
        super().__init__(
            constraint_id="changeover_001",
            constraint_type=ConstraintType.CHANGEOVER,
            name="Changeover Times",
            description="Sequence-dependent setup times must be included in schedule",
            hard=True,
        )


@dataclass
class ResourceConstraint(Constraint):
    """Resource availability constraint (labor, materials, etc.)"""

    def __init__(
        self, resource_id: str, resource_name: str, max_capacity: float, time_window: tuple
    ):
        super().__init__(
            constraint_id=f"resource_{resource_id}",
            constraint_type=ConstraintType.RESOURCE,
            name=f"Resource - {resource_name}",
            description=f"Resource {resource_name} capacity must not exceed {max_capacity}",
            hard=True,
        )
        self.resource_id = resource_id
        self.max_capacity = max_capacity
        self.time_window = time_window


class ConstraintManager:
    """
    Manages collection of constraints for a scheduling problem

    Provides methods to add, remove, and validate constraints.
    """

    def __init__(self):
        self.constraints: List[Constraint] = []

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint"""
        self.constraints.append(constraint)

    def remove_constraint(self, constraint_id: str) -> None:
        """Remove a constraint by ID"""
        self.constraints = [c for c in self.constraints if c.constraint_id != constraint_id]

    def get_hard_constraints(self) -> List[Constraint]:
        """Get all hard constraints"""
        return [c for c in self.constraints if c.hard]

    def get_soft_constraints(self) -> List[Constraint]:
        """Get all soft constraints"""
        return [c for c in self.constraints if not c.hard]

    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[Constraint]:
        """Get constraints of specific type"""
        return [c for c in self.constraints if c.constraint_type == constraint_type]

    def validate_all(self, schedule: Any) -> tuple[bool, List[tuple[Constraint, ValidationResult, str]]]:
        """
        Validate all constraints

        Args:
            schedule: Schedule to validate

        Returns:
            Tuple of (all_valid, [(constraint, result, message)])
        """
        results = []
        all_valid = True

        for constraint in self.constraints:
            result, message = constraint.validate(schedule)
            results.append((constraint, result, message))

            if constraint.hard and result == ValidationResult.INVALID:
                all_valid = False

        return all_valid, results

    def calculate_penalty(self, schedule: Any) -> float:
        """
        Calculate total penalty for soft constraint violations

        Args:
            schedule: Schedule to evaluate

        Returns:
            Total penalty value
        """
        total_penalty = 0.0

        for constraint in self.get_soft_constraints():
            result, _ = constraint.validate(schedule)
            if result == ValidationResult.INVALID:
                total_penalty += constraint.penalty_weight

        return total_penalty
