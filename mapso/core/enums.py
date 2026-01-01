"""
Enumerations for MAPSO

Defines enum types for status tracking, solver types, and other categorical values.
Note: JobStatus, MachineState, and ShiftType are defined in models.py for convenience.
"""

from enum import Enum


class SolverType(Enum):
    """Type of optimization solver"""

    CPSAT = "cpsat"  # Google OR-Tools CP-SAT
    SIMULATED_ANNEALING = "simulated_annealing"  # Metaheuristic
    GENETIC_ALGORITHM = "genetic_algorithm"  # Not implemented yet
    HYBRID = "hybrid"  # Combined approach
    BASELINE = "baseline"  # Heuristic (FIFO, EDD, SPT)


class HeuristicType(Enum):
    """Type of baseline heuristic"""

    FIFO = "fifo"  # First In First Out
    EDD = "edd"  # Earliest Due Date
    SPT = "spt"  # Shortest Processing Time
    PRIORITY = "priority"  # By job priority
    RANDOM = "random"  # Random assignment


class ForecastMethod(Enum):
    """Forecasting method"""

    SARIMA = "sarima"  # Seasonal ARIMA
    PROPHET = "prophet"  # Facebook Prophet
    LSTM = "lstm"  # Long Short-Term Memory
    ENSEMBLE = "ensemble"  # Ensemble of methods
    MOVING_AVERAGE = "moving_average"  # Simple moving average


class ConstraintType(Enum):
    """Type of scheduling constraint"""

    ASSIGNMENT = "assignment"  # Job-to-machine assignment
    CAPACITY = "capacity"  # Machine capacity
    TEMPORAL = "temporal"  # Time windows, precedence
    RESOURCE = "resource"  # Labor, materials
    CHANGEOVER = "changeover"  # Setup times
    SHIFT = "shift"  # Shift availability


class ObjectiveComponent(Enum):
    """Components of multi-objective function"""

    LATENESS = "lateness"  # Minimize lateness
    TARDINESS = "tardiness"  # Minimize tardiness
    MAKESPAN = "makespan"  # Minimize total time
    SETUP_TIME = "setup_time"  # Minimize changeover time
    COST = "cost"  # Minimize cost
    ENERGY = "energy"  # Minimize energy
    UTILIZATION = "utilization"  # Maximize utilization


class NeighborhoodOperator(Enum):
    """Neighborhood operators for local search"""

    SWAP = "swap"  # Swap two jobs
    SHIFT = "shift"  # Move job to different time
    REASSIGN = "reassign"  # Assign job to different machine
    RESEQUENCE = "resequence"  # Reverse subsequence
    INSERT = "insert"  # Insert job at different position


class ValidationResult(Enum):
    """Result of constraint validation"""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNKNOWN = "unknown"


class OptimizationStatus(Enum):
    """Status of optimization run"""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    OPTIMAL = "optimal"  # Optimal solution found
    FEASIBLE = "feasible"  # Feasible but not proven optimal
    INFEASIBLE = "infeasible"  # No feasible solution
    TIMEOUT = "timeout"  # Stopped due to timeout
    ERROR = "error"  # Error occurred
    CANCELLED = "cancelled"  # Cancelled by user
