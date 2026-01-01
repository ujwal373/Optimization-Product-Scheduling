"""
Global constants for MAPSO

Mathematical constants, default parameters, and system-wide settings.
"""

from typing import Dict

# Time constants (in minutes)
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

# Priority levels
PRIORITY_CRITICAL = 1
PRIORITY_HIGH = 2
PRIORITY_MEDIUM = 3
PRIORITY_LOW = 4
PRIORITY_MINIMAL = 5

# Optimization defaults
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_CPSAT_THREADS = 8
DEFAULT_SA_TEMPERATURE = 100.0
DEFAULT_SA_COOLING_RATE = 0.95
DEFAULT_SA_MAX_ITERATIONS = 10000

# Objective weights (must sum to 1.0)
DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "lateness": 0.4,
    "setup_time": 0.2,
    "cost": 0.2,
    "energy": 0.2,
}

# Trust-region optimization defaults
DEFAULT_TRUST_REGION_RADIUS = 0.1
DEFAULT_TRUST_REGION_MAX_ITER = 50
DEFAULT_TRUST_REGION_EPSILON = 1e-6

# Forecasting defaults
DEFAULT_FORECAST_HORIZON_DAYS = 30
DEFAULT_FORECAST_CONFIDENCE = 0.95
SARIMA_DEFAULT_SEASONAL_PERIOD = 7  # Weekly seasonality

# Data generation defaults
DEFAULT_N_SKUS = 30
DEFAULT_N_MACHINES = 10
DEFAULT_N_JOBS = 200
DEFAULT_HISTORICAL_DAYS = 365

# Changeover time ranges (minutes)
CHANGEOVER_LOW_MIN = 5
CHANGEOVER_LOW_MAX = 30
CHANGEOVER_MEDIUM_MIN = 30
CHANGEOVER_MEDIUM_MAX = 120
CHANGEOVER_HIGH_MIN = 120
CHANGEOVER_HIGH_MAX = 480

# Machine utilization
TARGET_UTILIZATION = 0.75
MAX_UTILIZATION = 0.95

# Shift definitions (standard factory shifts)
SHIFT_1_START = "06:00"
SHIFT_1_END = "14:00"
SHIFT_2_START = "14:00"
SHIFT_2_END = "22:00"
SHIFT_3_START = "22:00"
SHIFT_3_END = "06:00"

# Cost parameters
OVERTIME_MULTIPLIER = 1.5
LATE_JOB_PENALTY_PER_DAY = 1000.0
ENERGY_PEAK_MULTIPLIER = 1.3
ENERGY_OFFPEAK_MULTIPLIER = 0.7

# Tolerance and epsilon values
EPSILON = 1e-9
FLOAT_TOLERANCE = 1e-6

# Validation thresholds
MAX_JOBS_PER_SCHEDULE = 10000
MAX_MACHINES = 100
MAX_SKUS = 500
MIN_DUE_DATE_DAYS = 1
MAX_DUE_DATE_DAYS = 365

# Logging
LOG_DATE_FORMAT = "YYYY-MM-DD HH:mm:ss"
LOG_LEVEL_DEFAULT = "INFO"

# API limits
API_MAX_JOBS_PER_REQUEST = 1000
API_MAX_TIMEOUT_SECONDS = 600  # 10 minutes
API_DEFAULT_PAGE_SIZE = 50

# Visualization
GANTT_HEIGHT_PER_MACHINE = 50
GANTT_MIN_HEIGHT = 300
GANTT_MAX_HEIGHT = 1200
COLOR_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# File formats
SUPPORTED_INPUT_FORMATS = [".json", ".csv", ".xlsx"]
SUPPORTED_OUTPUT_FORMATS = [".json", ".csv", ".xlsx", ".pkl"]

# Random seed for reproducibility
DEFAULT_RANDOM_SEED = 42
