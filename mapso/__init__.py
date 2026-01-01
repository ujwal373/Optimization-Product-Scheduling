"""
MAPSO - Multi-Agent Production Scheduling Optimizer

A production-ready multi-agent scheduling system for industrial manufacturing
and supply chain optimization.

This package implements:
- 5 specialized agents (Demand Forecast, Capacity, Changeover, Cost/Energy, Orchestrator)
- 3-layer optimization (CP-SAT, Simulated Annealing, Continuous)
- Multi-objective optimization (lateness, setup time, cost, energy)
- Mathematical rigor with formal problem formulations
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Version information
VERSION = __version__

__all__ = [
    "VERSION",
    "__version__",
]
