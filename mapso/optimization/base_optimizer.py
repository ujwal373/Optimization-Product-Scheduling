"""
Base optimizer interface

Defines the standard interface that all optimizers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from mapso.core.models import Schedule, Job, Machine, ChangeoverMatrix, SKU
from mapso.core.enums import OptimizationStatus
from mapso.utils.logging_config import get_logger


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers

    All optimization algorithms (CP-SAT, Simulated Annealing, etc.)
    must implement this interface.
    """

    def __init__(self, optimizer_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize optimizer

        Args:
            optimizer_name: Name of this optimizer
            config: Configuration dictionary
        """
        self.optimizer_name = optimizer_name
        self.config = config or {}
        self.logger = get_logger(f"optimizer.{optimizer_name}")

        # Statistics
        self.status = OptimizationStatus.NOT_STARTED
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.iterations = 0
        self.objective_value = float("inf")

    @abstractmethod
    def optimize(
        self,
        jobs: List[Job],
        machines: List[Machine],
        skus: List[SKU],
        changeover_matrix: ChangeoverMatrix,
        start_date: datetime,
        end_date: datetime,
        objective_weights: Optional[Dict[str, float]] = None,
        timeout: Optional[int] = None,
    ) -> Schedule:
        """
        Run optimization and return schedule

        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            skus: List of SKUs
            changeover_matrix: Changeover time/cost matrix
            start_date: Schedule start date
            end_date: Schedule end date
            objective_weights: Weights for multi-objective function
            timeout: Maximum time in seconds (None = no limit)

        Returns:
            Optimized schedule
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimizer statistics

        Returns:
            Dictionary with statistics
        """
        computation_time = 0.0
        if self.start_time and self.end_time:
            computation_time = (self.end_time - self.start_time).total_seconds()

        return {
            "optimizer": self.optimizer_name,
            "status": self.status.value,
            "computation_time_seconds": computation_time,
            "iterations": self.iterations,
            "objective_value": self.objective_value,
            "config": self.config,
        }

    def reset(self) -> None:
        """Reset optimizer state"""
        self.status = OptimizationStatus.NOT_STARTED
        self.start_time = None
        self.end_time = None
        self.iterations = 0
        self.objective_value = float("inf")

    def _start_optimization(self) -> None:
        """Mark optimization as started"""
        self.status = OptimizationStatus.RUNNING
        self.start_time = datetime.now()
        self.logger.info(f"{self.optimizer_name} optimization started")

    def _end_optimization(self, status: OptimizationStatus) -> None:
        """Mark optimization as finished"""
        self.status = status
        self.end_time = datetime.now()
        computation_time = (self.end_time - self.start_time).total_seconds()
        self.logger.info(
            f"{self.optimizer_name} optimization finished: {status.value} "
            f"(time: {computation_time:.2f}s, objective: {self.objective_value:.2f})"
        )
