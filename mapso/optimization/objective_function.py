"""
Multi-objective function implementation

Implements the weighted sum objective function:

f(x) = α·lateness + β·setup_time + γ·cost + δ·energy

Where:
- α, β, γ, δ ≥ 0 and α + β + γ + δ = 1 (weights)
- Each component is normalized to [0, 1] range
"""

from typing import Dict, List, Optional
import numpy as np

from mapso.core.models import Schedule, SKU, ChangeoverMatrix
from mapso.evaluation.metrics import ScheduleMetrics
from mapso.utils.logging_config import get_logger

logger = get_logger("objective_function")


class MultiObjectiveFunction:
    """
    Multi-objective function for schedule optimization

    Combines multiple objectives into weighted sum:
    - Lateness (minimize late jobs)
    - Setup time (minimize changeovers)
    - Cost (minimize production + overtime + setup costs)
    - Energy (minimize energy consumption)

    Each component is normalized using min-max scaling.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalization_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize multi-objective function

        Args:
            weights: Dictionary of {component: weight}
            normalization_bounds: Dictionary of {component: {min, max}}
        """
        # Default weights
        if weights is None:
            weights = {
                "lateness": 0.4,
                "setup_time": 0.2,
                "cost": 0.2,
                "energy": 0.2,
            }

        # Validate weights sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(
                f"Weights sum to {total_weight}, normalizing to 1.0"
            )
            weights = {k: v / total_weight for k, v in weights.items()}

        self.weights = weights

        # Default normalization bounds (will be updated based on observed values)
        if normalization_bounds is None:
            normalization_bounds = {
                "lateness": {"min": 0.0, "max": 1000.0},
                "setup_time": {"min": 0.0, "max": 100.0},
                "cost": {"min": 0.0, "max": 100000.0},
                "energy": {"min": 0.0, "max": 10000.0},
            }

        self.normalization_bounds = normalization_bounds

        # Track observed values for adaptive normalization
        self.observed_values = {
            "lateness": [],
            "setup_time": [],
            "cost": [],
            "energy": [],
        }

    def evaluate(
        self,
        schedule: Schedule,
        changeover_matrix: ChangeoverMatrix,
        skus: List[SKU],
        normalize: bool = True,
    ) -> float:
        """
        Evaluate multi-objective function for a schedule

        Args:
            schedule: Schedule to evaluate
            changeover_matrix: Changeover matrix
            skus: List of SKUs
            normalize: Whether to normalize components

        Returns:
            Weighted objective value (lower is better)
        """
        # Calculate all metrics
        metrics = ScheduleMetrics.calculate_all_metrics(
            schedule, changeover_matrix, skus
        )

        # Extract components
        lateness = metrics.get("total_lateness_hours", 0.0)
        setup_time = metrics.get("total_setup_time_hours", 0.0)
        cost = metrics.get("total_cost", 0.0)
        energy = metrics.get("total_energy_kwh", 0.0)

        # Record observed values
        self.observed_values["lateness"].append(lateness)
        self.observed_values["setup_time"].append(setup_time)
        self.observed_values["cost"].append(cost)
        self.observed_values["energy"].append(energy)

        # Normalize if requested
        if normalize:
            lateness_norm = self._normalize("lateness", lateness)
            setup_norm = self._normalize("setup_time", setup_time)
            cost_norm = self._normalize("cost", cost)
            energy_norm = self._normalize("energy", energy)
        else:
            lateness_norm = lateness
            setup_norm = setup_time
            cost_norm = cost
            energy_norm = energy

        # Weighted sum
        objective_value = (
            self.weights.get("lateness", 0.0) * lateness_norm
            + self.weights.get("setup_time", 0.0) * setup_norm
            + self.weights.get("cost", 0.0) * cost_norm
            + self.weights.get("energy", 0.0) * energy_norm
        )

        return objective_value

    def evaluate_components(
        self,
        schedule: Schedule,
        changeover_matrix: ChangeoverMatrix,
        skus: List[SKU],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all components separately

        Args:
            schedule: Schedule to evaluate
            changeover_matrix: Changeover matrix
            skus: List of SKUs

        Returns:
            Dictionary with component values (raw, normalized, weighted)
        """
        metrics = ScheduleMetrics.calculate_all_metrics(
            schedule, changeover_matrix, skus
        )

        components = {}

        # Lateness
        lateness_raw = metrics.get("total_lateness_hours", 0.0)
        lateness_norm = self._normalize("lateness", lateness_raw)
        components["lateness"] = {
            "raw": lateness_raw,
            "normalized": lateness_norm,
            "weighted": self.weights.get("lateness", 0.0) * lateness_norm,
            "weight": self.weights.get("lateness", 0.0),
        }

        # Setup time
        setup_raw = metrics.get("total_setup_time_hours", 0.0)
        setup_norm = self._normalize("setup_time", setup_raw)
        components["setup_time"] = {
            "raw": setup_raw,
            "normalized": setup_norm,
            "weighted": self.weights.get("setup_time", 0.0) * setup_norm,
            "weight": self.weights.get("setup_time", 0.0),
        }

        # Cost
        cost_raw = metrics.get("total_cost", 0.0)
        cost_norm = self._normalize("cost", cost_raw)
        components["cost"] = {
            "raw": cost_raw,
            "normalized": cost_norm,
            "weighted": self.weights.get("cost", 0.0) * cost_norm,
            "weight": self.weights.get("cost", 0.0),
        }

        # Energy
        energy_raw = metrics.get("total_energy_kwh", 0.0)
        energy_norm = self._normalize("energy", energy_raw)
        components["energy"] = {
            "raw": energy_raw,
            "normalized": energy_norm,
            "weighted": self.weights.get("energy", 0.0) * energy_norm,
            "weight": self.weights.get("energy", 0.0),
        }

        return components

    def _normalize(self, component: str, value: float) -> float:
        """
        Normalize value to [0, 1] using min-max scaling

        Args:
            component: Component name
            value: Raw value

        Returns:
            Normalized value in [0, 1]
        """
        bounds = self.normalization_bounds.get(component, {"min": 0.0, "max": 1.0})
        min_val = bounds["min"]
        max_val = bounds["max"]

        if max_val == min_val:
            return 0.0

        normalized = (value - min_val) / (max_val - min_val)

        # Clip to [0, 1]
        return max(0.0, min(1.0, normalized))

    def update_normalization_bounds(self, percentile: float = 95) -> None:
        """
        Update normalization bounds based on observed values

        Uses percentile-based approach to handle outliers.

        Args:
            percentile: Percentile to use for max bound (e.g., 95)
        """
        for component, values in self.observed_values.items():
            if len(values) < 10:
                continue  # Need more samples

            min_val = np.min(values)
            max_val = np.percentile(values, percentile)

            self.normalization_bounds[component] = {
                "min": min_val,
                "max": max_val,
            }

            logger.info(
                f"Updated {component} bounds: [{min_val:.2f}, {max_val:.2f}]"
            )

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update objective weights

        Args:
            weights: New weights (must sum to 1)
        """
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.weights = weights
        logger.info(f"Updated weights: {weights}")

    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.weights.copy()

    def get_normalization_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get current normalization bounds"""
        return {k: v.copy() for k, v in self.normalization_bounds.items()}
