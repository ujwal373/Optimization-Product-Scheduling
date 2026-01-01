"""
Objective function components for production scheduling

Defines individual objective components and multi-objective combinations.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any
from mapso.core.enums import ObjectiveComponent
import numpy as np


@dataclass
class ObjectiveFunction:
    """
    Single objective function component

    Attributes:
        component: Type of objective
        name: Human-readable name
        description: Detailed description
        weight: Weight in multi-objective function (0-1)
        minimize: Whether to minimize (True) or maximize (False)
        evaluator: Function to evaluate this objective
    """

    component: ObjectiveComponent
    name: str
    description: str
    weight: float = 1.0
    minimize: bool = True
    evaluator: Optional[Callable] = None

    def evaluate(self, schedule: Any) -> float:
        """
        Evaluate objective for a schedule

        Args:
            schedule: Schedule to evaluate

        Returns:
            Objective value (raw, not normalized)
        """
        if self.evaluator is None:
            raise ValueError(f"No evaluator function provided for {self.name}")

        try:
            value = self.evaluator(schedule)
            return value if self.minimize else -value  # Negate for maximization
        except Exception as e:
            raise RuntimeError(f"Error evaluating {self.name}: {str(e)}")


@dataclass
class MultiObjectiveFunction:
    """
    Multi-objective function as weighted sum

    Objective: minimize Σ w_i · f_i(x)

    Where:
        w_i = weights (normalized to sum to 1)
        f_i = individual objective components (normalized to [0, 1])

    Attributes:
        objectives: List of objective functions
        weights: Weights for each objective (must sum to 1)
        normalization_params: Parameters for normalizing each objective
    """

    objectives: List[ObjectiveFunction]
    weights: Dict[str, float]
    normalization_params: Dict[str, Dict[str, float]]

    def __post_init__(self):
        """Validate weights sum to 1"""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Set weights in objective functions
        for obj in self.objectives:
            component_name = obj.component.value
            if component_name in self.weights:
                obj.weight = self.weights[component_name]

    def normalize(self, value: float, component: str) -> float:
        """
        Normalize objective value to [0, 1] range

        Args:
            value: Raw objective value
            component: Objective component name

        Returns:
            Normalized value in [0, 1]
        """
        if component not in self.normalization_params:
            # No normalization parameters, return as-is
            return value

        params = self.normalization_params[component]
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)

        if max_val == min_val:
            return 0.0

        # Min-max normalization
        normalized = (value - min_val) / (max_val - min_val)

        # Clip to [0, 1]
        return max(0.0, min(1.0, normalized))

    def evaluate(self, schedule: Any, normalize: bool = True) -> float:
        """
        Evaluate multi-objective function

        Args:
            schedule: Schedule to evaluate
            normalize: Whether to normalize individual objectives

        Returns:
            Weighted sum of objectives
        """
        total = 0.0

        for obj in self.objectives:
            # Evaluate raw objective
            raw_value = obj.evaluate(schedule)

            # Normalize if requested
            if normalize:
                value = self.normalize(raw_value, obj.component.value)
            else:
                value = raw_value

            # Add weighted component
            total += obj.weight * value

        return total

    def evaluate_components(self, schedule: Any) -> Dict[str, float]:
        """
        Evaluate all components separately

        Args:
            schedule: Schedule to evaluate

        Returns:
            Dictionary of {component_name: value}
        """
        components = {}

        for obj in self.objectives:
            raw_value = obj.evaluate(schedule)
            norm_value = self.normalize(raw_value, obj.component.value)

            components[obj.component.value] = {
                "raw": raw_value,
                "normalized": norm_value,
                "weighted": obj.weight * norm_value,
                "weight": obj.weight,
            }

        return components

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update objective weights

        Args:
            new_weights: New weight values (must sum to 1)
        """
        # Validate sum to 1
        total = sum(new_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.weights = new_weights

        # Update objective functions
        for obj in self.objectives:
            component_name = obj.component.value
            if component_name in new_weights:
                obj.weight = new_weights[component_name]

    def update_normalization_params(
        self, component: str, min_val: float, max_val: float
    ) -> None:
        """
        Update normalization parameters for a component

        Args:
            component: Component name
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
        """
        self.normalization_params[component] = {"min": min_val, "max": max_val}


class ObjectiveBuilder:
    """
    Builder for multi-objective functions

    Provides convenient methods to construct standard objective functions.
    """

    @staticmethod
    def build_default(
        lateness_weight: float = 0.4,
        setup_weight: float = 0.2,
        cost_weight: float = 0.2,
        energy_weight: float = 0.2,
    ) -> MultiObjectiveFunction:
        """
        Build default multi-objective function

        Objective: minimize (w1·lateness + w2·setup_time + w3·cost + w4·energy)

        Args:
            lateness_weight: Weight for lateness
            setup_weight: Weight for setup time
            cost_weight: Weight for cost
            energy_weight: Weight for energy

        Returns:
            MultiObjectiveFunction instance
        """
        # Validate weights
        total = lateness_weight + setup_weight + cost_weight + energy_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        objectives = [
            ObjectiveFunction(
                component=ObjectiveComponent.LATENESS,
                name="Total Lateness",
                description="Sum of lateness across all jobs (hours)",
                weight=lateness_weight,
                minimize=True,
            ),
            ObjectiveFunction(
                component=ObjectiveComponent.SETUP_TIME,
                name="Total Setup Time",
                description="Sum of changeover times (hours)",
                weight=setup_weight,
                minimize=True,
            ),
            ObjectiveFunction(
                component=ObjectiveComponent.COST,
                name="Total Cost",
                description="Production cost + overtime + setup costs ($)",
                weight=cost_weight,
                minimize=True,
            ),
            ObjectiveFunction(
                component=ObjectiveComponent.ENERGY,
                name="Total Energy",
                description="Energy consumption (kWh)",
                weight=energy_weight,
                minimize=True,
            ),
        ]

        weights = {
            "lateness": lateness_weight,
            "setup_time": setup_weight,
            "cost": cost_weight,
            "energy": energy_weight,
        }

        # Initialize with placeholder normalization params
        normalization_params = {
            "lateness": {"min": 0.0, "max": 1000.0},
            "setup_time": {"min": 0.0, "max": 100.0},
            "cost": {"min": 0.0, "max": 100000.0},
            "energy": {"min": 0.0, "max": 10000.0},
        }

        return MultiObjectiveFunction(
            objectives=objectives,
            weights=weights,
            normalization_params=normalization_params,
        )

    @staticmethod
    def build_makespan_only() -> MultiObjectiveFunction:
        """Build objective function focused only on makespan"""
        objectives = [
            ObjectiveFunction(
                component=ObjectiveComponent.MAKESPAN,
                name="Makespan",
                description="Total schedule duration (hours)",
                weight=1.0,
                minimize=True,
            )
        ]

        return MultiObjectiveFunction(
            objectives=objectives,
            weights={"makespan": 1.0},
            normalization_params={"makespan": {"min": 0.0, "max": 1000.0}},
        )

    @staticmethod
    def build_custom(
        component_weights: Dict[str, float],
        normalization_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> MultiObjectiveFunction:
        """
        Build custom multi-objective function

        Args:
            component_weights: Dictionary of {component_name: weight}
            normalization_bounds: Optional normalization parameters

        Returns:
            MultiObjectiveFunction instance
        """
        # Validate weights
        total = sum(component_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        # Map component names to ObjectiveComponent enum
        component_map = {
            "lateness": ObjectiveComponent.LATENESS,
            "tardiness": ObjectiveComponent.TARDINESS,
            "makespan": ObjectiveComponent.MAKESPAN,
            "setup_time": ObjectiveComponent.SETUP_TIME,
            "cost": ObjectiveComponent.COST,
            "energy": ObjectiveComponent.ENERGY,
            "utilization": ObjectiveComponent.UTILIZATION,
        }

        objectives = []
        for name, weight in component_weights.items():
            if name not in component_map:
                raise ValueError(f"Unknown objective component: {name}")

            objectives.append(
                ObjectiveFunction(
                    component=component_map[name],
                    name=name.replace("_", " ").title(),
                    description=f"Objective: {name}",
                    weight=weight,
                    minimize=name != "utilization",  # Maximize utilization
                )
            )

        # Default normalization params
        if normalization_bounds is None:
            normalization_bounds = {name: {"min": 0.0, "max": 1000.0} for name in component_weights}

        return MultiObjectiveFunction(
            objectives=objectives,
            weights=component_weights,
            normalization_params=normalization_bounds,
        )
