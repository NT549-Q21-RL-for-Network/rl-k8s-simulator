"""Skeleton runtime interfaces for real-cluster integration."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class K8sMetricsCollector:
    config: Dict[str, Any]

    def collect(self) -> Dict[str, Any]:
        raise NotImplementedError("Implement Prometheus/K8s metrics collection here.")


@dataclass
class K8sActionExecutor:
    config: Dict[str, Any]

    def execute(self, action_id: int, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError("Implement kubectl/K8s API action execution here.")


@dataclass
class K8sSelfHealingAgent:
    model: Any

    def predict_action(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action.item() if hasattr(action, "item") else action)
