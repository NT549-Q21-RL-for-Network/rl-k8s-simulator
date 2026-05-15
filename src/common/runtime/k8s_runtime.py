"""Compatibility module.

Legacy imports may still reference:
- src.common.runtime.k8s_runtime.K8sMetricsCollector
- src.common.runtime.k8s_runtime.K8sActionExecutor
- src.common.runtime.k8s_runtime.K8sSelfHealingAgent

The concrete implementations now live in smaller modules.
"""

from .metrics_collector import K8sMetricsCollector
from .action_executor import K8sActionExecutor
from .agent import K8sSelfHealingAgent

__all__ = ["K8sMetricsCollector", "K8sActionExecutor", "K8sSelfHealingAgent"]
