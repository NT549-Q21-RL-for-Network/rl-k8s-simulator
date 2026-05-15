"""Scenario-focused environment for pod failure / crash loop."""
from typing import Dict
import numpy as np

from src.common.core.envs import K8sSelfHealingEnv
from src.common.core.failure_scenarios import FailureScenario


class PodFailureEnv(K8sSelfHealingEnv):
    """Environment constrained to pod crash loop failures."""

    TARGET_SCENARIOS = ("pod_crash_loop",)

    def _generate_failed_state(self) -> Dict[str, float]:
        scenario_name = np.random.choice(self.TARGET_SCENARIOS)
        scenario = FailureScenario.SCENARIOS[scenario_name].copy()
        self._scenario_name = scenario_name

        state = scenario["state"].copy()
        for key, val in state.items():
            noise = np.random.uniform(-0.08, 0.08)
            if key in ["node_ready_status", "pending_pods", "crashloop_flag", "failed_pods"]:
                max_vals = {"node_ready_status": 3, "pending_pods": 100, "crashloop_flag": 20, "failed_pods": 50}
                state[key] = float(np.clip(val + noise * max_vals.get(key, 1), 0, max_vals.get(key, 1)))
            else:
                state[key] = float(np.clip(val + noise, 0.0, 1.0))
        return state
