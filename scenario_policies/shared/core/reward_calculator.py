"""Hàm thưởng cho bài toán self-healing."""

from typing import Any, Dict
import numpy as np


class RewardCalculator:
    ALPHA = 8.0
    BETA = 1.5
    GAMMA = 2.0
    DELTA = 4.0
    STEP_PENALTY = 0.06
    RECOVERY_BONUS = 40.0
    COLLAPSE_PENALTY = -50.0

    @staticmethod
    def calculate(prev_state: Dict[str, Any], curr_state: Dict[str, Any], action_id: int, episode_steps: int) -> float:
        prev_health = RewardCalculator._calculate_health(prev_state)
        curr_health = RewardCalculator._calculate_health(curr_state)
        stability_delta = curr_health - prev_health
        stability_reward = RewardCalculator.ALPHA * max(0.0, stability_delta)

        overhead_penalty = RewardCalculator.BETA * RewardCalculator._calculate_overhead(curr_state)
        impact_penalty = RewardCalculator.GAMMA * RewardCalculator._get_action_impact(action_id)
        sla_cost = RewardCalculator.DELTA * RewardCalculator._calculate_sla_penalty(curr_state)

        return stability_reward - overhead_penalty - impact_penalty - sla_cost - RewardCalculator.STEP_PENALTY

    @staticmethod
    def _calculate_health(state: Dict[str, Any]) -> float:
        error_rate = state.get("error_rate_5xx", 0.0)
        pending_pods = state.get("pending_pods", 0.0) / 100.0
        crashloop = state.get("crashloop_flag", 0.0) / 20.0
        node_status = state.get("node_ready_status", 0.0) / 3.0
        latency = (state.get("p90_latency", 0.0) + state.get("p99_latency", 0.0)) / 2.0
        throughput_loss = 1.0 - state.get("throughput", 1.0)

        cpu = state.get("cpu_utilization", 0.0)
        memory = state.get("memory_usage", 0.0)
        critical_resource = cpu > 0.98 or memory > 0.98

        health = (
            (1.0 - error_rate) * 0.30
            + (1.0 - node_status) * 0.25
            + (1.0 - latency) * 0.20
            + (1.0 - pending_pods) * 0.10
            + (1.0 - throughput_loss) * 0.10
            + (1.0 - crashloop) * 0.05
        )

        if critical_resource:
            health *= 0.5

        return float(np.clip(health, 0.0, 1.0))

    @staticmethod
    def _calculate_overhead(state: Dict[str, Any]) -> float:
        cpu = state.get("cpu_utilization", 0.0)
        memory = state.get("memory_usage", 0.0)
        network = state.get("network_bandwidth", 0.0)
        return (cpu + memory + network) / 3.0

    @staticmethod
    def _get_action_impact(action_id: int) -> float:
        action_impacts = {
            0: 0.0,
            1: 0.50,
            2: 0.12,
            3: 0.20,
            4: 0.75,
            5: 0.40,
            6: 0.05,
        }
        return action_impacts.get(action_id, 0.25)

    @staticmethod
    def _calculate_sla_penalty(state: Dict[str, Any]) -> float:
        p99_latency = state.get("p99_latency", 0.0)
        p90_latency = state.get("p90_latency", 0.0)
        error_rate = state.get("error_rate_5xx", 0.0)

        latency_violation = max(0.0, p99_latency - 0.3) ** 1.5
        latency_p90_violation = max(0.0, p90_latency - 0.25) ** 1.3 * 0.5
        error_violation = max(0.0, error_rate - 0.02) ** 1.2

        penalty = latency_violation + latency_p90_violation + error_violation
        return float(min(1.0, penalty))
