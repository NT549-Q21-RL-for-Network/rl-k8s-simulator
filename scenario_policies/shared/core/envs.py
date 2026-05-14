"""Các môi trường Gym cho self-healing."""

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_space import ActionSpace
from .failure_scenarios import FailureScenario
from .reward_calculator import RewardCalculator


class K8sSelfHealingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 100)
        self.observation_step_interval = self.config.get("step_interval_sec", 10)
        self.num_deployments = self.config.get("num_deployments", 5)
        self.num_nodes = self.config.get("num_nodes", 3)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = ActionSpace.get_action_space()

        self.current_step = 0
        self.episode_rewards: list[float] = []
        self.current_state: Dict[str, float] | None = None
        self.prev_state: Dict[str, float] | None = None
        self._scenario_name: str | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_rewards = []
        self.current_state = self._generate_failed_state()
        self.prev_state = self.current_state.copy()
        obs = self._encode_observation(self.current_state)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action_name = ActionSpace.ACTIONS[int(action)]["name"]
        self._execute_action(int(action), action_name)

        assert self.current_state is not None
        self.prev_state = self.current_state.copy()
        self.current_state = self._collect_metrics()

        reward = RewardCalculator.calculate(self.prev_state, self.current_state, int(action), self.current_step)
        self.episode_rewards.append(reward)

        recovered = self._is_recovered()
        collapsed = self._is_collapsed()
        truncated = self.current_step >= self.max_steps
        terminated = recovered or collapsed

        obs = self._encode_observation(self.current_state)
        info = {
            "action": action_name,
            "recovered": recovered,
            "collapsed": collapsed,
            "episode_reward": float(sum(self.episode_rewards)),
            "scenario": self._scenario_name,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _generate_failed_state(self) -> Dict[str, float]:
        scenario = FailureScenario.sample_scenario()
        self._scenario_name = scenario["name"]
        return scenario["state"].copy()

    def _execute_action(self, action: int, action_name: str) -> None:
        s = self.current_state
        assert s is not None

        if action == 1:
            s["crashloop_flag"] = max(0.0, s["crashloop_flag"] - 2.0)
            s["failed_pods"] = max(0.0, s["failed_pods"] - 2.0)
            s["pending_pods"] = max(0.0, s["pending_pods"] - 2.0)
            s["error_rate_5xx"] = max(0.0, s["error_rate_5xx"] * 0.85)
            s["throughput"] = min(1.0, s["throughput"] + 0.04)
        elif action == 2:
            s["pending_pods"] = max(0.0, s["pending_pods"] - 6.0)
            s["throughput"] = min(1.0, s["throughput"] + 0.10)
            s["cpu_utilization"] = max(0.05, s["cpu_utilization"] - 0.08)
            s["memory_usage"] = max(0.05, s["memory_usage"] - 0.08)
            s["error_rate_5xx"] = max(0.0, s["error_rate_5xx"] * 0.91)
        elif action == 3:
            s["cpu_utilization"] = max(0.05, s["cpu_utilization"] - 0.05)
            s["memory_usage"] = max(0.05, s["memory_usage"] - 0.05)
            s["throughput"] *= 0.90
        elif action == 4:
            s["node_ready_status"] = max(0.0, s["node_ready_status"] - 1.0)
            s["pending_pods"] = min(100.0, s["pending_pods"] + 5.0)
        elif action == 5:
            s["node_ready_status"] = max(0.0, s["node_ready_status"] - 0.8)
            s["error_rate_5xx"] = max(0.0, s["error_rate_5xx"] * 0.95)
        elif action == 6:
            s["node_ready_status"] = max(0.0, s["node_ready_status"] - 0.8)
            s["pending_pods"] = max(0.0, s["pending_pods"] - 5.0)
            s["error_rate_5xx"] = max(0.0, s["error_rate_5xx"] * 0.96)
            s["throughput"] = min(1.0, s["throughput"] + 0.025)

    def _collect_metrics(self) -> Dict[str, float]:
        assert self.current_state is not None
        state: Dict[str, float] = {}
        for key, val in self.current_state.items():
            if key == "node_ready_status":
                drift = max(0.0, val - np.random.uniform(0, 0.12))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.05), 0, 3))
            elif key == "pending_pods":
                drift = max(0.0, val - np.random.uniform(0.8, 2.5))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.3), 0, 100))
            elif key == "crashloop_flag":
                drift = max(0.0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 20))
            elif key == "failed_pods":
                drift = max(0.0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 50))
            else:
                target = np.random.uniform(0.08, 0.25)
                drift = val * 0.98 + target * 0.02
                state[key] = float(np.clip(drift + np.random.normal(0, 0.015), 0, 1))
        return state

    def _is_recovered(self) -> bool:
        assert self.current_state is not None
        s = self.current_state
        return (
            s["error_rate_5xx"] < 0.02
            and s["pending_pods"] < 2
            and s["crashloop_flag"] < 1
            and s["node_ready_status"] < 1
        )

    def _is_collapsed(self) -> bool:
        assert self.current_state is not None
        s = self.current_state
        return s["error_rate_5xx"] > 0.5 and s["node_ready_status"] >= 3

    def _encode_observation(self, state: Dict[str, float]) -> np.ndarray:
        continuous = np.array(
            [
                state["cpu_utilization"],
                state["memory_usage"],
                state["disk_io"],
                state["network_bandwidth"],
                state["p90_latency"],
                state["p99_latency"],
                state["error_rate_5xx"],
                state["throughput"],
            ],
            dtype=np.float32,
        )

        discrete = np.array(
            [
                state["node_ready_status"] / 3,
                state["pending_pods"] / 100,
                state["crashloop_flag"] / 20,
                state["failed_pods"] / 50,
            ],
            dtype=np.float32,
        )

        return np.clip(np.concatenate([continuous, discrete]), 0.0, 1.0)


class K8sSelfHealingEnvV31(K8sSelfHealingEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        assert self.current_state is not None
        self.current_state = self._generate_failed_state_severe()
        self.prev_state = self.current_state.copy()
        obs = self._encode_observation(self.current_state)
        return obs, {}

    def _generate_failed_state_severe(self) -> Dict[str, float]:
        scenario = FailureScenario.sample_scenario()
        self._scenario_name = scenario["name"]
        state = scenario["state"].copy()
        state["error_rate_5xx"] = min(1.0, state["error_rate_5xx"] * 1.3)
        state["cpu_utilization"] = min(1.0, state["cpu_utilization"] * 1.2)
        state["memory_usage"] = min(1.0, state["memory_usage"] * 1.2)
        state["p99_latency"] = min(1.0, state["p99_latency"] * 1.25)
        state["pending_pods"] = min(100.0, state["pending_pods"] * 1.3)
        state["node_ready_status"] = min(3.0, state["node_ready_status"] * 1.2)
        return state

    def _collect_metrics(self) -> Dict[str, float]:
        assert self.current_state is not None
        state: Dict[str, float] = {}
        for key, val in self.current_state.items():
            if key == "node_ready_status":
                drift = max(0.0, val - np.random.uniform(0, 0.12))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.07), 0, 3))
            elif key == "pending_pods":
                drift = max(0.0, val - np.random.uniform(0.8, 2.5))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.3), 0, 100))
            elif key == "crashloop_flag":
                drift = max(0.0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 20))
            elif key == "failed_pods":
                drift = max(0.0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 50))
            else:
                target = np.random.uniform(0.08, 0.25)
                drift = val * 0.96 + target * 0.04
                state[key] = float(np.clip(drift + np.random.normal(0, 0.025), 0, 1))
        return state

    def _is_recovered(self) -> bool:
        assert self.current_state is not None
        s = self.current_state
        return (
            s["error_rate_5xx"] < 0.01
            and s["pending_pods"] < 1
            and s["crashloop_flag"] < 0.5
            and s["node_ready_status"] < 0.5
        )
