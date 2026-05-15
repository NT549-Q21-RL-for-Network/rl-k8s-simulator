from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.core.reward_calculator import RewardCalculator
from src.common.training.baseline import BaselineRandomAgent
from src.common.training.evaluation import evaluate_agent


def _clip(v: float, low: float, high: float) -> float:
    return float(max(low, min(high, v)))


def _to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _map_row_to_state(row: Dict[str, str], max_rps_ref: float) -> Dict[str, float]:
    http_rps = max(0.0, _to_float(row, "http_rps", 0.0))
    failed_rate = _clip(_to_float(row, "http_req_failed_rate", 0.0), 0.0, 1.0)

    # Artifacts currently expose p95/p99 in ms.
    p95_ms = max(0.0, _to_float(row, "http_req_duration_p95_ms", 0.0))
    p99_ms = max(0.0, _to_float(row, "http_req_duration_p99_ms", 0.0))

    throughput = _clip(http_rps / max(max_rps_ref, 1.0), 0.0, 1.0)
    p90_latency = _clip(p95_ms / 2000.0, 0.0, 1.0)  # proxy from p95
    p99_latency = _clip(p99_ms / 3000.0, 0.0, 1.0)

    phase = (row.get("phase") or "").strip().lower()
    fault_active = str(row.get("fault_active", "False")).strip().lower() in {"true", "1", "yes"}

    # These metrics are not directly present in current artifacts; estimate lightly
    # from observed traffic/latency/error so PPO can still train end-to-end.
    cpu_utilization = _clip(0.15 + 0.55 * throughput + 0.15 * failed_rate, 0.0, 1.0)
    memory_usage = _clip(0.18 + 0.45 * throughput + 0.10 * p90_latency, 0.0, 1.0)
    disk_io = _clip(0.08 + 0.25 * throughput + (0.08 if fault_active else 0.0), 0.0, 1.0)
    network_bandwidth = _clip(0.10 + 0.60 * throughput, 0.0, 1.0)

    if "chaos_active" in phase or fault_active:
        pending_pods = 12.0
        failed_pods = 6.0 if failed_rate > 0.02 else 3.0
        crashloop_flag = 2.0 if "pod" in phase else 0.0
    elif "recovery" in phase:
        pending_pods = 6.0
        failed_pods = 2.0
        crashloop_flag = 0.5
    else:
        pending_pods = 2.0
        failed_pods = 0.5
        crashloop_flag = 0.0

    return {
        "cpu_utilization": cpu_utilization,
        "memory_usage": memory_usage,
        "disk_io": disk_io,
        "network_bandwidth": network_bandwidth,
        "p90_latency": p90_latency,
        "p99_latency": p99_latency,
        "error_rate_5xx": failed_rate,
        "throughput": throughput,
        "node_ready_status": 0.0,
        "pending_pods": float(pending_pods),
        "crashloop_flag": float(crashloop_flag),
        "failed_pods": float(failed_pods),
    }


def _encode_observation(state: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            state["cpu_utilization"],
            state["memory_usage"],
            state["disk_io"],
            state["network_bandwidth"],
            state["p90_latency"],
            state["p99_latency"],
            state["error_rate_5xx"],
            state["throughput"],
            _clip(state["node_ready_status"] / 3.0, 0.0, 1.0),
            _clip(state["pending_pods"] / 100.0, 0.0, 1.0),
            _clip(state["crashloop_flag"] / 20.0, 0.0, 1.0),
            _clip(state["failed_pods"] / 50.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def _apply_action_effect(state: Dict[str, float], action: int) -> None:
    if action == 1:  # restart_pod
        state["crashloop_flag"] = max(0.0, state["crashloop_flag"] - 2.0)
        state["failed_pods"] = max(0.0, state["failed_pods"] - 2.0)
        state["pending_pods"] = max(0.0, state["pending_pods"] - 1.0)
        state["error_rate_5xx"] = _clip(state["error_rate_5xx"] * 0.85, 0.0, 1.0)
    elif action == 2:  # scale_up
        state["pending_pods"] = max(0.0, state["pending_pods"] - 3.0)
        state["throughput"] = _clip(state["throughput"] + 0.08, 0.0, 1.0)
        state["cpu_utilization"] = _clip(state["cpu_utilization"] - 0.06, 0.0, 1.0)
        state["memory_usage"] = _clip(state["memory_usage"] - 0.06, 0.0, 1.0)
    elif action == 3:  # scale_down
        state["throughput"] = _clip(state["throughput"] * 0.92, 0.0, 1.0)
        state["cpu_utilization"] = _clip(state["cpu_utilization"] + 0.04, 0.0, 1.0)
        state["memory_usage"] = _clip(state["memory_usage"] + 0.04, 0.0, 1.0)


@dataclass
class ArtifactReplayEnv(gym.Env):
    trajectories: List[List[Dict[str, float]]]
    max_steps: int = 80

    metadata = {"render_modes": ["human"]}

    def __post_init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)
        self._traj: List[Dict[str, float]] = []
        self._idx = 0
        self.current_state: Dict[str, float] | None = None
        self.prev_state: Dict[str, float] | None = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._traj = random.choice(self.trajectories)
        self._idx = 0
        self.current_state = dict(self._traj[self._idx])
        self.prev_state = dict(self.current_state)
        return _encode_observation(self.current_state), {}

    def step(self, action: int):
        assert self.current_state is not None
        self.current_step += 1
        self.prev_state = dict(self.current_state)

        # Agent effect first
        _apply_action_effect(self.current_state, int(action))

        # Replay-driven external dynamics
        self._idx = min(self._idx + 1, len(self._traj) - 1)
        target = self._traj[self._idx]
        for k in self.current_state.keys():
            blended = 0.55 * self.current_state[k] + 0.45 * target[k]
            noise = np.random.normal(0.0, 0.01)
            if k in {"node_ready_status", "pending_pods", "crashloop_flag", "failed_pods"}:
                max_v = {"node_ready_status": 3.0, "pending_pods": 100.0, "crashloop_flag": 20.0, "failed_pods": 50.0}[k]
                self.current_state[k] = float(np.clip(blended + noise * max_v * 0.05, 0.0, max_v))
            else:
                self.current_state[k] = float(np.clip(blended + noise, 0.0, 1.0))

        reward = RewardCalculator.calculate(self.prev_state, self.current_state, int(action), self.current_step)
        recovered = (
            self.current_state["error_rate_5xx"] < 0.02
            and self.current_state["pending_pods"] < 2.0
            and self.current_state["crashloop_flag"] < 1.0
        )
        collapsed = (
            self.current_state["error_rate_5xx"] > 0.5
            and self.current_state["p99_latency"] > 0.95
        )
        truncated = self.current_step >= self.max_steps or self._idx >= len(self._traj) - 1
        terminated = recovered or collapsed

        info = {"recovered": recovered, "collapsed": collapsed}
        return _encode_observation(self.current_state), float(reward), bool(terminated), bool(truncated), info


def plot_training_curve(monitor_csv: Path, output_png: Path, rolling_window: int = 20) -> None:
    if not monitor_csv.exists():
        return

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    with monitor_csv.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episode_rewards.append(float(row["r"]))
                episode_lengths.append(int(float(row["l"])))
            except Exception:
                continue

    if not episode_rewards:
        return

    xs = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.float32)

    win = max(1, min(rolling_window, len(rewards)))
    kernel = np.ones(win, dtype=np.float32) / win
    rewards_ma = np.convolve(rewards, kernel, mode="same")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(xs, rewards, alpha=0.35, label="Episode reward")
    axes[0].plot(xs, rewards_ma, linewidth=2.0, label=f"Moving avg ({win})")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Convergence")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(xs, lengths, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode length")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def load_trajectories(artifacts_root: Path, scenario: str, min_len: int, max_rps_ref: float) -> List[List[Dict[str, float]]]:
    if scenario == "network_delay":
        base = artifacts_root / "chaos-network-delay-api-gateway"
    elif scenario == "pod_failure":
        base = artifacts_root / "chaos-pod-kill-product"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    csv_files = sorted(base.glob("*/combined-report.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No combined-report.csv found under {base}")

    trajectories: List[List[Dict[str, float]]] = []
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if len(rows) < min_len:
            continue
        traj = [_map_row_to_state(r, max_rps_ref=max_rps_ref) for r in rows]
        trajectories.append(traj)

    if not trajectories:
        raise RuntimeError(f"No trajectory with min_len>={min_len} for scenario={scenario}")
    return trajectories


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO from real artifacts trajectories")
    p.add_argument("--scenario", choices=["network_delay", "pod_failure"], required=True)
    p.add_argument("--artifacts-root", type=Path, required=True)
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--min-traj-len", type=int, default=30)
    p.add_argument("--max-rps-ref", type=float, default=1000.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rolling-window", type=int, default=20)
    p.add_argument("--output-dir", type=Path, default=Path("training_results/ppo"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(
        artifacts_root=args.artifacts_root,
        scenario=args.scenario,
        min_len=args.min_traj_len,
        max_rps_ref=args.max_rps_ref,
    )
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir / f"run_{run_id}_{args.scenario}_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_env = ArtifactReplayEnv(trajectories=trajectories, max_steps=args.max_steps)
    monitor_csv = run_dir / "monitor.csv"
    env = Monitor(base_env, filename=str(monitor_csv))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    agent_metrics = evaluate_agent(model, env, num_episodes=args.eval_episodes)
    baseline = BaselineRandomAgent(env)
    baseline_metrics = evaluate_agent(baseline, env, num_episodes=args.eval_episodes)

    model_path = run_dir / "ppo_model.zip"
    model.save(str(model_path))
    curve_path = run_dir / "training_curve.png"
    plot_training_curve(monitor_csv=monitor_csv, output_png=curve_path, rolling_window=args.rolling_window)

    metadata = {
        "run_id": run_id,
        "scenario": args.scenario,
        "source": "artifacts",
        "artifacts_root": str(args.artifacts_root),
        "num_trajectories": len(trajectories),
        "timesteps": args.timesteps,
        "monitor_csv": str(monitor_csv),
        "training_curve_png": str(curve_path),
        "agent_metrics": agent_metrics,
        "baseline_metrics": baseline_metrics,
        "model_path": str(model_path),
    }
    with (run_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n=== TRAINING FROM ARTIFACTS DONE ===")
    print(f"scenario: {args.scenario}")
    print(f"num_trajectories: {len(trajectories)}")
    print(f"success_rate(agent): {agent_metrics['success_rate']:.3f}")
    print(f"success_rate(baseline): {baseline_metrics['success_rate']:.3f}")
    print(f"avg_reward(agent): {agent_metrics['avg_reward']:.3f}")
    print(f"training_curve: {curve_path}")
    print(f"model_path: {model_path}")


if __name__ == "__main__":
    main()
