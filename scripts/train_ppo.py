from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO

from scenario_policies.network_delay.env_network_delay import NetworkDelayEnv
from scenario_policies.pod_failure.env_pod_failure import PodFailureEnv
from scenario_policies.shared.training.baseline import BaselineRandomAgent
from scenario_policies.shared.training.evaluation import evaluate_agent


def get_env_class(scenario: str):
    if scenario == "network_delay":
        return NetworkDelayEnv
    if scenario == "pod_failure":
        return PodFailureEnv
    raise ValueError(f"Unsupported scenario: {scenario}")


def train_once(args: argparse.Namespace) -> Dict:
    env_cls = get_env_class(args.scenario)
    env_config = {
        "max_steps": args.max_steps,
        "step_interval_sec": args.step_interval_sec,
        "num_deployments": args.num_deployments,
        "num_nodes": args.num_nodes,
    }

    env = env_cls(config=env_config)

    tensorboard_log = str(args.log_dir) if args.enable_tensorboard else None

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
        tensorboard_log=tensorboard_log,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    agent_metrics = evaluate_agent(model, env, num_episodes=args.eval_episodes)
    baseline = BaselineRandomAgent(env)
    baseline_metrics = evaluate_agent(baseline, env, num_episodes=args.eval_episodes)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir / f"run_{run_id}_{args.scenario}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "ppo_model.zip"
    model.save(str(model_path))

    metadata = {
        "run_id": run_id,
        "scenario": args.scenario,
        "train_config": {
            "timesteps": args.timesteps,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "max_steps": args.max_steps,
            "step_interval_sec": args.step_interval_sec,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "enable_tensorboard": args.enable_tensorboard,
        },
        "agent_metrics": agent_metrics,
        "baseline_metrics": baseline_metrics,
        "model_path": str(model_path),
    }

    meta_file = run_dir / "training_metadata.json"
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    summary = {
        "run_dir": str(run_dir),
        "agent_metrics": agent_metrics,
        "baseline_metrics": baseline_metrics,
    }
    with (args.output_dir.parent / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on selected K8s fault scenario")
    parser.add_argument("--scenario", choices=["network_delay", "pod_failure"], required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--step-interval-sec", type=int, default=10)
    parser.add_argument("--num-deployments", type=int, default=5)
    parser.add_argument("--num-nodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("ppo_k8s_logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.enable_tensorboard:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    metadata = train_once(args)

    print("\n=== TRAINING DONE ===")
    print(f"scenario: {metadata['scenario']}")
    print(f"success_rate(agent): {metadata['agent_metrics']['success_rate']:.3f}")
    print(f"success_rate(baseline): {metadata['baseline_metrics']['success_rate']:.3f}")
    print(f"avg_reward(agent): {metadata['agent_metrics']['avg_reward']:.3f}")
    print(f"model_path: {metadata['model_path']}")


if __name__ == "__main__":
    main()
