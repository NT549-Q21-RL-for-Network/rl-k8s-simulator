from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenario_policies.shared.runtime.k8s_runtime import K8sSelfHealingAgent

RUNNING = True


def _handle_sigterm(signum, frame):
    global RUNNING
    RUNNING = False


def parse_obs(raw: str | None) -> np.ndarray:
    if not raw:
        # near-healthy default observation, already normalized [0,1]
        return np.array([0.3, 0.3, 0.2, 0.3, 0.15, 0.2, 0.02, 0.8, 0.0, 0.02, 0.0, 0.0], dtype=np.float32)

    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 12:
        raise ValueError("OBSERVATION must have exactly 12 comma-separated values")
    arr = np.array(vals, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained PPO model in inference mode")
    parser.add_argument("--model-path", default="/app/models/ppo_model.zip")
    parser.add_argument("--interval-sec", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--observation", default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    model = PPO.load(str(model_path))
    agent = K8sSelfHealingAgent(model=model)

    print(f"[infer] model={model_path}")
    print(f"[infer] interval_sec={args.interval_sec}")

    if args.once:
        obs = parse_obs(args.observation)
        action = agent.predict_action(obs)
        print(json.dumps({"mode": "once", "action_id": action, "observation": obs.tolist()}))
        return

    while RUNNING:
        obs = parse_obs(args.observation)
        action = agent.predict_action(obs)
        print(json.dumps({"mode": "loop", "action_id": action, "observation": obs.tolist()}), flush=True)
        time.sleep(max(1, args.interval_sec))

    print("[infer] shutdown")


if __name__ == "__main__":
    main()
