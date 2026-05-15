from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.runtime.k8s_runtime import K8sActionExecutor, K8sMetricsCollector, K8sSelfHealingAgent

RUNNING = True


def _handle_sigterm(signum, frame):
    global RUNNING
    RUNNING = False


def _clamp(v: float, low: float, high: float) -> float:
    return float(max(low, min(high, v)))


def _round_for_output(value, decimals: int):
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, list):
        return [_round_for_output(item, decimals) for item in value]
    if isinstance(value, dict):
        return {k: _round_for_output(v, decimals) for k, v in value.items()}
    return value


def parse_obs(raw: str | None) -> np.ndarray:
    if not raw:
        return np.array([0.3, 0.3, 0.2, 0.3, 0.15, 0.2, 0.02, 0.8, 0.0, 0.02, 0.0, 0.0], dtype=np.float32)

    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 12:
        raise ValueError("OBSERVATION must have exactly 12 comma-separated values")
    arr = np.array(vals, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def metrics_to_observation(metrics: dict, max_rps: float) -> np.ndarray:
    cpu = _clamp(metrics.get("cpu_utilization", 0.0), 0.0, 1.0)
    mem = _clamp(metrics.get("memory_usage", 0.0), 0.0, 1.0)
    disk = _clamp(metrics.get("disk_io", 0.0), 0.0, 1.0)

    net_bytes_per_sec = max(float(metrics.get("network_bandwidth", 0.0)), 0.0)
    net_norm = _clamp(net_bytes_per_sec / 125_000_000.0, 0.0, 1.0)

    p90_ms = max(float(metrics.get("p90_latency", 0.0)), 0.0) * 1000.0
    p99_ms = max(float(metrics.get("p99_latency", 0.0)), 0.0) * 1000.0
    p90_norm = _clamp(p90_ms / 2000.0, 0.0, 1.0)
    p99_norm = _clamp(p99_ms / 3000.0, 0.0, 1.0)

    error_rate = _clamp(metrics.get("http_req_failed_rate", 0.0), 0.0, 1.0)
    throughput = _clamp(float(metrics.get("throughput", 0.0)) / max(max_rps, 1.0), 0.0, 1.0)

    node_not_ready = _clamp(float(metrics.get("node_not_ready", 0.0)) / 3.0, 0.0, 1.0)
    pending = _clamp(float(metrics.get("pending_pods", 0.0)) / 100.0, 0.0, 1.0)
    crash = _clamp(float(metrics.get("crashloop_flag", 0.0)) / 20.0, 0.0, 1.0)
    failed = _clamp(float(metrics.get("failed_pods", 0.0)) / 50.0, 0.0, 1.0)

    return np.array(
        [cpu, mem, disk, net_norm, p90_norm, p99_norm, error_rate, throughput, node_not_ready, pending, crash, failed],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained PPO model in inference mode")
    parser.add_argument("--model-path", default="/app/models/ppo_model.zip")
    parser.add_argument("--interval-sec", type=int, default=int(os.getenv("POLL_INTERVAL_SEC", "5")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--observation", default=None)
    parser.add_argument("--prometheus-url", default=os.getenv("PROMETHEUS_URL", ""))
    parser.add_argument("--metrics-namespace", default=os.getenv("METRICS_NAMESPACE", "mini-ecommerce"))
    parser.add_argument("--ksm-job-regex", default=os.getenv("KSM_JOB_REGEX", ".*kube-state-metrics.*"))
    parser.add_argument("--pod-regex", default=os.getenv("POD_REGEX", ".*"))
    parser.add_argument("--deployment-regex", default=os.getenv("DEPLOYMENT_REGEX", ".*"))
    parser.add_argument("--preferred-workload", default=os.getenv("PREFERRED_WORKLOAD", "product-service"))
    parser.add_argument("--action-log-path", default=os.getenv("ACTION_LOG_PATH", "/tmp/rl-agent/actions.jsonl"))
    parser.add_argument("--action-cooldown-sec", type=int, default=int(os.getenv("ACTION_COOLDOWN_SEC", "15")))
    parser.add_argument("--min-replicas", type=int, default=int(os.getenv("MIN_REPLICAS", "1")))
    parser.add_argument("--max-replicas", type=int, default=int(os.getenv("MAX_REPLICAS", "10")))
    parser.add_argument("--max-rps", type=float, default=float(os.getenv("MAX_RPS", "1000")))
    parser.add_argument("--output-decimals", type=int, default=int(os.getenv("OUTPUT_DECIMALS", "4")))
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN", "true").lower() == "true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    collector = K8sMetricsCollector(
        config={
            "prometheus_url": args.prometheus_url,
            "timeout_sec": 5,
            "metrics_namespace": args.metrics_namespace,
            "ksm_job_regex": args.ksm_job_regex,
            "pod_regex": args.pod_regex,
        }
    )
    executor = K8sActionExecutor(
        config={
            "dry_run": args.dry_run,
            "action_namespace": args.metrics_namespace,
            "pod_regex": args.pod_regex,
            "deployment_regex": args.deployment_regex,
            "preferred_workload": args.preferred_workload,
            "action_log_path": args.action_log_path,
            "cooldown_sec": args.action_cooldown_sec,
            "min_replicas": args.min_replicas,
            "max_replicas": args.max_replicas,
        }
    )

    model = PPO.load(str(model_path))
    agent = K8sSelfHealingAgent(model=model)

    print(f"[infer] model={model_path}")
    print(f"[infer] interval_sec={args.interval_sec}")
    print(f"[infer] prometheus_url={args.prometheus_url or '(disabled)'}")
    print(f"[infer] metrics_namespace={args.metrics_namespace}")
    print(f"[infer] ksm_job_regex={args.ksm_job_regex}")
    print(f"[infer] pod_regex={args.pod_regex}")
    print(f"[infer] deployment_regex={args.deployment_regex}")
    print(f"[infer] preferred_workload={args.preferred_workload}")
    print(f"[infer] action_log_path={args.action_log_path}")
    print(f"[infer] action_cooldown_sec={args.action_cooldown_sec}")
    print(f"[infer] min_replicas={args.min_replicas}")
    print(f"[infer] max_replicas={args.max_replicas}")
    print(f"[infer] output_decimals={args.output_decimals}")
    print(f"[infer] dry_run={args.dry_run}")

    if args.once:
        if args.prometheus_url:
            metrics = collector.collect()
            obs = metrics_to_observation(metrics, args.max_rps)
            source = "prometheus"
        else:
            obs = parse_obs(args.observation)
            metrics = {}
            source = "manual"

        action = agent.predict_action(obs)
        result = executor.execute(
            action,
            context={
                "source": source,
                "metrics": metrics,
                "observation": obs.tolist(),
            },
        )
        payload = {
            "mode": "once",
            "source": source,
            "action_id": action,
            "observation": obs.tolist(),
            "metrics": metrics,
            "execution": result,
        }
        print(json.dumps(_round_for_output(payload, args.output_decimals)))
        return

    while RUNNING:
        if args.prometheus_url:
            metrics = collector.collect()
            obs = metrics_to_observation(metrics, args.max_rps)
            source = "prometheus"
        else:
            obs = parse_obs(args.observation)
            metrics = {}
            source = "manual"

        action = agent.predict_action(obs)
        result = executor.execute(
            action,
            context={
                "source": source,
                "metrics": metrics,
                "observation": obs.tolist(),
            },
        )
        payload = {
            "mode": "loop",
            "source": source,
            "action_id": action,
            "observation": obs.tolist(),
            "metrics": metrics,
            "execution": result,
        }

        print(
            json.dumps(_round_for_output(payload, args.output_decimals)),
            flush=True,
        )
        time.sleep(max(1, args.interval_sec))

    print("[infer] shutdown")


if __name__ == "__main__":
    main()
