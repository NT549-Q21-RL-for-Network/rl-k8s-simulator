"""Danh mục kịch bản lỗi và hàm lấy mẫu trạng thái ban đầu."""

from typing import Any, Dict
import numpy as np


class FailureScenario:
    SCENARIOS = {
        "node_failure": {
            "description": "Node down / network partition",
            "state": {
                "cpu_utilization": 0.2,
                "memory_usage": 0.1,
                "disk_io": 0.05,
                "network_bandwidth": 0.05,
                "p90_latency": 0.8,
                "p99_latency": 0.95,
                "error_rate_5xx": 0.30,
                "throughput": 0.2,
                "node_ready_status": 3.0,
                "pending_pods": 60.0,
                "crashloop_flag": 0.0,
                "failed_pods": 30.0,
            },
        },
        "pod_crash_loop": {
            "description": "CrashLoopBackOff - app bug",
            "state": {
                "cpu_utilization": 0.3,
                "memory_usage": 0.4,
                "disk_io": 0.1,
                "network_bandwidth": 0.2,
                "p90_latency": 0.4,
                "p99_latency": 0.5,
                "error_rate_5xx": 0.15,
                "throughput": 0.5,
                "node_ready_status": 0.0,
                "pending_pods": 5.0,
                "crashloop_flag": 12.0,
                "failed_pods": 8.0,
            },
        },
        "resource_exhaustion": {
            "description": "High CPU/Memory pressure",
            "state": {
                "cpu_utilization": 0.95,
                "memory_usage": 0.90,
                "disk_io": 0.7,
                "network_bandwidth": 0.5,
                "p90_latency": 0.6,
                "p99_latency": 0.75,
                "error_rate_5xx": 0.05,
                "throughput": 0.3,
                "node_ready_status": 1.0,
                "pending_pods": 25.0,
                "crashloop_flag": 2.0,
                "failed_pods": 5.0,
            },
        },
        "network_degradation": {
            "description": "High latency / packet loss",
            "state": {
                "cpu_utilization": 0.5,
                "memory_usage": 0.4,
                "disk_io": 0.2,
                "network_bandwidth": 0.85,
                "p90_latency": 0.9,
                "p99_latency": 1.0,
                "error_rate_5xx": 0.20,
                "throughput": 0.4,
                "node_ready_status": 0.0,
                "pending_pods": 10.0,
                "crashloop_flag": 3.0,
                "failed_pods": 4.0,
            },
        },
    }

    @staticmethod
    def sample_scenario() -> Dict[str, Any]:
        scenario_name = np.random.choice(list(FailureScenario.SCENARIOS.keys()))
        scenario = FailureScenario.SCENARIOS[scenario_name].copy()
        scenario["name"] = scenario_name

        state = scenario["state"].copy()
        for key, val in state.items():
            noise = np.random.uniform(-0.08, 0.08)
            if key in {"node_ready_status", "pending_pods", "crashloop_flag", "failed_pods"}:
                max_vals = {"node_ready_status": 3, "pending_pods": 100, "crashloop_flag": 20, "failed_pods": 50}
                cap = max_vals.get(key, 1)
                state[key] = float(np.clip(val + noise * cap, 0, cap))
            else:
                state[key] = float(np.clip(val + noise, 0.0, 1.0))

        scenario["state"] = state
        return scenario
