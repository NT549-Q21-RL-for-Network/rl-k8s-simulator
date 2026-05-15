"""Định nghĩa không gian trạng thái cho môi trường self-healing."""

from gymnasium import spaces
import numpy as np


class StateSpace:
    CONTINUOUS_METRICS = {
        "cpu_utilization": (0.0, 1.0),
        "memory_usage": (0.0, 1.0),
        "disk_io": (0.0, 1.0),
        "network_bandwidth": (0.0, 1.0),
        "p90_latency": (0.0, 1.0),
        "p99_latency": (0.0, 1.0),
        "error_rate_5xx": (0.0, 1.0),
        "throughput": (0.0, 1.0),
    }

    DISCRETE_METRICS = {
        "node_ready_status": (0, 3),
        "pending_pods": (0, 100),
        "crashloop_flag": (0, 20),
        "failed_pods": (0, 50),
    }

    @staticmethod
    def get_observation_space() -> spaces.Dict:
        continuous_dim = len(StateSpace.CONTINUOUS_METRICS)
        discrete_ranges = [
            max_val - min_val + 1
            for _, (min_val, max_val) in StateSpace.DISCRETE_METRICS.items()
        ]
        return spaces.Dict(
            {
                "continuous": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(continuous_dim,),
                    dtype=np.float32,
                ),
                "discrete": spaces.MultiDiscrete(discrete_ranges),
            }
        )
