"""Tiện ích vẽ biểu đồ diagnostics."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def create_diagnostic_plots(agent_metrics: Dict, baseline_metrics: Dict, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = ["success_rate", "avg_recovery_steps", "avg_reward"]
    agent_vals = [agent_metrics.get(k, 0) for k in labels]
    base_vals = [baseline_metrics.get(k, 0) for k in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(labels))
    ax.plot(x, agent_vals, marker="o", label="agent")
    ax.plot(x, base_vals, marker="o", label="baseline")
    ax.set_xticks(list(x), labels)
    ax.set_title("Agent vs Baseline")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = run_dir / "diagnostic_plots.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
