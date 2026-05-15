from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class K8sSelfHealingAgent:
    model: Any

    def predict_action(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action.item() if hasattr(action, "item") else action)
