"""Hàm đánh giá agent trên nhiều episode."""

from typing import Dict

import numpy as np


def evaluate_agent(agent, env, num_episodes: int = 10) -> Dict[str, float]:
    successes = 0
    recovery_steps = []
    episode_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        info = {}

        while True:
            action, _ = agent.predict(obs, deterministic=True)
            action_int = int(action.item() if hasattr(action, "item") else action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break

        if info.get("recovered", False):
            successes += 1
            recovery_steps.append(steps)

        episode_rewards.append(total_reward)

    return {
        "success_rate": successes / num_episodes if num_episodes else 0.0,
        "avg_recovery_steps": float(np.mean(recovery_steps)) if recovery_steps else -1.0,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
    }
