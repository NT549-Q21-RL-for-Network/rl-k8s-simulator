"""Baseline policy cho so sánh."""


class BaselineRandomAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, observation, deterministic=True):
        return self.env.action_space.sample(), None

    def get_vec_normalize_env(self):
        return None
