"""Định nghĩa không gian hành động cho agent."""

from gymnasium import spaces


class ActionSpace:
    ACTIONS = {
        0: {"name": "idle", "description": "Do nothing, observe"},
        1: {"name": "restart_pod", "description": "Kill and restart a failed pod", "param": "pod_index"},
        2: {"name": "scale_up", "description": "Increase pod replicas by 1", "param": "deployment_index"},
        3: {"name": "scale_down", "description": "Decrease pod replicas by 1", "param": "deployment_index"},
        4: {"name": "drain_node", "description": "Drain node for maintenance", "param": "node_index"},
        5: {"name": "cordon_node", "description": "Mark node as unschedulable", "param": "node_index"},
        6: {"name": "uncordon_node", "description": "Mark node as schedulable", "param": "node_index"},
    }

    @staticmethod
    def get_action_space() -> spaces.Discrete:
        return spaces.Discrete(len(ActionSpace.ACTIONS))

    @staticmethod
    def describe_action(action_id: int) -> str:
        action = ActionSpace.ACTIONS.get(action_id, {"name": "unknown", "description": ""})
        return f"{action['name']}: {action['description']}"
