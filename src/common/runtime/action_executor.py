from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from kubernetes import client
from kubernetes import config as k8s_config
from kubernetes.client import exceptions as k8s_exceptions


logger = logging.getLogger(__name__)


@dataclass
class K8sActionExecutor:
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        self.dry_run = bool(self.config.get("dry_run", True))
        self.namespace = str(self.config.get("action_namespace", "mini-ecommerce"))
        self.pod_regex = re.compile(str(self.config.get("pod_regex", ".*")))
        self.deployment_regex = re.compile(str(self.config.get("deployment_regex", ".*")))
        self.preferred_workload = str(self.config.get("preferred_workload", "product-service")).strip()
        self.scale_step = max(1, int(self.config.get("scale_step", 1)))
        self.min_replicas = max(0, int(self.config.get("min_replicas", 1)))
        self.max_replicas = max(self.min_replicas, int(self.config.get("max_replicas", 10)))
        self.cooldown_sec = max(0, int(self.config.get("cooldown_sec", 15)))
        self.action_log_path = str(self.config.get("action_log_path", "/tmp/rl-agent/actions.jsonl"))

        self._last_action_at = 0.0
        self._last_action_id = -1
        self._core_v1: client.CoreV1Api | None = None
        self._apps_v1: client.AppsV1Api | None = None

        if not self.dry_run:
            self._ensure_k8s_clients()

    def _ensure_k8s_clients(self) -> None:
        if self._core_v1 is not None and self._apps_v1 is not None:
            return
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        self._core_v1 = client.CoreV1Api()
        self._apps_v1 = client.AppsV1Api()

    @staticmethod
    def _action_name(action_id: int) -> str:
        names = {
            0: "idle",
            1: "restart_pod",
            2: "scale_up",
            3: "scale_down",
            4: "drain_node",
            5: "cordon_node",
            6: "uncordon_node",
        }
        return names.get(action_id, "unknown")

    def _write_action_log(self, payload: Dict[str, Any]) -> None:
        try:
            log_dir = os.path.dirname(self.action_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.action_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning("cannot write action log: %s", exc)

    def _list_candidate_pods(self):
        assert self._core_v1 is not None
        pods = self._core_v1.list_namespaced_pod(namespace=self.namespace).items
        candidates = []
        for pod in pods:
            name = pod.metadata.name if pod.metadata else ""
            if not name or not self.pod_regex.match(name):
                continue
            if pod.status and pod.status.phase != "Running":
                continue
            candidates.append(pod)
        return candidates

    def _list_candidate_deployments(self):
        assert self._apps_v1 is not None
        deps = self._apps_v1.list_namespaced_deployment(namespace=self.namespace).items
        candidates = []
        for dep in deps:
            name = dep.metadata.name if dep.metadata else ""
            if not name or not self.deployment_regex.match(name):
                continue
            candidates.append(dep)
        return candidates

    def _select_pod_for_restart(self):
        candidates = self._list_candidate_pods()
        if not candidates:
            return None
        preferred = [p for p in candidates if p.metadata and p.metadata.name.startswith(self.preferred_workload)]
        pool = preferred if preferred else candidates
        pool.sort(key=lambda p: p.metadata.creation_timestamp or datetime.min.replace(tzinfo=timezone.utc))
        return pool[0]

    def _select_deployment_for_scale(self, direction: str):
        candidates = self._list_candidate_deployments()
        if not candidates:
            return None

        preferred = [d for d in candidates if d.metadata and d.metadata.name.startswith(self.preferred_workload)]
        pool = preferred if preferred else candidates

        if direction == "up":
            valid = [d for d in pool if (d.spec.replicas or 0) < self.max_replicas]
            if not valid:
                return None
            valid.sort(key=lambda d: (d.spec.replicas or 0, d.metadata.name))
            return valid[0]

        valid = [d for d in pool if (d.spec.replicas or 0) > self.min_replicas]
        if not valid:
            return None
        valid.sort(key=lambda d: (-(d.spec.replicas or 0), d.metadata.name))
        return valid[0]

    def _restart_pod(self) -> Dict[str, Any]:
        assert self._core_v1 is not None
        pod = self._select_pod_for_restart()
        if pod is None or pod.metadata is None:
            return {"ok": False, "message": "no running pod matched for restart"}
        name = pod.metadata.name
        self._core_v1.delete_namespaced_pod(name=name, namespace=self.namespace, grace_period_seconds=0)
        return {"ok": True, "pod": name, "namespace": self.namespace, "message": "pod deleted for restart"}

    def _scale_deployment(self, direction: str) -> Dict[str, Any]:
        assert self._apps_v1 is not None
        dep = self._select_deployment_for_scale(direction=direction)
        if dep is None or dep.metadata is None:
            return {"ok": False, "message": f"no deployment available for scale_{direction}"}

        name = dep.metadata.name
        current = dep.spec.replicas or 0
        if direction == "up":
            desired = min(self.max_replicas, current + self.scale_step)
        else:
            desired = max(self.min_replicas, current - self.scale_step)

        if desired == current:
            return {
                "ok": True,
                "deployment": name,
                "replicas_before": current,
                "replicas_after": desired,
                "message": "replicas unchanged",
            }

        body = {"spec": {"replicas": desired}}
        self._apps_v1.patch_namespaced_deployment_scale(name=name, namespace=self.namespace, body=body)
        return {
            "ok": True,
            "deployment": name,
            "namespace": self.namespace,
            "replicas_before": current,
            "replicas_after": desired,
        }

    def execute(self, action_id: int, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        context = context or {}
        action_id = int(action_id)
        action_name = self._action_name(action_id)
        now = time.time()

        base_payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action_id": action_id,
            "action_name": action_name,
            "dry_run": self.dry_run,
            "namespace": self.namespace,
            "context": context,
        }

        if self.dry_run:
            payload = {**base_payload, "executed": False, "ok": True, "message": "action execution is dry-run"}
            self._write_action_log(payload)
            return payload

        if action_id == 0:
            payload = {**base_payload, "executed": False, "ok": True, "message": "idle action"}
            self._write_action_log(payload)
            return payload

        if self.cooldown_sec > 0 and self._last_action_at > 0 and (now - self._last_action_at) < self.cooldown_sec:
            payload = {
                **base_payload,
                "executed": False,
                "ok": False,
                "message": "cooldown active",
                "cooldown_sec": self.cooldown_sec,
                "since_last_action_sec": round(now - self._last_action_at, 3),
                "last_action_id": self._last_action_id,
            }
            self._write_action_log(payload)
            return payload

        try:
            self._ensure_k8s_clients()
            if action_id == 1:
                detail = self._restart_pod()
            elif action_id == 2:
                detail = self._scale_deployment(direction="up")
            elif action_id == 3:
                detail = self._scale_deployment(direction="down")
            else:
                detail = {"ok": False, "message": f"action '{action_name}' not implemented"}

            executed = bool(detail.get("ok")) and action_id in {1, 2, 3}
            if executed:
                self._last_action_at = now
                self._last_action_id = action_id

            payload = {**base_payload, "executed": executed, **detail}
            self._write_action_log(payload)
            return payload
        except k8s_exceptions.ApiException as exc:
            payload = {
                **base_payload,
                "executed": False,
                "ok": False,
                "message": "kubernetes api error",
                "status": getattr(exc, "status", None),
                "reason": getattr(exc, "reason", str(exc)),
                "body": getattr(exc, "body", None),
            }
            self._write_action_log(payload)
            return payload
        except Exception as exc:
            payload = {**base_payload, "executed": False, "ok": False, "message": "executor failure", "error": str(exc)}
            self._write_action_log(payload)
            return payload
