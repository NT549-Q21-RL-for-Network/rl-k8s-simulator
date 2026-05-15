from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

import requests
from kubernetes import client
from kubernetes import config as k8s_config
from kubernetes.client import exceptions as k8s_exceptions


logger = logging.getLogger(__name__)


def _zero_metrics() -> Dict[str, float]:
    return {
        "http_rps": 0.0,
        "http_5xx_rps": 0.0,
        "http_req_failed_rate": 0.0,
        "cpu_utilization": 0.0,
        "memory_usage": 0.0,
        "disk_io": 0.0,
        "network_bandwidth": 0.0,
        "p90_latency": 0.0,
        "p99_latency": 0.0,
        "pending_pods": 0.0,
        "crashloop_flag": 0.0,
        "failed_pods": 0.0,
        "node_not_ready": 0.0,
        "throughput": 0.0,
    }


@dataclass
class K8sMetricsCollector:
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        self.prometheus_url = str(self.config.get("prometheus_url", "")).rstrip("/")
        self.timeout_sec = float(self.config.get("timeout_sec", 2.0))
        self.retry_after_sec = float(self.config.get("retry_after_sec", 20.0))
        self.metrics_namespace = str(self.config.get("metrics_namespace", "mini-ecommerce"))
        self.ksm_job_regex = str(self.config.get("ksm_job_regex", ".*kube-state-metrics.*"))
        self.pod_regex = str(self.config.get("pod_regex", ".*"))
        self._next_allowed_probe = 0.0

    def _prometheus_ready(self) -> bool:
        if not self.prometheus_url:
            return False
        now = time.time()
        if now < self._next_allowed_probe:
            return False
        try:
            response = requests.get(f"{self.prometheus_url}/-/ready", timeout=self.timeout_sec)
            if response.status_code == 200:
                self._next_allowed_probe = 0.0
                return True
        except Exception as exc:
            logger.warning("prometheus not reachable: %s", exc)
        self._next_allowed_probe = now + self.retry_after_sec
        return False

    def _query_value(self, promql: str) -> float | None:
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": promql},
                timeout=self.timeout_sec,
            )
            if response.status_code != 200:
                logger.warning(
                    "prometheus query failed status=%s query=%s body=%s",
                    response.status_code,
                    promql,
                    response.text[:300],
                )
                return None
            payload = response.json()
            if payload.get("status") != "success":
                return None
            result = payload.get("data", {}).get("result", [])
            if not result:
                return None
            return float(result[0]["value"][1])
        except Exception as exc:
            logger.warning("prometheus query failed: %s", exc)
            return None

    def _query_first(self, candidates: Iterable[str], default: float = 0.0) -> float:
        for q in candidates:
            v = self._query_value(q)
            if v is not None:
                return v
        return default

    def collect(self) -> Dict[str, Any]:
        if not self._prometheus_ready():
            return _zero_metrics()

        ns = self.metrics_namespace
        ksm = self.ksm_job_regex
        pod_re = self.pod_regex

        http_rps = self._query_first(
            [
                f'sum(rate(http_requests_total{{namespace="{ns}"}}[1m]))',
                f'sum(rate(http_server_requests_seconds_count{{namespace="{ns}"}}[1m]))',
                f'sum(rate(http_request_duration_seconds_count{{namespace="{ns}"}}[1m]))',
                f'sum(rate(nginx_ingress_controller_requests{{namespace="{ns}"}}[1m]))',
            ],
            default=0.0,
        )

        http_5xx_rps = self._query_first(
            [
                f'sum(rate(http_requests_total{{namespace="{ns}",status=~"5.."}}[1m]))',
                f'sum(rate(http_server_requests_seconds_count{{namespace="{ns}",status=~"5.."}}[1m]))',
                f'sum(rate(http_request_duration_seconds_count{{namespace="{ns}",code=~"5.."}}[1m]))',
                f'sum(rate(nginx_ingress_controller_requests{{namespace="{ns}",status=~"5.."}}[1m]))',
            ],
            default=0.0,
        )

        p90_latency = self._query_first(
            [
                f'histogram_quantile(0.90, sum(rate(http_request_duration_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))',
                f'histogram_quantile(0.90, sum(rate(http_server_requests_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))',
                f'histogram_quantile(0.90, sum(rate(nginx_ingress_controller_request_duration_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))',
            ],
            default=0.0,
        )

        p99_latency = self._query_first(
            [
                f'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))',
                f'histogram_quantile(0.99, sum(rate(http_server_requests_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))',
                f'histogram_quantile(0.99, sum(rate(nginx_ingress_controller_request_duration_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))',
            ],
            default=0.0,
        )

        cpu_used = self._query_first(
            [
                f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",pod=~"{pod_re}",container!="",container!="POD"}}[1m]))',
                f'sum(rate(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate{{namespace="{ns}",pod=~"{pod_re}"}}[1m]))',
            ],
            default=0.0,
        )
        cpu_limit = self._query_first(
            [
                f'sum(kube_pod_container_resource_limits{{namespace="{ns}",pod=~"{pod_re}",resource="cpu",unit="core"}})',
                f'sum(kube_pod_container_resource_limits_cpu_cores{{namespace="{ns}",pod=~"{pod_re}"}})',
                f'sum(kube_pod_container_resource_requests{{namespace="{ns}",pod=~"{pod_re}",resource="cpu",unit="core"}})',
                f'sum(kube_pod_container_resource_requests_cpu_cores{{namespace="{ns}",pod=~"{pod_re}"}})',
            ],
            default=0.0,
        )

        mem_used = self._query_first(
            [
                f'sum(container_memory_working_set_bytes{{namespace="{ns}",pod=~"{pod_re}",container!="",container!="POD"}})',
                f'sum(container_memory_usage_bytes{{namespace="{ns}",pod=~"{pod_re}",container!="",container!="POD"}})',
            ],
            default=0.0,
        )
        mem_limit = self._query_first(
            [
                f'sum(kube_pod_container_resource_limits{{namespace="{ns}",pod=~"{pod_re}",resource="memory",unit="byte"}})',
                f'sum(kube_pod_container_resource_limits_memory_bytes{{namespace="{ns}",pod=~"{pod_re}"}})',
                f'sum(kube_pod_container_resource_requests{{namespace="{ns}",pod=~"{pod_re}",resource="memory",unit="byte"}})',
                f'sum(kube_pod_container_resource_requests_memory_bytes{{namespace="{ns}",pod=~"{pod_re}"}})',
            ],
            default=0.0,
        )

        fs_used = self._query_first(
            [
                f'sum(container_fs_usage_bytes{{namespace="{ns}",pod=~"{pod_re}",container!="",container!="POD"}})',
                f'sum(kubelet_volume_stats_used_bytes{{namespace="{ns}"}})',
            ],
            default=0.0,
        )
        fs_cap = self._query_first(
            [
                f'sum(container_fs_limit_bytes{{namespace="{ns}",pod=~"{pod_re}",container!="",container!="POD"}})',
                f'sum(kubelet_volume_stats_capacity_bytes{{namespace="{ns}"}})',
            ],
            default=0.0,
        )

        net_rx = self._query_first(
            [f'sum(rate(container_network_receive_bytes_total{{namespace="{ns}",pod=~"{pod_re}"}}[1m]))'],
            default=0.0,
        )
        net_tx = self._query_first(
            [f'sum(rate(container_network_transmit_bytes_total{{namespace="{ns}",pod=~"{pod_re}"}}[1m]))'],
            default=0.0,
        )

        # Deduplicate across duplicated scrape targets and count only active app namespace pods.
        pending_pods = self._query_first(
            [f'sum(max by (namespace,pod) (kube_pod_status_phase{{job=~"{ksm}",namespace="{ns}",pod=~"{pod_re}",phase="Pending"}} == 1))'],
            default=0.0,
        )
        failed_pods = self._query_first(
            [
                (
                    f'sum(max by (namespace,pod) ('
                    f'(kube_pod_status_phase{{job=~"{ksm}",namespace="{ns}",pod=~"{pod_re}",phase="Unknown"}} == 1) '
                    f'or on (namespace,pod) (sum by (namespace,pod) '
                    f'(kube_pod_container_status_waiting_reason{{job=~"{ksm}",namespace="{ns}",pod=~"{pod_re}",'
                    f'reason=~"CrashLoopBackOff|ImagePullBackOff|ErrImagePull|CreateContainerError|RunContainerError"}} == 1) > 0)'
                    f'))'
                )
            ],
            default=0.0,
        )
        crashloop = self._query_first(
            [
                f'sum(max by (namespace,pod,container) (kube_pod_container_status_waiting_reason{{job=~"{ksm}",namespace="{ns}",pod=~"{pod_re}",reason="CrashLoopBackOff"}} == 1))'
            ],
            default=0.0,
        )

        node_not_ready = self._query_first(
            [
                (
                    f'sum(max by (node) (kube_node_status_condition{{job=~"{ksm}",condition="Ready",status=~"false|unknown"}} == 1) '
                    f'and on (node) max by (node) (kube_pod_info{{job=~"{ksm}",namespace="{ns}"}}))'
                )
            ],
            default=0.0,
        )

        return {
            "http_rps": max(http_rps, 0.0),
            "http_5xx_rps": max(http_5xx_rps, 0.0),
            "http_req_failed_rate": (http_5xx_rps / http_rps) if http_rps > 0 else 0.0,
            "cpu_utilization": (cpu_used / cpu_limit) if cpu_limit > 0 else 0.0,
            "memory_usage": (mem_used / mem_limit) if mem_limit > 0 else 0.0,
            "disk_io": (fs_used / fs_cap) if fs_cap > 0 else 0.0,
            "network_bandwidth": max(net_rx + net_tx, 0.0),
            "p90_latency": max(p90_latency, 0.0),
            "p99_latency": max(p99_latency, 0.0),
            "pending_pods": max(pending_pods, 0.0),
            "crashloop_flag": max(crashloop, 0.0),
            "failed_pods": max(failed_pods, 0.0),
            "node_not_ready": max(node_not_ready, 0.0),
            "throughput": max(http_rps, 0.0),
        }


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
        names = {0: "idle", 1: "restart_pod", 2: "scale_up", 3: "scale_down", 4: "drain_node", 5: "cordon_node", 6: "uncordon_node"}
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
            return {"ok": True, "deployment": name, "replicas_before": current, "replicas_after": desired, "message": "replicas unchanged"}

        body = {"spec": {"replicas": desired}}
        self._apps_v1.patch_namespaced_deployment_scale(name=name, namespace=self.namespace, body=body)
        return {"ok": True, "deployment": name, "namespace": self.namespace, "replicas_before": current, "replicas_after": desired}

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


@dataclass
class K8sSelfHealingAgent:
    model: Any

    def predict_action(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action.item() if hasattr(action, "item") else action)
