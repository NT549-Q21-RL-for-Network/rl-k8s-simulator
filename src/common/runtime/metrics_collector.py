from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import requests


logger = logging.getLogger(__name__)


def zero_metrics() -> Dict[str, float]:
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
            return zero_metrics()

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
