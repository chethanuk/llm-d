"""Render regression for the optimized-baseline CPU vLLM overlays.

Each replica reserves most of a NUMA node's memory (64Gi request, 32Gi KV
cache), so two replicas scheduled onto one host collide on a single NUMA node.
The decode Deployment must ask the scheduler to spread replicas across hosts.
"""

from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

CPU_DIR = Path(__file__).resolve().parents[2] / "guides/optimized-baseline/modelserver/cpu/vllm"


def _render(overlay: str) -> list:
    kubectl = shutil.which("kubectl")
    if kubectl:
        cmd = [kubectl, "kustomize", str(CPU_DIR / overlay)]
    else:
        kustomize = shutil.which("kustomize")
        if not kustomize:
            pytest.skip("kubectl or kustomize is required for the render regression")
        cmd = [kustomize, "build", str(CPU_DIR / overlay)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _selects_own_pods(term: dict, pod_labels: dict) -> bool:
    pod_term = term.get("podAffinityTerm") or {}
    if pod_term.get("topologyKey") != "kubernetes.io/hostname":
        return False
    # A namespace override could point the term away from the Deployment's pods.
    if pod_term.get("namespaces") or pod_term.get("namespaceSelector") is not None:
        return False
    label_selector = pod_term.get("labelSelector") or {}
    # matchExpressions could exclude the pods even when matchLabels match.
    if label_selector.get("matchExpressions"):
        return False
    selector = label_selector.get("matchLabels") or {}
    return bool(selector) and all(pod_labels.get(k) == v for k, v in selector.items())


@pytest.mark.parametrize("overlay", ["base", "gke"])
def test_replicas_prefer_different_hosts(overlay):
    deployments = [d for d in _render(overlay) if d["kind"] == "Deployment"]
    assert len(deployments) == 1, [d["metadata"]["name"] for d in deployments]
    deploy = deployments[0]
    assert (deploy["spec"].get("replicas") or 1) >= 2
    pod = deploy["spec"]["template"]
    affinity = (pod["spec"].get("affinity") or {}).get("podAntiAffinity") or {}
    terms = affinity.get("preferredDuringSchedulingIgnoredDuringExecution") or []
    pod_labels = pod["metadata"].get("labels") or {}
    assert any(
        t.get("weight", 0) > 0 and _selects_own_pods(t, pod_labels) for t in terms
    ), "no hostname podAntiAffinity selecting the decode pods: replicas can share one host and NUMA node"
