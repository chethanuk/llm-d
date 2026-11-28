"""Regression coverage for the default model served by the CPU vLLM overlays.

A gated Hugging Face repo returns 403 for a valid token that has not accepted the
licence for that specific repo, which crash-loops the decode pods. The CPU overlays
must serve an ungated model, and the guide metadata and pod labels must agree with it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
# Heuristic: orgs whose HF repos commonly require per-repo licence acceptance.
# Gating is set per repo, so this guards against regressions; it does not prove a model is ungated.
GATED_ORGS = {"meta-llama", "mistralai", "google"}
OVERLAYS = [
    pytest.param("optimized-baseline", "base", id="optimized-baseline-base"),
    pytest.param("optimized-baseline", "gke", id="optimized-baseline-gke"),
    pytest.param("precise-prefix-cache-routing", "base", id="precise-prefix-cache-routing-base"),
    pytest.param("precise-prefix-cache-routing", "gke", id="precise-prefix-cache-routing-gke"),
]


def render(guide: str, provider: str) -> str:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        pytest.skip("kubectl is required for the kustomize render regression")
    result = subprocess.run(
        [kubectl, "kustomize", str(ROOT / f"guides/{guide}/modelserver/cpu/vllm/{provider}")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def serving(rendered: str) -> tuple[dict, dict]:
    """Return (Deployment, vllm serve container)."""
    for doc in yaml.safe_load_all(rendered):
        if doc and doc.get("kind") == "Deployment":
            for container in doc["spec"]["template"]["spec"]["containers"]:
                if container.get("command") == ["vllm", "serve"]:
                    return doc, container
    raise AssertionError("no Deployment with a `vllm serve` container")


@pytest.mark.parametrize(("guide", "provider"), OVERLAYS)
def test_cpu_overlay_serves_ungated_model_consistently(guide: str, provider: str) -> None:
    deployment, container = serving(render(guide, provider))
    model = container["args"][0]

    org, _, basename = model.rpartition("/")
    assert org not in GATED_ORGS, f"{model} is from an org known to gate models: a token without licence access gets 403"

    guide_yaml = yaml.safe_load((ROOT / f"guides/{guide}/guide.yaml").read_text())
    assert model == guide_yaml["support"]["accelerators"]["cpu"]["model"]

    labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert labels["llm-d.ai/model"] == basename
    assert deployment["spec"]["selector"]["matchLabels"]["llm-d.ai/model"] == basename

    if guide == "precise-prefix-cache-routing":
        args = container["args"]
        config = json.loads(args[args.index("--kv-events-config") + 1])
        assert config["topic"].endswith(f"@{model}")
