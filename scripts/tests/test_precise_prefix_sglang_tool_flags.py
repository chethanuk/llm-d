"""Regression coverage for Qwen3 tool calling on the precise-prefix SGLang GPU overlays.

The overlays serve Qwen3-32B with SGLang. Without ``--tool-call-parser`` and
``--reasoning-parser``, SGLang returns tool calls and thinking text as plain
content instead of structured fields.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_ROOT = ROOT / "guides/precise-prefix-cache-routing/modelserver/gpu/sglang"
OVERLAYS = ["base", "gke"]
MODEL = "--model-path=Qwen/Qwen3-32B"


def render(overlay: str) -> str:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        pytest.skip("kubectl is required for the kustomize render regression")
    result = subprocess.run(
        [kubectl, "kustomize", str(OVERLAY_ROOT / overlay)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def modelserver_args(rendered: str) -> list[str]:
    deployment = next(
        doc
        for doc in yaml.safe_load_all(rendered)
        if doc and doc.get("kind") == "Deployment"
    )
    containers = deployment["spec"]["template"]["spec"]["containers"]
    return next(c for c in containers if c["name"] == "modelserver")["args"]


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_qwen3_sglang_overlay_enables_tool_and_reasoning_parsers(overlay: str) -> None:
    args = modelserver_args(render(overlay))

    assert MODEL in args
    assert args.count("--tool-call-parser=qwen") == 1
    assert args.count("--reasoning-parser=qwen3") == 1
