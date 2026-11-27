"""Regression coverage for Qwen3 tool calling on the pod-snapshot GKE overlays.

The overlays serve Qwen3-32B but start the model server with no parser flags.
Without them, tool calls and thinking text come back as plain content instead
of structured fields.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_ROOT = ROOT / "guides/pod-snapshot/modelserver/gpu"
MODEL = "Qwen/Qwen3-32B"

CASES = [
    pytest.param(
        "vllm",
        [
            "--enable-auto-tool-choice",
            "--tool-call-parser=hermes",
            "--reasoning-parser=qwen3",
        ],
        id="vllm-gke",
    ),
    pytest.param(
        "sglang",
        ["--tool-call-parser=qwen", "--reasoning-parser=qwen3"],
        id="sglang-gke",
    ),
]


def render(engine: str) -> str:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        pytest.skip("kubectl is required for the kustomize render regression")
    result = subprocess.run(
        [kubectl, "kustomize", str(OVERLAY_ROOT / engine / "gke")],
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


@pytest.mark.parametrize("engine, flags", CASES)
def test_qwen3_gke_overlay_enables_tool_and_reasoning_parsers(
    engine: str, flags: list[str]
) -> None:
    args = modelserver_args(render(engine))

    # The parser names below are only right for Qwen3.
    assert MODEL in args
    # Match on the flag name so a later override (argparse keeps the last) fails too.
    for flag in flags:
        name = flag.split("=")[0]
        assert [a for a in args if a.split("=")[0] == name] == [flag], args
