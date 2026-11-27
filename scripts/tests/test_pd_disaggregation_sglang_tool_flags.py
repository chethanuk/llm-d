"""Regression coverage for gpt-oss tool calling on the P/D SGLang overlays.

Every ``pd-disaggregation`` SGLang overlay serves gpt-oss-120b through a
``DisaggregatedSet`` with prefill and decode roles. Without SGLang's
``--tool-call-parser=gpt-oss`` and ``--reasoning-parser=gpt-oss`` the server
cannot return tool calls or reasoning content. SGLang has no
``--enable-auto-tool-choice`` flag, so it is not checked here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_ROOT = ROOT / "guides/pd-disaggregation/modelserver/gpu/sglang"
OVERLAYS = ["base", "aws", "coreweave", "gke"]
MODEL_ARG = "--model-path=openai/gpt-oss-120b"


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


def modelserver_containers(rendered: str) -> dict[str, dict]:
    disaggregated_set = next(
        doc
        for doc in yaml.safe_load_all(rendered)
        if doc and doc.get("kind") == "DisaggregatedSet"
    )
    containers = {}
    for role in disaggregated_set["spec"]["roles"]:
        pod_spec = role["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
        for container in pod_spec["containers"]:
            if container["name"] == "modelserver":
                containers[role["name"]] = container
    return containers


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_gpt_oss_sglang_overlay_enables_tool_and_reasoning_parsers(overlay: str) -> None:
    containers = modelserver_containers(render(overlay))

    assert sorted(containers) == ["decode", "prefill"]
    for role, container in containers.items():
        args = container["args"]
        assert MODEL_ARG in args, role
        # SGLang's argparse keeps the last value, so any second parser arg would override gpt-oss.
        for flag in ("--tool-call-parser", "--reasoning-parser"):
            values = [arg for arg in args if arg.split("=", 1)[0] == flag]
            assert values == [f"{flag}=gpt-oss"], (role, values)
