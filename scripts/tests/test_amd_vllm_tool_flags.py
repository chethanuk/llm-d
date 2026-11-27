"""Regression coverage for Qwen3 tool calling on the AMD vLLM overlays.

The AMD overlays serve Qwen3-32B with vLLM. Without ``--enable-auto-tool-choice``
and the hermes and qwen3 parsers, vLLM rejects requests that carry ``tools`` and
returns thinking text as plain content. The TPU and SGLang overlays already set them.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OVERLAYS = [
    "guides/optimized-baseline/modelserver/amd/vllm/base",
    "guides/optimized-baseline/modelserver/amd/vllm/amd-ci",
    "guides/precise-prefix-cache-routing/modelserver/amd/vllm/base",
    "guides/predicted-latency-routing/modelserver/amd/vllm/base",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/lmcache-connector/cpu/base",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/lmcache-connector/cpu/amd-ci",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/lmcache-connector/fs/base",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/lmcache-connector/fs/amd-ci",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/native/cpu/base",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/native/cpu/amd-ci",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/native/fs/base",
    "guides/tiered-prefix-cache/modelserver/amd/vllm/native/fs/amd-ci",
]
MODEL = "Qwen/Qwen3-32B"


def render(overlay: str) -> str:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        pytest.skip("kubectl is required for the kustomize render regression")
    result = subprocess.run(
        [kubectl, "kustomize", str(ROOT / overlay)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def serve_argv(rendered: str) -> list[str]:
    """argv of the modelserver container, with a shell script entrypoint unpacked."""
    container = next(
        c
        for doc in yaml.safe_load_all(rendered)
        if doc and doc.get("kind") == "Deployment"
        for c in doc["spec"]["template"]["spec"]["containers"]
        if c["name"] == "modelserver"
    )
    command = [str(a) for a in container.get("command") or []]
    args = [str(a) for a in container.get("args") or []]
    if command[-1:] == ["-c"]:
        script = args[0]
        subprocess.run([command[0], "-n"], input=script, text=True, check=True, timeout=10)
        # Skip setup lines (e.g. the Triton sed patch) and read from the exec.
        start = script.index("exec vllm serve")
        joined = script[start:].replace("\\\n", " ").strip()
        # A dropped line continuation would run the remaining flags as a separate command.
        assert "\n" not in joined, "serve script must be a single continued command"
        argv = shlex.split(joined, comments=True)
        return argv[1:]
    return command + args


def serve_tokens(argv: list[str]) -> list[str]:
    """Split "--flag=value" so it reads the same as "--flag value"."""
    tokens = []
    for arg in argv:
        tokens += arg.split("=", 1) if arg.startswith("--") else [arg]
    return tokens


def flag_values(tokens: list[str], flag: str) -> list[str]:
    # A flag left dangling at the end yields None, so it fails the exact-match asserts.
    return [tokens[i + 1] if i + 1 < len(tokens) else None for i, t in enumerate(tokens) if t == flag]


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_qwen3_amd_vllm_overlay_enables_tool_and_reasoning_parsers(overlay: str) -> None:
    argv = serve_argv(render(overlay))
    tokens = serve_tokens(argv)

    assert MODEL in argv
    assert tokens.count("--enable-auto-tool-choice") == 1
    assert flag_values(tokens, "--tool-call-parser") == ["hermes"]
    assert flag_values(tokens, "--reasoning-parser") == ["qwen3"]
