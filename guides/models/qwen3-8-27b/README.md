# Qwen3.8-27B on H100 with Tiered Prefix Cache

## Overview

This guide serves [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) on NVIDIA H100,
one GPU per replica, with KV-cache offloading to CPU RAM and, optionally, a shared filesystem.

Qwen3.8-27B is a hybrid-attention model: only 16 of its 64 layers are full attention, and the other
48 are Gated DeltaNet (linear attention) layers that keep a fixed-size state per sequence instead of a
per-token KV cache. Its KV cache per token is therefore small (64 KiB in bf16), so a 27B model
fits on a single H100 with a large KV pool. The checkpoint also ships an MTP head, which the guide
uses for speculative decoding.

The guide composes the [Tiered Prefix Cache](../../tiered-prefix-cache/README.md) foundation guide:
each overlay here layers a model-specific delta (model, TP=1, MTP, tool-call and reasoning parsers,
tier size, image pin) on top of the matching `tiered-prefix-cache` native overlay, and the router uses
the same CPU-aware prefix-cache scoring with a capacity sized for this model. For how offloading works
and when to add each tier, read that guide; this one covers only what is specific to Qwen3.8-27B.

## What you get

| Parameter | Value |
| --- | --- |
| Model | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (FP8 weights, about 29 GiB including the MTP head and vision tower) |
| Replicas | 2, each on 1× H100 80 GB (TP=1) |
| Attention | Hybrid: 16 full-attention + 48 Gated DeltaNet layers; vLLM's hybrid KV-cache manager and prefix caching stay on (the default) |
| MTP speculative decoding | On, 3 draft tokens |
| KV cache offload | vLLM `OffloadingConnector`, 32 GiB CPU tier per replica (`VARIANT=cpu`), plus a shared ReadWriteMany filesystem tier (`VARIANT=fs`) |
| Router | Standalone llm-d Router with GPU and CPU prefix-cache scoring, queue and KV-utilization scoring |
| Parsers | `--tool-call-parser=qwen3_coder`, `--reasoning-parser=qwen3` |

## Supported Accelerators and Model Servers

<!-- guide:support start -->
| Accelerator | `ACCELERATOR_TYPE` | Served model | vLLM | Notes |
| --- | --- | --- | --- | --- |
| NVIDIA GPU | `gpu` | `Qwen/Qwen3.8-27B-FP8` | 🟡 community | H100 80 GB reference · 2 replicas × TP=1 (2 GPUs) · FP8 · MTP · `CONNECTOR=native`, `VARIANT`: `cpu`, `fs` |

✅ validated: covered by a nightly E2E workflow · 🟡 community: maintained by the hardware vendor or community, not covered by nightly E2E · ❌ not supported: tracked in the linked issue · — no configuration.
<!-- guide:support end -->

### KV offloading options

`CONNECTOR` picks the offloading implementation and `VARIANT` the tier:

| `CONNECTOR` | `VARIANT` | Tiers | Notes |
| --- | --- | --- | --- |
| `native` | `cpu` (default) | HBM → CPU RAM | vLLM `OffloadingConnector`, 32 GiB per replica in `/dev/shm` |
| `native` | `fs` | HBM → CPU RAM → shared filesystem | `TieringOffloadingSpec` with an `fs` secondary tier at `/mnt/files-storage`; replicas reuse each other's blocks |
| `mooncake-store` | — | — | Coming in a follow-up |

The native connector writes every block it computes to the tier, so the CPU tier holds copies of HBM
blocks. At 32 GiB it is smaller than the ~39 GiB on-GPU KV pool, so `VARIANT=cpu` mostly duplicates
HBM. `VARIANT=fs` adds a shared tier far larger than HBM and is the recommended option.

## Prerequisites

- Have the [proper client tools installed on your local system](../../../helpers/client-setup/README.md) to use this guide.
- A Kubernetes cluster with 2 NVIDIA H100 80 GB GPUs (one per replica) and about 96 GiB of memory per
  replica pod: the CPU tier lives in `/dev/shm` and counts toward the pod's memory limit.
- The overlays pin `vllm/vllm-openai:v0.32.0-cu129` (CUDA 12.9), which runs on NVIDIA drivers older
  than r580. If your nodes run driver r580 or newer, you can use the plain `v0.32.0` tag (CUDA 13.0)
  instead by changing `newTag` in [`modelserver/gpu/vllm/native/common/kustomization.yaml`](modelserver/gpu/vllm/native/common/kustomization.yaml).
- For `VARIANT=fs`, a StorageClass that provides ReadWriteMany PVCs (see
  [Filesystem tier](../../tiered-prefix-cache/README.md#filesystem-tier)).
- Create a [HuggingFace token](../../../helpers/hf-token.md) and export it as `HF_TOKEN` in your shell.
- The overlays set `VLLM_USE_RUST_FRONTEND=1` (vLLM's Rust OpenAI frontend). If a verify step fails,
  remove it from [`modelserver/gpu/vllm/native/common/patch-vllm.yaml`](modelserver/gpu/vllm/native/common/patch-vllm.yaml)
  to fall back to the Python frontend.

### Get the guide

Every command below runs from a local clone of the [llm-d repository](https://github.com/llm-d/llm-d):

<!-- guide:prerequisites.clone start -->
<!-- llm-d-cicd:skip start -->
```bash
export BRANCH=main
git clone https://github.com/llm-d/llm-d.git && cd llm-d && git checkout ${BRANCH}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.clone end -->

### Configure the environment

Set the guide-specific environment variables (`HF_TOKEN` must already be exported, see [Prerequisites](#prerequisites)):

<!-- guide:env.static start -->
```bash
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
export GUIDE_NAME=qwen3-8-27b
export NAMESPACE=llm-d-qwen3-8-27b
export ACCELERATOR_TYPE=gpu # options: gpu
export MODEL_SERVER=vllm # options: vllm
export CONNECTOR=native # options: native; offloading implementation: vLLM OffloadingConnector
export VARIANT=cpu # options: cpu, fs; offload tier: cpu = CPU RAM, fs = CPU RAM + shared filesystem
export INFRA_PROVIDER=base # options: base, gke
export STORAGE_CLASS= # VARIANT=fs only: StorageClass of the shared PVC, empty for the cluster default
export MODEL=Qwen/Qwen3.8-27B-FP8 # the model every overlay serves
source ${REPO_ROOT}/guides/env.sh # defines GAIE_VERSION, ROUTER_CHART_VERSION, router chart URLs, and CURL_TEST_IMAGE
```
<!-- guide:env.static end -->

Install the Gateway API Inference Extension CRDs:

<!-- guide:prerequisites.gaie start -->
```bash
# GAIE_URL is automatically calculated from GAIE_VERSION at ${REPO_ROOT}/guides/env.sh
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml
```
<!-- guide:prerequisites.gaie end -->

Create a target namespace for the installation:

<!-- guide:prerequisites.namespace start -->
```bash
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
```
<!-- guide:prerequisites.namespace end -->

Create the `llm-d-hf-token` secret in your target namespace:

<!-- guide:prerequisites.secrets start -->
<!-- llm-d-cicd:skip start -->
```bash
kubectl create secret generic llm-d-hf-token \
  --from-literal="HF_TOKEN=${HF_TOKEN}" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.secrets end -->

## Deploy

### 1. Deploy the llm-d Router

Prepare the paths to the `helm` values files for the `llm-d` router:

<!-- guide:deploy.router_values start -->
```bash
# Paths to values files
export ROUTER_BASE_VALUES="${REPO_ROOT}/guides/recipes/router/base.values.yaml"
export ROUTER_VALUES="${REPO_ROOT}/guides/models/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml"
```
<!-- guide:deploy.router_values end -->

Deploy the router in [Standalone Mode](../../../docs/architecture/router/proxy.md). Keep the release
name `${GUIDE_NAME}`: the verification steps look up the `${GUIDE_NAME}-epp` Service by that name.

<!-- guide:deploy.standalone start -->
```bash
helm install ${GUIDE_NAME} \
  ${ROUTER_STANDALONE_CHART} \
  -f ${ROUTER_BASE_VALUES} \
  -f ${ROUTER_VALUES} \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```
<!-- guide:deploy.standalone end -->

> [!NOTE]
> The router's `lruCapacityPerServer` is sized for the 32 GiB CPU tier and bf16 KV. If you change the
> tier size or the KV-cache dtype, update it in [`router/qwen3-8-27b.values.yaml`](router/qwen3-8-27b.values.yaml).

### 2. Create the shared storage (`VARIANT=fs` only)

<!-- guide:deploy.storage start -->
```bash
# only when VARIANT=fs:
# ReadWriteMany PVC for the filesystem tier, mounted at /mnt/files-storage
# (the tiered-prefix-cache manifest, reused unchanged)
envsubst < ${REPO_ROOT}/guides/tiered-prefix-cache/manifests/pvc.yaml | kubectl apply -n ${NAMESPACE} -f -
```
<!-- guide:deploy.storage end -->

### 3. Deploy the Model Server

Apply the Kustomize overlay for your offloading tier and infrastructure provider:

<!-- guide:deploy.modelserver start -->
```bash
kubectl apply -n ${NAMESPACE} \
  -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/${CONNECTOR}/${VARIANT}/${INFRA_PROVIDER}/
```
<!-- guide:deploy.modelserver end -->

Wait for the model servers to become ready (the first start downloads about 29 GiB of weights):

```bash
kubectl wait --for=condition=Ready pod -l llm-d.ai/guide=${GUIDE_NAME} -n ${NAMESPACE} --timeout=1800s
```

## Verify

### 1. Get the IP of the Proxy

<!-- guide:verify.endpoint.standalone start -->
```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```
<!-- guide:verify.endpoint.standalone end -->

### 2. Check the shared storage (`VARIANT=fs` only)

<!-- guide:verify.tests.pvc start -->
```bash
# only when VARIANT=fs:
kubectl get pvc llm-d-kv-cache-storage -n ${NAMESPACE}
```
<!-- guide:verify.tests.pvc end -->

### 3. Send Test Requests

Check that `MODEL` matches the model the servers load:

<!-- guide:verify.tests.model start -->
```bash
# Fail fast if MODEL does not match the model the servers load.
# `false` (not `exit 1`) fails the step without closing an interactive shell.
POD=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/guide=${GUIDE_NAME} -o jsonpath='{.items[0].metadata.name}')
SERVED=$(kubectl get --raw "/api/v1/namespaces/${NAMESPACE}/pods/${POD}:8000/proxy/v1/models" \
  | grep -o '"id": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
echo "model servers serve: ${SERVED}"
[ "${SERVED}" = "${MODEL}" ] \
  || { echo "MODEL=${MODEL} does not match the served model: export MODEL=${SERVED}" >&2; false; }
```
<!-- guide:verify.tests.model end -->

Send a completion request through the router:

<!-- guide:verify.tests.request start -->
```bash
kubectl run curl-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'curl -sS -X POST "http://${IP}/v1/completions" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"prompt\": \"How are you today?\"}"'
```
<!-- guide:verify.tests.request end -->

Check that tool calls are parsed:

<!-- guide:verify.tests.tool_call start -->
```bash
# --enable-auto-tool-choice + --tool-call-parser=qwen3_coder must turn the
# model's tool call into a structured `tool_calls` entry. Thinking is off
# so the token budget goes to the call.
kubectl run tool-call-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'R=$(curl -sS -X POST "http://${IP}/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\": \"${MODEL}\", \"max_tokens\": 512, \"chat_template_kwargs\": {\"enable_thinking\": false},
        \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Paris right now?\"}],
        \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\",
          \"description\": \"Get the current weather for a city\",
          \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}],
        \"tool_choice\": \"auto\"}")
    echo "${R}" | grep -Eq "\"tool_calls\": *\[ *\{" && echo "${R}" | grep -q "get_weather" \
      && echo "OK: tool call parsed" \
      || { echo "FAIL: no tool_calls in ${R}"; exit 1; }'
```
<!-- guide:verify.tests.tool_call end -->

Check that reasoning is separated from the answer:

<!-- guide:verify.tests.reasoning start -->
```bash
# --reasoning-parser=qwen3 must split the thinking into its own field.
# Older vLLM names it `reasoning_content`, newer `reasoning`: accept both.
kubectl run reasoning-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'R=$(curl -sS -X POST "http://${IP}/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\": \"${MODEL}\", \"max_tokens\": 2048, \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 times 23?\"}]}")
    echo "${R}" | grep -Eq "\"reasoning(_content)?\": *\"[^\"]" \
      && echo "OK: reasoning separated from content" \
      || { echo "FAIL: empty or missing reasoning field in ${R}"; exit 1; }'
```
<!-- guide:verify.tests.reasoning end -->

### 4. Check KV-Cache Offloading

Send requests that share a long prefix:

<!-- guide:verify.tests.shared_prefix start -->
```bash
# 10 requests that share a ~8k-token prefix and differ only in the last sentence.
# The connector stores whole chunks only: 784-token blocks x blocks_per_chunk 4 = 3136 tokens.
kubectl run prefix-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'P=$(for i in $(seq 1 470); do printf "Evicted KV-cache blocks move to CPU RAM and are loaded back instead of recomputed. "; done)
    for i in $(seq 1 10); do
      curl -sS -o /dev/null -w "request ${i}: HTTP %{http_code}\n" -X POST "http://${IP}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"${P} Question ${i}: what is llm-d?\", \"max_tokens\": 16}"
    done'
```
<!-- guide:verify.tests.shared_prefix end -->

Read the offload counters of each model server:

<!-- guide:verify.tests.offload_metrics start -->
```bash
# Offload counters of every model server pod, read through the
# Kubernetes API server proxy (no port-forward needed).
for pod in $(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/guide=${GUIDE_NAME} -o jsonpath='{.items[*].metadata.name}'); do
  echo "== ${pod}"
  kubectl get --raw "/api/v1/namespaces/${NAMESPACE}/pods/${pod}:8000/proxy/metrics" \
    | grep -E '^vllm:(kv_offload_(store|load)_bytes_total|external_prefix_cache_(hits|queries)_total)' || true
done
```
<!-- guide:verify.tests.offload_metrics end -->

Fail if no block was written to the offload tier:

<!-- guide:verify.tests.offload_gate start -->
```bash
# Gate: the native OffloadingConnector writes every computed block to
# the offload tier, so after the burst above the bytes stored, summed
# over all model server pods, must be above zero.
STORED=0
for pod in $(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/guide=${GUIDE_NAME} -o jsonpath='{.items[*].metadata.name}'); do
  STORED=$(kubectl get --raw "/api/v1/namespaces/${NAMESPACE}/pods/${pod}:8000/proxy/metrics" \
    | awk -v s="${STORED}" '/^vllm:kv_offload_store_bytes_total/ {s += $NF} END {print s + 0}')
done
echo "vllm:kv_offload_store_bytes_total (all pods): ${STORED}"
awk -v s="${STORED}" 'BEGIN {exit !(s > 0)}' \
  || { echo "no KV-cache blocks were written to the offload tier: the connector is not active" >&2; false; }
```
<!-- guide:verify.tests.offload_gate end -->

With `VARIANT=fs`, check that one replica loads a prefix that the other wrote to the shared tier
(`vllm:external_prefix_cache_hits_total` on the second pod should rise):

<!-- guide:verify.tests.shared_tier start -->
```bash
# only when VARIANT=fs and CONNECTOR=native:
# Prime a new prefix on pod A, then send it to pod B, which has never seen
# it and must load it from the shared filesystem tier: B's external prefix
# cache hits must go up. ~8k tokens, so it spans whole 3136-token offload
# chunks (784-token blocks x 4).
READY=($(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/guide=${GUIDE_NAME} \
  -o jsonpath='{range .items[*]}{.metadata.name} {.status.podIP} {.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' \
  | awk 'NF == 3 && $3 == "True" {print $1, $2}'))
hits() {
  kubectl get --raw "/api/v1/namespaces/${NAMESPACE}/pods/$1:8000/proxy/metrics" \
    | awk '/^vllm:external_prefix_cache_hits_total/ {s += $NF} END {print s + 0}'
}
if [ ${#READY[@]} -lt 4 ]; then
  echo "needs at least 2 ready model server pods, found $((${#READY[@]} / 2)): skipping"
else
POD_B=${READY[2]}
BEFORE=$(hits ${POD_B})
kubectl run shared-tier-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="FIRST=${READY[1]}" \
  --env="SECOND=${READY[3]}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'P=$(for i in $(seq 1 470); do printf "The shared filesystem tier lets every replica reuse KV-cache blocks written by another. "; done)
    for ip in ${FIRST} ${SECOND}; do
      curl -fsS -o /dev/null -w "${ip}: HTTP %{http_code}\n" -X POST "http://${ip}:8000/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"${P}\", \"max_tokens\": 1}" || exit 1
      sleep 10
    done'
AFTER=$(hits ${POD_B})
echo "vllm:external_prefix_cache_hits_total on ${POD_B}: ${BEFORE} -> ${AFTER}"
awk -v b="${BEFORE}" -v a="${AFTER}" 'BEGIN {exit !(a > b)}' \
  || { echo "${POD_B} did not load the prefix from the shared filesystem tier" >&2; false; }
fi
```
<!-- guide:verify.tests.shared_tier end -->

## Benchmark

[`inference-perf.yaml`](inference-perf.yaml) runs an
[`inference-perf`](https://github.com/kubernetes-sigs/inference-perf) job with a shared-prefix
workload sized for `VARIANT=fs`: the prefixes (~98 GiB) overflow HBM and are reloaded from the shared
filesystem tier, which is far larger than HBM. The CPU tier only holds copies of HBM blocks, so
`VARIANT=cpu` gains little on this workload. Set the per-run parameters, render it, and apply it:

```bash
# One pass is 500 prompts; use a multiple of 500 so each prefix is revisited after eviction.
export CONCURRENCY_LEVEL=32
export NUM_REQUESTS=1500

kubectl delete job inference-perf -n ${NAMESPACE} --ignore-not-found
envsubst < ${REPO_ROOT}/guides/models/${GUIDE_NAME}/inference-perf.yaml | kubectl apply -n ${NAMESPACE} -f -
kubectl logs -n ${NAMESPACE} job/inference-perf -f
```

### Benchmark Results

No results yet. They will come from the nightly benchmark run and be published here with the
cluster, image digest, and benchmark parameters of each run.

## Cleanup

To remove the deployed components:

<!-- guide:cleanup.modelserver start -->
```bash
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/${CONNECTOR}/${VARIANT}/${INFRA_PROVIDER}
```
<!-- guide:cleanup.modelserver end -->

<!-- guide:cleanup.storage start -->
```bash
# only when VARIANT=fs:
kubectl delete -n ${NAMESPACE} -f ${REPO_ROOT}/guides/tiered-prefix-cache/manifests/pvc.yaml --ignore-not-found=true
```
<!-- guide:cleanup.storage end -->

<!-- guide:cleanup.rest start -->
```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}
```
<!-- llm-d-cicd:skip start -->
```bash
kubectl delete namespace ${NAMESPACE}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:cleanup.rest end -->

## Reference environment

Status: **community**. The overlays target 2× NVIDIA H100 80 GB on Kubernetes (`base`) and GKE
(`gke`) with vLLM `v0.32.0-cu129`, but this guide has **not yet been validated on a cluster** and is
not covered by a nightly E2E workflow. Treat the sizing (32 GiB CPU tier, router capacity) as a
starting point until a validated run is recorded here.

```text
Tier: 2 (Deployable)
Reference environment: 2x NVIDIA H100 80 GB, Kubernetes (base) and GKE, vLLM v0.32.0-cu129.
Gaps to Tier 3: no nightly E2E job on GKE H100, so no status badge in the release matrix.
Gaps to Tier 4: no benchmark results recorded against this guide's overlays.
```

See [Guides: Definition and Policy](../../GUIDES-POLICY.md) for what the tiers mean.
