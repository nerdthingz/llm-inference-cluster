# Production LLM Inference Cluster: vLLM + KServe + Kubernetes + A/B Traffic Routing

A local implementation of the inference infrastructure pattern used by production AI teams to serve LLMs at scale — vLLM for high-throughput inference, KServe for Kubernetes-native model serving, Nginx for A/B traffic routing, and Prometheus/Grafana for observability.

## Architecture

```
                    ┌─────────────┐
   Client Request → │    Nginx    │  (80/20 traffic split)
                    └──────┬──────┘
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌───────────────┐    ┌───────────────┐
        │ vLLM (Qwen     │    │ vLLM (Qwen    │
        │ 2.5-1.5B) 80%  │    │ 2.5-0.5B) 20% │
        └───────────────┘    └───────────────┘
                │                     │
                └──────────┬──────────┘
                            ▼
                    ┌──────────────┐
                    │  Prometheus  │ → Grafana Dashboard
                    └──────────────┘
```

GitHub Actions CI/CD: lint → test → build Docker image → push to GitHub Container Registry on every push to `main`.

## Tech Stack

| Component | Role |
|---|---|
| vLLM | PagedAttention-based inference engine, OpenAI-compatible API |
| Docker / Docker Compose | Container packaging and local orchestration |
| Kubernetes (Kind) + KServe | Model-serving abstraction on Kubernetes |
| Nginx Ingress | Weighted A/B traffic routing |
| Prometheus + Grafana | Metrics collection and live dashboards |
| GitHub Actions | CI/CD: lint, test, build, push |

## What Was Built and Verified

### 1. vLLM Inference Serving (Docker Compose) — fully working
Two models served via vLLM's OpenAI-compatible API on a local GPU (GTX 1650, 4GB VRAM):
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct`

### 2. Batch Size Benchmarking — fully working, real numbers
Benchmarked both models at batch sizes 1, 4, 16, and 32 using concurrent async requests (`benchmark/benchmark_qwen.py`, `benchmark/benchmark_qwen_small.py`). Results saved in `qwen_results.json` and `qwen_small_results.json`.

**Qwen2.5-1.5B-Instruct:**

| Batch Size | Tokens/sec | P50 Latency |
|---|---|---|
| 1 | 4.39 | 22.8s |
| 4 | 11.86 | 33.7s |
| 16 | 44.16 | 36.2s |
| 32 | 44.02 | 72.7s |

**Qwen2.5-0.5B-Instruct:**

| Batch Size | Tokens/sec | P50 Latency |
|---|---|---|
| 1 | 8.28 | 12.1s |
| 4 | 36.34 | 11.0s |
| 16 | 129.03 | 12.4s |
| 32 | 130.42 | 24.5s |

**Why throughput scales with batch size:** vLLM's continuous batching adds new requests into a running batch as soon as GPU capacity allows, rather than waiting for the entire batch to finish — keeping the GPU busy instead of idle between requests. PagedAttention allocates each request's KV cache in fixed-size blocks rather than one large contiguous reservation per request, eliminating memory fragmentation and allowing more sequences to share GPU memory concurrently. Throughput plateaus at batch 16→32 because the workload becomes memory-bound rather than compute-bound — note these absolute numbers are constrained by the GTX 1650's 4GB VRAM and an older vLLM version (`v0.6.3`, pinned for Turing-architecture compatibility — see Known Limitations); the scaling *pattern* is what generalizes to production hardware.

### 3. Nginx 80/20 A/B Routing — config built and validated independently, not run concurrently
`nginx/nginx.conf` implements weighted traffic splitting via `split_clients`, routing 80% of requests to the larger model and 20% to the smaller one, with an `X-Backend-Used` response header for verification. A live-traffic test script (`benchmark/ab_test.py`) was built to fire concurrent requests through Nginx and tally the actual split plus per-backend latency. See Known Limitations for why this wasn't run end-to-end on the dev GPU.

### 4. Prometheus + Grafana Monitoring — fully working
Prometheus scrapes vLLM's native `/metrics` endpoint; Grafana dashboard (`monitoring/grafana/dashboards/vllm.json`) visualizes throughput, requests running/waiting, GPU KV cache usage, and P50/P95/P99 time-per-output-token, live, during benchmark runs.

### 5. Kubernetes + KServe — installed and deployed, validated on CPU
Kind cluster with cert-manager, KServe (RawDeployment mode), and Nginx Ingress Controller all installed and running. `InferenceService` CRDs (`k8s/inferenceservice-qwen.yaml`, `k8s/inferenceservice-qwen-small.yaml`) successfully deploy vLLM containers, and a model was confirmed loading and initializing correctly. See Known Limitations for GPU passthrough and stability constraints.

### 6. CI/CD (GitHub Actions) — fully working
`.github/workflows/ci-cd.yaml` runs on every push to `main`: lint (flake8) → test (pytest) → build Docker image → push to `ghcr.io`, tagged with both `latest` and the commit SHA.

## Known Limitations

Built and tested on a GTX 1650 — 4GB VRAM, Turing architecture (compute capability 7.5). VRAM size was the root constraint behind every limitation below.

- **vLLM pinned to v0.6.3.** Current vLLM (`:latest`) hardcodes FlashInfer's `BatchPrefillWithPagedKVCache` kernel for prefill, which throws a CUDA `invalid argument` error on Turing GPUs. v0.6.3 defaults to FlashAttention2, which Turing supports. Hardware/software compatibility boundary, not a config mistake.
- **Live Nginx A/B routing not run end-to-end.** Needs both models' weights + two KV caches in VRAM at once — doesn't fit in 4GB alongside Windows/WSL2/Docker Desktop overhead. Nginx config and the A/B test script (`benchmark/ab_test.py`) are built and ready to run on a higher-VRAM GPU (6GB+, e.g. RTX 3050).
- **No GPU inside Kubernetes.** Kind doesn't expose GPU devices to pods by default. Installed the NVIDIA device plugin — it ran but never advertised a GPU resource, since the NVIDIA Container Toolkit isn't wired into Kind's nested container runtime on Windows/WSL2. `InferenceService`s were validated in CPU mode instead.
- **CPU pod inside K8s was memory-constrained.** Confirmed a pod loading model weights and initializing vLLM successfully inside KServe — proves the deployment pipeline (KServe → Deployment → Service → Ingress) works. Sustained run hit `OOMKilled` from the dev machine's limited system RAM, not a K8s config issue.

In short: every limitation traces back to 4GB VRAM (and thr ram that i had), not the architecture itself.

## Repository Structure

```
llm-inference-cluster/
├── docker-compose.yml
├── Dockerfile
├── nginx/
│   └── nginx.conf
├── benchmark/
│   ├── benchmark_qwen.py
│   ├── benchmark_qwen_small.py
│   ├── ab_test.py
│   └── test_benchmark.py
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── dashboards/vllm.json
│       └── provisioning/
├── k8s/
│   ├── inferenceservice-qwen.yaml
│   ├── inferenceservice-qwen-small.yaml
│   ├── nginx-ingress.yaml
│   └── patch-rbac-proxy.json
├── .github/workflows/ci-cd.yaml
├── qwen_results.json
└── qwen_small_results.json
```

## Running This Project

**Prerequisites:** Docker Desktop with GPU support, NVIDIA GPU (Turing or newer), `kubectl`, `kind`, `helm`.

```bash
# 1. Start a single model
docker compose --profile qwen up

# 2. Run benchmarks
python benchmark/benchmark_qwen.py qwen_results.json

# 3. Start monitoring stack (separate terminal)
docker compose --profile monitoring up
# Prometheus: http://localhost:9090 | Grafana: http://localhost:3000 (admin/admin)

# 4. Kubernetes deployment
kind create cluster --name llm-cluster
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.13.0 -n kserve --create-namespace
helm install kserve oci://ghcr.io/kserve/charts/kserve --version v0.13.0 -n kserve --set kserve.controller.deploymentMode=RawDeployment
kubectl apply -f k8s/inferenceservice-qwen.yaml
```
