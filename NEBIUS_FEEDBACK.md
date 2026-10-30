# Nebius AI Cloud & NVIDIA Nemotron: Production Engineering Feedback Report

> **Project:** VeriKernel — Physics-Aware AI Code Optimization Referee  
> **Author / Maintainer:** Bhoomish Gupta  
> **Platforms Evaluated:** Nebius Token Factory (`api.tokenfactory.nebius.com/v1`), Nebius AI Cloud Compute Sandbox  
> **Models Evaluated:** `nvidia/nemotron-3-super-120b-a12b`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`  
> **Workload Evaluated:** Automated High-Performance Computing (HPC) Navier-Stokes C++ Stencil Synthesis, Multi-Turn Autonomous Refactoring, and Differential Determinism Verification ($N=30$ Study)

---

## Executive Summary

During the architecture and benchmarking of **VeriKernel**, our engineering team conducted an intensive stress-test of the **Nebius Token Factory**, evaluating NVIDIA Nemotron models across multi-turn agentic self-correction loops, tight compiler sandboxes, and hardware-pinned OpenMP multi-core execution.

This document serves as our official technical feedback deliverable for the Nebius / NVIDIA Hackathon. Rather than qualitative impressions, this report provides **quantitative telemetry, failure attribution breakdowns, transport profile analyses, and concrete architectural recommendations** for the Nebius and NVIDIA platform engineering teams.

---

## 1. The "Reasoning Token Trap" & Context Ceiling Dynamics

### The Phenomenon
When querying reasoning-oriented models such as `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` and `nvidia/nemotron-3-super-120b-a12b` for complex C++ HPC refactoring, the model engages in internal chain-of-thought (CoT) planning before emitting visible syntax. 

During our initial development phase with exploratory scripts ([scripts/check_token_factory.py](scripts/check_token_factory.py)), setting conservative limits such as `max_tokens=256` or `max_tokens=2500` resulted in sudden, opaque execution failures:
- The API returned `finish_reason: "length"`.
- The completion payload was either completely empty (`choice.message.content == ""`) or truncated in the middle of a loop boundary predicate (e.g., `for (size_t i = 1; i < n - `).
- Downstream compiler passes failed with syntax errors (`expected ')' before ';' token`), causing the agent loop to reject the model output as malformed syntax rather than recognizing it as context exhaustion.

```
+---------------------------------------------------------------------------------------------------+
|                            THE REASONING TOKEN CEILING DYNAMICS                                   |
+---------------------------------------------------------------------------------------------------+
|  [Conservative: max_tokens = 2,500]                                                              |
|  ├─ Internal Chain-of-Thought Reasoning: 2,150 tokens consumed (invisible or pre-pended)         |
|  └─ Remaining Output Window: 350 tokens ──► TRUNCATED MID-LOOP (Compile Failure)                |
|                                                                                                   |
|  [Tuned Production: max_tokens = 16,384]                                                          |
|  ├─ Internal Chain-of-Thought Reasoning: 2,800 tokens (Algorithmic & Parity Strategy)            |
|  └─ Unbounded Output Window: 1,850 tokens ──► COMPLETE C++ COMPILATION & VERIFIED STENCIL        |
+---------------------------------------------------------------------------------------------------+
```

### The Solution & Telemetry
In `agent_loop.py`, we expanded `MAX_TOKENS = 16384` and introduced an explicit system prompt reasoning directive:
```text
REASONING DIRECTIVE:
Keep your internal reasoning focused and efficient. Formulate your parallelization 
strategy quickly and proceed directly to generating the complete C++ header. 
Do not generate unbounded reasoning traces.
```

### Empirical Study Defect Attribution
In our $N=30$ empirical study comparing **Arm A (One-Shot Ungated)** against **Arm B (VeriKernel Gated)**:
- In **Arm A**, out of 26 failed one-shot runs, **96.2% (25 runs) were genuine algorithmic defects** where the model voluntarily finished generation (`finish_reason: "stop"`, e.g., using 1,472 or 2,062 tokens against the 16,384 ceiling). The model was not token-starved; it simply hallucinated invalid OpenMP loop bounds.
- In **Arm B**, across 30 multi-turn recovery sessions, **0.0% unrecovered algorithmic bugs occurred**. The only 2 runs that failed to reach verification cleared Attempts 1–3 and hit the 16,384-token boundary on Attempt 4 due to the accumulation of four consecutive complete C++ kernel rewrites in the chat context.

> **Key Recommendation for Nebius:** Expose `usage.reasoning_tokens` distinctly in the `usage` metadata (adhering to the latest OpenAI completion spec) and provide streaming chunk metadata (`delta.reasoning_content`) so client orchestrators can monitor reasoning budget burn independently from output token consumption.

---

## 2. Token Factory Latency, TTFT, and Queue Dynamics

### Benchmark Telemetry
We benchmarked API response latencies on the `https://api.tokenfactory.nebius.com/v1/` endpoint across varying hours and workload sizes:

| Workload Stage | Prompt Size | Completion Size | Time-To-First-Token (TTFT) | Total Generation Latency | Effective Generation Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Probe Ping** (`check_token_factory.py`) | 34 tokens | 12 tokens | **210 ms – 420 ms** | 0.82 s | ~15.1 tok/s |
| **Attempt 1 (Initial Seed Prompt)** | 980 tokens | 1,420 tokens | **1.20 s – 2.10 s** | 16.4 s – 24.8 s | ~68.4 tok/s |
| **Attempt 2 (Retry + Referee Feedback)** | 2,150 tokens | 1,890 tokens | **1.65 s – 2.45 s** | 22.1 s – 31.6 s | ~72.1 tok/s |
| **Attempt 3 (Retry + Tavily Stencil Docs)** | 3,840 tokens | 2,210 tokens | **1.95 s – 2.80 s** | 28.5 s – 45.5 s | ~69.8 tok/s |

### Transport Layer Profile & Socket Drops
During peak cluster batching, end-to-end multi-turn requests occasionally required 35–50 seconds. Standard client libraries with default 10-second or 30-second read timeouts threw `httpx.ReadTimeout` exceptions before the first token block was emitted.

To harden the pipeline, we configured an explicit high-resilience transport in `agent_loop.py`:
```python
http_client = httpx.Client(
    timeout=httpx.Timeout(180.0, connect=30.0, read=180.0, write=30.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
)
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    http_client=http_client,
)
```

> **Key Recommendation for Nebius:** During deep reasoning phases or cluster load queuing before first token emission, emit periodic SSE heartbeat comments (`: keep-alive\n\n`) on streaming endpoints to prevent intermediate HTTP proxies and standard API clients from dropping idle TCP connections.

---

## 3. Inference Economics: One-Shot Fallacy vs. Gated Verification

Nebius Token Factory features transparent, competitive inference pricing for `nvidia/nemotron-3-super-120b-a12b`:
- **Prompt Pricing:** \$0.30 per 1,000,000 input tokens
- **Completion Pricing:** \$0.90 per 1,000,000 output tokens

Our $N=30$ study analyzed the true economics of HPC code synthesis:

```
+---------------------------------------------------------------------------------------------------+
|                         INFERENCE ECONOMICS: TRUE COST PER USABLE KERNEL                          |
+---------------------------------------------------------------------------------------------------+
|  [Arm A: One-Shot Ungated (Standard LLM Prompting)]                                               |
|  ├─ Mean API Cost per Run: $0.0060 USD                                                            |
|  ├─ Pass Rate: 13.3% (Only 4 of 30 kernels valid)                                                 |
|  ├─ Wasted Spend: $0.1560 on 26 non-compiling or unphysical kernels                               |
|  └─ TRUE FINANCIAL COST PER USABLE KERNEL: $0.0449 USD                                            |
|                                                                                                   |
|  [Arm B: VeriKernel Gated + Nemotron Retry Loop]                                                  |
|  ├─ Mean API Cost per Run: $0.0115 USD (Only +$0.0055 USD per run)                                |
|  ├─ Pass Rate: 93.3% (28 of 30 kernels certified)                                                 |
|  ├─ Mean Iterations to Pass: 2.10 attempts                                                        |
|  └─ TRUE FINANCIAL COST PER USABLE KERNEL: $0.0123 USD  ──► 3.6x CHEAPER PER VALID ASSET!         |
+---------------------------------------------------------------------------------------------------+
```

### The "One-Shot Fallacy"
Naively looking at single API calls suggests that ungated generation is cheaper (\$0.0060 vs. \$0.0115). However, because ungated LLMs produce code that fails compilation (83.3%) or silently violates Navier-Stokes physics (3.3%), **86.7% of ungated spend is completely wasted**. 

By coupling Nemotron-3-Super with the VeriKernel referee gate and targeted Tavily error retrieval, the model converges on verified Red-Black Gauss-Seidel stencils in ~2 iterations, yielding **3.6× lower cost per production-ready kernel**.

---

## 4. Virtualized CPU Threading & OpenMP Behavior on Cloud Infrastructure

### The Challenge of Virtualized vCPUs
Evaluating multi-threaded C++ OpenMP code on shared cloud virtual machines introduces two severe confounding variables:
1. **Thread Migration Jitter:** In unpinned containers, the Linux kernel scheduler migrates threads across physical cores, corrupting L2/L3 cache residency and causing $\pm 25\%$ timing variance on memory-bound stencils.
2. **False Parallel Speedups:** Naive OpenMP pragmas (`#pragma omp parallel for` placed over lexicographical Gauss-Seidel) appear 5×–13× faster on virtualized cores because cache lines are read concurrently, but race conditions corrupt simulation state.

### The VeriKernel Architectural Solution
In our Docker sandbox ([Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml)), we implemented strict core binding:
```yaml
security_opt:
  - no-new-privileges:true
pids_limit: 128
mem_limit: 2g
cpuset: "0-7"
environment:
  - OMP_PROC_BIND=close
  - OMP_PLACES=cores
  - OMP_NUM_THREADS=8
```

### Results on 8 Dedicated Pinned Cores
- **Differential Determinism (Stage 3):** Differential sweeps across $T \in \{1, 2, 4, 8\}$ threads over 3 repeated trials cleanly isolated race conditions with maximum thread divergence $\Delta = 3.79 \times 10^{-5}$ without the $4\times$ memory bloat of ThreadSanitizer.
- **Pinned Bandwidth Benchmark (Stage 5):** On an out-of-L2 $513 \times 513$ grid (100 sweeps), pinning reduced runtime spread to $< 3.2\%$ and reliably measured real throughput jumps:
  - Baseline serial Gauss-Seidel: **20.97 GB/s** (59.70 ms)
  - Verified Red-Black OpenMP Kernel: **88.86 GB/s** (4.24× speedup live) up to **437.5 GB/s peak**.
  - *Cache Footprint Architecture Note:* The working set of the $513 \times 513$ double-precision arrays ($\text{psi} + \omega$) is $2 \times 513^2 \times 8\text{ B} \approx 4.21\text{ MB}$. This footprint is strictly out-of-L2 (exceeding individual ~1 MB per-core L2 caches), but resides within the CPU's shared 32 MB L3 cache. Peak throughput (437.5 GB/s) reflects multi-core on-die L3 cache bandwidth and SIMD vector register reuse rather than saturated off-chip DRAM bus limits (~40–80 GB/s on cloud VMs).

---

## 5. Platform UX, Developer Tooling & Constructive Recommendations

### What Nebius Did Exceptionally Well

1. **Flawless OpenAI SDK Drop-In Compatibility:**  
   Switching our entire multi-turn retry harness to Nebius required nothing more than:
   ```python
   client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=NEBIUS_API_KEY)
   ```
   No custom SDK wrappers or non-standard authentication signatures were necessary.
2. **Cluster Stability & High Concurrency:**  
   Across our $N=30$ study, smoke test sweeps, and Break-It Gallery live executions, the endpoint handled sustained parallel queries with **zero 500, 502, or 503 gateway drops**.
3. **Model Selection:**  
   `nvidia/nemotron-3-super-120b-a12b` proved exceptionally capable at parsing structured C++ compiler diagnostics and translating Tavily-retrieved algorithmic hints into bit-exact Red-Black indexing logic.

### Constructive Feedback & Feature Requests for Nebius

| Feature Area | Current Limitation | Proposed Engineering Enhancement |
| :--- | :--- | :--- |
| **Console Architecture** | Token Factory (`tokenfactory.nebius.com`) and Nebius AI Cloud (`console.nebius.com`) are separated into two distinct portals with different auth and billing states. | **Unified Console:** Consolidate Token Factory API key management, model exploration, and VM infrastructure under a single unified dashboard. |
| **API Cost Metadata** | The API returns token counts (`prompt_tokens`, `completion_tokens`), but does not provide actual dollar amounts. | **In-Response Billing Telemetry:** Return computed `usage.cost_usd` or rate tier headers in HTTP responses to eliminate manual client-side rate tables. |
| **Reasoning Visibility** | Internal reasoning tokens count against `completion_tokens` but are not separated in standard completion responses. | **Granular Reasoning Counters:** Expose `usage.reasoning_tokens` and return streaming chunks under `delta.reasoning_content` to match the latest frontier reasoning standards. |
| **Deterministic Seed Pinning** | Passing `seed=42` with `temperature=0.0` does not consistently yield identical token sequences across distributed GPU batches. | **Bit-Exact Seed Pinning:** Provide reproducible seed enforcement for scientific benchmarks and compliance auditing. |
| **API Key Sub-Scoping** | API keys currently possess full project-wide permissions. | **Scoped Sub-Keys:** Enable creating rate-limited or spend-capped sub-keys (e.g., maximum \$10/day budget) for hackathon demos and public testing. |

---

## Conclusion

The combination of **Nebius Token Factory** infrastructure and **NVIDIA Nemotron** models provides an enterprise-grade engine for scientific code generation. By addressing reasoning-token visibility and unifying platform consoles, Nebius has the opportunity to establish Token Factory as the premier cloud runtime for mission-critical, physics-grounded AI engineering.

*The VeriKernel team thanks Nebius and NVIDIA for supporting this research through the Builders Program and Token Factory access.*
