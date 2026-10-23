# VeriKernel: Physics-Aware AI Code Optimization Referee

> *"Faster code is worthless if the physics is wrong."*

VeriKernel is an automated high-performance computing (HPC) verification engine and agentic self-correction harness. It evaluates and repairs AI-generated C++ computational stencils against **multi-threaded determinism**, **numerical conservation invariants**, and published empirical fluid dynamics benchmarks.

Standard AI code generation models (such as deep reasoning systems) frequently optimize parallel loops by introducing data races or aggressive floating-point flags (`-ffast-math`) that report 5×–13× speedups while silently corrupting physics simulations. VeriKernel provides a strict, immutable referee sandbox that intercepts silent corruption and uses execution-grounded failure diagnostics to guide LLMs toward mathematically valid parallel algorithms (such as Red-Black Gauss-Seidel with AVX vectorization).

## Architecture Overview

VeriKernel enforces a strict separation between an **Immutable Test Harness** (`harness_cavity.cpp`) and a **Mutable Computational Kernel** (`candidate_kernel.hpp`):

```
                        +----------------------------+
                        |  AI Agent Prompt / Seed    |
                        |   (NVIDIA Nemotron-3)      |
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        |    candidate_kernel.hpp    |
                        +--------------+-------------+
                                       |
                   +-------------------v-------------------+
                   |         5-STAGE REFEREE GATE          |
                   +---------------------------------------+
                   |  [Stage 1] Static Allow-List Linter   | ---> Fail: LINT_FAIL
                   |  [Stage 2] Hardened Sandboxed Build   | ---> Fail: COMPILE_FAIL
                   |  [Stage 3] Multi-Thread Determinism   | ---> Fail: RACE_NON_DETERMINISTIC
                   |  [Stage 4] Ghia Physics & L2 Oracle   | ---> Fail: PHYSICS_FAIL
                   |  [Stage 5] Pinned Bandwidth Benchmark | ---> Fail: BENCH_FAIL
                   +-------------------+-------------------+
                                       |
                 +---------------------+---------------------+
                 | (Pass)                                    | (Fail)
                 v                                           v
    +--------------------------+               +----------------------------+
    |  Referee Certificate     |               |   Load-Bearing Tavily API  |
    |  - Median Latency & GB/s |               |   Failure Signature Search |
    |  - Zero Race Drift Check |               +--------------+-------------+
    |  - Ghia Centerline RMSE  |                              |
    +--------------------------+               +--------------v-------------+
                                               | Multi-Turn Retry Prompt    |
                                               | (Diagnostics + Citations)  |
                                               +--------------+-------------+
                                                              |
                                                              +--> Back to Agent
```

### The 5-Stage Verification Pipeline

1. **Stage 1 — Static Allow-List Linter:** Regex and AST inspect the candidate header. It strictly permits `<cmath>`, `<omp.h>`, `<immintrin.h>`, `<algorithm>`, `<cstddef>`, and `<cstdint>`. Forbids raw file I/O (`<fstream>`), containers (`<vector>`), system process calls (`fork`, `system`), and illegal precision pragmas (`#pragma GCC optimize("fast-math")`).
2. **Stage 2 — Sandboxed Compilation:** Compiles the harness with the candidate kernel using `-O3 -march=native -fopenmp -Wall -std=c++17` inside an isolated container (`gcc:13`) or native hardened environment with a 20-second timeout.
3. **Stage 3 — Differential Multi-Thread Determinism Gate:** Executes a 30-time-step sweep across thread counts T∈{1,2,4,8} with 3 repeated runs each on a 65×65 grid. If max cross-thread difference exceeds 10−11, the kernel is rejected for race conditions (`RACE_NON_DETERMINISTIC`) without the memory bloat of ThreadSanitizer.
4. **Stage 4 — Physics & Mathematical Oracle:** Runs a full 2D Navier-Stokes Lid-Driven Cavity simulation (Re=100, Δt=0.001, grid 129×129) until steady state (max∣ωn+1−ωn∣≤10−6). Validates against:
   - **Relative**L2​**Norm:** ∥ψtest​−ψref​∥L2​​/∥ψref​∥L2​​≤10−5 compared to precomputed single-threaded reference binary `reference_cavity_129.bin`.
   - **Empirical Ground Truth:** Centerline velocity profile u(0.5,y) and v(x,0.5) against the 17 benchmark points of **Ghia, Ghia, & Shin (1982)** (RMSE≤0.05).
5. **Stage 5 — Pinned Out-of-Cache Benchmark:** Executes 100 fixed Poisson sweeps on an out-of-L2 513×513 grid over 5 trials with thread pinning (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`). Computes median execution time and effective memory bandwidth (GB/s).

## Break-It Gallery (Adversarial Verification Suite)

To prove that the referee deterministically isolates bugs without requiring live LLM token usage, VeriKernel includes 5 pre-baked adversarial kernels in `gallery/`:

| Kernel File | Injected Defect | Caught By Gate | Real-World Failure Mode |
| --- | --- | --- | --- |
| `bug_1_naive_omp_race.hpp` | Naive `#pragma omp parallel for` over lexicographical Gauss-Seidel | **Stage 3: Determinism** | Run-to-run drift Δ=3.87×10−5>10−11. Appears 5.7× faster, but produces non-deterministic physics. |
| `bug_2_fast_mathcheat.hpp` | `#pragma GCC optimize("fast-math")` compiler pragma | **Stage 1: Linter** | Violates IEEE-754 precision; static regex halts execution before compilation. |
| `bug_3_boundary_off_by_one.hpp` | Boundary loop indexing terminates at n−2 instead of n−1 | **Stage 4: Physics** | Momentum transfer severed from lid; Ghia RMSE collapses to 0.3872 (>0.05). |
| `bug_4_fp32_downgrade.hpp` | Inner Poisson relaxation converted from `double` to `float` | **Stage 4: Physics** | Numerical precision limits cause residual to stall above 10−6; hits 15,000-step timeout. |
| `bug_5_wrong_redblack_index.hpp` | Checkerboard parity checked via 1D index `(i * (n-1) + j) % 2` | **Stage 4: Physics** | Bypasses alternating color update; achieves fake 13.3× speedup (279 GB/s) with 99% relative error. |

## Empirical Evaluation Study (N=30)

We conducted an empirical benchmark comparing standard **Ungated One-Shot LLM Synthesis** against **Referee-Gated Iterative Self-Correction** across 3 distinct HPC prompting prompts (10 trials each, N=30 total):

| Metric | Ungated One-Shot (Baseline) | VeriKernel Closed-Loop (Gated) |
| --- | --- | --- |
| **Pass Rate (All 5 Gates)** | **10.0%** (3/30) | **93.3%** (28/30) |
| **Stage 1 / 2 Fails (Linter / Syntax)** | 6 / 30 | 0 / 30 (after ≤2 retries) |
| **Stage 3 Fails (Data Races / Drift)** | 14 / 30 | 0 / 30 (resolved via Red-Black) |
| **Stage 4 Fails (Physics / Numerical)** | 7 / 30 | 2 / 30 (max retries exhausted) |
| **Median Valid Speedup** | 1.00× (Baseline fallback) | 3.82×**over baseline** |
| **Average Attempts to Pass** | N/A (1-attempt cap) | 2.1 attempts |
| **Average Token Factory Cost** | ~$0.012 per run | ~$0.041 per verified kernel |

## Model Inference & Load-Bearing Tavily Integration

### NVIDIA Nemotron via Nebius Token Factory

- **Model Endpoint:** `nvidia/nemotron-3-super` / `nvidia/nemotron-3-ultra` hosted on Nebius Token Factory.
- **Budgeting & Parameters:** Temperature set to `0.2` for deterministic code generation. Context capped at `max_tokens=2500` to prevent token runaways on inner loops.
- **Cost Transparency:** Each iteration logs prompt tokens, completion tokens, and real dollar cost based on Nebius Token Factory billing rates.

### Load-Bearing Tavily Search

Rather than using search as decorative context before code generation, Tavily is triggered **only upon referee failure**:

1. When a candidate fails a gate, the referee extracts the exact failure signature (e.g. `RACE_NON_DETERMINISTIC`, `max_thread_delta=3.87e-05`).
2. The recovery orchestrator maps this signature to a targeted query:

   ```
   "Gauss-Seidel OpenMP data race red-black ordering C++ stencil"
   ```
3. Top search results and documentation snippets are injected directly into the retry prompt alongside the harness's quantitative telemetry, steering the model toward valid mathematical structures.

## Quickstart & Reproducibility

### Prerequisites

- Linux / WSL2 (Ubuntu 22.04+ recommended)
- `g++` 11+ with OpenMP support
- `python` 3.10+
- `docker` (optional, for containerized sandbox builds)

### 1. Clone & Setup

```
git clone https://github.com/BhoomishGupta/VeriKernel.git
cd VeriKernel

# Install Python requirements
pip install -r requirements.txt
```

### 2. Generate Golden Ground Truth

Before running full verification, generate the single-threaded baseline reference:

```
make ref
```

*This compiles `harness_cavity.cpp` and runs 13,676 steps to steady state, exporting `reference_cavity_129.bin`.*

### 3. Run the Verification Referee

Run the referee on the baseline kernel:

```
make test
```

*Expected output:*

```
[Stage 1] LINT_PASS: Static linter checks passed
[Stage 2] COMPILE_PASS: Containerized build passed with gcc:13
[Stage 3] DETERMINISM_PASS: Passed with zero drift across threads (max_delta=0.00e+00)
[Stage 4] PHYSICS_PASS: Passed: relative_l2=0.00e+00, Ghia U RMSE=0.0020 in 13676 steps
[Stage 5] BENCH_PASS: Throughput: 21.04 GB/s (Median latency: 59.57 ms)
```

### 4. Run Stage 5 Bandwidth Benchmark

```
make bench
```

### 5. Verify the Adversarial Break-It Gallery

Execute all 5 gallery defect kernels to verify that the gates trigger properly:

```
make gallery
```

## Citation & References

- **Empirical Fluid Dynamics Benchmark:** Ghia, U., Ghia, K. N., & Shin, C. T. (1982). *High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method.* Journal of Computational Physics, 48(3), 387-411.
- **Hardware Platform:** Benchmarked on x86_64 multi-core Linux with OpenMP thread pinning (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`).

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.