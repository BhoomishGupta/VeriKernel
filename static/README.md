# VeriKernel: Physics-Aware AI Code Optimization Referee

[![Live Evaluator](https://img.shields.io/badge/Live_Evaluator-VIP_Judge_Passkey-00ffff?style=for-the-badge&logo=fastapi)](https://bhoomishgupta.github.io/VeriKernel/?backend=https://thank-regulate-swell.ngrok-free.dev&judge_token=verikernel-judge-2026)
[![Static Fallback Demo](https://img.shields.io/badge/Static_Demo-Zero--Backend_Mirror-brightgreen?style=for-the-badge&logo=githubpages)](https://bhoomishgupta.github.io/VeriKernel/)
[![N=30 Benchmark](https://img.shields.io/badge/N=30_Benchmark-93.3%25_Verified-blue?style=for-the-badge)](https://bhoomishgupta.github.io/VeriKernel/#tab-evidence)
[![Engineering Feedback](https://img.shields.io/badge/Nebius_Feedback-Engineering_Report-purple?style=for-the-badge)](NEBIUS_FEEDBACK.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> *"Faster code is worthless if the physics is wrong."*

---

### 🚀 Official Submission & Live Evaluation Links for Hackathon Judges

| Evaluation Target | Public Link | Operational Status & Description |
| :--- | :--- | :--- |
| **1. Live VIP Ingress** | **[Launch Live Evaluator](https://bhoomishgupta.github.io/VeriKernel/?backend=https://thank-regulate-swell.ngrok-free.dev&judge_token=verikernel-judge-2026)** | Connected to multi-core VM backend via persistent ngrok HTTPS tunnel (`thank-regulate-swell.ngrok-free.dev`). Grants evaluation privileges via passkey `verikernel-judge-2026`. |
| **2. Standalone Static Mirror** | **[Launch Static Demo](https://bhoomishgupta.github.io/VeriKernel/)** | Zero-dependency static edge fallback hosted on GitHub Pages. Preserves historical telemetry, $N=30$ study data, interactive code diffs, and Ghia plots forever (\$0.00 cost). |
| **3. Technical Feedback Report** | **[View NEBIUS_FEEDBACK.md](NEBIUS_FEEDBACK.md)** | Comprehensive engineering report on Nebius Token Factory inference latency, reasoning token ceilings, thread pinning, and token economics. |

---

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
3. **Stage 3 — Differential Multi-Thread Determinism Gate:** Executes a 30-time-step sweep across thread counts $T \in \{1, 2, 4, 8\}$ with 3 repeated runs each on a $65 \times 65$ grid. If max cross-thread difference exceeds $10^{-11}$, the kernel is rejected for race conditions (`RACE_NON_DETERMINISTIC`) without the memory bloat of ThreadSanitizer.
4. **Stage 4 — Physics & Mathematical Oracle:** Runs a full 2D Navier-Stokes Lid-Driven Cavity simulation ($\text{Re}=100$, $\Delta t=0.001$, grid $129 \times 129$) until steady state ($\max |\omega^{n+1} - \omega^n| \le 10^{-6}$). Validates against:
   - **Relative $L_2$ Norm:** $\|\psi_{\text{test}} - \psi_{\text{ref}}\|_{L_2} / \|\psi_{\text{ref}}\|_{L_2} \le 10^{-5}$ compared to precomputed single-threaded reference binary `reference_cavity_129.bin`.
   - **Empirical Ground Truth:** Centerline velocity profile $u(0.5, y)$ and $v(x, 0.5)$ against the 17 benchmark points of **Ghia, Ghia, & Shin (1982)** ($\text{RMSE} \le 0.05$).
5. **Stage 5 — Pinned Out-of-Cache Benchmark:** Executes 100 fixed Poisson sweeps on an out-of-L2 $513 \times 513$ grid over 5 trials with thread pinning (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`). Computes median execution time and effective memory bandwidth (GB/s).

## Break-It Gallery (Adversarial Verification Suite)

To prove that the referee deterministically isolates bugs without requiring live LLM token usage, VeriKernel includes pre-baked adversarial kernels in `gallery/`:

| Kernel File | Injected Defect | Caught By Gate | Real-World Failure Mode |
| --- | --- | --- | --- |
| `bug_1_fast_mathcheat.hpp` | `#pragma GCC optimize("fast-math")` compiler pragma | **Stage 1: Linter** | Violates IEEE-754 precision; static regex halts execution before compilation. |
| `bug_2_naive_omp_race.hpp` | Naive `#pragma omp parallel for` over lexicographical Gauss-Seidel | **Stage 3: Determinism** | Run-to-run drift $\Delta = 3.87 \times 10^{-5} > 10^{-11}$. Appears 5.7× faster, but produces non-deterministic physics. |
| `bug_3_boundary_off_by_one.hpp` | Boundary loop indexing terminates at $n - 2$ instead of $n - 1$ | **Stage 4: Physics** | Momentum transfer severed from lid; Ghia RMSE collapses to $0.3872 > 0.05$. |
| `bug_4_fp32_downgrade.hpp` | Inner Poisson relaxation converted from `double` to `float` | **Stage 4: Physics** | Numerical precision limits cause residual to stall above $10^{-6}$; hits 15,000-step timeout. |
| `bug_5_wrong_redblack_index.hpp` | Checkerboard parity checked via 1D index `(i * (n-1) + j) % 2` | **Stage 4: Physics** | Bypasses alternating color update; achieves fake 13.3× speedup (279 GB/s) with 99.0% relative error. |
| `bug_6_wave_naive_startup.hpp` | Naive 1st-order Taylor startup for 2D acoustic wave equation | **Stage 4: Physics (Wave)** | Truncation error degrades observed multi-grid convergence order from $p \approx 2.00$ to $p = 1.008 < 1.90$. |

## Empirical Evaluation Study (N=30)

We conducted an empirical benchmark comparing standard **Ungated One-Shot LLM Synthesis (Arm A)** against **Referee-Gated Iterative Self-Correction (Arm B)** across **3 distinct HPC prompting formulations $\times$ 10 trials each ($N=30$ total runs per arm)**:
1. **Direct Pragma Optimization:** Prompt requesting OpenMP parallelization for the inner Poisson Gauss-Seidel loop.
2. **Algorithmic & Parity Restructuring:** Prompt requesting race-free checkerboard/red-black schemes to eliminate loop dependencies.
3. **Hardware & Bandwidth Optimization:** Prompt requesting memory bandwidth optimization with OpenMP SIMD directives on out-of-L2 grids.

> **Identical Experimental Settings:** Both Arm A and Arm B used identical system prompts, identical seed user prompts, identical model weights (`nvidia/nemotron-3-super-120b-a12b`), identical sampling temperature ($T=0.2$), and identical context ceilings (`max_tokens=16384`).

**Model:** `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory  
**Workload:** 2D Navier-Stokes Lid-Driven Cavity ($\text{Re} = 100$, $129 \times 129$ steady-state & $513 \times 513$ pinned benchmark)

| Evaluation Metric | Arm A: One-Shot Ungated (Standard LLM) | Arm B: Referee-Gated + Tavily Loop (VeriKernel) | Impact / Significance |
| :--- | :--- | :--- | :--- |
| **Verification Pass Rate** | **4/30 (13.3%)** | **28/30 (93.3%)** | **+80.0% absolute reliability gain** |
| Local Compiler Failures (Stage 2) | 25/30 runs | 2/30 runs (only after 4 multi-turn retries) | Self-correction eliminates syntax & OpenMP loop bugs |
| Physics / Invariant Drift (Stage 4) | 1/30 runs (stealth failure) | 0/30 runs (Relative $L_2 \le 10^{-5}$, Ghia RMSE $\le 0.05$) | Stage 4 blocks bogus speedups with invalid physics |
| **Failure Attribution: Genuine Bugs** | **25 / 26 (96.2%)** | **0 / 2 (0.0%)** | **0% unrecovered algorithmic bugs in Arm B** |
| **Failure Attribution: Token Truncation**| 1 / 26 (3.8%) | 2 / 2 (100.0%) | 16k ceiling reached during attempt 4 refactoring |
| **Mean Attempts to Pass** | 1.00 attempt (no recovery) | **2.10 attempts** | Rapid convergence in ~2 iterations |
| **Median Verified Speedup** | N/A (13.3% baseline sample) | **13.66x over baseline** | Peak performance: **20.86x (437.5 GB/s)** |
| **Mean Inference Cost per Run** | $0.0060 USD | **$0.0115 USD** | Only +$0.0055 USD per fully certified kernel |
| **True Cost per Usable Kernel** | $0.0449 USD | **$0.0123 USD** | **VeriKernel is 3.6x cheaper per valid asset** |

**Why Did 25 of 30 One-Shot Runs Fail to Compile (Defending the Baseline)?**  
A compiler failure rate of 83.3% in Arm A might initially appear high, raising questions about whether the baseline was a "strawman". However, inspection of the compiler diagnostic logs reveals that 25 of the 26 Arm A failures were caused by a single ubiquitous compiler rejection rule: **OpenMP §2.6 Canonical Loop Form**.  
Nemotron-3-Super repeatedly generated loop headers of the form:
```cpp
for (std::size_t i = 1; i + 1 < n; ++i)
```
While `i + 1 < n` is valid ISO C++, OpenMP specifications require loop controlling predicates inside `#pragma omp parallel for` to strictly adhere to canonical form: `var < lb` or `var < ub`. Both GCC and Clang reject compound predicates with `error: invalid controlling predicate`.  
Because one-shot LLM generation provides zero compiler feedback, the model repeated this pattern across all 3 prompt styles. In VeriKernel (Arm B), Stage 2 intercepted this exact GCC error and returned the line number and error message in the Attempt 2 prompt. Nemotron immediately corrected the predicate to `for (std::size_t i = 1; i < n - 1; ++i)`. This confirms that compiler-in-the-loop gating is not a strawman, but a hard requirement for synthesizing valid OpenMP HPC code.

**Defect Attribution:** Of the 26 failed runs in Arm A, 96.2% (25 runs) were genuine algorithmic and syntax defects where the model voluntarily stopped generating tokens (e.g., Run 11 used 2,062 tokens, Run 21 used 1,472 tokens against the 16,384 ceiling). The model was not token-starved; it simply lacked compiler feedback. In contrast, Arm B had 0.0% genuine code defects—its only 2 unverified runs occurred when multi-turn refactoring reached the 16,384 token ceiling on Attempt 4.

**The Case for Stage 4 Physics Gating:** In Arm A Run 14, the model generated code that compiled cleanly and achieved a median throughput of 234.54 GB/s (11.18× speedup). Without VeriKernel, this kernel would have passed standard performance benchmarks. VeriKernel's Stage 4 caught a 99.0% relative $L_2$ error and Ghia RMSE of 0.3884, halting an unphysical kernel from reaching production.

## Model Inference & Load-Bearing Tavily Integration

### NVIDIA Nemotron via Nebius Token Factory

- **Model Endpoint:** `nvidia/nemotron-3-super-120b-a12b` hosted on Nebius Token Factory.
- **Budgeting & Parameters:** Temperature set to `0.2` for reproducible code generation. Context window ceiling set to `max_tokens=16384` (preventing internal chain-of-thought truncation on complex stencils; see [NEBIUS_FEEDBACK.md](NEBIUS_FEEDBACK.md)).
- **Cost Transparency:** Each iteration logs prompt tokens, completion tokens, and real dollar cost based on Nebius Token Factory billing rates.

### Load-Bearing Tavily Search

Rather than using search as decorative context before code generation, Tavily is triggered **only upon referee failure**:

1. When a candidate fails a gate, the referee extracts the exact failure signature (e.g. `RACE_NON_DETERMINISTIC`, `max_thread_delta=3.87e-05`).
2. The recovery orchestrator maps this signature to a targeted query:

   ```
   "Gauss-Seidel OpenMP data race red-black ordering C++ stencil"
   ```
3. Top search results and documentation snippets are injected directly into the retry prompt alongside the harness's quantitative telemetry, steering the model toward valid mathematical structures.

### Tavily Search Ablation & Qualitative Diagnostic Analysis

- **Ablation Limitation Note:** In this $N=30$ study, Arm B evaluates the unified VeriKernel agent loop coupling compiler/runtime/physics referee telemetry with targeted Tavily retrieval. We formally acknowledge as an experimental limitation that an isolated three-arm ablation (Arm C: Referee-Gated retry *without* Tavily web search) was not evaluated in this initial batch. An isolated Arm C trial is planned as an upcoming empirical extension.
- **Qualitative Load-Bearing Telemetry:** When naive parallelization of lexicographical Gauss-Seidel is flagged by Stage 3 (Determinism Gate: `RACE_NON_DETERMINISTIC`, thread drift $\Delta = 3.87 \times 10^{-5} > 10^{-11}$), LLMs operating solely on prompt instructions routinely attempt naive pragma variants (`#pragma omp parallel for schedule(guided, 8)` or reduction clauses), which repeatedly fail determinism. 
- **Retrieval-Driven Transition:** VeriKernel's recovery orchestrator triggers Tavily specifically on this failure code with the diagnostic query: `"Gauss-Seidel OpenMP data race red-black ordering C++ stencil"`. Tavily injects the canonical two-phase Red-Black checkerboard pattern (`(i + j) % 2 == 0` for Red, `(i + j) % 2 == 1` for Black) into Attempt 2. This steers Nemotron to replace the serial dependency chain with a mathematically exact two-pass barrier, reducing cross-thread divergence from $\Delta = 3.87 \times 10^{-5}$ to bit-exact $\Delta = 0.00 \times 10^{00}$ and enabling Stage 3 to pass.

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

### 4. Run One-Command Verification & Benchmark Suite (Step 10)

Run the master verification suite that executes the baseline cavity solver (5 stages), the 2D wave multi-grid convergence order ($p \ge 1.90$), all 6 adversarial Break-It Gallery defect kernels, and the Stage 5 pinned memory bandwidth benchmark:

```bash
make bench
```

*(Note: `make check` is also provided as an alias).*

### 5. Run Isolated Stage 5 Bandwidth Benchmark

To execute solely the Stage 5 out-of-L2 ($513 \times 513$) memory bandwidth benchmark with thread pinning:

```bash
make bench-bandwidth
```

### 6. Verify the Adversarial Break-It Gallery

Execute all 6 gallery defect kernels to verify that each designated gate deterministically rejects defects:

```bash
make gallery
```

## Token Economics Summary

Inference evaluated on `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory (\$0.30/M input tokens, \$0.90/M output tokens):

| Inference Metric | Arm A: One-Shot Ungated (Standard LLM) | Arm B: Referee-Gated + Tavily (VeriKernel) | Economic Impact |
| :--- | :--- | :--- | :--- |
| **Verification Pass Rate** | 13.3% (4 / 30 kernels) | **93.3% (28 / 30 kernels)** | **+80.0% reliability gain** |
| **Mean Cost per Invocation** | \$0.0060 USD | \$0.0115 USD | +\$0.0055 USD per session |
| **Unrecovered Session Spend** | 86.7% (\$0.1560 total) | **6.7% (\$0.0230 total)** | Only 2/30 sessions unrecovered |
| **Effective Cost per Usable Kernel** | **\$0.0449 USD** | **\$0.0123 USD** | **3.6× cheaper per production asset** |

> **Economic Insight & Accounting Details:** One-shot generation appears cheap on paper (\$0.0060/call) but wastes 86.7% of budget producing syntactically malformed or mathematically non-deterministic code. VeriKernel's automated self-correction loop amortizes multi-turn costs across a 93.3% pass rate, delivering certified high-performance code at more than one-third of the effective cost.
>
> *Note on Cost Accounting:* "Unrecovered Session Spend" strictly measures runs that reached the 4-attempt ceiling without passing verification (26/30 in Arm A vs. only 2/30 in Arm B). Intermediate retry attempts within successful Arm B sessions (averaging 2.10 attempts per passed run) are fully accounted for in the bottom-line **Effective Cost per Usable Kernel** (\$0.3450 total spend / 28 verified kernels = **\$0.0123 USD**), proving that VeriKernel remains **3.6× cheaper per usable production asset** even after paying for all retry tokens.

## Nebius Token Factory & NVIDIA Technical Feedback

As part of the Nebius & NVIDIA hackathon deliverables, we have published a dedicated engineering report:

📄 **[Read the Full Engineering Report: NEBIUS_FEEDBACK.md](NEBIUS_FEEDBACK.md)**

Key topics covered in the report:
1. **The Reasoning Token Ceiling Trap:** Why conservative `max_tokens` ceilings cause internal chain-of-thought truncation, and how expanding to 16,384 tokens unlocked a 93.3% pass rate.
2. **Cluster Latency & Throughput Profile:** Real-world TTFT and generation speeds across prompt sizes on `api.tokenfactory.nebius.com/v1/`.
3. **Inference Economics ($N=30$ Study):** Why ungated generation wasted 86.7% of tokens on invalid code (\$0.0449/usable kernel), whereas VeriKernel's Nemotron loop achieved \$0.0123/usable kernel (**3.6× cheaper per usable production asset**).
4. **Cloud vCPU OpenMP Thread-Pinning:** Overcoming virtualized CPU scheduling jitter using `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.
5. **Platform UX & Roadmap Requests:** Granular `usage.reasoning_tokens`, bit-exact seed pinning, and console consolidation.

## Future Roadmap

1. **3D Cavity & Spectral Navier-Stokes:** Extend the immutable harness from 2D ($129^2$, $513^2$) to 3D staggered grids ($128^3$) and pseudo-spectral methods for high-Reynolds regimes ($\text{Re} \ge 1000$).
2. **GPU Kernel Synthesis (CUDA & Triton):** Integrate GPU Referee gates evaluating warp divergence, shared memory bank conflicts, and NVIDIA Nsight Compute roofline benchmarks.
3. **Formal Invariant Proving:** Integrate symbolic execution (SymPy / Z3) into Stage 1 to formally prove boundary conditions and stencil symmetries before compilation.
4. **Adaptive Tiered Reasoning:** Dynamic model routing escalating from `Nemotron-3-Nano` for syntax repairs to `Nemotron-3-Super` for complex cache blocking and SIMD AVX-512 vectorization.

## Citation & References

- **Empirical Fluid Dynamics Benchmark:** Ghia, U., Ghia, K. N., & Shin, C. T. (1982). *High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method.* Journal of Computational Physics, 48(3), 387-411.
- **Hardware Platform:** Benchmarked on x86_64 multi-core Linux with OpenMP thread pinning (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`).

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.