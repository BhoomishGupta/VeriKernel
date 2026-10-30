### VeriKernel Empirical Verification Study (N=30 Benchmark)
**Model:** `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory  
**Workload:** 2D Navier-Stokes Lid-Driven Cavity (Re = 100, 29<sup>2</sup> steady-state & 13<sup>2</sup> pinned benchmark)

| Evaluation Metric | Arm A: One-Shot Ungated (Standard LLM) | Arm B: Referee-Gated + Tavily Loop (VeriKernel) | Impact / Significance |
| :--- | :--- | :--- | :--- |
| **Verification Pass Rate** | **4/30 (13.3%)** | **28/30 (93.3%)** | **+80.0% absolute reliability gain** |
| Local Compiler Failures (Stage 2) | 25/30 runs | 2/30 runs (only after 4 multi-turn retries) | Self-correction eliminates syntax & OpenMP loop bugs |
| Physics / Invariant Drift (Stage 4) | 1/30 runs (stealth failure) | 0/30 runs ( \le 10^{-5}$, Ghia RMSE $\le 0.05$) | Stage 4 blocks bogus speedups with invalid physics |
| **Failure Attribution: Genuine Bugs** | **25 / 26 (96.2%)** | **0 / 2 (0.0%)** | **0% unrecovered algorithmic bugs in Arm B** |
| **Failure Attribution: Token Truncation**| 1 / 26 (3.8%) | 2 / 2 (100.0%) | 16k ceiling reached during attempt 4 refactoring |
| **Mean Attempts to Pass** | 1.00 attempt (no recovery) | **2.10 attempts** | Rapid convergence in ~2 iterations |
| **Median Verified Speedup** | N/A (13.3% baseline sample) | **13.66x over baseline** | Peak performance: **20.86x (437.5 GB/s)** |
| **Mean Inference Cost per Run** | $0.0060 USD | **$0.0115 USD** | Only +$0.0055 USD per fully certified kernel |
| **True Cost per Usable Kernel** | $0.0449 USD | **$0.0123 USD** | **VeriKernel is 3.6x cheaper per valid asset** |
