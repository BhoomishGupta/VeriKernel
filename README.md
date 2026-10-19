# VeriKernel

VeriKernel is a physics-aware C++ HPC code verification engine for iterative stencil kernels.
It provides an immutable lid-driven cavity benchmark harness and a 5-stage referee pipeline
that combines static policy checks, reproducible compilation, determinism testing, physics
regression checks, and pinned-thread timing runs.

## Repository Layout

- `harness_cavity.cpp` – 2D lid-driven cavity harness (`129x129`, `Re=100`) with OpenMP timing and determinism checks over `1/2/4/8` threads.
- `baseline_kernel.hpp` – baseline lexicographical Gauss-Seidel Poisson stencil sweep.
- `referee.py` – 5-stage referee runner.
- `Makefile` – convenience targets: `make bench`, `make test`, `make clean`.

## 5-Stage Referee Architecture

```text
┌──────────────────────────────┐
│ Stage 1: Static Linter       │
│ - reject banned pragmas      │
│ - enforce include allowlist  │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Stage 2: Container Build     │
│ - g++ -O3 -march=native      │
│ -fopenmp (docker gcc:13)     │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Stage 3: Determinism         │
│ - rerun 1/2/4/8 thread cases │
│ - compare array deltas       │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Stage 4: Physics Verification│
│ - Ghia et al. references     │
│ - streamfunction/vorticity   │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Stage 5: Timing Benchmark    │
│ - pinned OMP threads         │
│ - capture wall-clock runtime │
└──────────────────────────────┘
```

## Build and Run

```bash
make bench
make test
make clean
```

## Token Factory / NVIDIA Nemotron

VeriKernel can be used as the deterministic referee layer beneath an LLM-driven optimization
loop. In a Token Factory workflow, generated kernel candidates are submitted to the referee,
which gates acceptance on deterministic numerics and physics checks before publishing scores.
NVIDIA Nemotron-based agents can call this pipeline to benchmark candidate stencil updates and
accept only safe, reproducible speedups.
