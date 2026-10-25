#!/usr/bin/env python3
"""
VeriKernel: Multi-Attempt Nemotron Self-Correction Loop
Iteratively optimizes C++ physics kernels against the 5-Stage Referee,
incorporating quantitative diagnostic telemetry and Tavily failure recovery.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any
from dotenv import load_dotenv
import httpx
from openai import OpenAI

from tavily_recovery import retrieve_recovery_guidance

load_dotenv()

REPO_ROOT = pathlib.Path(__file__).resolve().parent
REFEREE_PATH = REPO_ROOT / "referee.py"
BASELINE_PATH = REPO_ROOT / "baseline_kernel.hpp"
CANDIDATE_PATH = REPO_ROOT / "candidate_kernel.hpp"
CERT_PATH = REPO_ROOT / "REFEREE_CERTIFICATE.md"
RUNS_DIR = REPO_ROOT / "cache" / "agent_runs"

MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"
MAX_ATTEMPTS = 4
MAX_TOKENS = 16384  # Expanded to allow deep reasoning + full C++ emission without cutoff
PRICE_PER_M_INPUT = 0.30
PRICE_PER_M_OUTPUT = 0.90

# Robust HTTP transport with generous connection/read timeouts
http_client = httpx.Client(
    timeout=httpx.Timeout(180.0, connect=30.0, read=180.0, write=30.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
)

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    http_client=http_client,
)

SYSTEM_PROMPT = """You are a Principal High-Performance Computing (HPC) & Computational Physics Engineer.
Your task is to optimize a C++17 2D Navier-Stokes computational stencil for multi-core x86_64 CPUs using OpenMP and SIMD vectorization.

REASONING DIRECTIVE:
Keep your internal reasoning focused and efficient. Formulate your parallelization strategy quickly and proceed directly to generating the complete C++ header. Do not generate unbounded reasoning traces.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. IMMUTABLE CONTRACT: Output ONLY the C++ header code inside a single ```cpp ... ``` codeblock. Do not include markdown preamble or conversational explanations outside the codeblock.
2. NAMESPACE & FUNCTION SIGNATURES: All functions must reside inside `namespace baseline` with exact signatures:
   - `inline std::size_t idx(std::size_t i, std::size_t j, std::size_t n)`
   - `inline void gauss_seidel_poisson_sweep(double* psi, const double* omega, std::size_t n, double h, std::size_t sweeps = 1)`
   - `inline void vorticity_transport_step(double* omega_new, const double* omega, const double* psi, std::size_t n, double h, double dt, double re)`
3. ALLOWED HEADERS ONLY: `<algorithm>`, `<cmath>`, `<cstddef>`, `<cstdint>`, `<immintrin.h>`, `<omp.h>`.
   Any other include (<iostream>, <vector>, <fstream>, <cstdlib>) triggers an instant Stage 1 rejection.
4. STRICT COMPILER HYGIENE: Never use `#pragma GCC optimize("fast-math")` or inline assembly.
5. DETERMINISM & SAFETY: Lexicographical Gauss-Seidel CANNOT be parallelized with `#pragma omp parallel for` because of loop-carried dependencies (adjacent reads/writes).
   To parallelize without data races, you MUST refactor into a 2-pass Red-Black Gauss-Seidel or Successive Over-Relaxation (SOR) stencil with checkerboard parity `(i + j) % 2 == 0` (Red) then `(i + j) % 2 == 1` (Black).
6. NUMERICAL FIDELITY: Maintain 64-bit precision (`double`). Do not downgrade accumulators to float32. Preserve all boundary indices (loops run `1 <= i < n - 1` and `1 <= j < n - 1`).
"""


def extract_cpp_code(response_text: str) -> str:
    """Extracts raw C++ code, prioritizing blocks declaring namespace baseline."""
    matches = re.findall(r"```(?:cpp|c\+\+|c)?\s*(.*?)\s*```", response_text, re.DOTALL)
    for m in reversed(matches):
        if "namespace baseline" in m or "gauss_seidel_poisson_sweep" in m:
            return m.strip()
    if matches:
        return matches[-1].strip()

    unclosed = re.search(r"```(?:cpp|c\+\+|c)?\s*(.*)", response_text, re.DOTALL)
    if unclosed:
        code = unclosed.group(1).strip()
        if "namespace baseline" in code:
            return code

    return response_text.strip()


def run_referee(kernel_path: pathlib.Path) -> tuple[bool, list[dict[str, Any]]]:
    """Executes referee.py against the candidate kernel and returns structured telemetry."""
    cmd = [sys.executable, str(REFEREE_PATH), "--kernel", str(kernel_path), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=60.0)
    try:
        telemetry = json.loads(proc.stdout)
        all_passed = all(stage.get("ok", False) for stage in telemetry)
        return all_passed, telemetry
    except Exception as exc:
        return False, [{"stage": "Referee Execution", "ok": False, "status_tag": "PARSE_ERROR", "detail": f"{exc} | stdout: {proc.stdout[:300]}"}]


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000 * PRICE_PER_M_INPUT) + (completion_tokens / 1_000_000 * PRICE_PER_M_OUTPUT)


def generate_certificate(telemetry: list[dict[str, Any]], total_cost: float, total_tokens: int) -> None:
    """Generates the official REFEREE_CERTIFICATE.md upon passing all 5 stages."""
    import hashlib
    kernel_code = CANDIDATE_PATH.read_text(encoding="utf-8")
    sha256_hash = hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()

    det = next((s for s in telemetry if s.get("stage") == "Stage 3 (Multi-Thread Determinism)"), {})
    phys = next((s for s in telemetry if s.get("stage") == "Stage 4 (Physics & Math Oracle)"), {})
    bench = next((s for s in telemetry if s.get("stage") == "Stage 5 (Memory Bandwidth Benchmark)"), {})

    cert_text = f"""# VeriKernel Official Verification Certificate
**Issued by:** VeriKernel 5-Stage Physics-Aware Referee  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Model Workhorse:** {MODEL_ID} (Nebius Token Factory)  
**SHA-256 Signature:** `{sha256_hash}`  

---

### Verification Summary
| Verification Stage | Referee Gate | Measured Result | Threshold / Requirement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Static Linter | Allowed headers & pragmas | Whitelist only; zero fast-math | **PASSED** |
| **Stage 2** | Sandbox Compile | `-O3 -march=native -fopenmp` | Clean zero-warning compilation | **PASSED** |
| **Stage 3** | Thread Determinism | `{det.get('detail', 'Zero drift across threads')}` | $\\Delta \\le 10^{{-11}}$ across $T \\in \\{{1, 2, 4, 8\\}}$ | **PASSED** |
| **Stage 4** | Ghia & $L_2$ Oracle | `{phys.get('detail', 'Passed')}` | Relative $L_2 \\le 10^{{-5}}$, Ghia RMSE $\\le 0.05$ | **PASSED** |
| **Stage 5** | Pinned Benchmark | `{bench.get('detail', 'Completed')}` | 100 sweeps on $513\\times 513$ grid | **PASSED** |

---

### Economic & Compute Telemetry
* **Total Tokens Consumed:** {total_tokens:,} tokens
* **Total Nebius Inference Cost:** ${total_cost:.5f} USD
* **Artifact:** `candidate_kernel.hpp` is mathematically verified against Navier-Stokes ground truth.
"""
    CERT_PATH.write_text(cert_text, encoding="utf-8")
    print(f"\n[Certificate] Generated official certification at: {CERT_PATH}")


def call_llm_with_retry(messages: list[dict[str, str]], retries: int = 3) -> Any:
    """Executes chat completion with retry backoff for transient network issues."""
    for attempt_idx in range(1, retries + 1):
        try:
            return client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.2,
            )
        except Exception as exc:
            if attempt_idx < retries:
                wait_sec = attempt_idx * 2
                print(f"    [Network Glitch] {exc}. Retrying in {wait_sec}s ({attempt_idx}/{retries})...")
                time.sleep(wait_sec)
            else:
                raise exc


def run_agent_optimization():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_code = BASELINE_PATH.read_text(encoding="utf-8")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    attempts_history = []

    print("======================================================================")
    print(" VeriKernel: Autonomous Nemotron HPC Optimization & Self-Correction")
    print(f" Engine: {MODEL_ID} | Max Attempts: {MAX_ATTEMPTS} | Max Tokens: {MAX_TOKENS}")
    print("======================================================================")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Optimize the following baseline Lid-Driven Cavity C++ kernel for 8-core CPU performance.\n"
                "Target OpenMP multi-threading and SIMD vectorization while strictly preserving thread determinism and numerical convergence.\n"
                "Keep internal reasoning concise and immediately output the complete C++ header inside ```cpp ... ```:\n\n"
                f"```cpp\n{baseline_code}\n```"
            ),
        },
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n>>> [Attempt {attempt}/{MAX_ATTEMPTS}] Querying Nemotron via Nebius Token Factory...")
        t0 = time.time()

        try:
            res = call_llm_with_retry(messages, retries=3)
        except Exception as exc:
            print(f"[API Error] Failed to query Nebius Token Factory after retries: {exc}")
            break

        dt = time.time() - t0
        usage = res.usage
        p_tok = usage.prompt_tokens
        c_tok = usage.completion_tokens
        call_cost = compute_cost(p_tok, c_tok)

        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_cost += call_cost

        raw_response = res.choices[0].message.content or ""
        reasoning = getattr(res.choices[0].message, "reasoning_content", None)
        candidate_code = extract_cpp_code(raw_response)

        print(f"    Inference completed in {dt:.2f}s ({p_tok} in / {c_tok} out | Cost: ${call_cost:.5f})")
        if reasoning:
            print(f"    Reasoning trace: {reasoning.strip()[:140]}...")
        print(f"    Extracted C++ code: {len(candidate_code)} characters")

        # Write candidate to disk
        CANDIDATE_PATH.write_text(candidate_code, encoding="utf-8")

        # Run 5-Stage Referee
        print(">>> [Referee] Evaluating candidate kernel through 5-stage pipeline...")
        all_passed, telemetry = run_referee(CANDIDATE_PATH)

        # Print stage summaries
        for s in telemetry:
            tag = "[PASS]" if s.get("ok") else "[FAIL]"
            print(f"    {tag} {s.get('stage')}: {s.get('status_tag')} - {s.get('detail')}")

        attempts_history.append({
            "attempt": attempt,
            "cost_usd": call_cost,
            "tokens": {"prompt": p_tok, "completion": c_tok},
            "telemetry": telemetry,
            "passed": all_passed,
        })

        if all_passed:
            print("\n======================================================================")
            print(" VERIFICATION SUCCESS: Candidate kernel passed all 5 Referee stages!")
            print(f" Total Tokens: {total_prompt_tokens + total_completion_tokens:,} | Total Cost: ${total_cost:.5f}")
            print("======================================================================")
            generate_certificate(telemetry, total_cost, total_prompt_tokens + total_completion_tokens)
            break

        if attempt == MAX_ATTEMPTS:
            print(f"\n[Budget Exceeded] Reached maximum attempts ({MAX_ATTEMPTS}) without passing all stages.")
            break

        # Locate first failing stage to trigger load-bearing Tavily recovery
        failing_stage = next((s for s in telemetry if not s.get("ok")), None)
        stage_name = failing_stage.get("stage", "Unknown") if failing_stage else "Unknown"
        status_tag = failing_stage.get("status_tag", "FAIL") if failing_stage else "FAIL"
        detail = failing_stage.get("detail", "") if failing_stage else ""

        print(f"\n>>> [Load-Bearing Tavily] Referee tripped at {stage_name} ({status_tag}). Dispatching query...")
        recovery = retrieve_recovery_guidance(stage_name, status_tag, detail)
        print(f"    Query : {recovery['query']}")
        print(f"    Cached: {recovery.get('cached', False)}")

        # Construct self-correction retry prompt
        retry_prompt = (
            f"Your previous C++ implementation FAILED VeriKernel verification at {stage_name}.\n"
            f"Referee Failure Diagnostic:\n"
            f"• Status: {status_tag}\n"
            f"• Telemetry: {detail}\n\n"
            f"External Technical Guidance (Retrieved via Tavily):\n"
            f"{recovery.get('formatted_guidance', '')}\n\n"
            f"Refactor your implementation to resolve this exact failure. "
            f"Ensure the entire file is wrapped inside `namespace baseline {{ ... }}`. "
            f"Keep reasoning concise and output ONLY the complete, corrected C++ code inside ```cpp ... ```."
        )

        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": retry_prompt})

    # Save complete run history
    run_log = REPO_ROOT / "cache" / "latest_agent_run.json"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    run_log.write_text(json.dumps(attempts_history, indent=2), encoding="utf-8")
    print(f"[Trace] Full run history saved to: {run_log}")


if __name__ == "__main__":
    run_agent_optimization()
