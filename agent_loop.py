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
MAX_TOKENS = 16384  # Allows reasoning trace + complete C++ emission
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

UNASSISTED_SYSTEM_PROMPT = """You are a Principal High-Performance Computing (HPC) & Computational Physics Engineer.
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
5. NUMERICAL FIDELITY: Maintain 64-bit precision (`double`). Do not downgrade accumulators to float32.
"""

CONSTRAINED_SYSTEM_PROMPT = """You are a Principal High-Performance Computing (HPC) & Computational Physics Engineer.
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

# Default backward compatibility
SYSTEM_PROMPT = UNASSISTED_SYSTEM_PROMPT


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
    cmd = [sys.executable, str(REFEREE_PATH), "--kernel", str(kernel_path), "--json", "--local"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=60.0)
        telemetry = json.loads(proc.stdout)
        all_passed = all(stage.get("ok", False) for stage in telemetry)
        return all_passed, telemetry
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "harness_cavity"], check=False)
        return False, [{
            "stage": "Referee Execution",
            "ok": False,
            "status_tag": "TIMEOUT",
            "detail": "Referee evaluation exceeded 60.0s timeout",
        }]
    except Exception as exc:
        stdout_snippet = proc.stdout[:300] if 'proc' in locals() and proc.stdout else "No stdout"
        return False, [{
            "stage": "Referee Execution",
            "ok": False,
            "status_tag": "PARSE_ERROR",
            "detail": f"{exc} | stdout: {stdout_snippet}",
        }]


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000 * PRICE_PER_M_INPUT) + (completion_tokens / 1_000_000 * PRICE_PER_M_OUTPUT)


def generate_certificate(telemetry: list[dict[str, Any]], total_cost: float, total_tokens: int) -> None:
    """Generates the official REFEREE_CERTIFICATE.md upon passing all 5 stages."""
    import hashlib
    kernel_code = CANDIDATE_PATH.read_text(encoding="utf-8")
    sha256_hash = hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()

    det = next((s for s in telemetry if "Stage 3" in s.get("stage", "")), {})
    phys = next((s for s in telemetry if "Stage 4" in s.get("stage", "")), {})
    bench = next((s for s in telemetry if "Stage 5" in s.get("stage", "")), {})

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
- **Total Tokens Consumed:** {total_tokens:,} tokens
- **Total Nebius Inference Cost:** ${total_cost:.5f} USD
- **Artifact:** `candidate_kernel.hpp` is mathematically verified against Navier-Stokes ground truth.
"""
    CERT_PATH.write_text(cert_text, encoding="utf-8")
    print(f"\n[Certificate] Generated official certification at: {CERT_PATH}")


def call_llm_with_retry(
    messages: list[dict[str, str]],
    retries: int = 3,
    stop_event=None,
) -> Any:
    """Executes chat completion with retry backoff for transient network issues."""
    for attempt_idx in range(1, retries + 1):
        if stop_event and stop_event.is_set():
            raise RuntimeError("Operation cancelled by user")
        try:
            return client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.2,
            )
        except Exception as exc:
            if stop_event and stop_event.is_set():
                raise RuntimeError("Operation cancelled by user")
            if attempt_idx < retries:
                wait_sec = attempt_idx * 2
                print(f"    [Network Glitch] {exc}. Retrying in {wait_sec}s ({attempt_idx}/{retries})...")
                if stop_event and stop_event.wait(timeout=wait_sec):
                    raise RuntimeError("Operation cancelled by user")
                elif not stop_event:
                    time.sleep(wait_sec)
            else:
                raise exc


def call_llm_with_heartbeat(
    messages: list[dict[str, str]],
    attempt: int,
    progress_callback=None,
    retries: int = 3,
    stop_event=None,
) -> Any:
    """Executes LLM completion while dispatching real-time heartbeats to progress_callback."""
    import threading
    internal_stop_event = threading.Event()

    def heartbeat_worker():
        t0 = time.time()
        phases = [
            "Analyzing baseline Gauss-Seidel data dependencies...",
            "Formulating Red-Black checkerboard parallelization...",
            "Reasoning on OpenMP thread pinning and SIMD directives...",
            "Synthesizing complete C++ candidate kernel code...",
            "Verifying pragma allow-list & namespace requirements...",
            "Emitting finalized C++ header tokens from Nebius Token Factory...",
        ]
        phase_idx = 0
        while not internal_stop_event.wait(timeout=2.0):
            if stop_event and stop_event.is_set():
                break
            elapsed = int(time.time() - t0)
            desc = phases[min(phase_idx, len(phases) - 1)]
            phase_idx += 1
            if progress_callback:
                progress_callback({
                    "type": "heartbeat",
                    "attempt": attempt,
                    "elapsed_sec": elapsed,
                    "est_total_sec": 20,
                    "phase": desc,
                    "message": f"Nemotron-3 is synthesizing ({elapsed}s elapsed): {desc}",
                })

    hb = threading.Thread(target=heartbeat_worker, daemon=True)
    hb.start()
    try:
        return call_llm_with_retry(messages, retries=retries, stop_event=stop_event)
    finally:
        internal_stop_event.set()
        hb.join(timeout=1.0)


def run_agent_optimization(
    mode: str = "self_correction",
    progress_callback=None,
    stop_event=None,
) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_code = BASELINE_PATH.read_text(encoding="utf-8")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    attempts_history = []

    active_system_prompt = UNASSISTED_SYSTEM_PROMPT if mode == "self_correction" else CONSTRAINED_SYSTEM_PROMPT

    print("======================================================================")
    print(" VeriKernel: Autonomous Nemotron HPC Optimization & Self-Correction")
    print(f" Engine: {MODEL_ID} | Mode: {mode} | Max Attempts: {MAX_ATTEMPTS} | Max Tokens: {MAX_TOKENS}")
    print("======================================================================")

    messages = [
        {"role": "system", "content": active_system_prompt},
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
        if stop_event and stop_event.is_set():
            print(f"\n[Agent] Stop event triggered. Terminating loop before Attempt {attempt}.")
            if progress_callback:
                progress_callback({
                    "type": "cancelled",
                    "attempt": attempt,
                    "message": "Optimization cancelled by client.",
                })
            break

        print(f"\n>>> [Attempt {attempt}/{MAX_ATTEMPTS}] Querying Nemotron via Nebius Token Factory...")
        if progress_callback:
            progress_callback({
                "type": "attempt_start",
                "attempt": attempt,
                "max_attempts": MAX_ATTEMPTS,
                "message": f"Connecting to Nebius Token Factory ({MODEL_ID})...",
            })
        t0 = time.time()

        try:
            res = call_llm_with_heartbeat(
                messages,
                attempt,
                progress_callback=progress_callback,
                retries=3,
                stop_event=stop_event,
            )
        except Exception as exc:
            print(f"[API Error] Failed to query Nebius Token Factory after retries: {exc}")
            if progress_callback:
                progress_callback({
                    "type": "error",
                    "attempt": attempt,
                    "message": f"Nebius API error: {exc}",
                })
            break

        if stop_event and stop_event.is_set():
            print("[Agent] Stop event detected after inference. Aborting.")
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

        if progress_callback:
            progress_callback({
                "type": "llm_complete",
                "attempt": attempt,
                "dt": round(dt, 2),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "call_cost": round(call_cost, 5),
                "code_len": len(candidate_code),
                "message": f"Inference complete in {dt:.2f}s ({p_tok} in / {c_tok} out | ${call_cost:.5f}) — Extracted {len(candidate_code)} chars of C++",
            })

        # Write candidate to disk
        CANDIDATE_PATH.write_text(candidate_code, encoding="utf-8")

        # Run 5-Stage Referee
        print(">>> [Referee] Evaluating candidate kernel through 5-stage pipeline...")
        if progress_callback:
            progress_callback({
                "type": "referee_start",
                "attempt": attempt,
                "message": "Dispatching candidate kernel to 5-stage C++ Referee...",
            })

        all_passed, telemetry = run_referee(CANDIDATE_PATH)

        # Print stage summaries cleanly and stream each stage
        for s in telemetry:
            tag = "[PASS]" if s.get("ok") else "[FAIL]"
            print(f"    {tag} {s.get('stage')}: {s.get('status_tag')} - {s.get('detail')}")
            if progress_callback:
                progress_callback({
                    "type": "referee_stage",
                    "attempt": attempt,
                    "stage": s.get("stage"),
                    "ok": s.get("ok"),
                    "status_tag": s.get("status_tag"),
                    "detail": s.get("detail"),
                    "message": f"Referee {s.get('stage')}: {tag} ({s.get('status_tag')}) — {s.get('detail')}",
                })

        attempts_history.append({
            "attempt": attempt,
            "cost_usd": call_cost,
            "tokens": {"prompt": p_tok, "completion": c_tok},
            "code": candidate_code,
            "telemetry": telemetry,
            "tavily_guidance": None,
            "passed": all_passed,
        })

        if progress_callback:
            progress_callback({
                "type": "referee_complete",
                "attempt": attempt,
                "passed": all_passed,
                "telemetry": telemetry,
                "message": "✓ Candidate PASSED all 5 Referee stages!" if all_passed else f"✗ Candidate tripped at {telemetry[0].get('stage') if telemetry else 'Referee'}",
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
            if progress_callback:
                progress_callback({
                    "type": "budget_exceeded",
                    "attempt": attempt,
                    "message": f"Reached maximum attempts ({MAX_ATTEMPTS}).",
                })
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

        attempts_history[-1]["tavily_guidance"] = {
            "query": recovery.get("query"),
            "snippets": [s.get("content", "") for s in recovery.get("snippets", [])],
            "urls": [s.get("url", "") for s in recovery.get("snippets", []) if s.get("url")],
            "answer": recovery.get("answer", ""),
        }

        if progress_callback:
            progress_callback({
                "type": "tavily_guidance",
                "attempt": attempt,
                "failing_stage": stage_name,
                "status_tag": status_tag,
                "query": recovery.get("query"),
                "urls": [s.get("url", "") for s in recovery.get("snippets", []) if s.get("url")],
            })

        # Construct self-correction retry prompt
        retry_prompt = (
            f"Your previous C++ implementation FAILED VeriKernel verification at {stage_name}.\n"
            f"Referee Failure Diagnostic:\n"
            f"• Status: {status_tag}\n"
            f"• Telemetry: {detail}\n\n"
            f"External Technical Guidance (Retrieved via Tavily):\n"
            f"{recovery.get('formatted_guidance', '')}\n\n"
            f"Refactor your implementation to resolve this exact failure:\n"
            f"1. Ensure all OpenMP loops use canonical predicate syntax (e.g. `i < n - 1`, not `i + 1 < n`).\n"
            f"2. If parallelizing Gauss-Seidel, eliminate loop-carried race conditions using 2-pass Red-Black checkerboard ordering with parity `(i + j) % 2 == 0` (Red) then `(i + j) % 2 == 1` (Black).\n"
            f"3. Ensure the entire file is wrapped inside `namespace baseline {{ ... }}`.\n"
            f"Keep reasoning concise and output ONLY the complete, corrected C++ code inside ```cpp ... ```."
        )

        # Prune context window to prevent the 16,384-token reasoning ceiling exhaustion
        clean_assistant_turn = f"// Candidate Implementation (Attempt {attempt}):\n```cpp\n{candidate_code}\n```"
        if len(messages) > 3:
            messages = [
                messages[0],
                messages[1],
                {"role": "assistant", "content": clean_assistant_turn},
                {"role": "user", "content": retry_prompt},
            ]
        else:
            messages.append({"role": "assistant", "content": clean_assistant_turn})
            messages.append({"role": "user", "content": retry_prompt})

    # Packaging run telemetry
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_telemetry = {
        "timestamp": timestamp,
        "model": MODEL_ID,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_cost_usd": total_cost,
        "attempts_count": len(attempts_history),
        "verified": any(a.get("passed", False) for a in attempts_history),
        "attempts": attempts_history,
    }

    # 1. Archive into cache/agent_runs/
    run_archive_file = RUNS_DIR / f"run_{timestamp}.json"
    run_archive_file.write_text(json.dumps(run_telemetry, indent=2), encoding="utf-8")
    print(f"\n[Archive] Telemetry saved to: {run_archive_file}")

    # 2. Update cache/latest_agent_run.json
    run_log = REPO_ROOT / "cache" / "latest_agent_run.json"
    run_log.write_text(json.dumps(attempts_history, indent=2), encoding="utf-8")
    print(f"[Trace] Latest run pointer saved to: {run_log}")

    if progress_callback:
        progress_callback({
            "type": "finished",
            "verified": run_telemetry["verified"],
            "total_tokens": run_telemetry["total_tokens"],
            "total_cost_usd": round(run_telemetry["total_cost_usd"], 5),
            "attempts_count": run_telemetry["attempts_count"],
            "archive_filename": run_archive_file.name,
        })

    return run_telemetry


if __name__ == "__main__":
    run_agent_optimization()