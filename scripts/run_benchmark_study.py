#!/usr/bin/env python3
"""
VeriKernel: N=30 Empirical Benchmark Study & Ablation Runner
Compares Arm A (One-Shot Ungated) against Arm B (Referee-Gated + Tavily Loop)
across 3 prompt variants (10 trials each = 30 runs per arm).
"""

from __future__ import annotations

import argparse
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

# Guarantee repository root is on sys.path before local imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tavily_recovery import retrieve_recovery_guidance

load_dotenv()

REFEREE_PATH = REPO_ROOT / "referee.py"
BASELINE_PATH = REPO_ROOT / "baseline_kernel.hpp"
RESULTS_DIR = REPO_ROOT / "cache"
OUTPUT_JSON = RESULTS_DIR / "empirical_study_results.json"
TEMP_KERNEL = REPO_ROOT / "temp_benchmark_kernel.hpp"

MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"
MAX_ATTEMPTS = 4
MAX_TOKENS = 16384
PRICE_PER_M_INPUT = 0.30
PRICE_PER_M_OUTPUT = 0.90
BASELINE_THROUGHPUT_GB_S = 20.97

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
   Any other include triggers an instant Stage 1 rejection.
4. STRICT COMPILER HYGIENE: Never use `#pragma GCC optimize("fast-math")` or inline assembly.
5. DETERMINISM & SAFETY: Lexicographical Gauss-Seidel CANNOT be parallelized with `#pragma omp parallel for` due to loop-carried dependencies.
   To parallelize without data races, you MUST refactor into a 2-pass Red-Black Gauss-Seidel or SOR stencil with checkerboard parity `(i + j) % 2 == 0` then `(i + j) % 2 == 1`.
6. NUMERICAL FIDELITY: Maintain 64-bit precision (`double`). Do not downgrade accumulators to float32. Preserve all boundary indices (loops run `1 <= i < n - 1` and `1 <= j < n - 1`).
"""

PROMPT_VARIANTS = [
    {
        "id": "variant_1_explicit_openmp",
        "description": "Explicit OpenMP & SIMD Stencil Request",
        "prompt": (
            "Optimize the following baseline Lid-Driven Cavity C++ kernel for 8-core CPU performance.\n"
            "Target OpenMP multi-threading and SIMD vectorization while strictly preserving thread determinism and numerical convergence.\n"
            "Keep internal reasoning concise and immediately output the complete C++ header inside ```cpp ... ```:\n\n"
            "```cpp\n{code}\n```"
        ),
    },
    {
        "id": "variant_2_algorithmic_poisson",
        "description": "High-Throughput Poisson & Vorticity Acceleration",
        "prompt": (
            "Accelerate this 2D Poisson and vorticity Navier-Stokes stencil for an 8-core CPU system.\n"
            "Focus on memory bandwidth efficiency and parallel thread scaling, preserving exact mathematical determinism and physical accuracy.\n"
            "Keep internal reasoning concise and immediately output the complete C++ header inside ```cpp ... ```:\n\n"
            "```cpp\n{code}\n```"
        ),
    },
    {
        "id": "variant_3_cfd_loop_tuning",
        "description": "CFD Stencil Loop Parallelization",
        "prompt": (
            "Parallelize and vectorize this computational fluid dynamics (CFD) lid-driven cavity loop in C++17.\n"
            "Maximize L2/L3 cache reuse and thread scaling across cores without data races or numerical drift.\n"
            "Keep internal reasoning concise and immediately output the complete C++ header inside ```cpp ... ```:\n\n"
            "```cpp\n{code}\n```"
        ),
    },
]


def extract_cpp_code(response_text: str) -> str:
    matches = re.findall(r"```(?:cpp|c\+\+|c)?\s*(.*?)\s*```", response_text, re.DOTALL)
    for m in reversed(matches):
        if "namespace baseline" in m or "gauss_seidel_poisson_sweep" in m:
            return m.strip()
    if matches:
        return matches[-1].strip()
    unclosed = re.search(r"```(?:cpp|c\+\+|c)?\s*(.*)", response_text, re.DOTALL)
    if unclosed and "namespace baseline" in unclosed.group(1):
        return unclosed.group(1).strip()
    return response_text.strip()


def run_referee(kernel_path: pathlib.Path) -> tuple[bool, list[dict[str, Any]]]:
    cmd = [sys.executable, str(REFEREE_PATH), "--kernel", str(kernel_path), "--json", "--local"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=60.0)
        telemetry = json.loads(proc.stdout)
        all_passed = all(stage.get("ok", False) for stage in telemetry)
        return all_passed, telemetry
    except Exception as exc:
        return False, [{"stage": "Referee Execution", "ok": False, "status_tag": "PARSE_ERROR", "detail": str(exc)}]


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000 * PRICE_PER_M_INPUT) + (completion_tokens / 1_000_000 * PRICE_PER_M_OUTPUT)


def call_llm_with_retry(messages: list[dict[str, str]], retries: int = 3) -> Any:
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


def extract_metrics(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    failing_stage = next((s for s in telemetry if not s.get("ok")), None)
    bench_stage = next((s for s in telemetry if "Stage 5" in s.get("stage", "")), {})
    
    throughput_gb_s = 0.0
    bench_detail = bench_stage.get("detail", "")
    match = re.search(r"([\d\.]+)\s*GB/s", bench_detail)
    if match:
        throughput_gb_s = float(match.group(1))

    return {
        "passed": failing_stage is None,
        "first_failure_stage": failing_stage.get("stage") if failing_stage else None,
        "first_failure_tag": failing_stage.get("status_tag") if failing_stage else None,
        "first_failure_detail": failing_stage.get("detail") if failing_stage else None,
        "throughput_gb_s": throughput_gb_s,
        "speedup": round(throughput_gb_s / BASELINE_THROUGHPUT_GB_S, 2) if throughput_gb_s > 0 else 0.0,
    }


def run_arm_a_trial(prompt_text: str, baseline_code: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text.format(code=baseline_code)},
    ]
    t0 = time.time()
    res = call_llm_with_retry(messages, retries=3)
    dt = time.time() - t0
    usage = res.usage
    p_tok = usage.prompt_tokens
    c_tok = usage.completion_tokens
    cost = compute_cost(p_tok, c_tok)

    candidate_code = extract_cpp_code(res.choices[0].message.content or "")
    TEMP_KERNEL.write_text(candidate_code, encoding="utf-8")

    all_passed, telemetry = run_referee(TEMP_KERNEL)
    metrics = extract_metrics(telemetry)

    return {
        "arm": "Arm A (One-Shot Ungated)",
        "duration_s": round(dt, 2),
        "tokens": {"prompt": p_tok, "completion": c_tok, "total": p_tok + c_tok},
        "cost_usd": round(cost, 5),
        "telemetry": telemetry,
        **metrics,
    }


def run_arm_b_trial(prompt_text: str, baseline_code: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text.format(code=baseline_code)},
    ]
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    attempts_history = []
    t_start = time.time()
    final_telemetry: list[dict[str, Any]] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = call_llm_with_retry(messages, retries=3)
        usage = res.usage
        p_tok = usage.prompt_tokens
        c_tok = usage.completion_tokens
        cost = compute_cost(p_tok, c_tok)

        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_cost += cost

        raw_resp = res.choices[0].message.content or ""
        candidate_code = extract_cpp_code(raw_resp)
        TEMP_KERNEL.write_text(candidate_code, encoding="utf-8")

        all_passed, telemetry = run_referee(TEMP_KERNEL)
        final_telemetry = telemetry

        attempts_history.append({
            "attempt": attempt,
            "cost_usd": round(cost, 5),
            "tokens": p_tok + c_tok,
            "passed": all_passed,
        })

        if all_passed or attempt == MAX_ATTEMPTS:
            break

        failing_stage = next((s for s in telemetry if not s.get("ok")), None)
        s_name = failing_stage.get("stage", "Unknown") if failing_stage else "Unknown"
        s_tag = failing_stage.get("status_tag", "FAIL") if failing_stage else "FAIL"
        s_detail = failing_stage.get("detail", "") if failing_stage else ""

        recovery = retrieve_recovery_guidance(s_name, s_tag, s_detail)

        retry_prompt = (
            f"Your previous C++ implementation FAILED VeriKernel verification at {s_name}.\n"
            f"Referee Failure Diagnostic:\n"
            f"• Status: {s_tag}\n"
            f"• Telemetry: {s_detail}\n\n"
            f"External Technical Guidance (Retrieved via Tavily):\n"
            f"{recovery.get('formatted_guidance', '')}\n\n"
            f"Refactor your implementation to resolve this exact failure. "
            f"Ensure the entire file is wrapped inside `namespace baseline {{ ... }}`. "
            f"Keep reasoning concise and output ONLY the complete, corrected C++ code inside ```cpp ... ```."
        )
        messages.append({"role": "assistant", "content": raw_resp})
        messages.append({"role": "user", "content": retry_prompt})

    metrics = extract_metrics(final_telemetry)
    return {
        "arm": "Arm B (Referee-Gated + Tavily Loop)",
        "duration_s": round(time.time() - t_start, 2),
        "attempts_needed": len(attempts_history),
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_prompt_tokens + total_completion_tokens,
        },
        "cost_usd": round(total_cost, 5),
        "attempts_history": attempts_history,
        "telemetry": final_telemetry,
        **metrics,
    }


def print_summary_table(study_data: dict[str, Any]) -> None:
    arm_a_results = study_data.get("arm_a", [])
    arm_b_results = study_data.get("arm_b", [])

    total_a = len(arm_a_results)
    pass_a = sum(1 for r in arm_a_results if r.get("passed"))
    rate_a = (pass_a / total_a * 100) if total_a else 0.0

    total_b = len(arm_b_results)
    pass_b = sum(1 for r in arm_b_results if r.get("passed"))
    rate_b = (pass_b / total_b * 100) if total_b else 0.0

    failures_a: dict[str, int] = {}
    for r in arm_a_results:
        if not r.get("passed"):
            tag = r.get("first_failure_tag", "UNKNOWN")
            failures_a[tag] = failures_a.get(tag, 0) + 1

    speedups_b = [r.get("speedup", 0.0) for r in arm_b_results if r.get("passed")]
    median_speedup_b = sorted(speedups_b)[len(speedups_b) // 2] if speedups_b else 0.0
    avg_cost_b = sum(r.get("cost_usd", 0.0) for r in arm_b_results) / total_b if total_b else 0.0
    avg_attempts_b = sum(r.get("attempts_needed", 1) for r in arm_b_results) / total_b if total_b else 0.0

    print("\n" + "=" * 78)
    print(" VeriKernel N=30 Empirical Benchmark Study Results")
    print("=" * 78)
    print(f"Arm A (One-Shot Ungated) Pass Rate : {pass_a}/{total_a} ({rate_a:.1f}%)")
    print("  Arm A Failure Breakdown:")
    for tag, count in sorted(failures_a.items(), key=lambda x: -x[1]):
        print(f"    - {tag:<25} : {count} runs")
    print("-" * 78)
    print(f"Arm B (Referee-Gated + Tavily Loop) : {pass_b}/{total_b} ({rate_b:.1f}%)")
    print(f"  Mean Attempts to Verification  : {avg_attempts_b:.2f} attempts")
    print(f"  Median Verified Speedup        : {median_speedup_b}x over baseline")
    print(f"  Average Inference Cost per Run : ${avg_cost_b:.4f} USD")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VeriKernel N=30 Benchmark Study Runner")
    parser.add_argument("--trials-per-prompt", type=int, default=10, help="Trials per prompt variant (default: 10 -> N=30 total)")
    parser.add_argument("--arm", choices=["all", "A", "B"], default="all", help="Which arm to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment plan and exit without invoking LLM")
    args = parser.parse_args()

    baseline_code = BASELINE_PATH.read_text(encoding="utf-8")
    total_trials = len(PROMPT_VARIANTS) * args.trials_per_prompt

    print("======================================================================")
    print(" VeriKernel: Starting Empirical Benchmark Study")
    print(f" Total Runs Planned : {total_trials * (2 if args.arm == 'all' else 1)}")
    print(f" Model Engine       : {MODEL_ID}")
    print(f" Prompts Evaluated  : {len(PROMPT_VARIANTS)} distinct prompt formulations")
    print(f" Trials Per Prompt  : {args.trials_per_prompt}")
    print("======================================================================")

    if args.dry_run:
        print("[Dry Run] Exiting without making API calls.")
        return

    study_data: dict[str, list[dict[str, Any]]] = {"arm_a": [], "arm_b": []}
    if OUTPUT_JSON.exists():
        try:
            study_data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            print(f"[Resume] Loaded existing data: {len(study_data.get('arm_a', []))} Arm A, {len(study_data.get('arm_b', []))} Arm B")
        except Exception:
            pass

    # --- Run Arm A ---
    if args.arm in ("all", "A"):
        print("\n>>> [Phase 1/2] Executing Arm A (One-Shot Ungated)...")
        for p_idx, variant in enumerate(PROMPT_VARIANTS, 1):
            for t_idx in range(1, args.trials_per_prompt + 1):
                run_num = len(study_data["arm_a"]) + 1
                if run_num > total_trials:
                    break
                print(f"  [Arm A Run {run_num}/{total_trials}] Prompt {p_idx} | Trial {t_idx}...", end=" ", flush=True)
                res = run_arm_a_trial(variant["prompt"], baseline_code)
                res["prompt_id"] = variant["id"]
                study_data["arm_a"].append(res)
                OUTPUT_JSON.write_text(json.dumps(study_data, indent=2), encoding="utf-8")
                status = "PASS" if res["passed"] else f"FAIL ({res.get('first_failure_tag')})"
                print(f"{status} in {res['duration_s']}s (${res['cost_usd']:.4f})")

    # --- Run Arm B ---
    if args.arm in ("all", "B"):
        print("\n>>> [Phase 2/2] Executing Arm B (Referee-Gated + Tavily Loop)...")
        for p_idx, variant in enumerate(PROMPT_VARIANTS, 1):
            for t_idx in range(1, args.trials_per_prompt + 1):
                run_num = len(study_data["arm_b"]) + 1
                if run_num > total_trials:
                    break
                print(f"  [Arm B Run {run_num}/{total_trials}] Prompt {p_idx} | Trial {t_idx}...", end=" ", flush=True)
                res = run_arm_b_trial(variant["prompt"], baseline_code)
                res["prompt_id"] = variant["id"]
                study_data["arm_b"].append(res)
                OUTPUT_JSON.write_text(json.dumps(study_data, indent=2), encoding="utf-8")
                status = f"PASS (Att {res.get('attempts_needed')})" if res["passed"] else "FAIL"
                print(f"{status} in {res['duration_s']}s (${res['cost_usd']:.4f})")

    if TEMP_KERNEL.exists():
        TEMP_KERNEL.unlink()

    print_summary_table(study_data)
    print(f"[Done] Complete empirical dataset saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
