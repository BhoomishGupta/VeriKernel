#!/usr/bin/env python3
"""
VeriKernel: Physics-Aware AI Code Optimization Referee
Dual-Workload Verification Pipeline:
  - cavity : 2D Incompressible Navier-Stokes (Ghia RMSE + L2 Reference)
  - wave   : 2D Acoustic Wave Equation (Observed Convergence Order p >= 1.90)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import asdict, dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent

# Workload 1: Cavity Defaults
DEFAULT_HARNESS_CAVITY = REPO_ROOT / "harness_cavity.cpp"
DEFAULT_KERNEL_CAVITY = REPO_ROOT / "baseline_kernel.hpp"
DEFAULT_BINARY_CAVITY = REPO_ROOT / "build" / "harness_cavity"
REFERENCE_BINARY = REPO_ROOT / "reference_cavity_129.bin"

# Workload 2: Wave Defaults
DEFAULT_HARNESS_WAVE = REPO_ROOT / "harness_wave.cpp"
DEFAULT_KERNEL_WAVE = REPO_ROOT / "wave_baseline.hpp"
DEFAULT_BINARY_WAVE = REPO_ROOT / "build" / "harness_wave"

BANNED_PATTERNS = [
    r"#pragma\s+GCC\s+optimize\s*\(\s*\"fast-math\"\s*\)",
    r"#pragma\s+clang\s+fp\s+contract\s*\(\s*fast\s*\)",
    r"#pragma\s+(?:GCC\s+)?unroll\b",
    r"#pragma\s+clang\s+loop\s+unroll",
    r"-ffast-math",
    r"\bfork\s*\(",
    r"\bsystem\s*\(",
    r"\b__asm__\b",
]

# Strict kernel allow-list per VeriKernel HPC contract
ALLOWED_KERNEL_INCLUDES = {
    "algorithm",
    "cmath",
    "cstddef",
    "cstdint",
    "immintrin.h",
    "omp.h",
}


@dataclass
class StageResult:
    stage: str
    ok: bool
    status_tag: str
    detail: str


def run_command(
    command: list[str],
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=False, text=True, capture_output=True, env=env, timeout=timeout
    )


def stage1_static_lint(kernel_path: pathlib.Path) -> StageResult:
    content = kernel_path.read_text(encoding="utf-8")

    for pattern in BANNED_PATTERNS:
        if re.search(pattern, content):
            return StageResult(
                "Stage 1",
                False,
                "LINT_FAIL",
                f"Forbidden pattern/pragma detected: {pattern}",
            )

    include_pattern = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE)
    includes = include_pattern.findall(content)
    unauthorized = sorted({inc for inc in includes if inc not in ALLOWED_KERNEL_INCLUDES})
    if unauthorized:
        return StageResult(
            "Stage 1",
            False,
            "LINT_FAIL",
            f"Unauthorized headers in kernel: {', '.join(unauthorized)}",
        )

    return StageResult("Stage 1", True, "LINT_PASS", "Static linter checks passed")


def check_docker_available() -> bool:
    """Check if docker daemon is running AND gcc:13 image is cached."""
    daemon_ok = (
        subprocess.run(
            ["docker", "info"], check=False, text=True, capture_output=True
        ).returncode
        == 0
    )
    if not daemon_ok:
        return False
    image_ok = (
        subprocess.run(
            ["docker", "image", "inspect", "gcc:13"],
            check=False,
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )
    return image_ok


def stage2_compile(
    harness_path: pathlib.Path,
    kernel_path: pathlib.Path,
    binary_path: pathlib.Path,
    force_local: bool = False,
) -> StageResult:
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    base_flags = [
        "-O3",
        "-march=native",
        "-fopenmp",
        "-Wall",
        "-std=c++17",
    ]

    # Map path relative to REPO_ROOT for container mount
    try:
        kernel_rel = kernel_path.resolve().relative_to(REPO_ROOT.resolve())
        container_kernel_arg = f'-DKERNEL_HEADER="/workspace/{kernel_rel}"'
    except ValueError:
        container_kernel_arg = None  # Outside repo root, force local

    local_kernel_arg = f'-DKERNEL_HEADER="{kernel_path.resolve()}"'

    # 1. Containerized build (if available, not forced local, and path is within workspace)
    if not force_local and check_docker_available() and (container_kernel_arg is not None):
        user_flags = (
            ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
        )
        container_cmd = [
            "docker",
            "run",
            "--rm",
            *user_flags,
            "-v",
            f"{REPO_ROOT}:/workspace",
            "-w",
            "/workspace",
            "gcc:13",
            "g++",
            *base_flags,
            container_kernel_arg,
            str(harness_path.relative_to(REPO_ROOT)),
            "-o",
            str(binary_path.relative_to(REPO_ROOT)),
        ]
        try:
            compile_result = run_command(container_cmd, timeout=25.0)
            if compile_result.returncode == 0:
                return StageResult("Stage 2", True, "COMPILE_PASS", "Containerized build passed with gcc:13")
            return StageResult("Stage 2", False, "COMPILE_FAIL", f"Containerized build failed:\n{compile_result.stderr}")
        except subprocess.TimeoutExpired:
            return StageResult("Stage 2", False, "COMPILE_TIMEOUT", "Container compilation timed out")

    # 2. Local g++ build fallback
    local_cmd = [
        "g++",
        *base_flags,
        local_kernel_arg,
        str(harness_path),
        "-o",
        str(binary_path),
    ]
    try:
        compile_result = run_command(local_cmd, timeout=20.0)
        if compile_result.returncode != 0:
            return StageResult("Stage 2", False, "COMPILE_FAIL", f"Local g++ build failed:\n{compile_result.stderr}")
        return StageResult("Stage 2", True, "COMPILE_PASS", "Local g++ build passed")
    except subprocess.TimeoutExpired:
        return StageResult("Stage 2", False, "COMPILE_TIMEOUT", "Local compilation timed out")


def run_harness_json(binary_path: pathlib.Path, mode: str = "--all") -> dict:
    env = dict(os.environ)
    env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})

    result = run_command([str(binary_path), mode], env=env, timeout=60.0)
    if result.returncode != 0 and mode not in ("--all", "--bench"):
        raise RuntimeError(
            f"Binary exited with code {result.returncode}\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

    match = re.search(
        r"__VERIKERNEL_JSON_START__\s*(\{.*?\})\s*__VERIKERNEL_JSON_END__",
        result.stdout,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            f"Failed to find __VERIKERNEL_JSON_START__ delimiter.\nOutput:\n{result.stdout}\nError:\n{result.stderr}"
        )

    # Sanitize C++ non-standard tokens (nan, -nan, inf, -inf) into null so json.loads succeeds
    raw_json = match.group(1)
    clean_json = re.sub(r':\s*(-?nan|-?inf)\b', ': null', raw_json, flags=re.IGNORECASE)
    return json.loads(clean_json)


def stage3_determinism(telemetry: dict, workload: str = "cavity") -> StageResult:
    if workload == "wave":
        drift = telemetry.get("max_thread_drift", 1.0)
        if drift is None or drift > 1e-11:
            drift_str = f"{float(drift):.2e}" if drift is not None else "NaN"
            return StageResult(
                "Stage 3",
                False,
                "RACE_NON_DETERMINISTIC",
                f"Determinism violation: max_thread_drift={drift_str} (req <= 1e-11)",
            )
        return StageResult(
            "Stage 3",
            True,
            "DETERMINISM_PASS",
            f"Passed with zero drift across threads (max_delta={drift:.2e})",
        )

    # Cavity Workload
    det = telemetry.get("stage3_determinism", {})
    passed = det.get("pass", False)
    max_thread = det.get("max_thread_delta", 1.0)
    max_repeat = det.get("max_repeat_delta", 1.0)

    if not passed or max(max_thread, max_repeat) > 1e-11:
        return StageResult(
            "Stage 3",
            False,
            "RACE_NON_DETERMINISTIC",
            f"Determinism violation: max_thread_delta={max_thread:.2e}, max_repeat_delta={max_repeat:.2e}",
        )
    return StageResult(
        "Stage 3",
        True,
        "DETERMINISM_PASS",
        f"Passed with zero drift across threads (max_delta={max(max_thread, max_repeat):.2e})",
    )


def stage4_physics(telemetry: dict, workload: str = "cavity") -> StageResult:
    if workload == "wave":
        p = telemetry.get("observed_order_p")
        finest_l2 = telemetry.get("finest_grid_l2")
        p_str = f"{float(p):.4f}" if p is not None else "NaN"
        l2_str = f"{float(finest_l2):.2e}" if finest_l2 is not None else "NaN"

        if p is None or p < 1.90:
            return StageResult(
                "Stage 4",
                False,
                "PHYSICS_FAIL",
                f"Failed: observed order p={p_str} (req >= 1.90), finest L2={l2_str}. Truncation error detected.",
            )
        return StageResult(
            "Stage 4",
            True,
            "PHYSICS_PASS",
            f"Passed: observed order p={p:.4f} (req >= 1.90), finest L2={l2_str}",
        )

    # Cavity Workload
    phy = telemetry.get("stage4_physics", {})
    passed = phy.get("pass", False)
    rel_l2 = phy.get("relative_l2")
    ghia_u = phy.get("ghia_u_rmse")
    converged_steps = phy.get("converged_steps", 0)

    rel_l2_str = f"{float(rel_l2):.2e}" if rel_l2 is not None else "NaN/Inf"
    ghia_u_str = f"{float(ghia_u):.4f}" if ghia_u is not None else "NaN/Inf"

    if not passed:
        return StageResult(
            "Stage 4",
            False,
            "PHYSICS_FAIL",
            f"Failed: relative_l2={rel_l2_str} (req <= 1e-5), Ghia U RMSE={ghia_u_str} (req <= 0.05), steps={converged_steps}",
        )
    return StageResult(
        "Stage 4",
        True,
        "PHYSICS_PASS",
        f"Passed: relative_l2={rel_l2_str}, Ghia U RMSE={ghia_u_str} in {converged_steps} steps",
    )


def stage5_timing(telemetry: dict, workload: str = "cavity") -> StageResult:
    if workload == "wave":
        throughput = telemetry.get("throughput_gb_s", 0.0)
        median_ms = telemetry.get("median_ms", 0.0)

        if median_ms <= 0.0 or throughput <= 0.0:
            return StageResult(
                "Stage 5",
                False,
                "BENCH_FAIL",
                f"Invalid benchmark measurements: median_ms={median_ms}, throughput={throughput} GB/s",
            )
        return StageResult(
            "Stage 5",
            True,
            "BENCH_PASS",
            f"Throughput: {throughput:.2f} GB/s (Median latency: {median_ms:.2f} ms)",
        )

    # Cavity Workload
    bench = telemetry.get("stage5_benchmark", {})
    median_time = bench.get("median_time_s", 0.0)
    throughput = bench.get("throughput_gb_s", 0.0)

    if median_time <= 0.0 or throughput <= 0.0:
        return StageResult(
            "Stage 5",
            False,
            "BENCH_FAIL",
            f"Invalid benchmark measurements: time={median_time}s, throughput={throughput} GB/s",
        )
    return StageResult(
        "Stage 5",
        True,
        "BENCH_PASS",
        f"Throughput: {throughput:.2f} GB/s (Median latency: {median_time*1000:.2f} ms)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="VeriKernel 5-Stage Verification Referee")
    parser.add_argument("--kernel", type=pathlib.Path, default=None, help="Path to candidate C++ header")
    parser.add_argument("--workload", type=str, choices=["cavity", "wave"], default=None, help="Evaluation workload")
    parser.add_argument("--harness", type=pathlib.Path, default=None, help="Path to test harness source")
    parser.add_argument("--binary", type=pathlib.Path, default=None, help="Output compiled binary path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON summary")
    parser.add_argument("--local", action="store_true", help="Force local g++ compilation without Docker")
    args = parser.parse_args()

    # Automatic Workload Resolution
    workload = args.workload
    if workload is None:
        if args.kernel and "wave" in str(args.kernel).lower():
            workload = "wave"
        elif args.harness and "wave" in str(args.harness).lower():
            workload = "wave"
        else:
            workload = "cavity"

    # Default paths according to resolved workload
    if workload == "wave":
        kernel_path = (args.kernel or DEFAULT_KERNEL_WAVE).resolve()
        harness_path = (args.harness or DEFAULT_HARNESS_WAVE).resolve()
        binary_path = (args.binary or DEFAULT_BINARY_WAVE).resolve()
    else:
        kernel_path = (args.kernel or DEFAULT_KERNEL_CAVITY).resolve()
        harness_path = (args.harness or DEFAULT_HARNESS_CAVITY).resolve()
        binary_path = (args.binary or DEFAULT_BINARY_CAVITY).resolve()

    if not kernel_path.exists():
        print(f"Kernel file not found: {kernel_path}", file=sys.stderr)
        return 2
    if not harness_path.exists():
        print(f"Harness file not found: {harness_path}", file=sys.stderr)
        return 2

    # Auto-generate baseline reference if missing (Cavity only)
    if workload == "cavity" and not REFERENCE_BINARY.exists():
        print(f"Generating missing reference binary at {REFERENCE_BINARY}...", file=sys.stderr)
        stage2_compile(harness_path, DEFAULT_KERNEL_CAVITY, binary_path, force_local=args.local)
        run_command([str(binary_path), "--generate-ref"], timeout=120.0)

    stages: list[StageResult] = []

    # Stage 1: Static Linter
    stages.append(stage1_static_lint(kernel_path))
    if not stages[-1].ok:
        return finalize(stages, args.json)

    # Stage 2: Compilation
    stages.append(stage2_compile(harness_path, kernel_path, binary_path, force_local=args.local))
    if not stages[-1].ok:
        return finalize(stages, args.json)

    # Stages 3, 4, 5 Execution
    mode = "--bench" if workload == "wave" else "--all"
    try:
        telemetry = run_harness_json(binary_path, mode=mode)
    except Exception as exc:
        stages.append(StageResult("Execution", False, "RUNTIME_ERROR", str(exc)))
        return finalize(stages, args.json)

    stages.append(stage3_determinism(telemetry, workload=workload))
    stages.append(stage4_physics(telemetry, workload=workload))
    stages.append(stage5_timing(telemetry, workload=workload))

    return finalize(stages, args.json)


def finalize(stages: list[StageResult], print_json: bool) -> int:
    failed = any(not r.ok for r in stages)
    if print_json:
        print(json.dumps([asdict(r) for r in stages], indent=2))
    else:
        for r in stages:
            print(f"[{r.stage}] {r.status_tag}: {r.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
