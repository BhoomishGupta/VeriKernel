#!/usr/bin/env python3
"""VeriKernel 5-stage referee for cavity harness submissions."""

from __future__ import annotations

import argparse
import pathlib
import re
import os
import subprocess
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_HARNESS = REPO_ROOT / "harness_cavity.cpp"
DEFAULT_BINARY = REPO_ROOT / "build" / "harness_cavity"
BASELINE_HEADER = REPO_ROOT / "baseline_kernel.hpp"

BANNED_PATTERNS = [
    r"#pragma\s+GCC\s+optimize\s*\(\s*\"fast-math\"\s*\)",
    r"#pragma\s+clang\s+fp\s+contract\s*\(\s*fast\s*\)",
    r"-ffast-math",
]

ALLOWED_INCLUDES = {
    "algorithm",
    "array",
    "cmath",
    "cstddef",
    "iomanip",
    "iostream",
    "limits",
    "string",
    "vector",
    "omp.h",
    "baseline_kernel.hpp",
}


@dataclass
class StageResult:
    stage: str
    ok: bool
    detail: str


def run_command(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True, env=env)


def stage1_static_lint(harness_path: pathlib.Path) -> StageResult:
    content = harness_path.read_text(encoding="utf-8")

    for pattern in BANNED_PATTERNS:
        if re.search(pattern, content):
            return StageResult("Stage 1", False, f"Banned pattern detected: {pattern}")

    include_pattern = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE)
    includes = include_pattern.findall(content)
    unauthorized = sorted({inc for inc in includes if inc not in ALLOWED_INCLUDES})
    if unauthorized:
        return StageResult("Stage 1", False, f"Unauthorized includes: {', '.join(unauthorized)}")

    return StageResult("Stage 1", True, "Static linter checks passed")


def stage2_compile(harness_path: pathlib.Path, binary_path: pathlib.Path) -> StageResult:
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    container_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-w",
        "/workspace",
        "gcc:13",
        "g++",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-std=c++17",
        str(harness_path.relative_to(REPO_ROOT)),
        "-o",
        str(binary_path.relative_to(REPO_ROOT)),
    ]

    local_cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-std=c++17",
        str(harness_path),
        "-o",
        str(binary_path),
    ]

    docker_exists = subprocess.run(["which", "docker"], check=False, text=True, capture_output=True).returncode == 0

    if docker_exists:
        compile_result = run_command(container_cmd)
        if compile_result.returncode != 0:
            return StageResult("Stage 2", False, f"Containerized build failed:\n{compile_result.stderr}")
        return StageResult("Stage 2", True, "Containerized build passed with gcc:13")

    compile_result = run_command(local_cmd)
    if compile_result.returncode != 0:
        return StageResult("Stage 2", False, f"Local fallback build failed:\n{compile_result.stderr}")
    return StageResult("Stage 2", True, "Container runtime unavailable; local g++ fallback build passed")


def parse_metric(line: str, key: str) -> float:
    match = re.search(rf"{re.escape(key)}=([-+0-9.eE]+)", line)
    if not match:
        raise ValueError(f"Missing metric {key} in line: {line}")
    return float(match.group(1))


def run_harness(binary_path: pathlib.Path, env: dict[str, str] | None = None) -> list[str]:
    result = run_command([str(binary_path)], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Harness failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result.stdout.splitlines()


def stage3_determinism(binary_path: pathlib.Path) -> StageResult:
    try:
        lines = run_harness(binary_path)
    except RuntimeError as exc:
        return StageResult("Stage 3", False, str(exc))

    det_lines = [line for line in lines if line.startswith("DETERMINISM ")]
    if len(det_lines) != 4:
        return StageResult("Stage 3", False, "Expected 4 determinism lines (threads 1/2/4/8)")

    for line in det_lines:
        rr_psi = parse_metric(line, "RUN_TO_RUN_PSI_DELTA")
        rr_omega = parse_metric(line, "RUN_TO_RUN_OMEGA_DELTA")
        x_psi = parse_metric(line, "VS_T1_PSI_DELTA")
        x_omega = parse_metric(line, "VS_T1_OMEGA_DELTA")
        if max(rr_psi, rr_omega, x_psi, x_omega) > 1e-12:
            return StageResult("Stage 3", False, f"Determinism violation: {line}")

    return StageResult("Stage 3", True, "Determinism checks passed across 1/2/4/8 threads")


def stage4_physics(binary_path: pathlib.Path) -> StageResult:
    try:
        lines = run_harness(binary_path)
    except RuntimeError as exc:
        return StageResult("Stage 4", False, str(exc))

    summary = next((line for line in lines if line.startswith("PHYSICS_PASS=")), None)
    if summary is None:
        return StageResult("Stage 4", False, "Missing PHYSICS_PASS summary")

    if summary.strip() != "PHYSICS_PASS=1":
        return StageResult("Stage 4", False, "Physics verification failed against hardcoded references")

    return StageResult("Stage 4", True, "Physics verification passed")


def stage5_timing(binary_path: pathlib.Path) -> StageResult:
    env = dict(os.environ)
    env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
    try:
        lines = run_harness(binary_path, env=env)
    except RuntimeError as exc:
        return StageResult("Stage 5", False, str(exc))

    timing_lines = [line for line in lines if line.startswith("THREADS=")]
    if len(timing_lines) != 4:
        return StageResult("Stage 5", False, "Missing thread timing lines")

    timings = []
    for line in timing_lines:
        timings.append(parse_metric(line, "TIME"))

    if any(t <= 0.0 for t in timings):
        return StageResult("Stage 5", False, "Non-positive timing measurement encountered")

    return StageResult("Stage 5", True, "Pinned-thread timing benchmark completed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VeriKernel 5-stage referee checks")
    parser.add_argument("--harness", type=pathlib.Path, default=DEFAULT_HARNESS)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    args = parser.parse_args()

    harness_path = args.harness.resolve()
    binary_path = args.binary.resolve()

    if not harness_path.exists():
        print(f"Harness source not found: {harness_path}", file=sys.stderr)
        return 2
    if not BASELINE_HEADER.exists():
        print(f"Missing baseline kernel header: {BASELINE_HEADER}", file=sys.stderr)
        return 2

    stages = [
        stage1_static_lint(harness_path),
        stage2_compile(harness_path, binary_path),
        stage3_determinism(binary_path),
        stage4_physics(binary_path),
        stage5_timing(binary_path),
    ]

    failed = False
    for result in stages:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{result.stage}] {status}: {result.detail}")
        failed = failed or not result.ok

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
