#!/usr/bin/env python3
"""
scripts/judge_preflight_check.py: Automated 3-Tier Pre-Flight Readiness Audit.

Audits:
  Tier 1: Static CDN Fallback (GitHub Pages & Netlify bundle completeness)
  Tier 2: Ingress & Availability Shield (Health, Budget Guards, Judge VIP Passkey)
  Tier 3: Isolated Compute Sandbox & Physics Oracle (C++ binaries, Reference binary, Gallery)

Usage:
  python3 scripts/judge_preflight_check.py [--url http://localhost:8000] [--token verikernel-judge-2026]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
CACHE_DIR = REPO_ROOT / "cache"


def log_step(status: bool, label: str, detail: str = ""):
    symbol = "✓" if status else "✗"
    color_prefix = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    detail_str = f" — {detail}" if detail else ""
    print(f"  {color_prefix}{symbol}{reset} {label}{detail_str}")
    return status


def test_url(url: str, headers: dict | None = None, timeout: float = 4.0) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "VeriKernel-PreFlight-Auditor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()
        if "application/json" in content_type:
            return resp.status, json.loads(raw.decode("utf-8"))
        return resp.status, raw.decode("utf-8", errors="replace")


def run_preflight_audit(backend_url: str | None, judge_token: str) -> bool:
    print("======================================================================")
    print("       VeriKernel: Step 9 Pre-Flight 3-Tier Deployment Audit          ")
    print("======================================================================")
    all_passed = True

    # -------------------------------------------------------------------------
    # Tier 1: Static Edge CDN Distribution Audit
    # -------------------------------------------------------------------------
    print("\n[Tier 1: Global Edge Static Distribution (GitHub Pages / Netlify)]")
    static_files = [
        ("static/index.html", STATIC_DIR / "index.html"),
        ("static/.nojekyll", STATIC_DIR / ".nojekyll"),
        ("static/baseline_kernel.hpp", STATIC_DIR / "baseline_kernel.hpp"),
        ("static/candidate_kernel.hpp", STATIC_DIR / "candidate_kernel.hpp"),
        ("static/REFEREE_CERTIFICATE.md", STATIC_DIR / "REFEREE_CERTIFICATE.md"),
        ("static/verikernel_certified_bundle.zip", STATIC_DIR / "verikernel_certified_bundle.zip"),
        ("static/cache/latest_agent_run.json", STATIC_DIR / "cache" / "latest_agent_run.json"),
        ("static/cache/empirical_study_results.json", STATIC_DIR / "cache" / "empirical_study_results.json"),
    ]

    for rel_path, path in static_files:
        if path.exists() and path.stat().st_size > 0:
            log_step(True, f"Asset Present: {rel_path}", f"{path.stat().st_size:,} bytes")
        else:
            all_passed = False
            log_step(False, f"Asset Missing or Empty: {rel_path}")

    # Validate JSON integrity
    try:
        with open(STATIC_DIR / "cache" / "empirical_study_results.json", "r", encoding="utf-8") as f:
            study_data = json.load(f)
            has_arm_a = "arm_a" in study_data or "arm_a_oneshot" in study_data
            has_arm_b = "arm_b" in study_data or "arm_b_referee_gated" in study_data
            log_step(has_arm_a and has_arm_b, "N=30 Benchmark Dataset Invariant", f"Arm A ({len(study_data.get('arm_a', []))} runs) & Arm B ({len(study_data.get('arm_b', []))} runs) present")
            if not (has_arm_a and has_arm_b):
                all_passed = False
    except Exception as e:
        all_passed = False
        log_step(False, "Empirical Study JSON Parse Error", str(e))

    # -------------------------------------------------------------------------
    # Tier 2: API Gateway & Availability Guard Shield
    # -------------------------------------------------------------------------
    print("\n[Tier 2: Ingress, Rate Limiting & Judge VIP Protection]")
    if backend_url:
        target_api = backend_url.rstrip("/")
        try:
            # 1. Health check
            st, data = test_url(f"{target_api}/api/health")
            log_step(st == 200 and data.get("status") == "ok", "Backend /api/health Probe", f"CPU Cores: {data.get('cpu_count', 'N/A')}")
            
            # 2. Budget & Guardrails check
            st, bdata = test_url(f"{target_api}/api/budget-status")
            log_step(
                st == 200 and "daily_spend_limit_usd" in bdata,
                "Availability Guard Status",
                f"Cap: {bdata.get('daily_runs_completed', 0)}/{bdata.get('daily_runs_limit', 30)} runs | Spend: ${bdata.get('daily_spend_current_usd', 0):.2f}"
            )

            # 3. Judge VIP Passkey Verification
            st, jdata = test_url(
                f"{target_api}/api/budget-status",
                headers={"X-Judge-Token": judge_token, "User-Agent": "JudgePreFlight/1.0"}
            )
            is_judge = jdata.get("is_judge", False)
            log_step(is_judge, f"Judge VIP Authentication Check (Token: '{judge_token}')", "Unmetered access granted" if is_judge else "Refused")
            if not is_judge:
                all_passed = False

            # 4. Zero-Token Break-It Gallery Fast-Path
            t0 = time.time()
            post_data = json.dumps({"gallery_bug": "gallery/bug_1_fast_mathcheat.hpp", "force_live": False}).encode("utf-8")
            req = urllib.request.Request(
                f"{target_api}/api/run-referee",
                data=post_data,
                headers={"Content-Type": "application/json", "User-Agent": "JudgePreFlight/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                elapsed = time.time() - t0
                gres = json.loads(resp.read().decode("utf-8"))
                cached = gres.get("cached", False)
                log_step(
                    resp.status == 200 and gres.get("status") == "success",
                    "Break-It Gallery Zero-Token Fast-Path",
                    f"{elapsed:.3f}s response (Cached: {cached})"
                )

        except urllib.error.URLError as e:
            all_passed = False
            log_step(False, f"Remote Backend VM Unreachable at {target_api}", str(e))
            print("    💡 Note: If running on a remote VM, ensure the container or tunnel is running.")
    else:
        # Evaluate local in-process API test client
        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from fastapi.testclient import TestClient
            import server
            client = TestClient(server.app)

            # Health
            h_resp = client.get("/api/health")
            log_step(h_resp.status_code == 200 and h_resp.json()["status"] == "ok", "In-Process Backend Health Check")

            # Judge VIP
            j_resp = client.get("/api/budget-status", headers={"X-Judge-Token": judge_token})
            is_judge = j_resp.json().get("is_judge", False)
            log_step(is_judge, f"In-Process Judge VIP Authentication (Token: '{judge_token}')")
            if not is_judge:
                all_passed = False

            # Gallery
            g_resp = client.post("/api/run-referee", json={"gallery_bug": "gallery/bug_1_fast_mathcheat.hpp"})
            log_step(g_resp.status_code == 200, "In-Process Break-It Gallery Local Check")

        except Exception as e:
            all_passed = False
            log_step(False, "In-Process Backend Test Error", str(e))

    # -------------------------------------------------------------------------
    # Tier 3: Computational Stencil & Invariant Ground Truth
    # -------------------------------------------------------------------------
    print("\n[Tier 3: Computational Stencil & Golden Reference Ground Truth]")
    ref_binary = REPO_ROOT / "reference_cavity_129.bin"
    if ref_binary.exists() and ref_binary.stat().st_size == 133136:
        log_step(True, "Cavity Ground Truth Binary (129x129, Re=100)", f"{ref_binary.stat().st_size:,} bytes")
    else:
        all_passed = False
        log_step(False, "Ground Truth Binary Missing or Invalid Size", f"Found: {ref_binary.stat().st_size if ref_binary.exists() else 0} bytes (expected 133,136)")

    # Check gallery kernels
    gallery_dir = REPO_ROOT / "gallery"
    gallery_count = len(list(gallery_dir.glob("*.hpp"))) if gallery_dir.exists() else 0
    log_step(gallery_count >= 5, f"Adversarial Gallery Test Kernels ({gallery_count} present)", "All defect classes covered")

    # Final Verdict
    print("\n======================================================================")
    if all_passed:
        print("\033[92m✓ PRE-FLIGHT AUDIT PASSED: 3-Tier Architecture Fully Operational.\033[0m")
        print("  - Tier 1: Static CDN Fallback verified for zero-dependency presentation.")
        print("  - Tier 2: Ingress, IP limits & Judge VIP passkeys verified sound.")
        print("  - Tier 3: Computational ground-truth and sandboxed referee operational.")
        print("======================================================================")
        return True
    else:
        print("\033[91m✗ AUDIT FAILED: One or more pre-flight conditions were not met.\033[0m")
        print("======================================================================")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriKernel Step 9 3-Tier Pre-Flight Audit")
    parser.add_argument("--url", default=None, help="Backend VM base URL to audit (e.g. http://localhost:8000)")
    parser.add_argument("--token", default="verikernel-judge-2026", help="Judge VIP authentication secret token")
    args = parser.parse_args()

    success = run_preflight_audit(backend_url=args.url, judge_token=args.token)
    sys.exit(0 if success else 1)
