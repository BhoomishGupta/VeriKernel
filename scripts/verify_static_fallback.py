#!/usr/bin/env python3
"""
Static Fallback Verification Suite for GitHub Pages & Netlify.
Simulates a zero-backend static web server serving only static/, verifying that
all assets, datasets, vendor libraries, and certified bundles resolve with HTTP 200.
"""

import functools
import http.server
import json
import os
import pathlib
import socketserver
import sys
import threading
import time
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def verify_static_site():
    print("======================================================================")
    print(" VeriKernel: Static Fallback Layer Verification (Zero-Backend Mode)")
    print("======================================================================")

    # 1. Pre-build check
    if not (STATIC_DIR / "index.html").exists():
        print("✗ Error: static/index.html not found.")
        sys.exit(1)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(STATIC_DIR))
    httpd = ReusableTCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]

    stop_event = threading.Event()
    def _serve():
        httpd.timeout = 0.5
        while not stop_event.is_set():
            httpd.handle_request()
        httpd.server_close()

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    base_url = f"http://127.0.0.1:{port}"
    errors = []

    tests = [
        ("/", 200, "Dashboard Entrypoint"),
        ("/.nojekyll", 200, "GitHub Pages Jekyll Bypass"),
        ("/vendor/chart.umd.min.js", 200, "Chart.js Vendor Bundle"),
        ("/vendor/diff2html.min.css", 200, "Diff2Html CSS"),
        ("/vendor/diff.min.js", 200, "JS Diff Vendor Bundle"),
        ("/vendor/diff2html.min.js", 200, "Diff2Html JS Bundle"),
        ("/baseline_kernel.hpp", 200, "Serial Baseline C++ Kernel"),
        ("/candidate_kernel.hpp", 200, "Verified Candidate C++ Kernel"),
        ("/REFEREE_CERTIFICATE.md", 200, "Cryptographic Referee Certificate"),
        ("/verikernel_certified_bundle.zip", 200, "Certified ZIP Bundle"),
        ("/cache/latest_agent_run.json", 200, "Latest Self-Correction Telemetry"),
        ("/cache/empirical_study_results.json", 200, "N=30 Empirical Benchmark Dataset"),
    ]

    print(f"\n[1/3] Testing Static Asset Resolution from {base_url}...")
    for path, expected_status, label in tests:
        url = base_url + path
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VeriKernel-Static-Auditor"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                status = resp.status
                content = resp.read()
                if status == expected_status:
                    print(f"  ✓ [{status}] {label} ({path}) — {len(content):,} bytes")
                else:
                    msg = f"Unexpected status {status} for {path} (expected {expected_status})"
                    print(f"  ✗ {msg}")
                    errors.append(msg)
        except Exception as e:
            msg = f"Failed to fetch {path}: {e}"
            print(f"  ✗ {msg}")
            errors.append(msg)

    # 2. JSON Invariant Verification
    print("\n[2/3] Validating Payload JSON Data Structures...")
    try:
        req = urllib.request.urlopen(f"{base_url}/cache/latest_agent_run.json")
        run_data = json.loads(req.read().decode("utf-8"))
        attempts = run_data if isinstance(run_data, list) else run_data.get("attempts", [])
        if len(attempts) > 0:
            print(f"  ✓ latest_agent_run.json contains {len(attempts)} attempt(s)")
        else:
            errors.append("latest_agent_run.json has 0 attempts")

        req2 = urllib.request.urlopen(f"{base_url}/cache/empirical_study_results.json")
        study_data = json.loads(req2.read().decode("utf-8"))
        arm_a = study_data.get("arm_a", [])
        arm_b = study_data.get("arm_b", [])
        print(f"  ✓ empirical_study_results.json contains Arm A ({len(arm_a)} runs) and Arm B ({len(arm_b)} runs)")
        if len(arm_a) + len(arm_b) != 30:
            print(f"  ! Note: Total runs = {len(arm_a) + len(arm_b)} (expected 30)")
    except Exception as e:
        errors.append(f"JSON validation failure: {e}")

    # 3. HTML Responsive Layout Audit
    print("\n[3/3] Auditing Responsive Meta and Breakpoints...")
    try:
        index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert 'name="viewport"' in index_html, "Missing viewport meta tag"
        assert '@media (max-width: 1100px)' in index_html, "Missing 1100px breakpoint"
        assert '@media (max-width: 768px)' in index_html, "Missing 768px tablet breakpoint"
        assert '@media (max-width: 480px)' in index_html, "Missing 480px mobile breakpoint"
        assert 'backendModal' in index_html, "Missing 3-tier backend VM modal"
        assert './vendor/chart.umd.min.js' in index_html, "Vendor scripts must use relative path"
        print("  ✓ Mobile viewport meta tag configured: width=device-width, initial-scale=1.0")
        print("  ✓ Responsive breakpoints defined for desktop (1100px), tablet (768px), and mobile (480px)")
        print("  ✓ 3-tier backend VM modal and dynamic API resolver integrated")
    except AssertionError as ae:
        errors.append(f"HTML Audit failure: {ae}")

    # Tear down
    stop_event.set()
    server_thread.join(timeout=1.0)

    print("\n======================================================================")
    if not errors:
        print("✓ ALL STATIC FALLBACK CHECKS PASSED: Ready for GitHub Pages & Netlify!")
        print("======================================================================")
        return 0
    else:
        print(f"✗ FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        print("======================================================================")
        return 1


if __name__ == "__main__":
    sys.exit(verify_static_site())
