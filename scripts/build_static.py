#!/usr/bin/env python3
"""
Static Bundle Builder for GitHub Pages & Netlify Deployment.
Materializes all symlinks into standalone static assets, copies cache payloads,
creates .nojekyll for GitHub Pages, and verifies offline bundle completeness.
"""

import json
import os
import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
CACHE_DIR = REPO_ROOT / "cache"


def build_static_bundle():
    print(f"==> Building standalone static distribution in {STATIC_DIR}...")
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Materialize root-level certified artifacts
    files_to_copy = [
        ("baseline_kernel.hpp", REPO_ROOT / "baseline_kernel.hpp"),
        ("candidate_kernel.hpp", REPO_ROOT / "candidate_kernel.hpp"),
        ("REFEREE_CERTIFICATE.md", REPO_ROOT / "REFEREE_CERTIFICATE.md"),
        ("verikernel_certified_bundle.zip", CACHE_DIR / "verikernel_certified_bundle.zip"),
    ]

    for dest_name, src_path in files_to_copy:
        dest_path = STATIC_DIR / dest_name
        if dest_path.is_symlink() or dest_path.exists():
            dest_path.unlink()
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            print(f"  [Copy] {src_path.name} -> static/{dest_name} ({dest_path.stat().st_size} bytes)")
        else:
            print(f"  [WARN] Source file not found: {src_path}")

    # 2. Materialize cache/ directory inside static/
    static_cache = STATIC_DIR / "cache"
    if static_cache.is_symlink():
        static_cache.unlink()
    static_cache.mkdir(parents=True, exist_ok=True)

    cache_files = [
        "latest_agent_run.json",
        "empirical_study_results.json",
        "empirical_study_summary.md",
        "smoke_test_results.json",
        "tavily_cache.json",
        "verikernel_certified_bundle.zip",
    ]

    for fname in cache_files:
        src = CACHE_DIR / fname
        dest = static_cache / fname
        if src.exists():
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            shutil.copy2(src, dest)
            print(f"  [Copy] cache/{fname} -> static/cache/{fname}")

    # Copy agent_runs directory
    src_agent_runs = CACHE_DIR / "agent_runs"
    dest_agent_runs = static_cache / "agent_runs"
    if dest_agent_runs.is_symlink() or dest_agent_runs.exists():
        shutil.rmtree(dest_agent_runs)
    if src_agent_runs.exists():
        shutil.copytree(src_agent_runs, dest_agent_runs)
        count = len(list(dest_agent_runs.glob("*.json")))
        print(f"  [Copy] cache/agent_runs/ ({count} runs) -> static/cache/agent_runs/")

    # 3. Materialize gallery/ directory inside static/ for defect inspection
    src_gallery = REPO_ROOT / "gallery"
    dest_gallery = STATIC_DIR / "gallery"
    if dest_gallery.is_symlink() or dest_gallery.exists():
        shutil.rmtree(dest_gallery)
    if src_gallery.exists():
        shutil.copytree(src_gallery, dest_gallery)
        g_count = len(list(dest_gallery.glob("*.hpp")))
        print(f"  [Copy] gallery/ ({g_count} defect kernels) -> static/gallery/")

    # 4. Create .nojekyll for GitHub Pages
    nojekyll = STATIC_DIR / ".nojekyll"
    nojekyll.write_text("# Disable Jekyll for GitHub Pages\n", encoding="utf-8")
    print("  [Init] static/.nojekyll created")

    # 5. Verify integrity of all required JSON files
    critical_json = [
        static_cache / "latest_agent_run.json",
        static_cache / "empirical_study_results.json",
    ]
    for cj in critical_json:
        if not cj.exists():
            print(f"  [ERROR] Critical static file missing: {cj}")
            sys.exit(1)
        try:
            with open(cj, "r", encoding="utf-8") as f:
                json.load(f)
            print(f"  [Verify] Valid JSON: {cj.name}")
        except Exception as e:
            print(f"  [ERROR] Invalid JSON in {cj}: {e}")
            sys.exit(1)

    print("\n✓ Standalone static distribution successfully verified.")
    print(f"  Static payload size: {sum(f.stat().st_size for f in STATIC_DIR.rglob('*') if f.is_file()) / 1024:.1f} KB")


if __name__ == "__main__":
    build_static_bundle()
