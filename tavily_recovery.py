#!/usr/bin/env python3
"""
VeriKernel: Load-Bearing Tavily Search on Referee Gate Failure
Dispatches targeted technical queries based on failure signatures, extracts
concrete algorithmic fixes, and caches results locally on disk.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = pathlib.Path(__file__).resolve().parent
CACHE_FILE = REPO_ROOT / "cache" / "tavily_cache.json"

SIGNATURE_QUERY_MAP: dict[str, str] = {
    "RACE_NON_DETERMINISTIC": (
        "Gauss-Seidel OpenMP data race red-black ordering C++ stencil parallelization"
    ),
    "LINT_FAIL_FASTMATH": (
        "why fast-math breaks numerical convergence Navier-Stokes CFD IEEE 754"
    ),
    "LINT_FAIL_HEADERS": (
        "C++ HPC stencil minimal headers memory layout performance"
    ),
    "COMPILE_FAIL": (
        "OpenMP C++ SIMD stencil pragma syntax error gcc compilation fix"
    ),
    "PHYSICS_FAIL_STALL": (
        "Gauss-Seidel Poisson solver convergence stalling Thom boundary conditions"
    ),
    "PHYSICS_FAIL_PARITY": (
        "Red-Black Gauss-Seidel checkerboard 2D grid index parity 5-point stencil"
    ),
    "PHYSICS_FAIL_GENERAL": (
        "Lid-driven cavity Navier-Stokes streamfunction vorticity boundary discretization"
    ),
}


def _load_cache() -> dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def get_diagnostic_query(stage: str, status_tag: str, detail: str) -> str:
    """Derive the optimal search query from the Referee's failure signature."""
    status_upper = status_tag.upper()
    detail_lower = detail.lower()

    if "RACE" in status_upper or "DETERMINISM" in status_upper:
        return SIGNATURE_QUERY_MAP["RACE_NON_DETERMINISTIC"]

    if "LINT" in status_upper:
        if "fast-math" in detail_lower:
            return SIGNATURE_QUERY_MAP["LINT_FAIL_FASTMATH"]
        return SIGNATURE_QUERY_MAP["LINT_FAIL_HEADERS"]

    if "COMPILE" in status_upper:
        return SIGNATURE_QUERY_MAP["COMPILE_FAIL"]

    if "PHYSICS" in status_upper:
        if "steps=15000" in detail_lower or "stalls" in detail_lower:
            return SIGNATURE_QUERY_MAP["PHYSICS_FAIL_STALL"]
        if "wrong_redblack" in detail_lower or "parity" in detail_lower or "0.99" in detail_lower:
            return SIGNATURE_QUERY_MAP["PHYSICS_FAIL_PARITY"]
        return SIGNATURE_QUERY_MAP["PHYSICS_FAIL_GENERAL"]

    return f"C++ OpenMP numerical stencil {stage} {status_tag}"


def retrieve_recovery_guidance(
    stage: str,
    status_tag: str,
    detail: str,
    max_results: int = 3,
) -> dict[str, Any]:
    """
    Queries Tavily for targeted remediation guidance.
    Checks disk cache first to conserve API credits.
    """
    query = get_diagnostic_query(stage, status_tag, detail)
    cache = _load_cache()

    cache_key = f"{stage}::{status_tag}::{query}"
    if cache_key in cache:
        cached_entry = cache[cache_key]
        cached_entry["cached"] = True
        return cached_entry

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        return {
            "query": query,
            "cached": False,
            "error": "TAVILY_API_KEY unconfigured",
            "snippets": [
                {
                    "title": "Red-Black Ordering for Parallel Gauss-Seidel",
                    "url": "https://en.wikipedia.org/wiki/Red-black_Gauss-Seidel",
                    "content": (
                        "Lexicographical Gauss-Seidel has loop-carried dependencies that cause data races "
                        "under OpenMP. Refactor into Red-Black (checkerboard) ordering: update (i+j)%2==0 in Pass 1, "
                        "then (i+j)%2==1 in Pass 2 with #pragma omp parallel for."
                    ),
                }
            ],
            "formatted_guidance": (
                "Algorithmic Recommendation:\n"
                "• Lexicographical Gauss-Seidel cannot be parallelized naively with #pragma omp parallel for.\n"
                "• Convert the sweep into a 2-pass Red-Black Gauss-Seidel (even parity cells then odd parity cells).\n"
                "• Ensure (i + j) % 2 is evaluated across 2D coordinates, not flattened 1D indices."
            ),
        }

    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )

        snippets = []
        for r in response.get("results", []):
            snippets.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:350],
            })

        tavily_answer = response.get("answer", "")
        formatted_guidance = ""
        if tavily_answer:
            formatted_guidance += f"Summary Analysis: {tavily_answer}\n\n"

        formatted_guidance += "Technical Citations:\n"
        for idx, s in enumerate(snippets, 1):
            formatted_guidance += f"[{idx}] {s['title']} ({s['url']})\n    {s['content']}\n"

        result = {
            "query": query,
            "cached": False,
            "answer": tavily_answer,
            "snippets": snippets,
            "formatted_guidance": formatted_guidance.strip(),
        }

        cache[cache_key] = result
        _save_cache(cache)
        return result

    except Exception as exc:
        return {
            "query": query,
            "cached": False,
            "error": str(exc),
            "snippets": [],
            "formatted_guidance": f"Tavily search failed: {exc}",
        }


if __name__ == "__main__":
    print("Testing Tavily Recovery Dispatcher...")
    report = retrieve_recovery_guidance(
        stage="Stage 3",
        status_tag="RACE_NON_DETERMINISTIC",
        detail="Determinism violation: max_thread_delta=3.87e-05 across threads",
    )
    print(f"Query Triggered: {report['query']}")
    print(f"Cached Result  : {report['cached']}")
    print("\nGuidance Text:\n" + report["formatted_guidance"])