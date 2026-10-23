#!/usr/bin/env python3
"""FastAPI Execution Server with Global Lock and Queue Tracking for VeriKernel."""

from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import sys
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = pathlib.Path(__file__).resolve().parent
REFEREE_PATH = REPO_ROOT / "referee.py"
DEFAULT_KERNEL = REPO_ROOT / "baseline_kernel.hpp"
ESTIMATED_RUN_SECONDS = 4

app = FastAPI(title="VeriKernel Execution API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency state
execution_lock = asyncio.Lock()
waiting_queue_count = 0


class ExecutionRequest(BaseModel):
    kernel_code: str | None = Field(
        default=None,
        description="Inline C++ kernel code. If omitted, runs default baseline_kernel.hpp.",
    )
    gallery_bug: str | None = Field(
        default=None,
        description="Relative path to a Break-It Gallery kernel (e.g., 'gallery/bug_1_naive_omp_race.hpp').",
    )


def run_referee_sync(kernel_path: pathlib.Path) -> dict:
    """Execute referee.py in a synchronous subprocess and return parsed JSON output."""
    cmd = [sys.executable, str(REFEREE_PATH), "--kernel", str(kernel_path), "--json"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30.0,
    )
    try:
        telemetry = json.loads(proc.stdout)
    except json.JSONDecodeError:
        telemetry = {
            "error": "Failed to parse referee output",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    return telemetry


async def execute_referee_task(req: ExecutionRequest) -> AsyncGenerator[str, None]:
    """Manages queue position notifications, lock acquisition, and referee execution."""
    global waiting_queue_count

    # Determine kernel target
    temp_kernel_path: pathlib.Path | None = None
    if req.gallery_bug:
        kernel_path = REPO_ROOT / req.gallery_bug
        if not kernel_path.exists():
            yield f"data: {json.dumps({'error': f'Gallery target not found: {req.gallery_bug}'})}\n\n"
            return
    elif req.kernel_code:
        # Write candidate code to temporary file
        temp_dir = REPO_ROOT / ".cache_run"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_kernel_path = temp_dir / f"candidate_{asyncio.get_event_loop().time():.4f}.hpp"
        temp_kernel_path.write_text(req.kernel_code, encoding="utf-8")
        kernel_path = temp_kernel_path
    else:
        kernel_path = DEFAULT_KERNEL

    # 1. Inspect queue state before acquiring lock
    if execution_lock.locked():
        waiting_queue_count += 1
        queue_ahead = waiting_queue_count
        wait_time = queue_ahead * ESTIMATED_RUN_SECONDS
        run_plural = "runs" if queue_ahead > 1 else "run"
        queue_msg = f"{queue_ahead} {run_plural} ahead of you — ~{wait_time}s wait"

        yield f"data: {json.dumps({'status': 'queued', 'queue_position': queue_ahead, 'message': queue_msg})}\n\n"

        try:
            await execution_lock.acquire()
        finally:
            waiting_queue_count = max(0, waiting_queue_count - 1)
    else:
        await execution_lock.acquire()

    # 2. Lock acquired: Execute benchmark safely without cross-thread CPU interference
    try:
        yield f"data: {json.dumps({'status': 'running', 'message': 'Executing verification pipeline...'})}\n\n"

        # Offload subprocess to threadpool to avoid stalling the asyncio event loop
        telemetry = await asyncio.to_thread(run_referee_sync, kernel_path)

        yield f"data: {json.dumps({'status': 'completed', 'data': telemetry})}\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'status': 'error', 'detail': str(exc)})}\n\n"

    finally:
        execution_lock.release()
        if temp_kernel_path and temp_kernel_path.exists():
            temp_kernel_path.unlink(missing_ok=True)


@app.post("/api/run-referee-stream")
async def run_referee_stream_endpoint(req: ExecutionRequest):
    """Server-Sent Events endpoint streaming queue position and final referee results."""
    return StreamingResponse(
        execute_referee_task(req),
        media_type="text/event-stream",
    )


@app.post("/api/run-referee")
async def run_referee_json_endpoint(req: ExecutionRequest):
    """Standard serialized JSON endpoint waiting for execution lock."""
    global waiting_queue_count

    if execution_lock.locked():
        waiting_queue_count += 1
        try:
            await execution_lock.acquire()
        finally:
            waiting_queue_count = max(0, waiting_queue_count - 1)
    else:
        await execution_lock.acquire()

    try:
        kernel_path = REPO_ROOT / req.gallery_bug if req.gallery_bug else DEFAULT_KERNEL
        result = await asyncio.to_thread(run_referee_sync, kernel_path)
        return {"status": "success", "results": result}
    finally:
        execution_lock.release()


@app.get("/api/queue-status")
async def get_queue_status():
    """Lightweight polling endpoint to inspect current VM execution state."""
    is_busy = execution_lock.locked()
    return {
        "busy": is_busy,
        "waiting": waiting_queue_count,
        "estimated_wait_seconds": waiting_queue_count * ESTIMATED_RUN_SECONDS,
    }