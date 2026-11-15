"""
Executor server: runs model-written Python in a persistent Jupyter kernel.

The agent sends code over HTTP; this server runs it in the kernel and sends
back what happened (printed text and errors). It is meant to run inside the
locked-down Docker container (compose.yaml), which is what sandboxes the code.

Endpoints:
    GET  /health   is the kernel alive?
    POST /execute  run code, return its outputs
    POST /reset    restart the kernel with a clean slate

Run sandboxed, from the repo root:
    docker compose up -d --build
Run locally for development only (no sandbox, never for model code):
    uv run uvicorn server:app --host 127.0.0.1 --port 8000
"""

import asyncio
import os
import queue
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from jupyter_client import AsyncKernelManager
from pydantic import BaseModel


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

# /data and /output in Docker; the project's data/ and output/ folders locally.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))

# Set to "1" only by compose.yaml, i.e. when running inside the locked-down
# container. The agent refuses to send model code to an executor that doesn't
# report this, so a server accidentally started on Windows is never used.
SANDBOXED = os.environ.get("EXECUTOR_SANDBOXED") == "1"

DEFAULT_TIMEOUT = 60        # seconds per execution
MAX_OUTPUT_CHARS = 100_000  # safety cap; the agent trims further before the model sees it

HERE = Path(__file__).resolve().parent

# Runs once in every fresh kernel, before any model code. The kernel's working
# directory is output/, so this folder is added to sys.path for the helper
# imports. pd and np are preloaded under their usual names.
KERNEL_SETUP = f"""
import sys
sys.path.insert(0, {str(HERE)!r})
import pandas as pd
import numpy as np
from analyst_helpers import profile, compare_keys, check
from cleaning import (load, current, undo, drop_duplicates, drop_rows, set_missing,
                      fill_missing, merge_variants, strip_spaces, placeholders_to_missing,
                      to_numbers, to_dates, custom_fix, show_log, save_clean, CleaningError)
"""

# Terminal color codes ("\x1b[0;31m") that kernel tracebacks contain.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


# --------------------------------------------------------------------------
# Request and response shapes
# --------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    code: str
    timeout: float = DEFAULT_TIMEOUT


class ExecuteResponse(BaseModel):
    status: str                 # "ok", "error" or "timeout"
    stdout: str = ""            # printed output, plus the value of the last line
    stderr: str = ""            # warnings and anything printed to stderr
    error: str | None = None    # a clean traceback when status is "error"
    truncated: bool = False     # True if stdout was cut at MAX_OUTPUT_CHARS
    kernel_restarted: bool = False  # True if a stuck kernel was replaced; the
                                    # agent must tell the model its state is gone
    kernel: int = 0             # which kernel ran this (see Kernel.generation)


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------

class KernelDied(Exception):
    """The kernel process ended while we were waiting for its output.
    Almost always the memory limit: Linux kills a process that exceeds it."""


class Kernel:
    """Owns one Jupyter kernel and runs code in it, one request at a time."""

    def __init__(self) -> None:
        self.manager: AsyncKernelManager | None = None
        self.client = None
        # Counts kernel starts (first start, resets, crash and timeout restarts).
        # All clients share one kernel, so a client that sees this change
        # knows another client reset it and its tables are gone.
        self.generation = 0
        # One execution at a time: concurrent requests would mix their outputs.
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # The kernel talks to this server over 127.0.0.1 with signed messages.
        # The "TCP without encryption" startup warning concerns other programs
        # on the same machine; inside the container there are none.
        self.manager = AsyncKernelManager(kernel_name="python3")
        # Relative paths in model code (e.g. "summary.csv") land in output/.
        await self.manager.start_kernel(cwd=str(OUTPUT_DIR))
        self.client = self.manager.client()
        self.client.start_channels()
        await self.client.wait_for_ready(timeout=60)
        await self._execute(KERNEL_SETUP, timeout=30)
        self.generation += 1

    async def stop(self) -> None:
        if self.client:
            self.client.stop_channels()
        if self.manager:
            await self.manager.shutdown_kernel(now=True)

    async def restart(self) -> None:
        """Fresh kernel, for /reset. Waits for any running execution first."""
        async with self.lock:
            await self._restart()

    async def _restart(self) -> None:
        await self.stop()
        await self.start()

    async def is_alive(self) -> bool:
        return bool(self.manager and await self.manager.is_alive())

    async def run(self, code: str, timeout: float) -> ExecuteResponse:
        """Run code for a request. The lock makes other requests wait their turn."""
        async with self.lock:
            response = await self._execute(code, timeout)
            response.kernel = self.generation   # after any restart _execute did
            return response

    async def _execute(self, code: str, timeout: float) -> ExecuteResponse:
        """Send code to the kernel and collect everything it produces.

        Callers must already hold the lock, or (like start()) run before any
        request can arrive; asyncio.Lock isn't re-entrant.
        """
        # Every output message for this code carries msg_id as its parent.
        msg_id = self.client.execute(code)
        try:
            return await asyncio.wait_for(self._collect(msg_id), timeout)
        except TimeoutError:
            pass
        except KernelDied:
            await self._restart()
            return ExecuteResponse(
                status="error",
                kernel_restarted=True,
                error="The kernel crashed while running this code, most likely because "
                      "it ran out of memory. It was restarted: all variables and imports "
                      "are gone and files must be loaded again. Use less memory, e.g. load "
                      "only the needed columns (usecols=...) or process data in chunks.")

        # Too slow: interrupt the code, like Jupyter's stop button.
        await self.manager.interrupt_kernel()
        if await self._drain(msg_id):
            # Stopped cleanly: the kernel and its variables are still usable.
            return ExecuteResponse(
                status="timeout",
                error=f"Execution stopped after {timeout:.0f} seconds. "
                      "Variables from earlier steps are still available.")

        # Some code (C extensions, certain sleeps) ignores interrupts. A kernel
        # in that state can't be trusted, so replace it.
        await self._restart()
        return ExecuteResponse(
            status="timeout",
            kernel_restarted=True,
            error=f"Execution stopped after {timeout:.0f} seconds. The code could not "
                  "be interrupted, so the kernel was restarted: all variables and "
                  "imports are gone and files must be loaded again.")

    async def _collect(self, msg_id: str) -> ExecuteResponse:
        """Read output messages until the kernel reports it's idle again."""
        stdout, stderr = [], []
        error = None

        while True:
            # Outputs arrive on the iopub channel. Waiting 1 second at a time
            # lets us notice a kernel killed for using too much memory, instead
            # of waiting out the full timeout for messages that never come.
            try:
                msg = await self.client.get_iopub_msg(timeout=1)
            except queue.Empty:
                if not await self.manager.is_alive():
                    raise KernelDied()
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue  # left over from an earlier request; ignore it
            kind, content = msg["msg_type"], msg["content"]

            if kind == "stream":
                (stdout if content["name"] == "stdout" else stderr).append(content["text"])

            elif kind in ("execute_result", "display_data"):
                # The last line's value, or display(): keep the plain-text form.
                data = content["data"]
                if "text/plain" in data:
                    stdout.append(data["text/plain"] + "\n")

            elif kind == "error":
                error = ANSI_ESCAPE.sub("", "\n".join(content["traceback"]))

            elif kind == "status" and content["execution_state"] == "idle":
                break  # the kernel finished this code

        text = "".join(stdout)
        truncated = len(text) > MAX_OUTPUT_CHARS
        return ExecuteResponse(
            status="error" if error else "ok",
            stdout=text[:MAX_OUTPUT_CHARS],
            stderr="".join(stderr)[:MAX_OUTPUT_CHARS],
            error=error,
            truncated=truncated,
        )

    async def _drain(self, msg_id: str) -> bool:
        """After an interrupt, read the remaining messages for that request so
        they don't leak into the next one. Returns True if the kernel reported
        idle (the code really stopped) within 10 seconds, False otherwise."""
        try:
            await asyncio.wait_for(self._collect(msg_id), 10)
            return True
        except (TimeoutError, KernelDied):
            return False


kernel = Kernel()


# --------------------------------------------------------------------------
# The web app
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await kernel.start()
    yield
    await kernel.stop()


app = FastAPI(title="Analyst executor", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok" if await kernel.is_alive() else "dead", "kernel": kernel.generation,
            "sandboxed": SANDBOXED}


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    if not await kernel.is_alive():
        # The kernel crashed (e.g. out of memory): the client should call /reset.
        raise HTTPException(status_code=503, detail="Kernel is not running. Call /reset.")
    return await kernel.run(request.code, request.timeout)


@app.post("/reset")
async def reset():
    await kernel.restart()
    return {"status": "ok", "kernel": kernel.generation}