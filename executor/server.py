"""
Executor server: runs model-written Python in a persistent Jupyter kernel.

The agent sends code over HTTP; this server runs it in the kernel and sends
back what happened (printed text and errors). In stage 3 this whole file
runs inside the Docker container, so the code it executes is sandboxed.

Endpoints:
    GET  /health   is the kernel alive?
    POST /execute  run code, return its outputs
    POST /reset    restart the kernel with a clean slate

Run locally for development (NOT for model code; it has no sandbox):
    uv run uvicorn server:app --host 127.0.0.1 --port 8000
Run sandboxed (stage 3), from the repo root:
    docker compose up -d --build
"""

import asyncio      # Python's built-in system for running async code. FastAPI is async,
                    # and we use asyncio for timeouts and to allow one execution at a time.
import os           # Reads environment variables, so settings can change without code edits.
import queue        # Only for queue.Empty: the exception raised when no message arrives in time.
import re           # Regular expressions; used to strip terminal color codes from tracebacks.
from contextlib import asynccontextmanager  # Turns a function with `yield` into the
                                            # setup/teardown pair FastAPI's lifespan expects.
from pathlib import Path  # Object-oriented file paths that work on Windows and Linux alike.

from fastapi import FastAPI, HTTPException  # The web framework, and its error response.
from jupyter_client import AsyncKernelManager  # Starts, stops and restarts kernels.
from pydantic import BaseModel  # Declares the shape of request/response JSON.
                                # FastAPI uses it to validate input automatically.


# --------------------------------------------------------------------------
# Settings (environment variables, with defaults for running locally)
# --------------------------------------------------------------------------

# In Docker these become /data and /output (stage 3). Locally they default to
# the project's own data/ and output/ folders, one level above executor/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))

# Set to "1" only by compose.yaml, i.e. when running inside the locked-down
# container. The agent refuses to send model code to an executor that doesn't
# report this, so a server accidentally started on Windows is never used.
SANDBOXED = os.environ.get("EXECUTOR_SANDBOXED") == "1"

DEFAULT_TIMEOUT = 60        # seconds per execution, from the plan
MAX_OUTPUT_CHARS = 100_000  # safety cap on returned text. The agent trims further
                            # (to ~5,000) before sending anything to the model.

# The folder holding this file and analyst_helpers.py.
HERE = Path(__file__).resolve().parent

# Runs once in every fresh kernel, before any model code.
#   - The kernel runs in output/, so Python wouldn't find analyst_helpers.py
#     next to this file. sys.path is the list of folders Python searches when
#     importing; adding ours makes `from analyst_helpers import ...` work.
#     {str(HERE)!r} puts the path in as a quoted Python string.
#   - pd and np are imported up front, under the names everyone uses, so the
#     model's first block can't fail with "name 'pd' is not defined".
#   - The finding helpers (analyst_helpers.py) and the fix helpers with their
#     change log (cleaning.py) are imported by name, ready to call.
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

# Terminal color codes look like "\x1b[0;31m". Tracebacks from the kernel
# contain them (for colored output in a terminal); they're noise in our results.
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
        # Counts kernel starts: 1 for the first, +1 for every reset, crash or
        # stuck-code restart. Every program using this sandbox (the UI, the
        # CLI, the eval, the tests) shares ONE kernel, so a client compares
        # this number with the one it got from its own reset: if it changed,
        # someone else reset the kernel and its tables are gone.
        self.generation = 0
        # A lock lets only one execution run at a time. Two requests sharing a
        # kernel at once would mix up their outputs.
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # "python3" is the kernel ipykernel registers, using this same Python.
        # The server and kernel talk over ports on 127.0.0.1 (this machine
        # only), and every message is signed with a random key. The startup
        # warning about "TCP without encryption" is about other programs on the
        # same machine; inside the container (stage 3) there are none.
        self.manager = AsyncKernelManager(kernel_name="python3")
        # cwd = the kernel's working directory, so relative paths like
        # "summary.csv" are saved into output/.
        await self.manager.start_kernel(cwd=str(OUTPUT_DIR))
        self.client = self.manager.client()
        self.client.start_channels()  # opens the connections to the kernel
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

        Callers must already hold the lock (or, like start(), run before any
        request can arrive). asyncio.Lock can't be acquired twice by the same
        code path, so start() calls this directly instead of run().
        """
        # execute() sends the code and returns a message ID. Every output
        # message the kernel sends back for this code carries that ID as
        # its "parent", which is how we know which outputs belong to it.
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

        # Too slow: ask the kernel to stop the code (Jupyter's stop button).
        await self.manager.interrupt_kernel()
        if await self._drain(msg_id):
            # It stopped and reported idle, so the kernel is usable and every
            # variable created before this run still exists.
            return ExecuteResponse(
                status="timeout",
                error=f"Execution stopped after {timeout:.0f} seconds. "
                      "Variables from earlier steps are still available.")

        # It did not stop. How a kernel is interrupted differs by operating
        # system, and some code (C extensions, certain sleeps) ignores it.
        # A kernel in that state can't be trusted, so replace it.
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
            # The "iopub" channel is where the kernel broadcasts all outputs.
            # Wait at most 1 second per message, so that while code runs quietly
            # we regularly check the kernel is still alive. Without this, a
            # kernel killed for using too much memory would leave us waiting
            # for messages that never come, until the full timeout.
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
                # Anything printed. "name" says whether it went to stdout or stderr.
                (stdout if content["name"] == "stdout" else stderr).append(content["text"])

            elif kind in ("execute_result", "display_data"):
                # execute_result = the value of the last line (like df.head()).
                # display_data = something shown explicitly with display().
                # Both can carry several formats of the same thing (plain text,
                # HTML...); the model reads text, so we keep the plain version.
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
    # Everything before `yield` runs when the server starts...
    await kernel.start()
    yield
    # ...and everything after it runs when the server shuts down.
    await kernel.stop()


app = FastAPI(title="Analyst executor", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok" if await kernel.is_alive() else "dead", "kernel": kernel.generation,
            "sandboxed": SANDBOXED}


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    if not await kernel.is_alive():
        # 503 = "service unavailable": the kernel crashed (e.g. out of memory).
        # The agent should call /reset and tell the model its variables are gone.
        raise HTTPException(status_code=503, detail="Kernel is not running. Call /reset.")
    return await kernel.run(request.code, request.timeout)


@app.post("/reset")
async def reset():
    await kernel.restart()
    return {"status": "ok", "kernel": kernel.generation}