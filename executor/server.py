"""
Executor server: runs model-written Python in a persistent Jupyter kernel.

The agent sends code over HTTP; this server runs it in the kernel and sends
back what happened (printed text, errors, charts). In stage 3 this whole file
runs inside the Docker container, so the code it executes is sandboxed.

Endpoints:
    GET  /health   is the kernel alive?
    POST /execute  run code, return its outputs
    POST /reset    restart the kernel with a clean slate

Run locally:
    uv run uvicorn server:app --host 127.0.0.1 --port 8000
"""

import asyncio      
import base64       
import os           
import re           
from contextlib import asynccontextmanager  
from pathlib import Path 
from fastapi import FastAPI, HTTPException  
from jupyter_client import AsyncKernelManager  
from pydantic import BaseModel  


# --------------------------------------------------------------------------
# Settings (environment variables, with defaults for running locally)
# --------------------------------------------------------------------------

# In Docker these become /data and /output. Locally they default to
# the project's own data/ and output/ folders, one level above executor/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))
CHARTS_DIR = OUTPUT_DIR / "charts"

DEFAULT_TIMEOUT = 60        # seconds per execution, from the plan
MAX_OUTPUT_CHARS = 100_000  # safety cap on returned text. The agent trims further
                            # (to ~3,000) before sending anything to the model.

# Runs once in every fresh kernel, before any model code.
#   The inline backend is how Jupyter shows charts: each finished figure is
#   sent back as a PNG instead of opening a window.
KERNEL_SETUP = """
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
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


class Chart(BaseModel):
    id: str           # e.g. "chart_003", used by the final report to refer to it
    png_base64: str   # the image itself, for the UI to display


class ExecuteResponse(BaseModel):
    status: str                 # "ok", "error" or "timeout"
    stdout: str = ""            # printed output, plus the value of the last line
    stderr: str = ""            # warnings and anything printed to stderr
    error: str | None = None    # a clean traceback when status is "error"
    charts: list[Chart] = []
    truncated: bool = False     # True if stdout was cut at MAX_OUTPUT_CHARS
    kernel_restarted: bool = False  # True if a stuck kernel was replaced; the
                                    # agent must tell the model its state is gone


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------

class Kernel:
    """Owns one Jupyter kernel and runs code in it, one request at a time."""

    def __init__(self) -> None:
        self.manager: AsyncKernelManager | None = None
        self.client = None
        self.chart_count = 0
        # A lock lets only one execution run at a time. Two requests sharing a
        # kernel at once would mix up their outputs.
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

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
        # chart_count is NOT reset, so new charts never overwrite earlier files.
        await self.stop()
        await self.start()

    async def is_alive(self) -> bool:
        return bool(self.manager and await self.manager.is_alive())

    async def run(self, code: str, timeout: float) -> ExecuteResponse:
        """Run code for a request. The lock makes other requests wait their turn."""
        async with self.lock:
            return await self._execute(code, timeout)

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
        stdout, stderr, charts = [], [], []
        error = None

        while True:
            # The "iopub" channel is where the kernel broadcasts all outputs.
            msg = await self.client.get_iopub_msg()
            if msg["parent_header"].get("msg_id") != msg_id:
                continue  # left over from an earlier request; ignore it
            kind, content = msg["msg_type"], msg["content"]

            if kind == "stream":
                # Anything printed. "name" says whether it went to stdout or stderr.
                (stdout if content["name"] == "stdout" else stderr).append(content["text"])

            elif kind in ("execute_result", "display_data"):
                # execute_result = the value of the last line (like df.head()).
                # display_data = something shown explicitly, e.g. a chart.
                # Both carry several formats of the same thing; we take the PNG
                # if there is one, else the plain-text version.
                data = content["data"]
                if "image/png" in data:
                    charts.append(self._save_chart(data["image/png"]))
                elif "text/plain" in data:
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
            charts=charts,
            truncated=truncated,
        )

    async def _drain(self, msg_id: str) -> bool:
        """After an interrupt, read the remaining messages for that request so
        they don't leak into the next one. Returns True if the kernel reported
        idle (the code really stopped) within 10 seconds, False otherwise."""
        try:
            await asyncio.wait_for(self._collect(msg_id), 10)
            return True
        except TimeoutError:
            return False

    def _save_chart(self, png_base64: str) -> Chart:
        """Write a chart to output/charts/ and give it a stable ID."""
        self.chart_count += 1
        chart_id = f"chart_{self.chart_count:03d}"  # :03d pads to 3 digits: 001, 002...
        (CHARTS_DIR / f"{chart_id}.png").write_bytes(base64.b64decode(png_base64))
        return Chart(id=chart_id, png_base64=png_base64)


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
    return {"status": "ok" if await kernel.is_alive() else "dead"}


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
    return {"status": "ok"}