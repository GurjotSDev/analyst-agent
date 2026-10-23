"""
All settings in one place. Every value can be overridden with an environment
variable of the same name prefixed with ANALYST_ (e.g. ANALYST_MODEL), so you
can try another model without editing code:

    $env:ANALYST_MODEL = "qwen3.5:9b"     (PowerShell)
"""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ANALYST_{name}", default)


# --- The model (Ollama, on this PC) ---
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = _env("MODEL", "qwen3.5:9b")
THINK = _env("THINK", "true").lower() == "true"   # let the model reason before answering
NUM_CTX = int(_env("NUM_CTX", "32768"))            # context window, in tokens

# --- The sandboxed executor (Docker, published on this PC only) ---
EXECUTOR_URL = _env("EXECUTOR_URL", "http://127.0.0.1:8000")
EXEC_TIMEOUT = float(_env("EXEC_TIMEOUT", "60"))   # seconds per code execution

# --- Limits from the plan ---
MAX_STEPS = int(_env("MAX_STEPS", "15"))            # code executions per user message
MAX_RESULT_CHARS = int(_env("MAX_RESULT_CHARS", "5000"))  # output the model sees per step

# --- Folders on THIS PC (the container sees them as /data and /output) ---
# This file is analyst-agent/agent/src/agent/config.py, so parents[3] is the
# repo root, analyst-agent/.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(_env("DATA_DIR", str(REPO_ROOT / "data")))
OUTPUT_DIR = Path(_env("OUTPUT_DIR", str(REPO_ROOT / "output")))