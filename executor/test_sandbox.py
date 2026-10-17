"""
Stage 3 checks: does the sandbox actually contain model-written code?

Each check sends code to the executor that tries to do something it must NOT be
able to do, exactly as model code would, and confirms it fails.
Start the containers first (from the repo root):
    docker compose up -d --build
Then, from executor/:
    uv run python test_sandbox.py
"""

import httpx

BASE = "http://127.0.0.1:8000"


def execute(code: str) -> dict:
    r = httpx.post(f"{BASE}/execute", json={"code": code, "timeout": 30}, timeout=60)
    r.raise_for_status()
    return r.json()


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


def last_line(r: dict) -> str:
    """The final line of an error traceback, e.g. 'OSError: Read-only file system'."""
    return (r["error"] or "").strip().splitlines()[-1] if r["error"] else r["stdout"].strip()


# 0. This executor says it's the sandboxed one (only compose.yaml sets that).
health = httpx.get(f"{BASE}/health").json()
check("reports sandboxed", health.get("sandboxed") is True, str(health))

# 1. No internet: looking up or connecting to any outside address fails.
r = execute("import urllib.request\nurllib.request.urlopen('https://pypi.org', timeout=5)")
check("no internet", r["status"] == "error", last_line(r))

# 2. Can't reach your PC either (host.docker.internal is Docker's name for it).
r = execute("import socket\nsocket.create_connection(('host.docker.internal', 11434), timeout=5)")
check("can't reach host (e.g. Ollama)", r["status"] == "error", last_line(r))

# 3. Your data folder can be read but never changed.
r = execute("import os\nprint(sorted(os.listdir('/data')))")
check("can read /data", r["status"] == "ok", r["stdout"].strip())
r = execute("open('/data/hacked.txt', 'w').write('x')")
check("can't write /data", r["status"] == "error", last_line(r))

# 4. The container's own files are read-only (it can't rewrite the server).
r = execute("open('/app/server.py', 'a').write('# changed')")
check("can't modify server code", r["status"] == "error", last_line(r))

# 5. Not running as root.
r = execute("import os\nprint(os.getuid())")
check("runs as non-root user", r["stdout"].strip() == "1000", f"uid={r['stdout'].strip()}")

# 6. Output folder IS writable (the one place results are meant to go).
r = execute("open('/output/sandbox_test.txt', 'w').write('ok')\nprint('written')")
check("can write /output", r["stdout"].strip() == "written", last_line(r))
execute("import os\nos.remove('/output/sandbox_test.txt')")

# 7. Memory is capped: grabbing far more than the 4 GB limit gets the kernel
#    killed (never the server or your PC). The server should notice within
#    seconds, restart the kernel, and say what happened.
import time
start = time.perf_counter()
r = execute("x = bytearray(6 * 1024**3)")
seconds = time.perf_counter() - start
check("memory limit", r["status"] == "error" and r["kernel_restarted"] and seconds < 25,
      f"{seconds:.0f}s, {(r['error'] or '')[:60]}")
r = execute("print('alive')")
check("usable after memory test", r["stdout"].strip() == "alive", repr(r["stdout"]))