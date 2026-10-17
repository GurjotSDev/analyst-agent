"""
Stage 2 checks for the executor server. Start the server first, in another terminal:
    uv run uvicorn server:app --host 127.0.0.1 --port 8000
Then run:
    uv run python test_executor.py
"""

import httpx  # HTTP client: sends requests to the server, like a browser would.

BASE = "http://127.0.0.1:8000"


def execute(code: str, timeout: float = 60) -> dict:
    # The HTTP request gets extra time on top of the kernel's own timeout, so
    # the server can interrupt the code and reply before httpx gives up.
    r = httpx.post(f"{BASE}/execute", json={"code": code, "timeout": timeout},
                   timeout=timeout + 30)
    r.raise_for_status()  # raises an exception for any 4xx/5xx response
    return r.json()


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


# 1. Is the server up and the kernel alive?
health = httpx.get(f"{BASE}/health").json()
check("health", health["status"] == "ok", str(health))

# 2. Printed output comes back.
r = execute("print(1 + 1)")
check("print", r["status"] == "ok" and r["stdout"].strip() == "2", repr(r["stdout"]))

# 3. Variables persist between executions (the reason we use a kernel).
execute("x = 41")
r = execute("print(x + 1)")
check("state persists", r["stdout"].strip() == "42", repr(r["stdout"]))

# 4. The last line's value is returned, like in a notebook.
r = execute("import pandas as pd\npd.DataFrame({'a': [1, 2]})")
check("last-line value", "a" in r["stdout"], repr(r["stdout"][:60]))

# 5. Errors come back as a clean traceback, not a crash.
r = execute("1 / 0")
check("error", r["status"] == "error" and "ZeroDivisionError" in (r["error"] or ""),
      (r["error"] or "").splitlines()[-1])

# 6. A chart comes back as a PNG and is saved with an ID.
r = execute("import matplotlib.pyplot as plt\nplt.plot([1, 3, 2])\nplt.show()")
check("chart", len(r["charts"]) == 1, str([c["id"] for c in r["charts"]]))

# 7. Long-running code is stopped. Either the interrupt works and variables
#    survive, or the server restarts the kernel and says so. Both are correct;
#    which one happens depends on the operating system.
r = execute("import time\ntime.sleep(30)", timeout=3)
restarted = r["kernel_restarted"]
check("timeout", r["status"] == "timeout",
      "(kernel restarted)" if restarted else "(interrupted, state kept)")
r = execute("print(x)")
if restarted:
    check("usable after timeout", r["status"] == "error" and "NameError" in (r["error"] or ""),
          "fresh kernel, x is gone as expected")
else:
    check("usable after timeout", r["stdout"].strip() == "41",
          f"status={r['status']} stdout={r['stdout']!r} error={(r['error'] or '')[-120:]!r}")
execute("x = 41")  # make sure x exists again for the next checks

# 7b. Code that ignores the stop request: the server must restart the kernel
#     and stay usable, never hang or mix outputs into the next request.
r = execute("import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "time.sleep(30)", timeout=3)
check("unstoppable code", r["status"] == "timeout" and r["kernel_restarted"],
      r["error"][:60] if r["error"] else "")
r = execute("print('still working')")
check("usable after restart", r["stdout"].strip() == "still working", repr(r["stdout"]))

# 8. pandas 3 is what's installed.
r = execute("import pandas as pd\nprint(pd.__version__)")
check("pandas 3", r["stdout"].startswith("3."), r["stdout"].strip())

# 9. Reset clears variables.
execute("x = 41")
httpx.post(f"{BASE}/reset", timeout=90).raise_for_status()
r = execute("print(x)")
check("reset clears state", r["status"] == "error" and "NameError" in (r["error"] or ""))