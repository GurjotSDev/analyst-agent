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
r = execute("import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 3, 2])\n"
            "ax.set_title('Test chart')\n"
            "ax.set_xlabel('Step')\n"
            "ax.set_ylabel('Value')\n"
            "plt.show()")
check("chart", len(r["charts"]) == 1, str([c["id"] for c in r["charts"]]))
# Delete the test chart so test runs don't leave files in output/charts.
# The kernel's working folder is output/, so "charts/..." finds it both
# locally and in Docker.
for chart in r["charts"]:
    execute(f"import os\nos.remove('charts/{chart['id']}.png')")

# 6a. Charts still save if the charts folder was deleted while the server
#     runs (by you on your PC, or by model code). This used to crash with
#     FileNotFoundError and an HTTP 500.
execute("import shutil\nshutil.rmtree('charts', ignore_errors=True)")
r = execute("fig, ax = plt.subplots()\nax.plot([1, 2])\nax.set_title('T')\n"
            "ax.set_xlabel('x')\nax.set_ylabel('y')\nplt.show()")
check("chart after folder deleted", r["status"] == "ok" and len(r["charts"]) == 1, r["status"])
for chart in r["charts"]:
    execute(f"import os\nos.remove('charts/{chart['id']}.png')")

# 6b. The house chart style is active (no top/right box lines, fixed colours).
r = execute("import matplotlib.pyplot as plt\n"
            "print(plt.rcParams['axes.spines.top'], plt.rcParams['axes.prop_cycle'].by_key()['color'][0])")
check("chart style applied", r["stdout"].strip() == "False #2a78d6", repr(r["stdout"].strip() or r["error"]))

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

# 8b. The data-quality helpers are preloaded in every kernel.
r = execute("print(callable(profile), callable(compare_keys), callable(merge_variants))")
check("helpers preloaded", r["stdout"].strip() == "True True True", repr(r["stdout"].strip() or r["error"]))

# 8c. pd, np and plt are preloaded too (the model's first chart failed without plt).
r = execute("print(pd.__name__, np.__name__, plt.__name__)")
check("pd/np/plt preloaded", r["stdout"].strip() == "pandas numpy matplotlib.pyplot",
      repr(r["stdout"].strip() or r["error"]))

# 9. Reset clears variables.
execute("x = 41")
httpx.post(f"{BASE}/reset", timeout=90).raise_for_status()
r = execute("print(x)")
check("reset clears state", r["status"] == "error" and "NameError" in (r["error"] or ""))
r = execute("print(callable(profile))")
check("helpers back after reset", r["stdout"].strip() == "True")