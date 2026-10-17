# Executor image: the FastAPI server + Jupyter kernel that runs model-written code.
# Built and started by compose.yaml in the repo root; see that file for the
# sandbox settings (no internet, read-only data, resource limits).

# Base image: Debian Linux with Python 3.14 preinstalled. "slim" leaves out
# compilers and other build tools we don't need, which keeps the image small
# and gives model code fewer tools to misuse.
FROM python:3.14-slim

# Copy only the uv program out of Astral's official image. "COPY --from" can
# copy files out of any other image, so we get uv without installing it.
# Pinned version: rebuilding next month gives the same uv. Keep this at or
# above your local `uv --version`, so it can read the uv.lock you created.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

# uv settings for building images (from uv's Docker guide):
#   UV_COMPILE_BYTECODE: precompile .py files to .pyc now, so the container
#     starts faster (and doesn't need to write them later; the filesystem
#     will be read-only).
#   UV_LINK_MODE=copy: copy package files instead of linking them from uv's
#     cache. Docker's build cache lives elsewhere, so links would break.
#   UV_PYTHON_DOWNLOADS=never: always use the image's Python 3.14, never
#     download another one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# A normal, unprivileged user to run everything as. By default containers run
# as root; if model code ever found a way out of the container, root would
# make that far worse. Created before copying any code so this step stays cached.
RUN useradd --create-home --uid 1000 analyst

WORKDIR /app

# Dependencies first, code second. Docker caches each step and only reruns a
# step when its inputs change, so editing server.py later skips this slow
# install entirely.
#   --locked: install exactly what uv.lock says; fail if it's out of date.
#   --no-dev: skip development-only packages (httpx for the tests).
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY server.py ./

# Put the project's virtual environment first on PATH, so `uvicorn` and the
# kernel's `python` are the venv's. DATA_DIR / OUTPUT_DIR are the mount points
# compose.yaml attaches your folders to.
ENV PATH="/app/.venv/bin:$PATH" \
    DATA_DIR=/data \
    OUTPUT_DIR=/output

# From here on, everything runs as the unprivileged user.
USER analyst

# Documentation only: says the server listens on 8000. It doesn't open anything.
EXPOSE 8000

# 0.0.0.0 = "all network interfaces INSIDE this container". The container's
# own 127.0.0.1 is unreachable from outside it, even from the proxy next to it.
# Who can actually reach port 8000 is decided by compose.yaml, not here.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]