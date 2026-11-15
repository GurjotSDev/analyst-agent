# Executor image: the FastAPI server and the Jupyter kernel that runs
# model-written code. The sandbox settings (no internet, read-only data,
# resource limits) are in compose.yaml.

# slim: no compilers or build tools, so a smaller image and fewer tools for
# model code to misuse.
FROM python:3.14-slim

# Pinned uv; keep it at or above the local version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

# Precompile bytecode (the filesystem is read-only at runtime), copy packages
# instead of linking to uv's cache, and always use the image's Python.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Run as an unprivileged user, never root.
RUN useradd --create-home --uid 1000 analyst

WORKDIR /app

# Dependencies before code, so editing the code doesn't rerun the install.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY server.py analyst_helpers.py cleaning.py report.py ./

# DATA_DIR / OUTPUT_DIR are where compose.yaml mounts the data/ and output/ folders.
ENV PATH="/app/.venv/bin:$PATH" \
    DATA_DIR=/data \
    OUTPUT_DIR=/output

USER analyst

EXPOSE 8000

# 0.0.0.0 inside the container; compose.yaml decides who can reach the port.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
