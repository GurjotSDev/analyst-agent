# UI image: the Gradio web app and the agent loop (our own code, not model
# code: model-written code still runs only in the executor sandbox).
# Built and started by compose.yaml in the repo root.

FROM python:3.14-slim

# uv, copied out of Astral's image (same version as the executor's).
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first (slow, cached), our code second (fast). The agent is a
# package, so it's installed in two steps:
#   1. --no-install-project: everything it depends on, without the agent itself
#   2. after copying src/: the agent package
# --extra ui adds Gradio, which only this image needs (see pyproject.toml).
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --extra ui --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra ui

# GRADIO_TEMP_DIR: Gradio keeps a copy of every uploaded file here. /tmp is
# in memory (compose.yaml), so uploads never reach a disk and vanish when the
# container stops. Analytics off, as everywhere else.
# PYTHONUNBUFFERED: print immediately, so messages show in `docker compose logs ui`.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GRADIO_TEMP_DIR=/tmp/gradio \
    GRADIO_ANALYTICS_ENABLED=False

USER app
EXPOSE 7860
CMD ["python", "-m", "agent.ui"]