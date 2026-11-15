# UI image: the Gradio web app and the agent loop. Model-written code never
# runs here; it runs only in the executor sandbox.

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first (cached), then the agent package itself. --extra ui
# adds Gradio, which only this image needs.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --extra ui --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra ui

# Gradio keeps uploaded files in GRADIO_TEMP_DIR; /tmp is in memory
# (compose.yaml), so uploads never reach a disk.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GRADIO_TEMP_DIR=/tmp/gradio \
    GRADIO_ANALYTICS_ENABLED=False

USER app
EXPOSE 7860
CMD ["python", "-m", "agent.ui"]
