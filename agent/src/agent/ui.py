"""
The web UI: upload CSV files, chat with the cleanup bot, approve each block
of code before it runs, and download the cleaned files and the report.

    docker compose up -d --build      (then open http://127.0.0.1:7860)

or, without Docker (needs the 'ui' extra, which installs Gradio):

    uv run --extra ui python -m agent.ui

This file only draws a Session (session.py) and connects the buttons to it.
All the logic lives in Session, so it can be tested without a browser.

Privacy: share=False (no public Gradio link), analytics off, the chat's
"share to Hugging Face" button removed, the server listens only where
config.UI_HOST says (this machine), and uploads are deleted on New session and
on exit (and at startup, in case the app crashed).
"""

import atexit
import os
import signal
import sys

# Must be set before Gradio is imported: it reads these once, at import.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr   # noqa: E402  (imported after the setting above on purpose)

from . import config  # noqa: E402
from .session import Session, make_session  # noqa: E402

LOG_HEADERS = ["#", "Table", "Change", "What changed", "Reason", "Example"]


def build_app(session: Session) -> gr.Blocks:
    with gr.Blocks(title="Cleanup bot", analytics_enabled=False, fill_height=True) as app:
        gr.Markdown("## Cleanup bot\nChecks your CSV files, proposes fixes, and applies only the ones you approve.")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=480, buttons=["copy"],  # no "share"
                                  placeholder="Upload CSV files on the right, then ask the bot to check them.")
                # Shown only while code is waiting for approval.
                code = gr.Code(label="Code waiting for your approval", language="python",
                               interactive=False, visible=False)
                with gr.Row(visible=False) as approval:
                    run = gr.Button("Run it", variant="primary", scale=1)
                    skip = gr.Button("Don't run", scale=1)
                    feedback = gr.Textbox(placeholder="Or tell the model what to change, then press Enter",
                                          show_label=False, scale=4)
                # Hidden while code is waiting, so there's one thing to do at a time.
                with gr.Row() as composer:
                    box = gr.Textbox(placeholder="e.g. Check these files for problems and tell me what you would fix.",
                                     show_label=False, scale=5, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1)
            with gr.Column(scale=2):
                upload = gr.File(label="Upload CSV files", file_count="multiple", file_types=[".csv"])
                files = gr.Markdown()
                # The first rows of the latest upload: check it's the right file
                # before asking. Hidden until something is uploaded.
                with gr.Accordion("Preview", open=False, visible=False) as preview_box:
                    preview = gr.Dataframe(interactive=False, wrap=True)
                auto = gr.Checkbox(label="Run code without asking", value=False,
                                   info="Off: you approve every block of code before it runs.")
                new = gr.Button("New session")
                gr.Markdown(f"Model: {config.MODEL}. New session deletes the uploaded files.")

        # Full width below the chat: the change log's six columns need the room.
        with gr.Tabs():
            with gr.Tab("Change log"):
                log = gr.Dataframe(headers=LOG_HEADERS, interactive=False, wrap=True, value=[],
                                   column_widths=["4%", "12%", "16%", "20%", "16%", "32%"])
            with gr.Tab("Downloads"):
                downloads = gr.File(label="Cleaned files and report", file_count="multiple",
                                    interactive=False)

        # Every handler redraws the same components from the session.
        outputs = [chat, code, approval, composer, box, feedback, files, log, downloads, upload,
                   preview_box, preview, send, run, skip]

        def file_list() -> str:
            if not session.files:
                return "**Files:** none yet. Upload CSV files above."
            lines = []
            for f in session.files:
                name = f.split("/")[-1]
                if name in session.file_info:
                    name += f": {session.file_info[name]:,} rows × {session.columns[name]} columns"
                lines.append(f"- {name}")
            return "**Files:**\n" + "\n".join(lines)

        def preview_view():
            if session.preview is None:
                return gr.update(visible=False), gr.update(value=None)
            name, header, rows = session.preview
            width = len(header)
            rows = [(r + [""] * width)[:width] for r in rows]   # ragged rows would break the table
            return (gr.update(visible=True, label=f"Preview: first {len(rows)} rows of {name}"),
                    gr.update(value={"headers": header, "data": rows}))

        def view(box_text="", busy=False):
            """busy=True while the model or the sandbox is working: the buttons
            are greyed out, so it's clear a click would only queue up."""
            waiting = session.pending_code is not None
            return (list(session.messages),
                    gr.update(value=session.pending_code or "", visible=waiting),
                    gr.update(visible=waiting),
                    gr.update(visible=not waiting),
                    box_text,   # the message box: cleared once the message is taken
                    "",         # the feedback box
                    file_list(),
                    session.log_rows,
                    [str(p) for p in session.downloads] or None,
                    None,       # empty the upload box once its files are copied
                    *preview_view(),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy))

        # Generators: Gradio redraws on every yield, so replies and results
        # appear as they happen. The change log and downloads are read from
        # the sandbox once, at the end of each turn.
        def finish():
            session.refresh()
            return view()

        def on_send(text):
            yield view(box_text=text, busy=True)   # keep the text visible until it's been taken
            for _ in session.send(text):
                yield view(busy=True)
            yield finish()

        def on_run():
            yield view(busy=True)
            for _ in session.approve():
                yield view(busy=True)
            yield finish()

        def on_skip(text=""):
            yield view(busy=True)
            for _ in session.decline(text):
                yield view(busy=True)
            yield finish()

        def on_upload(paths):
            session.add_files(paths or [])
            return view()

        def on_new():
            session.new()
            return view()

        # One shared queue slot ("agent") for everything that talks to the
        # model or the sandbox: two clicks can never run at the same time.
        busy = {"concurrency_id": "agent", "concurrency_limit": 1}
        send.click(on_send, box, outputs, **busy)
        box.submit(on_send, box, outputs, **busy)
        run.click(on_run, None, outputs, **busy)
        skip.click(on_skip, None, outputs, **busy)
        feedback.submit(on_skip, feedback, outputs, **busy)
        upload.upload(on_upload, upload, outputs, **busy)
        new.click(on_new, None, outputs, **busy)
        # The chat's own trash icon can't be hidden in Gradio 6.31. Clearing
        # only the screen would leave the model remembering everything, so the
        # icon does exactly what New session does.
        chat.clear(on_new, None, outputs, **busy)
        auto.change(session.set_auto, auto, None)
        app.load(view, None, outputs)   # a page refresh shows the current conversation
    return app


def error_app(message: str) -> gr.Blocks:
    """Shown instead of the chat when the sandbox or the model can't be reached."""
    with gr.Blocks(title="Cleanup bot", analytics_enabled=False) as app:
        gr.Markdown(f"## Cleanup bot can't start\n\n{message}\n\n"
                    "Fix that, then restart the UI: `docker compose restart ui`")
    return app


def main() -> None:
    try:
        session = make_session()
        app = build_app(session)
        # Delete the uploads when the app stops. `docker compose stop` sends
        # SIGTERM, which by default ends Python WITHOUT running atexit
        # functions; turning it into a normal exit makes atexit run.
        atexit.register(session.close)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    except Exception as e:   # ExecutorNotSafe, ModelNotAvailable, a connection error
        print(f"Can't start: {e}")
        app = error_app(f"**{type(e).__name__}:** {e}")
    # share=False: no public link. ssr_mode=False: no Node.js server needed.
    # footer_links: leave out "api" (this app isn't meant to be scripted).
    # allowed_paths: the only folder Gradio may serve downloads from.
    app.launch(server_name=config.UI_HOST, server_port=config.UI_PORT, share=False, ssr_mode=False,
               footer_links=["gradio", "settings"], allowed_paths=[str(config.OUTPUT_DIR / "clean")],
               max_file_size="500mb")


if __name__ == "__main__":
    main()