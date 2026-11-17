"""
The web UI, in two tabs:
- Clean: upload CSV files, chat with the cleanup bot, approve each block of
  code before it runs, and download the cleaned files and the report.
- Ask questions: ask about this session's tables (files cleaned and saved on
  the Clean tab, plus CSVs uploaded on this tab). The model
  writes read-only SQL, which runs automatically and is always shown.

    docker compose up -d --build      (then open http://127.0.0.1:7860)

or, without Docker (needs the 'ui' extra, which installs Gradio):

    uv run --extra ui python -m agent.ui

This file only draws a Session (session.py) and an AskSession
(sql_session.py) and connects the buttons to them. All the logic lives in
those two classes, so it can be tested without a browser.

Privacy: share=False (no public Gradio link), analytics off, the chats'
"share to Hugging Face" buttons removed, the server listens only where
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
from .sql_session import AskSession, make_ask_session  # noqa: E402

LOG_HEADERS = ["#", "Table", "Change", "What changed", "Reason", "Example"]
NO_TABLES = "No tables yet. Clean and save files in the Clean tab, or upload CSV files above."


def build_app(session: Session, ask: AskSession) -> gr.Blocks:
    with gr.Blocks(title="Cleanup bot", analytics_enabled=False, fill_height=True) as app:
        gr.Markdown("## Cleanup bot\nChecks your CSV files, proposes fixes, and applies only the ones you "
                    "approve. Then ask questions about the cleaned tables.")
        with gr.Tabs():
            # ---------------------------------------------------------------
            # Clean tab: the original layout, unchanged.
            # ---------------------------------------------------------------
            with gr.Tab("Clean"):
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
                        # A dropdown and a button rather than the x on gr.File,
                        # which is less dependable in Gradio 6.31. Hidden with no files.
                        with gr.Row(visible=False) as remove_row:
                            remove_pick = gr.Dropdown(label="Remove a file", choices=[], value=None, scale=3)
                            remove = gr.Button("Remove", variant="stop", scale=1)
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

            # ---------------------------------------------------------------
            # Ask questions tab: read-only SQL over output/clean/. Queries
            # run without approval (they can't change anything), but each
            # one is shown in the chat as a "Ran a query" section.
            # ---------------------------------------------------------------
            with gr.Tab("Ask questions") as ask_tab:
                with gr.Row():
                    with gr.Column(scale=3):
                        ask_chat = gr.Chatbot(label="Conversation", height=480, buttons=["copy"],  # no "share"
                                              placeholder="Ask a question about the cleaned tables listed on the right.")
                        with gr.Row():
                            ask_box = gr.Textbox(placeholder="e.g. How many rows does each table have?",
                                                 show_label=False, scale=5)
                            ask_send = gr.Button("Send", variant="primary", scale=1)
                    with gr.Column(scale=2):
                        ask_upload = gr.File(label="Upload CSV files (treated as already clean)",
                                             file_count="multiple", file_types=[".csv"])
                        ask_files = gr.Markdown()
                        with gr.Row(visible=False) as ask_remove_row:
                            ask_remove_pick = gr.Dropdown(label="Remove a file", choices=[], value=None, scale=3)
                            ask_remove = gr.Button("Remove", variant="stop", scale=1)
                        tables = gr.Markdown()
                        refresh = gr.Button("Refresh tables")
                        ask_new = gr.Button("New conversation")
                        gr.Markdown("The tables are read-only: questions can't change any file. "
                                    "New conversation clears only this tab's chat; New session "
                                    "(Clean tab) resets both tabs.")
                # Full width: a result can have many columns. Hidden until a
                # query succeeds: an empty Dataframe in Gradio 6.31 draws
                # placeholder headers "1 2 3", which look like a real result.
                result = gr.Dataframe(label="Last result", interactive=False, wrap=True,
                                      max_height=400, value=None, visible=False)

        # ===================================================================
        # Clean tab: every handler redraws the same components from the session.
        # ===================================================================
        outputs = [chat, code, approval, composer, box, feedback, files, log, downloads, upload,
                   preview_box, preview, remove_row, remove_pick, send, run, skip, remove]

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

        def remove_view(names: list[str]):
            """The remove dropdown and its row: hidden when there's nothing to remove."""
            return gr.update(visible=bool(names)), gr.update(choices=names, value=None)

        def view(box_text="", busy=False):
            """busy=True while the model or the sandbox is working: the buttons
            are greyed out, so it's clear a click would only queue up.
            box_text=None leaves the message box as it is (a redraw caused by
            the other tab must not wipe what you're typing)."""
            waiting = session.pending_code is not None
            return (list(session.messages),
                    gr.update(value=session.pending_code or "", visible=waiting),
                    gr.update(visible=waiting),
                    gr.update(visible=not waiting),
                    gr.update() if box_text is None else box_text,  # cleared once the message is taken
                    "",         # the feedback box
                    file_list(),
                    session.log_rows,
                    [str(p) for p in session.downloads] or None,
                    None,       # empty the upload box once its files are copied
                    *preview_view(),
                    *remove_view([f.split("/")[-1] for f in session.files]),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy))

        def saved_files() -> list:
            """Each saved file with its modification time, so saving the same
            table again also counts as a change."""
            return [(p, p.stat().st_mtime) for p in session.downloads if p.exists()]

        # The Clean tab's turn handlers also redraw the Ask tab (both_outputs),
        # because saving or removing a file there changes the Ask tables.
        # While a turn runs, the Ask outputs are skipped: nothing changes there.
        def busy_view(box_text=""):
            return (*view(box_text=box_text, busy=True), *[gr.skip()] * len(ask_outputs))

        # Generators: Gradio redraws on every yield, so replies and results
        # appear as they happen. The change log and downloads are read from
        # the sandbox once, at the end of each turn.
        def finish():
            before = saved_files()
            session.refresh()
            if saved_files() != before:
                # This turn saved files: give the Ask tab the new tables.
                ask.refresh_tables()
            return (*view(), *ask_view(box_text=None))

        def on_send(text):
            yield busy_view(box_text=text)   # keep the text visible until it's been taken
            for _ in session.send(text):
                yield busy_view()
            yield finish()

        def on_run():
            yield busy_view()
            for _ in session.approve():
                yield busy_view()
            yield finish()

        def on_skip(text=""):
            yield busy_view()
            for _ in session.decline(text):
                yield busy_view()
            yield finish()

        def on_upload(paths):
            session.add_files(paths or [])
            return view(box_text=None)

        def on_remove(name):
            """Remove a Clean upload. If it was cleaned and saved, its table
            leaves the Downloads and the Ask tab too (refresh, then the Ask
            tables)."""
            if not name:
                gr.Warning("Pick a file to remove first.")
            else:
                problem = session.remove_file(name)
                if problem:
                    gr.Warning(problem)
                else:
                    session.refresh()
                    ask.refresh_tables()
            return (*view(box_text=None), *ask_view(box_text=None))

        def on_new():
            """New session resets both tabs: chats, uploads on both, tables."""
            session.new()
            ask.reset()
            return (*view(), *ask_view())

        # ===================================================================
        # Ask questions tab: the same pattern, drawn from the AskSession.
        # ===================================================================
        ask_outputs = [ask_chat, ask_box, ask_upload, ask_files, ask_remove_row, ask_remove_pick, tables, result,
                       ask_send, refresh, ask_new, ask_remove]

        def ask_file_list() -> str:
            if not ask.files:
                return "**Uploaded here:** none. CSV files uploaded here are used as they are, without cleaning."
            lines = [f"- {n}: {ask.file_info[n][0]:,} rows × {ask.file_info[n][1]} columns" for n in ask.files]
            return "**Uploaded here:**\n" + "\n".join(lines)

        def table_list() -> str:
            if ask.schema_error:
                return f"**Tables:** can't be read right now.\n\n{ask.schema_error}"
            lines = [f"- **{t['name']}** ({ask.table_origin.get(t['name'], 'cleaned')}): "
                     f"{t['rows']:,} rows × {len(t['columns'])} columns" for t in ask.tables]
            text = "**Tables:**\n" + "\n".join(lines) if lines else f"**Tables:** {NO_TABLES}"
            if ask.problems:
                text += "\n\n**Not loaded:**\n" + "\n".join(f"- {p}" for p in ask.problems)
            return text

        def result_view():
            r = ask.last_result
            if r is None:
                return gr.update(value=None, visible=False)
            width = len(r["columns"])
            rows = [(list(row) + [None] * width)[:width] for row in r["rows"]]
            label = f"Last result: {r['row_count']:,} rows"
            if r["truncated"] or len(rows) < r["row_count"]:
                label += f" (first {len(rows):,} shown)"
            return gr.update(value={"headers": r["columns"], "data": rows}, label=label, visible=True)

        def ask_view(box_text="", busy=False):
            """box_text=None leaves the question box as it is (see view())."""
            return (list(ask.messages),
                    gr.update() if box_text is None else box_text,
                    None,       # empty the upload box once its files are copied
                    ask_file_list(),
                    *remove_view(list(ask.files)),
                    table_list(),
                    result_view(),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy),
                    gr.update(interactive=not busy))

        def on_ask(text):
            yield ask_view(box_text=text, busy=True)
            for _ in ask.send(text):
                yield ask_view(busy=True)
            yield ask_view()

        def on_refresh():
            ask.refresh_tables()
            return ask_view()

        def on_ask_new():
            ask.new()
            return ask_view()

        def on_ask_upload(paths):
            ask.add_files(paths or [])
            return ask_view(box_text=None)

        def on_ask_remove(name):
            if not name:
                gr.Warning("Pick a file to remove first.")
            else:
                problem = ask.remove_file(name)
                if problem:
                    gr.Warning(problem)
            return ask_view(box_text=None)

        # One shared queue slot ("agent") for everything that talks to the
        # model or the sandbox, in both tabs: two clicks can never run at the
        # same time, so the one local model is never asked two things at once.
        busy = {"concurrency_id": "agent", "concurrency_limit": 1}
        both = outputs + ask_outputs   # handlers that can change both tabs
        send.click(on_send, box, both, **busy)
        box.submit(on_send, box, both, **busy)
        run.click(on_run, None, both, **busy)
        skip.click(on_skip, None, both, **busy)
        feedback.submit(on_skip, feedback, both, **busy)
        upload.upload(on_upload, upload, outputs, **busy)
        remove.click(on_remove, remove_pick, both, **busy)
        new.click(on_new, None, both, **busy)
        # The chat's own trash icon can't be hidden in Gradio 6.31. Clearing
        # only the screen would leave the model remembering everything, so the
        # icon does exactly what New session does.
        chat.clear(on_new, None, both, **busy)
        auto.change(session.set_auto, auto, None)

        ask_send.click(on_ask, ask_box, ask_outputs, **busy)
        ask_box.submit(on_ask, ask_box, ask_outputs, **busy)
        refresh.click(on_refresh, None, ask_outputs, **busy)
        ask_tab.select(on_refresh, None, ask_outputs, **busy)   # opening the tab re-reads the tables
        ask_new.click(on_ask_new, None, ask_outputs, **busy)
        ask_chat.clear(on_ask_new, None, ask_outputs, **busy)   # same reason as chat.clear above
        ask_upload.upload(on_ask_upload, ask_upload, ask_outputs, **busy)
        ask_remove.click(on_ask_remove, ask_remove_pick, ask_outputs, **busy)

        # A page refresh shows the current conversations.
        app.load(view, None, outputs)
        app.load(ask_view, None, ask_outputs)
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
        ask = make_ask_session(session)   # shares the model and the sandbox client
        app = build_app(session, ask)
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
