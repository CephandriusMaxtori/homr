"""Manual smoke test for the GUI handlers: exercises the real event functions
(Show pages / Convert) with a real PDF, without a browser.

    .venv\Scripts\python.exe _smoke_click.py show <pdf>
    .venv\Scripts\python.exe _smoke_click.py convert <image-or-folder> [output_dir]
"""

import sys
import traceback
from pathlib import Path

import gradio as gr

import homr.gui as gui


def handlers(app: object) -> dict[str, object]:
    """The click handlers of a built app, by name."""
    return {block.name: block.fn for block in app.fns.values() if block.name}


def show(pdf: str) -> None:
    app = gui.build_app()
    found = handlers(app)
    status, gallery, choices, log, preview = list(found["on_render_pages"](pdf, 300))[-1]
    print("STATUS :", status)
    print("GALLERY:", [(Path(p).name, caption) for p, caption in gallery])
    print("CHOICES:", choices)
    print("STATE  :", type(preview).__name__, len(preview.pages), "pages")
    print("SELECT :", [Path(p).name for p in preview.selected_pages([preview.pages[1]])])

    try:
        list(found["on_render_pages"](None, 300))
    except gr.Error as error:
        print("NO PDF :", error)
    try:
        list(found["on_render_pages"](pdf, 10))
    except gr.Error as error:
        print("BAD DPI:", error)


def convert(target: str, output_dir: str = "") -> None:
    app = gui.build_app()
    found = handlers(app)
    inputs = gui.collect_inputs([target] if Path(target).is_file() else [], "")
    print("collected:", inputs, flush=True)
    options = gui.GuiOptions(
        title_detection=False, output_dir=output_dir, merge=len(inputs) > 1
    )
    updates = 0
    for status, downloads, previews, log in gui.run_convert(inputs, options):
        updates += 1
        print("=" * 70, flush=True)
        print("STATUS:", status, flush=True)
        print("DOWNLOADS:", downloads, flush=True)
        print("PREVIEWS:", previews, flush=True)
        print("LOG TAIL:", "\n".join(log.strip().splitlines()[-5:]), flush=True)
    print("=" * 70)
    print(f"updates streamed: {updates}")


if __name__ == "__main__":
    try:
        if sys.argv[1] == "show":
            show(sys.argv[2])
        else:
            convert(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    except Exception:
        traceback.print_exc()
        sys.exit(1)