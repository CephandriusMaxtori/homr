"""A local browser GUI for homr.

``homr`` is a command line tool first and foremost. This module wraps the very
same pipeline in a small Gradio app (installed with the ``gui`` extra) so that
the flags of ``homr.main`` can be used without memorizing them::

    homr-gui                # or: python -m homr.gui

Everything that talks to the neural network (and therefore pulls in
onnxruntime) is imported lazily inside the job, so the small helpers below stay
importable - and testable - with nothing but the standard library.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, TextIO, TypeVar

from homr.simple_logging import eprint

if TYPE_CHECKING:
    from gradio import Blocks

    from homr.main import ProcessingConfig
    from homr.music_xml_generator import XmlGeneratorArguments

T = TypeVar("T")

#: Everything homr accepts as input, see ``homr.main.get_all_image_files_in_folder``.
SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf")

#: Downloads and previews are served from here, which is passed to
#: ``Blocks.launch(allowed_paths=...)`` - Gradio refuses to serve arbitrary paths.
STAGING_ROOT = Path(tempfile.gettempdir()) / "homr-gui"

_CSS = """
.gradio-container { max-width: 1500px !important; }
footer { display: none !important; }
.homr-log textarea { font-family: Consolas, "Courier New", monospace; font-size: 12px; }
.homr-pages { border: 1px solid var(--border-color-primary); border-radius: 8px; }
"""

# homr replaces ``sys.stderr`` to follow a job, so only one job may run at a time.
_JOB_LOCK = threading.Lock()
_ACTIVE_CONTEXT: JobContext | None = None
_ACTIVE_LOCK = threading.Lock()


class GpuChoice(Enum):
    """The ``--gpu`` modes of ``homr.main``, as radio buttons.

    The members are plain strings so that they can be handed to Gradio as is.
    """

    AUTO = "Auto"
    CPU = "Force CPU"
    GPU = "Force GPU"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class GpuFlags:
    """Which parts of the pipeline may use a GPU execution provider."""

    transformer: bool
    segnet: bool
    coreml_encoder: bool


def resolve_gpu_flags(
    choice: GpuChoice,
    *,
    cuda: bool,
    rocm: bool,
    coreml: bool,
    coreml_encoder: bool,
) -> GpuFlags:
    """Mirrors the ``--gpu``/``--coreml-encoder`` logic of ``homr.main.main()``.

    The availability of the execution providers is passed in so that this can be
    tested without onnxruntime.
    """
    if choice is GpuChoice.CPU:
        return GpuFlags(transformer=False, segnet=False, coreml_encoder=False)
    if choice is GpuChoice.GPU:
        return GpuFlags(transformer=True, segnet=True, coreml_encoder=False)
    # CUDA/ROCm speeds up the whole pipeline. CoreML only helps segnet: its fp16
    # models are slower on the CPU EP than the fp32 ones, and the CoreML EP
    # cannot run the decoder. The CoreML encoder is a separate opt-in which
    # only applies when the transformer isn't already on CUDA/ROCm.
    transformer = cuda or rocm
    segnet = cuda or rocm or coreml
    return GpuFlags(
        transformer=transformer,
        segnet=segnet,
        coreml_encoder=coreml_encoder and not transformer and coreml,
    )


@dataclass(frozen=True)
class GuiOptions:
    """The flags of ``homr.main`` that the GUI exposes."""

    debug: bool = False
    cache: bool = False
    title_detection: bool = True
    gpu: GpuChoice = GpuChoice.AUTO
    coreml_encoder: bool = False
    large_page: bool = False
    metronome: int | None = None
    tempo: int | None = None
    selected_staff: int = -1
    write_staff_positions: bool = False
    read_staff_positions: bool = False
    output_dir: str = ""
    merge: bool = True

    def validate(self) -> None:
        """Raises ``ValueError`` for option combinations homr would reject late."""
        if self.metronome is not None and self.metronome <= 0:
            raise ValueError("Metronome needs a bpm value greater than 0.")
        if self.tempo is not None and self.tempo <= 0:
            raise ValueError("Tempo needs a bpm value greater than 0.")
        if self.selected_staff < -1:
            raise ValueError("Staff number must be -1 (all staffs) or a positive number.")

    def to_processing_config(self, gpu_flags: GpuFlags) -> ProcessingConfig:
        from homr.main import ProcessingConfig  # noqa: PLC0415

        return ProcessingConfig(
            self.debug,
            self.cache,
            self.write_staff_positions,
            self.read_staff_positions,
            self.selected_staff,
            gpu_flags.transformer,
            gpu_flags.segnet,
            gpu_flags.coreml_encoder,
            self.title_detection,
        )

    def to_xml_generator_args(self) -> XmlGeneratorArguments:
        from homr.music_xml_generator import XmlGeneratorArguments  # noqa: PLC0415

        return XmlGeneratorArguments(self.large_page, self.metronome, self.tempo)


@dataclass(frozen=True)
class JobState:
    """Progress of a running job: written by the worker, polled by the GUI."""

    step: int = 0
    total: int = 0
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.step / self.total)


@dataclass
class JobContext:
    """Handed to a running job so that it can report progress and check for a cancel."""

    cancel: threading.Event
    state: JobState = field(default_factory=JobState)

    def update(self, message: str, step: int | None = None, total: int | None = None) -> None:
        """Publishes a new progress state; ``step``/``total`` keep their value if omitted."""
        previous = self.state
        self.state = JobState(
            step=previous.step if step is None else step,
            total=previous.total if total is None else total,
            message=message,
        )

    def log(self, message: str) -> None:
        """Adds a line to the GUI log - the same text also lands on stderr."""
        eprint(message)


def request_cancel() -> bool:
    """Stops the running job after the current image. Returns False if nothing runs."""
    with _ACTIVE_LOCK:
        context = _ACTIVE_CONTEXT
    if context is None:
        return False
    context.cancel.set()
    return True


class _LogStream(io.TextIOBase):
    """A ``sys.stderr`` replacement that tees what homr logs into a queue.

    homr logs through ``eprint`` (``print(file=sys.stderr)``), so swapping out
    ``sys.stderr`` is all it takes to follow a running job. Everything is still
    passed on to the original stream, which keeps the terminal output intact.
    """

    def __init__(self, original: TextIO, sink: Queue[Any]) -> None:
        self._original = original
        self._sink = sink
        self._pending = ""

    def write(self, text: str) -> int:
        self._original.write(text)
        self._split(text)
        return len(text)

    def flush(self) -> None:
        self._original.flush()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def _split(self, text: str) -> None:
        """Splits into lines; ``\\r`` counts too, the model downloads use it."""
        self._pending += text.replace("\r", "\n")
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if line.strip():
                self._sink.put(line.rstrip())

    def flush_pending(self) -> None:
        """Emits output of a last line that never got its newline."""
        if self._pending.strip():
            self._sink.put(self._pending.rstrip())
        self._pending = ""


@dataclass(frozen=True)
class JobUpdate:
    """One slice of a running job: new log output, its progress, and at the very end its outcome."""

    log: str = ""
    state: JobState = field(default_factory=JobState)
    result: Any | None = None
    error: BaseException | None = None
    done: bool = False


def stream_job(
    job: Callable[[JobContext], T],
    poll_interval: float = 0.2,
    heartbeat: float = 1.0,
) -> Iterator[JobUpdate]:
    """Runs *job* in a worker thread and yields updates until it is finished.

    A conversion takes minutes and blocks, so it gets its own thread and
    reports through ``sys.stderr`` (see ``_LogStream``). The last update
    carries either ``result`` or ``error``. Updates are sent whenever the job
    logs something or changes its progress, and at least every *heartbeat*
    seconds so that a quiet job still refreshes the progress bar.
    """
    global _ACTIVE_CONTEXT  # noqa: PLW0603 - the cancel button needs to find the job
    if not _JOB_LOCK.acquire(blocking=False):
        raise RuntimeError("Another homr job is still running.")
    queue: Queue[str | None] = Queue()
    context = JobContext(cancel=threading.Event())
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = job(context)
        except BaseException as error:  # noqa: BLE001 - the GUI reports it
            outcome["error"] = error
            eprint(traceback.format_exc())
        finally:
            queue.put(None)

    stream = _LogStream(sys.stderr, queue)
    thread = threading.Thread(target=target, name="homr-gui-job", daemon=True)
    log = ""
    state = context.state
    last_update = time.monotonic()
    original_stderr = sys.stderr
    with _ACTIVE_LOCK:
        _ACTIVE_CONTEXT = context
    sys.stderr = stream
    try:
        thread.start()
        while True:
            finished = False
            try:
                item = queue.get(timeout=poll_interval)
            except Empty:
                if not thread.is_alive():
                    break
            else:
                if item is None:
                    finished = True
                else:
                    log += item + "\n"
            now = time.monotonic()
            if finished or log or state != context.state or now - last_update >= heartbeat:
                state = context.state
                last_update = now
                update = JobUpdate(log=log, state=state, done=finished)
                log = ""
                yield update
            if finished:
                break
    finally:
        sys.stderr = original_stderr
        thread.join()
        stream.flush_pending()
        with _ACTIVE_LOCK:
            _ACTIVE_CONTEXT = None
        _JOB_LOCK.release()
    yield JobUpdate(
        log=log,
        state=state,
        result=outcome.get("result"),
        error=outcome.get("error"),
        done=True,
    )


@dataclass
class ConversionReport:
    """What a single GUI run produced."""

    outputs: list[str] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    previews: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    cancelled: bool = False

    def summary(self) -> str:
        if self.outputs and not self.failures:
            return f"Wrote {len(self.outputs)} MusicXML file(s)."
        if not self.outputs and not self.failures:
            return "Nothing was converted."
        parts = []
        if self.outputs:
            parts.append(f"{len(self.outputs)} MusicXML file(s) written")
        if self.failures:
            parts.append(f"{len(self.failures)} file(s) failed")
        if self.cancelled:
            parts.append("stopped early")
        return ", ".join(parts) + "."

    def as_markdown(self) -> str:
        lines = [f"**{self.summary()}**"]
        for failure in self.failures:
            lines.append(f"- {failure}")
        return "\n".join(lines)


def _folder_images(folder: str) -> list[str]:
    """All images of *folder*, skipping the teaser/debug files of earlier runs."""
    from homr.main import get_all_image_files_in_folder  # noqa: PLC0415

    return get_all_image_files_in_folder(folder)


def collect_inputs(files: Sequence[str], folder: str = "") -> list[str]:
    """The images and PDFs to process: the chosen files plus the content of *folder*."""
    inputs: list[str] = []
    for entry in files:
        path = str(entry).strip()
        if not path:
            continue
        if os.path.isdir(path):
            inputs += _folder_images(path)
        elif os.path.isfile(path):
            inputs.append(path)
        else:
            raise ValueError(f"{path} is not a valid file or directory.")
    folder = folder.strip()
    if folder:
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a folder.")
        inputs += _folder_images(folder)
    # The same file can be picked twice, once as a file and once via a folder.
    return list(dict.fromkeys(inputs))


def _merge_xml(xml_paths: list[str], name: str, output_dir: str, ctx: JobContext) -> list[str]:
    """Merges MusicXML files into one, the way ``homr.main.run_homr`` does."""
    if len(xml_paths) < 2:
        return xml_paths
    from homr.relieur import process_concat  # noqa: PLC0415

    destination = output_dir or os.path.dirname(xml_paths[0])
    os.makedirs(destination, exist_ok=True)
    merged = os.path.join(destination, f"{name}_merged.musicxml")
    ctx.log(f"Merging {len(xml_paths)} files into {merged}")
    element, _, _ = process_concat(xml_paths)
    ET.ElementTree(element).write(merged, encoding="UTF-8", xml_declaration=True)
    for path in xml_paths:
        if os.path.exists(path):
            os.remove(path)
    return [merged]


def _new_staging_dir() -> Path:
    """A fresh directory for the downloads and previews of one run."""
    path = STAGING_ROOT / f"run-{uuid.uuid4().hex}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_file(path: str, staging: Path, index: int) -> str:
    """Copies *path* next to the other run artifacts so that Gradio can serve it."""
    destination = staging / f"{index:03d}_{os.path.basename(path)}"
    shutil.copyfile(path, destination)
    return str(destination)


def run_conversion(inputs: Sequence[str], options: GuiOptions, ctx: JobContext) -> ConversionReport:
    """Runs the homr pipeline over *inputs*; the GUI counterpart of ``homr.main.main()``.

    Raises ``ValueError`` on invalid input, returns the failures of single files
    in the report instead of giving up on a whole batch.
    """
    import onnxruntime as ort  # noqa: PLC0415

    from homr.main import download_weights, process_image  # noqa: PLC0415
    from homr.onnx_providers import (  # noqa: PLC0415
        coreml_available,
        cuda_available,
        rocm_available,
    )
    from homr.pdf_utils import render_pdf_to_image  # noqa: PLC0415
    from homr.title_detection import download_ocr_weights  # noqa: PLC0415

    options.validate()
    ctx.update("Downloading the models, this is only needed once", 0, len(inputs))
    ort.set_default_logger_severity(2 if options.debug else 3)
    gpu_flags = resolve_gpu_flags(
        options.gpu,
        cuda=cuda_available(),
        rocm=rocm_available(),
        coreml=coreml_available(),
        coreml_encoder=options.coreml_encoder,
    )
    ctx.log(
        f"Using {'the GPU' if gpu_flags.segnet else 'the CPU'} for segmentation, "
        f"{'the GPU' if gpu_flags.transformer else 'the CPU'} for symbol recognition"
    )
    download_weights(gpu_flags.segnet, gpu_flags.transformer, gpu_flags.coreml_encoder)
    if options.title_detection:
        download_ocr_weights()

    config = options.to_processing_config(gpu_flags)
    xml_generator_args = options.to_xml_generator_args()
    report = ConversionReport()
    parts: list[str] = []

    def convert(path: str, step: int) -> str | None:
        """Converts one image, returns its MusicXML (moved to the output folder)."""
        if ctx.cancel.is_set():
            return None
        ctx.update(f"Converting {os.path.basename(path)}", step, len(inputs))
        xml_file = process_image(path, config, xml_generator_args)
        if options.output_dir:
            destination_dir = os.path.abspath(os.path.expanduser(options.output_dir))
            os.makedirs(destination_dir, exist_ok=True)
            destination = os.path.join(destination_dir, os.path.basename(xml_file))
            if os.path.abspath(destination) != os.path.abspath(xml_file):
                shutil.move(xml_file, destination)
                xml_file = destination
        return xml_file

    for step, path in enumerate(inputs, start=1):
        if ctx.cancel.is_set():
            report.cancelled = True
            break
        try:
            ctx.log("=========================================")
            ctx.log(f"Processing {path}")
            if path.lower().endswith(".pdf"):
                pages = render_pdf_to_image(path)
                ctx.log(f"Rendered {len(pages)} page(s) from {os.path.basename(path)}")
                page_parts = [
                    xml
                    for offset, page in enumerate(pages)
                    if (xml := convert(page, step)) is not None
                ]
                if ctx.cancel.is_set():
                    report.cancelled = True
                if page_parts:
                    name = os.path.splitext(os.path.basename(path))[0]
                    parts += _merge_xml(page_parts, name, options.output_dir, ctx)
            else:
                xml_file = convert(path, step)
                if xml_file is not None:
                    parts.append(xml_file)
        except Exception as error:
            ctx.log(f"An error occurred while processing {path}: {error}")
            report.failures.append(f"**{os.path.basename(path)}**: {error}")

    if options.merge and len(parts) > 1:
        name = os.path.splitext(os.path.basename(parts[0]))[0]
        parts = _merge_xml(parts, name, options.output_dir, ctx)

    if parts:
        staging = _new_staging_dir()
        for index, path in enumerate(parts, start=1):
            report.outputs.append(path)
            report.downloads.append(stage_file(path, staging, index))
            teaser = _teaser_of(path)
            if teaser is not None:
                report.previews.append(stage_file(teaser, staging, 1000 + index))

    ctx.update(report.summary(), len(inputs), len(inputs))
    return report


def _teaser_of(xml_path: str) -> str | None:
    """The teaser image of a conversion, or None if there is none.

    ``homr.main.process_image`` writes it next to the image it converted, so a
    merged file has no teaser of its own - the pages keep theirs.
    """
    for candidate in (
        os.path.splitext(xml_path)[0] + "_teaser.png",
        os.path.splitext(os.path.basename(xml_path))[0] + "_teaser.png",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def render_pdf_pages(pdf_path: str, staging: Path, dpi: int = 300) -> list[str]:
    """Rasterizes every page of *pdf_path* to a JPG in *staging*.

    At the default 300 DPI this matches what the pipeline would render anyway
    (``homr.pdf_utils.render_pdf_to_image``), so selecting single pages does not
    change the recognition quality. Unlike that function the pages go somewhere
    else than next to the PDF, and they survive as JPGs the GUI can show.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415

    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(pdf_path)
    paths: list[str] = []
    try:
        for index, page in enumerate(pdf):
            bitmap = page.render(scale=scale)
            destination = staging / f"page-{index + 1:03d}.jpg"
            bitmap.to_pil().convert("RGB").save(destination, "JPEG", quality=92)
            paths.append(str(destination))
    finally:
        pdf.close()
    return paths


@dataclass(frozen=True)
class PdfPreview:
    """The rendered pages of a PDF, ready to be picked in the GUI."""

    source: str
    pages: list[str]

    @property
    def choices(self) -> list[tuple[str, str]]:
        """Checkbox labels and values: "Page 1" -> path of the rendered JPG."""
        return [(f"Page {index + 1}", path) for index, path in enumerate(self.pages)]

    def gallery(self) -> list[tuple[str, str]]:
        return [(path, f"Page {index + 1}") for index, path in enumerate(self.pages)]

    def selected_pages(self, selection: Sequence[str]) -> list[str]:
        """The chosen pages, in document order; everything if nothing is selected."""
        if not selection:
            return list(self.pages)
        chosen = set(selection)
        return [path for path in self.pages if path in chosen]


def preview_pdf(pdf_path: str, dpi: int = 300, ctx: JobContext | None = None) -> PdfPreview:
    """Renders a PDF so that single pages can be converted instead of all of them."""
    staging = _new_staging_dir()
    if ctx is not None:
        ctx.update("Rendering the pages", 0, 0)
    pages = render_pdf_pages(pdf_path, staging, dpi)
    if ctx is not None:
        ctx.log(f"Rendered {len(pages)} page(s) from {os.path.basename(pdf_path)} at {dpi} DPI")
    return PdfPreview(source=pdf_path, pages=pages)


def download_models(gpu: GpuChoice, coreml_encoder: bool, ctx: JobContext) -> str:
    """Downloads the models without converting anything, like ``homr --init``."""
    from homr.main import download_weights  # noqa: PLC0415
    from homr.onnx_providers import (  # noqa: PLC0415
        coreml_available,
        cuda_available,
        rocm_available,
    )
    from homr.title_detection import download_ocr_weights  # noqa: PLC0415

    ctx.update("Downloading the models", 0, 0)
    flags = resolve_gpu_flags(
        gpu,
        cuda=cuda_available(),
        rocm=rocm_available(),
        coreml=coreml_available(),
        coreml_encoder=coreml_encoder,
    )
    download_weights(flags.segnet, flags.transformer, flags.coreml_encoder)
    download_ocr_weights()
    return "**The models are ready.**"


def run_init(gpu: GpuChoice, coreml_encoder: bool) -> Iterator[tuple[str, str]]:
    """Event handler of the "Download models" button."""
    log = ""
    result: Any = None
    error: BaseException | None = None
    for update in stream_job(lambda ctx: download_models(gpu, coreml_encoder, ctx)):
        log += update.log
        result, error = update.result, update.error
    if error is not None:
        yield f"**Failed:** {error}", log
    elif result is not None:
        yield str(result), log


def run_convert(
    inputs: Sequence[str],
    options: GuiOptions,
) -> Iterator[tuple[str, list[str], list[tuple[str, str]], str]]:
    """Event handler of the "Convert" button: yields (status, downloads, previews, log)."""
    import gradio as gr  # noqa: PLC0415

    progress = gr.Progress()
    log = ""
    report = ConversionReport()
    try:
        for update in stream_job(lambda ctx: run_conversion(inputs, options, ctx)):
            log += update.log
            progress(update.state.fraction, desc=update.state.message)
            if update.error is not None:
                yield f"**Failed:** {update.error}", [], [], log
                return
            if isinstance(update.result, ConversionReport):
                report = update.result
            if update.done:
                yield report.as_markdown(), report.downloads, _previews(report), log
            else:
                yield _status_markdown(update.state, report), report.downloads, [], log
        progress(1.0, desc="Done")
    except Exception as error:
        yield f"**Failed:** {error}", [], [], log


def _status_markdown(state: JobState, report: ConversionReport) -> str:
    message = state.message or "Starting..."
    if report.failures:
        return f"{message}\n\n" + report.as_markdown()
    return message


def _previews(report: ConversionReport) -> list[tuple[str, str]]:
    """Gallery entries of the teaser images, which show the detected staves."""
    return [(preview, os.path.basename(preview)) for preview in report.previews]


def _theme() -> Any:
    """The look of the GUI: warm paper and the deep green of the homr demo page."""
    import gradio as gr  # noqa: PLC0415

    return gr.Theme(
        primary_hue=gr.themes.Color(
            c50="#f2f7f5",
            c100="#e0ece8",
            c200="#c2d9d3",
            c300="#9cc0b7",
            c400="#74a79b",
            c500="#4f8f81",
            c600="#2f7567",
            c700="#0f5c4c",
            c800="#0a463a",
            c900="#083329",
            c950="#041d17",
        ),
        font=["Iowan Old Style", "Palatino Linotype", "Palatino", "Georgia", "serif"],
        font_mono=["Consolas", "Courier New", "monospace"],
    )


def build_app() -> Blocks:
    """Builds the Gradio app of the homr GUI.

    The theme and CSS are passed to ``launch()`` instead, which is where Gradio 6
    expects them.
    """
    import gradio as gr  # noqa: PLC0415

    with gr.Blocks(title="homr", fill_width=True) as demo:
        gr.Markdown(
            "# homr\n"
            "Optical music recognition: turns a picture or a PDF of sheet music "
            "into MusicXML, which you can open with MuseScore, Sibelius or music21."
        )
        with gr.Row():
            with gr.Column(scale=3, min_width=320):
                with gr.Tabs():
                    with gr.Tab("Files"):
                        files = gr.Files(
                            label="Images or PDFs",
                            file_types=list(SUPPORTED_SUFFIXES),
                            type="filepath",
                        )
                        folder = gr.Textbox(
                            label="...or a whole folder",
                            placeholder=r"C:\scores\bach",
                        )
                        merge = gr.Checkbox(
                            label="Merge everything into a single MusicXML file",
                            value=True,
                        )
                    with gr.Tab("PDF pages"):
                        pdf_file = gr.File(
                            label="PDF",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        dpi = gr.Number(
                            label="Resolution (DPI)",
                            value=300,
                            info="300 is what the pipeline uses for PDFs anyway.",
                            precision=0,
                        )
                        render_button = gr.Button("Show pages", variant="primary")
                        page_grid = gr.Gallery(
                            label="Pages",
                            columns=3,
                            height=280,
                            interactive=True,
                            elem_classes="homr-pages",
                        )
                        page_choice = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Pages to convert",
                            info="Nothing selected converts every page.",
                        )
                with gr.Row():
                    convert_button = gr.Button("Convert", variant="primary")
                    cancel_button = gr.Button("Stop", variant="stop")
                with gr.Row():
                    models_button = gr.Button("Download models")
                    clear_button = gr.Button("Clear log", variant="secondary")
            with gr.Column(scale=2, min_width=280):
                with gr.Accordion("Inference", open=True):
                    gpu = gr.Radio(
                        choices=[choice.value for choice in GpuChoice],
                        value=GpuChoice.AUTO.value,
                        label="Processor",
                    )
                    coreml_encoder = gr.Checkbox(
                        label="Run the encoder on the Apple GPU (CoreML)",
                        info="Apple Silicon only, and it needs a long startup.",
                    )
                    title_detection = gr.Checkbox(
                        label="Detect the title (slower)",
                        value=True,
                    )
                with gr.Accordion("Output", open=False):
                    large_page = gr.Checkbox(label="Render onto larger pages")
                    metronome = gr.Number(label="Add a metronome mark (bpm)", precision=0)
                    tempo = gr.Number(label="Add a tempo mark (bpm)", precision=0)
                    output_dir = gr.Textbox(
                        label="Save into this folder",
                        placeholder="Leave empty to save next to the input, like the CLI",
                    )
                    selected_staff = gr.Number(
                        label="Only convert this staff",
                        value=-1,
                        info="-1 converts every staff of the image.",
                        precision=0,
                    )
                with gr.Accordion("Debugging", open=False):
                    debug = gr.Checkbox(
                        label="Write debug images",
                        info="Writes the intermediate steps next to the input.",
                    )
                    cache = gr.Checkbox(label="Reuse the cached segmentation")
                    with gr.Row():
                        write_staff_positions = gr.Checkbox(label="Save staff positions")
                        read_staff_positions = gr.Checkbox(label="Use saved positions")
                    gr.Markdown("_Saved positions are read from a `.txt` file next to the image._")
        status = gr.Markdown("Pick an image, a PDF or a folder to get started.")
        with gr.Row():
            downloads = gr.Files(label="MusicXML", interactive=False)
            previews = gr.Gallery(label="Detected staves", columns=2, height=320)
        log_box = gr.Textbox(
            label="Log",
            lines=14,
            max_lines=14,
            interactive=False,
            autoscroll=True,
            elem_classes="homr-log",
        )

        option_widgets = [
            gpu,
            coreml_encoder,
            title_detection,
            large_page,
            metronome,
            tempo,
            output_dir,
            selected_staff,
            debug,
            cache,
            write_staff_positions,
            read_staff_positions,
        ]

        def read_options(*values: Any) -> GuiOptions:
            """Builds the options from the widgets in ``option_widgets`` order."""
            (
                gpu_choice,
                use_coreml,
                detect_title,
                big_pages,
                metronome_bpm,
                tempo_bpm,
                destination,
                staff,
                write_debug,
                use_cache,
                write_positions,
                read_positions,
            ) = values
            options = GuiOptions(
                debug=write_debug,
                cache=use_cache,
                title_detection=detect_title,
                gpu=GpuChoice(gpu_choice),
                coreml_encoder=use_coreml,
                large_page=big_pages,
                metronome=int(metronome_bpm) if metronome_bpm else None,
                tempo=int(tempo_bpm) if tempo_bpm else None,
                selected_staff=int(staff) if staff is not None else -1,
                write_staff_positions=write_positions,
                read_staff_positions=read_positions,
                output_dir=(destination or "").strip(),
                merge=True,
            )
            options.validate()
            return options

        def on_convert(
            chosen: Any,
            chosen_folder: str,
            do_merge: bool,
            selection: Any,
            preview: PdfPreview | None,
            *option_values: Any,
        ) -> Iterator[tuple[str, list[str], list[tuple[str, str]], str]]:
            """Converts picked pages if there are any, else files and folders."""
            try:
                if preview is not None:
                    inputs = preview.selected_pages(selection or [])
                    if not inputs:
                        raise ValueError("No page selected.")
                else:
                    inputs = collect_inputs(
                        [entry.name for entry in chosen or []], chosen_folder or ""
                    )
                    if not inputs:
                        raise ValueError("Pick an image, a PDF or a folder first.")
                options = replace(read_options(*option_values), merge=do_merge)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            yield from run_convert(inputs, options)

        def on_render_pages(
            pdf: Any, resolution: Any
        ) -> Iterator[tuple[str, list[Any], list[str], str, PdfPreview | None]]:
            """Rasterizes a PDF so single pages can be converted."""
            path = getattr(pdf, "name", None) or (pdf or "")
            if not path:
                raise gr.Error("Pick a PDF first.")
            dpi_value = int(resolution) if resolution else 300
            if dpi_value < 72 or dpi_value > 1200:
                raise gr.Error("The resolution must be between 72 and 1200 DPI.")
            log = ""
            error: BaseException | None = None
            preview: PdfPreview | None = None
            for update in stream_job(lambda ctx: preview_pdf(str(path), dpi_value, ctx)):
                log += update.log
                error = update.error
                if isinstance(update.result, PdfPreview):
                    preview = update.result
            if error is not None or preview is None:
                yield f"**Failed:** {error}", [], [], log, None
                return
            yield (
                f"**{len(preview.pages)} page(s)** - pick the ones to convert, "
                "then press Convert.",
                preview.gallery(),
                preview.choices,
                log,
                preview,
            )

        preview_state = gr.State(None)
        convert_button.click(
            on_convert,
            inputs=[files, folder, merge, page_choice, preview_state, *option_widgets],
            outputs=[status, downloads, previews, log_box],
        )
        render_button.click(
            on_render_pages,
            inputs=[pdf_file, dpi],
            outputs=[status, page_grid, page_choice, log_box, preview_state],
        )
        cancel_button.click(
            lambda: (
                "**Stopping** - the current image is finished first."
                if request_cancel()
                else "Nothing is running."
            ),
            outputs=status,
        )
        models_button.click(
            run_init,
            inputs=[gpu, coreml_encoder],
            outputs=[status, log_box],
        )
        clear_button.click(lambda: "", outputs=log_box)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="homr-gui", description="A local web GUI for homr, an optical music recognition tool"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to listen on")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        allowed_paths=[str(STAGING_ROOT)],
        show_error=True,
        theme=_theme(),
        css=_CSS,
    )


if __name__ == "__main__":
    main()
