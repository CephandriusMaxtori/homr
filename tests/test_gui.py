import importlib.util
import io
import os
import sys
import tempfile
import threading
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Queue
from typing import Any
from unittest import mock

from homr.gui import (
    ConversionReport,
    GpuChoice,
    GuiOptions,
    JobContext,
    JobState,
    PdfPreview,
    _LogStream,
    _merge_xml,
    _new_staging_dir,
    _previews,
    _teaser_of,
    collect_inputs,
    request_cancel,
    resolve_gpu_flags,
    run_convert,
    stage_file,
    stream_job,
)
from homr.simple_logging import eprint


def _can_import_homr_main() -> bool:
    """Whether homr.main imports here, i.e. the inference extras are installed."""
    return (
        importlib.util.find_spec("onnxruntime") is not None
        and importlib.util.find_spec("rapidocr") is not None
    )


HAS_INFERENCE = _can_import_homr_main()


class TestResolveGpuFlags(unittest.TestCase):
    def test_auto_keeps_everything_on_the_cpu_without_a_gpu(self) -> None:
        flags = resolve_gpu_flags(
            GpuChoice.AUTO, cuda=False, rocm=False, coreml=False, coreml_encoder=False
        )

        self.assertFalse(flags.transformer)
        self.assertFalse(flags.segnet)
        self.assertFalse(flags.coreml_encoder)

    def test_auto_runs_the_transformer_on_cuda_and_rocm(self) -> None:
        for provider in ("cuda", "rocm"):
            with self.subTest(provider=provider):
                flags = resolve_gpu_flags(
                    GpuChoice.AUTO,
                    cuda=provider == "cuda",
                    rocm=provider == "rocm",
                    coreml=False,
                    coreml_encoder=False,
                )

                self.assertTrue(flags.transformer)
                self.assertTrue(flags.segnet)

    def test_auto_only_speeds_up_segnet_with_coreml(self) -> None:
        flags = resolve_gpu_flags(
            GpuChoice.AUTO, cuda=False, rocm=False, coreml=True, coreml_encoder=False
        )

        self.assertFalse(flags.transformer)
        self.assertTrue(flags.segnet)

    def test_auto_enables_the_coreml_encoder_only_without_cuda(self) -> None:
        enabled = resolve_gpu_flags(
            GpuChoice.AUTO, cuda=False, rocm=False, coreml=True, coreml_encoder=True
        )
        on_cuda = resolve_gpu_flags(
            GpuChoice.AUTO, cuda=True, rocm=False, coreml=True, coreml_encoder=True
        )

        self.assertTrue(enabled.coreml_encoder)
        self.assertFalse(on_cuda.coreml_encoder)

    def test_forced_cpu_ignores_an_available_gpu(self) -> None:
        flags = resolve_gpu_flags(
            GpuChoice.CPU, cuda=True, rocm=True, coreml=True, coreml_encoder=True
        )

        self.assertEqual(
            flags,
            resolve_gpu_flags(
                GpuChoice.AUTO, cuda=False, rocm=False, coreml=False, coreml_encoder=False
            ),
        )
        self.assertFalse(flags.coreml_encoder)

    def test_forced_gpu_ignores_the_missing_execution_provider(self) -> None:
        flags = resolve_gpu_flags(
            GpuChoice.GPU, cuda=False, rocm=False, coreml=False, coreml_encoder=True
        )

        self.assertTrue(flags.transformer)
        self.assertTrue(flags.segnet)
        self.assertFalse(flags.coreml_encoder)


class TestGuiOptions(unittest.TestCase):
    def test_defaults_match_the_command_line(self) -> None:
        options = GuiOptions()

        options.validate()
        self.assertTrue(options.title_detection)
        self.assertEqual(options.selected_staff, -1)
        self.assertEqual(options.gpu, GpuChoice.AUTO)

    def test_rejects_bpm_values_that_are_not_positive(self) -> None:
        for field in ("metronome", "tempo"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                if field == "metronome":
                    GuiOptions(metronome=0).validate()
                else:
                    GuiOptions(tempo=0).validate()

    def test_rejects_a_staff_number_below_the_all_staffs_marker(self) -> None:
        with self.assertRaises(ValueError):
            GuiOptions(selected_staff=-2).validate()

    @unittest.skipUnless(HAS_INFERENCE, "needs the inference extras (onnxruntime, rapidocr)")
    def test_maps_the_options_onto_the_processing_config(self) -> None:
        options = GuiOptions(
            debug=True,
            cache=True,
            title_detection=False,
            selected_staff=2,
            write_staff_positions=True,
            read_staff_positions=True,
            coreml_encoder=False,
        )
        flags = resolve_gpu_flags(
            GpuChoice.AUTO, cuda=True, rocm=False, coreml=False, coreml_encoder=False
        )

        config = options.to_processing_config(flags)

        self.assertTrue(config.enable_debug)
        self.assertTrue(config.enable_cache)
        self.assertTrue(config.write_staff_positions)
        self.assertTrue(config.read_staff_positions)
        self.assertEqual(config.selected_staff, 2)
        self.assertTrue(config.transformer_use_gpu)
        self.assertTrue(config.segnet_use_gpu)
        self.assertFalse(config.coreml_encoder)
        self.assertFalse(config.title_detection)


class TestLogStream(unittest.TestCase):
    def test_splits_writes_into_lines(self) -> None:
        queue: Queue[str] = Queue()
        stream = _LogStream(io.StringIO(), queue)

        stream.write("first line\nsecond line\n")

        self.assertEqual(["first line", "second line"], list(queue.queue))
        self.assertEqual("", stream._pending)

    def test_carriage_returns_end_a_line_for_the_download_progress(self) -> None:
        queue: Queue[str] = Queue()
        stream = _LogStream(io.StringIO(), queue)

        stream.write("Downloaded 1 MB\rDownloaded 2 MB\r")

        self.assertEqual(["Downloaded 1 MB", "Downloaded 2 MB"], list(queue.queue))

    def test_keeps_an_incomplete_line_until_it_is_finished(self) -> None:
        queue: Queue[str] = Queue()
        stream = _LogStream(io.StringIO(), queue)

        stream.write("Processing page")
        self.assertEqual([], list(queue.queue))

        stream.write(" 1 of 4\n")
        self.assertEqual(["Processing page 1 of 4"], list(queue.queue))

    def test_drops_blank_lines(self) -> None:
        queue: Queue[str] = Queue()
        stream = _LogStream(io.StringIO(), queue)

        stream.write("\n\n  \r\nsomething\n")

        self.assertEqual(["something"], list(queue.queue))

    def test_passes_everything_on_to_the_original_stream(self) -> None:
        original = io.StringIO()
        stream = _LogStream(original, Queue())

        written = stream.write("visible in the terminal\n")

        self.assertEqual(original.getvalue(), "visible in the terminal\n")
        self.assertEqual(len("visible in the terminal\n"), written)

    def test_flush_pending_emits_a_trailing_partial_line(self) -> None:
        queue: Queue[str] = Queue()
        stream = _LogStream(io.StringIO(), queue)

        stream.write("no newline at the end")
        stream.flush_pending()
        stream.flush_pending()

        self.assertEqual(["no newline at the end"], list(queue.queue))

    def test_is_a_writable_stream_that_cannot_be_read(self) -> None:
        stream = _LogStream(io.StringIO(), Queue())

        self.assertTrue(stream.writable())
        self.assertFalse(stream.readable())

    def test_survives_being_used_as_the_sys_stderr(self) -> None:
        queue: Queue[str] = Queue()
        stream = _LogStream(io.StringIO(), queue)
        previous = sys.stderr
        sys.stderr = stream
        try:
            eprint("through stderr")
        finally:
            sys.stderr = previous

        self.assertEqual(["through stderr"], list(queue.queue))


class TestStreamJob(unittest.TestCase):
    def test_reports_the_log_of_the_job_and_its_result(self) -> None:
        def job(ctx: JobContext) -> str:
            ctx.log("first")
            ctx.update("halfway", 1, 2)
            ctx.log("second")
            return "result"

        updates = list(stream_job(job, poll_interval=0.01))

        self.assertTrue(updates[-1].done)
        self.assertEqual("result", updates[-1].result)
        self.assertIsNone(updates[-1].error)
        log = "".join(update.log for update in updates)
        self.assertIn("first", log)
        self.assertIn("second", log)
        self.assertEqual("halfway", updates[-1].state.message)
        self.assertEqual(JobState(step=1, total=2, message="halfway"), updates[-1].state)

    def test_reports_a_failing_job_instead_of_raising(self) -> None:
        def job(ctx: JobContext) -> str:
            raise RuntimeError("no staffs found")

        updates = list(stream_job(job, poll_interval=0.01))

        error = updates[-1].error
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual("no staffs found", str(error))
        self.assertIsNone(updates[-1].result)

    def test_restores_stderr_and_releases_the_job_lock(self) -> None:
        before = sys.stderr

        list(stream_job(lambda ctx: None, poll_interval=0.01))

        self.assertIs(sys.stderr, before)
        # A second job must be possible, i.e. the lock was released.
        self.assertEqual("ok", list(stream_job(lambda ctx: "ok", poll_interval=0.01))[-1].result)

    def test_refuses_a_second_job_while_one_runs(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow(ctx: JobContext) -> None:
            started.set()
            release.wait(timeout=5)

        def second(ctx: JobContext) -> None:
            return None

        updates: Iterator[object] = stream_job(slow, poll_interval=0.01)
        next(updates)
        self.assertTrue(started.wait(timeout=5))
        try:
            with self.assertRaises(RuntimeError):
                list(stream_job(second, poll_interval=0.01))
        finally:
            release.set()
            list(updates)


class TestCancel(unittest.TestCase):
    def test_cancel_reaches_the_running_job(self) -> None:
        seen: list[bool] = []
        started = threading.Event()

        def job(ctx: JobContext) -> None:
            started.set()
            while not ctx.cancel.wait(timeout=0.01):
                pass
            seen.append(ctx.cancel.is_set())

        updates = stream_job(job, poll_interval=0.01)
        next(updates)
        self.assertTrue(started.wait(timeout=5))

        self.assertTrue(request_cancel())
        list(updates)
        self.assertEqual([True], seen)

    def test_cancel_without_a_job_is_reported(self) -> None:
        self.assertFalse(request_cancel())


class TestCollectInputs(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "page.png").write_bytes(b"")
        (self.root / "score.pdf").write_bytes(b"")
        (self.root / "score_notes.txt").write_text("ignored")

    def test_keeps_the_chosen_files_in_order(self) -> None:
        png = str(self.root / "page.png")
        pdf = str(self.root / "score.pdf")

        self.assertEqual([pdf, png], collect_inputs([pdf, png]))

    @unittest.skipUnless(HAS_INFERENCE, "the folder scan comes from homr.main")
    def test_expands_a_folder_and_skips_the_debug_files(self) -> None:
        teaser = self.root / "page_teaser.png"
        teaser.write_bytes(b"")

        found = collect_inputs([], str(self.root))

        self.assertEqual(
            sorted([str(self.root / "page.png"), str(self.root / "score.pdf")]), sorted(found)
        )
        self.assertNotIn(str(teaser), found)

    def test_a_file_is_only_used_once(self) -> None:
        png = str(self.root / "page.png")

        with mock.patch("homr.gui._folder_images", return_value=[png]):
            self.assertEqual([png], collect_inputs([png], str(self.root)))

    def test_a_folder_among_the_files_is_expanded(self) -> None:
        pdf = str(self.root / "score.pdf")

        with mock.patch("homr.gui._folder_images", return_value=[pdf]):
            self.assertEqual([pdf], collect_inputs([str(self.root)]))

    def test_rejects_a_path_that_does_not_exist(self) -> None:
        with self.assertRaises(ValueError):
            collect_inputs([str(self.root / "missing.png")])

    def test_rejects_a_folder_that_does_not_exist(self) -> None:
        with self.assertRaises(ValueError):
            collect_inputs([], str(self.root / "missing"))

    def test_ignores_empty_entries(self) -> None:
        self.assertEqual([], collect_inputs(["", "  "], ""))


class TestStaging(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_stages_a_copy_of_every_result(self) -> None:
        source = Path(self.folder.name) / "page.musicxml"
        source.write_text("<score-partwise/>")
        staging = _new_staging_dir()

        staged = Path(stage_file(str(source), staging, 7))

        self.assertEqual("007_page.musicxml", staged.name)
        self.assertEqual("<score-partwise/>", staged.read_text())
        self.assertTrue(staged.is_relative_to(staging))

    def test_every_run_gets_its_own_directory(self) -> None:
        first, second = _new_staging_dir(), _new_staging_dir()

        self.assertNotEqual(first, second)

    def test_finds_the_teaser_next_to_the_result(self) -> None:
        folder = Path(self.folder.name)
        xml = folder / "page.musicxml"
        xml.write_text("<score-partwise/>")

        self.assertIsNone(_teaser_of(str(xml)))

        teaser = folder / "page_teaser.png"
        teaser.write_bytes(b"")
        self.assertEqual(str(teaser), _teaser_of(str(xml)))


class TestMergeXml(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def _part(self, name: str, note: str) -> str:
        path = self.root / f"{name}.musicxml"
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<score-partwise version="4.0"><part-list>'
            f'<score-part id="P{note}"/>'
            '</part-list><part id="P1"><measure number="1"/></part></score-partwise>'
        )
        return str(path)

    def test_merges_into_one_file_next_to_the_inputs(self) -> None:
        first = self._part("page1", "1")
        second = self._part("page2", "2")
        ctx = JobContext(cancel=threading.Event())

        merged = _merge_xml([first, second], "score", "", ctx)

        self.assertEqual([str(self.root / "score_merged.musicxml")], merged)
        self.assertTrue(os.path.exists(merged[0]))
        self.assertFalse(os.path.exists(first))
        self.assertFalse(os.path.exists(second))

    def test_merges_into_the_output_folder(self) -> None:
        destination = self.root / "out"
        first = self._part("page1", "1")
        second = self._part("page2", "2")
        ctx = JobContext(cancel=threading.Event())

        merged = _merge_xml([first, second], "score", str(destination), ctx)

        self.assertEqual([str(destination / "score_merged.musicxml")], merged)
        self.assertTrue(os.path.exists(merged[0]))

    def test_a_single_file_is_left_alone(self) -> None:
        only = self._part("page1", "1")

        self.assertEqual([only], _merge_xml([only], "score", "", JobContext(threading.Event())))


class TestRunConvert(unittest.TestCase):
    """The generator behind the Convert button, with the pipeline stubbed out."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def _fake_conversion(
        self, results: ConversionReport
    ) -> Callable[[Any, GuiOptions, JobContext], ConversionReport]:
        def run_conversion(inputs: Any, options: GuiOptions, ctx: JobContext) -> ConversionReport:
            ctx.log("Processing " + str(inputs[0]))
            return results

        return run_conversion

    def test_yields_the_status_and_the_downloads_of_the_result(self) -> None:
        written = self.root / "page.musicxml"
        written.write_text("<score-partwise/>")
        staged = self.root / "staged.musicxml"
        staged.write_text("<score-partwise/>")
        report = ConversionReport(outputs=[str(written)], downloads=[str(staged)])

        with mock.patch("homr.gui.run_conversion", self._fake_conversion(report)):
            updates = list(run_convert(["page.png"], GuiOptions()))

        status, downloads, previews, log = updates[-1]
        self.assertIn("Wrote 1 MusicXML file(s).", status)
        self.assertEqual([str(staged)], downloads)
        self.assertEqual([], previews)
        self.assertIn("Processing page.png", log)

    def test_reports_a_failing_job_without_downloads(self) -> None:
        def failing(inputs: Any, options: GuiOptions, ctx: JobContext) -> ConversionReport:
            raise RuntimeError("No noteheads found")

        with mock.patch("homr.gui.run_conversion", failing):
            updates = list(run_convert(["page.png"], GuiOptions()))

        status, downloads, previews, _ = updates[-1]
        self.assertIn("No noteheads found", status)
        self.assertEqual([], downloads)
        self.assertEqual([], previews)

    def test_previews_the_teaser_of_every_result(self) -> None:
        report = ConversionReport(
            outputs=["a.musicxml"], downloads=["a.musicxml"], previews=["001_a_teaser.png"]
        )

        self.assertEqual([("001_a_teaser.png", "001_a_teaser.png")], _previews(report))


class TestPdfPreview(unittest.TestCase):
    """The PDF page picker: render, label and pick pages."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def _preview(self, count: int = 3) -> PdfPreview:
        pages = []
        for index in range(count):
            path = self.root / f"page-{index + 1:03d}.jpg"
            path.write_bytes(b"")
            pages.append(str(path))
        return PdfPreview(source="score.pdf", pages=pages)

    def test_labels_the_pages_in_order(self) -> None:
        preview = self._preview(3)

        self.assertEqual(["Page 1", "Page 2", "Page 3"], [c[0] for c in preview.choices])
        self.assertEqual(preview.pages, [c[1] for c in preview.choices])
        self.assertEqual(["Page 1", "Page 2", "Page 3"], [c[1] for c in preview.gallery()])

    def test_an_empty_selection_means_every_page(self) -> None:
        preview = self._preview()

        self.assertEqual(preview.pages, preview.selected_pages([]))
        self.assertEqual(preview.pages, preview.selected_pages(None))

    def test_only_the_ticked_pages_are_converted(self) -> None:
        preview = self._preview()

        self.assertEqual([preview.pages[1]], preview.selected_pages([preview.pages[1]]))

    def test_the_selection_is_ordered_like_the_document(self) -> None:
        preview = self._preview()
        picked = [preview.pages[2], preview.pages[0]]

        self.assertEqual([preview.pages[0], preview.pages[2]], preview.selected_pages(picked))

    def test_a_selection_from_another_preview_is_ignored(self) -> None:
        preview = self._preview()
        other = str(self.root / "elsewhere.jpg")
        Path(other).write_bytes(b"")

        self.assertEqual([], preview.selected_pages([other]))


class TestConversionReport(unittest.TestCase):
    def test_summarizes_a_successful_run(self) -> None:
        report = ConversionReport(outputs=["a.musicxml", "b.musicxml"])

        self.assertEqual("Wrote 2 MusicXML file(s).", report.summary())
        self.assertEqual("**Wrote 2 MusicXML file(s).**", report.as_markdown())

    def test_lists_the_files_that_failed(self) -> None:
        report = ConversionReport(
            outputs=["a.musicxml"], failures=["**page2.png**: No noteheads found"]
        )

        self.assertIn("1 MusicXML file(s) written, 1 file(s) failed.", report.summary())
        self.assertIn("page2.png", report.as_markdown())

    def test_reports_an_empty_run(self) -> None:
        self.assertEqual("Nothing was converted.", ConversionReport().summary())


if __name__ == "__main__":
    unittest.main()
