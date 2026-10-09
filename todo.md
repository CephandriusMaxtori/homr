# todo.md — client-side port of homr to GitHub Pages

Everything that still needs doing for a **fully client-side** (no server, no GPU box)
web build of homr that runs on GitHub Pages.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[!]` blocked

## Decisions already made

| Topic | Choice |
| --- | --- |
| Model hosting | Hugging Face Hub (public repo) — see `HuggingFaceModels.md` |
| Output fidelity | Parity with the Python reference, within a **documented tolerance** |
| Browser targets | WebGPU primary, single-threaded WASM fallback; both tested |
| Repo layout | Parallel `web/` tree; Python stays the reference implementation and keeps shipping the CLI, Docker and Gradio GUI |

## Key constraints discovered during investigation

Re-verify these before designing around them; each one invalidates work if it is wrong.

- [x] GitHub Release assets send **no** `Access-Control-Allow-Origin`, so a browser
      cannot fetch them. Confirmed by header dump. → hence Hugging Face.
- [x] `@techstark/opencv-js` 5.0.0 exports 1622 symbols and covers almost all of the
      `cv2` surface we need. **Missing: `Subdiv2D`** and the **`createCLAHE` factory**.
      The `CLAHE` class itself *is* exported.
- [x] GitHub Pages cannot send COOP/COEP, so `crossOriginIsolated` is false and
      `SharedArrayBuffer` is unavailable → onnxruntime-web's WASM backend is
      **single-threaded**. This is why WebGPU is the primary path.
- [x] The three models are already ONNX. No PyTorch conversion or re-export is needed.
- [x] Verified tensor interfaces (onnxruntime 1.30, CPU EP):

  | Model | Input | Output |
  | --- | --- | --- |
  | segnet | `input` `[batch, 3, 320, 320]`, **dynamic batch**, float32 or float16 | `output` `[batch, 6, 320, 320]` |
  | encoder | `input` `[1, 1, 256, 1280]`, **fixed** | `output` `[1, 1280, 512]` |
  | decoder | 39 inputs / 39 outputs | 39 outputs |

  Decoder details that the port depends on:
  - `rhythms`, `pitchs`, `lifts`, `articulations`, `slurs` → `tensor(int64)`, `[1, 1]`
  - `context` → float, `[1, 'cache_exists', 512]`
  - `cache_len` → `tensor(int64)`, `[1]`
  - `cache_in0..31` → `[1, 8, 'seq_len', 64]` (32 tensors = `decoder_depth 8` × 4)
  - logits: `out_rhythms` 260, `out_pitchs` 72, `out_lifts` 7, `out_positions` 5,
    `out_articulations` 62, `out_slurs` 5 — **these confirm the vocabulary sizes**
  - `attention` is `[2]` — the (x, y) pair written as `<!-- imgpos: … -->`
  - fp16 variant is fp16 on **all** float tensors including the cache
  - Note the decoder's *first* input is int64 in both variants, so a naive
    "is this graph fp16" probe over input types reports fp32 for the fp16 decoder
- [x] `fractions.Fraction` is load-bearing in the MusicXML generator (dict keys,
      `sorted()`, `min`/`max`) and needs a real BigInt implementation in TypeScript.
- [x] Three different rounding conventions are in play — Python banker's rounding,
      JS `Math.round` half-up, and OpenCV's `cvRound` half-away-from-zero. Getting
      this wrong silently shifts crops and pitches.
- [x] Dead code that does not need porting: `bounding_boxes._merge_groups_recursive`,
      `transformer/utils.py:softmax`, `bounding_boxes.create_lines` +
      `cv2.HoughLinesP` and all of `staff_position_save_load.py` (only reachable via
      `--read-staff-positions`), and `staff_dewarping.warp_image_randomly*`
      (training-only augmentation, the only use of `np.random`).
- [x] Zero RNG on the inference path. Title detection is the only threading
      (`ThreadPoolExecutor` + `Future`).

---

## Block 0 — Repo hygiene

- [x] Push the outstanding `main` commit to `origin/main`
- [x] Document the Hugging Face model mirror (`HuggingFaceModels.md`)
- [x] Add `scripts/hf_model_manifest.py` to report exact `.onnx` sizes and checksums
- [x] Create `web/` scaffold: Vite + TypeScript (strict) + Vitest + Playwright
- [x] Add `web/scripts/copy-wasm.mjs` (ORT wasm assets) and `rasterize-svg.mjs`
- [x] Update `.gitignore` for the web build
- [ ] Add `.github/workflows/web.yml` — build, test, deploy to `gh-pages`
- [ ] Add a "Web demo" section to `README.md`
- [ ] Add a `docs/web-parity.md` stub describing the tolerance policy

## Block 1 — Phase 0 spike (**go/no-go gate**) — ⚠️ PARTIAL

**Verdict: not established.** Findings in **`docs/spike-results.md`**.

Settled: model hosting works end to end, ONNX interfaces verified, ORT wiring fixed,
WASM fp32 confirmed running. **Not settled: whether this is fast enough in a browser.**
WebGPU rows failed to get an adapter; the WASM decoder failed on a bug in our own
harness. See "Open questions" in the spike doc.

- [x] Create the public HF repo → **`ngbcoder/Homr-onnx`**
- [x] Download + unzip all six `.onnx`; each loads in `onnxruntime` 1.30 and matches
      the expected interface. Segnet smoke test: `[1,3,320,320]` fp16 → `[1,6,320,320]`
- [x] Upload to HF, with a model card (`README.md` at the repo root)
- [x] **CORS verified** on all six files: `access-control-allow-origin: *`,
      `accept-ranges: bytes`, HTTP 206 on ranged GET, remote sizes match local
- [x] Write `web/public/models.json`, pinned by revision
      (revision `fbe8be58a31a38fc3e110b0c9c874053342b7e3b`)
- [x] Generate golden tensors from the Python reference
      (`scripts/dump_spike_tensors.py`): real 256×1280 staff canvases, encoder
      outputs, and reference decode step counts
- [x] Confirm **WASM fp32** runs: segnet 5.19 s/batch, encoder 2.11 s/forward
- [!] Benchmark **decoder** on WASM — blocked: `contextReduced` shape bug in our own
      harness (`bench.ts`), not a model problem. `context` is `[1, 'cache_exists', 512]`
      and the reduced slice must match the cache sequence length passed in the same step.
- [!] Benchmark **WebGPU** (fp16) — blocked: `requestAdapter()` returned `null` in the
      run. An earlier run did report `intel/gen-12lp`, so availability is unproven and
      the right headless flags are unsettled
- [x] Probe `findContours` return shape → out-param form, returns `void`
- [x] Probe `RotatedRect` / `MatVector` ergonomics → named-field object; `.size()`/`.get(i)`
- [!] Probe `cv.CLAHE` construction — class is present and constructible, but `apply()`
      did not round-trip a pixel value. Probe bug, not a known gap. **On the critical path.**
- [!] Probe `cv.reduce` — errored; `CV_16U` likely needs `data_u16`, not `data_u8`
- [!] Probe `boxPoints` — never reached; `minAreaRect` consumed the failure first
- [x] **Decision recorded:** WebGPU → fp16, WASM → fp32; fp16 segnet + fp16 encoder +
      fp32 decoder (97.7 MB); no int8 quantisation for now

Findings that changed the plan (all in `docs/spike-results.md`):

- **`cv.Subdiv2D` is absent** — still the last structural unknown. The dewarper needs a
  lattice-derived triangulation. Prototype this first in Block 4.
- **`minAreaRect` returns `{center, size, angle}`, not a tuple**, and geometry results
  are `MatVector`s that need `.delete()`. `bounding_boxes.py` destructures
  `box[0][0]` in ~40 places, so a wrapper type is required.
- **`findContours` uses out-params** and returns `void` — the 4.x JS form is gone.
- **`imread` / `imwrite` unavailable** (`imgcodecs` disabled) — use `<canvas>`.
- **The shipped OpenCV.js typings do not match the JS binding.** Probe at runtime, as
  `web/src/spike/opencv-probe.ts` does.
- **`ort.env.wasm.wasmPaths` must be absolute**, and must be a *directory prefix*
  rather than explicit `{wasm, mjs}` paths, or WebGPU is silently disabled.
- **All three ORT wasm variants must be served** (plain / jsep / asyncify) — 66 MB.
- **`io_binding` has no browser equivalent** — the port uses `Tensor` binding.
- **The decoder's first input is `int64` in both fp32 and fp16 variants**, so probing
  input types to detect precision reports the fp16 decoder as fp32.

## Block 2 — Phase 1: pure-logic port (~2,500 lines, low risk)

Delivers `EncodedSymbol[][] → MusicXML` with no image at all.

- [ ] `constants.ts`
- [ ] `point_mapping.ts` (`identity`, `undo_crop`, `undo_resize`, `chain`)
- [ ] `image_utils.ts`, `staff_regions.ts`
- [ ] `resize.ts` — PIL's `Image.resize` default is **bicubic**; preserve it
- [ ] `model.ts` — drop the `cv2.putText` debug drawing
- [ ] `fraction.ts` — BigInt-backed `Fraction` with hash/eq/ordering
- [ ] `vocabulary.ts` — token orders are baked into the ONNX logits; get them exact
      (rhythm 260, pitch 72, lift 7, articulation 62, slur 5, position 5; `pitch` is
      built `reversed`, `position = {".", upper, upper2, lower, lower2}`)
- [ ] `xml.ts` — an `xml.etree`-alike builder + serializer, matching the quirks:
      no pretty-printing, short empty elements rendered as `<dot />` **with a space**,
      attribute order = insertion order, text escapes `& < >` (not quotes),
      attribute escapes `& < > " \n \r \t`
- [ ] `music_xml_generator.ts`
- [ ] `relieur.ts` — multi-page merge
- [ ] Golden-token harness: a Python script dumps `EncodedSymbol[][]` + expected
      MusicXML to JSON; Vitest replays and byte-diffs
- [ ] Port the cases from `tests/test_music_xml_generator.py`

## Block 3 — Phase 2: onnxruntime-web layer

- [ ] `models.ts` — URL resolution from the manifest; Cache Storage keyed by URL so a
      revision bump invalidates naturally
- [ ] EP selection: WebGPU → fp16, WASM → fp32 (fp16 is *slower* on the CPU provider —
      see `homr/main.py:349-354`)
- [ ] Replace `io_binding` (`bind_cpu_input`, `run_with_iobinding`, `OrtValue`) with
      `Tensor` binding; none of that API exists in the browser
- [ ] `segnet.ts` — patch extraction with white padding, batching, `argmax`, `merge_patches`
- [ ] `tromr.ts` — encoder forward + greedy decoder with the 32 KV-cache tensors,
      `cache_len` bookkeeping, EOS handling
- [ ] No module-level singletons (`_segnet_inference`, `inference` in
      `staff_parsing_tromr.py`) — use an explicit context object, since a page can run
      several images
- [ ] Golden-tensor harness: Python dumps segnet I/O, encoder context and per-step
      decoder outputs as `.npz`; Vitest asserts within tolerance

## Block 4 — Phase 3: CV layer

Port in pipeline order. Mark each with the perf note attached.

- [ ] `cvwrap.ts` — typed wrapper over `@techstark/opencv-js`
- [ ] `pyround.ts` — audited `pyRound` (banker's), `jsRound` (half-up), `cvRound`
      (half-away-from-zero). Sites that will visibly break if wrong:
      `staff_parsing.py:257` (crop rect), `model.py:251` (**pitch**),
      `resize.py:18` (page height), `note_detection.py:40,41` (notehead centres),
      `staff_detection.py:443` (unit size)
- [ ] `autocrop.ts`
- [ ] `bounding_boxes.ts`
      ⚠ **rewrite**: `_merge_groups_optimized` is O(n²) with ~9 cv calls per pair —
      2000 boxes ⇒ ~18M JS↔WASM crossings. Pure-TS convex clip + spatial hash.
- [ ] `noise_filtering.ts`
      ⚠ **vectorise**: `create_grid` runs one `filter2D` per 20×20 tile (~13,440 per
      page) and `apply_noise_filter` ~27,000 draw calls into a throwaway debug image
- [ ] `bar_line_detection.ts`, `brace_dot_detection.ts`, `note_detection.ts`
- [ ] `find_peaks.ts`
      ⚠ `np.argsort(...)[::-1]` is **not** a stable descending sort — ties come out in
      reverse index order. `find_peaks` keeps peaks greedily in this order.
- [ ] `staff_detection.ts`
      ⚠ **rewrite**: `find_horizontal_lines` has a per-pixel Python loop building a
      numpy scalar at a time → use `cv.reduce`. `find_staff_anchors` is O(symbols × 6 ×
      fragments) with a `rotatedRectangleIntersection` per pair.
- [ ] `staff_dewarping.ts`
      ⚠ `cv2.Subdiv2D` is unavailable. The control points form a regular 3-row lattice
      (`calculate_span_and_optimal_points` + `add_image_edges_to_lines`), so build the
      triangulation from lattice topology instead of a general Delaunay.
      ⚠ `find_simplex` is O(points × triangles) and `inverse_transform_point` re-inverts
      the affine per candidate triangle — precompute inverses and add a uniform grid index.
      ⚠ This module deliberately uses **float32** throughout; per-site decisions needed
- [ ] `staff_parsing.ts`
      ⚠ `Staff.get_at` is a linear scan over ~200 grid points called from inner loops
      — the grid is x-sorted, so binary search
- [ ] `main.ts` — orchestration only (no argparse, no filesystem)

## Block 5 — Phase 4: output and UI

- [ ] MusicXML download
- [ ] Verovio WASM preview of the result (`verovio` npm package)
- [ ] PDF input via `pdfjs-dist`, replacing `pypdfium2`
- [ ] Drag-and-drop, multi-file selection
- [ ] Web Worker + staged progress reporting so the tab never freezes
- [ ] Segnet prediction cache in IndexedDB (replaces the `.npy` + `lzma` cache)
- [ ] Title detection — **deferred**; equivalent to the existing `--no-title` flag.
      RapidOCR/PP-OCR via onnxruntime-web is possible later

## Block 6 — Phase 5: parity and CI

- [ ] Python reference run over N validation images → reference MusicXML
      (candidates: `figures/tabi`, the `validation/` corpora, the released
      `smb_homr.db` / `polish-scores_homr.db` reference DBs)
- [ ] Playwright harness that drives the built page on the same images
- [ ] Compare via the repo's own scorer (`validation/ned_score.py`) and record the
      achieved NED in `docs/web-parity.md`
- [ ] Document the tolerance policy and any known divergences
- [ ] CI: lint, typecheck, unit tests, parity smoke test, deploy to `gh-pages`

## Open questions

- [!] Exact decoder step budget and ms/step on WebGPU — unknown until Block 1. This is
      the single number that decides whether the project is viable.
- [ ] Whether the WASM fallback is usable at all, or should be shipped as
      "best effort, expect minutes per page".
- [ ] Whether to quantise (e.g. int8) to shrink the 97.7 MB download, at the cost of
      accuracy and a deviation from the Python reference.
- [ ] Whether `createCLAHE` being unavailable (if `new cv.CLAHE` also fails) needs a
      hand-rolled CLAHE. It is on the critical path at `main.py:124` and `main.py:209`.