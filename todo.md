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
- [ ] Create `web/` scaffold: Vite + TypeScript (strict) + Vitest
- [ ] Add `.github/workflows/web.yml` — build, test, deploy to `gh-pages`
- [ ] Update `.gitignore` (`web/node_modules`, `web/dist`, `web/hf-models`, `*.local`)
- [ ] Add a "Web demo" section to `README.md`
- [ ] Add a `docs/web-parity.md` stub describing the tolerance policy

## Block 1 — Phase 0 spike (**go/no-go gate**)

Do not write the port until this passes. If decoder throughput is unusable, stop.

- [ ] Create the public HF repo (e.g. `liebharc/homr-onnx`), follow `HuggingFaceModels.md`
- [ ] Download + unzip all six `.onnx`; verify each loads in `onnxruntime`
- [ ] Upload to HF; verify CORS with the curl check in `HuggingFaceModels.md` §4 step 5
- [ ] Write `web/public/models.json`, pinned by revision
- [ ] Benchmark **segnet**: one full 1920×2800 page ≈ 54 patches at 320×320, batch 8
- [ ] Benchmark **encoder**: one 1×1×256×1280 forward
- [ ] Benchmark **decoder**: a full greedy decode of one staff, count the steps and
      measure ms/step (it needs 32 dynamic KV-cache tensors per step)
- [ ] Record all six numbers on **WebGPU** *and* **WASM**
- [ ] Confirm `new cv.CLAHE(1.0, 8, 8)` is constructible (only the factory is missing)
- [ ] Confirm whether `findContours` returns `(contours, hierarchy)` or just `contours`
- [ ] Confirm the `RotatedRect` / `MatVector` / `Mat` ergonomics and `Mat` lifetime rules
- [ ] Confirm `cv.reduce` handles the `find_horizontal_lines` histogram replacement
- [ ] **Decision:** default precision + EP per browser, written into the spike notes

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