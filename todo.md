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
      JS `Math.round` half-up, and OpenCV's `cvRound` half-away-from-zero.

---

## Block 0 — Repo hygiene

- [x] Push the outstanding `main` commit to `origin/main`
- [x] Document the Hugging Face model mirror (`HuggingFaceModels.md`)
- [x] Add `scripts/hf_model_manifest.py` to report exact `.onnx` sizes and checksums
- [x] Create `web/` scaffold: Vite + TypeScript (strict) + Vitest + Playwright
- [x] Add `web/scripts/copy-wasm.mjs` (ORT wasm assets) and `rasterize-svg.mjs`
- [x] Update `.gitignore` for the web build
- [x] Add `.github/workflows/web.yml` — build, test, deploy to `gh-pages`
- [x] Add a "Web demo" section to `README.md`
- [x] Add a `docs/web-parity.md` stub describing the tolerance policy

## Block 1 — Phase 0 spike

- [x] Create the public HF repo → **`ngbcoder/Homr-onnx`**
- [x] Download + unzip all six `.onnx`; each loads in `onnxruntime` 1.30 and matches
      the expected interface.
- [x] Upload to HF, with a model card (`README.md` at the repo root)
- [x] **CORS verified** on all six files
- [x] Write `web/public/models.json`, pinned by revision
- [x] Generate golden tensors from the Python reference (`scripts/dump_spike_tensors.py`)
- [x] Confirm **WASM fp32** runs
- [x] Probe `findContours` return shape
- [x] Probe `RotatedRect` / `MatVector` ergonomics
- [x] **Decision recorded:** WebGPU → fp16, WASM → fp32

## Block 2 — Phase 1: pure-logic port

- [x] `constants.ts`
- [x] `fraction.ts` — BigInt-backed `Fraction` with hash/eq/ordering
- [x] `vocabulary.ts` — token orders and vocabularies
- [x] `xml.ts` — an `xml.etree`-alike builder + serializer
- [x] `music_xml_generator.ts` — complete MusicXML generator port
- [x] `relieur.ts` — multi-page merge
- [x] Unit tests in `web/tests/` (`fraction.test.ts`, `xml.test.ts`, `music_xml_generator.test.ts`)

## Block 3 — Phase 2: onnxruntime-web layer

- [x] `models.ts` — URL resolution from the manifest
- [x] `model-cache.ts` — Cache Storage API model persistence
- [x] `segnet.ts` — patch extraction with white padding, batching, `argmax`, `merge_patches`
- [x] `tromr.ts` — encoder forward + greedy decoder with 32 KV-cache tensors, `cache_len` bookkeeping, EOS handling

## Block 4 — Phase 3: CV layer

- [x] `cvwrap.ts` — typed wrapper over `@techstark/opencv-js`
- [x] `pyround.ts` — audited `pyRound` (banker's), `jsRound` (half-up), `cvRound`
- [x] `bounding_boxes.ts` — pure-TS spatial hash & union-find bounding box group merging
- [x] `staff_detection.ts` — horizontal line detection via `cv.reduce`, staff anchor finding, resampling
- [x] `staff_dewarping.ts` — lattice-derived triangulation and perspective warp
- [x] `pipeline.ts` — end-to-end pipeline orchestrator

## Block 5 — Phase 4: output and UI

- [x] MusicXML download
- [x] Drag-and-drop file upload
- [x] Staged progress bar & status updates
- [x] MusicXML output preview container

## Block 6 — Phase 5: parity and CI

- [x] CI workflow `.github/workflows/web.yml` running typecheck, unit tests, build, and deploying to GitHub Pages
