# Phase 0 spike — findings

What was **verified** while preparing the browser port, and what was **not**.

The point of this document is to replace assumptions with facts before writing the
port. It is not a benchmark report: the performance measurements were not completed,
and the sections that would contain them say so rather than estimating.

Verified on **2026-10-08**. Sections are labelled accordingly.

## Status: partially complete

| Area | State |
| --- | --- |
| Model hosting (Hugging Face) | ✅ verified end to end |
| Python reference metrics | ✅ measured |
| ONNX tensor interfaces | ✅ verified against onnxruntime 1.30 |
| onnxruntime-web wiring | ✅ verified (loads, runs) |
| OpenCV.js surface | ⚠️ partially verified — see below |
| Browser performance | ❌ **not measured** |
| Viability verdict | ❌ **not established** |

The go/no-go question is still open. What follows is the groundwork that makes
answering it cheap, not an answer to it.

## Verified: model hosting

The checkpoints are mirrored to <https://huggingface.co/ngbcoder/Homr-onnx> and serve
with CORS enabled, which the GitHub release assets do not. All six files verified:
`access-control-allow-origin: *`, `accept-ranges: bytes`, HTTP 206 on ranged GET,
remote sizes matching local.

See `HuggingFaceModels.md` for the full workflow.

## Verified: Python reference (CPU, this machine)

From `web/public/spike/reference.json`, produced by `scripts/dump_spike_tensors.py`
running the real pipeline over `figures/tabi.svg` rasterised to a 1920×2235 page.

| | |
| --- | --- |
| segnet + staff detection | 24.30 s |
| staffs detected | 8 (4 grand-staff groups after merging) |
| encoder, per staff | 1.09 s |
| **decode steps per staff** | **127, 166, 143** |
| **decode, per step (CPU)** | **30.98, 38.84, 42.57 ms** |

**Mean 145.3 greedy steps per staff.** This is the number the viability decision turns
on: the decoder runs once per staff, and each step binds 32 KV-cache tensors, so a page
of this size costs roughly 8 × 145 ≈ 1,160 decoder invocations. At the measured CPU
rate that would be ~40 s of decoding alone, which is why the browser measurement
matters.

Reference decode output for staff 0 begins:
`clef_G2, keySignature_4, timeSignature/8, note_2./C4/#, …`

## Verified: ONNX tensor interfaces

Confirmed by loading all six checkpoints in onnxruntime 1.30 (`scripts/validate_homr_models.py`).
Segnet smoke test: `[1,3,320,320]` fp16 → `[1,6,320,320]`.

| Model | Input | Output |
| --- | --- | --- |
| segnet | `input` `[batch, 3, 320, 320]`, **dynamic batch** | `output` `[batch, 6, 320, 320]` |
| encoder | `input` `[1, 1, 256, 1280]`, **fixed** | `output` `[1, 1280, 512]` |
| decoder | 39 inputs / 39 outputs | 39 outputs |

Decoder detail:

- `rhythms`, `pitchs`, `lifts`, `articulations`, `slurs` → `tensor(int64)`, `[1, 1]`
- `context` → float, `[1, 'cache_exists', 512]`
- `cache_len` → `tensor(int64)`, `[1]`
- `cache_in0..31` → `[1, 8, 'seq_len', 64]` (32 = `decoder_depth 8` × 4)
- logits: rhythm **260**, pitch **72**, lift **7**, position **5**, articulation **62**, slur **5**
  — these confirm the vocabulary sizes in `homr/transformer/vocabulary.py`
- `attention` is `[2]` — the (x, y) pair written as `<!-- imgpos: … -->`
- fp16 variants are fp16 on **all** float tensors, including the KV cache

⚠️ The decoder's **first input is `int64`** in *both* variants. A naive "is this graph
fp16?" probe over input types reports the fp16 decoder as fp32. This caused a false
failure during the spike.

## Verified: the decoder KV cache starts empty, not zero-filled

`decoder_inference.py:init_cache(cache_len=0)` allocates
`np.zeros((1, heads, 0, head_dim))` — a tensor with a **zero-length** sequence axis,
holding **zero elements**.

Feeding that shape a full `heads * headDim` buffer fails:

```
Tensor's size(0) does not match data length(512)
```

because `1 * 8 * 0 * 64` is 0, not 512. The data buffer has to be empty too. This was
the cause of the first decoder benchmark failure and is a real constraint on the port:
the step-0 cache is an empty tensor, not a zero-filled one.

Confirmed against the graph:

| tensor | declared shape |
| --- | --- |
| `context` | `[1, 'cache_exists', 512]` |
| `cache_in0` | `[1, 8, 'seq_len', 64]` |
| `cache_out0` | `[1, 8, 'seq_len + 1', 64]` |

So after step 0 the cache grows to `seq_len = 1`, and `cache_exists` (the context's
second axis) must agree with the cache length passed in the same step.

## Verified: WebGPU sessions must be created once and reused

The WebGPU execution provider permits only **one** session to be initialising at a
time. Two distinct failures follow from getting this wrong:

1. Releasing a session and immediately creating another trips
   `another WebGPU EP inference session is being created.`, because `release()` is
   asynchronous and teardown has not finished when the next `create()` begins.
2. If a session creation never settles, every subsequent attempt reports the same
   error — the original failure is masked.

The spike originally called `release()` after each benchmark, which is why three
sessions failed in a row after the first. Sessions are now cached by
`provider::url` and reused for the page lifetime. This is also what the app should do:
a page converts several images through the same three sessions, so re-creating them
per image would be wasteful.

## Verified: onnxruntime-web wiring

- `wasmPaths` must be an **absolute prefix**. A relative `"ort/"` is parsed as a bare
  module specifier and fails with
  `Failed to resolve module specifier 'ort/ort-wasm-simd-threaded.jsep.mjs'`.
- Passing explicit `{wasm, mjs}` paths **forces the plain build and silently disables
  WebGPU**. Pass the directory and let ORT select the variant.
- ORT requests the `.asyncify` variant in some configurations. All three wasm builds
  (`plain`, `jsep`, `asyncify`) must be served; `scripts/copy-wasm.mjs` copies 66 MB.
- `io_binding` (`bind_cpu_input`, `bind_ortvalue_input`, `run_with_iobinding`,
  `OrtValue`) has **no** browser equivalent. The port uses `Tensor` binding throughout.
- **WASM (fp32) runs.** Confirmed: segnet fp32 and encoder fp32 both completed and
  produced the expected output shapes.

## Verified: OpenCV.js surface

`@techstark/opencv-js` 5.0.0 exports **1622 symbols**. The shipped `.d.ts` files
describe the OpenCV **C++** API and do **not** match the JavaScript binding, so these
were probed at runtime by `web/src/spike/opencv-probe.ts` rather than read off typings.

Confirmed at runtime:

| symbol | status |
| --- | --- |
| `createCLAHE` | **absent** |
| `CLAHE` (class) | present, constructible via `new`; `getClipLimit()` returns the clip limit |
| `Subdiv2D` | **absent** (only `Subdiv2D_PTLOC_*` enum constants) |
| `findContours` | out-param form `(image, contours, hierarchy, mode, method)`, returns `void` |
| `minAreaRect` | returns a **named-field object** `{center, size, angle}`, not an indexable tuple |
| `MatVector` | usable: `.size()`, `.get(i)`, `.push_back()`, `.delete()` |
| `imread` / `imwrite` | unavailable (`imgcodecs` disabled) — use `<canvas>` |
| `boxPoints`, `reduce` | **not yet confirmed** — see below |

### OpenCV.js probe still to run

`web/src/spike/opencv-probe.ts` reports these, but the probe itself needs fixing first:

- `CLAHE.apply` reported a pixel of `-1`, meaning `data_u8` was not populated by that
  construction path. Likely a `matFromArray` vs `new Mat(...)` difference, not a missing
  capability — `createCLAHE` is confirmed absent, but whether `CLAHE` round-trips
  correctly is **unconfirmed**.
- `reduce` errored. `cv.reduce` is present in the export list, so this is a probe bug,
  most likely `CV_16U` needing `data_u16` rather than `data_u8`. **Unconfirmed.**
- `findContours` returned `contours.size() = 16` where 1 was expected, on a known
  single-square image. The out-param call shape is right, but the input handling is
  wrong (probably needs `threshold` first, or a non-inverted `CHAIN_APPROX_SIMPLE`
  source). **Unconfirmed.**
- `boxPoints` was never reached, because `minAreaRect` ran first and consumed the
  failure. The typings claim `Point2f[]`; the binding likely returns a `MatVector`.
  **Unconfirmed.**

None of these block the port's structure, but `CLAHE` in particular is on the critical
path (`main.py:124`, `main.py:209`), so its true status matters.

## Not measured: browser performance

No performance conclusion can be drawn. Two attempts were made; neither produced a
usable result.

**Attempt 1** — all providers failed to initialise:

```
no available backend found. ERR: [webgpu] TypeError: Failed to resolve module
specifier 'ort/ort-wasm-simd-threaded.jsep.mjs'
```

Cause: `wasmPaths` was set to a relative `"ort/"`. Fixed — see the verified section.
The `asyncify` variant was also missing from `public/ort/`; now copied.

**Attempt 2** — after the fix:

| model | provider | prec | result |
| --- | --- | --- | --- |
| segnet | webgpu | fp16 | **failed** — bare status `9943672`, no message |
| encoder | webgpu | fp16 | **failed** — `another WebGPU EP inference session is being created.` |
| decoder | webgpu | fp16 | **failed** — same |
| segnet | wasm | fp32 | 5.19 s/batch (8 patches of 320×320) |
| encoder | wasm | fp32 | 2.11 s/forward |
| decoder | wasm | fp32 | **not completed** |

All three WebGPU rows are **failures, not measurements**.

An adapter *is* obtained — `requestAdapter()` returns `intel/gen-12lp`, so
`navigator.gpu` and the adapter are both fine. The failure is in onnxruntime-web's WebGPU
execution provider initialising: session creation throws a bare `9943672` with no
message, and that creation never settles. Every later attempt then reports the
follow-on `another WebGPU EP inference session is being created.` error, which masks
the original cause.

The session-reuse fix (above) resolved the cascade, but the underlying initialisation
failure stands. **This looks like a headless-environment limitation** rather than a
model or manifest problem — a software or headless Chromium cannot always bring up the
WebGPU EP. It needs either a headed browser or a machine where the EP initialises, and
should not be treated as evidence that WebGPU is unusable in production.

The harness now falls back to WASM-only when no adapter is obtained, so runs are no
longer wasted, but the decoder WASM number is still **missing** — the run was
interrupted before finishing. That is the measurement the viability decision needs.

## Decisions that stand regardless of performance

These follow from file sizes and API shapes, not from benchmarks:

- **Bundle: fp16 segnet (27.34 MB) + fp16 encoder (25.24 MB) + fp32 decoder (45.12 MB)
  = 97.7 MB.** fp16 is not automatically smaller: the fp16 *decoder* is 89.61 MB against
  45.12 MB for fp32, so the decoder stays fp32 on both paths.
- **Precision pairing** (in `preferredPrecision()`): WebGPU → fp16, WASM → fp32. Note
  this is the same conclusion `homr/main.py` reaches for the CPU execution provider,
  where fp16 models are slower than fp32 ones.
- **No int8 quantisation** for now: it would cut the download substantially but
  diverges from the Python reference, which is the parity baseline.
- **`Subdiv2D` is absent**, so the dewarper needs a different triangulation. The dewarp
  control points form a regular 3-row lattice, so the triangulation can be derived from
  lattice topology rather than a general Delaunay. Still the highest-risk geometry
  change.
- **`RotatedRect` is an object, not a tuple.** `bounding_boxes.py` destructures
  `box[0][0]` / `box[1][1]` / `box[2]` in ~40 places, so a wrapper type is needed.
- **Geometry results are `MatVector`s** that must be `.delete()`d or the WASM heap
  leaks. Another reason to rewrite the O(n²) overlap loops in pure TypeScript instead of
  calling OpenCV per pair.

## Open questions

1. **[blocking]** Is WebGPU actually available and fast enough? Needs a run with a
   reliable adapter. The single WASM data point suggests the decoder is the risk.
2. **[blocking]** Fix the `contextReduced` shape bug so the decoder can be measured at
   all.
3. Does `cv.CLAHE` round-trip correctly, or does CLAHE need hand-rolling? It is on the
   critical path.
4. Confirm `cv.reduce` and `boxPoints` (probe bugs, not known gaps).
5. Is the single-threaded WASM fallback acceptable at all, or should it be presented as
   unsupported? Depends on question 1.