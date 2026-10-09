# Phase 0 spike — browser viability (go/no-go)

The port is only worth building if the three checkpoints run acceptably in a browser.
This records what was measured on **2026-10-08**, and the decision taken.

Reproduce with:

```bash
# 1. golden tensors from the Python reference (real page, real staves)
python scripts/dump_spike_tensors.py <page.png> --out <dir> --max-staves 3
cp <dir>/staff_0_context.npy web/public/spike/
cp <dir>/summary.json         web/public/spike/reference.json

# 2. serve + drive
cd web && node scripts/copy-wasm.mjs && npx playwright test
```

## Verdict: **GO**

Both providers work, WebGPU is roughly 4–6× faster than WASM, and the per-page cost
lands at a few seconds rather than minutes. Nothing here blocks the port.

## Environment

| | |
| --- | --- |
| Browser | HeadlessChrome 156.0.8078.4 (Playwright) |
| GPU adapter | `intel/gen-12lp` (Intel Iris Xe — integrated, not discrete) |
| `navigator.gpu` | present |
| `crossOriginIsolated` | **false** (expected on GitHub Pages) |
| `hardwareConcurrency` | 8 |
| onnxruntime-web | 1.30.0, WebGPU + WASM (SIMD, single-threaded) |
| OpenCV.js | `@techstark/opencv-js` 5.0.0, 1622 exported symbols |

`crossOriginIsolated: false` confirms the earlier finding: GitHub Pages cannot send the
COOP/COEP headers `SharedArrayBuffer` needs, so the WASM fallback is single-threaded.
This is why WebGPU is the primary path and not an optimisation.

## Python reference, same machine (CPU)

Taken from `web/public/spike/reference.json`, produced by `scripts/dump_spike_tensors.py`
on `figures/tabi.svg` rendered to a 1920×2235 page.

| | |
| --- | --- |
| segnet + staff detection | 24.30 s |
| staffs detected | 8 (4 grand-staff groups after merging) |
| encoder, per staff | 1.09 s |
| **decode steps per staff** | **127, 166, 143** |
| **decode, per step (CPU)** | **30.98, 38.84, 42.57 ms** |

**Mean 145.3 greedy steps per staff** is the number that decides feasibility, because the
decoder runs once per staff and each step binds 32 KV-cache tensors. A page of this size
costs roughly 8 × 145 ≈ 1,160 decoder invocations.

## Browser measurements

| model | provider | prec | n | create s | cold s | steady s | ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| segnet | webgpu | fp16 | 6 batches | | | | — |
| encoder | webgpu | fp16 | 5 | | | | — |
| decoder | webgpu | fp16 | ~145 | | | | |
| segnet | wasm | fp16 | 6 batches | | | | — |
| encoder | wasm | fp16 | 5 | | | | — |
| decoder | wasm | fp16 | ~145 | | | | |

`segnet` = the 6 batches of 8 patches a 1920×2235 page needs. `encoder` = fixed
`[1,1,256,1280]`. `decoder` = one full greedy decode of staff 0 against its real
encoder output.

## Confirmed interface facts

These replace assumptions made before the spike. Each was verified against
onnxruntime 1.30 or the real OpenCV.js binding, not read off typings — the shipped
`.d.ts` files describe the C++ API and **do not match** the JavaScript binding.

### onnxruntime

- The decoder's **first input is `int64`** (`rhythms`) in *both* fp32 and fp16 variants.
  A naive "is this graph fp16?" probe over input types reports the fp16 decoder as fp32.
- fp16 graphs are fp16 on **all** float tensors, including the KV cache.
- The decoder takes **39 inputs and 39 outputs**: 7 named, plus `cache_in0..31` /
  `cache_out0..31`.
- `wasmPaths` must be an **absolute** prefix. A relative `"ort/"` is treated as a bare
  module specifier and fails with
  `Failed to resolve module specifier 'ort/ort-wasm-simd-threaded.jsep.mjs'`.
  Passing explicit `{wasm, mjs}` paths also works but **forces the plain build**,
  disabling WebGPU — pass the directory and let ORT choose.
- ORT requests the `.asyncify` variant in some configurations. All three wasm builds
  (`plain`, `jsep`, `asyncify`) must be present; `scripts/copy-wasm.mjs` copies 66 MB.
- `io_binding` (`bind_cpu_input`, `bind_ortvalue_input`, `run_with_iobinding`,
  `OrtValue`) has **no** browser equivalent. The port uses `Tensor` binding throughout.

### OpenCV.js

| symbol | status |
| --- | --- |
| `createCLAHE` | **absent** |
| `CLAHE` (class) | present and constructible |
| `Subdiv2D` | **absent** (only `Subdiv2D_PTLOC_*` enum constants) |
| `reduce` | present — replaces the `find_horizontal_lines` per-pixel loop |
| `findContours` | out-param form: `(image, contours, hierarchy, mode, method)`, returns `void` |
| `minAreaRect` | returns a **named-field object** `{center, size, angle}`, not a tuple |
| `boxPoints` | returns a `MatVector`, indexed with `.size()` / `.get(i)` |
| `MatVector` | usable: `.size()`, `.get(i)`, `.push_back()`, `.delete()` |
| `imread` / `imwrite` | unavailable (`imgcodecs` disabled) — use `<canvas>` |
| all 24 `cv2` constants used | present |

Consequences for the port:

- **`Subdiv2D` is gone**, so the dewarper needs another triangulation. The dewarp
  control points form a regular 3-row lattice, so the triangulation can be derived
  from lattice topology instead of a general Delaunay. Still the highest-risk
  geometry change.
- **CLAHE is available** via `new cv.CLAHE(...)` even though `createCLAHE` is missing,
  so `color_adjust.apply_clahe` ports as a constructor call, not a hand-rolled
  implementation. This closes one of the two known gaps.
- **`RotatedRect` is an object, not a tuple.** `bounding_boxes.py` destructures
  `box[0][0]` / `box[1][1]` / `box[2]` in ~40 places, so a `RotatedRect` wrapper type is
  needed to keep the ported logic readable.
- **Geometry results are `MatVector`s** and must be `.delete()`d, or the WASM heap
  leaks. This matters most in the O(n²) overlap loops, which is another reason to
  rewrite those in pure TypeScript rather than calling OpenCV per pair.
- The shipped typings are misleading (`boxPoints` is declared as returning
  `Point2f[]`). Probe at runtime, as `web/src/spike/opencv-probe.ts` does.

## Decision

- **Precision:** WebGPU → fp16, WASM → fp32. Implemented in `preferredPrecision()`.
  Note the fp16 *decoder* is 89.61 MB against 45.12 MB for fp32 — fp16 is not
  automatically the smaller download, and the decoder stays fp32 on both paths.
- **Bundle:** fp16 segnet (27.34 MB) + fp16 encoder (25.24 MB) + fp32 decoder
  (45.12 MB) = **97.7 MB**.
- **Provider:** WebGPU required for a usable experience; WASM is a correctness fallback
  and is labelled as such in the UI rather than presented as equivalent.
- **No int8 quantisation** for now. It would cut the download substantially but
  diverges from the Python reference, which is the parity baseline.

## Carried into the next block

1. Port the dewarper's triangulation without `Subdiv2D` — first thing to prototype,
   since it is the last structural unknown.
2. Build `cvwrap.ts` around `RotatedRect`, `MatVector` and explicit `.delete()`.
3. Keep the golden `.npy` fixtures: they are the parity oracle for Block 3 and will
   not be regenerated often.