# Hosting the ONNX models on Hugging Face (for the browser build)

This document covers one thing only: **how to mirror the three ONNX models that
homr already uses onto the Hugging Face Hub, so that a client-side (GitHub Pages)
build can download them from the browser.**

It does not cover the port itself — see `todo.md` for that.

---

## 1. Why we need a new home at all

homr already downloads its models from a GitHub Release tag at runtime:

```
https://github.com/liebharc/homr/releases/download/onnx_checkpoints/<name>.zip
```

That works fine for the Python CLI (`homr/main.py:download_weights`). It does **not**
work for a browser, because release assets are served without CORS headers.

Verified against the live release asset:

```
$ curl -sSL -r 0-0 -D - -o NUL \
    https://github.com/liebharc/homr/releases/download/onnx_checkpoints/segnet_308-..._fp16.zip

HTTP/1.1 302 Found
Location: https://release-assets.githubusercontent.com/github-production-release-asset/...
HTTP/1.1 206 Partial Content
Content-Type: application/octet-stream
ETag: "0x8DE5F751DACDEA0"
Server: Windows-Azure-Blob/1.0 Microsoft-HTTPAPI/2.0
```

The final response — served from Azure Blob behind `release-assets.githubusercontent.com` —
has **no `Access-Control-Allow-Origin` header**. A `fetch()` from
`https://liebharc.github.io` is therefore blocked by the same-origin policy.
`onnxruntime-web` needs to `fetch()` the model bytes to build a session, so this is a
hard blocker, not something we can work around in JavaScript.

Hugging Face, by contrast, sets CORS headers on model files:

```
$ curl -sD - -o NUL -H 'Origin: https://liebharc.github.io' \
    https://huggingface.co/<owner>/<repo>/resolve/main/<file>.onnx

access-control-allow-origin: *
access-control-allow-headers: Content-Range, Content-Type, Content-Disposition, ETag
accept-ranges: bytes
content-type: application/octet-stream
```

Two things matter here for onnxruntime-web:

- `access-control-allow-origin` is present, so `fetch()` succeeds cross-origin.
- `accept-ranges: bytes` is present, so ranged/streamed reads work, and
  `ETag` / `X-Linked-Etag` are exposed, so we can do conditional requests.

Options we considered and rejected:

| Option | Why not |
| --- | --- |
| GitHub Releases | No CORS (above). |
| `cdn.jsdelivr.net/gh/...` | CORS is fine, but jsDelivr caps GitHub-backed files at 20 MB. All six models are larger. |
| Commit the models into the repo / `gh-pages` | Would work (same origin), but adds ~130 MB to every clone and bloats the repo permanently. |
| Cloudflare R2 / a CORS proxy | Works, but adds a bucket, credentials, and a second thing to keep alive. HF is the smaller operational burden. |

---

## 2. What needs to be uploaded

Three models, in fp32 and fp16. The names come straight out of the source of truth:

- `homr/segmentation/config.py` → `model_name = "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f"`
- `homr/transformer/configs.py` → `model_name = "pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6"`

| File | Role | Release zip | Raw `.onnx` |
| --- | --- | --- | --- |
| `segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx` | segmentation, fp32 | 50.75 MB | **54.66 MB** |
| `segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx` | segmentation, fp16 | 25.24 MB | **27.34 MB** |
| `encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx` | transformer encoder, fp32 | 46.05 MB | **50.41 MB** |
| `encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx` | transformer encoder, fp16 | 22.89 MB | **25.24 MB** |
| `decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx` | transformer decoder, fp32 | 35.78 MB | **45.12 MB** |
| `decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx` | transformer decoder, fp16 | 82.54 MB | **89.61 MB** |

Raw sizes are exact — obtained by reading each ZIP's central directory over a ranged
request, without downloading the files. Regenerate them with:

```
python scripts/hf_model_manifest.py
python scripts/hf_model_manifest.py --local hf-models   # also print SHA256 of local files
```

Note the release asset is named after the model **without** its extension:
`segnet_308-..._fp16.zip`, not `.onnx.zip`. That matches
`homr/main.py:download_weights`, which does `os.path.basename(model).split(".")[0] + ".zip"`.

Each asset contains exactly **one** entry — the bare `.onnx`, no nested directory —
so unzipping in step 2 below yields a flat directory.

**Two things do *not* need to be hosted — they already live in this repo:**

- `homr/transformer/tokenizer_rhythm.json` (2556 B)
- `homr/transformer/tokenizer_lift.json` (331 B)
- `homr/transformer/tokenizer_pitch.json` (1581 B)
- `homr/transformer/tokenizer_note.json` (275 B)

These get bundled into the web app at build time. (In practice the runtime token
vocabularies are rebuilt from source in `homr/transformer/vocabulary.py`; the JSON files
are only consulted by the training code.)

**fp16 is not automatically smaller.** The fp16 decoder is *twice* the size of the fp32
one (89.61 MB vs 45.12 MB) because half-precision weights barely compress in a ZIP.
Budget:

| Bundle | Total download |
| --- | --- |
| all fp32 | **150.2 MB** |
| segnet fp16 + encoder fp16 + decoder fp32 | **97.7 MB** |
| all fp16 | **292.4 MB** |

We plan to ship **fp16 segnet + fp16 encoder + fp32 decoder** (97.7 MB), and fall back
to the fp32 segnet/encoder on the single-threaded WASM path (see `todo.md`, Block 1).
97.7 MB per visitor is still a lot — this is why caching in the browser is not optional.

---

## 3. Create the repository

Pick a name — the model names are already long. Something like
`ngbcoder/Homr-onnx`.

1. Create a **public** model repo at <https://huggingface.co/new>.

   > It must be public, otherwise the browser needs a token. A token in client-side
   > JavaScript is a public token; there is no way to keep a secret in a static page.

2. Note that a model repo's default **visibility for the file tree** is fine as-is.
   We only need `resolve/` to be world-readable, which public repos give us.

3. Create a token with **`read`** scope only:
   <https://huggingface.co/settings/tokens> → *Fine-grained* → *Read access to contents
   of all public gated repos you can access*. It is needed **only for the upload**, not
   at runtime.

4. Authenticate locally (the `hf` CLI is not installed here yet; see step 1 below).

---

## 4. Upload the models

### Step 1 — install the CLI

`hf` is not currently on this machine. `uv` is, so the least invasive option is:

```powershell
uv tool install huggingface_hub
hf auth login          # paste the read token from step 3
```

A plain `pip install -U huggingface_hub` works just as well if you prefer a venv.

### Step 2 — download the `.onnx` files from the release

Each release asset is a ZIP containing **exactly one** `.onnx` at the top level — no
nested directory. Verified for the fp16 segnet asset: 1 entry,
`segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx`.

```powershell
$base = 'https://github.com/liebharc/homr/releases/download/onnx_checkpoints'
$dest = 'hf-models'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$names = @(
  'segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx',
  'segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx',
  'encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx',
  'encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx',
  'decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx',
  'decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx'
)

foreach ($name in $names) {
  $zip = Join-Path $dest "$name.zip"
  curl.exe -L -o $zip "$base/$name.zip"
  Expand-Archive -Path $zip -DestinationPath $dest -Force
  Remove-Item $zip
}

Get-ChildItem $dest -Filter *.onnx | Select-Object Name, Length
```

Expect six files. Verify the segnet file at minimum, since everything downstream
depends on it.

### Step 3 — sanity-check the models still load

Do not upload a file you have not round-tripped. The Python side is the reference:

```powershell
poetry run python -c "import onnxruntime as ort; s=ort.InferenceSession('hf-models/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx'); print(s.get_inputs()[0].name, s.get_inputs()[0].shape, s.get_outputs()[0].name)"
```

Expect `input` with shape `[batch_size, 3, 320, 320]` and output `output` — this is
what `homr/segmentation/inference_segnet.py:87-88` assumes.

### Step 4 — upload

```powershell
hf upload ngbcoder/Homr-onnx $dest \
  --repo-type model \
  --exclude "*.zip"
```

Or drive it file by file so progress is visible:

```powershell
foreach ($f in Get-ChildItem $dest -Filter *.onnx) {
  hf upload ngbcoder/Homr-onnx $f.FullName --repo-type model
}
```

`.onnx` files go to git-lfs automatically (the HF Hub is LFS-backed), so there is no
step to forget.

### Step 5 — verify CORS before writing any web code

This is the check that actually matters. Run it from a machine that is not in a
corporate proxy, and confirm `access-control-allow-origin` is present:

```powershell
curl.exe -sD - -o NUL -H 'Origin: https://liebharc.github.io' `
  'https://huggingface.co/ngbcoder/Homr-onnx/resolve/main/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx'
```

You are looking for:

```
access-control-allow-origin: *
accept-ranges: bytes
content-type: application/octet-stream
```

If `access-control-allow-origin` is missing, the upload did not land in a public repo.

---

## 5. Wiring it into the web app

Nothing in the Python package needs to change. The browser build gets its own manifest.

**`web/public/models.json`** — pinned by revision so a model swap is an explicit,
reviewable commit rather than a silent behaviour change:

```json
{
  "repo": "ngbcoder/Homr-onnx",
  "revision": "main",
  "models": {
    "segnet": {
      "fp16": "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx",
      "fp32": "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx"
    },
    "encoder": {
      "fp16": "encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx",
      "fp32": "encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx"
    },
    "decoder": {
      "fp16": "decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx",
      "fp32": "decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx"
    }
  }
}
```

**`web/src/models.ts`** — resolves a name + precision to a URL, and picks the
precision from the execution provider:

```ts
const RESOLVE = 'https://huggingface.co/{repo}/resolve/{revision}/{file}';

export function modelUrl(
  manifest: Manifest,
  key: 'segnet' | 'encoder' | 'decoder',
  fp16: boolean,
): string {
  const file = manifest.models[key][fp16 ? 'fp16' : 'fp32'];
  return RESOLVE
    .replace('{repo}', manifest.repo)
    .replace('{revision}', manifest.revision)
    .replace('{file}', file);
}
```

Rules for the rest of the model layer (tracked in `todo.md`, Block 2):

- **WebGPU → fp16**, **WASM → fp32.** The Python code already documents this: fp16
  models are *slower* on the CPU execution provider (`homr/main.py:349-354`), and the
  WASM provider on GitHub Pages is single-threaded because Pages cannot send the
  COOP/COEP headers needed for `SharedArrayBuffer`.
- **Cache aggressively.** ~88 MB per visitor is not acceptable to re-download on every
  page load. Store the bytes in the Cache Storage API keyed by the full URL, so a
  model swap (`revision` changes → URL changes) invalidates naturally.
- **Do not zip-unpack in the browser.** The release assets are ZIPs; HF serves the raw
  `.onnx`, which is what onnxruntime-web wants.

---

## 6. Maintaining the mirror

homr trains and re-exports these models. When a new checkpoint is published to the
`onnx_checkpoints` release tag:

1. Re-run step 2 and 3 from §4 against the new tag.
2. Copy the new model name into `homr/segmentation/config.py` and
   `homr/transformer/configs.py` (these drive the Python download path).
3. Upload to HF, then update `web/public/models.json` — filenames *and* `revision` —
   in the same commit.
4. Bump any segmentation cache key: `homr/segmentation/config.py` derives
   `segmentation_version` from the filename, which invalidates the `--cache` files.
5. Re-run the parity harness (`todo.md`, Block 6) before merging.

**Rule of thumb: the Python release tag stays the source of truth; HF is a mirror.**
Never fix a model in only one of the two places.

---

## 7. Troubleshooting

**`TypeError: Failed to fetch` in the browser console.**
Almost always CORS. Check the DevTools Network tab: a blocked request shows as CORS
with no status code, or as a 200 from `huggingface.co` that never reaches the page.
Re-run the §4 step 5 curl. If curl shows the header but the browser still fails, check
that you are not behind a proxy that strips `Origin`.

**`404` on `resolve/main/<file>`.**
The filename does not match. They are case-sensitive and include the full training
hash — copy them out of the download listing, do not retype them.

**`Invalid ONNX file` / session creation fails.**
Usually a truncated upload. HF's UI shows the file size per row; compare against
`scripts/hf_model_manifest.py` output. Re-upload with `hf upload --force`.

**Model loads but every prediction is garbage.**
The dtype/precision pairing is wrong — fp16 weights fed as fp32 inputs, or vice versa.
`homr/segmentation/inference_segnet.py:90-98` shows the binding: fp16 models take
`np.float16` input, fp32 models take `np.float32`.

**Session creation is very slow or the tab hangs.**
Expected on the WASM path. Use WebGPU where available and treat the WASM path as a
fallback. See the perf gate in `todo.md`, Block 1.