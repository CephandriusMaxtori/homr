"""Print exact sizes and SHA256 checksums for the ONNX checkpoints we mirror to Hugging Face.

The GitHub release assets are ZIPs; the browser fetches the raw ``.onnx`` from the
Hugging Face Hub. This script reports the size and digest of the files that actually
get uploaded, so ``web/public/models.json`` and the caching layer can be checked
against something concrete.

It works without any models being downloaded: the script reads each release ZIP's
central directory over a ranged HTTP request instead of pulling the whole file.

See ``HuggingFaceModels.md`` for the surrounding workflow.

Usage:
    python scripts/hf_model_manifest.py
    python scripts/hf_model_manifest.py --local hf-models   # also hash files on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

RELEASE_BASE = (
    "https://github.com/liebharc/homr/releases/download/onnx_checkpoints"
)

# Kept in sync with homr/segmentation/config.py and homr/transformer/configs.py.
MODEL_FILES = (
    "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
    "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx",
    "encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx",
    "encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx",
    "decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx",
    "decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6_fp16.onnx",
)

# GitHub release assets answer 404 to HEAD but honour ranged GET.
RANGE_HEADER = "Range"
EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_DIR_ENTRY = 46
TAIL_BYTES = 70_000


@dataclass
class RemoteEntry:
    name: str
    compressed_size: int
    uncompressed_size: int


class ReleaseAssetError(RuntimeError):
    pass


def _get(url: str, start: int | None = None, end: int | None = None) -> tuple[bytes, int]:
    """Fetch a byte range and return (payload, total_size)."""
    request = urllib.request.Request(url, headers={"User-Agent": "homr-hf-manifest"})
    if start is not None:
        end = start if end is None else end
        request.add_header(RANGE_HEADER, f"bytes={start}-{end}")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            total = int(payload.__len__())
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                total = int(content_range.rsplit("/", 1)[1])
            return payload, total
    except urllib.error.HTTPError as e:
        raise ReleaseAssetError(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise ReleaseAssetError(f"network error for {url}: {e.reason}") from e


def read_zip_entry_size(url: str) -> RemoteEntry:
    """Return the single entry of a release ZIP without downloading it.

    Reads the trailing 70 KB, locates the end-of-central-directory record, then reads
    the central directory itself. Each release asset holds exactly one ``.onnx``.
    """
    head, total = _get(url, 0, 0)
    if not head:  # pragma: no cover - defensive
        raise ReleaseAssetError(f"empty response for {url}")

    tail_start = max(0, total - TAIL_BYTES)
    tail, _ = _get(url, tail_start, total - 1)

    eocd = tail.rfind(EOCD_SIGNATURE)
    if eocd < 0:
        raise ReleaseAssetError(f"no end-of-central-directory record in {url}")

    entry_count = struct.unpack_from("<H", tail, eocd + 10)[0]
    cd_size = struct.unpack_from("<I", tail, eocd + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, eocd + 16)[0]

    if entry_count != 1:
        raise ReleaseAssetError(f"expected 1 entry in {url}, found {entry_count}")

    cd_start = cd_offset - tail_start
    if cd_start < 0:
        cd, _ = _get(url, cd_offset, cd_offset + cd_size - 1)
        base = cd_offset
    else:
        cd = tail[cd_start : cd_start + cd_size]
        base = tail_start

    compressed = struct.unpack_from("<I", cd, 20)[0]
    uncompressed = struct.unpack_from("<I", cd, 24)[0]
    name_len = struct.unpack_from("<H", cd, 28)[0]
    name = cd[CENTRAL_DIR_ENTRY : CENTRAL_DIR_ENTRY + name_len].decode("utf-8")

    return RemoteEntry(name=name, compressed_size=compressed, uncompressed_size=uncompressed)


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _mb(value: int) -> str:
    return f"{value / (1024 * 1024):8.2f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        metavar="DIR",
        help="also hash .onnx files in this directory (after downloading + unzipping)",
    )
    args = parser.parse_args()

    print(f"{'file':<74} {'zip':>11} {'onnx':>11}")
    print("-" * 98)

    manifest = {}
    failures = []
    for name in MODEL_FILES:
        # Matches homr/main.py:download_weights, which splits on "." and appends ".zip":
        #   base_name = os.path.basename(model).split(".")[0]  ->  base_name + ".zip"
        zip_name = name.removesuffix(".onnx") + ".zip"
        url = f"{RELEASE_BASE}/{zip_name}"
        try:
            entry = read_zip_entry_size(url)
        except ReleaseAssetError as e:
            print(f"{name:<74} {'ERROR':>11}   {e}")
            failures.append(name)
            continue

        if entry.name != name:
            print(f"{name:<74} {'MISMATCH':>11}   zip contains {entry.name!r}")
            failures.append(name)
            continue

        manifest[name] = entry.uncompressed_size
        print(f"{name:<74} {_mb(entry.compressed_size)} {_mb(entry.uncompressed_size)}")

    fp32 = [n for n in MODEL_FILES if not n.endswith("_fp16.onnx")]
    # The download sweet spot: half precision for the conv-heavy models, fp32 for the
    # decoder (the fp16 decoder is a larger file and no faster on the WASM path).
    mixed = [n for n in MODEL_FILES if n.endswith("_fp16.onnx") and "decoder" not in n]
    mixed += [n for n in MODEL_FILES if n.startswith("decoder") and not n.endswith("_fp16.onnx")]

    def total_of(names: list[str]) -> str:
        known = [manifest[n] for n in names if n in manifest]
        if len(known) != len(names):
            return "  (incomplete)"
        return f"{sum(known) / (1024 * 1024):.1f} MB"

    print()
    print(f"all fp32                        : {total_of(fp32)}")
    print(f"segnet+encoder fp16, decoder fp32: {total_of(mixed)}")
    print(f"all six                         : {total_of(list(MODEL_FILES))}")

    if args.local:
        print()
        print("local checksums:")
        for name in MODEL_FILES:
            path = os.path.join(args.local, name)
            if os.path.isfile(path):
                print(f"{name:<74} {sha256_of(path)}")
            else:
                print(f"{name:<74} missing from {args.local}")

    if failures:
        print()
        print(f"failed for {len(failures)} file(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())