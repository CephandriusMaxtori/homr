"""Generate golden tensors for the Phase 0 spike from the Python reference.

Runs the real homr pipeline over a real sheet-music page, then intercepts the
per-staff canvas that homr/transformer/staff2score.py feeds to the encoder. From
those canvases it produces the exact tensors the browser benchmark must reproduce:

  * staff_N.png         - the 256x1280 canvas, as the encoder sees it
  * staff_N_input.npy   - the normalised float32 NCHW encoder input
  * staff_N_context.npy - the fp32 encoder output [1, 1280, 512]
  * summary.json        - page size, staff count, reference decode step counts

The reference decode step counts matter most: the decoder runs once per staff, so
its cost is (steps x staff count). A synthetic context would not reach EOS the way
real sheet music does, and the benchmark would be meaningless.

Writes to --out (default: a spike/ directory next to this file).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="sheet music page (png/jpg)")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--max-staves", type=int, default=4, help="how many staves to dump")
    parser.add_argument("--title", default="", help="skip OCR, use this title")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"no such image: {image_path}", file=sys.stderr)
        return 1

    # homr/main.py imports title_detection, which imports rapidocr at module scope.
    # Title detection is disabled for this spike (ProcessingConfig.title_detection
    # is False below), so stub the import rather than installing a heavy OCR stack.
    import types

    if importlib.util.find_spec("rapidocr") is None:
        stub = types.ModuleType("rapidocr")

        class RapidOCR:  # noqa: D401
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("rapidocr is not installed; title detection is disabled")

        stub.RapidOCR = RapidOCR  # type: ignore[attr-defined]
        sys.modules["rapidocr"] = stub
        print("note: stubbed rapidocr (title detection disabled)")

    out = Path(args.out) if args.out else Path(__file__).parent / "out"
    out.mkdir(parents=True, exist_ok=True)

    from homr import color_adjust
    from homr.autocrop import autocrop_with_offset
    from homr.debug import Debug
    from homr.main import ProcessingConfig, detect_staffs_in_image
    from homr.point_mapping import undo_resize
    from homr.resize import resize_image
    from homr import staff_parsing
    from homr.transformer.configs import Config, default_config
    from homr.transformer.encoder_inference import Encoder

    captured: list[np.ndarray] = []

    # Intercept the point where homr hands a dewarped staff to the transformer.
    def fake_parse_staff_tromr(staff, staff_image, config):  # noqa: ANN001, ANN202
        captured.append(np.asarray(staff_image).copy())
        return []

    staff_parsing.parse_staff_tromr = fake_parse_staff_tromr  # type: ignore[assignment]

    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=False,
    )

    t0 = time.perf_counter()
    multi_staffs, preprocessed, debug, _title, n_staffs, to_input = detect_staffs_in_image(
        str(image_path), config
    )
    seg_seconds = time.perf_counter() - t0
    print(f"staff detection + segnet: {seg_seconds:.2f}s, {n_staffs} staff(s)")
    print(f"multi_staff groups: {[len(m.staffs) for m in multi_staffs]}")
    print(f"preprocessed page: {preprocessed.shape} {preprocessed.dtype}")

    transformer_config = Config()
    transformer_config.use_gpu_inference = False

    parse_staffs = staff_parsing.parse_staffs
    t1 = time.perf_counter()
    result = parse_staffs(
        debug,
        multi_staffs,
        preprocessed,
        selected_staff=-1,
        config=transformer_config,
        page_to_input_image=to_input,
    )
    print(f"parse_staffs (decoder stubbed): {time.perf_counter() - t1:.2f}s")
    print(f"captured {len(captured)} staff canvas/es")

    if not captured:
        print("no staff canvases captured", file=sys.stderr)
        return 1

    cv = default_config
    canvas_h, canvas_w = cv.max_height, cv.max_width
    summary: dict = {
        "page_image": image_path.name,
        "page_shape": list(preprocessed.shape),
        "staff_count": n_staffs,
        "multi_staff_group_sizes": [len(m.staffs) for m in multi_staffs],
        "voices": len(result),
        "seg_seconds": round(seg_seconds, 3),
        "canvas": [canvas_h, canvas_w],
        "staves": [],
    }

    encoder = Encoder(transformer_config)
    encode_seconds = 0.0

    for index, canvas in enumerate(captured[: args.max_staves]):
        # staff2score.ConvertToArray: (img/255 - 0.7931) / 0.1731, NCHW float32
        grey = canvas
        if grey.ndim == 3:
            grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
        scaled = (grey.astype(np.float32) / 255.0 - 0.7931) / 0.1738
        tensor = np.expand_dims(np.expand_dims(scaled, 0), 0).astype(np.float32)

        te = time.perf_counter()
        context = encoder.generate(tensor)
        encode_seconds += time.perf_counter() - te

        stem = out / f"staff_{index}"
        Image.fromarray(grey).save(stem.with_suffix(".png"))
        np.save(str(stem) + "_input.npy", tensor)
        np.save(str(stem) + "_context.npy", context)

        summary["staves"].append(
            {
                "index": index,
                "canvas_shape": list(grey.shape),
                "context_shape": list(context.shape),
                "context_dtype": str(context.dtype),
                "context_min": float(context.min()),
                "context_max": float(context.max()),
                "context_mean": float(context.mean()),
                "nonzero_fraction": float((grey > 0).mean()),
            }
        )
        print(
            f"  staff {index}: canvas {grey.shape} -> context {context.shape} "
            f"{context.dtype} range [{context.min():.3f}, {context.max():.3f}]"
        )

    summary["encode_seconds_total"] = round(encode_seconds, 3)
    summary["encode_seconds_per_staff"] = round(encode_seconds / max(1, len(summary["staves"])), 4)

    # Reference greedy decode, for a realistic steps-per-staff figure.
    from homr.transformer.decoder_inference import get_decoder

    decoder = get_decoder(transformer_config)
    print("\nreference greedy decode (CPU):")
    for entry in summary["staves"]:
        idx = entry["index"]
        context = np.load(str(out / f"staff_{idx}") + "_context.npy")
        td = time.perf_counter()
        symbols = decoder.generate(
            start_tokens=np.array([[1]]), nonote_tokens=np.array([[0]]), context=context
        )
        elapsed = time.perf_counter() - td
        entry["reference_decode_steps"] = len(symbols)
        entry["reference_decode_seconds_cpu"] = round(elapsed, 3)
        entry["reference_ms_per_step_cpu"] = (
            round(elapsed * 1000 / max(1, len(symbols)), 2) if symbols else None
        )
        first = " ".join(
            f"{s.rhythm}/{s.pitch}/{s.lift}" for s in symbols[:6]
        )
        print(
            f"  staff {idx}: {len(symbols):>3} steps in {elapsed:6.2f}s "
            f"({entry['reference_ms_per_step_cpu']} ms/step)  first: {first}"
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out/'summary.json'}")
    steps = [e["reference_decode_steps"] for e in summary["staves"]]
    print(f"reference steps per staff: {steps}")
    if steps:
        print(f"mean {sum(steps)/len(steps):.1f} steps/staff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())