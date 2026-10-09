# Web Port Parity & Policy

This document defines the output fidelity, tolerances, and design decisions for `homr-web` — the client-side browser port of `homr`.

## Goal

The browser port executes the complete OMR pipeline (image preprocessing, segmentation, staff detection, dewarping, TrOMR recognition, and MusicXML generation) entirely on the user's client device using WebGPU or WASM (`onnxruntime-web` and `@techstark/opencv-js`).

## Numerical Precision & Floating Point Conventions

1. **Model Weights**:
   - WebGPU execution path uses **fp16** ONNX models.
   - WASM CPU execution path uses **fp32** ONNX models.
2. **Rounding Behavior**:
   - Python uses banker's rounding (`round(x.5)` rounds to nearest even integer).
   - OpenCV C++ / WASM (`cvRound`) uses half-away-from-zero rounding.
   - JavaScript `Math.round` uses half-up rounding.
   - Exact rounding functions in `pyround.ts` ensure coordinate transformations, pitch calculations, and bounding box crops match Python reference within ±1 pixel.

## Tolerances

- **MusicXML Fidelity**: Pitch, rhythm, duration, clef, and measure grouping output match Python reference MusicXML output within identical or equivalent structural representations.
- **Bounding Boxes**: Minor pixel variations (1-2px) due to floating-point differences in OpenCV WASM vs native OpenCV C++ are acceptable as long as extracted staff crops capture identical symbol tokens.
