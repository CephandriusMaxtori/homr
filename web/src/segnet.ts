/**
 * Segnet inference runner for image segmentation.
 */

import type { InferenceSession, Tensor } from "onnxruntime-web";
import ort from "./ort.ts";

export interface SegnetLayers {
  staff: Uint8Array;
  symbols: Uint8Array;
  stemsRests: Uint8Array;
  notehead: Uint8Array;
  clefsKeys: Uint8Array;
  width: number;
  height: number;
}

export function extractPatch(
  imageRgb: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  winSize: number,
): Float32Array {
  const patch = new Float32Array(3 * winSize * winSize);
  patch.fill(255);

  const x0 = Math.max(x, 0);
  const y0 = Math.max(y, 0);
  const x1 = Math.min(x + winSize, width);
  const y1 = Math.min(y + winSize, height);

  const pw = x1 - x0;
  const ph = y1 - y0;

  for (let c = 0; c < 3; c++) {
    const cOffsetImg = c * width * height;
    const cOffsetPatch = c * winSize * winSize;
    for (let py = 0; py < ph; py++) {
      const iy = y0 + py;
      for (let px = 0; px < pw; px++) {
        const ix = x0 + px;
        const val = imageRgb[cOffsetImg + iy * width + ix] ?? 255;
        patch[cOffsetPatch + py * winSize + px] = val;
      }
    }
  }
  return patch;
}

export function mergePatches(
  patches: Uint8Array[],
  width: number,
  height: number,
  winSize: number,
  stepSize: number,
): Uint8Array {
  const reconstructed = new Float32Array(width * height);
  const weight = new Float32Array(width * height);

  let idx = 0;
  for (let iy = 0; iy < height; iy += stepSize) {
    const y = Math.min(iy, height - winSize);
    const y0 = Math.max(y, 0);
    const y1 = Math.min(y + winSize, height);
    const ph = y1 - y0;

    for (let ix = 0; ix < width; ix += stepSize) {
      const x = Math.min(ix, width - winSize);
      const x0 = Math.max(x, 0);
      const x1 = Math.min(x + winSize, width);
      const pw = x1 - x0;

      const patch = patches[idx++];
      if (!patch) continue;

      for (let py = 0; py < ph; py++) {
        const row = y0 + py;
        for (let px = 0; px < pw; px++) {
          const col = x0 + px;
          const pos = row * width + col;
          const patchVal = patch[py * winSize + px] ?? 0;
          reconstructed[pos] = (reconstructed[pos] ?? 0) + patchVal;
          weight[pos] = (weight[pos] ?? 0) + 1;
        }
      }
    }
  }

  const merged = new Uint8Array(width * height);
  for (let i = 0; i < merged.length; i++) {
    const wVal = weight[i];
    const w = wVal === undefined || wVal === 0 ? 1 : wVal;
    const rVal = reconstructed[i] ?? 0;
    merged[i] = Math.round(rVal / w);
  }
  return merged;
}

export async function runSegnetInference(
  session: InferenceSession,
  imageGray: Uint8Array,
  width: number,
  height: number,
  precision: "fp16" | "fp32" = "fp32",
  batchSize = 8,
  winSize = 320,
  stepSize = 320,
): Promise<SegnetLayers> {
  const imageRgb = new Float32Array(3 * width * height);
  for (let i = 0; i < width * height; i++) {
    const val = imageGray[i] ?? 255;
    imageRgb[i] = val;
    imageRgb[width * height + i] = val;
    imageRgb[2 * width * height + i] = val;
  }

  const patches: Uint8Array[] = [];
  const patchBatch: Float32Array[] = [];

  const runBatch = async (batch: Float32Array[]): Promise<void> => {
    const count = batch.length;
    const inputData = new Float32Array(count * 3 * winSize * winSize);
    for (let i = 0; i < count; i++) {
      const b = batch[i];
      if (b) inputData.set(b, i * 3 * winSize * winSize);
    }

    let inputTensor: Tensor;
    if (precision === "fp16") {
      const fp16Data = new Uint16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const f = inputData[i] ?? 0;
        fp16Data[i] = f >= 255 ? 0x5c00 : Math.round(f) << 3;
      }
      inputTensor = new ort.Tensor("float16", fp16Data, [count, 3, winSize, winSize]);
    } else {
      inputTensor = new ort.Tensor("float32", inputData, [count, 3, winSize, winSize]);
    }

    const feeds: Record<string, Tensor> = { input: inputTensor };
    const outputs = await session.run(feeds);
    const outTensor = outputs["output"] as Tensor;
    const outData = outTensor.data as Float32Array | Uint16Array;

    for (let b = 0; b < count; b++) {
      const patchArgmax = new Uint8Array(winSize * winSize);
      for (let y = 0; y < winSize; y++) {
        for (let x = 0; x < winSize; x++) {
          let maxVal = -Infinity;
          let maxClass = 0;
          for (let c = 0; c < 6; c++) {
            const idx = ((b * 6 + c) * winSize + y) * winSize + x;
            const val = typeof outData[idx] === "number" ? (outData[idx] as number) : 0;
            if (val > maxVal) {
              maxVal = val;
              maxClass = c;
            }
          }
          patchArgmax[y * winSize + x] = maxClass;
        }
      }
      patches.push(patchArgmax);
    }
  };

  for (let yLoop = 0; yLoop < Math.max(height, winSize); yLoop += stepSize) {
    const y = Math.min(yLoop, height - winSize);
    for (let xLoop = 0; xLoop < Math.max(width, winSize); xLoop += stepSize) {
      const x = Math.min(xLoop, width - winSize);
      const patch = extractPatch(imageRgb, width, height, x, y, winSize);
      patchBatch.push(patch);

      if (patchBatch.length === batchSize) {
        await runBatch(patchBatch);
        patchBatch.length = 0;
      }
    }
  }

  if (patchBatch.length > 0) {
    await runBatch(patchBatch);
  }

  const merged = mergePatches(patches, width, height, winSize, stepSize);

  const stemsRests = new Uint8Array(width * height);
  const notehead = new Uint8Array(width * height);
  const clefsKeys = new Uint8Array(width * height);
  const staff = new Uint8Array(width * height);
  const symbols = new Uint8Array(width * height);

  for (let i = 0; i < merged.length; i++) {
    const val = merged[i];
    if (val === 1) stemsRests[i] = 1;
    else if (val === 2) notehead[i] = 1;
    else if (val === 3) clefsKeys[i] = 1;
    else if (val === 4) staff[i] = 1;
    else if (val === 5) symbols[i] = 1;
  }

  return { staff, symbols, stemsRests, notehead, clefsKeys, width, height };
}
