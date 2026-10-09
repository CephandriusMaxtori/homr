/**
 * TrOMR transformer model runner (Encoder & Greedy Decoder).
 */

import type { InferenceSession, Tensor } from "onnxruntime-web";
import { CONFIG, NORMALIZATION } from "./models.ts";
import ort from "./ort.ts";

import { EncodedSymbol, Vocabulary } from "./vocabulary.ts";

const vocab = new Vocabulary();

function reverseVocab(map: Record<string, number>): Record<number, string> {
  const rev: Record<number, string> = {};
  for (const [k, v] of Object.entries(map)) {
    rev[v] = k;
  }
  return rev;
}

const invRhythm = reverseVocab(vocab.rhythm);
const invPitch = reverseVocab(vocab.pitch);
const invLift = reverseVocab(vocab.lift);
const invArticulation = reverseVocab(vocab.articulation);
const invSlur = reverseVocab(vocab.slur);
const invPosition = reverseVocab(vocab.position);

export function prepareEncoderInput(
  imageGray: Uint8Array,
  width: number,
  height: number,
  maxHeight = 256,
  maxWidth = 1280,
): Float32Array {
  const canvas = new Float32Array(maxHeight * maxWidth);

  const whiteNorm = (1.0 - NORMALIZATION.mean) / NORMALIZATION.std;
  canvas.fill(whiteNorm);

  const copyH = Math.min(height, maxHeight);
  const copyW = Math.min(width, maxWidth);

  for (let y = 0; y < copyH; y++) {
    for (let x = 0; x < copyW; x++) {
      const pixel = (imageGray[y * width + x] ?? 255) / 255.0;
      const norm = (pixel - NORMALIZATION.mean) / NORMALIZATION.std;
      canvas[y * maxWidth + x] = norm;
    }
  }

  return canvas;
}

function argmaxLastStep(tensor: Tensor): number {
  const dims = tensor.dims;
  const width = dims[dims.length - 1] ?? 1;
  const data = tensor.data as Float32Array | Uint16Array;
  const stride = tensor.type === "float16" ? 2 : 4;
  const start = (data.length / stride - width) * stride;

  let bestIndex = 0;
  let bestVal = -Infinity;

  for (let i = 0; i < width; i++) {
    const idx = start + i * stride;
    let val = 0;
    if (tensor.type === "float16") {
      const h = (data as Uint16Array)[idx] ?? 0;
      const sign = (h & 0x8000) >> 15;
      const exp = (h & 0x7c00) >> 10;
      const frac = h & 0x03ff;
      val = exp === 0 ? frac * 2 ** -24 : (frac / 1024 + 1) * 2 ** (exp - 15);
      if (sign) val = -val;
    } else {
      val = (data as Float32Array)[idx] ?? 0;
    }
    if (val > bestVal) {
      bestVal = val;
      bestIndex = i;
    }
  }
  return bestIndex;
}

export async function runEncoder(
  encoderSession: InferenceSession,
  imageGray: Uint8Array,
  width: number,
  height: number,
  precision: "fp16" | "fp32" = "fp32",
): Promise<Tensor> {
  const inputData = prepareEncoderInput(imageGray, width, height);
  let inputTensor: Tensor;

  if (precision === "fp16") {
    const fp16Data = new Uint16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
      const f = inputData[i] ?? 0;
      fp16Data[i] = f >= 1 ? 0x3c00 : 0x0000;
    }
    inputTensor = new ort.Tensor("float16", fp16Data, [1, 1, 256, 1280]);
  } else {
    inputTensor = new ort.Tensor("float32", inputData, [1, 1, 256, 1280]);
  }

  const feeds = { input: inputTensor };
  const outputs = await encoderSession.run(feeds);
  return outputs["output"] as Tensor;
}

export async function runGreedyDecoder(
  decoderSession: InferenceSession,
  context: Tensor,
  precision: "fp16" | "fp32" = "fp32",
): Promise<EncodedSymbol[]> {
  const dtype = precision === "fp16" ? "float16" : "float32";
  const heads = CONFIG.decoder_heads;
  const headDim = CONFIG.headDim;
  const kvCount = CONFIG.kvTensorCount;

  const contextData = context.data as Float32Array | Uint16Array;
  const contextReduced = new ort.Tensor(
    dtype,
    contextData.slice(0, CONFIG.encoder_dim),
    [1, 1, CONFIG.encoder_dim],
  );

  const makeToken = (val: number): Tensor =>
    new ort.Tensor("int64", BigInt64Array.from([BigInt(val)]), [1, 1]);

  let rhythmToken = makeToken(CONFIG.bos_token);
  let pitchToken = makeToken(CONFIG.nonote_token);
  let liftToken = makeToken(CONFIG.nonote_token);
  let articulationToken = makeToken(CONFIG.nonote_token);
  let slurToken = makeToken(CONFIG.nonote_token);

  let cache: Tensor[] = Array.from(
    { length: kvCount },
    () => new ort.Tensor(dtype, precision === "fp16" ? new Uint16Array(0) : new Float32Array(0), [1, heads, 0, headDim]),
  );

  const symbols: EncodedSymbol[] = [];

  for (let step = 0; step < CONFIG.max_seq_len; step++) {
    const feeds: Record<string, Tensor> = {
      rhythms: rhythmToken,
      pitchs: pitchToken,
      lifts: liftToken,
      articulations: articulationToken,
      slurs: slurToken,
      context: step === 0 ? context : contextReduced,
      cache_len: new ort.Tensor("int64", BigInt64Array.from([BigInt(step)]), [1]),
    };

    cache.forEach((tensor, i) => {
      feeds[`cache_in${i}`] = tensor;
    });

    const out = await decoderSession.run(feeds);

    const rhythmIdx = argmaxLastStep(out["out_rhythms"] as Tensor);
    if (rhythmIdx === CONFIG.eos_token) break;

    const pitchIdx = argmaxLastStep(out["out_pitchs"] as Tensor);
    const liftIdx = argmaxLastStep(out["out_lifts"] as Tensor);
    const articulationIdx = argmaxLastStep(out["out_articulations"] as Tensor);
    const slurIdx = argmaxLastStep(out["out_slurs"] as Tensor);
    const positionIdx = argmaxLastStep(out["out_positions"] as Tensor);

    const symbol = new EncodedSymbol(
      invRhythm[rhythmIdx] ?? "[UNK]",
      invPitch[pitchIdx] ?? ".",
      invLift[liftIdx] ?? ".",
      invArticulation[articulationIdx] ?? ".",
      invSlur[slurIdx] ?? ".",
      invPosition[positionIdx] ?? ".",
    );

    symbols.push(symbol);

    rhythmToken = makeToken(rhythmIdx);
    pitchToken = makeToken(pitchIdx);
    liftToken = makeToken(liftIdx);
    articulationToken = makeToken(articulationIdx);
    slurToken = makeToken(slurIdx);

    cache = Array.from(
      { length: kvCount },
      (_, i) => out[`cache_out${i}`] as Tensor,
    );
  }

  return symbols;
}
