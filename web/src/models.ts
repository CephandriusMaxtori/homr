/**
 * Model manifest: names, revisions and URLs for the ONNX checkpoints.
 *
 * The browser cannot read the GitHub release assets (no Access-Control-Allow-Origin),
 * so the checkpoints are mirrored to the Hugging Face Hub. See
 * `HuggingFaceModels.md` at the repository root for the rationale.
 */

/** homr/transformer/configs.py — used by homr/transformer/staff2score.py. */
export const CONFIG = {
  channels: 1,
  patch_size: 16,
  max_height: 256,
  max_width: 1280,
  max_seq_len: 608,
  pad_token: 0,
  bos_token: 1,
  eos_token: 2,
  nonote_token: 0,
  encoder_depth: 8,
  encoder_dim: 512,
  encoder_heads: 8,
  decoder_depth: 8,
  decoder_dim: 512,
  decoder_heads: 8,
  /**
   * decoder_inference.py:init_cache allocates decoder_depth * 4 KV tensors, each
   * shaped [batch, heads, seq_len, head_dim].
   */
  get kvTensorCount(): number {
    return CONFIG.decoder_depth * 4;
  },
  get headDim(): number {
    return CONFIG.decoder_dim / CONFIG.decoder_heads;
  },
} as const;

/** homr/transformer/staff2score.py:ConvertToArray */
export const NORMALIZATION = {
  mean: 0.7931,
  std: 0.1738,
} as const;

/** homr/segmentation/inference_segnet.py:extract_patch — fixed window. */
export const SEGNET = {
  winSize: 320,
  /** homr/main.py:get_predictions passes step_size=320. */
  stepSize: 320,
  /** homr/segmentation/inference_segnet.py:inference default batch_size. */
  batchSize: 8,
  /** Class indices, from the argmax over axis 0 in inference_segnet.inference. */
  classes: {
    stemsRests: 1,
    notehead: 2,
    clefsKeys: 3,
    staff: 4,
    symbols: 5,
  } as const,
} as const;

export type ModelKey = "segnet" | "encoder" | "decoder";
export type Precision = "fp16" | "fp32";

export interface ModelManifest {
  repo: string;
  revision: string;
  baseUrlTemplate: string;
  models: Record<ModelKey, Record<Precision, string>>;
}

/** Fetch the manifest that ships with the build. */
export async function loadManifest(url = "models.json"): Promise<ModelManifest> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load model manifest from ${url}: HTTP ${response.status}`);
  }
  const manifest = (await response.json()) as ModelManifest;
  for (const key of ["segnet", "encoder", "decoder"] as const) {
    if (!manifest.models?.[key]?.fp16 || !manifest.models?.[key]?.fp32) {
      throw new Error(`Manifest is missing an entry for "${key}"`);
    }
  }
  return manifest;
}

/** Resolve a checkpoint to a fully-qualified URL. */
export function modelUrl(
  manifest: ModelManifest,
  key: ModelKey,
  precision: Precision,
): string {
  return manifest.baseUrlTemplate
    .replace("{repo}", manifest.repo)
    .replace("{revision}", manifest.revision)
    .replace("{file}", manifest.models[key][precision]);
}

/**
 * WebGPU gets the fp16 weights; the single-threaded WASM backend gets fp32.
 *
 * homr/main.py already documents this for the CPU execution provider: the fp16
 * models are slower there than the fp32 ones. GitHub Pages cannot send the
 * COOP/COEP headers required for SharedArrayBuffer, so the WASM path cannot be
 * made multi-threaded.
 */
export function preferredPrecision(hasWebGpu: boolean): Precision {
  return hasWebGpu ? "fp16" : "fp32";
}