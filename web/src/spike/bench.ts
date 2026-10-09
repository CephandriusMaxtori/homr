/**
 * Phase 0 spike: are these three models actually viable in a browser?
 *
 * The decoder is the risk. It needs 32 dynamic KV-cache tensors bound per step, and it
 * runs once per staff, so page cost is (steps x staves) decoder invocations. If a step
 * costs tens of milliseconds there is no usable product, so this measures it before
 * any of the port is written.
 *
 * The KV-cache wiring mirrors homr/transformer/decoder_inference.py:ScoreDecoder
 * exactly, including the "full context on step 0, first frame only afterwards"
 * behaviour, whose rationale is in the comment at decoder_inference.py:83-86.
 */

import ort, { configureOrt } from "../ort.ts";

import {
  CONFIG,
  SEGNET,
  type ModelKey,
  type ModelManifest,
  type Precision,
  modelUrl,
} from "../models.ts";

export type ProviderName = "webgpu" | "wasm";
export type Dtype = "float16" | "float32";

export interface ProviderReport {
  provider: ProviderName;
  available: boolean;
  detail: string;
}

export interface BenchResult {
  provider: ProviderName;
  key: ModelKey;
  precision: Precision;
  sessionCreateSeconds: number;
  firstRunSeconds: number;
  steadyRunSeconds: number;
  iterations: number;
  /** Decoder only: how many greedy steps until EOS. */
  steps?: number;
  /** Decoder only: steadyRunSeconds / steps. */
  msPerStep?: number;
  outputShape?: number[];
  /** Progress lines, so a long decode is not indistinguishable from a hang. */
  trace: string[];
  error?: string;
}

export function detectWebGpu(): ProviderReport {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) {
    return { provider: "webgpu", available: false, detail: "navigator.gpu is undefined" };
  }
  return { provider: "webgpu", available: true, detail: "navigator.gpu present" };
}

function now(): number {
  return performance.now();
}

function describeError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

async function createSession(url: string, provider: ProviderName): Promise<ort.InferenceSession> {
  configureOrt();
  return ort.InferenceSession.create(url, {
    executionProviders: [provider],
    graphOptimizationLevel: "all",
  });
}

function zeros(dtype: Dtype, length: number): Uint16Array | Float32Array {
  return dtype === "float16" ? new Uint16Array(length) : new Float32Array(length);
}

/** half-precision (binary16) -> number, for argmax over fp16 logits. */
function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exponent = (h & 0x7c00) >> 10;
  const fraction = h & 0x03ff;
  let value: number;
  if (exponent === 0) {
    value = fraction * 2 ** -24;
  } else if (exponent === 0x1f) {
    value = fraction ? Number.NaN : Number.POSITIVE_INFINITY;
  } else {
    value = (fraction / 1024 + 1) * 2 ** (exponent - 15);
  }
  return sign ? -value : value;
}

/**
 * argmax over the last axis of the final sequence position.
 * numpy/TF argmax returns the FIRST maximum on ties, so match that.
 */
function argmaxLastStep(tensor: ort.Tensor, dtype: Dtype): number {
  const dims = tensor.dims;
  const width = dims[dims.length - 1] ?? 1;
  const stride = dtype === "float16" ? 2 : 4;
  const data = tensor.data as Uint16Array | Float32Array;
  // The decoder emits [1, 1, width]; step through the tail element-wise.
  const start = (data.length / stride - width) * stride;
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  const halves = data as Uint16Array;
  const floats = data as Float32Array;
  for (let i = 0; i < width; i++) {
    const offset = start + i * stride;
    const value = dtype === "float16" ? halfToFloat(halves[offset] ?? 0) : (floats[offset] ?? 0);
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/** Benchmark one full segnet pass: the patch batches a 1920x2235 page needs. */
export async function benchSegnet(
  manifest: ModelManifest,
  provider: ProviderName,
  precision: Precision,
): Promise<BenchResult> {
  const result: BenchResult = {
    provider,
    key: "segnet",
    precision,
    sessionCreateSeconds: 0,
    firstRunSeconds: 0,
    steadyRunSeconds: 0,
    iterations: 0,
    trace: [],
  };
  try {
    const t0 = now();
    const session = await createSession(modelUrl(manifest, "segnet", precision), provider);
    result.sessionCreateSeconds = (now() - t0) / 1000;

    const dtype: Dtype = precision === "fp16" ? "float16" : "float32";
    const size = SEGNET.winSize;
    const channels = 3;
    const input = new ort.Tensor(
      dtype,
      dtype === "float16"
        ? new Uint16Array(SEGNET.batchSize * channels * size * size).fill(0x3c00)
        : new Float32Array(SEGNET.batchSize * channels * size * size).fill(255),
      [SEGNET.batchSize, channels, size, size],
    );

    const t1 = now();
    const first = await session.run({ input });
    result.firstRunSeconds = (now() - t1) / 1000;
    result.outputShape = [...(first["output"] as ort.Tensor).dims];

    // A 1920x2235 page needs ceil(2235/320) x ceil(1920/320) = 7 x 6 = 42 windows,
    // which is 6 batches of 8 at batch_size=8 (inference_segnet.py).
    const iterations = 6;
    const t2 = now();
    for (let i = 0; i < iterations; i++) await session.run({ input });
    result.steadyRunSeconds = (now() - t2) / 1000;
    result.iterations = iterations;
    await session.release();
  } catch (error) {
    result.error = describeError(error);
  }
  return result;
}

/** Benchmark a single encoder forward at the fixed [1, 1, 256, 1280] input. */
export async function benchEncoder(
  manifest: ModelManifest,
  provider: ProviderName,
  precision: Precision,
): Promise<BenchResult> {
  const result: BenchResult = {
    provider,
    key: "encoder",
    precision,
    sessionCreateSeconds: 0,
    firstRunSeconds: 0,
    steadyRunSeconds: 0,
    iterations: 0,
    trace: [],
  };
  try {
    const t0 = now();
    const session = await createSession(modelUrl(manifest, "encoder", precision), provider);
    result.sessionCreateSeconds = (now() - t0) / 1000;

    const dtype: Dtype = precision === "fp16" ? "float16" : "float32";
    const h = CONFIG.max_height;
    const w = CONFIG.max_width;
    const input = new ort.Tensor(dtype, zeros(dtype, h * w).fill(0), [1, CONFIG.channels, h, w]);

    const t1 = now();
    const first = await session.run({ input });
    result.firstRunSeconds = (now() - t1) / 1000;
    result.outputShape = [...(first["output"] as ort.Tensor).dims];

    const iterations = 5;
    const t2 = now();
    for (let i = 0; i < iterations; i++) await session.run({ input });
    result.steadyRunSeconds = (now() - t2) / 1000;
    result.iterations = iterations;
    await session.release();
  } catch (error) {
    result.error = describeError(error);
  }
  return result;
}

/**
 * Greedy-decode one staff.
 *
 * `contextOverride` should be a real encoder output for a real dewarped staff
 * (from scripts/dump_spike_tensors.py). A synthetic context would not hit EOS the
 * way real sheet music does, making the step count — the number that decides
 * viability — meaningless.
 */
export async function benchDecoder(
  manifest: ModelManifest,
  provider: ProviderName,
  precision: Precision,
  contextOverride?: ort.Tensor,
): Promise<BenchResult> {
  const result: BenchResult = {
    provider,
    key: "decoder",
    precision,
    sessionCreateSeconds: 0,
    firstRunSeconds: 0,
    steadyRunSeconds: 0,
    iterations: 0,
    trace: [],
  };
  try {
    const dtype: Dtype = precision === "fp16" ? "float16" : "float32";
    const t0 = now();
    const session = await createSession(modelUrl(manifest, "decoder", precision), provider);
    result.sessionCreateSeconds = (now() - t0) / 1000;

    const context =
      contextOverride ??
      new ort.Tensor(dtype, zeros(dtype, 1280 * CONFIG.encoder_dim), [1, 1280, CONFIG.encoder_dim]);
    // decoder_inference.py:73 -- only the first frame is passed after step 0.
    const contextReduced = new ort.Tensor(
      dtype,
      (context.data as Uint16Array | Float32Array).slice(0, CONFIG.encoder_dim),
      [1, 1, CONFIG.encoder_dim],
    );

    const heads = CONFIG.decoder_heads;
    const headDim = CONFIG.headDim;
    const kvCount = CONFIG.kvTensorCount;

    const token = (value: number): ort.Tensor =>
      new ort.Tensor("int64", BigInt64Array.from([BigInt(value)]), [1, 1]);

    const runDecode = async (): Promise<number> => {
      let rhythm = token(CONFIG.bos_token);
      let pitch = token(CONFIG.nonote_token);
      let lift = token(CONFIG.nonote_token);
      let articulation = token(CONFIG.nonote_token);
      let slur = token(CONFIG.nonote_token);

      // Step 0 starts from an empty cache: decoder_inference.py:init_cache(0).
      let cache: ort.Tensor[] = Array.from(
        { length: kvCount },
        () => new ort.Tensor(dtype, zeros(dtype, heads * headDim), [1, heads, 0, headDim]),
      );

      let steps = 0;
      const runStart = now();
      for (let step = 0; step < CONFIG.max_seq_len; step++) {
        const feeds: Record<string, ort.Tensor> = {
          rhythms: rhythm,
          pitchs: pitch,
          lifts: lift,
          articulations: articulation,
          slurs: slur,
          context: step === 0 ? context : contextReduced,
          cache_len: new ort.Tensor("int64", BigInt64Array.from([BigInt(step)]), [1]),
        };
        cache.forEach((tensor, i) => {
          feeds[`cache_in${i}`] = tensor;
        });

        const out = await session.run(feeds);

        const rhythmSample = argmaxLastStep(out["out_rhythms"] as ort.Tensor, dtype);
        if (rhythmSample === CONFIG.eos_token) {
          result.trace.push(
            `EOS at step ${step} after ${((now() - runStart) / 1000).toFixed(2)}s`,
          );
          break;
        }

        steps++;
        rhythm = token(rhythmSample);
        pitch = token(argmaxLastStep(out["out_pitchs"] as ort.Tensor, dtype));
        lift = token(argmaxLastStep(out["out_lifts"] as ort.Tensor, dtype));
        articulation = token(argmaxLastStep(out["out_articulations"] as ort.Tensor, dtype));
        slur = token(argmaxLastStep(out["out_slurs"] as ort.Tensor, dtype));

        cache = Array.from(
          { length: kvCount },
          (_, i) => out[`cache_out${i}`] as ort.Tensor,
        );
      }
      return steps;
    };

    const t1 = now();
    const coldSteps = await runDecode();
    result.firstRunSeconds = (now() - t1) / 1000;

    const t2 = now();
    result.trace.push(`cold decode: ${coldSteps} steps in ${result.firstRunSeconds.toFixed(2)}s`);
    const steps = await runDecode();
    result.steadyRunSeconds = (now() - t2) / 1000;
    result.iterations = steps;
    result.steps = steps;
    result.msPerStep = steps > 0 ? (result.steadyRunSeconds * 1000) / steps : 0;
    if (coldSteps !== steps) {
      // Not fatal, but a sign the first run was not representative.
      result.error = `note: cold decode produced ${coldSteps} steps, steady produced ${steps}`;
    }
    await session.release();
  } catch (error) {
    result.error = describeError(error);
  }
  return result;
}

export async function runAll(
  manifest: ModelManifest,
  providers: ProviderName[],
  precisions: Precision[],
  context?: ort.Tensor,
): Promise<{ providers: ProviderReport[]; results: BenchResult[] }> {
  const report: { providers: ProviderReport[]; results: BenchResult[] } = {
    providers: [detectWebGpu()],
    results: [],
  };
  for (const provider of providers) {
    for (const precision of precisions) {
      report.results.push(await benchSegnet(manifest, provider, precision));
      report.results.push(await benchEncoder(manifest, provider, precision));
      report.results.push(await benchDecoder(manifest, provider, precision, context));
    }
  }
  return report;
}