/**
 * Phase 0 spike driver.
 *
 * Runs the three feasibility questions in one page load and leaves the answers on
 * `window.__SPIKE__` for Playwright to collect:
 *
 *   1. Can the three checkpoints run at all in this browser, and how fast?
 *   2. Is the OpenCV.js WASM build missing anything the port needs?
 *   3. Does the decoder's 32-tensor KV cache actually bind and grow?
 *
 * The decoder is fed a REAL encoder output (public/spike/staff_0_context.npy,
 * dumped by scripts/dump_spike_tensors.py) so the step count is the same number the
 * Python reference produced. A synthetic context would not reach EOS the same way.
 */

import ort, { configureOrt } from "../ort.ts";

configureOrt();

import { loadManifest, preferredPrecision, type ModelManifest } from "../models.ts";
import {
  benchDecoder,
  benchEncoder,
  benchSegnet,
  detectWebGpu,
  type BenchResult,
  type ProviderName,
} from "./bench.ts";
import { probeOpenCv, type OpenCvReport } from "./opencv-probe.ts";
import { fetchNpy } from "./npy.ts";

interface SpikeReport {
  startedAt: string;
  userAgent: string;
  crossOriginIsolated: boolean;
  hardwareConcurrency: number;
  webgpu: { available: boolean; detail: string; adapter?: string };
  preferredPrecision: "fp16" | "fp32";
  reference: unknown;
  bench: BenchResult[];
  opencv: OpenCvReport | null;
  opencvError?: string;
  finishedAt?: string;
}

declare global {
  interface Window {
    __SPIKE__?: SpikeReport;
    __SPIKE_DONE__?: boolean;
  }
}

const status = document.getElementById("status") as HTMLDivElement;
const log = document.getElementById("log") as HTMLDivElement;
const table = document.getElementById("results") as HTMLTableElement;

function say(message: string): void {
  log.textContent += `${message}\n`;
  log.scrollTop = log.scrollHeight;
  console.log(`[spike] ${message}`);
}

function renderBench(results: BenchResult[]): void {
  const rows = results
    .map((r) => {
      const failed = r.error ? "bad" : "ok";
      return `<tr>
        <td>${r.key}</td>
        <td>${r.provider}</td>
        <td>${r.precision}</td>
        <td class="${failed}">${r.error ? "FAIL" : (r.steps ?? r.iterations)}</td>
        <td>${r.sessionCreateSeconds.toFixed(2)}</td>
        <td>${r.firstRunSeconds.toFixed(2)}</td>
        <td>${r.steadyRunSeconds.toFixed(2)}</td>
        <td>${r.msPerStep ? r.msPerStep.toFixed(1) : "-"}</td>
        <td class="${failed}">${r.error ? r.error.slice(0, 90) : "ok"}</td>
      </tr>`;
    })
    .join("");
  table.innerHTML = `<thead><tr>
      <th>model</th><th>provider</th><th>prec</th><th>n</th>
      <th>create s</th><th>cold s</th><th>steady s</th><th>ms/step</th><th>note</th>
    </tr></thead><tbody>${rows}</tbody>`;
}

async function adapterDescription(): Promise<string> {
  const gpu = (navigator as unknown as {
    gpu?: { requestAdapter(): Promise<{ info?: unknown; requestAdapterInfo?: () => Promise<unknown> } | null> };
  }).gpu;
  if (!gpu) return "n/a";
  const adapter = await gpu.requestAdapter();
  if (!adapter) return "requestAdapter() returned null";
  const info = (adapter as { info?: { vendor?: string; architecture?: string } }).info;
  return info ? `${info.vendor ?? "?"}/${info.architecture ?? "?"}` : "adapter obtained";
}

async function main(): Promise<void> {
  const report: SpikeReport = {
    startedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    hardwareConcurrency: navigator.hardwareConcurrency,
    webgpu: { ...detectWebGpu() },
    preferredPrecision: "fp32",
    reference: null,
    bench: [],
    opencv: null,
  };
  window.__SPIKE__ = report;

  status.textContent = "loading manifest…";
  let manifest: ModelManifest;
  try {
    manifest = await loadManifest("models.json");
  } catch (error) {
    status.textContent = `manifest failed: ${String(error)}`;
    window.__SPIKE_DONE__ = true;
    return;
  }
  say(`manifest ok: ${manifest.repo}@${manifest.revision.slice(0, 8)}`);

  try {
    const referenceResponse = await fetch("spike/reference.json");
    if (referenceResponse.ok) report.reference = await referenceResponse.json();
    say("loaded Python reference metrics");
  } catch {
    say("no reference metrics found");
  }

  // --- OpenCV.js probes (independent of the models) ------------------------
  status.textContent = "probing OpenCV.js…";
  try {
    report.opencv = await probeOpenCv();
    const missing = report.opencv.probes.filter((p) => !p.ok).map((p) => p.name);
    say(`opencv.js: ${report.opencv.probes.length} probes, ${report.opencv.symbolCount} symbols`);
    say(`opencv.js absent: ${missing.length ? missing.join(", ") : "none"}`);
    for (const probe of report.opencv.probes) {
      if (probe.name === "CLAHE.apply" || probe.name === "findContours" ||
          probe.name === "minAreaRect" || probe.name === "reduce" || probe.name === "Subdiv2D") {
        say(`  ${probe.ok ? "OK " : "NO "} ${probe.name}: ${probe.detail}`);
      }
    }
  } catch (error) {
    report.opencvError = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    say(`opencv.js failed: ${report.opencvError}`);
  }

  // --- real encoder output for the decoder --------------------------------
  status.textContent = "loading golden encoder output…";
  let context: ort.Tensor | undefined;
  try {
    const npy = await fetchNpy("spike/staff_0_context.npy");
    say(`golden context: ${npy.dtype} ${JSON.stringify(npy.shape)}`);
    if (npy.shape.length !== 3 || npy.shape[1] !== 1280) {
      throw new Error(`unexpected context shape ${JSON.stringify(npy.shape)}`);
    }
    context = new ort.Tensor("float32", npy.data as Float32Array, npy.shape);
  } catch (error) {
    say(`no golden context (${String(error)}); decoder will use a zero context`);
  }

  // --- benchmarks ---------------------------------------------------------
  report.webgpu.adapter = await adapterDescription();
  say(`webgpu adapter: ${report.webgpu.adapter}`);
  say(`crossOriginIsolated=${report.crossOriginIsolated} (Pages cannot set COOP/COEP)`);

  // Measure each provider at ITS preferred precision only.
  //
  // The full 2x2 matrix was tried first and is not worth its cost: every bench
  // downloads its own copy of the checkpoint (segnet 27-55 MB, encoder 25-50 MB,
  // decoder 45-90 MB), so all four combinations meant ~300 MB of model traffic and a
  // multi-minute run. The decision being made is "which precision per provider",
  // which the off-diagonal cells cannot inform.
  // `navigator.gpu` existing does not mean an adapter can actually be acquired: in a
  // headless run it can be present while requestAdapter() returns null, and ORT then
  // fails with a bare status code rather than a clear message. Fall back to WASM-only
  // when no adapter is actually obtained.
  const adapter = report.webgpu.adapter ?? "";
  const adapterWorks = report.webgpu.available && !/returned null/i.test(adapter);
  if (report.webgpu.available && !adapterWorks) {
    report.webgpu.available = false;
    report.webgpu.detail = `${report.webgpu.detail}; requestAdapter failed -> falling back to wasm`;
    say("webgpu: navigator.gpu present but no adapter obtained; using wasm only");
  }
  report.preferredPrecision = preferredPrecision(adapterWorks);

  const plan: { provider: ProviderName; precision: "fp16" | "fp32" }[] = adapterWorks
    ? [
        { provider: "webgpu", precision: "fp16" },
        { provider: "wasm", precision: "fp32" },
      ]
    : [{ provider: "wasm", precision: "fp32" }];
  say(`matrix: ${plan.map((p) => `${p.provider}/${p.precision}`).join(", ")}`);

  for (const { provider, precision: prec } of plan) {
    status.textContent = `segnet ${provider}/${prec}…`;
    const segnet = await benchSegnet(manifest, provider, prec);
    say(
      `segnet ${provider}/${prec}: ${segnet.error
        ? `FAIL ${segnet.error}`
        : `${segnet.steadyRunSeconds.toFixed(2)}s for ${segnet.iterations} batches (${(
            (segnet.steadyRunSeconds * 1000) /
            segnet.iterations
          ).toFixed(0)} ms/batch)`}`,
    );
    report.bench.push(segnet);
    renderBench(report.bench);

    status.textContent = `encoder ${provider}/${prec}…`;
    const encoder = await benchEncoder(manifest, provider, prec);
    say(
      `encoder ${provider}/${prec}: ${encoder.error
        ? `FAIL ${encoder.error}`
        : `${encoder.steadyRunSeconds.toFixed(2)}s for ${encoder.iterations} forwards (${(
            (encoder.steadyRunSeconds * 1000) /
            encoder.iterations
          ).toFixed(0)} ms)`}`,
    );
    report.bench.push(encoder);
    renderBench(report.bench);

    status.textContent = `decoder ${provider}/${prec} (up to 608 steps)…`;
    say(`decoder ${provider}/${prec}: running the full greedy decode, this is the long one…`);
    const decoder = await benchDecoder(manifest, provider, prec, context);
    say(
      `decoder ${provider}/${prec}: ${decoder.error
        ? `FAIL ${decoder.error}`
        : `${decoder.steps} steps, ${decoder.msPerStep?.toFixed(1)} ms/step, ${decoder.steadyRunSeconds.toFixed(
            2,
          )}s total`}`,
    );
    for (const line of decoder.trace) say(`  ${line}`);
    report.bench.push(decoder);
    renderBench(report.bench);
  }

  report.finishedAt = new Date().toISOString();
  status.textContent = "done";
  window.__SPIKE_DONE__ = true;
}

main().catch((error) => {
  status.textContent = `fatal: ${error instanceof Error ? error.stack : String(error)}`;
  say(`fatal: ${String(error)}`);
  window.__SPIKE_DONE__ = true;
});