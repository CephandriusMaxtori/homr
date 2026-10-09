/**
 * Single place where onnxruntime-web is configured.
 *
 * Two things matter here and both were verified during the Phase 0 spike:
 *
 * 1. We import from "onnxruntime-web/webgpu", not the default entry point. Only the
 *    webgpu build carries the WebGPU kernels; the default entry point has no
 *    `webgpu` execution provider at all. The webgpu build still supports the plain
 *    `wasm` provider, so one import covers both paths.
 *
 * 2. The .wasm binaries are copied to public/ort by scripts/copy-wasm.mjs and served
 *    from there. ORT would otherwise try to resolve them relative to its own module
 *    URL, which Vite rewrites during the build.
 */

import * as ort from "onnxruntime-web/webgpu";

/**
 * Absolute URL to public/ort/.
 *
 * ORT resolves these paths itself and rejects anything that is not absolute: a bare
 * "ort/" is treated as a bare module specifier and fails with
 * "Failed to resolve module specifier 'ort/ort-wasm-simd-threaded.jsep.mjs'".
 * Resolving against the document base also keeps the same build working at a domain
 * root and under a project path.
 */
export function ortWasmDir(): string {
  return new URL("ort/", document.baseURI).href;
}

let configured = false;

export function configureOrt(): typeof ort {
  if (configured) return ort;
  // Pass the directory as an absolute prefix, not explicit {wasm, mjs} paths. ORT
  // picks the filename per build: "ort-wasm-simd-threaded.wasm" for the wasm provider
  // and the ".jsep." variant when the webgpu provider is requested. Pinning the paths
  // explicitly would force the plain build even for WebGPU.
  ort.env.wasm.wasmPaths = ortWasmDir();
  // GitHub Pages cannot send the COOP/COEP headers that SharedArrayBuffer needs, so
  // numThreads > 1 would silently not take effect. Keep it at 1 rather than implying
  // parallelism the WASM fallback does not have.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  configured = true;
  return ort;
}

export type Ort = typeof ort;
export default ort;