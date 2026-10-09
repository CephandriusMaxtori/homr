// Copy the prebuilt WASM payloads out of node_modules into public/ so Vite serves
// them as static assets. Doing it this way (rather than importing them) keeps the
// 13-27 MB binaries out of the JS dependency graph and lets ort.env.wasm.wasmPaths
// point at a predictable location on both localhost and GitHub Pages.
//
//   node scripts/copy-wasm.mjs

import { copyFile, mkdir, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const ortDist = join(web, "node_modules", "onnxruntime-web", "dist");
const ortOut = join(web, "public", "ort");

// onnxruntime-web picks the filename by build, and "wasmPaths" as a directory prefix
// means whatever it asks for must exist. Verified against 1.30.0, which selects
// .asyncify for the webgpu provider in some configurations, so copy every variant
// rather than guessing. Each is fetched only if actually requested.
const files = [
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
];

await mkdir(ortOut, { recursive: true });

let total = 0;
for (const file of files) {
  const from = join(ortDist, file);
  try {
    await copyFile(from, join(ortOut, file));
    const info = await stat(join(ortOut, file));
    total += info.size;
    console.log(`  ${file.padEnd(40)} ${(info.size / 1024 / 1024).toFixed(2)} MB`);
  } catch (error) {
    console.error(`  MISSING ${file}: ${error.message}`);
    process.exitCode = 1;
  }
}
console.log(`copied ${files.length} files, ${(total / 1024 / 1024).toFixed(1)} MB -> public/ort/`);