// Drive the spike page directly with its own timeout and always write a report.
//
// The Playwright CLI runner and the spike page were fighting over the same dev
// server, so this script owns the browser lifecycle end to end and is safe to
// re-run while a `playwright test` invocation is stuck.
//
//   node scripts/run-spike.mjs [url] [reportPath]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const url = process.argv[2] ?? "http://localhost:5199/spike.html";
const reportPath = resolve(process.argv[3] ?? join(here, "..", "spike-report.json"));

const TIMEOUT_MS = Number(process.env.SPIKE_TIMEOUT_MS ?? 20 * 60 * 1000);

console.log(`launching chromium, navigating to ${url} (timeout ${TIMEOUT_MS} ms)`);

const browser = await chromium.launch({
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--ignore-gpu-blocklist",
    "--use-angle=default",
  ],
});

const page = await browser.newPage();
const failures = [];
const logs = [];
page.on("requestfailed", (r) => failures.push(`${r.url()} :: ${r.failure()?.errorText}`));
page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });

const done = await page
  .waitForFunction(() => window.__SPIKE_DONE__ === true, null, { timeout: TIMEOUT_MS })
  .then(() => true)
  .catch(() => false);

const report = await page.evaluate(() => window.__SPIKE__);
await browser.close();

const output = { ...(report ?? {}), completed: done, requestFailures: failures, consoleTail: logs.slice(-80) };
mkdirSync(dirname(reportPath), { recursive: true });
writeFileSync(reportPath, JSON.stringify(output, null, 2), "utf-8");

console.log(`\n================= homr web — Phase 0 spike =================`);
console.log(`completed        : ${done}`);
if (report) {
  console.log(`webgpu available : ${report.webgpu?.available} (${report.webgpu?.adapter ?? report.webgpu?.detail})`);
  console.log(`crossOriginIsolated: ${report.crossOriginIsolated}`);
  console.log(`preferred precision: ${report.preferredPrecision}`);
  console.log(`\nmodel       provider  prec  n        create s  cold s   steady s  ms/step`);
  for (const r of report.bench ?? []) {
    const n = r.steps ?? r.iterations;
    const ms = r.msPerStep ? r.msPerStep.toFixed(1) : "-";
    console.log(
      `  ${r.key.padEnd(10)} ${r.provider.padEnd(9)} ${r.precision.padEnd(5)} ${String(n).padEnd(8)} ` +
        `${r.sessionCreateSeconds.toFixed(2).padStart(8)}  ${r.firstRunSeconds.toFixed(2).padStart(7)}  ` +
        `${r.steadyRunSeconds.toFixed(2).padStart(8)}  ${ms.padStart(7)}`,
    );
    if (r.error) console.log(`      ! ${r.error.slice(0, 140)}`);
  }
  if (report.opencv) {
    const absent = report.opencv.probes.filter((p) => !p.ok).map((p) => p.name);
    console.log(`\nOpenCV.js (${report.opencv.symbolCount} symbols), absent: ${absent.join(", ") || "none"}`);
    for (const p of report.opencv.probes) {
      if (["CLAHE.apply", "findContours", "minAreaRect", "boxPoints", "MatVector", "reduce", "Subdiv2D"].includes(p.name)) {
        console.log(`  ${p.ok ? "present" : "ABSENT "}  ${p.name.padEnd(12)} ${p.detail.slice(0, 110)}`);
      }
    }
  }
  if (report.opencvError) console.log(`OpenCV.js failed: ${report.opencvError}`);
}
if (failures.length) {
  console.log(`\nfailed requests (${failures.length}):`);
  for (const f of failures.slice(0, 10)) console.log(`  ${f}`);
}
console.log(`\nreport -> ${reportPath}`);