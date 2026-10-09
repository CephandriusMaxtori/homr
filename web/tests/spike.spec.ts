import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const reportPath = join(here, "..", "spike-report.json");

/**
 * The spike is a measurement, not an assertion. It fails only if the page itself
 * breaks or nothing at all was measured — individual provider failures are recorded
 * in the report and judged by a human, because "WASM is slow" is a finding rather
 * than a bug.
 */
test("run the Phase 0 spike and record measurements", async ({ page }) => {
  const logs: string[] = [];
  page.on("console", (message) => logs.push(`${message.type()}: ${message.text()}`));
  page.on("pageerror", (error) => logs.push(`pageerror: ${error.message}`));

  const requestFailures: string[] = [];
  page.on("requestfailed", (request) => {
    requestFailures.push(`${request.url()} :: ${request.failure()?.errorText}`);
  });

  await page.goto("/spike.html", { waitUntil: "domcontentloaded" });

  // The whole thing (model downloads plus decodes) is slow by nature.
  await page.waitForFunction(() => window.__SPIKE_DONE__ === true, null, {
    timeout: 25 * 60 * 1000,
  });

  const report = await page.evaluate(() => window.__SPIKE__);

  expect(report, "window.__SPIKE__ must be populated").toBeTruthy();
  expect(report.bench.length, "at least some benchmarks must have run").toBeGreaterThan(0);

  const output = {
    ...report,
    requestFailures,
    consoleTail: logs.slice(-60),
  };
  writeFileSync(reportPath, JSON.stringify(output, null, 2), "utf-8");

  console.log("\n================= homr web — Phase 0 spike =================");
  console.log(`userAgent            : ${report.userAgent}`);
  console.log(`crossOriginIsolated  : ${report.crossOriginIsolated}`);
  console.log(`hardwareConcurrency  : ${report.hardwareConcurrency}`);
  console.log(`webgpu available     : ${report.webgpu.available} (${report.webgpu.adapter ?? report.webgpu.detail})`);
  console.log(`preferred precision  : ${report.preferredPrecision}`);

  if (report.reference) {
    const reference = report.reference as {
      seg_seconds: number;
      staff_count: number;
      multi_staff_group_sizes: number[];
      staves: { reference_decode_steps: number; reference_ms_per_step_cpu: number }[];
    };
    const total = reference.staves.reduce((a, s) => a + s.reference_decode_steps, 0);
    console.log(`\nPython reference (CPU, on this machine):`);
    console.log(`  segnet + staff detection : ${reference.seg_seconds.toFixed(2)} s`);
    console.log(`  decode steps per staff   : ${reference.staves.map((s) => s.reference_decode_steps).join(", ")}`);
    console.log(`  decode ms/step (CPU)      : ${reference.staves.map((s) => s.reference_ms_per_step_cpu).join(", ")}`);
    console.log(`  mean steps/staff          : ${(total / reference.staves.length).toFixed(1)}`);
  }

  console.log(`\nbrowser measurements:`);
  console.log(
    ["model", "provider", "prec", "n", "create s", "cold s", "steady s", "ms/step"]
      .map((h, i) => h.padEnd([12, 9, 5, 5, 9, 8, 10, 9][i] ?? 8))
      .join(""),
  );
  for (const row of report.bench) {
    const cells = [
      row.key,
      row.provider,
      row.precision,
      String(row.steps ?? row.iterations),
      row.sessionCreateSeconds.toFixed(2),
      row.firstRunSeconds.toFixed(2),
      row.steadyRunSeconds.toFixed(2),
      row.msPerStep ? row.msPerStep.toFixed(1) : "-",
    ];
    console.log("  " + cells.map((c, i) => c.padEnd([12, 9, 5, 5, 9, 8, 10, 9][i] ?? 8)).join(""));
    if (row.error) console.log(`      ! ${row.error.slice(0, 120)}`);
  }

  if (report.opencv) {
    const absent = report.opencv.probes.filter((p) => !p.ok).map((p) => p.name);
    console.log(`\nOpenCV.js (${report.opencv.symbolCount} symbols):`);
    for (const probe of report.opencv.probes) {
      if (["CLAHE.apply", "findContours", "minAreaRect", "reduce", "Subdiv2D", "createCLAHE"].includes(
        probe.name,
      )) {
        console.log(`  ${probe.ok ? "present" : "ABSENT "}  ${probe.name.padEnd(14)} ${probe.detail.slice(0, 90)}`);
      }
    }
    console.log(`  all absent: ${absent.length ? absent.join(", ") : "none"}`);
  }
  if (report.opencvError) console.log(`\nOpenCV.js failed: ${report.opencvError}`);

  if (requestFailures.length) {
    console.log(`\nfailed requests (${requestFailures.length}):`);
    for (const f of requestFailures.slice(0, 15)) console.log(`  ${f}`);
  }

  console.log(`\nreport written to ${reportPath}`);
  expect(existsSync(reportPath)).toBe(true);
});

declare global {
  interface Window {
    __SPIKE__?: Record<string, unknown> & {
      bench: {
        key: string;
        provider: string;
        precision: string;
        sessionCreateSeconds: number;
        firstRunSeconds: number;
        steadyRunSeconds: number;
        iterations: number;
        steps?: number;
        msPerStep?: number;
        error?: string;
      }[];
      reference?: unknown;
      opencv?: {
        symbolCount: number;
        probes: { name: string; ok: boolean; detail: string }[];
      } | null;
      opencvError?: string;
    };
    __SPIKE_DONE__?: boolean;
  }
}