// Poll the spike page's own state from outside the browser, so a long run can be
// watched without waiting blind. Reads the status text and log the page exposes.
//
//   node scripts/watch-spike.mjs [url]

import { chromium } from "playwright";

const url = process.argv[2] ?? "http://localhost:5199/spike.html";

// These flags match playwright.config.ts. A run with only
// --enable-unsafe-webgpu + --ignore-gpu-blocklist got "requestAdapter() returned
// null" on this machine, while the fuller set resolved an adapter
// (intel/gen-12lp), so keep the two in sync.
const browser = await chromium.launch({
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=default",
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
  ],
});
const page = await browser.newPage();

/** @type {Map<string, {count: number, bytes: number, done: boolean}>} */
const requests = new Map();
page.on("response", async (response) => {
  const url = response.url();
  const length = Number(response.headers()["content-length"] ?? 0);
  const entry = requests.get(url) ?? { count: 0, bytes: 0, done: false };
  entry.count++;
  entry.bytes += length;
  entry.done = true;
  requests.set(url, entry);
});

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });

const started = Date.now();
let lastLogLength = 0;

for (;;) {
  const snapshot = await page
    .evaluate(() => {
      const status = document.getElementById("status")?.textContent ?? "";
      const log = document.getElementById("log")?.textContent ?? "";
      return { status, log, done: window.__SPIKE_DONE__ === true };
    })
    .catch(() => null);

  if (!snapshot) {
    console.log("page gone");
    break;
  }

  // Print only new log lines so progress is legible.
  if (snapshot.log.length > lastLogLength) {
    const fresh = snapshot.log.slice(lastLogLength).trimEnd();
    for (const line of fresh.split("\n")) console.log(`  ${line}`);
    lastLogLength = snapshot.log.length;
  }

  const elapsed = ((Date.now() - started) / 1000).toFixed(0);
  console.log(`[${elapsed}s] status="${snapshot.status}"  (${requests.size} distinct responses)`);

  if (snapshot.done) {
    const report = await page.evaluate(() => window.__SPIKE__);
    console.log("\n--- finished ---");
    console.log(JSON.stringify(report?.bench ?? [], null, 2));
    break;
  }
  await new Promise((r) => setTimeout(r, 5000));
}

await browser.close();