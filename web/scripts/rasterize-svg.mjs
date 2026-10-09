// Rasterise an SVG to PNG at a fixed width using Chromium's own SVG renderer.
// Used to produce a real sheet-music page for the Phase 0 spike: figures/tabi.svg is
// a real MuseScore engraving, so it is a genuine OMR input.
//
//   node scripts/rasterize-svg.mjs <input.svg> <output.png> [width]

import { chromium } from "playwright";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const [, , inputArg, outputArg, widthArg] = process.argv;
if (!inputArg || !outputArg) {
  console.error("usage: node scripts/rasterize-svg.mjs <input.svg> <output.png> [width]");
  process.exit(1);
}
const width = Number(widthArg ?? 1920);

const svg = readFileSync(resolve(inputArg), "utf8");

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width, height: 1200 } });

await page.setContent(
  `<!doctype html><html><body style="margin:0;background:#fff">
     <div id="wrap" style="width:${width}px">${svg}</div>
   </body></html>`,
  { waitUntil: "load" },
);

const svgEl = page.locator("#wrap svg");
await svgEl.waitFor({ state: "attached" });

// Force the SVG to the target width and read back its intrinsic height.
const height = await page.evaluate((w) => {
  const el = document.querySelector("#wrap svg");
  const viewBox = el.viewBox?.baseVal;
  const ratio = viewBox && viewBox.height ? viewBox.height / viewBox.width : 1.414;
  el.setAttribute("width", String(w));
  el.setAttribute("height", String(Math.round(w * ratio)));
  el.style.width = `${w}px`;
  el.style.height = `${Math.round(w * ratio)}px`;
  return Math.round(w * ratio);
}, width);

await page.setViewportSize({ width, height });
await page.locator("#wrap").screenshot({ path: resolve(outputArg), type: "png" });

console.log(`wrote ${outputArg} at ${width}x${height}`);
await browser.close();