import { defineConfig } from "@playwright/test";

/**
 * Runs the Phase 0 spike page and writes the report to spike-report.json.
 *
 * WebGPU is requested explicitly. Headless Chromium can expose a software adapter
 * (SwiftShader) rather than the real GPU, which the report records so a slow WebGPU
 * number is never mistaken for a fast one.
 */
export default defineConfig({
  testDir: "./tests",
  timeout: 30 * 60 * 1000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5199",
    headless: true,
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,WebGPU",
            "--use-angle=default",
            "--ignore-gpu-blocklist",
            "--enable-gpu-rasterization",
          ],
        },
      },
    },
  ],
  webServer: {
    command: "npx vite --port 5199 --strictPort",
    url: "http://localhost:5199/spike.html",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});