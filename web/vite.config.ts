import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
  base: "./",
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    target: "es2022",
    // onnxruntime-web and opencv-js are large prebuilt WASM payloads.
    // Keep them out of the JS chunk so the app code can load first.
    rollupOptions: {
      output: {
        // Split the two large prebuilt runtimes into their own chunks so the app
        // code can parse while they download.
        manualChunks(id: string) {
          if (id.includes("onnxruntime")) return "ort";
          if (id.includes("@techstark/opencv-js")) return "opencv";
          return undefined;
        },
      },
    },
  },
  // The ORT WASM binaries and the OpenCV .wasm are copied into public/ by
  // scripts/copy-wasm.mjs; serve them as-is.
  assetsInclude: ["**/*.wasm"],
  server: {
    fs: { strict: false },
  },
  worker: { format: "es" },
});