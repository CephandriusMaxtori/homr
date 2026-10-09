/**
 * Web App Main Controller.
 */

import ort, { configureOrt } from "./ort.ts";
import { loadManifest, preferredPrecision } from "./models.ts";
import { ModelCache } from "./model-cache.ts";
import { processImagePipeline } from "./pipeline.ts";

const root = document.getElementById("app");

if (root) {
  root.innerHTML = `
    <div class="card">
      <div id="dropzone" class="dropzone">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--muted);margin:0 auto">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <p><strong>Click to choose a sheet music image</strong> or drag & drop here</p>
        <input type="file" id="fileInput" accept="image/*" style="display:none" />
      </div>
      <div id="fileInfo" style="margin-top:1rem;display:none;font-weight:500;color:var(--text)"></div>
      <div class="actions">
        <button id="convertBtn" class="btn" disabled>Convert to MusicXML</button>
        <button id="downloadBtn" class="btn btn-secondary" style="display:none">Download MusicXML</button>
      </div>

      <div id="progressSection" style="display:none;margin-top:1rem">
        <div class="progress-bar-container">
          <div id="progressBar" class="progress-bar"></div>
        </div>
        <div id="statusText" class="status-text">Ready</div>
      </div>
    </div>

    <div id="resultSection" class="card" style="display:none">
      <h3 style="font-size:1.1rem;margin-bottom:0.75rem">Generated MusicXML Output</h3>
      <div id="xmlPreview" class="preview-area"></div>
    </div>
  `;
}

const fileInput = document.getElementById("fileInput") as HTMLInputElement | null;
const dropzone = document.getElementById("dropzone");
const fileInfo = document.getElementById("fileInfo");
const convertBtn = document.getElementById("convertBtn") as HTMLButtonElement | null;
const downloadBtn = document.getElementById("downloadBtn") as HTMLButtonElement | null;
const progressSection = document.getElementById("progressSection");
const progressBar = document.getElementById("progressBar");
const statusText = document.getElementById("statusText");
const resultSection = document.getElementById("resultSection");
const xmlPreview = document.getElementById("xmlPreview");

let selectedFile: File | null = null;
let lastResultXml: string | null = null;

if (dropzone && fileInput) {
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer?.files.length) {
      handleFileSelected(e.dataTransfer.files[0]!);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files?.length) {
      handleFileSelected(fileInput.files[0]!);
    }
  });
}

function handleFileSelected(file: File): void {
  selectedFile = file;
  if (fileInfo) {
    fileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.style.display = "block";
  }
  if (convertBtn) {
    convertBtn.disabled = false;
  }
}

if (convertBtn) {
  convertBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    convertBtn.disabled = true;
    if (progressSection) progressSection.style.display = "block";
    updateProgress("Initializing runtime...", 5);

    try {
      configureOrt();
      const manifest = await loadManifest("models.json");
      const hasWebGpu = typeof (navigator as any).gpu !== "undefined";
      const precision = preferredPrecision(hasWebGpu);

      const cache = new ModelCache(manifest, {
        onProgress: (loaded, total, label) => {
          const pct = total > 0 ? Math.round((loaded / total) * 30) + 5 : 15;
          updateProgress(`Loading model ${label}...`, pct);
        },
      });

      const segnetBytes = await cache.load("segnet", precision);
      const encoderBytes = await cache.load("encoder", precision);
      const decoderBytes = await cache.load("decoder", precision);

      updateProgress("Creating inference sessions...", 35);
      const segnetSession = await ort.InferenceSession.create(segnetBytes);
      const encoderSession = await ort.InferenceSession.create(encoderBytes);
      const decoderSession = await ort.InferenceSession.create(decoderBytes);

      updateProgress("Reading image...", 40);
      const imgBitmap = await createImageBitmap(selectedFile);
      const canvas = document.createElement("canvas");
      canvas.width = imgBitmap.width;
      canvas.height = imgBitmap.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(imgBitmap, 0, 0);
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      const grayData = new Uint8Array(canvas.width * canvas.height);
      for (let i = 0; i < grayData.length; i++) {
        const r = imgData.data[i * 4] ?? 0;
        const g = imgData.data[i * 4 + 1] ?? 0;
        const b = imgData.data[i * 4 + 2] ?? 0;
        grayData[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      }

      const resultXml = await processImagePipeline(
        { segnet: segnetSession, encoder: encoderSession, decoder: decoderSession },
        grayData,
        canvas.width,
        canvas.height,
        {
          precision,
          onProgress: ({ stage, percent }) => updateProgress(stage, percent),
        },
      );

      lastResultXml = resultXml;
      if (xmlPreview) xmlPreview.textContent = resultXml;
      if (resultSection) resultSection.style.display = "block";
      if (downloadBtn) downloadBtn.style.display = "inline-flex";

    } catch (err: any) {
      updateProgress(`Error: ${err?.message ?? String(err)}`, 100);
    } finally {
      convertBtn.disabled = false;
    }
  });
}

if (downloadBtn) {
  downloadBtn.addEventListener("click", () => {
    if (!lastResultXml) return;
    const blob = new Blob([lastResultXml], { type: "application/xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "result.musicxml";
    a.click();
    URL.revokeObjectURL(url);
  });
}

function updateProgress(message: string, percent: number): void {
  if (statusText) statusText.textContent = message;
  if (progressBar) progressBar.style.width = `${percent}%`;
}
