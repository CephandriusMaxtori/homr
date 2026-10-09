/**
 * Browser-side model cache.
 *
 * The checkpoints total ~98 MB. Re-downloading them on every page load is not
 * acceptable, so bytes are kept in the Cache Storage API keyed by the full URL.
 *
 * Keying on the URL is deliberate: `models.json` pins a `revision`, so a model swap
 * changes the URL and therefore the cache key. A stale entry can never be served by
 * accident, and no explicit invalidation step is needed.
 *
 * onnxruntime-web will `fetch()` a model URL itself, but it does not persist what it
 * downloads, so we materialise the bytes once and hand ORT an ArrayBuffer.
 */

import type { ModelKey, ModelManifest, Precision } from "./models.ts";
import { modelUrl } from "./models.ts";

const CACHE_NAME = "homr-models-v1";

/** Show progress for downloads large enough that silence would look like a hang. */
export interface ProgressOptions {
  onProgress?: (loadedBytes: number, totalBytes: number, label: string) => void;
}

function isCacheStorageAvailable(): boolean {
  return typeof caches !== "undefined";
}

/** Total size if the server reports it, else 0 when unknown. */
async function contentLength(response: Response): Promise<number> {
  const header = response.headers.get("content-length");
  return header ? Number(header) : 0;
}

export class ModelCache {
  private readonly manifest: ModelManifest;
  private readonly progress: ProgressOptions;
  /** Session cache, so a second image in the same page does not re-download. */
  private readonly bytes = new Map<string, ArrayBuffer>();

  constructor(manifest: ModelManifest, progress: ProgressOptions = {}) {
    this.manifest = manifest;
    this.progress = progress;
  }

  private url(key: ModelKey, precision: Precision): string {
    return modelUrl(this.manifest, key, precision);
  }

  /**
   * Return the model bytes, from memory, then the Cache Storage API, then network.
   */
  async load(key: ModelKey, precision: Precision): Promise<ArrayBuffer> {
    const url = this.url(key, precision);
    const label = `${key}-${precision}`;

    const inMemory = this.bytes.get(url);
    if (inMemory) return inMemory;

    if (isCacheStorageAvailable()) {
      const cache = await caches.open(CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        const buffer = await hit.arrayBuffer();
        this.bytes.set(url, buffer);
        this.progress.onProgress?.(buffer.byteLength, buffer.byteLength, `${label} (cached)`);
        return buffer;
      }
    }

    const buffer = await this.download(url, label);
    this.bytes.set(url, buffer);

    if (isCacheStorageAvailable()) {
      const cache = await caches.open(CACHE_NAME);
      // Store a fresh Response so a partial body can never be persisted.
      await cache.put(url, new Response(buffer, { headers: { "content-type": "application/octet-stream" } }));
    }
    return buffer;
  }

  private async download(url: string, label: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Could not download ${label} from ${url}: HTTP ${response.status}`);
    }

    const total = await contentLength(response);

    // Stream when we can, so progress is real rather than an indeterminate spinner.
    if (!response.body || typeof ReadableStreamDefaultReader === "undefined") {
      const buffer = await response.arrayBuffer();
      this.progress.onProgress?.(buffer.byteLength, buffer.byteLength, label);
      return buffer;
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    let lastReport = 0;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        loaded += value.byteLength;
        // Throttle: a callback per chunk would dominate the cost for 98 MB.
        if (loaded - lastReport > 1_000_000) {
          lastReport = loaded;
          this.progress.onProgress?.(loaded, total, label);
        }
      }
    }

    const buffer = new ArrayBuffer(loaded);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) {
      view.set(chunk, offset);
      offset += chunk.byteLength;
    }
    this.progress.onProgress?.(loaded, loaded, label);
    return buffer;
  }

  /** Drop everything. For a "clear cached models" control in the UI. */
  async clear(): Promise<void> {
    this.bytes.clear();
    if (isCacheStorageAvailable()) {
      await caches.delete(CACHE_NAME);
    }
  }

  /** Bytes already resident, for a "models ready" summary in the UI. */
  residentBytes(): number {
    let total = 0;
    for (const buffer of this.bytes.values()) total += buffer.byteLength;
    return total;
  }
}