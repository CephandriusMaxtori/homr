/**
 * Typed wrapper over @techstark/opencv-js.
 */

import cvReady from "@techstark/opencv-js";

export type OpenCvInstance = typeof cvReady;

let cachedCv: OpenCvInstance | null = null;

export async function getCv(): Promise<OpenCvInstance> {
  if (cachedCv) return cachedCv;
  const candidate = await (cvReady as unknown as Promise<OpenCvInstance>);
  cachedCv = candidate;
  return cachedCv;
}

export function matFromArray(
  cv: OpenCvInstance,
  rows: number,
  cols: number,
  type: number,
  data: number[] | Uint8Array | Float32Array,
): any {
  return (cv as any).matFromArray(rows, cols, type, 1, data);
}
