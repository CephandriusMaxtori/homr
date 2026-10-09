/**
 * Minimal reader for NumPy `.npy` files (format v1.0/v2.0, C-order, 1-2 dims).
 *
 * The spike consumes tensors dumped by scripts/dump_spike_tensors.py, so we only
 * need to handle what that script writes: float32 and float16, shapes like
 * [1, 256, 1280] and [1, 1, 256, 1280].
 */

export interface NpyArray {
  dtype: "float32" | "float16";
  shape: number[];
  data: Float32Array | Uint16Array;
}

const MAGIC = 0x93;

export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const view = new DataView(buffer);
  if (view.getUint8(0) !== MAGIC) throw new Error("not a .npy file");
  const major = view.getUint8(6);

  let headerStart: number;
  let headerLength: number;
  if (major === 1) {
    headerLength = view.getUint16(8, true);
    headerStart = 10;
  } else {
    headerLength = view.getUint32(8, true);
    headerStart = 12;
  }

  const header = new TextDecoder()
    .decode(new Uint8Array(buffer, headerStart, headerLength))
    .trim();

  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortranOrder = /'fortran_order':\s*(True|False)/.exec(header)?.[1] === "True";
  const shape = (
    /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? ""
  )
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);

  if (fortranOrder) throw new Error("Fortran-ordered .npy is not supported");
  if (descr !== "<f4" && descr !== "<f2" && descr !== "|f4" && descr !== "|f2") {
    throw new Error(`unsupported dtype '${descr}' (expected little-endian f2 or f4)`);
  }

  const dataStart = headerStart + headerLength;
  const elementSize = descr.endsWith("f2") ? 2 : 4;
  const count = shape.reduce((a, b) => a * b, 1);
  const bytes = new Uint8Array(buffer, dataStart, count * elementSize);

  const data: Float32Array | Uint16Array =
    elementSize === 2
      ? new Uint16Array(bytes.buffer, bytes.byteOffset, count)
      : new Float32Array(bytes.buffer, bytes.byteOffset, count);

  return { dtype: elementSize === 2 ? "float16" : "float32", shape, data };
}

export async function fetchNpy(url: string): Promise<NpyArray> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch ${url}: HTTP ${response.status}`);
  return parseNpy(await response.arrayBuffer());
}