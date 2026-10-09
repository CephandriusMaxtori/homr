/**
 * Staff line detection, anchor finding, and staff resampling.
 */

import { number_of_lines_on_a_staff } from "./constants.ts";
import { getCv } from "./cvwrap.ts";

export interface StaffPoint {
  x: number;
  y: number[];
  angle: number;
}

export class Staff {
  grid: StaffPoint[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;

  constructor(grid: StaffPoint[]) {
    this.grid = grid;
    const first = grid[0];
    const last = grid[grid.length - 1];
    this.minX = first ? first.x : 0;
    this.maxX = last ? last.x : 0;
    let miny = Infinity;
    let maxy = -Infinity;
    for (const pt of grid) {
      for (const y of pt.y) {
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
      }
    }
    this.minY = Number.isFinite(miny) ? miny : 0;
    this.maxY = Number.isFinite(maxy) ? maxy : 0;
  }

  getAt(x: number): StaffPoint | null {
    if (this.grid.length === 0) return null;
    let low = 0;
    let high = this.grid.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      const pt = this.grid[mid];
      if (pt && pt.x === x) return pt;
      if (pt && pt.x < x) low = mid + 1;
      else high = mid - 1;
    }
    const idx = Math.max(0, Math.min(this.grid.length - 1, low));
    return this.grid[idx] ?? null;
  }
}

export async function findHorizontalLinesCv(
  imageMask: Uint8Array,
  width: number,
  height: number,
): Promise<number[][]> {
  const cv = await getCv();
  const src = (cv as any).matFromArray(height, width, (cv as any).CV_8UC1, 1, imageMask);
  const dst = new (cv as any).Mat();

  (cv as any).reduce(src, dst, 1, (cv as any).REDUCE_SUM, (cv as any).CV_32S);

  const rowSums = new Int32Array(dst.data32S ? dst.data32S.buffer : dst.data_u8.buffer);

  let mean = 0;
  for (let i = 0; i < height; i++) mean += rowSums[i] ?? 0;
  mean /= height;

  let variance = 0;
  for (let i = 0; i < height; i++) variance += ((rowSums[i] ?? 0) - mean) ** 2;
  const std = Math.sqrt(variance / height) || 1;

  const peaks: number[] = [];
  for (let y = 1; y < height - 1; y++) {
    const valY = rowSums[y] ?? 0;
    const valPrev = rowSums[y - 1] ?? 0;
    const valNext = rowSums[y + 1] ?? 0;
    const norm = (valY - mean) / std;
    if (norm > 1.0 && valY > valPrev && valY >= valNext) {
      peaks.push(y);
    }
  }

  src.delete();
  dst.delete();

  const completeGroups: number[][] = [];
  for (let i = 0; i + number_of_lines_on_a_staff <= peaks.length; i += number_of_lines_on_a_staff) {
    completeGroups.push(peaks.slice(i, i + number_of_lines_on_a_staff));
  }
  return completeGroups;
}
