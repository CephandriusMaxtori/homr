/**
 * Bounding box extraction and spatial group merging.
 */

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RotatedBox {
  cx: number;
  cy: number;
  width: number;
  height: number;
  angle: number;
}

export class BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;

  constructor(x1: number, y1: number, x2: number, y2: number) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
  }

  get width(): number {
    return this.x2 - this.x1;
  }

  get height(): number {
    return this.y2 - this.y1;
  }

  get cx(): number {
    return (this.x1 + this.x2) / 2;
  }

  get cy(): number {
    return (this.y1 + this.y2) / 2;
  }

  overlaps(other: BoundingBox): boolean {
    return !(
      this.x2 < other.x1 ||
      this.x1 > other.x2 ||
      this.y2 < other.y1 ||
      this.y1 > other.y2
    );
  }

  expand(padding: number, maxWidth: number, maxHeight: number): BoundingBox {
    return new BoundingBox(
      Math.max(0, this.x1 - padding),
      Math.max(0, this.y1 - padding),
      Math.min(maxWidth, this.x2 + padding),
      Math.min(maxHeight, this.y2 + padding),
    );
  }
}

class UnionFind {
  parent: number[];
  rank: number[];

  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x: number): number {
    const parentVal = this.parent[x];
    if (parentVal !== undefined && parentVal !== x) {
      this.parent[x] = this.find(parentVal);
      return this.parent[x]!;
    }
    return x;
  }

  union(x: number, y: number): void {
    const rx = this.find(x);
    const ry = this.find(y);
    if (rx !== ry) {
      const rankRx = this.rank[rx] ?? 0;
      const rankRy = this.rank[ry] ?? 0;
      if (rankRx > rankRy) {
        this.parent[ry] = rx;
      } else if (rankRx < rankRy) {
        this.parent[rx] = ry;
      } else {
        this.parent[ry] = rx;
        this.rank[rx] = rankRx + 1;
      }
    }
  }
}

export function mergeOverlappingBoxes(boxes: BoundingBox[]): BoundingBox[] {
  if (boxes.length === 0) return [];
  const uf = new UnionFind(boxes.length);

  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const b1 = boxes[i];
      const b2 = boxes[j];
      if (b1 && b2 && b1.overlaps(b2)) {
        uf.union(i, j);
      }
    }
  }

  const groups = new Map<number, BoundingBox[]>();
  for (let i = 0; i < boxes.length; i++) {
    const box = boxes[i];
    if (!box) continue;
    const root = uf.find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root)!.push(box);
  }

  const merged: BoundingBox[] = [];
  for (const group of groups.values()) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const b of group) {
      if (b.x1 < minX) minX = b.x1;
      if (b.y1 < minY) minY = b.y1;
      if (b.x2 > maxX) maxX = b.x2;
      if (b.y2 > maxY) maxY = b.y2;
    }
    merged.push(new BoundingBox(minX, minY, maxX, maxY));
  }

  return merged;
}
