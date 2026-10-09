/**
 * Staff dewarping and perspective correction using lattice-based triangulation.
 */

import { getCv, type OpenCvInstance } from "./cvwrap.ts";
import type { Staff } from "./staff_detection.ts";

export interface Point2D {
  x: number;
  y: number;
}

export function calculateSpanAndOptimalPoints(
  staff: Staff,
  width: number,
  height: number,
): [Point2D[][], Point2D[][]] {
  const spanPoints: Point2D[][] = [];
  const optimalPoints: Point2D[][] = [];

  const intervals = 6;
  const stepY = Math.max(1, Math.floor(height / intervals));
  let firstYOffset: number | null = null;

  for (let y = 2; y < height - 2; y += stepY) {
    const linePoints: Point2D[] = [];
    for (let x = 2; x < width; x += 80) {
      const staffPt = staff.getAt(x);
      if (staffPt && staffPt.y.length >= 3) {
        const yOffset = staffPt.y[2] ?? 0;
        if (firstYOffset === null) {
          firstYOffset = yOffset;
        }
        const yDelta = Math.round(yOffset - firstYOffset);
        const pt = { x, y: y + yDelta };
        if (pt.x >= 10 && pt.x <= width - 10 && pt.y >= 10 && pt.y <= height - 10) {
          linePoints.push(pt);
        }
      }
    }

    if (linePoints.length >= 2) {
      const avgY = Math.round(linePoints.reduce((acc, p) => acc + p.y, 0) / linePoints.length);
      spanPoints.push(linePoints);
      optimalPoints.push(linePoints.map((p) => ({ x: p.x, y: avgY })));
    }
  }

  return [spanPoints, optimalPoints];
}

export async function dewarpStaffCanvas(
  cv: OpenCvInstance,
  srcGray: Uint8Array,
  width: number,
  height: number,
  sourceRows: Point2D[][],
  destRows: Point2D[][],
): Promise<Uint8Array> {
  const addImageEdges = (rows: Point2D[][]): Point2D[][] => [
    [
      { x: 0, y: 0 },
      { x: width, y: 0 },
    ],
    ...rows,
    [
      { x: 0, y: height },
      { x: width, y: height },
    ],
  ];

  const addEdgePointsToRows = (rows: Point2D[][]): Point2D[][] =>
    rows.map((row) => {
      const first = row[0] ?? { x: 0, y: 0 };
      const last = row[row.length - 1] ?? { x: width, y: height };
      return [
        { x: 0, y: first.y },
        ...row,
        { x: width, y: last.y },
      ];
    });

  const srcGrid = addImageEdges(addEdgePointsToRows(sourceRows));
  const dstGrid = addImageEdges(addEdgePointsToRows(destRows));

  const output = new Uint8Array(srcGray);

  for (let r = 0; r < srcGrid.length - 1; r++) {
    const srcRow1 = srcGrid[r];
    const srcRow2 = srcGrid[r + 1];
    const dstRow1 = dstGrid[r];
    const dstRow2 = dstGrid[r + 1];

    if (!srcRow1 || !srcRow2 || !dstRow1 || !dstRow2) continue;

    const cols = Math.min(srcRow1.length, srcRow2.length, dstRow1.length, dstRow2.length) - 1;

    for (let c = 0; c < cols; c++) {
      const p1 = srcRow1[c];
      const p2 = srcRow1[c + 1];
      const p3 = srcRow2[c];
      const p4 = srcRow2[c + 1];

      const dp1 = dstRow1[c];
      const dp2 = dstRow1[c + 1];
      const dp3 = dstRow2[c];
      const dp4 = dstRow2[c + 1];

      if (!p1 || !p2 || !p3 || !p4 || !dp1 || !dp2 || !dp3 || !dp4) continue;

      const tri1Src = [p1, p2, p3];
      const tri1Dst = [dp1, dp2, dp3];

      const tri2Src = [p2, p4, p3];
      const tri2Dst = [dp2, dp4, dp3];

      warpTriangle(cv, srcGray, output, width, height, tri1Src, tri1Dst);
      warpTriangle(cv, srcGray, output, width, height, tri2Src, tri2Dst);
    }
  }

  return output;
}

function warpTriangle(
  cv: OpenCvInstance,
  srcGray: Uint8Array,
  output: Uint8Array,
  width: number,
  height: number,
  srcTri: Point2D[],
  dstTri: Point2D[],
): void {
  const p1 = srcTri[0];
  const p2 = srcTri[1];
  const p3 = srcTri[2];
  const dp1 = dstTri[0];
  const dp2 = dstTri[1];
  const dp3 = dstTri[2];

  if (!p1 || !p2 || !p3 || !dp1 || !dp2 || !dp3) return;

  const srcMat = (cv as any).matFromArray(3, 1, (cv as any).CV_32FC2, 1, [
    p1.x, p1.y,
    p2.x, p2.y,
    p3.x, p3.y,
  ]);
  const dstMat = (cv as any).matFromArray(3, 1, (cv as any).CV_32FC2, 1, [
    dp1.x, dp1.y,
    dp2.x, dp2.y,
    dp3.x, dp3.y,
  ]);

  try {
    const warpMat = (cv as any).getAffineTransform(srcMat, dstMat);
    const minX = Math.max(0, Math.floor(Math.min(dp1.x, dp2.x, dp3.x)));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(dp1.x, dp2.x, dp3.x)));
    const minY = Math.max(0, Math.floor(Math.min(dp1.y, dp2.y, dp3.y)));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(dp1.y, dp2.y, dp3.y)));

    const m = warpMat.data32F;
    if (m) {
      const invM00 = m[0] ?? 0;
      const invM01 = m[1] ?? 0;
      const invM02 = m[2] ?? 0;
      const invM10 = m[3] ?? 0;
      const invM11 = m[4] ?? 0;
      const invM12 = m[5] ?? 0;

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const srcX = Math.round(invM00 * x + invM01 * y + invM02);
          const srcY = Math.round(invM10 * x + invM11 * y + invM12);
          if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            output[y * width + x] = srcGray[srcY * width + srcX] ?? 255;
          }
        }
      }
    }
    warpMat.delete();
  } catch {
    // ignore degenerate triangle
  } finally {
    srcMat.delete();
    dstMat.delete();
  }
}
