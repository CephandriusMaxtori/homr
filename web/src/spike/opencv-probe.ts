/**
 * Phase 0 spike: probe what the OpenCV.js WASM build can actually do.
 *
 * The export list of @techstark/opencv-js 5.0.0 was read statically during planning
 * and two gaps were found: `Subdiv2D` is absent (only its PTLOC_* enum constants are
 * exported), and the `createCLAHE` factory is absent though the `CLAHE` class itself
 * is exported. This file confirms that at runtime, plus the API-shape questions that
 * decide how much wrapper code the port needs.
 *
 * Everything here is a yes/no or a shape probe. Nothing is timed.
 */

import cvReady from "@techstark/opencv-js";

type ProbeResult = {
  name: string;
  question: string;
  ok: boolean;
  detail: string;
};

export interface OpenCvReport {
  version: string;
  symbolCount: number;
  probes: ProbeResult[];
}

let cached: OpenCvReport | null = null;

/**
 * A Mat as the JS binding exposes it.
 *
 * Note `data_u8` is only populated for some constructors — `new cv.Mat(rows, cols,
 * type)` does not expose it reliably, whereas `matFromArray` does. Prefer
 * matFromArray when the contents matter.
 */
type MatLike = {
  rows: number;
  cols: number;
  data_u8?: Uint8Array;
  data32F?: Float32Array;
  delete(): void;
};

function describeError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

/** OpenCV.js exposes a factory default that may itself be a promise. */
export async function loadOpenCv(): Promise<typeof cvReady> {
  const candidate = await (cvReady as unknown as Promise<typeof cvReady>);
  return candidate;
}

export async function probeOpenCv(): Promise<OpenCvReport> {
  if (cached) return cached;

  const cv = await loadOpenCv();
  const probes: ProbeResult[] = [];

  const has = (name: string, fn: () => unknown, question: string): void => {
    try {
      const value = fn();
      const kind =
        value === null || value === undefined
          ? "null"
          : typeof value === "function"
            ? "function"
            : value?.constructor?.name ?? typeof value;
      probes.push({ name, question, ok: value !== undefined, detail: String(kind) });
    } catch (error) {
      probes.push({ name, question, ok: false, detail: describeError(error) });
    }
  };

  const buildInfo =
    (cv as unknown as { cv?: { getBuildInformation?: () => string } }).cv?.getBuildInformation?.() ??
    (cv as unknown as { getBuildInformation?: () => string }).getBuildInformation?.() ??
    "";
  const version = buildInfo.split("\n")[0] ?? "unknown";

  // --- The two known gaps -------------------------------------------------
  has("createCLAHE", () => (cv as unknown as Record<string, unknown>)["createCLAHE"], "Is the createCLAHE factory present?");
  has("CLAHE", () => (cv as unknown as Record<string, unknown>)["CLAHE"], "Is the CLAHE class present (so it can be constructed directly)?");
  has("Subdiv2D", () => (cv as unknown as Record<string, unknown>)["Subdiv2D"], "Is Subdiv2D (Delaunay) present? Expected: NO.");
  has("Subdiv2D_PTLOC_VERTEX", () => (cv as unknown as Record<string, unknown>)["Subdiv2D_PTLOC_VERTEX"], "Are the Subdiv2D enum constants present (implies the class was registered without its methods)?");

  // --- CLAHE construction: on the critical path at main.py:124 and :209 ----
  // The emscripten binding takes (clipLimit, tilesX, tilesY) but the reported
  // parameter count is discovered at runtime rather than assumed.
  try {
    type Clahe = { apply(src: unknown, dst: unknown): void; getClipLimit(): number };
    const cvx = cv as unknown as {
      CLAHE?: new (...values: number[]) => Clahe;
      matFromArray: (
        rows: number,
        cols: number,
        type: number,
        channels: number,
        data: ArrayLike<number>,
      ) => MatLike;
      CV_8UC1: number;
    };
    if (typeof cvx.CLAHE !== "function") throw new Error("cv.CLAHE is not constructible");

    // Try the documented arity first, then shorter ones, so the report says what the
    // binding actually accepts rather than just "it failed".
    let clahe: Clahe | undefined;
    const attempts: number[][] = [
      [1.0, 8, 8],
      [1.0, 8],
      [1.0],
    ];
    const errors: string[] = [];
    const CLAHE_ANY = cvx.CLAHE;
    for (const args of attempts) {
      try {
        clahe = new CLAHE_ANY(...args);
        break;
      } catch (error) {
        errors.push(`${args.length} args: ${describeError(error).slice(0, 70)}`);
      }
    }
    if (!clahe) throw new Error(errors.join(" | "));
    const claheFilter = clahe;

    const src = cvx.matFromArray(8, 8, cvx.CV_8UC1, 1, new Uint8Array(64).fill(120));
    const dst = cvx.matFromArray(8, 8, cvx.CV_8UC1, 1, new Uint8Array(64));
    claheFilter.apply(src, dst);
    const value = dst.data_u8?.[0] ?? -1;
    probes.push({
      name: "CLAHE.apply",
      question: "Can cv.CLAHE be constructed and applied?",
      ok: Number.isFinite(value) && value >= 0,
      detail: `8x8 grey tile -> first pixel ${value}, clipLimit=${claheFilter.getClipLimit()}`,
    });
    src.delete();
    dst.delete();
  } catch (error) {
    probes.push({
      name: "CLAHE.apply",
      question: "Can cv.CLAHE be constructed and applied?",
      ok: false,
      detail: describeError(error).slice(0, 160),
    });
  }

  // --- findContours return shape -----------------------------------------
  // Python does `contours, hierarchy = cv2.findContours(...)`. In OpenCV 4.x the
  // JS binding returns only `contours`; in 5.x it returns both. The port needs to know.
  try {
    const cvx = cv as unknown as {
      Mat: new (rows?: number, cols?: number, type?: number) => MatLike;
      MatVector: new () => {
        size(): number;
        get(i: number): MatLike;
        delete(): void;
      };
      matFromArray: (
        rows: number,
        cols: number,
        type: number,
        channels: number,
        data: ArrayLike<number>,
      ) => MatLike;
      findContours: (
        image: unknown,
        contours: unknown,
        hierarchy: unknown,
        mode: number,
        method: number,
      ) => void;
      RETR_TREE: number;
      CHAIN_APPROX_SIMPLE: number;
      CV_8UC1: number;
    };
    // A 20x20 image with a solid 8x8 white square in the middle: exactly one contour.
    const src = cvx.matFromArray(20, 20, cvx.CV_8UC1, 1, (() => {
      const px = new Uint8Array(20 * 20);
      for (let y = 6; y < 14; y++) for (let x = 6; x < 14; x++) px[y * 20 + x] = 255;
      return px;
    })());

    // The binding takes out-params (this is the form the typings declare), so the
    // call returns void and fills the MatVector.
    const contours = new cvx.MatVector();
    const hierarchy = new cvx.Mat(0, 0, cvx.CV_8UC1);
    cvx.findContours(src, contours, hierarchy, cvx.RETR_TREE, cvx.CHAIN_APPROX_SIMPLE);
    const count = contours.size();
    const first = count > 0 ? contours.get(0) : undefined;

    probes.push({
      name: "findContours",
      question: "Does findContours use out-params (MatVector) and how is it indexed?",
      ok: count === 1,
      detail: `out-param form: contours.size()=${count} (want 1), ` +
        `first=${first ? `${first.rows}x${first.cols}` : "none"}, ` +
        `has.size=${typeof contours.size === "function"} has.get=${typeof contours.get === "function"}`,
    });

    if (first) first.delete();
    contours.delete();
    hierarchy.delete();
    src.delete();
  } catch (error) {
    probes.push({
      name: "findContours",
      question: "Does findContours use out-params (MatVector) and how is it indexed?",
      ok: false,
      detail: describeError(error).slice(0, 160),
    });
  }

  // --- RotatedRect ergonomics ---------------------------------------------
  // bounding_boxes.py destructures the Python tuple (centre, size, angle) in ~40 places.
  try {
    const cvx = cv as unknown as {
      Mat: new (rows: number, cols: number, type: number) => MatLike;
      matFromArray: (
        rows: number,
        cols: number,
        type: number,
        channels: number,
        data: ArrayLike<number>,
      ) => MatLike;
      minAreaRect: (points: unknown) => unknown;
      boxPoints: (rect: unknown) => unknown;
      CV_32FC2: number;
    };
    // A rotated rectangle, so the fitted angle is not degenerate.
    const pts = cvx.matFromArray(4, 1, cvx.CV_32FC2, 1, [
      10.0, 10.0, 40.0, 16.0, 38.0, 30.0, 8.0, 24.0,
    ]);
    const rect = cvx.minAreaRect(pts) as Record<string, unknown>;
    const keys = Object.keys(rect);
    const center = rect["center"] as { x?: number; y?: number } | undefined;
    const size = rect["size"] as { width?: number; height?: number } | undefined;
    probes.push({
      name: "minAreaRect",
      question: "Is the return a named-field object (not an indexable tuple)?",
      ok: keys.includes("center") && keys.includes("size") && keys.includes("angle"),
      detail: `keys=[${keys.join(",")}] tupleIndexable=${
        (rect as Record<number, unknown>)[0] !== undefined
      } center=(${center?.x?.toFixed(2)},${center?.y?.toFixed(2)}) size=(${size?.width?.toFixed(2)},${size?.height?.toFixed(2)}) angle=${rect["angle"]}`,
    });
    pts.delete();
  } catch (error) {
    probes.push({
      name: "minAreaRect",
      question: "Is the return a named-field object (not an indexable tuple)?",
      ok: false,
      detail: describeError(error).slice(0, 160),
    });
  }

  // --- cv.reduce: replaces the per-pixel Python loop in find_horizontal_lines
  try {
    const cvx = cv as unknown as {
      matFromArray: (
        rows: number,
        cols: number,
        type: number,
        channels: number,
        data: ArrayLike<number>,
      ) => MatLike;
      reduce: (src: unknown, dst: unknown, dim: number, rtype: number, dtype: number) => void;
      REDUCE_SUM: number;
      CV_16U: number;
      CV_8UC1: number;
    };
    // Column 0 is 3,2,0,5 = 10; every other column is 1,1,1,1 = 4.
    const src = cvx.matFromArray(4, 5, cvx.CV_8UC1, 1, [
      3, 1, 1, 1, 1,
      2, 1, 1, 1, 1,
      0, 1, 1, 1, 1,
      5, 1, 1, 1, 1,
    ]);
    const dst = cvx.matFromArray(1, 5, cvx.CV_16U, 1, new Uint16Array(5));
    // dim=0 reduces across rows, giving a 1x5 row of column sums.
    cvx.reduce(src, dst, 0, cvx.REDUCE_SUM, cvx.CV_16U);
    const sums = Array.from(dst.data_u8?.slice(0, 5) ?? []);
    probes.push({
      name: "reduce",
      question: "Does cv.reduce replace the find_horizontal_lines Python loop?",
      ok: sums[0] === 10 && sums[4] === 4,
      detail: `column sums = [${sums.join(",")}] (want 10,4,4,4,4)`,
    });
    src.delete();
    dst.delete();
  } catch (error) {
    probes.push({
      name: "reduce",
      question: "Does cv.reduce replace the find_horizontal_lines Python loop?",
      ok: false,
      detail: describeError(error).slice(0, 160),
    });
  }

  // --- MatVector ergonomics ------------------------------------------------
  // Nearly every geometry result (contours, boxPoints, intersectConvexConvex) comes
  // back as a MatVector or a tuple, so how they are indexed decides the wrapper.
  try {
    const cvx = cv as unknown as {
      MatVector: new () => {
        size(): number;
        get(i: number): MatLike;
        push_back(v: unknown): void;
        delete(): void;
      };
      matFromArray: (
        rows: number,
        cols: number,
        type: number,
        channels: number,
        data: ArrayLike<number>,
      ) => MatLike;
      CV_32FC2: number;
    };
    const mv = new cvx.MatVector();
    const empty = new cvx.MatVector();
    const capacity = typeof mv.size === "function";
    const indexable = typeof mv.get === "function";
    // Writing then reading back is what a contour iteration loop needs.
    let roundTrip = "no";
    if (capacity && indexable) {
      const mat = cvx.matFromArray(1, 2, cvx.CV_32FC2, 1, [1.5, 2.5]);
      mv.push_back(mat);
      const got = mv.size() > 0 ? mv.get(0) : undefined;
      roundTrip =
        got && got.rows === 1 && got.cols === 2
          ? `push_back/get ok ${got.rows}x${got.cols}`
          : "push_back/get failed";
      if (got) got.delete();
      mat.delete();
    }
    mv.delete();
    empty.delete();
    probes.push({
      name: "MatVector",
      question: "Is MatVector usable (size/get/push_back) for contour iteration?",
      ok: capacity && indexable,
      detail: `${roundTrip}, .size=${capacity}, .get=${indexable}`,
    });
  } catch (error) {
    probes.push({
      name: "MatVector",
      question: "Is MatVector usable (size/get/push_back) for contour iteration?",
      ok: false,
      detail: describeError(error).slice(0, 160),
    });
  }

  // --- boxPoints: MatVector or Point2f[]? ----------------------------------
  try {
    const cvx = cv as unknown as {
      Mat: new (rows: number, cols: number, type: number) => MatLike;
      matFromArray: (
        rows: number,
        cols: number,
        type: number,
        channels: number,
        data: ArrayLike<number>,
      ) => MatLike;
      minAreaRect: (points: unknown) => unknown;
      boxPoints: (rect: unknown) => unknown;
      CV_32FC2: number;
    };
    const pts = cvx.matFromArray(4, 1, cvx.CV_32FC2, 1, [
      10.0, 10.0, 40.0, 16.0, 38.0, 30.0, 8.0, 24.0,
    ]);
    const rect = cvx.minAreaRect(pts) as {
      center?: { x: number; y: number };
      size?: { width: number; height: number };
      angle?: number;
    };
    const result = cvx.boxPoints(rect) as
      | {
          size?: () => number;
          get?: (i: number) => MatLike;
          length?: number;
          delete?: () => void;
        }
      | unknown[];
    const asVector = result as {
      size?: () => number;
      get?: (i: number) => MatLike;
      length?: number;
      delete?: () => void;
    };
    const isMatVector = typeof asVector.size === "function" && typeof asVector.get === "function";
    const isArray = Array.isArray(result);
    let shape = `unknown (typeof ${typeof result}, keys=[${Object.keys(result as object).join(",")}])`;
    if (isMatVector) {
      const n = asVector.size!();
      const first = n > 0 ? asVector.get!(0) : undefined;
      const coords = first?.data32F
        ? Array.from(first.data32F.slice(0, 8))
            .map((v) => v.toFixed(1))
            .join(",")
        : "no data32F";
      shape = `MatVector size=${n} first=[${coords}]`;
      if (first) first.delete();
    } else if (isArray) {
      shape = `Array length=${(result as unknown[]).length} first=${JSON.stringify(
        (result as unknown[])[0],
      )}`;
    }
    probes.push({
      name: "boxPoints",
      question: "Does boxPoints return a MatVector or a Point2f[] array?",
      ok: isMatVector || isArray,
      detail: `center=(${rect.center?.x?.toFixed(2)},${rect.center?.y?.toFixed(2)}) size=(${rect.size?.width?.toFixed(2)},${rect.size?.height?.toFixed(2)}) angle=${rect.angle?.toFixed(2)} -> ${shape}`,
    });
    if (typeof asVector.delete === "function") asVector.delete();
    pts.delete();
  } catch (error) {
    probes.push({
      name: "boxPoints",
      question: "Does boxPoints return a MatVector or a Point2f[] array?",
      ok: false,
      detail: describeError(error).slice(0, 160),
    });
  }

  // --- The rest of the cv2 surface the port relies on ----------------------
  for (const [name, question] of [
    ["threshold", "Is threshold available?"],
    ["adaptiveThreshold", "Is adaptiveThreshold available (staff_position_save_load only)?"],
    ["morphologyEx", "Is morphologyEx available?"],
    ["getStructuringElement", "Is getStructuringElement available?"],
    ["dilate", "Is dilate available?"],
    ["erode", "Is erode available?"],
    ["bitwise_and", "Is bitwise_and available?"],
    ["subtract", "Is subtract available?"],
    ["filter2D", "Is filter2D available (noise_filtering Laplacian)?"],
    ["fitEllipse", "Is fitEllipse available (notehead ellipses)?"],
    ["ellipse2Poly", "Is ellipse2Poly available?"],
    ["intersectConvexConvex", "Is intersectConvexConvex available (box overlap)?"],
    ["rotatedRectangleIntersection", "Is rotatedRectangleIntersection available (staff anchors)?"],
    ["pointPolygonTest", "Is pointPolygonTest available?"],
    ["getAffineTransform", "Is getAffineTransform available (dewarper)?"],
    ["invertAffineTransform", "Is invertAffineTransform available?"],
    ["warpAffine", "Is warpAffine available (dewarper)?"],
    ["fillConvexPoly", "Is fillConvexPoly available?"],
    ["calcHist", "Is calcHist available (autocrop)?"],
    ["contourArea", "Is contourArea available (autocrop)?"],
    ["boundingRect", "Is boundingRect available?"],
    ["HoughLinesP", "Is HoughLinesP available (--read-staff-positions only)?"],
  ] as const) {
    has(name, () => (cv as unknown as Record<string, unknown>)[name], question);
  }

  const symbolCount = Object.keys(cv as unknown as object).length;
  cached = { version, symbolCount, probes };
  return cached as OpenCvReport;
}