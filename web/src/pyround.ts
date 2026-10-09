/**
 * Audited rounding functions for Python banker's rounding, JS half-up rounding, and OpenCV cvRound.
 */

/**
 * Python's banker's rounding (round half to even).
 */
export function pyRound(x: number): number {
  if (!Number.isFinite(x)) return x;
  const floor = Math.floor(x);
  const diff = Math.abs(x - floor);
  if (Math.abs(diff - 0.5) < 1e-12) {
    return floor % 2 === 0 ? floor : floor + 1;
  }
  return Math.round(x);
}

/**
 * JS default half-up rounding.
 */
export function jsRound(x: number): number {
  return Math.round(x);
}

/**
 * OpenCV C++ cvRound (half away from zero).
 */
export function cvRound(x: number): number {
  return x >= 0 ? Math.floor(x + 0.5) : Math.ceil(x - 0.5);
}
