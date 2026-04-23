/**
 * One Euro Filter — adaptive low-pass filter for noisy real-time signals.
 * Low cutoff when signal is slow (aggressive smoothing), high cutoff when
 * signal is fast (low lag). Reference: https://hal.inria.fr/hal-00670496
 */

class ScalarFilter {
  private alpha: number;
  private y: number | null = null;

  constructor(alpha: number) {
    this.alpha = alpha;
  }

  setAlpha(alpha: number) {
    this.alpha = alpha;
  }

  filter(x: number): number {
    if (this.y === null) {
      this.y = x;
    } else {
      this.y = this.alpha * x + (1 - this.alpha) * this.y;
    }
    return this.y;
  }

  get last(): number | null {
    return this.y;
  }
}

function computeAlpha(cutoff: number, freq: number): number {
  const te = 1 / freq;
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / te);
}

export class OneEuroFilter {
  private freq: number;
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  private xFilt: ScalarFilter;
  private dxFilt: ScalarFilter;

  constructor(minCutoff = 1.0, beta = 0.3, dCutoff = 1.0, freq = 30) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilt  = new ScalarFilter(computeAlpha(minCutoff, freq));
    this.dxFilt = new ScalarFilter(computeAlpha(dCutoff, freq));
  }

  filter(x: number, timestamp?: number): number {
    // Optionally update frequency from wall-clock timestamps
    if (timestamp !== undefined) {
      // freq update handled externally via LandmarksFilter
    }

    const prevX = this.xFilt.last;
    const dx = prevX !== null ? (x - prevX) * this.freq : 0;
    const edx = this.dxFilt.filter(dx);

    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    this.xFilt.setAlpha(computeAlpha(cutoff, this.freq));
    return this.xFilt.filter(x);
  }
}

/**
 * Manages one OneEuroFilter per coordinate (x, y, z) per landmark.
 * Call filter() each frame to get smoothed landmarks.
 */
import type { PoseLandmarks } from "@/hooks/usePoseLandmarker";

export class LandmarksFilter {
  private filters: [OneEuroFilter, OneEuroFilter, OneEuroFilter][];

  constructor(minCutoff = 1.0, beta = 0.3, dCutoff = 1.0, freq = 30) {
    this.filters = Array.from({ length: 33 }, () => [
      new OneEuroFilter(minCutoff, beta, dCutoff, freq),
      new OneEuroFilter(minCutoff, beta, dCutoff, freq),
      new OneEuroFilter(minCutoff, beta, dCutoff, freq),
    ]);
  }

  filter(lms: PoseLandmarks): PoseLandmarks {
    return lms.map((lm, i) => ({
      x: this.filters[i][0].filter(lm.x),
      y: this.filters[i][1].filter(lm.y),
      z: this.filters[i][2].filter(lm.z),
      visibility: lm.visibility,
    }));
  }
}
