import { describe, expect, it } from "vitest";
import { Fraction } from "../src/fraction.ts";

describe("Fraction", () => {
  it("simplifies fractions", () => {
    const f = new Fraction(4, 8);
    expect(f.numerator).toBe(1n);
    expect(f.denominator).toBe(2n);
    expect(f.toString()).toBe("1/2");
  });

  it("handles negative denominators", () => {
    const f = new Fraction(3, -6);
    expect(f.numerator).toBe(-1n);
    expect(f.denominator).toBe(2n);
  });

  it("performs addition and subtraction", () => {
    const a = new Fraction(1, 4);
    const b = new Fraction(1, 2);
    expect(a.add(b).toString()).toBe("3/4");
    expect(b.sub(a).toString()).toBe("1/4");
  });

  it("performs multiplication and division", () => {
    const a = new Fraction(2, 3);
    const b = new Fraction(3, 4);
    expect(a.mul(b).toString()).toBe("1/2");
    expect(a.div(b).toString()).toBe("8/9");
  });

  it("compares correctly", () => {
    const a = new Fraction(1, 3);
    const b = new Fraction(1, 2);
    expect(a.lessThan(b)).toBe(true);
    expect(b.greaterThan(a)).toBe(true);
    expect(a.equals(new Fraction(2, 6))).toBe(true);
  });
});
