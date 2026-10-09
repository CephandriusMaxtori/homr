/**
 * BigInt-backed Fraction implementation matching Python's `fractions.Fraction`.
 */

function gcd(a: bigint, b: bigint): bigint {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x;
}

export class Fraction {
  readonly numerator: bigint;
  readonly denominator: bigint;

  constructor(num: bigint | number | string = 0n, den: bigint | number = 1n) {
    let n = typeof num === "string" ? BigInt(num) : BigInt(num);
    let d = BigInt(den);
    if (d === 0n) {
      throw new Error("ZeroDivisionError: denominator cannot be zero");
    }
    if (d < 0n) {
      n = -n;
      d = -d;
    }
    const g = gcd(n, d);
    this.numerator = n / g;
    this.denominator = d / g;
  }

  static from(value: Fraction | bigint | number | string): Fraction {
    if (value instanceof Fraction) return value;
    if (typeof value === "string") {
      if (value.includes("/")) {
        const parts = value.split("/");
        const n = parts[0] ?? "0";
        const d = parts[1] ?? "1";
        return new Fraction(BigInt(n), BigInt(d));
      }
      return new Fraction(BigInt(value));
    }
    return new Fraction(value);
  }

  add(other: Fraction | bigint | number | string): Fraction {
    const o = Fraction.from(other);
    return new Fraction(
      this.numerator * o.denominator + o.numerator * this.denominator,
      this.denominator * o.denominator,
    );
  }

  sub(other: Fraction | bigint | number | string): Fraction {
    const o = Fraction.from(other);
    return new Fraction(
      this.numerator * o.denominator - o.numerator * this.denominator,
      this.denominator * o.denominator,
    );
  }

  mul(other: Fraction | bigint | number | string): Fraction {
    const o = Fraction.from(other);
    return new Fraction(
      this.numerator * o.numerator,
      this.denominator * o.denominator,
    );
  }

  div(other: Fraction | bigint | number | string): Fraction {
    const o = Fraction.from(other);
    return new Fraction(
      this.numerator * o.denominator,
      this.denominator * o.numerator,
    );
  }

  compare(other: Fraction | bigint | number | string): number {
    const o = Fraction.from(other);
    const diff = this.sub(o);
    if (diff.numerator === 0n) return 0;
    return diff.numerator > 0n ? 1 : -1;
  }

  equals(other: Fraction | bigint | number | string): boolean {
    return this.compare(other) === 0;
  }

  lessThan(other: Fraction | bigint | number | string): boolean {
    return this.compare(other) < 0;
  }

  lessThanOrEqual(other: Fraction | bigint | number | string): boolean {
    return this.compare(other) <= 0;
  }

  greaterThan(other: Fraction | bigint | number | string): boolean {
    return this.compare(other) > 0;
  }

  greaterThanOrEqual(other: Fraction | bigint | number | string): boolean {
    return this.compare(other) >= 0;
  }

  abs(): Fraction {
    return this.numerator < 0n ? new Fraction(-this.numerator, this.denominator) : this;
  }

  toNumber(): number {
    return Number(this.numerator) / Number(this.denominator);
  }

  toString(): string {
    if (this.denominator === 1n) return this.numerator.toString();
    return `${this.numerator}/${this.denominator}`;
  }
}
