/**
 * Vocabulary definitions and EncodedSymbol representation for homr transformer outputs.
 */

import { Fraction } from "./fraction.ts";

export const nonote = ".";
export const empty = "_";

export const VALID_TIME_SIGNATURE_DENOMINATORS = [1, 2, 3, 4, 6, 8, 12, 16, 32, 48] as const;

export function buildDict(tokens: Iterable<string>): Record<string, number> {
  const result: Record<string, number> = {};
  let i = 0;
  for (const t of tokens) {
    if (t in result) throw new Error(`Duplicated entry for ${t}`);
    if (!t.trim() || t.trim() !== t) throw new Error(`Invalid whitespace in token: ${t}`);
    if (t.includes("&")) throw new Error(`& is reserved in token: ${t}`);
    result[t] = i++;
  }
  return result;
}

export function buildRhythm(): Record<string, number> {
  const rhythm: string[] = ["PAD", "BOS", "EOS", "chord"];
  rhythm.push("barline", "doublebarline", "bolddoublebarline");
  rhythm.push("repeatStart", "repeatEnd", "repeatEndStart");
  rhythm.push("voltaStart", "voltaStop", "voltaDiscontinue");

  for (let c = 3; c <= 5; c++) rhythm.push(`clef_F${c}`);
  for (let c = 1; c <= 5; c++) rhythm.push(`clef_C${c}`);
  for (let c = 1; c <= 2; c++) rhythm.push(`clef_G${c}`);
  rhythm.push("clef_TAB5");

  for (let c = -7; c <= 7; c++) rhythm.push(`keySignature_${c}`);
  for (const c of VALID_TIME_SIGNATURE_DENOMINATORS) rhythm.push(`timeSignature/${c}`);

  for (let c = 2; c <= 10; c++) rhythm.push(`rest_${c}m`);

  const kernBaseDurations = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64, 128];
  const dots = ["", ".", ".."];
  const grace = ["", "G"];

  for (const d of kernBaseDurations) {
    for (const g of grace) {
      for (const dot of dots) {
        rhythm.push(`note_${d}${g}${dot}`);
      }
    }
  }

  const irregularDurations = [7, 11, 13, 18, 20, 21, 22, 24, 26, 28, 30, 34, 36, 40, 48, 56, 96];
  for (const d of irregularDurations) rhythm.push(`note_${d}`);

  for (const d of kernBaseDurations) {
    for (const g of grace) {
      for (const dot of dots) {
        rhythm.push(`rest_${d}${g}${dot}`);
      }
    }
  }
  for (const d of irregularDurations) rhythm.push(`rest_${d}`);

  return buildDict(rhythm);
}

export function buildLift(): Record<string, number> {
  return buildDict([nonote, empty, "#", "##", "N", "b", "bb"]);
}

export function buildPosition(): Record<string, number> {
  return buildDict([nonote, "upper", "upper2", "lower", "lower2"]);
}

export function isLowerPosition(position: string): boolean {
  return position.startsWith("lower");
}

export function isUpperOrHasNoPosition(position: string): boolean {
  return !isLowerPosition(position);
}

export function buildArticulation(): Record<string, number> {
  const articulations = [
    nonote, empty,
    "accent", "accent_arpeggiate", "accent_arpeggiate_fermata", "accent_arpeggiate_staccato",
    "accent_arpeggiate_tenuto", "accent_breathMark", "accent_breathMark_fermata", "accent_fermata",
    "accent_fermata_tremolo", "accent_staccatissimo", "accent_staccato", "accent_staccato_tenuto",
    "accent_staccato_tremolo", "accent_staccato_trill", "accent_tenuto", "accent_tremolo",
    "accent_trill", "arpeggiate", "arpeggiate_breathMark", "arpeggiate_fermata",
    "arpeggiate_fermata_staccato", "arpeggiate_fermata_tenuto", "arpeggiate_staccatissimo",
    "arpeggiate_staccatissimo_staccato", "arpeggiate_staccato", "arpeggiate_staccato_tenuto",
    "arpeggiate_tenuto", "arpeggiate_tremolo", "arpeggiate_trill", "breathMark",
    "breathMark_fermata", "breathMark_fermata_tenuto", "breathMark_staccato",
    "breathMark_tenuto", "breathMark_tremolo", "breathMark_trill", "fermata",
    "fermata_staccato", "fermata_tenuto", "fermata_tremolo", "fermata_trill", "fermata_turn",
    "mordent", "staccatissimo", "staccatissimo_staccato", "staccatissimo_staccato_tenuto",
    "staccatissimo_staccato_tenuto_trill", "staccatissimo_tenuto", "staccato",
    "staccato_tenuto", "staccato_tremolo", "staccato_trill", "staccato_turn", "tenuto",
    "tenuto_tremolo", "tenuto_trill", "tremolo", "trill", "trill_turn", "turn"
  ];
  return buildDict(articulations);
}

export function buildSlur(): Record<string, number> {
  return buildDict([nonote, empty, "slurStart_slurStop", "slurStart", "slurStop"]);
}

export function buildPitch(): Record<string, number> {
  const pitch = [nonote, empty];
  const noteNames = ["C", "D", "E", "F", "G", "A", "B"];
  const list: string[] = [];
  for (let oct = 0; oct < 10; oct++) {
    for (const name of noteNames) {
      list.push(`${name}${oct}`);
    }
  }
  list.reverse();
  pitch.push(...list);
  return buildDict(pitch);
}

export function hasRhythmSymbolAPosition(rhythm: string): boolean {
  return rhythm.startsWith("note") || rhythm.startsWith("rest") || rhythm.startsWith("clef");
}

export class Vocabulary {
  readonly rhythm = buildRhythm();
  readonly lift = buildLift();
  readonly articulation = buildArticulation();
  readonly pitch = buildPitch();
  readonly slur = buildSlur();
  readonly position = buildPosition();
}

export class SymbolDuration {
  readonly baseDuration: Fraction;
  readonly dots: number;
  readonly actualNotes: number;
  readonly normalNotes: number;
  readonly fraction: Fraction;
  readonly kern: number;

  constructor(
    baseDuration: Fraction,
    dots: number,
    actualNotes: number,
    normalNotes: number,
    kern: number,
  ) {
    this.baseDuration = baseDuration;
    this.dots = dots;
    const actionNormal = new Fraction(actualNotes, normalNotes);
    this.actualNotes = Number(actionNormal.numerator);
    this.normalNotes = Number(actionNormal.denominator);
    this.kern = kern;
    this.fraction = this.toFraction();
  }

  private toFraction(): Fraction {
    let dur = this.baseDuration;
    let add = dur.div(2);
    for (let i = 0; i < this.dots; i++) {
      dur = dur.add(add);
      add = add.div(2);
    }
    if (this.actualNotes !== this.normalNotes) {
      dur = dur.mul(new Fraction(this.normalNotes, this.actualNotes));
    }
    return dur;
  }
}

export function priorPowerOfTwo(n: number): number {
  if (n < 1) return 1;
  return 1 << (31 - Math.clz32(n));
}

export function kernToSymbolDuration(kern: string): SymbolDuration {
  if (kern.endsWith("m")) {
    return new SymbolDuration(new Fraction(1), 0, 1, 1, 4);
  }
  let i = 0;
  while (i < kern.length && (kern[i] ?? "") >= "0" && (kern[i] ?? "") <= "9") {
    i++;
  }
  const baseStr = kern.slice(0, i);
  const rest = kern.slice(i);
  const base = baseStr ? Number.parseInt(baseStr, 10) : 4;
  let dots = 0;
  for (const ch of rest) if (ch === ".") dots++;

  if (kern.includes("G")) {
    return new SymbolDuration(new Fraction(0), dots, 1, 1, base);
  }
  if (base === 0) {
    return new SymbolDuration(new Fraction(1), dots, 1, 1, base);
  }
  if ((base & (base - 1)) === 0) {
    return new SymbolDuration(new Fraction(1, base), dots, 1, 1, base);
  } else {
    const normalNotes = priorPowerOfTwo(base);
    return new SymbolDuration(new Fraction(1, normalNotes), dots, base, normalNotes, normalNotes);
  }
}

export class EncodedSymbol {
  rhythm: string;
  pitch: string;
  lift: string;
  articulation: string;
  slur: string;
  position: string;
  coordinates: [number, number] | null;
  imageCoordinates: [number, number] | null = null;
  private durationCache: SymbolDuration | null = null;

  constructor(
    rhythm: string,
    pitch: string = nonote,
    lift: string = nonote,
    articulation: string = nonote,
    slur: string = nonote,
    position: string = nonote,
    coordinates: [number, number] | null = null,
  ) {
    this.rhythm = rhythm;
    this.pitch = pitch;
    this.lift = lift;
    this.articulation = articulation;
    this.slur = slur;
    this.position = position;
    this.coordinates = coordinates;
  }

  isControlSymbol(): boolean {
    return this.rhythm === "BOS" || this.rhythm === "EOS" || this.rhythm === "PAD";
  }

  isTuplet(): boolean {
    return this.removeTuplet().rhythm !== this.rhythm;
  }

  removeTuplet(): EncodedSymbol {
    const match = this.rhythm.match(/^(note|rest)_(\d+)(.*)$/);
    if (!match) return this;
    const durStr = match[2] ?? "4";
    let duration = Number.parseInt(durStr, 10);
    if (duration % 3 === 0) duration = (duration / 3) * 2;
    else if (duration % 5 === 0) duration = (duration / 5) * 4;
    else if (duration % 7 === 0) duration = (duration / 7) * 4;
    else return this;

    const res = new EncodedSymbol(
      `${match[1]}_${duration}${match[3] ?? ""}`,
      this.pitch,
      this.lift,
      this.articulation,
      this.slur,
      this.position,
      this.coordinates,
    );
    res.imageCoordinates = this.imageCoordinates;
    return res;
  }

  changeLift(lift: string): EncodedSymbol {
    const res = new EncodedSymbol(
      this.rhythm,
      this.pitch,
      lift,
      this.articulation,
      this.slur,
      this.position,
      this.coordinates,
    );
    res.imageCoordinates = this.imageCoordinates;
    return res;
  }

  toUpperPosition(): EncodedSymbol {
    if (isUpperOrHasNoPosition(this.position)) return this;
    const res = new EncodedSymbol(
      this.rhythm,
      this.pitch,
      this.lift,
      this.articulation,
      this.slur,
      this.position.replace("lower", "upper"),
      this.coordinates,
    );
    res.imageCoordinates = this.imageCoordinates;
    return res;
  }

  isValid(): boolean {
    const hasPos = hasRhythmSymbolAPosition(this.rhythm);
    const isNote = [this.lift, this.articulation, this.pitch, this.slur, this.position].map(
      (s) => s !== nonote,
    );
    return isNote.every((item) => item === hasPos);
  }

  getDuration(): SymbolDuration {
    if (this.durationCache) return this.durationCache;
    if (!this.rhythm.startsWith("note") && !this.rhythm.startsWith("rest")) {
      return new SymbolDuration(new Fraction(0), 0, 1, 1, 1);
    }
    const parts = this.rhythm.split("_");
    const kern = parts[1] ?? "4";
    this.durationCache = kernToSymbolDuration(kern);
    return this.durationCache;
  }

  toString(): string {
    return `${this.rhythm} ${this.pitch} ${this.lift} ${this.articulation} ${this.slur} ${this.position}`;
  }
}

export function groupIntoChords(symbols: EncodedSymbol[]): EncodedSymbol[][] {
  const chords: EncodedSymbol[][] = [];
  let isInChord = false;
  for (const s of symbols) {
    if (s.rhythm === "chord") {
      isInChord = true;
    } else if (isInChord && chords.length > 0) {
      const lastChord = chords[chords.length - 1];
      if (lastChord) lastChord.push(s);
      isInChord = false;
    } else {
      chords.push([s]);
    }
  }
  return chords;
}

export function sortTokenChords(
  symbols: EncodedSymbol[],
  keepChordSymbol = false,
): EncodedSymbol[][] {
  const chords: EncodedSymbol[][] = [];
  let isInChord = false;
  for (const s of symbols) {
    if (s.rhythm === "chord") {
      isInChord = true;
    } else if (isInChord && chords.length > 0) {
      const lastChord = chords[chords.length - 1];
      if (lastChord) {
        if (keepChordSymbol) {
          lastChord.push(new EncodedSymbol("chord"));
        }
        lastChord.push(s);
      }
      isInChord = false;
    } else {
      chords.push([s]);
    }
  }
  return chords.map((chord) =>
    chord.slice().sort((a, b) => (a.toString() > b.toString() ? -1 : 1)),
  );
}

export function removeDuplicatedSymbols(
  symbols: EncodedSymbol[],
  _cleanupTuplets = true,
): EncodedSymbol[] {
  const chords = groupIntoChords(symbols);
  const flattened: EncodedSymbol[] = [];
  for (const chord of chords) {
    let first = true;
    for (const symbol of chord) {
      if (!first) flattened.push(new EncodedSymbol("chord"));
      flattened.push(symbol);
      first = false;
    }
  }
  return flattened;
}
