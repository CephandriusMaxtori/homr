/**
 * Complete MusicXML generator port matching homr/music_xml_generator.py.
 */

import { duration_of_quarter } from "./constants.ts";
import { Fraction } from "./fraction.ts";
import {
  EncodedSymbol,
  empty,
  isLowerPosition,
  nonote,
  sortTokenChords,
} from "./vocabulary.ts";
import { Comment, Element, SubElement, tostring } from "./xml.ts";

export class XmlGeneratorArguments {
  large_page?: boolean;
  metronome?: number;
  tempo?: number;

  constructor(large_page?: boolean, metronome?: number, tempo?: number) {
    this.large_page = large_page;
    this.metronome = metronome;
    this.tempo = tempo;
  }
}

export class ConversionState {
  beats: number;
  division: number;
  nominator: Fraction;
  tremoloState: "start" | "stop" = "stop";
  voltaNumber = 1;
  lastVoltaMeasure = -10;

  constructor(division: number, nominator: Fraction) {
    this.beats = 4 * duration_of_quarter;
    this.division = division;
    this.nominator = nominator;
  }

  startVolta(measureNo: number): number {
    if (measureNo === this.lastVoltaMeasure + 1) {
      this.voltaNumber++;
    } else {
      this.voltaNumber = 1;
    }
    return this.voltaNumber;
  }

  stopVolta(measureNo: number): number {
    this.lastVoltaMeasure = measureNo;
    return this.voltaNumber;
  }

  toggleTremoloState(): "start" | "stop" {
    if (this.tremoloState === "start") {
      this.tremoloState = "stop";
    } else {
      this.tremoloState = "start";
    }
    return this.tremoloState;
  }
}

export class SymbolChord {
  symbols: EncodedSymbol[];
  tupletMark: string;

  constructor(symbols: EncodedSymbol[], tupletMark = "") {
    this.symbols = symbols;
    this.tupletMark = tupletMark;
  }

  toString(): string {
    return this.symbols.map((s) => s.toString()).join("&");
  }

  isBarline(): boolean {
    if (this.symbols.length === 0) return false;
    const first = this.symbols[0];
    const firstRhythm = first ? first.rhythm : "";
    return firstRhythm.includes("barline") || firstRhythm.includes("repeat");
  }

  getDuration(): Fraction {
    const notesRests = this.symbols
      .filter((s) => s.rhythm.startsWith("note") || s.rhythm.startsWith("rest"))
      .map((s) => s.getDuration().fraction);
    if (notesRests.length === 0) return new Fraction(0);
    let minDur = notesRests[0] ?? new Fraction(0);
    for (const d of notesRests) {
      if (d.lessThan(minDur)) minDur = d;
    }
    return minDur;
  }

  intoPositions(): SymbolChord[] {
    const buckets = new Map<string, EncodedSymbol[]>();
    for (const symbol of this.symbols) {
      const pos = symbol.position;
      if (!buckets.has(pos)) buckets.set(pos, []);
      buckets.get(pos)!.push(symbol);
    }
    const chords = Array.from(buckets.values()).map(
      (symbols) => new SymbolChord(symbols, this.tupletMark),
    );
    chords.sort((a, b) => {
      const aOnlyRest = a.symbols.every((s) => s.rhythm.startsWith("rest"));
      const bOnlyRest = b.symbols.every((s) => s.rhythm.startsWith("rest"));
      if (aOnlyRest === bOnlyRest) return 0;
      return aOnlyRest ? -1 : 1;
    });
    return chords;
  }
}

export function buildWork(titleText: string): Element {
  const work = new Element("work");
  SubElement(work, "work-title").text = titleText;
  return work;
}

export function buildIdentification(): Element {
  const ident = new Element("identification");
  const enc = SubElement(ident, "encoding");
  SubElement(enc, "software").text = "homr";
  return ident;
}

export function buildDefaults(args: XmlGeneratorArguments): Element {
  const defaults = new Element("defaults");
  if (args.large_page) {
    const pageLayout = SubElement(defaults, "page-layout");
    SubElement(pageLayout, "page-height").text = "300";
    SubElement(pageLayout, "page-width").text = "110";
  }
  return defaults;
}

export function getPartId(index: number): string {
  return `P${index + 1}`;
}

export function partMetadata(hasTwoStaves: boolean): [string, string, string, number] {
  if (hasTwoStaves) return ["Piano", "Piano", "keyboard.piano", 1];
  return ["Voice", "Voice", "voice", 54];
}

export function buildPartList(hasTwoStavesByPart: boolean[]): Element {
  const partList = new Element("part-list");
  for (let part = 0; part < hasTwoStavesByPart.length; part++) {
    const hasTwoStaves = hasTwoStavesByPart[part] ?? false;
    const partId = getPartId(part);
    const [partNameStr, instrumentNameStr, instrumentSoundStr, midiProgram] =
      partMetadata(hasTwoStaves);
    const scorePart = SubElement(partList, "score-part", { id: partId });
    SubElement(scorePart, "part-name").text = partNameStr;
    const scoreInstrument = SubElement(scorePart, "score-instrument", { id: `${partId}-I1` });
    SubElement(scoreInstrument, "instrument-name").text = instrumentNameStr;
    SubElement(scoreInstrument, "instrument-sound").text = instrumentSoundStr;
    const midiInstrument = SubElement(scorePart, "midi-instrument", { id: `${partId}-I1` });
    SubElement(midiInstrument, "midi-channel").text = (part + 1).toString();
    SubElement(midiInstrument, "midi-program").text = midiProgram.toString();
    SubElement(midiInstrument, "volume").text = "100";
    SubElement(midiInstrument, "pan").text = "0";
  }
  return partList;
}

export function buildOrGetAttributes(
  measure: Element,
  lastAttributes: Element | null,
  forceNew = false,
): Element {
  if (lastAttributes !== null && !forceNew) return lastAttributes;
  return SubElement(measure, "attributes");
}

export function buildOrGetBarline(measure: Element, location: string): Element {
  for (const child of measure.children) {
    if (child instanceof Element && child.tag === "barline" && child.get("location") === location) {
      return child;
    }
  }
  return SubElement(measure, "barline", { location });
}

export function buildKey(modelKey: EncodedSymbol, attributes: Element): void {
  const key = SubElement(attributes, "key");
  const parts = modelKey.rhythm.split("_");
  SubElement(key, "fifths").text = parts[1] ?? "0";
}

export function getStaff(symbol: EncodedSymbol): number {
  return isLowerPosition(symbol.position) ? 2 : 1;
}

export function getXmlVoice(staffNum: number, rhythmicLayer: number): number {
  return (staffNum - 1) * 4 + rhythmicLayer + 1;
}

interface TimedNoteEvent {
  staffNum: number;
  start: number;
  end: number;
  notes: Element[];
}

export function rebalanceMeasureVoices(measure: Element): void {
  const timedEvents: TimedNoteEvent[] = [];
  let currentTime = 0;
  let lastNoteStart = 0;

  for (const child of measure.children) {
    if (!(child instanceof Element)) continue;
    if (child.tag === "backup") {
      const dur = child.find("duration");
      if (dur?.text) currentTime -= Number.parseInt(dur.text, 10);
      continue;
    }
    if (child.tag === "forward") {
      const dur = child.find("duration");
      if (dur?.text) currentTime += Number.parseInt(dur.text, 10);
      continue;
    }
    if (child.tag !== "note") continue;

    const durEl = child.find("duration");
    const duration = durEl?.text ? Number.parseInt(durEl.text, 10) : 0;
    const staffEl = child.find("staff");
    const staffNum = staffEl?.text ? Number.parseInt(staffEl.text, 10) : 1;
    const isChordTone = child.find("chord") !== null;
    const start = isChordTone ? lastNoteStart : currentTime;
    const end = start + duration;

    const lastEvent = timedEvents[timedEvents.length - 1];
    if (
      isChordTone &&
      lastEvent &&
      lastEvent.staffNum === staffNum &&
      lastEvent.start === start &&
      lastEvent.end === end
    ) {
      lastEvent.notes.push(child);
    } else if (isChordTone) {
      timedEvents.push({ staffNum, start, end, notes: [child] });
    } else {
      lastNoteStart = start;
      currentTime += duration;
      timedEvents.push({ staffNum, start, end, notes: [child] });
    }
  }

  const byStaff = new Map<number, TimedNoteEvent[]>();
  for (const event of timedEvents) {
    if (!byStaff.has(event.staffNum)) byStaff.set(event.staffNum, []);
    byStaff.get(event.staffNum)!.push(event);
  }

  for (const [staffNum, events] of byStaff.entries()) {
    const sortedEvents = events.slice().sort((a, b) => a.start - b.start || a.end - b.end);
    let active: [number, number][] = [];
    for (const event of sortedEvents) {
      active = active.filter(([activeEnd]) => activeEnd > event.start);
      const usedVoices = new Set(active.map(([, voiceNo]) => voiceNo));
      let voiceNo = 1;
      while (usedVoices.has(voiceNo)) voiceNo++;
      active.push([event.end, voiceNo]);
      const xmlVoice = getXmlVoice(staffNum, voiceNo - 1).toString();
      for (const note of event.notes) {
        const voiceEl = note.find("voice");
        if (voiceEl) voiceEl.text = xmlVoice;
      }
    }
  }
}

export function convertTies(part: Element): void {
  const previous = new Map<string, Element[]>();
  for (const measure of part.findall("measure")) {
    for (const event of groupIntoEvents(measure)) {
      const first = event[0];
      if (!first) continue;
      const key = `${first.findtext("staff", "1")}:${first.findtext("voice", "1")}`;
      const before = previous.get(key);
      previous.set(key, event);
      if (before) tieEvent(before, event);
    }
  }
}

function groupIntoEvents(measure: Element): Element[][] {
  const events: Element[][] = [];
  for (const note of measure.findall("note")) {
    const lastEvent = events[events.length - 1];
    if (note.find("chord") !== null && lastEvent) {
      lastEvent.push(note);
    } else {
      events.push([note]);
    }
  }
  return events;
}

function tieEvent(before: Element[], after: Element[]): void {
  for (const startNote of before) {
    const begins = getSlur(startNote, "start");
    if (!begins) continue;
    const pitch = getNotePitch(startNote);
    if (!pitch) continue;
    for (const stopNote of after) {
      if (getNotePitch(stopNote) !== pitch) continue;
      const ends = getSlur(stopNote, "stop");
      if (!ends) continue;
      const [beginSlur, beginNotation] = begins;
      const [endSlur, endNotation] = ends;
      beginNotation.remove(beginSlur);
      endNotation.remove(endSlur);
      addTie(startNote, "start", beginNotation);
      addTie(stopNote, "stop", endNotation);
      break;
    }
  }
}

function getNotePitch(note: Element): string | null {
  const pitch = note.find("pitch");
  if (!pitch) return null;
  return `${pitch.findtext("step", "")}|${pitch.findtext("alter", "")}|${pitch.findtext("octave", "")}`;
}

function getSlur(note: Element, slurType: string): [Element, Element] | null {
  for (const notation of note.findall("notations")) {
    for (const slur of notation.findall("slur")) {
      if (slur.get("type") === slurType) return [slur, notation];
    }
  }
  return null;
}

function addTie(note: Element, tieType: string, notation: Element): void {
  const tie = new Element("tie", { type: tieType });
  const duration = note.find("duration");
  const position = duration ? note.children.indexOf(duration) + 1 : 0;
  note.insert(position, tie);
  notation.insert(0, new Element("tied", { type: tieType }));
}

export function buildClef(modelClef: EncodedSymbol, attributes: Element): void {
  const parts = modelClef.rhythm.split("_");
  const signAndLine = parts[1] ?? "G2";
  const clef = SubElement(attributes, "clef", { number: getStaff(modelClef).toString() });
  SubElement(clef, "sign").text = signAndLine[0] ?? "G";
  SubElement(clef, "line").text = signAndLine[1] ?? "2";
}

export function buildTimeSignature(
  modelTimeSignature: EncodedSymbol,
  attributes: Element,
  state: ConversionState,
): void {
  const time = SubElement(attributes, "time");
  const parts = modelTimeSignature.rhythm.split("/");
  const denominator = parts[1] ?? "4";
  const beats = Math.max(1, Math.floor(state.nominator.toNumber() * Number.parseInt(denominator, 10)));
  SubElement(time, "beats").text = beats.toString();
  SubElement(time, "beat-type").text = denominator;
  state.beats = beats;
}

export function buildBarlineStyle(barline: EncodedSymbol, xml: Element): void {
  const styleValue = barline.rhythm === "bolddoublebarline" ? "heavy-heavy" : "light-light";
  SubElement(xml, "bar-style").text = styleValue;
}

export function buildBarlineEnding(
  volta: EncodedSymbol,
  xml: Element,
  voltaNumber: number,
): void {
  let type_: string;
  if (volta.rhythm.startsWith("voltaStart")) type_ = "start";
  else if (volta.rhythm.startsWith("voltaStop")) type_ = "stop";
  else if (volta.rhythm.startsWith("voltaDiscontinue")) type_ = "discontinue";
  else throw new Error(`Unknown ending ${volta}`);
  SubElement(xml, "ending", { type: type_, number: voltaNumber.toString() });
}

export function buildRepeat(barline: EncodedSymbol, xml: Element): void {
  if (xml.find("repeat") !== null) return;
  const direction = barline.rhythm === "repeatStart" ? "forward" : "backward";
  SubElement(xml, "repeat", { direction });
}

const LIFT_TO_ALTER: Record<string, number> = {
  N: 0,
  "#": 1,
  "##": 2,
  b: -1,
  bb: -2,
};

const DURATION_NAMES: Record<number, string> = {
  0: "breve",
  1: "whole",
  2: "half",
  4: "quarter",
  8: "eighth",
  16: "16th",
  32: "32nd",
  64: "64th",
  128: "128th",
};

export function buildArticulations(
  note: Element,
  articulations: string,
  tupletMark: string,
  state: ConversionState,
): void {
  const notation = SubElement(note, "notations");
  const xmlArticulations: Element[] = [];
  const xmlOrnaments: Element[] = [];

  for (const articulation of articulations.split("_")) {
    if (articulation === "" || articulation === nonote) continue;
    if (articulation === "fermata") SubElement(notation, "fermata");
    else if (articulation === "arpeggiate") SubElement(notation, "arpeggiate");
    else if (articulation === "accent") xmlArticulations.push(new Element("accent"));
    else if (articulation === "mordent") xmlArticulations.push(new Element("mordent"));
    else if (articulation === "staccato") xmlArticulations.push(new Element("staccato"));
    else if (articulation === "staccatissimo") xmlArticulations.push(new Element("staccatissimo"));
    else if (articulation === "tenuto") xmlArticulations.push(new Element("tenuto"));
    else if (articulation === "tremolo") {
      const el = new Element("tremolo", { type: state.toggleTremoloState() });
      el.text = "3";
      xmlOrnaments.push(el);
    } else if (articulation === "trill") xmlOrnaments.push(new Element("trill-mark"));
    else if (articulation === "breathMark") xmlArticulations.push(new Element("breath-mark"));
    else if (articulation === "turn") xmlOrnaments.push(new Element("inverted-turn"));
    else if (articulation === "caesura") xmlArticulations.push(new Element("caesura"));
    else if (articulation === "doit") xmlArticulations.push(new Element("doit"));
    else if (articulation === "slurStart") SubElement(notation, "slur", { type: "start" });
    else if (articulation === "slurStop") SubElement(notation, "slur", { type: "stop" });
    else if (articulation === "tieStart") SubElement(notation, "tied", { type: "start" });
    else if (articulation === "tieStop") SubElement(notation, "tied", { type: "stop" });
  }

  if (tupletMark !== "") SubElement(notation, "tuplet", { type: tupletMark });

  if (xmlArticulations.length > 0) {
    const parent = SubElement(notation, "articulations");
    for (const child of xmlArticulations) parent.append(child);
  }

  if (xmlOrnaments.length > 0) {
    const parent = SubElement(notation, "ornaments");
    for (const child of xmlOrnaments) parent.append(child);
  }
}

export function buildSlurs(note: Element, slurs: string, slurNumber: number): void {
  let notation = note.find("notations");
  if (!notation) notation = SubElement(note, "notations");

  if (slurs === "_" || slurs === "" || slurs === nonote) return;
  if (slurs === "slurStart") {
    SubElement(notation, "slur", { type: "start", number: slurNumber.toString() });
  } else if (slurs === "slurStop") {
    SubElement(notation, "slur", { type: "stop", number: slurNumber.toString() });
  } else if (slurs === "slurStart_slurStop") {
    SubElement(notation, "slur", { type: "stop", number: slurNumber.toString() });
    SubElement(notation, "slur", { type: "start", number: slurNumber.toString() });
  }
}

export function buildImagePosition(xml: Element, symbol: EncodedSymbol): void {
  if (!symbol.imageCoordinates) return;
  const [x, y] = symbol.imageCoordinates;
  xml.append(new Comment(` imgpos: ${Math.round(x)}, ${Math.round(y)} `));
}

export function buildMultiMeasureRest(symbol: EncodedSymbol, attributes: Element): void {
  if (attributes.find("measure-style") !== null) return;
  const parts = symbol.rhythm.split("_");
  const durationStr = (parts[1] ?? "1m").replace("m", "");
  const duration = Number.parseInt(durationStr, 10);
  const style = SubElement(attributes, "measure-style");
  SubElement(style, "multiple-rest").text = duration.toString();
}

export function buildBackup(duration: Fraction, state: ConversionState): Element {
  const backup = new Element("backup");
  const val = Math.max(1, Math.floor(duration.toNumber() * state.division));
  SubElement(backup, "duration").text = val.toString();
  return backup;
}

export function buildNoteOrRest(
  modelNote: EncodedSymbol,
  rhythmicLayer: number,
  isChord: boolean,
  state: ConversionState,
  tupletMark: string,
): Element {
  const note = new Element("note");
  if (isChord) SubElement(note, "chord");
  const modelPitch = modelNote.pitch;
  const modelDuration = modelNote.getDuration();

  if (modelNote.rhythm.includes("G")) SubElement(note, "grace");

  if (modelPitch === empty) {
    if (modelDuration.fraction.numerator === 0n) SubElement(note, "rest", { measure: "yes" });
    else SubElement(note, "rest");
  } else if (modelPitch === nonote) {
    SubElement(note, "rest");
  } else {
    const pitch = SubElement(note, "pitch");
    const step = modelPitch[0] ?? "C";
    const octave = modelPitch[1] ?? "4";
    SubElement(pitch, "step").text = step;
    SubElement(pitch, "octave").text = octave;
    if (modelNote.lift !== nonote && modelNote.lift !== empty) {
      SubElement(pitch, "alter").text = (LIFT_TO_ALTER[modelNote.lift] ?? 0).toString();
    }
  }

  if (modelNote.rhythm.includes("G")) {
    const baseDuration = modelDuration.kern;
    SubElement(note, "type").text = DURATION_NAMES[baseDuration] ?? "quarter";
  } else if (modelDuration.fraction.numerator > 0n) {
    const baseDuration = modelDuration.kern === 0 ? 1 : modelDuration.kern;
    const durVal = Math.max(1, Math.floor(modelDuration.fraction.toNumber() * state.division));
    SubElement(note, "duration").text = durVal.toString();
    SubElement(note, "type").text = DURATION_NAMES[baseDuration] ?? "quarter";
  } else {
    SubElement(note, "duration").text = state.beats.toString();
    SubElement(note, "type").text = DURATION_NAMES[0] ?? "breve";
  }

  for (let i = 0; i < modelDuration.dots; i++) SubElement(note, "dot");

  if (modelDuration.actualNotes !== modelDuration.normalNotes) {
    const timeMod = SubElement(note, "time-modification");
    SubElement(timeMod, "actual-notes").text = modelDuration.actualNotes.toString();
    SubElement(timeMod, "normal-notes").text = modelDuration.normalNotes.toString();
  }

  const staffNum = getStaff(modelNote);
  const slurNumber = staffNum;
  SubElement(note, "voice").text = getXmlVoice(staffNum, rhythmicLayer).toString();
  SubElement(note, "staff").text = staffNum.toString();

  buildArticulations(note, modelNote.articulation, tupletMark, state);
  buildSlurs(note, modelNote.slur, slurNumber);
  buildImagePosition(note, modelNote);

  return note;
}

function groupNotes(notes: EncodedSymbol[]): Map<string, EncodedSymbol[]> {
  const groupsByDuration = new Map<string, EncodedSymbol[]>();
  let maxDur = new Fraction(0);
  for (const n of notes) {
    const frac = n.getDuration().fraction;
    if (frac.greaterThan(maxDur)) maxDur = frac;
  }

  for (const note of notes) {
    const duration = note.getDuration();
    const isGrace = note.rhythm.includes("G");
    let fraction: Fraction;
    if (isGrace) fraction = new Fraction(0);
    else if (duration.fraction.numerator === 0n) fraction = maxDur;
    else fraction = duration.fraction;

    const key = fraction.toString();
    if (!groupsByDuration.has(key)) groupsByDuration.set(key, []);
    groupsByDuration.get(key)!.push(note);
  }

  const sortedKeys = Array.from(groupsByDuration.keys()).sort((a, b) =>
    Fraction.from(a).compare(Fraction.from(b)),
  );
  const result = new Map<string, EncodedSymbol[]>();
  for (const k of sortedKeys) {
    const group = groupsByDuration.get(k);
    if (group) result.set(k, group);
  }
  return result;
}

export function buildNoteChord(
  noteChord: SymbolChord,
  state: ConversionState,
  chordDuration: Fraction,
): Element[] {
  const byDuration = groupNotes(noteChord.symbols);
  const result: Element[] = [];
  const durationKeys = Array.from(byDuration.keys());

  for (let i = 0; i < durationKeys.length; i++) {
    const key = durationKeys[i];
    if (!key) continue;
    const groupDuration = Fraction.from(key);
    const groupNotes = byDuration.get(key) ?? [];
    const notes = groupNotes.filter((n) => n.pitch !== empty && n.pitch !== nonote);
    const rests = groupNotes.filter((n) => n.pitch === empty || n.pitch === nonote);

    let isFirst = true;
    for (const note of notes) {
      result.push(buildNoteOrRest(note, i, !isFirst, state, noteChord.tupletMark));
      isFirst = false;
    }

    const firstRest = rests[0];
    if (firstRest) {
      if (notes.length > 0) {
        result.push(buildBackup(groupDuration, state));
      }
      result.push(buildNoteOrRest(firstRest, i, false, state, noteChord.tupletMark));
    }

    if (i !== durationKeys.length - 1 && groupDuration.greaterThan(0)) {
      result.push(buildBackup(groupDuration, state));
    }
  }

  let maxDur = new Fraction(0);
  for (const k of durationKeys) {
    const f = Fraction.from(k);
    if (f.greaterThan(maxDur)) maxDur = f;
  }
  if (chordDuration.lessThan(maxDur)) {
    result.push(buildBackup(maxDur.sub(chordDuration), state));
  }

  return result;
}

export function buildAddTimeDirection(args: XmlGeneratorArguments): Element | null {
  if (!args.metronome) return null;
  const direction = new Element("direction");
  const directionType = SubElement(direction, "direction-type");
  const metronome = SubElement(directionType, "metronome");
  SubElement(metronome, "beat-unit").text = "quarter";
  SubElement(metronome, "per-minute").text = args.metronome.toString();
  const tempo = args.tempo ?? args.metronome;
  SubElement(direction, "sound", { tempo: tempo.toString() });
  return direction;
}

function gcdBigInt(a: bigint, b: bigint): bigint {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x;
}

function lcmBigInt(a: bigint, b: bigint): bigint {
  if (a === 0n || b === 0n) return 0n;
  return (a * b) / gcdBigInt(a, b);
}

export function findCommonDivision(durations: Fraction[]): number {
  const denominators = durations
    .filter((d) => d.greaterThan(0))
    .map((d) => d.denominator);
  if (denominators.length === 0) return 1;
  let common = denominators[0] ?? 1n;
  for (let i = 1; i < denominators.length; i++) {
    const den = denominators[i];
    if (den !== undefined) common = lcmBigInt(common, den);
  }
  return Number(common);
}

export function findDivisionAndTimeSignatureNominator(
  voice: SymbolChord[],
): [number, Fraction] {
  const durations = [new Fraction(1, 4)];
  let durationInMeasure = new Fraction(0);
  const measureDuration: Fraction[] = [];

  for (const chord of voice) {
    if (chord.isBarline() && durationInMeasure.greaterThan(0)) {
      measureDuration.push(durationInMeasure);
      durationInMeasure = new Fraction(0);
    } else {
      for (const symbol of chord.symbols) {
        if (symbol.rhythm.startsWith("note") || symbol.rhythm.startsWith("rest")) {
          const frac = symbol.getDuration().fraction;
          if (frac.greaterThan(0)) durations.push(frac);
        }
      }
      const duration = chord.getDuration();
      if (duration.greaterThan(0)) {
        durationInMeasure = durationInMeasure.add(duration);
      }
    }
  }

  if (durationInMeasure.greaterThan(0)) measureDuration.push(durationInMeasure);

  if (measureDuration.length === 0) {
    return [findCommonDivision(durations), new Fraction(1)];
  }

  const sorted = measureDuration
    .slice()
    .sort((a, b) => a.compare(b));
  const nominator = sorted[Math.floor(sorted.length / 2)] ?? new Fraction(1);

  return [findCommonDivision(durations), nominator];
}

const BEFORE_NOTES_ORDER = ["clef", "keySignature", "timeSignature"];

export function splitMixedChord(symbols: EncodedSymbol[]): EncodedSymbol[][] {
  const notes = symbols.filter((s) => s.rhythm.startsWith("note") || s.rhythm.startsWith("rest"));
  if (notes.length === 0 || notes.length === symbols.length) return [symbols];

  const others = symbols.filter((s) => !s.rhythm.startsWith("note") && !s.rhythm.startsWith("rest"));
  const after = others.filter((s) => s.rhythm.includes("barline") || s.rhythm.includes("repeat"));
  const beforeMap = new Map<string, EncodedSymbol[]>();

  for (const s of others) {
    if (!after.includes(s)) {
      const parts = s.rhythm.split("_");
      const subParts = (parts[0] ?? "").split("/");
      const kind = subParts[0] ?? "";
      if (!beforeMap.has(kind)) beforeMap.set(kind, []);
      beforeMap.get(kind)!.push(s);
    }
  }

  const rank = new Map(BEFORE_NOTES_ORDER.map((k, i) => [k, i]));
  const ordered = Array.from(beforeMap.keys()).sort(
    (a, b) => (rank.get(a) ?? rank.size) - (rank.get(b) ?? rank.size),
  );

  const res: EncodedSymbol[][] = [];
  for (const kind of ordered) {
    const list = beforeMap.get(kind);
    if (list) res.push(list);
  }
  res.push(notes);
  for (const s of after) res.push([s]);
  return res;
}

export function splitMixedChords(groups: SymbolChord[]): SymbolChord[] {
  const result: SymbolChord[] = [];
  let splitBarline = false;

  for (const group of groups) {
    const parts = splitMixedChord(group.symbols);
    const lastResult = result[result.length - 1];
    const firstGroupSym = group.symbols[0];

    if (splitBarline && group.isBarline() && parts.length === 1 && lastResult && firstGroupSym) {
      const lastSym = lastResult.symbols[0];
      if (
        firstGroupSym.rhythm === "barline" &&
        lastSym &&
        lastSym.rhythm !== "barline"
      ) {
        splitBarline = false;
        continue;
      }
      result.pop();
    }
    for (const part of parts) {
      const firstPartSym = part[0];
      const isNotes = firstPartSym
        ? firstPartSym.rhythm.startsWith("note") || firstPartSym.rhythm.startsWith("rest")
        : false;
      result.push(new SymbolChord(part, isNotes ? group.tupletMark : ""));
    }
    const endRes = result[result.length - 1];
    splitBarline = parts.length > 1 && endRes ? endRes.isBarline() : false;
  }
  return result;
}

export function groupIntoChordsList(voice: EncodedSymbol[]): SymbolChord[] {
  return sortTokenChords(voice).map((s) => new SymbolChord(s));
}

export class TupletParser {
  static parse(groups: SymbolChord[]): SymbolChord[] {
    for (const measureGroups of TupletParser.splitIntoMeasures(groups)) {
      const savedMarks = measureGroups.map((g) => g.tupletMark);
      if (TupletParser.addTuplets(measureGroups)) continue;
      for (let i = 0; i < measureGroups.length; i++) {
        const mg = measureGroups[i];
        const saved = savedMarks[i];
        if (mg && saved !== undefined) mg.tupletMark = saved;
      }
    }
    return groups;
  }

  static getTupletDuration(group: SymbolChord) {
    for (const symbol of group.symbols) {
      if (symbol.rhythm.startsWith("note") || symbol.rhythm.startsWith("rest")) {
        const duration = symbol.getDuration();
        if (duration.normalNotes !== duration.actualNotes) return duration;
      }
    }
    return null;
  }

  static splitIntoMeasures(groups: SymbolChord[]): SymbolChord[][] {
    const measures: SymbolChord[][] = [];
    let currentMeasure: SymbolChord[] = [];
    for (const group of groups) {
      currentMeasure.push(group);
      if (group.isBarline()) {
        measures.push(currentMeasure);
        currentMeasure = [];
      }
    }
    if (currentMeasure.length > 0) measures.push(currentMeasure);
    return measures;
  }

  static addTuplets(groups: SymbolChord[]): boolean {
    let cursor = 0;
    while (cursor < groups.length) {
      const groupAtCursor = groups[cursor];
      if (!groupAtCursor) {
        cursor++;
        continue;
      }
      const duration = TupletParser.getTupletDuration(groupAtCursor);
      if (!duration) {
        cursor++;
        continue;
      }

      const start = cursor;
      const tupletFormat = `${duration.actualNotes}:${duration.normalNotes}`;
      const tupletSize = duration.actualNotes;

      while (cursor - start < tupletSize) {
        if (cursor >= groups.length) return false;
        const curGroup = groups[cursor];
        if (!curGroup) return false;
        const currentDuration = TupletParser.getTupletDuration(curGroup);
        if (!currentDuration) return false;
        const currentFormat = `${currentDuration.actualNotes}:${currentDuration.normalNotes}`;
        if (currentFormat !== tupletFormat) return false;
        cursor++;
      }

      const startGroup = groups[start];
      const stopGroup = groups[cursor - 1];
      if (startGroup) startGroup.tupletMark = "start";
      if (stopGroup) stopGroup.tupletMark = "stop";
    }
    return true;
  }
}

export function addTupletStartStop(groups: SymbolChord[]): SymbolChord[] {
  return TupletParser.parse(groups);
}

export function voiceHasTwoStaves(voice: EncodedSymbol[]): boolean {
  return voice.some((s) => isLowerPosition(s.position));
}

export function buildMeasures(
  args: XmlGeneratorArguments,
  voice: EncodedSymbol[],
  isFirstPart: boolean,
  hasTwoStaves = false,
): Element[] {
  let clock = new Fraction(0);
  let sounding: Fraction[] = [];

  const measures: Element[] = [];
  let measureNumber = 1;

  const groups = splitMixedChords(addTupletStartStop(groupIntoChordsList(voice)));
  const [division, nominator] = findDivisionAndTimeSignatureNominator(groups);
  const state = new ConversionState(division, nominator);

  let currentMeasure = new Element("measure", { number: measureNumber.toString() });

  const closeCurrentMeasure = (): void => {
    rebalanceMeasureVoices(currentMeasure);
    measures.push(currentMeasure);
    clock = new Fraction(0);
    sounding = [];
  };

  const firstAttributes = buildOrGetAttributes(currentMeasure, null);
  SubElement(firstAttributes, "divisions").text = Math.floor(division / 4).toString();
  if (hasTwoStaves) {
    SubElement(firstAttributes, "staves").text = "2";
    SubElement(firstAttributes, "part-symbol").text = "brace";
  }
  if (isFirstPart) {
    const direction = buildAddTimeDirection(args);
    if (direction) currentMeasure.append(direction);
  }

  let attributes: Element | null = firstAttributes;

  for (let groupNo = 0; groupNo < groups.length; groupNo++) {
    const group = groups[groupNo];
    if (!group) continue;
    const symbol = group.symbols[0];
    if (!symbol) continue;
    const rhythm = symbol.rhythm;
    const lastAttributes = attributes;
    attributes = null;

    if (rhythm.startsWith("note") || rhythm.startsWith("rest")) {
      if (group.symbols.length === 1 && rhythm.endsWith("m")) {
        attributes = buildOrGetAttributes(currentMeasure, lastAttributes);
        buildMultiMeasureRest(symbol, attributes);
      } else {
        const staffPositions = group.intoPositions();
        const advance = advanceToNextGroup(group, clock, sounding);
        clock = clock.add(advance);
        sounding = sounding.filter((end) => end.greaterThan(clock));

        for (let posNo = 0; posNo < staffPositions.length; posNo++) {
          const staffPos = staffPositions[posNo];
          if (!staffPos) continue;
          const chordDuration = posNo === staffPositions.length - 1 ? advance : new Fraction(0);
          for (const noteXml of buildNoteChord(staffPos, state, chordDuration)) {
            currentMeasure.append(noteXml);
          }
        }
      }
      continue;
    }

    if (rhythm === "newline") {
      const isLastMeasure = groupNo === groups.length - 1;
      if (!isLastMeasure) {
        SubElement(currentMeasure, "print", { "new-system": "yes" });
      }
    } else if (rhythm.startsWith("clef")) {
      attributes = buildOrGetAttributes(currentMeasure, lastAttributes, true);
      for (const s of group.symbols) {
        if (s.rhythm.startsWith("clef")) buildClef(s, attributes);
      }
    } else if (rhythm.startsWith("keySignature")) {
      attributes = buildOrGetAttributes(currentMeasure, lastAttributes);
      buildKey(symbol, attributes);
    } else if (rhythm.startsWith("timeSignature")) {
      attributes = buildOrGetAttributes(currentMeasure, lastAttributes);
      buildTimeSignature(symbol, attributes, state);
    } else if (rhythm.includes("barline")) {
      if (rhythm !== "barline") {
        const barline = buildOrGetBarline(currentMeasure, "right");
        buildBarlineStyle(symbol, barline);
      }
      closeCurrentMeasure();
      measureNumber++;
      currentMeasure = new Element("measure", { number: measureNumber.toString() });
    } else if (rhythm === "repeatStart") {
      closeCurrentMeasure();
      measureNumber++;
      currentMeasure = new Element("measure", { number: measureNumber.toString() });
      const barline = buildOrGetBarline(currentMeasure, "right");
      buildRepeat(symbol, barline);
    } else if (rhythm === "repeatEnd") {
      const barline = buildOrGetBarline(currentMeasure, "right");
      buildRepeat(symbol, barline);
      closeCurrentMeasure();
      measureNumber++;
      currentMeasure = new Element("measure", { number: measureNumber.toString() });
    } else if (rhythm === "repeatEndStart") {
      const barline1 = buildOrGetBarline(currentMeasure, "right");
      buildRepeat(new EncodedSymbol("repeatEnd"), barline1);
      closeCurrentMeasure();
      measureNumber++;
      currentMeasure = new Element("measure", { number: measureNumber.toString() });
      const barline2 = buildOrGetBarline(currentMeasure, "right");
      buildRepeat(new EncodedSymbol("repeatStart"), barline2);
    } else if (rhythm.startsWith("voltaStart")) {
      const voltaNumber = state.startVolta(measureNumber);
      const barline = buildOrGetBarline(currentMeasure, "left");
      buildBarlineEnding(symbol, barline, voltaNumber);
    } else if (rhythm.startsWith("voltaStop") || rhythm.startsWith("voltaDiscontinue")) {
      const voltaNumber = state.stopVolta(measureNumber);
      const barline = buildOrGetBarline(currentMeasure, "right");
      buildBarlineEnding(symbol, barline, voltaNumber);
    }
  }

  if (currentMeasure.children.length > 0) {
    closeCurrentMeasure();
  }

  if (firstAttributes.find("time") === null) {
    const timeEl = SubElement(firstAttributes, "time");
    const beats = Math.max(1, Math.floor(state.nominator.toNumber() * 4));
    SubElement(timeEl, "beats").text = beats.toString();
    SubElement(timeEl, "beat-type").text = "4";
  }

  return measures;
}

function advanceToNextGroup(
  group: SymbolChord,
  clock: Fraction,
  sounding: Fraction[],
): Fraction {
  const durations = group.symbols
    .filter((s) => s.rhythm.startsWith("note") || s.rhythm.startsWith("rest"))
    .map((s) => s.getDuration().fraction);
  const timed = durations.filter((d) => d.greaterThan(0));
  if (timed.length === 0) return new Fraction(0);

  for (const d of timed) sounding.push(clock.add(d));

  let minEnd: Fraction | null = null;
  for (const end of sounding) {
    if (end.greaterThan(clock)) {
      if (minEnd === null || end.lessThan(minEnd)) minEnd = end;
    }
  }
  return minEnd ? minEnd.sub(clock) : new Fraction(0);
}

export function buildPart(
  args: XmlGeneratorArguments,
  voice: EncodedSymbol[],
  index: number,
  hasTwoStaves: boolean,
): Element {
  const part = new Element("part", { id: getPartId(index) });
  const isFirstPart = index === 0;
  for (const measure of buildMeasures(args, voice, isFirstPart, hasTwoStaves)) {
    part.append(measure);
  }
  convertTies(part);
  return part;
}

export function generateXml(
  args: XmlGeneratorArguments,
  staffs: EncodedSymbol[][],
  title: string,
): Element {
  const root = new Element("score-partwise", { version: "4.0" });
  root.append(buildWork(title));
  root.append(buildIdentification());
  root.append(buildDefaults(args));

  const hasTwoStavesByPart = staffs.map((staff) => voiceHasTwoStaves(staff));
  root.append(buildPartList(hasTwoStavesByPart));

  for (let index = 0; index < staffs.length; index++) {
    const staff = staffs[index];
    const hasTwoStaves = hasTwoStavesByPart[index] ?? false;
    if (staff) {
      root.append(buildPart(args, staff, index, hasTwoStaves));
    }
  }
  return root;
}

export function xmlToString(element: Element): string {
  return `<?xml version="1.0" encoding="UTF-8"?>\n${tostring(element)}`;
}
