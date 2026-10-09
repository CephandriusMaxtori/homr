import { describe, expect, it } from "vitest";
import { EncodedSymbol } from "../src/vocabulary.ts";
import { generateXml, xmlToString, XmlGeneratorArguments } from "../src/music_xml_generator.ts";

describe("MusicXML Generator", () => {
  it("generates valid MusicXML for simple note sequence", () => {
    const clef = new EncodedSymbol("clef_G2", ".", ".", ".", ".", "upper");
    const key = new EncodedSymbol("keySignature_0", ".", ".", ".", ".", "upper");
    const time = new EncodedSymbol("timeSignature/4", ".", ".", ".", ".", "upper");
    const note = new EncodedSymbol("note_4", "C4", "_", "_", "_", "upper");
    const bar = new EncodedSymbol("barline", ".", ".", ".", ".", "upper");

    const symbols = [clef, key, time, note, bar];
    const xml = generateXml(new XmlGeneratorArguments(), [symbols], "Test Score");
    const xmlString = xmlToString(xml);

    expect(xmlString).toContain("<?xml version=");
    expect(xmlString).toContain("<score-partwise version=\"4.0\">");
    expect(xmlString).toContain("<work-title>Test Score</work-title>");
    expect(xmlString).toContain("<step>C</step>");
    expect(xmlString).toContain("<octave>4</octave>");
    expect(xmlString).toContain("<type>quarter</type>");
  });
});
