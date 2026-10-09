import { describe, expect, it } from "vitest";
import { Comment, Element, SubElement, tostring } from "../src/xml.ts";

describe("Element and tostring", () => {
  it("renders self-closing empty element with space", () => {
    const el = new Element("dot");
    expect(tostring(el)).toBe("<dot />");
  });

  it("renders attributes and text", () => {
    const el = new Element("measure", { number: "1" });
    const note = SubElement(el, "note");
    SubElement(note, "type").text = "quarter";
    expect(tostring(el)).toBe('<measure number="1"><note><type>quarter</type></note></measure>');
  });

  it("renders comment", () => {
    const c = new Comment(" imgpos: 10, 20 ");
    expect(tostring(c)).toBe("<!-- imgpos: 10, 20 -->");
  });
});
