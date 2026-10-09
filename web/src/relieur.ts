/**
 * Multi-page MusicXML combiner based on relieur (Apache 2.0).
 */

import { Element, tostring } from "./xml.ts";

function parseXmlString(xmlString: string): Element {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xmlString, "application/xml");
  const parserError = doc.querySelector("parsererror");
  if (parserError) {
    throw new Error(`XML Parse Error: ${parserError.textContent}`);
  }
  return domToElement(doc.documentElement);
}

function domToElement(node: globalThis.Element): Element {
  const attrib: Record<string, string> = {};
  for (let i = 0; i < node.attributes.length; i++) {
    const attr = node.attributes[i];
    if (attr) attrib[attr.name] = attr.value;
  }
  const el = new Element(node.tagName, attrib);

  let childText = "";
  for (let i = 0; i < node.childNodes.length; i++) {
    const child = node.childNodes[i];
    if (child) {
      if (child.nodeType === Node.TEXT_NODE) {
        childText += child.nodeValue ?? "";
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        el.append(domToElement(child as globalThis.Element));
      }
    }
  }
  el.text = childText.trim() || null;
  return el;
}

export function concatMusicXmlDocs(docs: string[]): string {
  if (docs.length === 0) return "";
  const firstDoc = docs[0];
  if (docs.length === 1 && firstDoc !== undefined) return firstDoc;
  if (!firstDoc) return "";

  const mainRoot = parseXmlString(firstDoc);
  const mainParts = mainRoot.findall("part");

  for (let docIdx = 1; docIdx < docs.length; docIdx++) {
    const docStr = docs[docIdx];
    if (!docStr) continue;
    const nextRoot = parseXmlString(docStr);
    const nextParts = nextRoot.findall("part");

    for (let partIdx = 0; partIdx < mainParts.length; partIdx++) {
      const mainPart = mainParts[partIdx];
      const nextPart = nextParts[partIdx];
      if (!mainPart || !nextPart) continue;

      const mainMeasures = mainPart.findall("measure");
      const currentLen = mainMeasures.length;
      const nextMeasures = nextPart.findall("measure");

      for (const measure of nextMeasures) {
        const origNum = Number.parseInt(measure.get("number") ?? "1", 10);
        const newNum = (origNum + currentLen).toString();
        measure.set("number", newNum);
        mainPart.append(measure);
      }
    }
  }

  return `<?xml version="1.0" encoding="UTF-8"?>\n${tostring(mainRoot)}`;
}
