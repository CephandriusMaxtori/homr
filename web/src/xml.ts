/**
 * Lightweight ElementTree-like DOM builder and serializer matching Python's xml.etree.ElementTree output.
 */

export class Element {
  tag: string;
  attrib: Record<string, string>;
  text: string | null = null;
  tail: string | null = null;
  children: (Element | Comment)[] = [];

  constructor(tag: string, attrib: Record<string, string> = {}) {
    this.tag = tag;
    this.attrib = { ...attrib };
  }

  append(child: Element | Comment): void {
    this.children.push(child);
  }

  insert(index: number, child: Element | Comment): void {
    this.children.splice(index, 0, child);
  }

  remove(child: Element | Comment): void {
    const idx = this.children.indexOf(child);
    if (idx !== -1) this.children.splice(idx, 1);
  }

  find(tag: string): Element | null {
    for (const child of this.children) {
      if (child instanceof Element && child.tag === tag) return child;
    }
    return null;
  }

  findall(tag: string): Element[] {
    const res: Element[] = [];
    for (const child of this.children) {
      if (child instanceof Element && child.tag === tag) res.push(child);
    }
    return res;
  }

  findtext(tag: string, defaultVal: string = ""): string {
    const el = this.find(tag);
    return el?.text ?? defaultVal;
  }

  get(key: string): string | undefined {
    return this.attrib[key];
  }

  set(key: string, value: string): void {
    this.attrib[key] = value;
  }

  [Symbol.iterator](): Iterator<Element | Comment> {
    return this.children[Symbol.iterator]();
  }
}

export class Comment {
  text: string;

  constructor(text: string) {
    this.text = text;
  }
}

export function SubElement(
  parent: Element,
  tag: string,
  attrib: Record<string, string> = {},
): Element {
  const el = new Element(tag, attrib);
  parent.append(el);
  return el;
}

function escapeText(str: string): string {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeAttr(str: string): string {
  return escapeText(str)
    .replace(/"/g, "&quot;")
    .replace(/\r/g, "&#13;")
    .replace(/\n/g, "&#10;")
    .replace(/\t/g, "&#9;");
}

export function tostring(element: Element | Comment): string {
  if (element instanceof Comment) {
    return `<!--${element.text}-->`;
  }

  let attrStr = "";
  for (const [k, v] of Object.entries(element.attrib)) {
    attrStr += ` ${k}="${escapeAttr(v)}"`;
  }

  if (element.children.length === 0 && (element.text === null || element.text === "")) {
    return `<${element.tag}${attrStr} />`;
  }

  let inner = element.text ? escapeText(element.text) : "";
  for (const child of element.children) {
    inner += tostring(child);
  }

  return `<${element.tag}${attrStr}>${inner}</${element.tag}>`;
}
