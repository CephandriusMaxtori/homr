"""Writes a small multi-page PDF (one staff per page) for smoke tests."""

import zlib
from pathlib import Path

OUT = Path(r"C:\Users\melis\AppData\Local\Temp\homr-smoke\score.pdf")
WIDTH, HEIGHT = 200, 300


def content_stream(index: int) -> bytes:
    """A PDF content stream drawing five staff lines for page *index*."""
    parts = ["1 J", "1 w", "0 0 0 RG"]
    for line in range(5):
        y = 60 + index * 10 + line * 10
        parts.append(f"20 {y} m 180 {y} l S")
    parts.append("BT /F1 12 Tf 20 250 Td (Synthetic page %d) Tj ET" % (index + 1))
    return "\n".join(parts).encode("latin-1")


def build() -> None:
    objects: list[bytes] = []

    page_count = 3
    font_obj = 3 + 2 * page_count
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(page_count))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    for index in range(page_count):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {4 + 2 * index} 0 R >>".encode()
        )
        stream = content_stream(index)
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n"
    ).encode()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(bytes(out))
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    build()