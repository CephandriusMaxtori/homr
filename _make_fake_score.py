"""Draws a tiny synthetic two-staff 'score' so the pipeline can be exercised
without downloading a real sample from the internet."""

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 900, 400
IMAGE = Image.new("L", (WIDTH, HEIGHT), 255)
draw = ImageDraw.Draw(IMAGE)

# Two staves of five lines each.
for top in (80, 240):
    for line in range(5):
        y = top + line * 10
        draw.line([(60, y), (WIDTH - 60, y)], fill=0, width=2)

# A treble-ish clef blob and a few noteheads with stems.
draw.ellipse([70, 70, 100, 130], outline=0, width=3)
for index in range(8):
    x = 160 + index * 90
    y = 130 + (index % 3) * 10
    draw.ellipse([x, y, x + 18, y + 14], fill=0)
    draw.line([(x + 16, y + 7), (x + 16, y - 45)], fill=0, width=3)

    lower_x = 160 + index * 90
    lower_y = 290 - (index % 2) * 10
    draw.ellipse([lower_x, lower_y, lower_x + 18, lower_y + 14], fill=0)
    draw.line([(lower_x + 16, lower_y + 7), (lower_x + 16, lower_y + 45)], fill=0, width=3)

# A bar line between the staves.
for x in range(2):
    bar_x = 300 + x * 300
    draw.line([(bar_x, 80), (bar_x, 280)], fill=0, width=2)

path = r"C:\Users\melis\AppData\Local\Temp\homr-smoke\synthetic.png"
Image.fromarray(np.zeros((1, 1), dtype=np.uint8))  # touch numpy as homr does
IMAGE.save(path)
print("wrote", path)