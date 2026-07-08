#!/usr/bin/env python3
"""Render terminal-style PNG from a text file."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render_terminal_png(textfile: Path, outfile: Path, width: int = 960) -> None:
    text = textfile.read_text(encoding="utf-8").rstrip("\n")
    lines = text.split("\n") if text else [""]

    font_path = "/System/Library/Fonts/Menlo.ttc"
    font_size = 14
    padding_x = 24
    padding_y = 28
    line_height = 20

    font = ImageFont.truetype(font_path, font_size)
    height = padding_y * 2 + line_height * len(lines)
    bg = (30, 30, 30)
    fg = (212, 212, 212)
    prompt = (78, 201, 176)
    dim = (140, 140, 140)
    green = (106, 153, 85)
    yellow = (220, 180, 90)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    y = padding_y
    for line in lines:
        if line.startswith("$ "):
            draw.text((padding_x, y), "$ ", font=font, fill=prompt)
            draw.text((padding_x + 18, y), line[2:], font=font, fill=fg)
        elif line.startswith("BEFORE:"):
            draw.text((padding_x, y), line[:7], font=font, fill=yellow)
            draw.text((padding_x + 70, y), line[7:], font=font, fill=fg)
        elif line.startswith("AFTER:"):
            draw.text((padding_x, y), line[:6], font=font, fill=green)
            draw.text((padding_x + 60, y), line[6:], font=font, fill=fg)
        elif line.startswith("#") or line.startswith("---"):
            draw.text((padding_x, y), line, font=font, fill=dim)
        else:
            draw.text((padding_x, y), line, font=font, fill=fg)
        y += line_height

    outfile.parent.mkdir(parents=True, exist_ok=True)
    image.save(outfile)


if __name__ == "__main__":
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 960
    render_terminal_png(Path(sys.argv[1]), Path(sys.argv[2]), width=w)
