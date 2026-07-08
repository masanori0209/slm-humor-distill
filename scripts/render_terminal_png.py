#!/usr/bin/env python3
"""Render terminal-style PNG from a text file (Japanese-safe)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Menlo は日本語グリフがない → 豆腐になる
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_SIZE = 15
LINE_HEIGHT = 24
PADDING_X = 24
PADDING_Y = 28


def load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=font)


def render_terminal_png(textfile: Path, outfile: Path, width: int = 1024) -> None:
    text = textfile.read_text(encoding="utf-8").rstrip("\n")
    lines = text.split("\n") if text else [""]

    font = load_font()
    height = PADDING_Y * 2 + LINE_HEIGHT * len(lines)
    bg = (30, 30, 30)
    fg = (212, 212, 212)
    prompt = (78, 201, 176)
    dim = (140, 140, 140)
    green = (106, 153, 85)
    yellow = (220, 180, 90)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    y = PADDING_Y
    for line in lines:
        x = PADDING_X
        if line.startswith("$ "):
            draw.text((x, y), "$ ", font=font, fill=prompt)
            x += text_width(draw, "$ ", font)
            draw.text((x, y), line[2:], font=font, fill=fg)
        elif line.startswith("BEFORE:"):
            prefix = "BEFORE:"
            draw.text((x, y), prefix, font=font, fill=yellow)
            x += text_width(draw, prefix, font) + 8
            draw.text((x, y), line[len(prefix) :].strip(), font=font, fill=fg)
        elif line.startswith("AFTER:"):
            prefix = "AFTER:"
            draw.text((x, y), prefix, font=font, fill=green)
            x += text_width(draw, prefix, font) + 8
            draw.text((x, y), line[len(prefix) :].strip(), font=font, fill=fg)
        elif line.startswith("#") or line.startswith("---"):
            draw.text((x, y), line, font=font, fill=dim)
        else:
            draw.text((x, y), line, font=font, fill=fg)
        y += LINE_HEIGHT

    outfile.parent.mkdir(parents=True, exist_ok=True)
    image.save(outfile)


if __name__ == "__main__":
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    render_terminal_png(Path(sys.argv[1]), Path(sys.argv[2]), width=w)
