#!/usr/bin/env python3
"""Render a text log file as a macOS-terminal-style PNG evidence image.

Usage: python render_terminal.py <input_text_file> <output_png> "<title command line>" [max_lines]
The text is rendered verbatim; if the log is huge it is truncated from the TOP
(keeping the final summary lines). Never fabricates content.
"""
import sys
import textwrap
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FALLBACK_FONT = "/System/Library/Fonts/Helvetica.ttc"
BG = (28, 28, 30)          # #1c1c1e
FG = (211, 211, 211)       # light gray
GREEN = (78, 201, 120)     # command line green
TITLEBAR = (58, 58, 62)
TITLEBAR_TEXT = (200, 200, 200)
WIDTH = 1400
FONT_SIZE = 15
WRAP_COLS = 110
LINE_H = 20
MARGIN_L = 16
MARGIN_T = 52  # below titlebar + command line area handled separately


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def main():
    if len(sys.argv) < 4:
        print("usage: render_terminal.py <input.txt> <out.png> '<title>' [max_lines]")
        sys.exit(2)
    in_path, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3]
    max_lines = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    with open(in_path, "r", errors="replace") as f:
        raw_lines = f.read().rstrip("\n").split("\n")

    # Wrap long lines, then truncate from the TOP keeping the tail (summaries).
    lines = []
    for ln in raw_lines:
        if len(ln) > WRAP_COLS:
            lines.extend(textwrap.wrap(ln, WRAP_COLS, subsequent_indent="    ",
                                       break_long_words=True, break_on_hyphens=False) or [ln])
        else:
            lines.append(ln)
    if len(lines) > max_lines:
        lines = ["... [truncated: showing last %d of %d lines] ..." % (max_lines, len(lines))] + lines[-max_lines:]

    font = load_font(FONT_PATH, FONT_SIZE) or load_font(FALLBACK_FONT, FONT_SIZE)
    tfont = load_font(FONT_PATH, 13) or load_font(FALLBACK_FONT, 13)

    cmd_text = "$ " + title
    cmd_wrapped = []
    for ln in textwrap.wrap(cmd_text, WRAP_COLS, break_long_words=True,
                             break_on_hyphens=False) or [cmd_text]:
        cmd_wrapped.append(ln)
    body = [""] + lines  # blank line after command
    min_height = 360
    height = max(MARGIN_T + LINE_H * len(cmd_wrapped) + LINE_H + LINE_H * len(body) + MARGIN_L, min_height)

    img = Image.new("RGB", (WIDTH, height), BG)
    d = ImageDraw.Draw(img)

    # macOS terminal style title bar
    d.rectangle([0, 0, WIDTH, 30], fill=TITLEBAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + i * 22, 9, 24 + i * 22, 21], fill=c)
    tw = d.textlength(title, font=tfont)
    d.text(((WIDTH - tw) / 2, 8), title, fill=TITLEBAR_TEXT, font=tfont)

    # command line(s) in green
    y = 36
    for ln in cmd_wrapped:
        d.text((MARGIN_L, y), ln, fill=GREEN, font=font)
        y += LINE_H
    y += LINE_H  # blank line after command

    for ln in body[1:]:  # skip the placeholder blank, already added spacing
        d.text((MARGIN_L, y), ln, fill=FG, font=font)
        y += LINE_H

    img.save(out_path)
    print("wrote %s (%dx%d)" % (out_path, img.width, img.height))


if __name__ == "__main__":
    main()
