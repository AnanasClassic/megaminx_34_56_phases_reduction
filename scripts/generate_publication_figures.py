#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def render(source: Path, output: Path, title: str) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [(row["method"], int(row["count"])) for row in csv.DictReader(handle)]
    width, height = 920, 500
    left, right, top, bottom = 82, 28, 55, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = math.ceil(math.log10(max(count for _, count in rows)))
    bar_step = plot_w / len(rows)
    bar_w = bar_step * 0.62

    def y(value: int) -> float:
        return top + plot_h * (1 - math.log10(max(value, 1)) / ymax)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        '<desc id="desc">Logarithmic bar chart of exact reductions and certificates first found at each beam width.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>',
    ]
    for exponent in range(ymax + 1):
        value = 10**exponent
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#d6d6d6"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">10^{exponent}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#222"/>')
    for index, (label, count) in enumerate(rows):
        x = left + bar_step * index + (bar_step - bar_w) / 2
        yy = y(count)
        bar_height = max(3.0, top + plot_h - yy)
        bar_top = top + plot_h - bar_height
        color = "#d95f02" if label == "exact reduction" else "#1f77b4"
        parts.append(f'<rect x="{x:.2f}" y="{bar_top:.2f}" width="{bar_w:.2f}" height="{bar_height:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.2f}" y="{bar_top-7:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{count:,}</text>')
        parts.append(f'<text transform="translate({x+bar_w/2:.2f},{top+plot_h+12}) rotate(45)" text-anchor="start" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    parts.append(f'<text transform="translate(18,{top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">states (log10 scale)</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    render(ROOT / "phase_3_4/figures/beam-coverage.csv", ROOT / "phase_3_4/figures/beam-coverage.svg", "Phase 3+4 certificate coverage")
    render(ROOT / "phase_5_6/figures/beam-coverage.csv", ROOT / "phase_5_6/figures/beam-coverage.svg", "Phase 5+6 certificate coverage")


if __name__ == "__main__":
    main()
