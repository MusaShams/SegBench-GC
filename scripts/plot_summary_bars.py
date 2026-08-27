"""Generate lightweight SVG bar charts from paper summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def svg_bar_chart(summary: dict, *, title: str, ylabel: str, lower_is_better: bool = False, group_key: str = "by_algorithm") -> str:
    by_algorithm = summary.get(group_key)
    if not isinstance(by_algorithm, dict) or not by_algorithm:
        raise ValueError(f"Summary JSON must contain a non-empty {group_key} object.")

    items = [(str(name), float(values["mean"])) for name, values in sorted(by_algorithm.items())]
    max_value = max(value for _, value in items)
    scale_max = max(max_value, 1e-8)
    width = 640
    height = 360
    margin_left = 80
    margin_bottom = 70
    plot_width = width - margin_left - 40
    plot_height = height - 70 - margin_bottom
    bar_gap = 36
    bar_width = max(40, int((plot_width - bar_gap * (len(items) + 1)) / max(len(items), 1)))

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 20 {height / 2})">{ylabel}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 30}" y2="{height - margin_bottom}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="55" x2="{margin_left}" y2="{height - margin_bottom}" stroke="black"/>',
    ]
    subtitle = "lower is better" if lower_is_better else "higher is better"
    elements.append(f'<text x="{width - 35}" y="52" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{subtitle}</text>')

    for index, (algorithm, value) in enumerate(items):
        x = margin_left + bar_gap + index * (bar_width + bar_gap)
        bar_height = 0.0 if scale_max == 0.0 else (value / scale_max) * plot_height
        y = height - margin_bottom - bar_height
        color = "#4C78A8" if algorithm == "iql" else "#F58518"
        elements.extend(
            [
                f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{color}"/>',
                f'<text x="{x + bar_width / 2}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>',
                f'<text x="{x + bar_width / 2}" y="{height - margin_bottom + 22}" text-anchor="middle" font-family="Arial" font-size="12">{algorithm}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--ylabel", required=True)
    parser.add_argument("--group-key", default="by_algorithm")
    parser.add_argument("--lower-is-better", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    svg = svg_bar_chart(
        summary,
        title=args.title,
        ylabel=args.ylabel,
        lower_is_better=args.lower_is_better,
        group_key=args.group_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
