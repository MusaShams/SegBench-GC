"""Validate ICLR 2027 formatting and submission requirements."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def main() -> None:
    required_files = (
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
        "natbib.sty",
        "fancyhdr.sty",
        "main.pdf",
        "main.log",
        "main.aux",
    )
    missing = [
        name for name in required_files if not (PAPER / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing ICLR submission files: {', '.join(missing)}"
        )

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    statements = (PAPER / "sections" / "07_statements.tex").read_text(
        encoding="utf-8"
    )
    log = (PAPER / "main.log").read_text(encoding="utf-8")
    aux = (PAPER / "main.aux").read_text(encoding="utf-8")

    if r"\usepackage{iclr2027_conference}" not in main_tex:
        raise ValueError("Paper does not use the ICLR 2027 style.")
    if re.search(r"^\s*\\iclrfinalcopy", main_tex, flags=re.MULTILINE):
        raise ValueError(
            "Submission must remain anonymous; iclrfinalcopy is enabled."
        )
    for heading in (
        "AI use statement",
        "Ethics statement",
        "Reproducibility statement",
    ):
        if heading not in statements:
            raise ValueError(f"Missing required paper section: {heading}")

    page_match = re.search(
        r"\\newlabel\{sec:main-text-end\}"
        r"\{\{[^}]*\}\{(\d+)\}",
        aux,
    )
    if page_match is None:
        raise ValueError("Could not determine main-text page count.")
    main_text_page = int(page_match.group(1))
    if main_text_page > 9:
        raise ValueError(
            f"ICLR main text exceeds 9 pages: {main_text_page}"
        )

    blocking_patterns = (
        r"LaTeX Warning: Citation .* undefined",
        r"LaTeX Warning: Reference .* undefined",
        r"Package .* Error:",
        r"Undefined control sequence",
        r"Overfull \\hbox",
    )
    for pattern in blocking_patterns:
        if re.search(pattern, log):
            raise ValueError(
                f"Blocking LaTeX diagnostic matched: {pattern}"
            )

    print(
        "ICLR submission checks passed: "
        f"anonymous, required statements present, "
        f"main text ends on page {main_text_page}."
    )


if __name__ == "__main__":
    main()
