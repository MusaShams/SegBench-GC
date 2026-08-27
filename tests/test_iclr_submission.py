from pathlib import Path


def test_iclr_source_is_anonymous_and_has_required_sections() -> None:
    main = Path("paper/main.tex").read_text(encoding="utf-8")
    statements = Path("paper/sections/07_statements.tex").read_text(
        encoding="utf-8"
    )

    assert r"\usepackage{iclr2027_conference}" in main
    assert "% \\iclrfinalcopy" in main
    assert "\n\\iclrfinalcopy" not in main
    assert "AI use statement" in statements
    assert "Ethics statement" in statements
    assert "Reproducibility statement" in statements


def test_official_iclr_style_assets_are_vendored() -> None:
    for name in (
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
        "natbib.sty",
        "fancyhdr.sty",
    ):
        assert (Path("paper") / name).is_file()


def test_iclr_submission_checker_tracks_page_limit() -> None:
    script = Path("scripts/check_iclr_submission.py").read_text(
        encoding="utf-8"
    )

    assert "sec:main-text-end" in script
    assert "main_text_page > 9" in script
