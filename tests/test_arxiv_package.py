import zipfile
from pathlib import Path

from scripts.build_arxiv_source import build_archive


def test_arxiv_manuscript_identifies_solo_author() -> None:
    source = Path("paper/arxiv.tex").read_text(encoding="utf-8")
    author_name = "".join(("Musa", " ", "Shams"))

    assert rf"\author{{{author_name}\\Independent Researcher}}" in source
    assert "iclrfinalcopy" not in source
    assert r"\input{sections/07_statements}" in source
    assert r"\appendix" in source
    statements = Path("paper/sections/07_statements.tex").read_text(
        encoding="utf-8"
    )
    assert "The author reviewed all AI-assisted work" in statements


def test_arxiv_source_archive_is_deterministic(tmp_path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_manifest = build_archive(first)
    second_manifest = build_archive(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest == second_manifest
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "arxiv.tex" in names
        assert "SOURCE_MANIFEST.json" in names
        assert "main.tex" not in names
        assert "main.pdf" not in names
