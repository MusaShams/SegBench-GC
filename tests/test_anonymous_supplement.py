from pathlib import Path
import zipfile

import pytest

import scripts.build_anonymous_supplement as anonymous_supplement
from scripts.build_anonymous_supplement import (
    FORBIDDEN,
    build_archive,
)


def test_anonymous_supplement_is_deterministic_and_scanned(
    tmp_path,
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_manifest = build_archive(first)
    second_manifest = build_archive(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest == second_manifest
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "README.md" in names
        assert "ARTIFACT_MANIFEST.json" in names
        assert "tfvc/checkin-notes.txt" not in names
        assert "paper/arxiv.tex" not in names
        assert "paper/arxiv.pdf" not in names
        assert "paper/arxiv_submission_metadata.txt" not in names
        for name in names:
            if name.endswith(
                (
                    ".py",
                    ".md",
                    ".tex",
                    ".yaml",
                    ".json",
                    ".csv",
                    ".svg",
                )
            ):
                text = archive.read(name).decode("utf-8")
                for forbidden in FORBIDDEN:
                    assert forbidden.lower() not in text.lower()


def test_identity_leak_aborts_without_partial_archive(
    tmp_path,
    monkeypatch,
) -> None:
    leak = tmp_path / "leak.txt"
    author_name = "".join(("Musa", " ", "Shams"))
    leak.write_text(author_name, encoding="utf-8")

    monkeypatch.setattr(
        anonymous_supplement,
        "included_files",
        lambda: [(leak, Path("leak.txt"))],
    )

    output = tmp_path / "anonymous.zip"

    with pytest.raises(ValueError, match="Anonymity scan failed"):
        anonymous_supplement.build_archive(output)

    assert not output.exists()
    assert not (tmp_path / ".anonymous.zip.tmp").exists()
