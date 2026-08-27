"""Build a deterministic, anonymity-scanned ICLR code supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bib",
    ".bst",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sty",
    ".svg",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_NAMES = {
    ".DS_Store",
    "build_anonymous_supplement.py",
    "test_arxiv_package.py",
    "iclr2027_submission_checklist.txt",
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.log",
    "main.out",
    "main.pdf",
    "arxiv.aux",
    "arxiv.bbl",
    "arxiv.blg",
    "arxiv.log",
    "arxiv.out",
    "arxiv.pdf",
    "arxiv.tex",
    "arxiv_submission_metadata.txt",
}
EXCLUDED_PARTS = {
    ".external",
    ".git",
    ".pytest_cache",
    ".tf",
    ".venv",
    "__pycache__",
    "adaptive_gcrl.egg-info",
    "runs",
    "tfvc",
}
ANONYMIZATION = {
    "https://dev.azure.com/bruhmoment123/": "",
    "MacTFVC": "",
    "/Users/Musa.Shams/Stuff/RL": "/path/to/segbench-gc",
    "/Users/Musa.Shams": "/home/anonymous",
    "/home/Musa.Shams": "/home/anonymous",
    "Musa.Shams": "anonymous",
}
FORBIDDEN = (
    "Musa.Shams",
    "Musa Shams",
    "MusaShams",
    "github.com/MusaShams",
    "0009-0005-1015-5342",
    "bruhmoment123",
    "adaptive-gcrl-research-unique",
    "project-655d82d0",
    "slumsdotgov",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "segbench-gc-anonymous.zip",
    )
    return parser.parse_args()


def tracked_paths() -> list[Path]:
    """Return repository-relative Git-tracked paths in deterministic order."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Building the anonymous supplement requires a Git checkout so "
            "local untracked files cannot silently enter the submission archive."
        ) from exc
    return [Path(item) for item in result.stdout.split("\0") if item]


def included_files() -> list[tuple[Path, Path]]:
    allowed_prefixes = (
        Path("configs"),
        Path("paper"),
        Path("scripts"),
        Path("src/adaptive_gcrl"),
        Path("tests"),
    )
    special = {
        Path("supplement/README.md"): Path("README.md"),
        Path("pyproject.toml"): Path("pyproject.toml"),
        Path("setup.py"): Path("setup.py"),
    }

    files: list[tuple[Path, Path]] = []
    for relative in tracked_paths():
        source = ROOT / relative
        if not source.is_file():
            continue
        if source.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if source.suffix in {".pyc", ".pyo"}:
            continue

        if relative in special:
            archive_path = special[relative]
        elif any(
            relative == prefix or prefix in relative.parents
            for prefix in allowed_prefixes
        ):
            archive_path = relative
        else:
            continue
        files.append((source, archive_path))

    return sorted(files, key=lambda item: item[1].as_posix())


def anonymous_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return payload
    text = payload.decode("utf-8")
    for source, replacement in ANONYMIZATION.items():
        text = text.replace(source, replacement)
    for forbidden in FORBIDDEN:
        if forbidden.lower() in text.lower():
            raise ValueError(
                f"Anonymity scan failed for {path}: {forbidden}"
            )
    return text.encode("utf-8")


def build_archive(output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)

    # Never leave a stale or partially written submission artifact behind.
    output.unlink(missing_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    temporary_output.unlink(missing_ok=True)

    manifest: dict[str, str] = {}
    try:
        with zipfile.ZipFile(
            temporary_output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source, relative in included_files():
                payload = anonymous_bytes(source)
                name = relative.as_posix()
                manifest[name] = hashlib.sha256(payload).hexdigest()
                info = zipfile.ZipInfo(
                    name,
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (
                    0o755 if source.stat().st_mode & 0o111 else 0o644
                ) << 16
                archive.writestr(info, payload)

            manifest_payload = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            info = zipfile.ZipInfo(
                "ARTIFACT_MANIFEST.json",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, manifest_payload)

        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise

    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_archive(args.output)
    print(
        f"Wrote {args.output} with {len(manifest)} files; "
        f"SHA-256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
