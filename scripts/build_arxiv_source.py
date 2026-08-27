"""Build a deterministic arXiv source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "segbench-gc-arxiv-source.zip",
    )
    return parser.parse_args()


def source_files() -> list[tuple[Path, Path]]:
    paths = [
        PAPER / "arxiv.tex",
        PAPER / "references.bib",
        PAPER / "natbib.sty",
        PAPER / "iclr2027_conference.bst",
    ]
    paths.extend(sorted((PAPER / "sections").glob("*.tex")))
    paths.extend(sorted((PAPER / "tables").glob("*.tex")))
    paths.extend(
        sorted(
            path
            for path in (PAPER / "figures").glob("*.pdf")
            if path.name.startswith("segbench_gc_")
        )
    )
    return [
        (path, path.relative_to(PAPER))
        for path in paths
    ]


def build_archive(output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source, relative in source_files():
            payload = source.read_bytes()
            name = relative.as_posix()
            manifest[name] = hashlib.sha256(payload).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
        manifest_payload = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        info = zipfile.ZipInfo(
            "SOURCE_MANIFEST.json",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, manifest_payload)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_archive(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"Wrote {args.output} with {len(manifest)} source files; "
        f"SHA-256 {digest}"
    )


if __name__ == "__main__":
    main()
