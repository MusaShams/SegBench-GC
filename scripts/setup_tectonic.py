"""Install a pinned portable Tectonic binary for paper compilation.

The Linux installer deliberately uses the static musl build so Ubuntu systems
with older glibc releases can run the same executable reproducibly.
"""

from __future__ import annotations

import argparse
import io
import platform
import shutil
import tarfile
import urllib.request
from pathlib import Path


VERSION = "0.16.9"
DEFAULT_OUTPUT = Path(".tools/tectonic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def release_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "x86_64-unknown-linux-musl"
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return "aarch64-unknown-linux-musl"
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return "x86_64-apple-darwin"
    if system == "darwin" and machine in {"aarch64", "arm64"}:
        return "aarch64-apple-darwin"
    raise RuntimeError(
        f"Unsupported platform for pinned Tectonic binary: {system}/{machine}"
    )


def download_binary(output: Path) -> Path:
    target = release_target()
    url = (
        "https://github.com/tectonic-typesetting/tectonic/releases/download/"
        f"tectonic%40{VERSION}/tectonic-{VERSION}-{target}.tar.gz"
    )
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        matching = [member for member in members if Path(member.name).name == "tectonic"]
        if len(matching) != 1:
            raise RuntimeError(
                f"Expected exactly one tectonic executable in {url}; found {len(matching)}"
            )
        source = archive.extractfile(matching[0])
        if source is None:
            raise RuntimeError("Could not extract Tectonic executable from release archive.")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        temporary.chmod(0o755)
        temporary.replace(output)
    return output


def main() -> None:
    args = parse_args()
    output = args.output
    if output.exists() and not args.force:
        print(f"Tectonic already exists at {output}; use --force to replace it.")
        return
    path = download_binary(output)
    print(f"Installed Tectonic {VERSION} at {path}")
    print("Linux uses the static musl build to avoid host glibc-version coupling.")


if __name__ == "__main__":
    main()
