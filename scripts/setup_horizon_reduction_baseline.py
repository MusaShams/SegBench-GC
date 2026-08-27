"""Install a pinned independent n-step GCSAC+BC reference implementation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REVISION = "c298aedcc505bc7a7b44b4d0c9318993f8b3f3fd"
ARCHIVE_URL = (
    "https://github.com/seohongpark/horizon-reduction/archive/"
    f"{REVISION}.tar.gz"
)
DEFAULT_SOURCE_DIR = Path(".external/horizon-reduction")
DEFAULT_VENV_DIR = Path(".horizon-reduction-venv")
COMMON_REQUIREMENTS = [
    "flax==0.8.5",
    "optax==0.2.3",
    "distrax==0.1.5",
    "ml-collections==0.1.1",
    "matplotlib==3.9.4",
    "moviepy==1.0.3",
    "wandb==0.17.9",
    "ogbench==1.2.1",
    "mujoco==3.3.7",
    "tqdm>=4.66",
]


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Archive member escapes destination: {member.name}")
    archive.extractall(destination)


def ensure_source(source_dir: Path, force: bool = False) -> Path:
    marker = source_dir / "agents" / "ngcsacbc.py"
    if marker.exists() and not force:
        return source_dir
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "horizon-reduction.tar.gz"
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, extract_dir)
        roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Expected one root directory in reference archive.")
        shutil.move(str(roots[0]), str(source_dir))
    if not marker.is_file():
        raise FileNotFoundError(f"Missing n-step GCSAC+BC implementation: {marker}")
    return source_dir


def ensure_venv(
    venv_dir: Path,
    *,
    force: bool = False,
    cpu: bool = False,
) -> Path:
    python = venv_dir / "bin" / "python"
    if force and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
        )
        jax_requirement = "jax==0.4.30" if cpu else "jax[cuda12]==0.4.30"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                jax_requirement,
                *COMMON_REQUIREMENTS,
            ],
            check=True,
        )
    return python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--force-source", action="store_true")
    parser.add_argument("--force-venv", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install CPU JAX instead of the default CUDA 12 build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ensure_source(args.source_dir, force=args.force_source)
    python = ensure_venv(
        args.venv_dir,
        force=args.force_venv,
        cpu=args.cpu,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            (
                "import jax, flax, ogbench; "
                "print('horizon-reduction baseline ready', "
                "jax.__version__, flax.__version__, jax.devices())"
            ),
        ],
        check=True,
    )
    print(f"revision={REVISION}")
    print(f"source={source}")
    print(f"python={python}")


if __name__ == "__main__":
    main()
