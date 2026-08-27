"""Download a pinned OGBench reference implementation and create its JAX environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REVISION = "1d4140997f60c52c6fb0702ec100dc988b18c548"
ARCHIVE_URL = f"https://github.com/seohongpark/ogbench/archive/{REVISION}.tar.gz"
DEFAULT_SOURCE_DIR = Path(".external/ogbench-official")
DEFAULT_VENV_DIR = Path(".official-venv")
REQUIREMENTS = [
    "jax==0.4.30",
    "jaxlib==0.4.30",
    "flax==0.8.5",
    "optax==0.2.3",
    "distrax==0.1.5",
    "ml-collections==0.1.1",
    "matplotlib==3.9.4",
    "moviepy==1.0.3",
    "wandb==0.17.9",
    "ogbench==1.2.1",
    "mujoco==3.3.7",
]


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Archive member escapes destination: {member.name}")
    archive.extractall(destination)


def ensure_source(source_dir: Path, force: bool = False) -> Path:
    marker = source_dir / "impls" / "agents" / "gciql.py"
    if marker.exists() and not force:
        return source_dir
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "ogbench.tar.gz"
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, extract_dir)
        extracted_roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(extracted_roots) != 1:
            raise RuntimeError("Expected exactly one root directory in the OGBench archive.")
        shutil.move(str(extracted_roots[0]), str(source_dir))
    if not marker.exists():
        raise FileNotFoundError(f"Official GCIQL implementation was not found at {marker}.")
    return source_dir


def enable_wandb_env_override(source_dir: Path) -> None:
    log_utils = source_dir / "impls" / "utils" / "log_utils.py"
    text = log_utils.read_text(encoding="utf-8")
    original = "        mode=mode,\n"
    replacement = "        mode=os.environ.get('WANDB_MODE', mode),\n"
    if replacement in text:
        return
    if original not in text:
        raise RuntimeError("Could not locate the official W&B mode assignment to patch.")
    log_utils.write_text(text.replace(original, replacement, 1), encoding="utf-8")


def ensure_venv(venv_dir: Path, force: bool = False) -> Path:
    python = venv_dir / "bin" / "python"
    if force and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", *REQUIREMENTS], check=True)
    return python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--force-source", action="store_true")
    parser.add_argument("--force-venv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = ensure_source(args.source_dir, force=args.force_source)
    enable_wandb_env_override(source_dir)
    python = ensure_venv(args.venv_dir, force=args.force_venv)
    subprocess.run(
        [
            str(python),
            "-c",
            "import jax, flax, ogbench; print('official baseline ready', jax.__version__, flax.__version__)",
        ],
        check=True,
    )
    print(f"source={source_dir}")
    print(f"python={python}")


if __name__ == "__main__":
    main()
