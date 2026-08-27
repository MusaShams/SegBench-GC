from pathlib import Path


def test_compile_paper_script_uses_tectonic_from_paper_directory() -> None:
    script = Path("scripts/compile_paper.sh").read_text(encoding="utf-8")

    assert 'repo_root="$(cd "$(dirname "$0")/.." && pwd)"' in script
    assert 'cd "$repo_root/paper"' in script
    assert 'if [[ -n "${TECTONIC:-}" ]]; then' in script
    assert 'elif [[ -x "$repo_root/.tools/tectonic" ]]; then' in script
    assert 'tectonic_bin="$repo_root/.tools/tectonic"' in script
    assert 'document="${1:-main.tex}"' in script
    assert '"$tectonic_bin" "$document"' in script
