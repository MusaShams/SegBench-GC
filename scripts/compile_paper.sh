#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root/paper"

if [[ -n "${TECTONIC:-}" ]]; then
  tectonic_bin="$TECTONIC"
elif [[ -x "$repo_root/.tools/tectonic" ]]; then
  tectonic_bin="$repo_root/.tools/tectonic"
else
  tectonic_bin="tectonic"
fi

document="${1:-main.tex}"
stem="${document%.tex}"
if ! command -v "$tectonic_bin" >/dev/null 2>&1; then
  echo "Tectonic is required. Run: .venv/bin/python scripts/setup_tectonic.py --force" >&2
  echo "or set TECTONIC=/path/to/tectonic." >&2
  exit 1
fi

rm -f "$stem.aux" "$stem.bbl" "$stem.blg" "$stem.log" "$stem.out"
"$tectonic_bin" "$document" --keep-logs --keep-intermediates
