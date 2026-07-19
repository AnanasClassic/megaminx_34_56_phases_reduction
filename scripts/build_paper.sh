#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAPER="$ROOT/paper"
OUTPUT="$ROOT/output/pdf/megaminx_boundary_layer_preprint.pdf"

command -v latexmk >/dev/null 2>&1 || {
  echo "error: latexmk is required to build the paper" >&2
  exit 2
}
command -v rg >/dev/null 2>&1 || {
  echo "error: ripgrep (rg) is required for the LaTeX log audit" >&2
  exit 2
}

cd "$PAPER"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

if [[ ! -s main.pdf ]] || ! tail -c 1024 main.pdf | rg -a -q '%%EOF'; then
  echo "error: paper/main.pdf is empty or truncated" >&2
  exit 2
fi

if rg -n \
  'LaTeX Warning:.*undefined|Citation .* undefined|Reference .* undefined|There were undefined references|Overfull \\hbox|Overfull \\vbox' \
  main.log; then
  echo "error: unresolved reference or overfull box in paper/main.log" >&2
  exit 2
fi
if [[ -f main.blg ]] && rg -n '^Warning--' main.blg; then
  echo "error: BibTeX warning in paper/main.blg" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
cp main.pdf "$OUTPUT"
if ! cmp -s main.pdf "$OUTPUT" || ! tail -c 1024 "$OUTPUT" | rg -a -q '%%EOF'; then
  echo "error: copied PDF differs from paper/main.pdf or is truncated" >&2
  exit 2
fi
echo "paper built: ${OUTPUT#$ROOT/}"
