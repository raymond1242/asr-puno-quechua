#!/usr/bin/env bash
# Build the paper. Usage:
#   bash paper/build.sh          one pass + bibtex + two passes (full build)
#   bash paper/build.sh watch    rebuild on every save (latexmk -pvc)
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "watch" ]]; then
    exec latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error main.tex
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
echo
echo "PDF: paper/main.pdf"
pdfinfo main.pdf 2>/dev/null | grep -E "^Pages" || true
