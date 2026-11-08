#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

run() {
    if ! "$@"; then
        echo "ERROR: $*" >&2
        exit 1
    fi
}

echo "[*] Compiling LaTeX report..."
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
bibtex main || true
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex

echo "[✓] Built main.pdf successfully!"
cp main.pdf /home/vishrut/calibrated-kd/docs/master_research_report.pdf
cp main.pdf /home/vishrut/.gemini/antigravity/brain/269ec839-7b13-4c7b-a244-7e54d8464da8/master_research_report.pdf
echo "[✓] Mirrored PDF to docs/ and artifact directory!"

