#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[*] Compiling LaTeX paper..."
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
bibtex main || true
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex

echo "[✓] Built latex/main.pdf successfully!"
