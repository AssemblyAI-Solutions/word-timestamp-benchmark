#!/usr/bin/env bash
# Backward-compat shim: regenerate methodology.pdf from methodology.html
# using the generic tools/render-pdf.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$HERE/tools/render-pdf.sh" "$HERE/methodology.html" "$HERE/methodology.pdf"
