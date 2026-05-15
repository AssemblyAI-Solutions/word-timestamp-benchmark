#!/usr/bin/env bash
# Render any HTML file in the repo to a PDF, using a local headless Chrome.
# Works for static pages (methodology.html) and JS-rendered ones (the report
# HTML uses Chart.js loaded from CDN).
#
# Usage:
#   tools/render-pdf.sh methodology.html methodology.pdf
#   tools/render-pdf.sh results-<...>.html sample-report.pdf
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $(basename "$0") <src.html> <dst.pdf>" >&2
  exit 2
fi

SRC="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
DST="$(cd "$(dirname "$2")" 2>/dev/null && pwd)/$(basename "$2") "  # tolerate missing dst
DST="${DST% }"

if [[ ! -f "$SRC" ]]; then
  echo "error: source not found: $SRC" >&2
  exit 1
fi

# Find a Chrome / Chromium binary — most-likely-installed first.
CANDIDATES=(
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
  "/Applications/Chromium.app/Contents/MacOS/Chromium"
  "$(command -v google-chrome 2>/dev/null || true)"
  "$(command -v chromium 2>/dev/null || true)"
  "$(command -v chrome 2>/dev/null || true)"
)
BROWSER=""
for c in "${CANDIDATES[@]}"; do
  if [[ -n "$c" && -x "$c" ]]; then
    BROWSER="$c"; break
  fi
done
if [[ -z "$BROWSER" ]]; then
  echo "error: no Chrome/Chromium binary found. Install Google Chrome." >&2
  exit 1
fi

echo "Rendering $SRC -> $DST"
echo "  via $BROWSER"

# --virtual-time-budget gives JavaScript (e.g. Chart.js) time to run before
# the print snapshot. 15000 ms is enough for charts to render from CDN.
# --run-all-compositor-stages-before-draw ensures layout is settled.
"$BROWSER" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --virtual-time-budget=15000 \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf="$DST" \
  "file://$SRC"

echo "Done: $(wc -c < "$DST" | tr -d ' ') bytes"
