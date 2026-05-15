#!/usr/bin/env bash
# Render methodology.html -> methodology.pdf using a local headless Chrome.
# Run this whenever methodology.html changes so the PDF in the repo stays in
# sync (GitHub renders PDFs inline, HTML only as source — the PDF is what a
# reader on github.com actually sees).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/methodology.html"
DST="$HERE/methodology.pdf"

if [[ ! -f "$SRC" ]]; then
  echo "error: $SRC not found" >&2
  exit 1
fi

# Find a Chrome/Chromium binary — order is most-likely-installed first.
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
    BROWSER="$c"
    break
  fi
done

if [[ -z "$BROWSER" ]]; then
  echo "error: no Chrome/Chromium binary found. Install Google Chrome or set" >&2
  echo "       \$BROWSER to a headless-capable browser binary." >&2
  exit 1
fi

echo "Rendering $SRC -> $DST"
echo "  via $BROWSER"

"$BROWSER" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$DST" \
  "file://$SRC"

echo "Done: $(wc -c < "$DST" | tr -d ' ') bytes"
