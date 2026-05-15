#!/usr/bin/env bash
# Populate ./corpus/ with N earnings22 files (mp3 → 16kHz mono wav, .nlp → plain .txt).
# Usage: ./populate_corpus.sh
set -euo pipefail

CORPUS_DIR="${CORPUS_DIR:-./corpus}"
IDS=(4329526 4351517 4372696 4420696 4423872)

MEDIA_BASE="https://media.githubusercontent.com/media/revdotcom/speech-datasets/main/earnings22/media"
NLP_BASE="https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings22/transcripts/nlp_references"

mkdir -p "$CORPUS_DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for id in "${IDS[@]}"; do
  echo "[$id] downloading mp3 + nlp..."
  curl -fsSL -o "$TMP/$id.mp3" "$MEDIA_BASE/$id.mp3" &
  curl -fsSL -o "$TMP/$id.nlp" "$NLP_BASE/$id.nlp" &
  wait

  echo "[$id] converting mp3 -> 16kHz mono wav..."
  ffmpeg -nostdin -loglevel error -y -i "$TMP/$id.mp3" \
    -ac 1 -ar 16000 "$CORPUS_DIR/$id.wav"

  echo "[$id] parsing nlp -> plain text..."
  python3 - "$TMP/$id.nlp" "$CORPUS_DIR/$id.txt" <<'PY'
import sys, csv
src, dst = sys.argv[1], sys.argv[2]
words = []
with open(src, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="|")
    for row in reader:
        tok = (row.get("token") or "").strip()
        if not tok:
            continue
        # filter non-word artifacts; keep alnum + apostrophes + hyphens
        if not any(c.isalnum() for c in tok):
            continue
        words.append(tok)
with open(dst, "w", encoding="utf-8") as f:
    f.write(" ".join(words) + "\n")
print(f"  wrote {len(words)} words")
PY
done

# Write corpus.json sidecar.
python3 - "$CORPUS_DIR" "${#IDS[@]}" <<'PY'
import json, sys
from pathlib import Path
dst = Path(sys.argv[1])
n = int(sys.argv[2])
(dst / "corpus.json").write_text(json.dumps({
    "name": "revdotcom/speech-datasets (earnings22)",
    "config": None,
    "split": None,
    "source_url": "https://github.com/revdotcom/speech-datasets/tree/main/earnings22",
    "num_utterances": n,
    "audio_col": "mp3",
    "text_col": "nlp_references",
    "notes": "Long-form earnings calls (30-60 min each). Needs pre-segmentation before MFA will reliably align.",
}, indent=2) + "\n")
print(f"  wrote {dst / 'corpus.json'}")
PY

echo
echo "Corpus contents:"
ls -lh "$CORPUS_DIR"
