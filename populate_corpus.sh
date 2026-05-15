#!/usr/bin/env bash
# Populate ./corpus/ with N short utterances from LibriSpeech dev-clean.
# LibriSpeech utterances are ~5-15 sec each, already 16 kHz mono — ideal for
# out-of-the-box MFA forced alignment.
#
# Usage:
#   ./populate_corpus.sh                  # default 50 utterances
#   NUM_UTTS=200 ./populate_corpus.sh     # more utterances
#
# Tarball is ~340 MB; cached in /tmp across runs.
set -euo pipefail

CORPUS_DIR="${CORPUS_DIR:-./corpus}"
NUM_UTTS="${NUM_UTTS:-50}"
TAR_URL="https://www.openslr.org/resources/12/dev-clean.tar.gz"
TAR_CACHE="/tmp/librispeech-dev-clean.tar.gz"
EXTRACT_DIR="/tmp/librispeech-dev-clean"

mkdir -p "$CORPUS_DIR"

if [[ ! -f "$TAR_CACHE" ]]; then
  echo "Downloading LibriSpeech dev-clean (~340 MB) -> $TAR_CACHE"
  curl -fL --progress-bar -o "$TAR_CACHE" "$TAR_URL"
else
  echo "Using cached tarball at $TAR_CACHE"
fi

if [[ ! -d "$EXTRACT_DIR/LibriSpeech/dev-clean" ]]; then
  echo "Extracting -> $EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  tar -xzf "$TAR_CACHE" -C "$EXTRACT_DIR"
fi

echo "Picking first $NUM_UTTS utterances and copying into $CORPUS_DIR/ ..."
python3 - "$EXTRACT_DIR/LibriSpeech/dev-clean" "$CORPUS_DIR" "$NUM_UTTS" <<'PY'
import sys, subprocess
from pathlib import Path

src_root, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])

# Build a map: utterance_id -> transcript_text
transcripts = {}
for trans_file in sorted(src_root.rglob("*.trans.txt")):
    for line in trans_file.read_text().splitlines():
        if not line.strip():
            continue
        utt_id, text = line.split(" ", 1)
        transcripts[utt_id] = text.strip()

# Walk flacs in sorted order, pick the first N
flacs = sorted(src_root.rglob("*.flac"))
count = 0
for flac in flacs:
    if count >= n:
        break
    utt_id = flac.stem
    if utt_id not in transcripts:
        continue
    wav_out = dst / f"{utt_id}.wav"
    txt_out = dst / f"{utt_id}.txt"
    # LibriSpeech flac is already 16 kHz mono; ffmpeg just decodes to wav.
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-i", str(flac), "-ac", "1", "-ar", "16000", str(wav_out)],
        check=True,
    )
    txt_out.write_text(transcripts[utt_id] + "\n")
    count += 1

print(f"  wrote {count} utterance pairs")

# Write corpus.json sidecar so the eval report can name the dataset.
import json
sidecar = {
    "name": "openslr/librispeech_asr",
    "config": "dev-clean",
    "split": "validation",
    "source_url": "https://www.openslr.org/12",
    "num_utterances": count,
    "audio_col": "flac",
    "text_col": "trans.txt",
}
(dst / "corpus.json").write_text(json.dumps(sidecar, indent=2) + "\n")
print(f"  wrote {dst / 'corpus.json'}")
PY

echo
echo "Corpus contents (first 10):"
ls "$CORPUS_DIR" | head -10
echo "..."
echo "Total files: $(ls "$CORPUS_DIR" | wc -l | tr -d ' ')"
