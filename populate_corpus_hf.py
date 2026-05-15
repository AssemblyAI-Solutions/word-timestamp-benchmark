"""Populate ./corpus/ from a Hugging Face audio dataset.

Pulls the first N rows of a dataset, decodes the audio column to 16 kHz mono
wav, and writes the transcript column as plain text. Output layout matches
what `timestamp_eval.py` expects: paired `<stem>.wav` + `<stem>.txt`.

Examples:
  # LibriSpeech dev-clean, 100 utterances
  python populate_corpus_hf.py --dataset openslr/librispeech_asr \\
    --config clean --split validation --num 100

  # CommonVoice English, 200 utterances
  python populate_corpus_hf.py --dataset mozilla-foundation/common_voice_17_0 \\
    --config en --split test --num 200 \\
    --audio-col audio --text-col sentence

  # AMI meeting corpus, microphone segments
  python populate_corpus_hf.py --dataset edinburghcstr/ami --config ihm \\
    --split test --num 50

Notes
-----
  * The dataset must already provide pre-segmented utterances (typically
    seconds-long, not minutes-long). MFA's default beam can't reliably align
    multi-minute single utterances; if you bring a long-form dataset, you'll
    need to pre-segment it before feeding it to MFA.
  * Use --audio-col / --text-col to override column names. The defaults
    (`audio` and `text` / `sentence` / `transcription`) cover most ASR
    datasets.
  * Some HF datasets gate access (CommonVoice, etc). Run `hf auth login`
    first if needed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from datasets import Audio, load_dataset


_TEXT_COL_CANDIDATES = ("text", "sentence", "transcription", "transcript", "normalized_text")


def _safe_stem(s: str) -> str:
    """Sanitize a string to a filesystem-safe stem."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "utt"


def _guess_text_col(row: dict, hint: str | None) -> str:
    if hint:
        if hint not in row:
            raise KeyError(f"--text-col {hint!r} not in dataset row (have: {list(row)})")
        return hint
    for c in _TEXT_COL_CANDIDATES:
        if c in row:
            return c
    raise KeyError(
        f"Couldn't auto-detect a transcript column. "
        f"Available columns: {list(row)}. Pass --text-col."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, help="HF dataset ID (e.g. openslr/librispeech_asr)")
    ap.add_argument("--config", default=None, help="Dataset config / subset name (e.g. clean, en)")
    ap.add_argument("--split", default="validation", help="Split (validation, test, train, ...)")
    ap.add_argument("--num", type=int, default=100, help="Number of utterances to copy")
    ap.add_argument("--audio-col", default="audio", help="Audio column name")
    ap.add_argument("--text-col", default=None, help="Transcript column name (auto-detected if omitted)")
    ap.add_argument("--id-col", default=None,
                    help="Column to use as filename stem. If omitted, falls back to "
                         "the audio file's basename, then to a numeric index.")
    ap.add_argument("--corpus-dir", default="./corpus", type=Path)
    args = ap.parse_args()

    args.corpus_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}"
          f"{('[' + args.config + ']') if args.config else ''}"
          f" split={args.split}, streaming={True}")
    ds = load_dataset(
        args.dataset, args.config,
        split=args.split, streaming=True, trust_remote_code=False,
    )
    # Disable HF's default audio decoder — we hand bytes to ffmpeg directly.
    # This avoids dragging in torch / torchcodec just for a one-off corpus pull.
    try:
        ds = ds.cast_column(args.audio_col, Audio(decode=False))
    except Exception:
        # Column may not be declared as Audio — fall back to whatever it is.
        pass

    text_col = None
    count = 0
    for i, row in enumerate(ds):
        if count >= args.num:
            break
        if text_col is None:
            text_col = _guess_text_col(row, args.text_col)

        audio = row.get(args.audio_col)
        if not audio:
            continue
        text = (row.get(text_col) or "").strip()
        if not text:
            continue

        # Derive a stem
        if args.id_col and args.id_col in row and row[args.id_col]:
            stem = _safe_stem(str(row[args.id_col]))
        elif isinstance(audio, dict) and audio.get("path"):
            stem = _safe_stem(Path(audio["path"]).stem)
        else:
            stem = f"utt_{i:06d}"

        wav_path = args.corpus_dir / f"{stem}.wav"
        txt_path = args.corpus_dir / f"{stem}.txt"
        if wav_path.exists() and txt_path.exists():
            count += 1
            continue

        # The audio column comes through in one of three shapes depending on
        # the dataset and whether decoding is disabled:
        #   {"bytes": b"...", "path": "..."}        decode=False, in-memory bytes
        #   {"path": "/abs/path/to/file"}            decode=False, on-disk path
        #   {"array": np.ndarray, "sampling_rate": int}  decoded (needs torch)
        # We hand bytes or path to ffmpeg directly; for arrays, fall back to
        # soundfile if it happens to be importable.
        if isinstance(audio, dict) and audio.get("bytes"):
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-i", "pipe:0", "-ac", "1", "-ar", "16000", str(wav_path)],
                input=audio["bytes"], check=True,
            )
        elif isinstance(audio, dict) and audio.get("path"):
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-i", audio["path"], "-ac", "1", "-ar", "16000", str(wav_path)],
                check=True,
            )
        elif isinstance(audio, dict) and "array" in audio:
            import soundfile as sf  # optional; only needed for pre-decoded arrays
            tmp_wav = args.corpus_dir / f"{stem}.tmp.wav"
            sf.write(str(tmp_wav), audio["array"], int(audio["sampling_rate"]))
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-i", str(tmp_wav), "-ac", "1", "-ar", "16000", str(wav_path)],
                check=True,
            )
            tmp_wav.unlink(missing_ok=True)
        elif isinstance(audio, str):
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-i", audio, "-ac", "1", "-ar", "16000", str(wav_path)],
                check=True,
            )
        else:
            print(f"  skipping {stem}: unrecognized audio shape {type(audio)}")
            continue

        txt_path.write_text(text + "\n")
        count += 1
        if count % 25 == 0:
            print(f"  {count} / {args.num}")

    # Write corpus.json sidecar so the eval report can name the dataset.
    sidecar = {
        "name": args.dataset,
        "config": args.config,
        "split": args.split,
        "source_url": f"https://huggingface.co/datasets/{args.dataset}",
        "num_utterances": count,
        "audio_col": args.audio_col,
        "text_col": text_col,
    }
    (args.corpus_dir / "corpus.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"\nWrote {count} utterance pairs to {args.corpus_dir}")
    print(f"Wrote corpus metadata to {args.corpus_dir / 'corpus.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
