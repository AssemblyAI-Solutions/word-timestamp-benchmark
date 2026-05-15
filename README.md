# Word-Level Timestamp Eval Harness

A reference harness for evaluating word-level timestamp accuracy of
speech-to-text providers. Designed to be **fair across providers** (same
reference, same alignment) and **transparent** (every step is a small,
auditable piece of code).

Out of the box it supports AssemblyAI, Deepgram, and OpenAI. Running multiple
providers in a single invocation produces a side-by-side HTML report.

---

> For a plain-language walkthrough with worked examples and diagrams, see
> **[`methodology.pdf`](./methodology.pdf)** (renders inline on GitHub) or
> [`methodology.html`](./methodology.html) (open in a browser — also linked
> from each generated report).
>
> For an example of what a finished report looks like — three providers
> on ~1,000 utterances of conversational speech — see
> **[`sample-report.pdf`](./sample-report.pdf)**.

## Methodology

1. **Reference timestamps via forced alignment.** Given audio + a
   corrected ground-truth transcript, run
   [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io)
   (MFA) to produce reference word start/end times. MFA is a standard,
   open-source forced aligner; using it (instead of an in-house aligner)
   keeps the reference reproducible by anyone.

2. **Hypothesis timestamps via the provider.** Transcribe the audio
   through each provider, requesting word-level timestamps and turning
   off optional post-processing where possible (`punctuate=False`, etc).

3. **Word-sequence alignment.** Reference and hypothesis word sequences
   are aligned by minimum edit distance (`jiwer.process_words`). Tokens
   are normalized identically on both sides (lowercased, outer punctuation
   stripped, apostrophes preserved) so any normalization choice is fair
   across providers.

4. **Multi-word substitution merging.** When a substitution chunk spans
   *N* hypothesis words against 1 reference word (e.g. ref `"1992"` →
   hyp `"nineteen ninety two"` under inverse text normalization), the
   harness merges the *N* hyp words into one virtual word whose start
   is the first word's start and end is the last word's end. This is
   the standard fix for the common ITN-induced bias where end-time
   error gets systematically understated on N:M substitutions. The
   mirror case (M ref words ↔ 1 hyp word) is also handled.

5. **Pooled-pair metrics.** All aligned (ref, hyp) word pairs across the
   corpus are pooled into one set, and metrics are computed on the pooled
   set:
   - **MAE / median / p90** of `|hyp - ref|` in milliseconds, for both
     word starts and word ends.
   - **% within tolerance** at 25, 50, 100, and 200 ms.

   Pooling (rather than weighted-averaging per-file percentiles) is the
   correct corpus-level aggregation for percentile metrics — a weighted
   mean of medians is not a median.

### Caveats worth knowing

- **MFA isn't ground truth.** Typical MFA alignment error is ~30–50 ms on
  clean read speech and more on noisy or spontaneous speech. The numbers
  here are bounded *below* by that floor. Treat absolute numbers as
  "provider performance relative to MFA `english_mfa`", not absolute
  perceptual accuracy. For an externally quoted absolute number, you'd
  need to forced-align + human-verify a sample to validate the reference
  itself first.
- **Word boundaries are inherently fuzzy** in continuous speech
  (coarticulation). MFA's boundary definition is used on both sides of
  the comparison; this is consistent, but it's a *choice* and is not the
  only valid one.
- **Provider-specific constraints.** Some models reject certain options
  in combination — e.g. AAI's `universal-3-pro` requires `format_text` at
  its default. The harness chooses options per-provider that produce
  comparable word outputs without crashing the request.
- **OpenAI word timestamps.** Word-level timestamps are only available
  via `whisper-1` (which needs `response_format=verbose_json` +
  `timestamp_granularities=["word"]`). The newer `gpt-4o-transcribe`
  and `gpt-4o-mini-transcribe` only support `response_format=json` and
  do not currently expose word timestamps via the public API; the harness
  rejects them with a clear error.

---

## Setup

The eval driver and MFA need to live in the same Python environment, since
the driver imports several Python deps *and* shells out to the `mfa` binary.
A dedicated conda env is the cleanest path on macOS / Linux.

### 1. Install conda (if you don't already have it)

```bash
brew install miniforge
conda init zsh          # or `conda init bash`
# close + reopen your terminal so `conda activate` is on PATH
```

### 2. Create a dedicated env with MFA

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner pip -y
conda activate aligner
```

The explicit `pip` is intentional — some miniforge builds don't include it
by default, and `conda run -n aligner pip ...` will silently fall back to
your system pip (installing into the wrong place) when pip is missing from
the env.

### 3. Install the Python deps *into the aligner env*

```bash
# from inside `conda activate aligner`:
pip install -r requirements.txt
```

If you skip the activate and use `conda run` instead, call the env's
Python directly so PATH ordering doesn't bite you:

```bash
conda run -n aligner python -m pip install -r requirements.txt
```

### 4. Download MFA acoustic + dictionary models

```bash
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa
```

Other languages are available — see the MFA docs for the matching
acoustic + dictionary pair.

### 5. Set provider API keys in `.env`

Put your keys in a `.env` next to `timestamp_eval.py`:

```bash
ASSEMBLYAI_API_KEY=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```

The harness loads this automatically via `python-dotenv`. You only need
keys for the providers you actually run.

---

## Corpus layout

MFA expects a directory of paired audio + transcript files sharing the same
stem:

```
corpus/
  earnings22_001.wav
  earnings22_001.txt
  earnings22_002.wav
  earnings22_002.txt
  ...
```

Requirements:
- Audio: any format ffmpeg can read; **mono 16 kHz wav is the canonical
  input**. The included loaders all produce that.
- Transcripts: plain text with the corrected ground-truth words. No
  timestamps, no speaker labels, no formatting markers. Case and
  punctuation are tolerated (MFA tokenizes), but cleaner is better.
- **Utterances should be seconds long, not minutes.** MFA's default beam
  search is tuned for utterances on the order of seconds. If you feed it
  a single 30 minute audio file paired with a 30 minute transcript, it
  treats that as one giant utterance and alignment will silently fail.
  Pre-segment long recordings with a VAD into utterance-sized chunks.

### Option A: LibriSpeech dev-clean (smoke test)

```bash
./populate_corpus.sh                # 50 utterances (~8 min audio)
NUM_UTTS=200 ./populate_corpus.sh   # more
```

LibriSpeech is clean read speech, ~5–15 seconds per utterance, already
16 kHz mono. MFA aligns the whole batch with no fuss. Fastest path to
a working eval.

### Option B: Hugging Face datasets (any ASR dataset)

```bash
# CommonVoice English test set
python populate_corpus_hf.py \
  --dataset mozilla-foundation/common_voice_17_0 \
  --config en --split test \
  --num 200 --text-col sentence

# AMI meeting microphone segments
python populate_corpus_hf.py \
  --dataset edinburghcstr/ami --config ihm --split test --num 100

# TED-LIUM 3
python populate_corpus_hf.py \
  --dataset LIUM/tedlium --config release3 --split test --num 100
```

The loader auto-detects common transcript column names (`text`, `sentence`,
`transcription`, `transcript`, `normalized_text`); pass `--text-col` to
override. Some datasets gate access — run `hf auth login` first if needed.

### Option C: bring your own audio + transcripts

Drop your paired `<stem>.wav` + `<stem>.txt` files into `./corpus/`
directly. The wavs should be 16 kHz mono (use ffmpeg to convert) and
each utterance should be tens of seconds at most.

Optionally drop a `corpus.json` next to the audio so the eval report
names the dataset properly in its header. Minimal schema:

```json
{
  "name": "Internal demo audio",
  "split": "v1",
  "source_url": "https://example.com/...",
  "num_utterances": 42,
  "notes": "5 speakers, mixed lapel + desk mic"
}
```

All fields are optional. The bundled `populate_corpus*.sh` and
`populate_corpus_hf.py` write this file automatically. You can also
override the displayed name at run time with
`--dataset-name "..."` on the eval CLI.

### Long-form (earnings22) — caveat emptor

`populate_corpus_earnings22.sh` pulls 5 long earnings calls from rev.ai's
public dataset. **You will need to pre-segment them before MFA succeeds.**
This script is included for reference / future segmentation work; it is
*not* a turnkey path to a working eval.

---

## Running

Make sure the aligner env is active (`conda activate aligner`) so both
`mfa` and the Python deps are on PATH.

### Single provider

```bash
python timestamp_eval.py \
  --corpus-dir ./corpus --mfa-output ./mfa_out --run-mfa \
  --providers assemblyai:universal-3-pro \
  # --out defaults to results-<UTC timestamp>.json so re-runs don't overwrite
```

Skip `--run-mfa` on re-runs (re-uses TextGrids in `--mfa-output`).

### Multi-provider, single run (recommended)

This produces one results JSON + one side-by-side HTML report:

```bash
python timestamp_eval.py \
  --corpus-dir ./corpus --mfa-output ./mfa_out --run-mfa \
  --providers assemblyai:universal-3-pro,deepgram:nova-3,openai:whisper-1 \
  # --out defaults to results-<UTC timestamp>.json so re-runs don't overwrite
```

The HTML report sits next to the JSON with the same stem (e.g.
`results-20260515T053046Z.json` → `results-20260515T053046Z.html`) and
opens directly in a browser — no server needed. Pass `--no-html` if you
only want JSON.

### Available models

| Provider     | Default          | Other accepted                              |
|--------------|------------------|---------------------------------------------|
| `assemblyai` | `universal-3-pro` | `universal-2`, `slam-1`, legacy enum values |
| `deepgram`   | `nova-3`         | any model the Deepgram API accepts          |
| `openai`     | `whisper-1`      | only `whisper-1` supports word timestamps   |

---

## Output

### What gets written

Every successful run writes four files:

| File | Purpose |
|------|---------|
| `results-<dataset>-<UTC timestamp>.json` | Machine-readable results: pooled metrics, per-file detail, run metadata. Stable schema. |
| `results-<dataset>-<UTC timestamp>.html` | Self-contained browser report: provider cards, side-by-side tables, charts, per-file detail. |
| `latest.json` | Symlink to the most-recent JSON. Stable filename for scripts. |
| `latest.html` | Symlink to the most-recent HTML. Stable URL for the `methodology.html` "latest results" link. |

The `<dataset>` slug is derived automatically from `corpus.json` (e.g. `peoples_speech-clean`) so a directory of past runs is self-describing at a glance. If there's no `corpus.json`, the slug is omitted and the filename is just `results-<timestamp>.json`. Override with `--dataset-name "..."` to set the slug explicitly.

`latest.*` always points at whatever the *most recently completed run* wrote — running a quick test after a real eval will repoint it. If you want a stable canonical pointer alongside `latest`, just `cp results-<dataset>-<ts>.html canonical.html` after the run.

Past runs are never deleted automatically — the directory grows over time. Clean up with `rm results-*-{prefix}-*.json` etc. when you don't need them.

### `results-<dataset>-<timestamp>.json`

```json
{
  "generated_at": "...",
  "corpus_dir": "./corpus",
  "tolerance_buckets_ms": [25, 50, 100, 200],
  "runs": [
    {
      "provider": "assemblyai",
      "model": "universal-3-pro",
      "files_scored": 50,
      "files_skipped_no_textgrid": 0,
      "files_failed": 0,
      "pooled": {
        "matched_pairs": 819,
        "ref_words": 850,
        "match_rate_pct": 96.4,
        "start_mae_ms": 65.9,
        "start_median_ms": 40.0,
        "start_p90_ms": 180.0,
        "end_mae_ms": 56.0,
        "end_median_ms": 35.0,
        "end_p90_ms": 160.0,
        "start_within_25ms_pct": 30.1,
        "start_within_50ms_pct": 43.2,
        "start_within_100ms_pct": 79.2,
        "start_within_200ms_pct": 99.3,
        "end_within_..."
      },
      "per_file": [ { "audio": "...", "status": "ok", "..." } ]
    }
  ]
}
```

### `results-<dataset>-<timestamp>.html`

A self-contained page with:
- Pooled comparison table across providers (start + end metrics).
- Bar charts of tolerance-bucket percentages (Chart.js via CDN).
- Per-file detail tables, collapsible per provider.
- A short caveats block explaining how to read the numbers.

A reasonable headline for a vendor comparison:

> Universal-3 Pro: start MAE 66 ms, 79 % of word starts within 100 ms
> (n = 819 aligned words across 50 LibriSpeech dev-clean utterances;
> reference = MFA `english_mfa`).

---

## File overview

| File | Purpose |
|------|---------|
| `timestamp_eval.py` | CLI driver: orchestrates MFA, transcription, alignment, metrics, report |
| `transcribers/base.py` | `Word` dataclass + provider registry |
| `transcribers/assemblyai.py` | AssemblyAI provider (Universal-3 Pro, Universal-2, legacy enum) |
| `transcribers/deepgram.py` | Deepgram provider (nova-3, etc) |
| `transcribers/openai.py` | OpenAI provider (whisper-1) |
| `report.py` | HTML report renderer |
| `templates/report.html.j2` | Jinja2 HTML template |
| `methodology.html` | Plain-language walkthrough with worked examples (open in a browser) |
| `methodology.pdf` | PDF render of the above — renders inline on GitHub. Regenerate with `tools/render-methodology-pdf.sh` after editing the HTML. |
| `sample-report.pdf` | Example output: full report on ~1,000 utterances of MLCommons/peoples_speech across three providers. |
| `tools/render-pdf.sh` | Headless-Chrome HTML-to-PDF helper. Works for the methodology page and for any generated report (waits for Chart.js to render before snapshotting). |
| `tools/render-methodology-pdf.sh` | Thin shim that calls `render-pdf.sh` for the methodology specifically. |
| `populate_corpus.sh` | LibriSpeech dev-clean utterance loader |
| `populate_corpus_hf.py` | Generic Hugging Face dataset loader |
| `populate_corpus_earnings22.sh` | Long-form earnings22 puller (needs pre-segmentation) |
| `requirements.txt` | Python dependencies |
