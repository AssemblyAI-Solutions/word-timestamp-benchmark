"""
Word-level timestamp evaluation harness for speech-to-text providers.

Methodology
-----------
  1. Take audio + corrected ground-truth transcript.
  2. Use Montreal Forced Aligner (MFA) to generate reference word timestamps.
  3. Transcribe the audio with one or more STT providers to get hypothesis
     word timestamps.
  4. Align reference and hypothesis word sequences via minimum edit distance
     (jiwer). Substitution spans of (1 ref word -> N hyp words) — common
     under inverse text normalization, e.g. ref "1992" -> hyp "nineteen ninety
     two" — are merged into a single virtual hyp word spanning the full N.
     This avoids systematically biasing end-time error on N:M substitutions.
  5. Pool all aligned (ref, hyp) pairs across the corpus into one set and
     compute timestamp error metrics on the pooled set:
       * MAE, median, p90 of |hyp - ref| in ms for start and end boundaries
       * % within {25, 50, 100, 200} ms tolerance buckets
     Pooling (rather than averaging per-file percentiles) is the mathematically
     correct corpus-level aggregation for percentile metrics.

Caveats
-------
  * MFA has ~30-50 ms of alignment error on clean read speech and more on
    noisy or spontaneous speech. It's a strong proxy, not "true" ground
    truth. For absolute accuracy claims, forced-align + human-verify a
    sample first.
  * Word boundaries in continuous speech are inherently fuzzy due to
    coarticulation. We use MFA's boundary definition consistently on both
    sides of the comparison, which is fair across providers but is a floor
    on absolute accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
from dotenv import load_dotenv
from textgrid import TextGrid

from transcribers import REGISTRY, TranscriptionError, Word, get_transcriber


# -----------------------------------------------------------------------------
# 1. Reference timestamps via Montreal Forced Aligner
# -----------------------------------------------------------------------------

def run_mfa_alignment(
    corpus_dir: Path,
    output_dir: Path,
    dictionary: str = "english_us_mfa",
    acoustic_model: str = "english_mfa",
    extra_args: Optional[List[str]] = None,
) -> None:
    """Invoke MFA's `align` command on the corpus.

    Expects MFA on PATH (i.e. the env has the `montreal-forced-aligner` conda
    package and the named `dictionary` + `acoustic_model` have been
    downloaded via `mfa model download`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mfa", "align",
        "--clean",
        "--overwrite",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def parse_textgrid(textgrid_path: Path) -> List[Word]:
    """Read an MFA TextGrid and return its words tier as `Word`s in ms."""
    tg = TextGrid.fromFile(str(textgrid_path))
    words: List[Word] = []
    for tier in tg.tiers:
        if tier.name.lower() != "words":
            continue
        for interval in tier.intervals:
            text = (interval.mark or "").strip()
            if not text:
                continue
            words.append(Word(
                text=_normalize_word(text),
                start_ms=int(round(interval.minTime * 1000)),
                end_ms=int(round(interval.maxTime * 1000)),
            ))
    return words


# -----------------------------------------------------------------------------
# 2. Word-sequence alignment + N:M substitution merging
# -----------------------------------------------------------------------------

def _merge_hyp_span(words: List[Word]) -> Word:
    """Merge consecutive hyp words into one virtual word.

    Used when N hyp words correspond to a single ref word in a substitute
    span (e.g. inverse-text-normalized expansions like "1992" -> "nineteen
    ninety two"). The merged word's start is the first word's start_ms and
    its end is the last word's end_ms. `word_count` accumulates so a future
    metric that wants "hyp tokens covered" can recover it.
    """
    if len(words) == 1:
        return words[0]
    return Word(
        text=" ".join(w.text for w in words),
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        word_count=sum(w.word_count for w in words),
    )


def _merge_ref_span(words: List[Word]) -> Word:
    """Merge consecutive ref words into one virtual word.

    Mirror of `_merge_hyp_span` for M ref words -> 1 hyp word
    (e.g. ref "do" "not" -> hyp "don't"). `word_count` accumulates so the
    match-rate metric correctly credits all M ref tokens as "covered" by
    the merged pair, rather than under-counting it as a single pair.
    """
    if len(words) == 1:
        return words[0]
    return Word(
        text=" ".join(w.text for w in words),
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        word_count=sum(w.word_count for w in words),
    )


def _emit_block(
    ref_span: List[Word],
    hyp_span: List[Word],
) -> List[Tuple[Optional[Word], Optional[Word]]]:
    """Emit (ref, hyp) pairs for a run of consecutive non-equal chunks.

    Cases:
      empty ref + N hyp    -> N pure insertions
      M ref + empty hyp    -> M pure deletions
      M ref + M hyp        -> M 1:1 substitution pairs
      1 ref + N hyp (N>1)  -> 1 pair: ref vs merged hyp span (ITN expansion)
      M ref + 1 hyp (M>1)  -> 1 pair: merged ref span vs hyp (ITN contraction)
      M ref + N hyp (else) -> 1 pair: merged ref vs merged hyp
                              (rare; covers complex N:M chunks like
                               compound ITN, e.g. ref "five fifty"
                               -> hyp "5:50")
    """
    out: List[Tuple[Optional[Word], Optional[Word]]] = []
    if not ref_span and not hyp_span:
        return out
    if not ref_span:
        return [(None, h) for h in hyp_span]
    if not hyp_span:
        return [(r, None) for r in ref_span]
    if len(ref_span) == len(hyp_span):
        for r, h in zip(ref_span, hyp_span):
            out.append((r, h))
        return out
    if len(ref_span) == 1:
        return [(ref_span[0], _merge_hyp_span(hyp_span))]
    if len(hyp_span) == 1:
        return [(_merge_ref_span(ref_span), hyp_span[0])]
    # General M:N — merge both sides into one comparable span.
    return [(_merge_ref_span(ref_span), _merge_hyp_span(hyp_span))]


def align_words(
    reference: List[Word],
    hypothesis: List[Word],
) -> List[Tuple[Optional[Word], Optional[Word]]]:
    """Align reference and hypothesis word sequences via min edit distance.

    Returns a list of (ref, hyp) pair tuples:
      (Word, Word)   matched pair (equal, or merged substitution span)
      (Word, None)   deletion: word in reference, no hypothesis counterpart
      (None, Word)   insertion: hypothesis word, no reference counterpart

    Implementation note: jiwer returns the edit-distance alignment as a
    sequence of `equal`/`substitute`/`insert`/`delete` chunks. The
    algorithm prefers minimal edits, so an ITN expansion like ref `"1992"`
    -> hyp `"nineteen" "ninety" "two"` typically comes back as a 1:1
    substitute (`"1992"` -> `"nineteen"`) plus 2 insertions, rather than
    as a single 1:3 substitute chunk. To make multi-word merging robust,
    we collapse any consecutive run of non-`equal` chunks into one M:N
    span and apply the merge rules to the combined span.
    """
    if not reference or not hypothesis:
        out: List[Tuple[Optional[Word], Optional[Word]]] = []
        out.extend((r, None) for r in reference)
        out.extend((None, h) for h in hypothesis)
        return out

    ref_texts = [w.text for w in reference]
    hyp_texts = [w.text for w in hypothesis]
    jiwer_out = jiwer.process_words(" ".join(ref_texts), " ".join(hyp_texts))

    pairs: List[Tuple[Optional[Word], Optional[Word]]] = []
    pending_ref: List[Word] = []
    pending_hyp: List[Word] = []

    def flush() -> None:
        nonlocal pending_ref, pending_hyp
        if pending_ref or pending_hyp:
            pairs.extend(_emit_block(pending_ref, pending_hyp))
            pending_ref = []
            pending_hyp = []

    for chunk in jiwer_out.alignments[0]:
        r_span = reference[chunk.ref_start_idx:chunk.ref_end_idx]
        h_span = hypothesis[chunk.hyp_start_idx:chunk.hyp_end_idx]

        if chunk.type == "equal":
            flush()
            for r, h in zip(r_span, h_span):
                pairs.append((r, h))
        else:
            # substitute / insert / delete -> accumulate into the current
            # non-equal block; flush only when we hit an equal chunk.
            pending_ref.extend(r_span)
            pending_hyp.extend(h_span)

    flush()
    return pairs


# -----------------------------------------------------------------------------
# 3. Pooled metrics
# -----------------------------------------------------------------------------

TOLERANCE_BUCKETS_MS = (25, 50, 100, 200)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _scalar_stats(deltas: List[float]) -> dict:
    if not deltas:
        return {
            "mae_ms": None, "median_ms": None, "p90_ms": None,
            **{f"within_{t}ms_pct": None for t in TOLERANCE_BUCKETS_MS},
        }
    abs_deltas = [abs(d) for d in deltas]
    n = len(abs_deltas)
    out = {
        "mae_ms": round(sum(abs_deltas) / n, 1),
        "median_ms": round(_percentile(abs_deltas, 50), 1),
        "p90_ms": round(_percentile(abs_deltas, 90), 1),
    }
    for t in TOLERANCE_BUCKETS_MS:
        out[f"within_{t}ms_pct"] = round(
            sum(1 for d in abs_deltas if d <= t) / n * 100, 2
        )
    return out


def metrics_from_pairs(
    pairs: List[Tuple[Optional[Word], Optional[Word]]],
    ref_total: int,
) -> dict:
    """Compute per-file or pooled metrics from a list of (ref, hyp) pairs.

    `ref_total` is the count of original reference words (used as the
    denominator for match-rate). The numerator is the sum of ref-side
    `word_count` across matched pairs — so a single merged pair that
    represents M ref tokens correctly contributes M to coverage.

    `matched_pairs` is the literal count of comparable pairs (post-merge)
    and is what every timestamp metric (MAE / median / p90 / buckets)
    operates over. `ref_words_matched` is the count of *original ref
    tokens* covered by those pairs — typically equal to matched_pairs
    when no merges happen, larger by N-1 per M:1 merge.
    """
    matched = [(r, h) for r, h in pairs if r is not None and h is not None]
    ref_words_matched = sum(r.word_count for r, _ in matched)
    start_deltas = [h.start_ms - r.start_ms for r, h in matched]
    end_deltas = [h.end_ms - r.end_ms for r, h in matched]

    out: dict = {
        "matched_pairs": len(matched),
        "ref_words_matched": ref_words_matched,
        "ref_words": ref_total,
        "match_rate_pct": (
            round(ref_words_matched / ref_total * 100, 2) if ref_total else 0.0
        ),
    }
    for label, deltas in (("start", start_deltas), ("end", end_deltas)):
        s = _scalar_stats(deltas)
        for k, v in s.items():
            out[f"{label}_{k}"] = v
    return out


# -----------------------------------------------------------------------------
# 4. Driver: run providers across the corpus, pool, report
# -----------------------------------------------------------------------------

def _process_one_file(
    audio_path: Path,
    tg_path: Path,
    transcriber,
    model: str,
    timeout_s: float,
) -> Tuple[dict, list, int]:
    """Run a single (transcribe + align + metrics) for one file.

    Runs in a worker thread. The transcriber's HTTP client is configured
    with `timeout_s` to bound the API call. Returns (metrics, pairs, n_ref)
    or raises on failure.
    """
    reference = parse_textgrid(tg_path)
    hyp_raw = transcriber(audio_path, model, None, timeout_s=timeout_s)
    hypothesis = [
        Word(text=_normalize_word(w.text), start_ms=w.start_ms, end_ms=w.end_ms)
        for w in hyp_raw
    ]
    pairs = align_words(reference, hypothesis)
    metrics = metrics_from_pairs(pairs, ref_total=len(reference))
    metrics["audio"] = audio_path.name
    metrics["hyp_words"] = len(hypothesis)
    metrics["status"] = "ok"
    return metrics, pairs, len(reference)


def run_provider_on_corpus(
    provider: str,
    model: str,
    corpus_dir: Path,
    mfa_output: Path,
    workers: int = 8,
    per_file_timeout_s: float = 120.0,
) -> dict:
    """Run a single provider across the corpus (parallel) and return a result block.

    Files are processed in parallel via a thread pool. Each per-file transcribe
    call is bounded by `per_file_timeout_s` (enforced via the underlying SDK's
    HTTP client timeout). A future that doesn't return within that time is
    reported as a failure and skipped — we never wait indefinitely on a
    wedged provider call.
    """
    transcriber = get_transcriber(provider)

    audio_paths = sorted(corpus_dir.glob("*.wav"))
    print(f"[{provider}:{model}] {len(audio_paths)} audio files, "
          f"workers={workers}, per-file timeout={per_file_timeout_s}s")

    per_file: List[dict] = []
    all_pairs: List[Tuple[Optional[Word], Optional[Word]]] = []
    total_ref_words = 0
    files_scored = files_skipped_no_tg = files_failed = 0

    work: List[Tuple[Path, Path]] = []
    for audio_path in audio_paths:
        tg_path = mfa_output / f"{audio_path.stem}.TextGrid"
        if not tg_path.exists():
            files_skipped_no_tg += 1
            per_file.append({
                "audio": audio_path.name,
                "status": "skipped_no_textgrid",
            })
            continue
        work.append((audio_path, tg_path))

    if not work:
        pooled = metrics_from_pairs([], ref_total=0)
        return {
            "provider": provider, "model": model,
            "files_scored": 0,
            "files_skipped_no_textgrid": files_skipped_no_tg,
            "files_failed": 0,
            "pooled": pooled, "per_file": per_file,
        }

    completed = 0
    total = len(work)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {
            pool.submit(
                _process_one_file, ap, tg, transcriber, model, per_file_timeout_s,
            ): ap
            for ap, tg in work
        }
        for fut in as_completed(future_to_path):
            ap = future_to_path[fut]
            completed += 1
            try:
                metrics, pairs, n_ref = fut.result()
                per_file.append(metrics)
                all_pairs.extend(pairs)
                total_ref_words += n_ref
                files_scored += 1
                print(
                    f"  [{completed}/{total}] {ap.name}: "
                    f"matched={metrics['matched_pairs']}/{metrics['ref_words']} "
                    f"start_mae={metrics['start_mae_ms']}ms "
                    f"end_mae={metrics['end_mae_ms']}ms"
                )
            except Exception as e:
                files_failed += 1
                per_file.append({
                    "audio": ap.name,
                    "status": "failed",
                    "error": str(e),
                })
                print(
                    f"  [{completed}/{total}] {ap.name} FAILED: {e}",
                    file=sys.stderr,
                )

    pooled = metrics_from_pairs(all_pairs, ref_total=total_ref_words)

    return {
        "provider": provider,
        "model": model,
        "files_scored": files_scored,
        "files_skipped_no_textgrid": files_skipped_no_tg,
        "files_failed": files_failed,
        "pooled": pooled,
        "per_file": per_file,
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_PUNCT = ".,!?;:\"()[]{}"


def _slugify(s: str, max_len: int = 40) -> str:
    """Filesystem-safe lowercased slug. Used to derive output filenames."""
    import re
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_").lower()
    return slug[:max_len].rstrip("_")


def _dataset_stem(corpus_info: dict) -> str:
    """Derive a short, filesystem-safe stem from corpus metadata.

    Prefers the bare dataset name (after any `/`), appended with the config
    if one is set. Returns empty string if no metadata is available, in which
    case the eval falls back to plain `results-<timestamp>`.
    """
    if not corpus_info:
        return ""
    name = corpus_info.get("name") or ""
    if not name:
        return _slugify(corpus_info.get("display_name", ""))
    # Strip namespace prefix (e.g. "MLCommons/peoples_speech" -> "peoples_speech")
    short = name.rsplit("/", 1)[-1]
    parts = [short]
    if corpus_info.get("config"):
        parts.append(corpus_info["config"])
    return _slugify("-".join(parts))


def _normalize_word(s: str) -> str:
    """Lowercase + strip outer punctuation. Apostrophes are preserved.

    Applied symmetrically to ref and hyp, so any normalization choice is
    fair across providers. The thing that matters is consistency, not the
    exact rule.
    """
    return s.strip().strip(_PUNCT).lower()


def parse_providers_arg(arg: str) -> List[Tuple[str, str]]:
    """Parse `--providers` into a list of (provider, model) tuples.

    Forms:
      assemblyai:universal-3-pro
      deepgram:nova-3,openai:whisper-1     (comma-separated multi-provider)
    """
    out = []
    for entry in arg.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"--providers entry {entry!r} must be 'provider:model'"
            )
        provider, model = entry.split(":", 1)
        provider, model = provider.strip(), model.strip()
        if provider not in REGISTRY:
            raise ValueError(
                f"unknown provider {provider!r}; available: {sorted(REGISTRY)}"
            )
        out.append((provider, model))
    if not out:
        raise ValueError("--providers may not be empty")
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir", required=True, type=Path,
        help="Directory with .wav files + matching .txt transcripts.",
    )
    parser.add_argument(
        "--mfa-output", required=True, type=Path,
        help="Directory where MFA TextGrids live (created with --run-mfa).",
    )
    parser.add_argument(
        "--run-mfa", action="store_true",
        help="Run MFA forced alignment before scoring. Skip on re-runs.",
    )
    parser.add_argument(
        "--providers",
        default="assemblyai:universal-3-pro",
        help=(
            "Comma-separated 'provider:model' pairs. Examples: "
            "'assemblyai:universal-3-pro', "
            "'assemblyai:universal-3-pro,deepgram:nova-3,openai:whisper-1'. "
            f"Available providers: {sorted(REGISTRY)}."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON path. Defaults to results-<UTC timestamp>.json so "
             "repeated runs don't overwrite each other. An HTML report will "
             "be written next to the JSON.",
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML report generation.",
    )
    parser.add_argument(
        "--env", type=Path, default=Path(".env"),
        help="Path to .env file with provider API keys.",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        help="Human-readable dataset/corpus name for the report header. "
             "Overrides anything in <corpus>/corpus.json. "
             "Example: 'LibriSpeech dev-clean (50 utts)'.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent worker threads per provider (default 8). Each worker "
             "transcribes one file at a time. Higher values speed up large runs "
             "but may hit provider rate limits.",
    )
    parser.add_argument(
        "--per-file-timeout", type=float, default=120.0,
        help="Hard timeout (seconds) on each per-file API call. Default 120 s. "
             "Bounded by the provider SDK's HTTP client timeout — past this, the "
             "file is marked failed and the worker moves on.",
    )
    args = parser.parse_args()

    # Line-buffered stdout so per-file progress prints show up in real time
    # even when the eval is run with stdout redirected to a file (default
    # Python behaviour is block-buffered to a pipe, which masks progress).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    if args.env.exists():
        load_dotenv(args.env)

    # Read corpus.json sidecar early so the dataset stem can be used in the
    # default output filename (e.g. results-peoples_speech-<ts>.json).
    corpus_info: dict = {}
    sidecar_path = args.corpus_dir / "corpus.json"
    if sidecar_path.exists():
        try:
            corpus_info = json.loads(sidecar_path.read_text())
        except Exception as e:
            print(f"warning: could not parse {sidecar_path}: {e}", file=sys.stderr)
    if args.dataset_name:
        corpus_info = {**corpus_info, "display_name": args.dataset_name}
    elif corpus_info.get("name"):
        parts = [corpus_info["name"]]
        if corpus_info.get("config"):
            parts.append(f"({corpus_info['config']}")
            if corpus_info.get("split"):
                parts[-1] += f"/{corpus_info['split']}"
            parts[-1] += ")"
        elif corpus_info.get("split"):
            parts.append(f"({corpus_info['split']})")
        corpus_info["display_name"] = " ".join(parts)

    # Default output filename: results-[<dataset>-]<UTC timestamp>.json so
    # back-to-back runs across different corpora are self-describing and
    # don't silently overwrite each other.
    if args.out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = _dataset_stem(corpus_info)
        args.out = Path(f"results-{stem}-{ts}.json" if stem else f"results-{ts}.json")

    providers = parse_providers_arg(args.providers)

    if args.run_mfa:
        print(f"Running MFA on {args.corpus_dir} -> {args.mfa_output}")
        run_mfa_alignment(args.corpus_dir, args.mfa_output)

    runs = []
    for provider, model in providers:
        print(f"\n=== Running {provider}:{model} ===")
        runs.append(run_provider_on_corpus(
            provider, model, args.corpus_dir, args.mfa_output,
            workers=args.workers,
            per_file_timeout_s=args.per_file_timeout,
        ))

    # Build the full results object.
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(args.corpus_dir),
        "corpus": corpus_info,
        "mfa_output": str(args.mfa_output),
        "tolerance_buckets_ms": list(TOLERANCE_BUCKETS_MS),
        "runs": runs,
    }

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {args.out}")

    # Side-by-side summary
    print("\n=== Pooled summary ===")
    print(f"{'provider:model':<40} {'files':<6} {'words':<8} "
          f"{'start MAE':<12} {'start p90':<12} "
          f"{'within 100':<12}")
    for run in runs:
        p = run["pooled"]
        label = f"{run['provider']}:{run['model']}"
        print(
            f"{label:<40} "
            f"{run['files_scored']:<6} "
            f"{p['matched_pairs']:<8} "
            f"{str(p['start_mae_ms']) + ' ms':<12} "
            f"{str(p['start_p90_ms']) + ' ms':<12} "
            f"{str(p['start_within_100ms_pct']) + '%':<12}"
        )

    if not args.no_html:
        from report import render_report  # local import to keep startup light
        html_path = args.out.with_suffix(".html")
        render_report(results, html_path)
        print(f"Wrote HTML report to {html_path}")

    # Maintain `latest.json` / `latest.html` as symlinks to the most-recent
    # run, so methodology.html (and other static docs) can point at a stable
    # filename and still surface the newest results.
    _update_latest_symlinks(args.out)

    return 0


def _update_latest_symlinks(json_path: Path) -> None:
    """Point latest.json / latest.html at the most-recent run."""
    parent = json_path.parent
    for ext in (".json", ".html"):
        target = json_path.with_suffix(ext)
        if not target.exists():
            continue
        link = parent / f"latest{ext}"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.name)
        except OSError as e:
            # Filesystems without symlink support (rare on macOS/Linux) — skip.
            print(f"warning: could not update {link}: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
