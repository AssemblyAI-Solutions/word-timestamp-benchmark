"""HTML report renderer.

Builds a self-contained HTML page from a results dict using a Jinja2
template. Chart.js is loaded from CDN, so the page is one file and
opens directly in a browser.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape


_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Consistent provider color palette (used in cards, tables, and charts).
_PROVIDER_PALETTE = ["#4f46e5", "#0891b2", "#ca8a04", "#db2777", "#16a34a", "#9333ea"]


def _provider_colors(runs: List[Dict[str, Any]]) -> Dict[str, str]:
    return {
        f"{r['provider']}:{r['model']}": _PROVIDER_PALETTE[i % len(_PROVIDER_PALETTE)]
        for i, r in enumerate(runs)
    }


def _short_provider(run: Dict[str, Any]) -> str:
    """A short display label, e.g. 'AssemblyAI universal-3-pro'."""
    display = {
        "assemblyai": "AssemblyAI",
        "deepgram": "Deepgram",
        "openai": "OpenAI",
    }
    return f"{display.get(run['provider'], run['provider'])} {run['model']}"


def _bests(runs: List[Dict[str, Any]], buckets: List[int]) -> Dict[str, str]:
    """For each metric, return the label of the run that wins.

    Lower is better for MAE/median/p90. Higher is better for tolerance %.
    Match rate is informational, not a "win" criterion.
    """
    out: Dict[str, str] = {}
    if not runs:
        return out
    labels = [f"{r['provider']}:{r['model']}" for r in runs]

    def winner(key: str, *, lower_is_better: bool) -> str:
        vals = []
        for lbl, r in zip(labels, runs):
            v = r["pooled"].get(key)
            if v is not None:
                vals.append((v, lbl))
        if not vals:
            return ""
        return min(vals)[1] if lower_is_better else max(vals)[1]

    for side in ("start", "end"):
        out[f"{side}_mae_ms"] = winner(f"{side}_mae_ms", lower_is_better=True)
        out[f"{side}_median_ms"] = winner(f"{side}_median_ms", lower_is_better=True)
        out[f"{side}_p90_ms"] = winner(f"{side}_p90_ms", lower_is_better=True)
        for t in buckets:
            out[f"{side}_within_{t}ms_pct"] = winner(
                f"{side}_within_{t}ms_pct", lower_is_better=False
            )
    return out


def _corpus_summary(corpus_dir: str) -> Dict[str, Any]:
    """Compute total audio duration and file count for the corpus."""
    p = Path(corpus_dir)
    if not p.exists():
        return {"files": 0, "total_duration_s": 0, "avg_duration_s": 0}
    wavs = sorted(p.glob("*.wav"))
    total = 0.0
    for w in wavs:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(w)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            total += float(out or 0)
        except Exception:
            pass
    n = len(wavs)
    return {
        "files": n,
        "total_duration_s": round(total, 1),
        "avg_duration_s": round(total / n, 1) if n else 0,
    }


def _build_headline(runs: List[Dict[str, Any]], buckets: List[int]) -> str:
    """A one-line auto-generated takeaway for the top of the report."""
    if not runs:
        return ""
    if len(runs) == 1:
        r = runs[0]
        p = r["pooled"]
        return (
            f"{_short_provider(r)}: start MAE {p.get('start_mae_ms')} ms, "
            f"{p.get('start_within_100ms_pct')}% of word starts within 100 ms "
            f"(n={p.get('matched_pairs')} aligned words across "
            f"{r['files_scored']} files)."
        )
    # Multi-provider: who wins start, who wins end, by how much.
    bests = _bests(runs, buckets)
    start_winner = bests.get("start_within_100ms_pct", "")
    end_winner = bests.get("end_within_100ms_pct", "")
    by_label = {f"{r['provider']}:{r['model']}": r for r in runs}
    parts = []
    if start_winner:
        wr = by_label[start_winner]
        wp = wr["pooled"]["start_within_100ms_pct"]
        parts.append(
            f"<strong>{_short_provider(wr)}</strong> leads on start times "
            f"({wp}% within 100 ms)"
        )
    if end_winner and end_winner != start_winner:
        wr = by_label[end_winner]
        wp = wr["pooled"]["end_within_100ms_pct"]
        parts.append(
            f"<strong>{_short_provider(wr)}</strong> leads on end times "
            f"({wp}% within 100 ms)"
        )
    elif end_winner:
        wr = by_label[end_winner]
        wp = wr["pooled"]["end_within_100ms_pct"]
        parts.append(f"and on end times ({wp}% within 100 ms)")
    return "; ".join(parts) + "."


def _truncate_name(name: str, max_len: int = 40) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len - 1] + "…"


def _build_chart_data(results: Dict[str, Any]) -> list:
    out = []
    buckets = results["tolerance_buckets_ms"]
    for run in results["runs"]:
        p = run["pooled"]
        row = {"label": f"{run['provider']}:{run['model']}"}
        for t in buckets:
            for side in ("start", "end"):
                key = f"{side}_within_{t}ms_pct"
                row[key] = p.get(key) or 0
        out.append(row)
    return out


def render_report(results: Dict[str, Any], html_path: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["short_provider"] = _short_provider
    env.filters["truncate_name"] = _truncate_name

    runs = results["runs"]
    buckets = results["tolerance_buckets_ms"]

    template = env.get_template("report.html.j2")
    html = template.render(
        generated_at=results["generated_at"],
        corpus_dir=results["corpus_dir"],
        corpus=results.get("corpus", {}),
        mfa_output=results["mfa_output"],
        tolerance_buckets_ms=buckets,
        runs=runs,
        provider_colors=_provider_colors(runs),
        bests=_bests(runs, buckets),
        corpus_summary=_corpus_summary(results["corpus_dir"]),
        headline=_build_headline(runs, buckets),
        runs_chart_data=_build_chart_data(results),
    )
    html_path.write_text(html)
