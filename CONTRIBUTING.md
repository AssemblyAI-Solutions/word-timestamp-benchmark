# Contributing to Word-Level Timestamp Benchmark

Thanks for taking a look — this is a small, focused harness and we welcome contributions that keep it that way.

## What's a good contribution

- **Bug fixes** in the alignment, metric, or report code (with a brief explanation of how to reproduce).
- **New provider integrations** in `transcribers/` — follow the pattern in `transcribers/base.py` (one file per provider, register via the decorator, return `List[Word]` in milliseconds).
- **Corpus loaders** for additional public datasets — extend `populate_corpus_hf.py` or add a sibling `populate_corpus_<dataset>.sh`. Make sure the loader writes a `corpus.json` sidecar so the report header is self-describing.
- **Methodology improvements** with clear before/after rationale (the canonical methodology lives in `methodology.html` — keep it in sync).

## What's not a good fit

- Changes that hide methodology details to make any provider look better. The whole value of this harness is being honest and reproducible.
- Heavyweight dependencies (e.g. torch, full HF datasets stack) added to the critical path. The transcribers + eval need to stay runnable in a vanilla MFA env.
- Per-provider tuning that makes the comparison unfair (different `format_text` defaults, different post-processing per provider, etc).

## Reporting issues

Please include:
- The corpus you ran on (dataset, size, `corpus.json` contents if you have one).
- The exact CLI invocation.
- The relevant snippet of `results-*.json` if it's a metrics issue, or the failing log line if it's a transcribe issue.

## Pull requests

- Keep PRs focused on one change at a time.
- Update `README.md` and `methodology.html` if you change something users see. If you edit `methodology.html`, regenerate the PDF with `tools/render-methodology-pdf.sh` so the GitHub-visible version stays in sync.
- Run the eval end-to-end on at least one small corpus (e.g. 50 LibriSpeech utts via `populate_corpus.sh`) before opening the PR.

## Code style

- Match the surrounding style — no major reformatting in a feature PR.
- Comments should explain *why*, not *what*. Don't write comments that just narrate the code.
- Don't add tests for trivial things, but do add a test or worked example when changing alignment or metrics math — that code is easy to break subtly.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](./LICENSE).
