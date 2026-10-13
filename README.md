# context-pager

Does relevance-scored eviction keep an assistant's answers intact on a fraction of the context,
and which scorer should do it? Findings: [REPORT.md](REPORT.md). Raw numbers: `results.md`, `results.json`. Solution ideas: [IDEAS.md](IDEAS.md).
Jev research: [NOTES-jev.md](NOTES-jev.md). Setup gotchas: [NOTES-practical.md](NOTES-practical.md).

## Layout

| path | what |
|---|---|
| `study/transcripts.py` | Claude Code `.jsonl` sessions to context items; stub rendering; deterministic cleaning |
| `study/selectors.py` | scorers (recency, random, BM25, bge-small embeddings, Kev) and the shared budget rule |
| `study/llm.py` | headless `claude -p` with no tools, hooks or CLAUDE.md |
| `study/run.py` | the experiment; every stage cached in `runs/<session-id>/`, so it resumes after interruption |
| `study/analyze.py` | statistics into `results.md` / `results.json`; makes no model calls |
| `tests/test_core.py` | budget filling, BM25, AUROC, cleaning, stub rendering |
| `pilot/` | the first 24-question pilot harness (superseded, kept for the record; its outputs are withheld, see Data) |
| `kev/` | not committed: `git clone https://github.com/jaredpalmer/kev.git` (Apache-2.0), used only as a local scoring server |

## Data

The study's questions, answers, grades, scores and summaries are **not** published. They were derived from my own
private Claude Code sessions and quote client and personal project content. What is public: all code, the
methodology, the run configuration (`runs/config.json`) and the aggregate results (`results.md`, `results.json`).
Anyone can regenerate the raw data from their own sessions with `uv run python -m study.run`, which reads
`~/.claude/projects` by default. `study.analyze` needs those regenerated `runs/` directories.
A fully public version built on open agent trajectories is planned.

## Reproduce

```bash
uv sync
git clone https://github.com/jaredpalmer/kev.git
uv run python tests/test_core.py                       # sanity
(cd kev && uv sync --extra serve && \
  uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009) &
uv run python -u -m study.run                           # hours; resumable
uv run python -m study.analyze
```

Requirements: the `claude` CLI logged in (it answers, judges and writes the questions), about 3 GB of disk for
model weights. Kev-0.8B runs on any Apple Silicon Mac; Kev-4B needs about 32 GB of RAM.
