# Practical notes from running this study

- **Headless Claude as an eval model on a subscription:** `--bare` only works with `ANTHROPIC_API_KEY`. Use
  `claude -p --setting-sources "" --strict-mcp-config --tools "" --no-session-persistence --system-prompt ...
  --output-format json` instead. Overhead is about 640 tokens, and no hooks, plugins or CLAUDE.md are loaded.
  `usage` in the JSON gives input, cache-creation and cache-read token counts.
- **Usage limits** come back as an error result containing "limit". The runner sleeps 15 minutes and retries.
  Caching every stage to disk meant a mid-run limit cost time and nothing else.
- **Kev on a 16 GB M4:** only Kev-0.8B fits. MLX serving, about 2 GB of downloads, first call ~3 s. It took
  385 ms per request with ~10 Noul questions and 249 ms with one. Install from source; `pip install kev` installs
  an unrelated package.
- **Kev-0.8B is a weak relevance judge.** In the smoke test, a pricing doc scored 0.48 for a pricing question.
  As a blind "needed later?" scorer its AUROC was 0.41.
- **Docs sites:** Mintlify docs serve raw markdown at `<page>.md` and an index at `/llms.txt`. One "confidence"
  page was ~60% inline React by size; fetching the `.md` avoids it.
- **Transcript anatomy** (38 large top-level sessions): tool output 71.5%, images 15.6%, thinking 5.8%, tool
  calls 4.6%, user text 1.4%, assistant text 1.2%. Subagent transcripts live in subdirectories, so
  `*/*.jsonl` counts only top-level sessions.
- **Auto mode** blocked a harness change that would have sent transcript chunks to an external API (TypeSafe),
  flagging it as data exfiltration. Any external scorer needs explicit user approval.
- **A "25% budget" comes out at about 36 to 38% of real tokens**, because pinned items and stubs count too. Report
  measured tokens, not the nominal budget.
