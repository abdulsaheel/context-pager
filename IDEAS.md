# Solution space beyond "score chunks with Jev/Kev"

Grounding fact from the 38 large top-level sessions on this machine: tool output is 71.5% of transcript
content, images 15.6%, assistant thinking 5.8%, tool calls 4.6%. What the user and the assistant actually
wrote to each other is 2.6%. Most of the context goes to material read once for a side task.

Ordered by what I'd try first. "Tested" means the overnight study measures it; see REPORT.md.

## 1. Collapse finished tool chains (tested: `text+recall`)
Once the assistant has answered a request, the tool calls and outputs behind that answer have usually done
their job. Keep the conversation text and turn every tool call and output into a one-line stub the model can
recall. This needs no model and no tuning, and it's deterministic, so it's easy to trust.

## 2. Paging instead of deleting (tested: every `+recall` condition)
Evicted items stay addressable (`#id`) and the model can ask for up to 5 of them back. This fixes the core
weakness of any pruning: relevance depends on questions that haven't been asked yet. The cost is an extra
model call when recall happens, and the outcome depends on stub quality.

## 3. Filter at ingestion, using the tool call's own intent (not yet tested; the next experiment)
The blind pager's problem is not knowing the next question. But when a tool result arrives, the reason it
was requested is already known: the assistant's preceding message and the tool call. Score the result's chunks
against that intent and keep only the matching chunks inline; store the rest out of context with a handle.
This is the cleanest fit for a decision model like Kev or Jev, because the query exists at the moment
pruning happens.

## 4. Prevent instead of cure: fork side quests
Run research and exploration in a child context (a subagent or forked session) and return only its
conclusion. Claude Code already does this with subagents. The open part is deciding **automatically** when a
user request is a side quest. That's a one-question Noul ("is this request a detour from `current_task`?"),
which is a good fit for a decision model.

## 5. Cascade scorers
Use BM25 or embeddings (microseconds to milliseconds, ~free) to shortlist the top 10-20% of items, and run the
decision model or a small LLM only on that shortlist. This is the classic retrieve-then-rerank setup from
information retrieval. Kev-0.8B costs ~0.8 s per request on an M4, so scoring every chunk on every turn isn't
viable locally, while rescoring a shortlist is.

## 6. Learn eviction from the transcripts themselves (most novel; needs a week, not a night)
Label every item in the ~1,400 local transcripts by whether it was **used later**: a later assistant message
quotes it, repeats its file path or error string, or the model re-reads it. That gives a free,
self-supervised "will this be needed again?" dataset with no human labelling. Train a small classifier on it
(Kev's fine-tuning loop or `jevlike` would work). This targets the actual distribution of Claude Code
sessions rather than generic relevance.

## 7. Stubs that carry the right hook
Recall only works if the stub tells the model what's behind it. The study's stubs are the first line of the
item. Better options: for a tool output, the command that produced it; for prose, the single line a scorer
rates as most informative. This is cheap and probably worth several points of recall accuracy.

## 8. Cache-aware eviction epochs
Rewriting history every turn breaks prompt caching, and cached prefixes are where long sessions get cheap.
Evict in batches at stable boundaries (for example every N turns, or once the uncached suffix passes a
threshold), so the cached prefix survives between evictions.

## 9. A pinned facts ledger
This is an extractive alternative to compaction: keep a small, append-only list of decisions, constraints,
paths and numbers, written as the session goes, always pinned. Compaction fails on exactly these details (see
the `compact` row in REPORT.md).

## Not worth it here
- **Deterministic cleaning** (strip HTML/JS, deduplicate repeated lines and outputs): 0-8% smaller on these
  transcripts, because Claude Code's tools already return mostly text and WebFetch pre-summarises pages. It
  only matters for raw HTTP or HTML tools.
- **Summarising compaction as the main strategy**: cheap but lossy on specifics; see the results.
