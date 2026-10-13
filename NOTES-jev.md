# Jev (TypeSafe AI): research notes, 2026-10-13

Compiled from TypeSafe's docs (docs.typesafe.ai, read as raw markdown), the launch post, and two independent
evaluations. Vendor claims are marked as such.

## What it is
A **decision model**, not a chatbot. You send a `state` (text or JSON) and typed questions. It returns
probabilities and never generates text. It suits classification, routing, scoring, guardrails, and picking among
candidates during extraction. It is not a replacement for the model behind Claude Code or Cursor.

- Company: TypeSafe AI, San Francisco, founded 2024 by Diogo Almeida (CEO, ex-OpenAI, RLHF co-author), Sasha
  Sheng and Erik Gafni. $40M seed with launch on 2026-09-15; ~$870M at a $7.5B valuation around 2026-10-09
  (a16z lead). Fewer than 25 employees, per press.
- Named after the economist W. S. Jevons (the Jevons paradox).

## How it works (as disclosed)
- A transformer that answers all questions in one parallel pass (no token-by-token decoding), at ~70 to 500 ms.
- Trained with "RLCD" (Reinforcement Learning for Calibrated Decisions), which targets honest probabilities.
  No paper; architecture, size, base model and data are undisclosed (reportedly synthetic data).
- Output is limited to the options you declare, so malformed output can't happen. A wrong option can still be picked.
- Same weights for every customer; no fine-tuning. You adapt it through state, instructions and criteria.
- Confidence is computed by formula from the probabilities, not by the model:
  Choice `(n * p_max - 1) / (n - 1)`; Score `1 - (prob-weighted distance from peak) / (same for uniform)`.

## API
- `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`
- Body: `{state, model: "jev-latest", questions: {id: {type, instructions, criteria}}}`
- Question types:
  - `noul`: returns 0..1 probability of yes.
  - `choice`: up to 255 options; returns choice, probabilities and confidence.
  - `score`: 2 to 10 ordered levels; returns a probability-weighted score, legend and confidence.
- Instructions and criteria can be JSON; point at state paths with backticks: `` `support.tickets[0].message` ``.
- `jev-1.13.0`:
  - Price: $0.042 per million input tokens; output tokens are free.
  - Context: 64k tokens total; 32k for the state plus the longest question.
  - Rate limits: 100K tokens/s and 80 requests/s (marked as changing).
  - Input: text only. English is best; CJK and other languages are weaker.
- Errors: 401, 422, 429 and 529. The SDKs retry with backoff.
- SDKs: `pip install typesafe-sdk` (Python 3.10+, MIT) and `npm i @typesafe-ai/sdk` (Node 20+, MIT).
  Environment: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`. Default timeout 10 s.
- Claude Code skill: `claude plugin marketplace add typesafe-ai/skills && claude plugin install typesafe@typesafe-ai`.
- Pin `jev-1.13.0` once confidence thresholds are tuned; the `jev-latest` alias moves when new versions ship.

## Known weaknesses (vendor's own "jaggedness" page)
- Reads instructions literally.
- Can't count or do maths; can't compare dates.
- Weak on multi-hop questions and double negatives.
- Accuracy drops with long irrelevant state ("context rot").
- Prompt injection inside the state can move answers.
- Biased toward the first Choice option.
- Can't generate text.

## Independent results
- **Vals AI** (12 systems):
  - Claim verification on 10-Ks: 0.975, tied with GPT-6 Astra at about 1/498th of the cost; best ECE (0.011).
  - LegalBench slice: last of 12, worst ECE (0.108).
  - Latency: ~0.1 s, flat up to 64 questions per call.
- **dev.to meta-review:**
  - 72.5% vs Fable 5.1 at 84.0% on a 200-item mixed set.
  - Behind the best LLM on 14 of 15 tasks in one pre-registered study.
  - One fitted temperature on 50 to 300 labels fixes most of the calibration error.

## Self-hosting
Jev weights aren't published, so you can't self-host it. Open clones:
- **Kev** (Apache-2.0): Qwen3.5 0.8B/4B/9B/27B with adapters, serving the same `/v1/systemone` API. Point
  `TYPESAFE_BASE_URL` at it. 4B needs a ~32 GB Mac; 0.8B runs on any Apple Silicon.
- **Laya** (ModernBERT), **SemIf/OpenJev** (logit readout from a frozen Qwen), **NanoJev**, and **jevlike**
  (a training kit).
