"""End-to-end study: which context-management strategy keeps answer quality at the lowest token cost?

Usage:  uv run python -m study.run            (resumable; every stage is cached under runs/<id>/)
Needs:  Kev server on :8009 (see README), `claude` CLI logged in.
"""
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from study import selectors as S
from study.llm import claude, parse_json
from study.transcripts import load_items, pick_transcripts, render, segment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
CONFIG = {
    "n_transcripts": 8,
    "questions_per_transcript": 10,
    "segment_chars": 240_000,   # ~60k tokens of history per transcript
    "budget_frac": 0.25,        # every selector keeps at most this share of the history
    "tail_items": 4,
    "max_recall": 5,
    "answer_model": "sonnet",
    "judge_models": ["opus", "sonnet"],
    "seed": 0,
    # never evaluate on the session that designed this study
    "exclude": ["e26ae883-b3dc-4521-9710-fc90c4e3e4fa"],
}

QGEN_SYS = "You write evaluation questions for long-conversation memory tests. Output only JSON."
QGEN = """Each ITEM below comes from a long coding-assistant session. For each item, write one question the user
might ask much later in the same session, answerable only from a specific fact in that item (a number, name,
path, error, decision, result). Rules:
- phrase it the way a user returning to an old topic would ("what was the ... we found earlier?")
- do NOT copy long or unusual strings from the item into the question; paraphrase
- not answerable from general knowledge; one fact; short gold answer taken from the item
- if an item has no such fact, use null
Output a JSON list in item order: [{{"id": N, "q": "...", "gold": "..."}} or {{"id": N, "q": null}}]

{items}"""

ANSWER_SYS = ("Answer the user's question using only the conversation history provided. Some history items may be "
              "EVICTED and shown only as one-line stubs. Be brief and specific. If the history does not contain the "
              "answer, reply exactly NOT FOUND.")
RECALL_SYS = ("Answer the user's question using only the conversation history provided. Items marked EVICTED are "
              "stubs; their full text can be paged back in. If you need evicted items, reply with ONLY a line "
              "`RECALL: #id, #id` listing up to {k} ids. Otherwise answer briefly and specifically, or reply exactly "
              "NOT FOUND if the answer is not in the history.")
SUMMARY_SYS = "You compact long conversations so work can continue in a fresh context window."
SUMMARY = ("Summarize this conversation so an assistant can continue it with no other context. Keep decisions, facts, "
           "numbers, file paths, errors and open tasks. At most about 2,000 words.\n\n{t}")
JUDGE_SYS = "You grade answers against gold answers. Output only JSON."
JUDGE = """Grade each candidate answer against the gold answer for its question.
2 = correct (all key facts in the gold answer, nothing contradicting it)
1 = partially correct (some key facts, or correct but hedged between alternatives)
0 = wrong, contradicting the gold, or NOT FOUND
Output a JSON list: [{{"k": key, "grade": 0|1|2}}, ...] covering every key.

{rows}"""

KEV_QUESTION_ASK = "Does `chunk` contain information that helps answer `query`?"
KEV_TASK_ASK = "Is `chunk` likely to be needed to continue the work described in `query`?"

_lock = threading.Lock()


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def save_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def make_questions(seg, cfg, d):
    path = os.path.join(d, "questions.json")
    if os.path.exists(path):
        return load_json(path)
    rng = random.Random(cfg["seed"])
    pinned = S.pinned_ids(seg, cfg["tail_items"])
    cutoff = int(len(seg) * 0.9)  # evidence never sits in the final 10% of history
    pool = [i for i in seg[:cutoff] if i["id"] not in pinned and i["kind"] != "tool_use" and len(i["text"]) >= 300]
    tools = [i for i in pool if i["kind"] == "tool_result"]
    texts = [i for i in pool if i["kind"] == "text"]
    sample = rng.sample(tools, min(9, len(tools))) + rng.sample(texts, min(5, len(texts)))
    sample.sort(key=lambda i: i["id"])

    def context_for(item):  # the user message that led to this item, for disambiguation
        prev = [i for i in seg[:item["id"]] if i["role"] == "user" and i["kind"] == "text"]
        return prev[-1]["text"][:300] if prev else ""

    block = "\n\n".join(f'ITEM id={i["id"]} ({i["role"]}/{i["kind"]})\nuser had asked: {context_for(i)}\n'
                        f'---\n{i["text"][:4000]}' for i in sample)
    out = parse_json(claude(QGEN.format(items=block), QGEN_SYS)[0])
    qs = [{"q": x["q"], "gold": x["gold"], "evidence": [x["id"]]} for x in out if x.get("q")]
    rng.shuffle(qs)
    qs = qs[:cfg["questions_per_transcript"]]
    save_json(path, qs)
    return qs


def task_query(seg):
    """What a blind pager knows at pruning time: the user's most recent requests."""
    users = [i["text"] for i in seg if i["role"] == "user" and i["kind"] == "text"]
    return "\n".join(users[-3:])[-1500:]


def compute_scores(seg, qs, d, embedder):
    path = os.path.join(d, "scores.json")
    scores = load_json(path, {})
    queries = {f"q{n}": q["q"] for n, q in enumerate(qs)}
    task = task_query(seg)
    if "kev" not in scores:
        kq, secs_q, n_req = S.kev_scores(seg, queries, KEV_QUESTION_ASK)
        kt, secs_t, _ = S.kev_scores(seg, {"task": task}, KEV_TASK_ASK)
        scores["kev"] = {**kq, **kt}
        scores["kev_timing"] = {"seconds_questions": secs_q, "seconds_task": secs_t, "requests": n_req}
        save_json(path, scores)
    if "bm25" not in scores:
        scores["recency"] = S.recency(seg)
        scores["random"] = S.random_scores(seg)
        scores["bm25"] = {k: S.bm25(seg, q) for k, q in {**queries, "task": task}.items()}
        scores["embed"] = {k: embedder.scores(seg, q) for k, q in {**queries, "task": task}.items()}
        save_json(path, scores)
    # json turns int keys into str; normalise back
    for name in ("recency", "random"):
        scores[name] = {int(k): v for k, v in scores[name].items()}
    for name in ("kev", "bm25", "embed"):
        scores[name] = {qk: {int(k): v for k, v in m.items()} for qk, m in scores[name].items()}
    return scores


def build_conditions(seg, qs, scores, summary, cfg):
    """{(n, condition): (context, recall_allowed, kept_ids)}"""
    budget = cfg["budget_frac"] * sum(len(i["text"]) for i in seg)
    pinned = S.pinned_ids(seg, cfg["tail_items"])
    full = render(seg)
    fill = lambda sc: S.fill_budget(seg, sc, budget, pinned)
    blind = {  # question-independent: computed once per transcript
        "recency+recall": fill(scores["recency"]),
        "kev-task+recall": fill(scores["kev"]["task"]),
        "stubs+recall": set(pinned),
        # structural, no model: keep the conversation itself, evict every tool call and tool output
        "text+recall": set(pinned) | {i["id"] for i in seg if i["kind"] == "text"},
    }
    conds = {}
    for n in range(len(qs)):
        qk = f"q{n}"
        conds[(n, "full")] = (full, False, None)
        conds[(n, "compact")] = (summary, False, None)
        for name, sc in (("recency", scores["recency"]), ("random", scores["random"]),
                         ("bm25-q", scores["bm25"][qk]), ("embed-q", scores["embed"][qk]),
                         ("kev-q", scores["kev"][qk])):
            keep = fill(sc)
            conds[(n, name)] = (render(seg, keep), False, keep)
        for name, keep in blind.items():
            conds[(n, name)] = (render(seg, keep), True, keep)
    return conds


def answer(seg, question, context, recall, keep, cfg):
    prompt = f"CONVERSATION HISTORY:\n{context}\n\nQUESTION: {question}"
    if not recall:
        text, tok = claude(prompt, ANSWER_SYS, cfg["answer_model"])
        return {"answer": text, "tokens": tok, "recalled": []}
    text, tok = claude(prompt, RECALL_SYS.format(k=cfg["max_recall"]), cfg["answer_model"])
    if not text.strip().startswith("RECALL"):
        return {"answer": text, "tokens": tok, "recalled": []}
    valid = {i["id"] for i in seg}
    ids = [int(x) for x in re.findall(r"#(\d+)", text) if int(x) in valid][:cfg["max_recall"]]
    context2 = render(seg, set(keep) | set(ids))
    text2, tok2 = claude(f"CONVERSATION HISTORY:\n{context2}\n\nQUESTION: {question}", ANSWER_SYS, cfg["answer_model"])
    return {"answer": text2, "tokens": tok + tok2, "recalled": ids}


def run_transcript(path, cfg, embedder):
    tag = os.path.basename(path)[:8]
    d = os.path.join(RUNS, tag)
    os.makedirs(d, exist_ok=True)
    seg = segment(load_items(path), cfg["segment_chars"])
    save_json(os.path.join(d, "meta.json"), {"path": path, "items": len(seg),
                                              "chars": sum(len(i["text"]) for i in seg)})
    print(f"[{tag}] {len(seg)} items", flush=True)

    qs = make_questions(seg, cfg, d)
    sp = os.path.join(d, "summary.txt")
    if not os.path.exists(sp):
        open(sp, "w").write(claude(SUMMARY.format(t=render(seg)), SUMMARY_SYS, cfg["answer_model"])[0])
    summary = open(sp).read()
    scores = compute_scores(seg, qs, d, embedder)
    print(f"[{tag}] {len(qs)} questions, scores ready", flush=True)

    conds = build_conditions(seg, qs, scores, summary, cfg)
    ap = os.path.join(d, "answers.json")
    answers = load_json(ap, [])
    have = {(a["n"], a["cond"]) for a in answers}

    def work(key):
        n, cond = key
        context, recall, keep = conds[key]
        r = answer(seg, qs[n]["q"], context, recall, keep, cfg)
        r.update(n=n, cond=cond, kept=sorted(keep) if keep is not None else None)
        with _lock:
            answers.append(r)
            save_json(ap, answers)

    todo = [k for k in conds if k not in have]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, todo))
    print(f"[{tag}] answers done ({len(answers)})", flush=True)

    for judge in cfg["judge_models"]:
        field = f"grade_{judge}"
        rows = [a for a in answers if field not in a]
        if not rows:
            continue
        random.Random(judge).shuffle(rows)  # blind: the judge never sees the condition
        text = "\n\n".join(f'key: {k}\nquestion: {qs[r["n"]]["q"]}\ngold: {qs[r["n"]]["gold"]}\n'
                           f'candidate: {r["answer"].strip()[:1500]}' for k, r in enumerate(rows))
        grades = {int(x["k"]): int(x["grade"]) for x in parse_json(claude(JUDGE.format(rows=text), JUDGE_SYS, judge)[0])}
        for k, r in enumerate(rows):
            r[field] = grades.get(k)
        save_json(ap, answers)
    print(f"[{tag}] judged", flush=True)


def main():
    cfg = CONFIG
    os.makedirs(RUNS, exist_ok=True)
    paths = load_json(os.path.join(RUNS, "transcripts.json"))
    if paths is None:
        paths = pick_transcripts(os.path.expanduser("~/.claude/projects"), cfg["n_transcripts"],
                                 cfg["segment_chars"], cfg["exclude"])
        save_json(os.path.join(RUNS, "transcripts.json"), paths)
    save_json(os.path.join(RUNS, "config.json"), cfg)
    only = sys.argv[1:]  # optional: run a subset by tag
    embedder = S.Embedder()
    for p in paths:
        if not only or os.path.basename(p)[:8] in only:
            run_transcript(p, cfg, embedder)


if __name__ == "__main__":
    main()
