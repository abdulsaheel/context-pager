"""Context-pager pilot: does relevance-scored eviction keep answer quality at a fraction of the tokens?

Conditions, per question over the same transcript segment:
  full     whole segment in context
  compact  an LLM summary of the segment (what auto-compaction does)
  pager    pinned items (user messages + recent tail) + chunks the scorer marks relevant;
           everything else replaced by a one-line stub

Scorer speaks TypeSafe's /v1/systemone contract, so Kev (local) and Jev (hosted) are the same code:
  SCORER_URL=http://localhost:8009 (default, Kev)   or   SCORER_URL=https://api.typesafe.ai + TYPESAFE_API_KEY
"""
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
SCORER_URL = os.environ.get("SCORER_URL", "http://localhost:8009")
SCORER_MODEL = os.environ.get("SCORER_MODEL", "kev-latest")
SEGMENT_CHARS = 240_000   # ~60k tokens of history per transcript
ITEM_CAP = 20_000         # single huge tool output gets truncated, as Claude Code does
CHUNK_CHARS = 6_000       # Kev-0.8B is validated to 8k tokens; stay well under
TAIL_ITEMS = 6            # most recent items are always kept
KEEP_P = float(os.environ.get("KEEP_P", "0.5"))
N_QUESTIONS = 8


# ---------- transcript -> items ----------

def load_items(path):
    items = []
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        content = (d.get("message") or {}).get("content")
        role = d["type"]
        if isinstance(content, str):
            items.append({"role": role, "kind": "text", "text": content})
            continue
        for b in content or []:
            t = b.get("type")
            if t == "text":
                items.append({"role": role, "kind": "text", "text": b["text"]})
            elif t == "tool_use":
                items.append({"role": role, "kind": "tool_use",
                              "text": f'{b.get("name")} {json.dumps(b.get("input"))[:2000]}'})
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "[image]") for x in c)
                items.append({"role": role, "kind": "tool_result", "text": str(c)[:ITEM_CAP]})
            elif t == "image":
                items.append({"role": role, "kind": "text", "text": "[image]"})
    # drop harness noise and empties
    items = [i for i in items if i["text"].strip() and not i["text"].startswith("<system-reminder>")]
    for n, i in enumerate(items):
        i["id"] = n
    return items


def segment(items):
    out, size = [], 0
    for i in items:
        if size + len(i["text"]) > SEGMENT_CHARS:
            break
        out.append(i)
        size += len(i["text"])
    return out


def render(items, stubs=frozenset()):
    parts = []
    for i in items:
        if i["id"] in stubs:
            first = i["text"].strip().splitlines()[0][:80]
            parts.append(f'[#{i["id"]} {i["role"]}/{i["kind"]} EVICTED, {len(i["text"])} chars: "{first}"]')
        else:
            parts.append(f'[#{i["id"]} {i["role"]}/{i["kind"]}]\n{i["text"]}')
    return "\n\n".join(parts)


# ---------- claude -p ----------

def claude(prompt, system, model="sonnet"):
    cmd = ["claude", "-p", "--setting-sources", "", "--strict-mcp-config", "--tools", "",
           "--model", model, "--no-session-persistence", "--system-prompt", system,
           "--output-format", "json"]
    for attempt in range(3):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
        try:
            d = json.loads(r.stdout)
            u = d.get("usage", {})
            tokens = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
            if not d.get("is_error"):
                return d["result"], tokens
        except ValueError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"claude failed: {r.stdout[-500:]} {r.stderr[-500:]}")


def parse_json(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    start = min(i for i in (s.find("["), s.find("{")) if i >= 0)
    return json.JSONDecoder().raw_decode(s[start:])[0]


# ---------- scorer (Kev / Jev) ----------

def score_chunk(text, questions):
    body = {
        "state": {"chunk": text},
        "model": SCORER_MODEL,
        "questions": {
            f"q{n}": {"type": "noul",
                      "instructions": {"question": q,
                                       "ask": "Does `chunk` contain information that helps answer `question`?"}}
            for n, q in enumerate(questions)
        },
    }
    headers = {"content-type": "application/json"}
    if os.environ.get("TYPESAFE_API_KEY") and "typesafe.ai" in SCORER_URL:
        headers["authorization"] = "Bearer " + os.environ["TYPESAFE_API_KEY"]
    req = urllib.request.Request(SCORER_URL + "/v1/systemone", json.dumps(body).encode(), headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                ans = json.load(r)["answers"]
            return [ans[f"q{n}"]["noul"] for n in range(len(questions))]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def pager_keep(seg, questions):
    """Per question: set of item ids kept in context. Pinned: all user text + recent tail."""
    pinned = {i["id"] for i in seg if i["role"] == "user" and i["kind"] == "text"}
    pinned |= {i["id"] for i in seg[-TAIL_ITEMS:]}
    candidates = [i for i in seg if i["id"] not in pinned]
    # an item is kept if any of its chunks scores relevant
    best = {i["id"]: [0.0] * len(questions) for i in candidates}
    jobs = [(i["id"], i["text"][o:o + CHUNK_CHARS]) for i in candidates for o in range(0, len(i["text"]), CHUNK_CHARS)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        for (iid, _), probs in zip(jobs, ex.map(lambda j: score_chunk(j[1], questions), jobs)):
            best[iid] = [max(a, b) for a, b in zip(best[iid], probs)]
    elapsed = time.time() - t0
    keep = [pinned | {iid for iid, p in best.items() if p[n] >= KEEP_P} for n in range(len(questions))]
    return keep, best, len(jobs), elapsed


# ---------- pipeline ----------

QGEN_SYS = "You write evaluation questions for long-context memory tests. Output only JSON."
QGEN = """Below is the start of a real coding-assistant conversation. Write {n} questions that a user might later ask
in this same conversation, whose answers depend on SPECIFIC details stated in it (exact numbers, file names, error
messages, decisions, command output, URLs). Rules:
- each answer must appear in an item BEFORE the last 10% of the transcript
- not answerable from general knowledge; one fact per question; mix tool output and discussion
- give the short gold answer and the item ids (the #N markers) that contain it
Output a JSON list: [{{"q": "...", "gold": "...", "evidence": [ids]}}]

TRANSCRIPT:
{t}"""
ANSWER_SYS = ("Answer the user's question using only the conversation history provided. Be brief and specific. "
              "If the history does not contain the answer, reply exactly NOT FOUND.")
SUMMARY_SYS = "You compact long conversations so work can continue in a fresh context window."
SUMMARY = ("Summarize this conversation so an assistant can continue it with no other context. Keep decisions, "
           "facts, numbers, file paths, errors and open tasks. Max ~2,000 words.\n\n{t}")
JUDGE_SYS = "You grade answers against gold answers. Output only JSON."
JUDGE = """Grade each candidate answer against the gold answer. 2 = correct, 1 = partially correct, 0 = wrong or NOT FOUND.
Output a JSON list of {{"k": key, "grade": 0|1|2}}.

{rows}"""


def run_transcript(path, tag):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    cache = lambda name: os.path.join(d, name)
    seg = segment(load_items(path))
    full = render(seg)
    print(f"[{tag}] {len(seg)} items, {len(full):,} chars", flush=True)

    if not os.path.exists(cache("questions.json")):
        qs, _ = claude(QGEN.format(n=N_QUESTIONS, t=full), QGEN_SYS)
        json.dump(parse_json(qs), open(cache("questions.json"), "w"), indent=1)
    qs = json.load(open(cache("questions.json")))
    questions = [q["q"] for q in qs]

    if not os.path.exists(cache("summary.txt")):
        open(cache("summary.txt"), "w").write(claude(SUMMARY.format(t=full), SUMMARY_SYS)[0])
    summary = open(cache("summary.txt")).read()

    if not os.path.exists(cache("pager.json")):
        keep, best, n_chunks, secs = pager_keep(seg, questions)
        json.dump({"keep": [sorted(k) for k in keep], "best": best, "chunks": n_chunks, "seconds": secs},
                  open(cache("pager.json"), "w"))
    pg = json.load(open(cache("pager.json")))
    print(f"[{tag}] scored {pg['chunks']} chunks in {pg['seconds']:.1f}s", flush=True)

    all_ids = {i["id"] for i in seg}
    contexts = {}
    for n, q in enumerate(questions):
        contexts[(n, "full")] = full
        contexts[(n, "compact")] = summary
        contexts[(n, "pager")] = render(seg, stubs=all_ids - set(pg["keep"][n]))

    if not os.path.exists(cache("answers.json")):
        def ask(key):
            n, cond = key
            a, tok = claude(f"CONVERSATION HISTORY:\n{contexts[key]}\n\nQUESTION: {questions[n]}", ANSWER_SYS)
            return key, a, tok
        with cf.ThreadPoolExecutor(4) as ex:
            res = list(ex.map(ask, contexts))
        json.dump([{"n": n, "cond": c, "answer": a, "tokens": t} for (n, c), a, t in res],
                  open(cache("answers.json"), "w"), indent=1)
    answers = json.load(open(cache("answers.json")))

    if not os.path.exists(cache("grades.json")):
        rows = answers[:]
        random.Random(0).shuffle(rows)  # judge never sees which condition produced which answer
        text = "\n\n".join(f'key: {k}\nquestion: {questions[r["n"]]}\ngold: {qs[r["n"]]["gold"]}\ncandidate: {r["answer"]}'
                           for k, r in enumerate(rows))
        g = parse_json(claude(JUDGE.format(rows=text), JUDGE_SYS)[0])
        grades = {x["k"]: x["grade"] for x in g}
        for k, r in enumerate(rows):
            r["grade"] = grades[k]
        json.dump(rows, open(cache("grades.json"), "w"), indent=1)
    graded = json.load(open(cache("grades.json")))

    # evidence recall: did the pager keep the items the gold answer came from?
    recall = [bool(set(q["evidence"]) & set(pg["keep"][n])) for n, q in enumerate(qs) if q.get("evidence")]
    return graded, recall


def report(results):
    by = {}
    for graded, _ in results:
        for r in graded:
            s = by.setdefault(r["cond"], {"pts": 0, "max": 0, "tok": 0, "n": 0})
            s["pts"] += r["grade"]; s["max"] += 2; s["tok"] += r["tokens"]; s["n"] += 1
    full_tok = by["full"]["tok"] / by["full"]["n"]
    print(f"\n{'condition':10} {'score':>8} {'avg input tokens':>18} {'vs full':>8}")
    for c in ("full", "compact", "pager"):
        s = by[c]
        print(f"{c:10} {s['pts'] / s['max'] * 100:7.1f}% {s['tok'] / s['n']:18,.0f} {s['tok'] / s['n'] / full_tok * 100:7.1f}%")
    rec = [x for _, r in results for x in r]
    print(f"\npager kept the gold evidence item for {sum(rec)}/{len(rec)} questions")


if __name__ == "__main__":
    paths = sys.argv[1:]
    results = list(cf.ThreadPoolExecutor(3).map(
        lambda p: run_transcript(p, os.path.basename(p)[:8]), paths))
    report(results)
