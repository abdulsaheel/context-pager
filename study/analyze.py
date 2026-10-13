"""Turn runs/ into results.md + results.json. Pure computation; no model calls.

Usage: uv run python -m study.analyze
Statistics: 95% CIs from a cluster bootstrap that resamples whole transcripts (questions within a
transcript are not independent), 10,000 resamples, seed 0. Paired differences use the same resamples.
"""
import json
import os

import numpy as np

from study import selectors as S
from study.run import CONFIG, RUNS, load_json
from study.transcripts import clean_items, load_items, render, segment

PRIMARY = "grade_opus"
BOOT = 10_000
ORDER = ["full", "compact", "recency", "random", "bm25-q", "embed-q", "kev-q",
         "recency+recall", "kev-task+recall", "stubs+recall", "text+recall"]
PAIRS = [("kev-q", "full"), ("kev-q", "bm25-q"), ("kev-q", "embed-q"), ("embed-q", "bm25-q"),
         ("bm25-q", "random"), ("compact", "full"), ("kev-task+recall", "recency+recall"),
         ("stubs+recall", "recency+recall"), ("stubs+recall", "full"), ("text+recall", "recency+recall"), ("text+recall", "full")]
BUDGETS = [0.05, 0.10, 0.25, 0.50]


def load_runs():
    runs = []
    for path in load_json(os.path.join(RUNS, "transcripts.json")):
        d = os.path.join(RUNS, os.path.basename(path)[:8])
        answers = load_json(os.path.join(d, "answers.json"), [])
        if not answers or any(PRIMARY not in a for a in answers):
            continue  # transcript not finished
        runs.append({"tag": os.path.basename(d), "path": path, "dir": d, "answers": answers,
                     "questions": load_json(os.path.join(d, "questions.json")),
                     "scores": load_json(os.path.join(d, "scores.json")),
                     "seg": segment(load_items(path), CONFIG["segment_chars"])})
    return runs


def cluster_boot(per_transcript, stat, rng):
    """per_transcript: list of arrays (one per transcript). stat: fn(concatenated array) -> float."""
    k = len(per_transcript)
    out = np.empty(BOOT)
    for b in range(BOOT):
        pick = rng.integers(0, k, k)
        out[b] = stat(np.concatenate([per_transcript[i] for i in pick]))
    return np.percentile(out, [2.5, 97.5])


def fmt_ci(mean, ci, pct=True):
    f = (lambda x: f"{x * 100:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(mean)} [{f(ci[0])}, {f(ci[1])}]"


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def end_to_end(runs, rng):
    conds = [c for c in ORDER if any(a["cond"] == c for r in runs for a in r["answers"])]
    # matrix per transcript: question x condition
    table = {}
    for r in runs:
        nq = len(r["questions"])
        g = {c: np.full(nq, np.nan) for c in conds}
        t = {c: np.full(nq, np.nan) for c in conds}
        for a in r["answers"]:
            if a["cond"] in g and a.get(PRIMARY) is not None:
                g[a["cond"]][a["n"]] = a[PRIMARY] / 2
                t[a["cond"]][a["n"]] = a["tokens"]
        table[r["tag"]] = (g, t)
    tags = list(table)
    full_tok = np.nanmean(np.concatenate([table[x][1]["full"] for x in tags]))
    rows = []
    for c in conds:
        scores = [table[x][0][c][~np.isnan(table[x][0][c])] for x in tags]
        allv = np.concatenate(scores)
        exact = [s == 1.0 for s in scores]
        tok = np.nanmean(np.concatenate([table[x][1][c] for x in tags]))
        rows.append({"condition": c, "n": int(len(allv)),
                     "score": float(allv.mean()), "score_ci": cluster_boot(scores, np.mean, rng).tolist(),
                     "exact": float(np.concatenate(exact).mean()),
                     "exact_ci": cluster_boot([e.astype(float) for e in exact], np.mean, rng).tolist(),
                     "tokens": float(tok), "tokens_vs_full": float(tok / full_tok)})
    diffs = []
    for a, b in PAIRS:
        if a not in conds or b not in conds:
            continue
        per = [table[x][0][a] - table[x][0][b] for x in tags]
        per = [p[~np.isnan(p)] for p in per]
        d = np.concatenate(per)
        diffs.append({"a": a, "b": b, "diff": float(d.mean()), "ci": cluster_boot(per, np.mean, rng).tolist(),
                      "a_wins": int((d > 0).sum()), "b_wins": int((d < 0).sum()), "ties": int((d == 0).sum())})
    # same table restricted to questions the full-context condition answered exactly (answerability check)
    sub = []
    for c in conds:
        v = np.concatenate([table[x][0][c][table[x][0]["full"] == 1.0] for x in tags])
        sub.append({"condition": c, "n": int(len(v)), "score": float(np.nanmean(v))})
    return rows, diffs, sub


def judges(runs):
    pairs = [(a["grade_opus"], a["grade_sonnet"]) for r in runs for a in r["answers"]
             if a.get("grade_opus") is not None and a.get("grade_sonnet") is not None]
    a, b = zip(*pairs)
    return {"n": len(pairs), "agreement": float(np.mean(np.array(a) == np.array(b))), "kappa": float(kappa(a, b))}


def retrieval(runs, rng):
    """Selector quality without any LLM: was the evidence item kept at budget X? AUROC over non-pinned items."""
    scorers = ["recency", "random", "bm25-q", "embed-q", "kev-q", "bm25-task", "embed-task", "kev-task"]
    rec = {(s, b): [] for s in scorers for b in BUDGETS}
    auc = {s: [] for s in scorers}
    for r in runs:
        seg, sc = r["seg"], r["scores"]
        total = sum(len(i["text"]) for i in seg)
        pinned = S.pinned_ids(seg, CONFIG["tail_items"])
        for n, q in enumerate(r["questions"]):
            ev = set(q["evidence"])
            for s in scorers:
                if s in ("recency", "random"):
                    m = sc[s]
                else:
                    name, kind = s.split("-")
                    m = sc[name][f"q{n}" if kind == "q" else "task"]
                m = {int(k): v for k, v in m.items()}
                for b in BUDGETS:
                    rec[(s, b)].append((r["tag"], bool(ev & S.fill_budget(seg, m, b * total, pinned))))
                auc[s].append((r["tag"], S.auroc({k: v for k, v in m.items() if k not in pinned}, ev)))
    tags = [r["tag"] for r in runs]
    group = lambda vals: [np.array([v for t, v in vals if t == tag], dtype=float) for tag in tags]
    out = []
    for s in scorers:
        row = {"scorer": s}
        for b in BUDGETS:
            g = group(rec[(s, b)])
            row[f"recall@{int(b * 100)}%"] = (float(np.concatenate(g).mean()), cluster_boot(g, np.mean, rng).tolist())
        g = [x[~np.isnan(x)] for x in group(auc[s])]
        row["auroc"] = (float(np.concatenate(g).mean()), cluster_boot(g, np.mean, rng).tolist())
        out.append(row)
    return out


def recall_behaviour(runs):
    out = {}
    for c in ("recency+recall", "kev-task+recall", "stubs+recall", "text+recall"):
        used = hit = missing = 0
        n = 0
        for r in runs:
            for a in r["answers"]:
                if a["cond"] != c:
                    continue
                n += 1
                ev = set(r["questions"][a["n"]]["evidence"])
                kept = set(a["kept"] or [])
                used += bool(a["recalled"])
                if not ev & kept:
                    missing += 1
                    hit += bool(ev & set(a["recalled"]))
        out[c] = {"n": n, "used_recall": used, "evidence_evicted": missing, "evidence_recalled": hit}
    return out


def systems(runs):
    kev = [r["scores"]["kev_timing"] for r in runs]
    nq = [len(r["questions"]) for r in runs]
    req = sum(k["requests"] for k in kev)
    clean = []
    for r in runs:
        full = len(render(r["seg"]))
        clean.append(len(render(r["seg"], texts=clean_items(r["seg"]))) / full)
    return {"kev_requests": req,
            "kev_ms_per_request_with_questions": 1000 * sum(k["seconds_questions"] for k in kev) / req,
            "kev_ms_per_request_task_only": 1000 * sum(k["seconds_task"] for k in kev) / req,
            "questions_per_request": float(np.mean(nq)),
            "clean_size_vs_full": [round(c, 3) for c in clean]}


def main():
    rng = np.random.default_rng(0)
    runs = load_runs()
    rows, diffs, sub = end_to_end(runs, rng)
    res = {"transcripts": [r["tag"] for r in runs], "questions": sum(len(r["questions"]) for r in runs),
           "end_to_end": rows, "paired": diffs, "answerable_subset": sub, "judges": judges(runs),
           "retrieval": retrieval(runs, rng), "recall": recall_behaviour(runs), "systems": systems(runs)}
    json.dump(res, open(os.path.join(os.path.dirname(RUNS), "results.json"), "w"), indent=1)

    L = [f"# Results ({len(runs)} transcripts, {res['questions']} questions)\n",
         f"Primary judge: Opus. Score = mean grade/2. 95% CI: cluster bootstrap over transcripts.\n",
         "## End-to-end answer quality vs context size\n",
         "| condition | score % [CI] | exact % [CI] | avg input tokens | vs full |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['condition']} | {fmt_ci(r['score'], r['score_ci'])} | {fmt_ci(r['exact'], r['exact_ci'])} "
                 f"| {r['tokens']:,.0f} | {r['tokens_vs_full'] * 100:.1f}% |")
    L += ["\n## Paired differences (score points, a minus b)\n", "| a | b | diff [CI] | a better | b better | tie |",
          "|---|---|---|---|---|---|"]
    for d in diffs:
        L.append(f"| {d['a']} | {d['b']} | {fmt_ci(d['diff'], d['ci'])} | {d['a_wins']} | {d['b_wins']} | {d['ties']} |")
    L += ["\n## Answerable subset (questions full context got exactly right)\n", "| condition | n | score % |", "|---|---|---|"]
    for s in sub:
        L.append(f"| {s['condition']} | {s['n']} | {s['score'] * 100:.1f} |")
    j = res["judges"]
    L += [f"\n## Judge agreement\n\nOpus vs Sonnet on {j['n']} answers: agreement {j['agreement'] * 100:.1f}%, "
          f"Cohen's kappa {j['kappa']:.3f}.\n",
          "## Selector quality without an LLM (evidence item kept?)\n",
          "| scorer | " + " | ".join(f"recall@{int(b * 100)}%" for b in BUDGETS) + " | AUROC |",
          "|---|" + "---|" * (len(BUDGETS) + 1)]
    for r in res["retrieval"]:
        cells = [fmt_ci(*r[f"recall@{int(b * 100)}%"]) for b in BUDGETS]
        L.append(f"| {r['scorer']} | " + " | ".join(cells) + f" | {fmt_ci(*r['auroc'], pct=False)} |")
    L += ["\n## Recall (paging) behaviour\n", "| condition | n | used recall | evidence evicted | evicted evidence recalled |",
          "|---|---|---|---|---|"]
    for c, v in res["recall"].items():
        L.append(f"| {c} | {v['n']} | {v['used_recall']} | {v['evidence_evicted']} | {v['evidence_recalled']} |")
    s = res["systems"]
    L += [f"\n## Systems\n\nKev-0.8B on Apple M4 (MLX): {s['kev_requests']} requests, "
          f"{s['kev_ms_per_request_with_questions']:.0f} ms/request with ~{s['questions_per_request']:.0f} questions, "
          f"{s['kev_ms_per_request_task_only']:.0f} ms/request for the single task question.\n",
          f"Deterministic cleaning, size vs full per transcript: {s['clean_size_vs_full']}"]
    open(os.path.join(os.path.dirname(RUNS), "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
