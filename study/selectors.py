"""Relevance scorers and a single budget-filling rule shared by all of them, so every selector
is compared at the same token budget."""
import json
import random
import re
import time
import urllib.request

import numpy as np
from rank_bm25 import BM25Okapi

KEV_URL = "http://localhost:8009/v1/systemone"
CHUNK_CHARS = 6_000  # Kev-0.8B is validated to 8k tokens; stay well under
_WORD = re.compile(r"[a-z0-9_]+")


def pinned_ids(seg, tail_items):
    """Always kept by every selector: everything the user typed, plus the most recent items."""
    ids = {i["id"] for i in seg if i["role"] == "user" and i["kind"] == "text"}
    return ids | {i["id"] for i in seg[-tail_items:]}


def fill_budget(seg, scores, budget_chars, pinned):
    """Pinned first, then items by descending score while they fit. Returns kept ids."""
    keep = set(pinned)
    used = sum(len(i["text"]) for i in seg if i["id"] in keep)
    for i in sorted((i for i in seg if i["id"] not in keep), key=lambda i: -scores[i["id"]]):
        if used + len(i["text"]) <= budget_chars:
            keep.add(i["id"])
            used += len(i["text"])
    return keep


# ---- scorers: each returns {item_id: score}, higher = keep ----

def recency(seg, query=None):
    return {i["id"]: i["id"] for i in seg}


def random_scores(seg, query=None, seed=0):
    rng = random.Random(seed)
    return {i["id"]: rng.random() for i in seg}


def _tok(text):
    return _WORD.findall(text.lower())


def bm25(seg, query):
    model = BM25Okapi([_tok(i["text"]) or ["_"] for i in seg])
    return dict(zip((i["id"] for i in seg), model.get_scores(_tok(query))))


class Embedder:
    """Dense retrieval: cosine similarity between the query and each item's best-matching chunk."""

    def __init__(self, name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self._cache = {}

    def _item_vecs(self, seg):
        key = hash(tuple((i["id"], len(i["text"]), i["text"][:64]) for i in seg))
        if key not in self._cache:
            chunks = [(i["id"], i["text"][o:o + 2000]) for i in seg for o in range(0, len(i["text"]), 2000)]
            vecs = self.model.encode([c for _, c in chunks], normalize_embeddings=True, batch_size=32)
            self._cache = {key: (chunks, vecs)}
        return self._cache[key]

    def scores(self, seg, query):
        chunks, vecs = self._item_vecs(seg)
        # bge models expect this prefix on queries
        q = self.model.encode(["Represent this sentence for searching relevant passages: " + query],
                              normalize_embeddings=True)[0]
        out = {i["id"]: -1.0 for i in seg}
        for (iid, _), s in zip(chunks, vecs @ q):
            out[iid] = max(out[iid], float(s))
        return out


def kev_scores(seg, queries, ask, workers=4):
    """One Kev request per chunk, every query asked in parallel against it (speculative fan-out).
    queries: {qkey: text}. ask: noul instruction mentioning `chunk` and `query`.
    Returns ({qkey: {item_id: p}}, seconds, n_requests). An item scores the max over its chunks."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(i["id"], i["text"][o:o + CHUNK_CHARS]) for i in seg for o in range(0, len(i["text"]), CHUNK_CHARS)]
    out = {k: {i["id"]: 0.0 for i in seg} for k in queries}

    def one(job):
        body = {"state": {"chunk": job[1]}, "model": "kev-latest",
                "questions": {k: {"type": "noul", "instructions": {"query": q, "ask": ask}} for k, q in queries.items()}}
        req = urllib.request.Request(KEV_URL, json.dumps(body).encode(), {"content-type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.load(r)["answers"]
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for (iid, _), ans in zip(jobs, ex.map(one, jobs)):
            for k in queries:
                out[k][iid] = max(out[k][iid], ans[k]["noul"])
    return out, time.time() - t0, len(jobs)


def auroc(scores, positives):
    """Probability a random evidence item outranks a random non-evidence item (ties count half)."""
    pos = np.array([s for i, s in scores.items() if i in positives])
    neg = np.array([s for i, s in scores.items() if i not in positives])
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
