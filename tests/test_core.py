"""Run: uv run python -m pytest -q tests   (or: uv run python tests/test_core.py)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from study import selectors as S
from study.transcripts import clean_items, clean_text, render


def seg_of(*texts, roles=None):
    roles = roles or ["assistant"] * len(texts)
    return [{"id": n, "role": r, "kind": "text" if r == "user" else "tool_result", "text": t}
            for n, (t, r) in enumerate(zip(texts, roles))]


def test_fill_budget_respects_budget_and_pins():
    seg = seg_of("u" * 10, "a" * 50, "b" * 50, "c" * 50, roles=["user", "assistant", "assistant", "assistant"])
    keep = S.fill_budget(seg, {0: 0, 1: 3, 2: 2, 3: 1}, budget_chars=70, pinned={0})
    assert keep == {0, 1}                       # pinned + best item that fits; next one would exceed 70
    assert sum(len(seg[i]["text"]) for i in keep) <= 70


def test_fill_budget_skips_oversized_but_continues():
    seg = seg_of("x" * 100, "y" * 10, "z" * 10)
    assert S.fill_budget(seg, {0: 9, 1: 5, 2: 1}, budget_chars=25, pinned=set()) == {1, 2}


def test_bm25_ranks_matching_item_first():
    seg = seg_of("the build failed with error E1234 in parser.py", "lunch menu: pasta", "weather is sunny")
    sc = S.bm25(seg, "which error broke the parser build?")
    assert max(sc, key=sc.get) == 0


def test_auroc():
    assert S.auroc({0: 0.9, 1: 0.1, 2: 0.2}, {0}) == 1.0
    assert S.auroc({0: 0.1, 1: 0.9, 2: 0.2}, {0}) == 0.0
    assert S.auroc({0: 0.5, 1: 0.5}, {0}) == 0.5


def test_clean_strips_script_and_repeats_but_keeps_facts():
    raw = "<script>var a=1;</script><p>Price: $0.042 per Mtok</p>\nnav\nnav\nnav\nnav"
    out = clean_text(raw)
    assert "$0.042" in out and "var a" not in out and out.count("nav") == 2


def test_clean_dedupes_identical_tool_results():
    seg = seg_of("same output " * 30, "same output " * 30)
    texts = clean_items(seg)
    assert texts[1] == "[identical to #0]"


def test_render_stubs_evicted():
    seg = seg_of("first line\nmore", "kept")
    out = render(seg, keep={1})
    assert "EVICTED" in out and "first line" in out and "more" not in out and "kept" in out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
