"""Claude Code session transcripts -> flat list of context items, and rendering back to a prompt."""
import glob
import json
import os
import re

ITEM_CAP = 20_000  # one huge tool output is truncated, as Claude Code itself does


def load_items(path):
    """One item per message block: user/assistant text, tool calls, tool results. Thinking is dropped
    (Claude Code does not resend old thinking either)."""
    items = []
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        role = d["type"]
        content = (d.get("message") or {}).get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for b in content or []:
            t = b.get("type")
            if t == "text":
                text, kind = b["text"], "text"
            elif t == "tool_use":
                text, kind = f'{b.get("name")} {json.dumps(b.get("input"))[:2000]}', "tool_use"
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "[image]") for x in c)
                text, kind = str(c)[:ITEM_CAP], "tool_result"
            elif t == "image":
                text, kind = "[image]", "text"
            else:
                continue
            if text.strip() and not text.startswith("<system-reminder>"):
                items.append({"role": role, "kind": kind, "text": text})
    for n, i in enumerate(items):
        i["id"] = n
    return items


def segment(items, max_chars):
    """Leading run of items whose text fits in max_chars: the 'history' a long session would carry."""
    out, size = [], 0
    for i in items:
        if size + len(i["text"]) > max_chars:
            break
        out.append(i)
        size += len(i["text"])
    return out


def stub(i):
    first = i["text"].strip().splitlines()[0][:100]
    return f'[#{i["id"]} {i["role"]}/{i["kind"]} EVICTED ({len(i["text"])} chars): {first}]'


def render(items, keep=None, texts=None):
    """keep=None renders everything; otherwise items not in keep become one-line stubs.
    texts optionally overrides item text (used by the cleaning condition)."""
    parts = []
    for i in items:
        if keep is not None and i["id"] not in keep:
            parts.append(stub(i))
        else:
            parts.append(f'[#{i["id"]} {i["role"]}/{i["kind"]}]\n{(texts or {}).get(i["id"], i["text"])}')
    return "\n\n".join(parts)


def pick_transcripts(root, n, min_chars, exclude=()):
    """Top-level sessions with at least min_chars of history, round-robin across projects
    (largest first within a project) so no single project dominates."""
    by_project = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "*.jsonl")), key=os.path.getsize, reverse=True):
        if os.path.getsize(p) < 500_000 or any(x in p for x in exclude):
            continue
        by_project.setdefault(os.path.dirname(p), []).append(p)
    picked, queues = [], list(by_project.values())
    while len(picked) < n and any(queues):
        for q in queues:
            while q:
                p = q.pop(0)
                if sum(len(i["text"]) for i in segment(load_items(p), min_chars)) >= 0.9 * min_chars:
                    picked.append(p)
                    break
            if len(picked) == n:
                break
    return picked


# ---- deterministic cleaning: no model, content-preserving rules ----

_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>\n]{1,200}>")
_BLANKS = re.compile(r"\n\s*\n+")


def clean_text(text):
    text = _SCRIPT.sub(" ", text)
    text = _TAG.sub(" ", text)
    lines, seen, out = text.splitlines(), {}, []
    for line in lines:
        key = line.strip()
        seen[key] = seen.get(key, 0) + 1
        if key and seen[key] > 2:  # boilerplate repeated within one output (nav, log spam)
            continue
        out.append(line.rstrip())
    return _BLANKS.sub("\n\n", "\n".join(out)).strip()


def clean_items(items):
    """Per-item cleaned text; a tool result identical to an earlier one becomes a back-reference."""
    texts, first_seen = {}, {}
    for i in items:
        t = clean_text(i["text"]) if i["kind"] == "tool_result" else i["text"]
        if i["kind"] == "tool_result" and len(t) > 200:
            if t in first_seen:
                t = f"[identical to #{first_seen[t]}]"
            else:
                first_seen[t] = i["id"]
        texts[i["id"]] = t
    return texts
