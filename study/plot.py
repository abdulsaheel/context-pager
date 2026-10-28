"""results.json -> docs/score-vs-tokens.svg. Standard library only; makes no model calls.

Usage: uv run python -m study.plot
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, L, R, T, B = 760, 460, 64, 24, 76, 56
COLORS = {"baseline": "#7a7f87", "question-aware": "#c2742b", "paging": "#1f6f8b"}
GROUP = {"full": "baseline", "compact": "baseline", "recency": "baseline", "random": "baseline",
         "bm25-q": "question-aware", "embed-q": "question-aware", "kev-q": "question-aware"}


# (dx, dy, anchor) label offsets so points in the crowded 35-40% column stay readable
LABEL = {"recency": (-9, 4, "end"), "random": (9, 4, "start"), "bm25-q": (-9, 4, "end"),
         "embed-q": (12, 20, "start"), "kev-q": (12, 5, "start"), "text+recall": (-9, 4, "end"),
         "stubs+recall": (9, -8, "start")}


def group(name):
    return GROUP.get(name, "paging")


def main():
    rows = json.load(open(os.path.join(ROOT, "results.json")))["end_to_end"]
    x = lambda v: L + (W - L - R) * v / 1.05
    y = lambda v: T + (H - T - B) * (1 - v)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" '
           f'font-size="12"><rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{L}" y="24" font-size="15" font-weight="600">Answer quality vs. context size '
           f'(7 sessions, 68 questions)</text>']
    for t in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t)}" y2="{y(t)}" stroke="#e6e8eb"/>'
                   f'<text x="{L - 8}" y="{y(t) + 4}" text-anchor="end" fill="#555">{int(t * 100)}%</text>')
    for t in (0, .25, .5, .75, 1):
        out.append(f'<text x="{x(t)}" y="{H - B + 18}" text-anchor="middle" fill="#555">{int(t * 100)}%</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle" fill="#333">input tokens vs. full context</text>')
    out.append(f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" fill="#333">'
               f'score (Opus grade / 2)</text>')
    for r in rows:
        c, g = r["condition"], group(r["condition"])
        cx, cy, col = x(r["tokens_vs_full"]), y(r["score"]), COLORS[g]
        lo, hi = r["score_ci"]
        partial = any(isinstance(v, float) and math.isnan(v) for v in (lo, hi))
        if not partial:
            out.append(f'<line x1="{cx}" x2="{cx}" y1="{y(lo)}" y2="{y(hi)}" stroke="{col}" stroke-width="1.5"/>')
        fill = "white" if partial else col
        out.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{fill}" stroke="{col}" stroke-width="2"/>')
        label = c + (" (partial, no CI)" if partial else "")
        dx, dy, anchor = LABEL.get(c, (9, 4, "start"))
        out.append(f'<text x="{cx + dx}" y="{cy + dy}" text-anchor="{anchor}" fill="#222">{label}</text>')
    lx = L + 8
    for name, col in COLORS.items():
        out.append(f'<circle cx="{lx}" cy="46" r="5" fill="{col}"/><text x="{lx + 10}" y="50" fill="#333">{name}</text>')
        lx += 20 + 7 * len(name)
    out.append("</svg>")
    path = os.path.join(ROOT, "docs", "score-vs-tokens.svg")
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
