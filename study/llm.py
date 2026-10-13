"""Headless Claude Code as the answer / judge / question-writer model. No tools, no hooks, no CLAUDE.md."""
import json
import re
import subprocess
import time


def claude(prompt, system, model="sonnet"):
    """Returns (text, input_tokens). input_tokens counts cached and uncached prompt tokens alike,
    so it measures context size, not price. Waits out usage limits instead of failing the run."""
    cmd = ["claude", "-p", "--setting-sources", "", "--strict-mcp-config", "--tools", "",
           "--model", model, "--no-session-persistence", "--system-prompt", system, "--output-format", "json"]
    last = ""
    for attempt in range(30):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
            d = json.loads(r.stdout)
            if not d.get("is_error"):
                u = d.get("usage", {})
                return d["result"], (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                                     + u.get("cache_read_input_tokens", 0))
            last = str(d.get("result"))
        except subprocess.TimeoutExpired:
            last = "timeout"
        except ValueError:
            last = f"bad output: {r.stdout[-200:]} {r.stderr[-200:]}"
        wait = 900 if "limit" in last.lower() else min(60, 5 * (attempt + 1))
        print(f"  claude retry {attempt + 1} in {wait}s: {last[:160]}", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"claude failed after retries: {last[:500]}")


def parse_json(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    start = min(i for i in (s.find("["), s.find("{")) if i >= 0)
    return json.JSONDecoder().raw_decode(s[start:])[0]
