#!/usr/bin/env python3
"""Three probes that decide whether a serving layer can sit on geniex serve.

1. Does it re-seed per request? The bundles ship "seed": 42 and Genie re-seeds
   from the config on every GenieDialog_reset, so a server that loads the
   config verbatim replays byte-identical output for a repeated prompt. Only
   controlling dialog creation fixes that -- which an HTTP proxy cannot do.
2. What happens past the compiled window? Genie hard-errors on overflow rather
   than truncating, so a serving layer must evict. Inheriting someone else's
   eviction policy is only acceptable if there is one.
3. Are stop sequences honoured? Genie wants a keyed object and silently ignores
   a bare list, so this is a real thing to get wrong.

Standing the servers up to point this at -- the geniex import, and the four
undocumented steps GenieAPIService needs -- is in "Reproducing the cross-server
comparison" in docs/GENIE_SERVER.md. Needs `pip install tokenizers`.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

from tokenizers import Tokenizer

# Same contract as bench_servers.py: the bundle location is an env var, not a
# path baked into the file. This script used to pin one machine's absolute
# path, which meant it crashed on import for everyone except its author -- on
# a script the README cites as the instrument behind a whole findings table.
BUNDLE = os.environ.get("GENIE_BUNDLE_DIR", "")
if not BUNDLE:
    sys.exit("set GENIE_BUNDLE_DIR to the bundle the server under test is serving")
TOK = Tokenizer.from_file(os.path.join(BUNDLE, "tokenizer.json"))
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18181"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qualcomm/qwen3-4b-ours"
CAP = sys.argv[3] if len(sys.argv) > 3 else "max_completion_tokens"


def ntok(t):
    return len(TOK.encode(t, add_special_tokens=False).ids)


def ask(messages, cap=48, extra=None, timeout=900):
    body = {"model": MODEL, "messages": messages, CAP: cap, "stream": False}
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")[:300]
        return {"http": e.code, "body": raw, "wall": time.perf_counter() - t0}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                "wall": time.perf_counter() - t0}
    m = d["choices"][0].get("message") or {}
    return {"text": m.get("content") or "",
            "finish": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage"), "wall": time.perf_counter() - t0}


def filler(n):
    unit = ("Memory bandwidth on a mobile accelerator is the binding "
            "constraint for token generation, and this sentence repeats. ")
    body = unit * max(1, n // max(1, ntok(unit)))
    while ntok(body) < n:
        body += unit
    return TOK.decode(TOK.encode(body, add_special_tokens=False).ids[:n])


print("=" * 68)
print("PROBE 1 -- does it re-seed per request?")
print("=" * 68)
p = [{"role": "user", "content":
      "Invent a short, unusual name for a coastal town. Reply with the name only."}]
outs = []
for i in range(3):
    r = ask(p, cap=24)
    t = (r.get("text") or r.get("error") or str(r.get("http")))[:90].replace("\n", " ")
    outs.append(r.get("text"))
    print("  run%d (%.1fs): %r" % (i + 1, r["wall"], t))
if all(o is not None for o in outs):
    uniq = len(set(outs))
    print("  -> %d distinct of 3. %s" % (
        uniq,
        "REPLAYS -- loads the config seed verbatim, no per-request re-seed"
        if uniq == 1 else "RE-SEEDS -- output varies across identical requests"))

print()
print("=" * 68)
print("PROBE 2 -- what happens past the compiled 8192 window?")
print("=" * 68)
for depth in (7000, 9000, 20000):
    msg = [{"role": "user", "content": filler(depth) + "\n\nSummarise in five words."}]
    r = ask(msg, cap=32, timeout=1200)
    if "text" in r:
        print("  ~%-6d tok: OK in %.1fs finish=%s usage=%s text=%r" % (
            depth, r["wall"], r["finish"], (r.get("usage") or {}).get("prompt_tokens"),
            (r["text"] or "")[:60].replace("\n", " ")))
    elif "http" in r:
        print("  ~%-6d tok: HTTP %s in %.1fs -- %s" % (
            depth, r["http"], r["wall"], r["body"][:160].replace("\n", " ")))
    else:
        print("  ~%-6d tok: %s after %.1fs" % (depth, r["error"], r["wall"]))

print()
print("=" * 68)
print("PROBE 3 -- are stop sequences honoured?")
print("=" * 68)
base = ask([{"role": "user", "content":
             "Count: one, two, three, four, five, six, seven, eight."}], cap=64)
print("  no stop     : %r" % (base.get("text") or "")[:100].replace("\n", " "))
withstop = ask([{"role": "user", "content":
                 "Count: one, two, three, four, five, six, seven, eight."}],
               cap=64, extra={"stop": ["four"]})
txt = withstop.get("text")
if txt is None:
    print("  stop=[four] : %s" % (withstop.get("error") or withstop.get("http")))
else:
    print("  stop=[four] : %r finish=%s" % (txt[:100].replace("\n", " "),
                                            withstop.get("finish")))
    b = (base.get("text") or "")
    if "four" in b and "four" not in txt:
        print("  -> HONOURED: 'four' present without stop, absent with it")
    elif txt == b:
        print("  -> IGNORED: byte-identical to the unstopped run")
    else:
        print("  -> UNCLEAR: output differs but 'four' handling is ambiguous")
