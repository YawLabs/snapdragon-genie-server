#!/usr/bin/env python3
"""Three probes that decide whether a serving layer can sit on geniex serve.

1. Does it re-seed per request? The bundles ship "seed": 42 and Genie re-seeds
   from the config on every GenieDialog_reset, so a server that loads the
   config verbatim replays byte-identical output for a repeated prompt. Only
   controlling dialog creation fixes that -- which an HTTP proxy cannot do.
2. What happens past the window? Genie hard-errors on overflow rather than
   truncating, so a serving layer must evict. Inheriting someone else's
   eviction policy is only acceptable if there is one. The window is read from
   the served bundle's genie_config.json (dialog.context.size) and the three
   prompt sizes derive from it -- one inside, one just past, one far past --
   because a literal here was right for exactly one bundle and silently
   changed meaning against every other.
3. Are stop sequences honoured? Genie wants a keyed object and silently ignores
   a bare list, so this is a real thing to get wrong.

Usage:
    GENIE_BUNDLE_DIR=<bundle> python src/probe_server_semantics.py [BASE] [MODEL] [CAP]

The three positionals point it at a server. The defaults are geniex serve's:
http://127.0.0.1:18181, qualcomm/qwen3-4b-ours, max_completion_tokens. For
this repo's server pass http://127.0.0.1:8123 qwen3-4b-npu max_tokens. CAP is
the spelling of the completion cap the server honours -- geniex ignores the
legacy `max_tokens` outright (measured in bench_servers.py). GENIE_BUNDLE_DIR
must be the bundle the server is serving: its tokenizer sizes the prompts and
its genie_config.json sets PROBE 2's window.

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

from prompt_depth import load_tokenizer, prompt_at

DEFAULT_BASE = "http://127.0.0.1:18181"
DEFAULT_MODEL = "qualcomm/qwen3-4b-ours"
DEFAULT_CAP = "max_completion_tokens"

# PROBE 2's prompt sizes as fractions of the window: inside it, just past it,
# far past it. On the 8192 bundle the findings table was measured against
# these come to 6963 / 9011 / 20480, which the old literal 7000 / 9000 / 20000
# approximated; on the launcher's default 8B tier (a 4096 prebuilt) the literal
# put all three rows past the window and the probe lost its only in-window
# control without saying so.
DEPTH_FRACTIONS = (0.85, 1.1, 2.5)


def read_window(bundle_dir):
    """`dialog.context.size` from the bundle's genie_config.json.

    The software cap Genie enforces on a dialog -- how much a client may send
    before the engine hard-errors -- which is the number PROBE 2 straddles.
    Not necessarily the length the bundle was compiled at (`--context-lengths`);
    genie_server.read_context_size explains the difference. Exits naming the
    path when the file or key is missing rather than falling back: the depths
    derive from this number, and a default would put back the literal it
    replaced.
    """
    path = os.path.join(bundle_dir, "genie_config.json")
    try:
        with open(path, encoding="utf-8") as f:
            size = int(json.load(f)["dialog"]["context"]["size"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit("cannot read dialog.context.size from %s (%s: %s) -- PROBE 2 "
                 "sizes its prompts from the served bundle's window" % (path, type(e).__name__, e))
    if size <= 0:
        sys.exit("dialog.context.size in %s is %d; the window must be positive" % (path, size))
    return size


def probe_depths(window):
    """The three PROBE 2 prompt sizes for `window`, per DEPTH_FRACTIONS."""
    return tuple(int(window * f) for f in DEPTH_FRACTIONS)


def ask(base, model, cap_key, messages, cap=48, extra=None, timeout=900):
    """One chat completion as a row of the probe's output.

    {"text", "finish", "usage", "wall"} on success; {"http", "body", "wall"} on
    a 4xx/5xx with the first 300 bytes of the server's error body; {"error",
    "wall"} on anything else -- including a 200 whose body is not
    OpenAI-shaped. Every failure is a row, never an exception: one bad answer
    costs one line of output, not the two probes after it. The shape check
    used to sit outside the try, so a 200 carrying valid JSON with no
    "choices" (an error envelope, a different API) raised KeyError out of
    here and ended the run mid-way.

    `text` is a str wherever it is present, which is what the printers assume
    (they slice it and call .replace). A message whose `content` is
    Anthropic-style content blocks is an error row for that reason, the same
    verdict this already gives a `message` that is not an object at all.
    """
    body = {"model": model, "messages": messages, cap_key: cap, "stream": False}
    if extra:
        body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions",
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
    choices = d.get("choices") if isinstance(d, dict) else None
    # `not isinstance(choices, list)` first: a "choices" that is an OBJECT is
    # truthy and not empty, so the length checks pass and `choices[0]` raised
    # KeyError: 0 out of here -- the exception this function promises never to
    # let out.
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {"error": "non-OpenAI body: %s" % json.dumps(d)[:120],
                "wall": time.perf_counter() - t0}
    c = choices[0]
    m = c.get("message") or {}
    if not isinstance(m, dict):
        # A choice whose "message" is a bare string (or a list of content
        # blocks) is as un-OpenAI-shaped as a body with no "choices" at all,
        # and `m.get` raised AttributeError straight out of here. The sibling
        # tools already treat this shape as real input from a third server on
        # the port (bench_endpoint._content_of, genie_smoke's shape clause).
        return {"error": "non-OpenAI body: %s" % json.dumps(d)[:120],
                "wall": time.perf_counter() - t0}
    text = m.get("content")
    # A well-formed message whose `content` is Anthropic-style content blocks
    # (a list, or one bare block) is the same un-OpenAI shape as the branch
    # above, one layer in -- and it used to come back as `text`, a LIST, which
    # every consumer here treats as a string: main() slices it and calls
    # .replace, raising AttributeError out of a function whose docstring
    # promises every failure is a row and never an exception. A row, then, not
    # "": three empty texts would have made seed_verdict announce "EMPTY -- 3
    # of 3 completions had no content" about a server that generated words,
    # which is a wrong finding rather than a missing one. `content: null` still
    # reads as empty, because a 200 with no content is a real answer this probe
    # reports on (README, and seed_verdict's EMPTY line).
    if text is not None and not isinstance(text, str):
        return {"error": "non-OpenAI body: %s" % json.dumps(d)[:120],
                "wall": time.perf_counter() - t0}
    return {"text": text or "",
            "finish": c.get("finish_reason"),
            "usage": d.get("usage"), "wall": time.perf_counter() - t0}


def seed_verdict(outs):
    """PROBE 1's verdict line for three completions, or None if any request
    failed (its row already printed the error).

    Three EMPTY completions are byte-identical too, and used to read as
    "REPLAYS" -- a verdict about seeding for a server that produced nothing to
    compare. The README documents that failure (a 200 with no content) on one
    of the engines this can be pointed at, so it gets its own line.
    """
    if any(o is None for o in outs):
        return None
    empty = sum(1 for o in outs if not o.strip())
    if empty:
        return "EMPTY -- %d of 3 completions had no content; no seed verdict" % empty
    uniq = len(set(outs))
    return "%d distinct of 3. %s" % (
        uniq,
        "REPLAYS -- loads the config seed verbatim, no per-request re-seed"
        if uniq == 1 else "RE-SEEDS -- output varies across identical requests")


def _failure(r):
    return r.get("error") or "HTTP %s -- %s" % (r.get("http"), (r.get("body") or "")[:160])


def stop_lines(base, withstop):
    """PROBE 3's output lines for the unstopped and stopped rows.

    A failed unstopped run used to print as `no stop : ''` with its error
    unseen, and the verdict then compared the stopped text against that empty
    string -- so a failed base plus an empty stopped completion said
    "IGNORED: byte-identical to the unstopped run" about a run that never
    happened. Either row failing now prints the failure and skips the verdict.
    """
    lines = []
    if "text" not in base:
        lines.append("  no stop     : %s" % _failure(base))
    else:
        lines.append("  no stop     : %r" % base["text"][:100].replace("\n", " "))
    if "text" not in withstop:
        lines.append("  stop=[four] : %s" % _failure(withstop))
    else:
        lines.append("  stop=[four] : %r finish=%s" % (
            withstop["text"][:100].replace("\n", " "), withstop.get("finish")))
    if "text" not in base or "text" not in withstop:
        lines.append("  -> NO VERDICT: a run failed, nothing to compare")
        return lines
    b, txt = base["text"], withstop["text"]
    if "four" in b and "four" not in txt:
        lines.append("  -> HONOURED: 'four' present without stop, absent with it")
    elif txt == b:
        lines.append("  -> IGNORED: byte-identical to the unstopped run")
    else:
        lines.append("  -> UNCLEAR: output differs but 'four' handling is ambiguous")
    return lines


def main(argv=None):
    argv = sys.argv if argv is None else argv
    base = argv[1] if len(argv) > 1 else DEFAULT_BASE
    model = argv[2] if len(argv) > 2 else DEFAULT_MODEL
    cap_key = argv[3] if len(argv) > 3 else DEFAULT_CAP
    # Same contract as bench_servers.py: the bundle location is an env var,
    # not a path baked into the file. This script used to pin one machine's
    # absolute path, which meant it crashed on import for everyone except its
    # author -- on a script the README cites as the instrument behind a whole
    # findings table. load_tokenizer names the path on every way that can
    # still go wrong.
    bundle = os.environ.get("GENIE_BUNDLE_DIR", "")
    tok = load_tokenizer(bundle)
    window = read_window(bundle)

    def q(messages, **kw):
        return ask(base, model, cap_key, messages, **kw)

    print("=" * 68)
    print("PROBE 1 -- does it re-seed per request?")
    print("=" * 68)
    p = [{"role": "user", "content":
          "Invent a short, unusual name for a coastal town. Reply with the name only."}]
    outs = []
    for i in range(3):
        r = q(p, cap=24)
        t = (r.get("text") or r.get("error") or str(r.get("http")))[:90].replace("\n", " ")
        outs.append(r.get("text"))
        print("  run%d (%.1fs): %r" % (i + 1, r["wall"], t))
    v = seed_verdict(outs)
    if v:
        print("  -> " + v)

    print()
    print("=" * 68)
    print("PROBE 2 -- what happens past the %d-token window (dialog.context.size)?" % window)
    print("=" * 68)
    for depth in probe_depths(window):
        msg = [{"role": "user", "content": prompt_at(tok, depth) + "\n\nSummarise in five words."}]
        r = q(msg, cap=32, timeout=1200)
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
    count = [{"role": "user", "content":
              "Count: one, two, three, four, five, six, seven, eight."}]
    base_row = q(count, cap=64)
    stop_row = q(count, cap=64, extra={"stop": ["four"]})
    for line in stop_lines(base_row, stop_row):
        print(line)


if __name__ == "__main__":
    main()
