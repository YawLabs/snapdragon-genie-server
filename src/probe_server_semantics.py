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
    python src/probe_server_semantics.py -h

The three positionals point it at a server. The defaults are geniex serve's:
  BASE   the server's base URL, http:// or https:// -- default http://127.0.0.1:18181
  MODEL  the model id to ask for                    -- default qualcomm/qwen3-4b-ours
  CAP    the request key the completion cap goes in -- default max_completion_tokens
For this repo's server pass http://127.0.0.1:8123 qwen3-4b-npu max_tokens. CAP
is the spelling of the completion cap the server honours -- geniex ignores the
legacy `max_tokens` outright (measured in bench_servers.py). GENIE_BUNDLE_DIR
must be the bundle the server is serving: its tokenizer sizes the prompts and
its genie_config.json sets PROBE 2's window.

Before PROBE 1 it asks BASE for /v1/models once. If nothing takes that
connection -- nothing listening on the port, a host that does not resolve --
it exits 1 naming the URL it tried, instead of spending ~20 s on eight rows
that all say the same refused connect and then exiting 0. Any answer at all,
an HTTP error included, is enough to go on: what the server does with a chat
completion is what the probes are for. A BASE that is not an http(s) URL, a
fourth positional and an unknown option are refused by name before anything
is loaded; -h, --help and /? print this and exit 0, before the bundle is
looked for.

Standing the servers up to point this at -- the geniex import, and the four
undocumented steps GenieAPIService needs -- is in "Reproducing the cross-server
comparison" in docs/GENIE_SERVER.md. Needs `pip install tokenizers`.
"""
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from prompt_depth import load_tokenizer, prompt_at

DEFAULT_BASE = "http://127.0.0.1:18181"
DEFAULT_MODEL = "qualcomm/qwen3-4b-ours"
DEFAULT_CAP = "max_completion_tokens"

# genie_smoke's spellings, so the two tools answer the same request for help.
HELP_FLAGS = ("-h", "--help", "/?")

# How long the one preflight request may take. It only has to find out whether
# anything takes the connection; a refused connect on Windows loopback comes
# back in about 2 s, and a server that is merely slow to answer is let through.
PREFLIGHT_SECS = 10

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
    here and ended the run mid-way. So did the Request itself: it was built
    before the try, and a base with no scheme (`--help` taken as BASE, or
    `127.0.0.1:8123`) raised ValueError("unknown url type") out of here as a
    16-line traceback. main() now refuses such a base by name first; this is
    the promise kept for any other caller.

    `text` is a str wherever it is present, which is what the printers assume
    (they slice it and call .replace). A message whose `content` is
    Anthropic-style content blocks is an error row for that reason, the same
    verdict this already gives a `message` that is not an object at all.
    """
    body = {"model": model, "messages": messages, cap_key: cap, "stream": False}
    if extra:
        body.update(extra)
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
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


def args_problem(args):
    """Why the positionals cannot be run, as a sentence, or None.

    `args` is argv without the program name, help flags already handled. The
    positionals used to be taken as they came: `--help` became BASE, and a
    fourth word was dropped without a mention.
    """
    for a in args:
        if a.startswith("-"):
            return ("unknown option %r: this tool takes only [BASE] [MODEL] "
                    "[CAP] -- -h for the usage" % a)
    if len(args) > 3:
        return ("%d arguments, and the usage is [BASE] [MODEL] [CAP]: %s would "
                "be ignored -- -h for the usage" % (len(args), " ".join(args[3:])))
    return base_problem(args[0]) if args else None


def base_problem(base):
    """Why `base` cannot be a server's base URL, naming it, or None.

    A scheme-less `127.0.0.1:8123` or `localhost:8123` reached urllib as an
    "unknown url type" on every one of the eight requests, and exited 0.
    """
    parts = urllib.parse.urlsplit(base)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return ("BASE %r is not an http(s) URL -- give the server's base as a "
                "client reaches it, e.g. http://127.0.0.1:8123 (this repo's "
                "server) or %s (geniex serve, the default). -h for the usage"
                % (base, DEFAULT_BASE))
    try:
        parts.port  # noqa: B018 -- reading it is the check: it raises on a bad port
    except ValueError as e:
        return "BASE %r has a port that is not a port number (%s)" % (base, e)
    return None


def nothing_listening(err):
    """True when the CONNECT was refused, i.e. nothing holds that port --
    genie_smoke.nothing_listening, for the same reason: only then is "try
    the other port" the right advice."""
    return isinstance(err, ConnectionRefusedError) or isinstance(
        getattr(err, "reason", None), ConnectionRefusedError)


def preflight(base, timeout=PREFLIGHT_SECS):
    """None if anything takes a connection at `base`, else why not, naming
    the URL tried.

    One GET of /v1/models before the three probes spend their eight
    requests. Against a dead port those came back as eight identical refused
    rows over ~20 s -- the URL printed nowhere, each row cut off before the
    word "refused", and exit 0. Only a failure to CONNECT stops the run:
    urllib raises URLError for everything up to and including sending the
    request, so that is "nothing is there". An HTTPError is an answer, and a
    timeout, reset or non-HTTP peer after the connect is a listener that took
    it -- the probes' own rows say what it then does, which is their job.
    """
    url = base + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return None
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError as e:
        why = "cannot reach %s (%s) -- nothing was probed" % (url, e.reason)
        if nothing_listening(e):
            why += (". Nothing is listening at %s. geniex serve is on 18181 (the "
                    "default BASE); this repo's server is on 8123 when started by "
                    "run-genie-server.ps1 -- pass its base, model and cap key, e.g. "
                    "http://127.0.0.1:8123 qwen3-4b-npu max_tokens" % base)
        return why
    except (OSError, ValueError, http.client.HTTPException):
        return None


def main(argv=None):
    argv = sys.argv if argv is None else argv
    args = argv[1:]
    if any(a in HELP_FLAGS for a in args):
        # Before the env read and the tokenizer load, both of which exit: the
        # usage in the docstring was unreachable, and with GENIE_BUNDLE_DIR set
        # `--help` became BASE and ended in a traceback. -OO strips __doc__.
        print((__doc__ or "Usage: probe_server_semantics.py [BASE] [MODEL] [CAP]").strip())
        return 0
    problem = args_problem(args)
    if problem:
        sys.exit(problem)
    base = args[0] if len(args) > 0 else DEFAULT_BASE
    model = args[1] if len(args) > 1 else DEFAULT_MODEL
    cap_key = args[2] if len(args) > 2 else DEFAULT_CAP
    # Same contract as bench_servers.py: the bundle location is an env var,
    # not a path baked into the file. This script used to pin one machine's
    # absolute path, which meant it crashed on import for everyone except its
    # author -- on a script the README cites as the instrument behind a whole
    # findings table. load_tokenizer names the path on every way that can
    # still go wrong.
    bundle = os.environ.get("GENIE_BUNDLE_DIR", "")
    tok = load_tokenizer(bundle)
    window = read_window(bundle)
    # After the bundle, which is local and names its own problems; before the
    # probes, which would each report the same missing server.
    problem = preflight(base)
    if problem:
        sys.exit(problem)

    def q(messages, **kw):
        return ask(base, model, cap_key, messages, **kw)

    # Which server this is about, once, where the output starts: the base URL
    # used to appear nowhere in it.
    print("probing %s -- model %s, cap sent as %s" % (base, model, cap_key))
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
