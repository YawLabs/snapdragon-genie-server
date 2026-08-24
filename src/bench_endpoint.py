#!/usr/bin/env python3
"""Measure prefill and decode against a RUNNING genie_server, over HTTP.

Why this exists as a committed script rather than an inline one-off: the
numbers it produces are the numbers the docs quote, and every one of them had
previously been measured by a throwaway script that was then thrown away -- so
the same measurement got rebuilt (and occasionally rebuilt WRONG) on each pass.
A benchmark whose method is not in the repo is a number nobody can check.

It deliberately talks to the HTTP endpoint, not to GenieEngine directly. The
endpoint is what a client actually experiences, and going through it keeps this
runnable against any OpenAI-compatible server -- so the NPU bundle and a
llama-server GPU/CPU leg can be compared with ONE tool on the SAME prompts,
which is exactly what the existing cross-engine table is missing.

  python src/bench_endpoint.py                         # default suite
  python src/bench_endpoint.py --base http://127.0.0.1:8080
  python src/bench_endpoint.py --decode-only --tokens 200

Pure stdlib, like the server. Needs a server already up; it starts nothing.

Method, and its limits:
  * DECODE is timed as the delta between two otherwise identical requests, one
    capped at 1 token and one at N. Subtracting cancels prefill, connection
    setup and template rendering, which a naive total/tokens figure folds into
    the rate and understates decode by a lot on short runs.
  * PREFILL is a request capped at 1 token, so wall time is prefill plus one
    decode step. At 500+ prompt tokens that single step is in the noise.
  * Thinking is disabled. A <think> block is real generated output but its
    length swings run to run, which makes it a variance source rather than a
    signal when what you want is tokens/sec.
  * Every measurement is preceded by a warmup request, because the first query
    against a freshly loaded dialog pays page-in costs that are not
    representative of steady state.

Read the results next to `docs/GENIE_SERVER.md`: on this engine throughput is
set by the window the BUNDLE WAS COMPILED AT, so a run is only comparable to
another run at the same /props n_ctx.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

# Roughly 4 characters per token for ordinary English prose. Only used to hit a
# target prompt SIZE -- the reported token counts are always the server's own,
# never this estimate.
CHARS_PER_TOKEN = 4
FILLER = "The quick brown fox jumps over the lazy dog near the riverbank. "


def _post(base, path, payload, timeout):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0


def _get(base, path, timeout=15):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def n_ctx(base):
    """The served window, from /props. None if the server does not offer it.

    Reported rather than assumed: it is the single variable that most changes
    what these numbers mean, and hardcoding it is how a result gets filed under
    the wrong bundle.
    """
    try:
        p = _get(base, "/props")
        return int(p["default_generation_settings"]["n_ctx"])
    except Exception:
        return None


def chat(base, model, prompt, max_tokens, timeout):
    body, wall = _post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Three spellings so this works against servers that honour any one of
        # them; a server that knows none just returns its default behaviour.
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }, timeout)
    usage = body.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall": wall,
    }


def prompt_of(target_tokens):
    """A prompt of about `target_tokens`, ending in a real instruction."""
    n = max(1, target_tokens) * CHARS_PER_TOKEN
    reps = n // len(FILLER) + 1
    return (FILLER * reps)[:n] + "\n\nSummarise the text above."


def measure_prefill(base, model, target, timeout):
    r = chat(base, model, prompt_of(target), 1, timeout)
    rate = r["prompt_tokens"] / r["wall"] if r["wall"] else 0.0
    print("  prefill   prompt=%-6d wall=%7.2fs   %8.1f tok/s"
          % (r["prompt_tokens"], r["wall"], rate), flush=True)
    return rate


def measure_decode(base, model, target, tokens, timeout):
    """Decode rate at a given context depth, prefill subtracted out."""
    p = prompt_of(target)
    short = chat(base, model, p, 1, timeout)
    long_ = chat(base, model, p, tokens + 1, timeout)
    steps = long_["completion_tokens"] - short["completion_tokens"]
    delta = long_["wall"] - short["wall"]
    if steps <= 0 or delta <= 0:
        # The model stopped early (hit EOS before the cap), so there is no
        # clean decode window to measure. Say so rather than printing a rate
        # derived from one or two tokens.
        print("  decode    depth=%-6d SKIPPED (model stopped after %d token(s))"
              % (long_["prompt_tokens"], max(0, steps)), flush=True)
        return None
    rate = steps / delta
    print("  decode    depth=%-6d %3d tokens in %6.2fs   %8.2f tok/s"
          % (long_["prompt_tokens"], steps, delta, rate), flush=True)
    return rate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8123",
                    help="server base URL (default %(default)s)")
    ap.add_argument("--model", default="qwen3-4b-npu")
    ap.add_argument("--tokens", type=int, default=200,
                    help="tokens to generate per decode measurement")
    ap.add_argument("--repeat", type=int, default=3,
                    help="decode repetitions at the shallow depth")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    try:
        _get(base, "/health")
    except Exception as e:
        sys.exit("no server at %s (%s) -- start genie_server.py first" % (base, e))

    ctx = n_ctx(base)
    print("endpoint %s   model=%s   n_ctx=%s"
          % (base, args.model, ctx if ctx else "unknown"), flush=True)
    if ctx is None:
        print("  (no /props -- results cannot be attributed to a window; on Genie",
              flush=True)
        print("   the compiled window sets throughput, so record it by hand)", flush=True)

    # Depths to probe. Kept inside the window with room for the generation, and
    # spread wide enough to show whether the curve is flat -- flatness is the
    # whole signature of a statically-shaped KV, so two clustered depths would
    # hide the finding rather than reveal it.
    limit = (ctx or 4096) - args.tokens - 256
    depths = [d for d in (500, 1500, 3000, 7000, 12000) if d < limit] or [min(500, max(1, limit))]

    print("\nwarmup", flush=True)
    chat(base, args.model, "Count from 1 to 5.", 24, args.timeout)

    if not args.decode_only:
        print("\nPREFILL (1-token cap; wall is prefill + one decode step)", flush=True)
        for d in depths:
            measure_prefill(base, args.model, d, args.timeout)

    if not args.prefill_only:
        print("\nDECODE (delta of N-token vs 1-token run at the same depth)", flush=True)
        rates = []
        for _ in range(max(1, args.repeat)):
            r = measure_decode(base, args.model, depths[0], args.tokens, args.timeout)
            if r:
                rates.append(r)
        if len(depths) > 1:
            # The deep sample is the point: if decode here matches decode at the
            # shallow depth, cost is set by the compiled window rather than by
            # how much context is actually resident.
            r = measure_decode(base, args.model, depths[-1], args.tokens, args.timeout)
            if r:
                rates.append(r)
        if rates:
            print("\n  decode median %.2f t/s over %d run(s)"
                  % (statistics.median(rates), len(rates)), flush=True)
            if len(depths) > 1 and len(rates) >= 2:
                spread = max(rates) - min(rates)
                print("  spread across depths %.2f t/s -- %s"
                      % (spread,
                         "flat, so cost tracks the COMPILED window, not the used context"
                         if spread < 0.25 * statistics.median(rates)
                         else "varies with depth on this engine"), flush=True)


if __name__ == "__main__":
    main()
