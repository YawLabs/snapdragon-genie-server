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
  python src/bench_endpoint.py --depths 250,3300,250,3300 --decode-every

The last form is worth knowing. A sweep that runs shallow-to-deep IN ORDER
cannot tell a real depth effect from the box drifting downward over the run --
both look like "slower at depth". Repeating depths in the list interleaves
them, so a genuine depth effect tracks the depth while drift shows up as a
monotonic slide regardless of it. That distinction has already overturned one
finding here.

Pure stdlib, like the server. Needs a server already up; it starts nothing.

Method, and its limits:
  * DECODE is timed as the delta between two otherwise identical requests, one
    capped at 1 token and one at N. Subtracting cancels prefill, connection
    setup and template rendering, which a naive total/tokens figure folds into
    the rate and understates decode by a lot on short runs. (Both requests
    really do pay full prefill: the server reuses its KV only when a prompt
    EXTENDS what the dialog holds, and the second request's prompt is a strict
    prefix of the first result, so it resets. If that ever changes, this
    subtraction quietly becomes wrong.)
  * PREFILL is a request capped at 1 token, so raw wall time is prefill PLUS
    one decode step. That step is not negligible at shallow depths -- on a
    slow-decoding bundle it was ~11% of a 469-token measurement -- so a cheap
    decode probe runs first and its per-step cost is subtracted. The probe's
    own figure and the raw wall are both printed, so the correction is visible
    rather than taken on trust.
  * Thinking is disabled. A <think> block is real generated output but its
    length swings run to run, which makes it a variance source rather than a
    signal when what you want is tokens/sec.
  * Every measurement is preceded by a warmup request, because the first query
    against a freshly loaded dialog pays page-in costs that are not
    representative of steady state.
  * A failed request (429 backpressure, a 400, a dropped connection) skips that
    data point and the sweep carries on. Losing a twenty-minute run to one
    transient 429 would be worse than a gap in the table.
  * CHECK THE BOX FOR CO-TENANTS FIRST. This tool cannot see them and will
    happily report a contended number as a clean one. On a shared machine that
    is not hypothetical: a batch measured here was invalidated by another
    session's llama-bench running concurrently, and one of ITS investigations
    was in turn invalidated by a resident genie_server busy-waiting on 2.7
    cores. Before a batch, confirm nothing else is loading the box, and record
    the wall-clock window with the numbers so an overlap can be reconstructed
    later instead of argued about.

Read the results next to `docs/GENIE_SERVER.md`: on this engine throughput is
set by the window the BUNDLE WAS COMPILED AT, so a run is only comparable to
another run at the same /props n_ctx.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# Roughly 4 characters per token for ordinary English prose. Only used to hit a
# target prompt SIZE -- the reported token counts are always the server's own,
# never this estimate.
CHARS_PER_TOKEN = 4
FILLER = "The quick brown fox jumps over the lazy dog near the riverbank. "


def _describe(err):
    """A one-line reason from a failed request, including the server's message.

    The server explains itself in the body ("server busy; NPU is single-flight",
    or a 400 naming the token counts). Printing only the status code throws that
    away at exactly the moment the reader needs it.
    """
    if isinstance(err, urllib.error.HTTPError):
        detail = ""
        try:
            detail = (json.loads(err.read()).get("error") or {}).get("message") or ""
        except Exception:
            pass
        return "HTTP %s%s" % (err.code, " -- %s" % detail if detail else "")
    return "%s: %s" % (type(err).__name__, err)


def _post(base, path, payload, timeout):
    """POST and time it. Returns (body, wall), or (None, reason) on failure."""
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
    except Exception as e:
        return None, _describe(e)
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
    """One completion. Returns a dict, or None after printing why it failed."""
    body, wall = _post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Three spellings so this works against servers that honour any one of
        # them; a server that knows none just returns its default behaviour.
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }, timeout)
    if body is None:
        print("  request failed (%s) -- skipping this point" % wall, flush=True)
        return None
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


def _delta_run(base, model, prompt, extra, timeout):
    """(one, many, steps, seconds) for `extra` more decode steps at this depth.

    Runs the same prompt capped at 1 token and at 1+extra. Everything that is
    not decode -- prefill, connection setup, template rendering -- occurs in
    both and cancels in the difference. Returns None if either request failed;
    steps==0 means the model hit EOS before the cap, so there is no window.
    """
    one = chat(base, model, prompt, 1, timeout)
    if one is None:
        return None
    many = chat(base, model, prompt, 1 + extra, timeout)
    if many is None:
        return None
    steps = many["completion_tokens"] - one["completion_tokens"]
    secs = many["wall"] - one["wall"]
    if steps <= 0 or secs <= 0:
        return (one, many, 0, 0.0)
    return (one, many, steps, secs)


def decode_probe(base, model, timeout, depth=500, steps=8):
    """Seconds per decode step, used to correct the prefill measurements.

    Cheap on purpose (a handful of tokens at a shallow depth). Decode on this
    engine is close to flat with depth, so one figure corrects the whole prefill
    sweep; where it is not flat the correction is small anyway, and the raw wall
    time is printed alongside so nothing is hidden.
    """
    r = _delta_run(base, model, prompt_of(depth), steps, timeout)
    if r is None or r[2] <= 0:
        return None
    return r[3] / r[2]


# A decode step legitimately costs 10-20% of the shortest prefill measurement
# here. Anything at or past half the sample is not a plausible correction, it is
# a probe that was taken while something else had the box -- and subtracting it
# silently inflates the result. Measured case: a probe of 1.546 s/token against
# a true 0.127 turned a real ~206 tok/s into a reported 549.
PROBE_MAX_SHARE = 0.5


def _probe_crosscheck(before, after, tol=1.5):
    """What the opening and closing decode probes say about the box holding.

    Split out of main() so it can be tested: it is a WARNING, and a warning
    that silently stops firing is worse than none -- the run then looks clean
    precisely when it is not. Both arguments are seconds per step; `after` is
    falsy when the closing probe failed.
    """
    if not after:
        return ("NOTE: the closing probe failed, so the corrections above "
                "could not be cross-checked.")
    a, b = 1.0 / before, 1.0 / after
    if max(a, b) / min(a, b) > tol:
        return ("WARNING: the decode probe read %.2f t/s before the sweep and "
                "%.2f t/s after it -- the box did not hold, so the corrections "
                "above are unreliable. Re-run quiet." % (a, b))
    return ("probe before/after: %.2f / %.2f t/s -- consistent, so the "
            "corrections above stand." % (a, b))


def measure_prefill(base, model, target, timeout, per_step=0.0):
    r = chat(base, model, prompt_of(target), 1, timeout)
    if r is None:
        return None
    raw = r["wall"]
    # The 1-token cap still generates one token, so raw wall is prefill plus a
    # step. Removing it is right -- but only when the step is credible against
    # THIS sample. Refusing loudly beats dividing by a floor: with per_step >=
    # raw the old code produced 4.69e+11 tok/s, which is not a number anyone
    # would notice was wrong in a table.
    if per_step and per_step >= PROBE_MAX_SHARE * raw:
        print("  prefill   prompt=%-6d wall=%7.2fs (RAW, uncorrected)  %8.1f tok/s"
              "   <- probe %.3f s/tok is >=%.0f%% of this sample; not subtracted"
              % (r["prompt_tokens"], raw, r["prompt_tokens"] / raw,
                 per_step, PROBE_MAX_SHARE * 100), flush=True)
        return r["prompt_tokens"] / raw
    wall = raw - per_step if per_step else raw
    rate = r["prompt_tokens"] / wall
    print("  prefill   prompt=%-6d wall=%7.2fs (raw %6.2fs)  %8.1f tok/s"
          % (r["prompt_tokens"], wall, raw, rate), flush=True)
    return rate


def measure_decode(base, model, target, tokens, timeout):
    """Decode rate at a given context depth, prefill subtracted out."""
    r = _delta_run(base, model, prompt_of(target), tokens, timeout)
    if r is None:
        return None
    _, many, steps, secs = r
    if steps <= 0:
        # The model stopped early (hit EOS before the cap), so there is no
        # clean decode window to measure. Say so rather than printing a rate
        # derived from one or two tokens.
        print("  decode    depth=%-6d SKIPPED (model stopped before the cap)"
              % many["prompt_tokens"], flush=True)
        return None
    rate = steps / secs
    print("  decode    depth=%-6d %3d tokens in %6.2fs   %8.2f tok/s"
          % (many["prompt_tokens"], steps, secs, rate), flush=True)
    return rate


def _verdict(shallow, deep):
    """Compare decode at two depths, keeping same-depth noise out of the claim.

    The point of this line is whether cost tracks the COMPILED window or the
    context actually in use -- a statement about the difference BETWEEN depths.
    Pooling both groups and taking max-minus-min folds run-to-run noise at one
    depth into that difference, which is enough to flip the verdict on a noisy
    box while the depths genuinely agree. So compare medians, and report the
    noise as its own number instead of letting it masquerade as a depth effect.
    """
    if not shallow or not deep:
        return
    ms, md = statistics.median(shallow), statistics.median(deep)
    delta = abs(ms - md)
    noise = max(max(shallow) - min(shallow), max(deep) - min(deep))
    print("  shallow median %.2f t/s (n=%d)   deep median %.2f t/s (n=%d)"
          % (ms, len(shallow), md, len(deep)), flush=True)
    print("  cross-depth delta %.2f t/s, same-depth noise %.2f t/s -- %s"
          % (delta, noise,
             "flat, so cost tracks the COMPILED window, not the used context"
             if delta < 0.25 * min(ms, md)
             else "varies with depth on this engine"), flush=True)
    if len(deep) == 1:
        print("  (deep depth sampled once -- --repeat-deep raises that)", flush=True)


DEFAULT_DEPTHS = (500, 1500, 3000, 7000, 12000)


def resolve_depths(spec, limit, ctx, tokens):
    """Depths to probe, given the caller's --depths and the token budget.

    Lifted out of main() so it can be tested without standing up a server: the
    health check runs first, so a bad --depths was previously unreachable from
    any test and only discoverable by a user hitting a traceback.

    Raises ValueError with a message fit to show a user; returns (depths, note)
    where note is a line to print or None. Depths past the budget are DROPPED
    and named -- a silently shortened sweep reads as "measured everything".
    """
    if not spec:
        return ([d for d in DEFAULT_DEPTHS if d < limit] or [min(500, limit)]), None
    try:
        asked = [int(d) for d in spec.split(",") if d.strip()]
    except ValueError as e:
        raise ValueError(
            "--depths wants comma-separated integers (%s)" % e) from e
    depths = [d for d in asked if 0 < d < limit]
    dropped = [d for d in asked if d not in depths]
    if not depths:
        raise ValueError("every requested depth exceeds the budget of %d tokens"
                         % limit)
    note = None
    if dropped:
        note = ("  note: dropped depth(s) %s -- past the %d-token budget "
                "(n_ctx %s minus --tokens %d minus margin)"
                % (", ".join(str(d) for d in dropped), limit, ctx, tokens))
    return depths, note


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
    ap.add_argument("--repeat-deep", type=int, default=1,
                    help="decode repetitions at the deep depth; each costs two "
                         "full deep prefills, hence the lower default")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--depths",
                    help="comma-separated prompt depths to probe, replacing the "
                         "default sweep. Depths that do not leave room for "
                         "--tokens are dropped with a note rather than silently.")
    ap.add_argument("--decode-every", action="store_true",
                    help="measure decode at EVERY depth instead of just the "
                         "shallowest and deepest, and print a per-depth table. "
                         "Use this to look for STEP changes: a bundle built at "
                         "several --context-lengths appears to carry one graph "
                         "per length and pick the smallest that fits, which "
                         "would show up as plateaus rather than a smooth slope.")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    try:
        _get(base, "/health")
    except Exception as e:
        sys.exit("no server at %s (%s) -- start genie_server.py first"
                 % (base, _describe(e)))

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
    if limit < 1:
        sys.exit("--tokens %d leaves no room in an n_ctx=%s window -- lower it"
                 % (args.tokens, ctx))
    try:
        depths, note = resolve_depths(args.depths, limit, ctx, args.tokens)
    except ValueError as e:
        sys.exit(str(e))
    if note:
        print(note, flush=True)

    print("\nwarmup", flush=True)
    chat(base, args.model, "Count from 1 to 5.", 24, args.timeout)

    probe_rate = None
    if not args.decode_only:
        per_step = decode_probe(base, args.model, args.timeout,
                                depth=min(500, depths[0]))
        probe_rate = (1.0 / per_step) if per_step else None
        if per_step:
            print("\ndecode probe: %.3f s/token (%.2f t/s), subtracted from each "
                  "prefill below" % (per_step, 1.0 / per_step), flush=True)
        else:
            per_step = 0.0
            print("\ndecode probe failed; prefill figures still include one "
                  "decode step and therefore read LOW", flush=True)
        print("\nPREFILL (one decode step removed; raw wall also shown)", flush=True)
        for d in depths:
            measure_prefill(base, args.model, d, args.timeout, per_step)
        if args.prefill_only and per_step:
            # With --prefill-only there is no decode phase to check the probe
            # against, and the probe is the ONLY thing shaping these numbers --
            # one taken during a blip corrupts every figure above silently.
            # Re-probe at the end: the two bracket the sweep, so agreement
            # means the box held throughout it.
            again = decode_probe(base, args.model, args.timeout,
                                 depth=min(500, depths[0]))
            print("\n  %s" % _probe_crosscheck(per_step, again), flush=True)

    if not args.prefill_only:
        print("\nDECODE (delta of N-token vs 1-token run at the same depth)", flush=True)

        def at(depth, reps):
            out = []
            for _ in range(max(1, reps)):
                r = measure_decode(base, args.model, depth, args.tokens, args.timeout)
                if r:
                    out.append(r)
            return out

        per_depth = []
        if args.decode_every:
            for d in depths:
                per_depth.append((d, at(d, args.repeat)))
        else:
            per_depth.append((depths[0], at(depths[0], args.repeat)))
            if len(depths) > 1:
                # The deep sample is the point: if decode here matches decode at
                # the shallow depth, cost is set by the compiled window rather
                # than by how much context is actually resident.
                per_depth.append((depths[-1], at(depths[-1], args.repeat_deep)))

        allr = [r for _, rs in per_depth for r in rs]
        if allr:
            print("\n  decode median %.2f t/s over %d run(s)"
                  % (statistics.median(allr), len(allr)), flush=True)
            # The prefill figures above were corrected using the probe's
            # per-step cost. Now that real decode rates exist, say whether the
            # probe agreed with them -- a probe taken during a blip corrupts
            # every prefill number in the run, and nothing else would reveal it.
            if probe_rate:
                med = statistics.median(allr)
                if med and (med / probe_rate > 2 or probe_rate / med > 2):
                    print("  WARNING: the decode probe read %.2f t/s but decode "
                          "measured %.2f t/s -- the probe was not representative, "
                          "so treat the corrected prefill figures above as "
                          "unreliable and re-run on a quiet box."
                          % (probe_rate, med), flush=True)
        if args.decode_every and len([1 for _, rs in per_depth if rs]) > 2:
            print("\n  per-depth medians (look for PLATEAUS, not a smooth slope):",
                  flush=True)
            prev = None
            for d, rs in per_depth:
                if not rs:
                    continue
                m = statistics.median(rs)
                # Flag the jumps rather than making the reader diff the column.
                mark = ""
                if prev is not None:
                    change = (m - prev) / prev * 100
                    mark = "  %+5.1f%%%s" % (change, "  <-- step" if abs(change) >= 8 else "")
                print("    depth %-6d %6.2f t/s%s" % (d, m, mark), flush=True)
                prev = m
        first = next((rs for _, rs in per_depth if rs), [])
        last = next((rs for _, rs in reversed(per_depth) if rs), [])
        if first is not last:
            _verdict(first, last)


if __name__ == "__main__":
    main()
