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
    really do pay full prefill: genie_server reuses its KV only when a prompt
    EXTENDS what the dialog holds, and the second request's prompt is a strict
    prefix of the first result, so it resets. llama-server would reuse the
    cached prefix by default -- its `cache_prompt` is on unless told otherwise
    -- so every request here sends `cache_prompt: false` explicitly, which
    genie_server ignores as an unknown key. If either server ever changes,
    this subtraction quietly becomes wrong.)
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
    representative of steady state. The warmup's ANSWER is also the only
    pre-flight that looks at what the server generates: an ignored output cap,
    an empty completion or a missing usage block ends the run right there,
    because each of them would otherwise produce a table that looks measured.
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
import http.client
import json
import os
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


def _int_env(name, default):
    """An int from the environment, or `default` with a line saying why not.

    A copy of genie_server's helper of the same name, not an import of it:
    pulling the server module into a bench tool would run its engine plumbing
    at import. The reason it exists is the same one recorded there -- a typo in
    an env var used to kill the process at IMPORT with a bare `invalid literal
    for int()` naming neither the variable nor the form it wanted, and because
    bench_contention imports this module at scope, the same typo killed that
    tool too.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("[bench] WARNING: %s=%r is not an integer; using %r instead."
              % (name, raw, default), flush=True)
        return default


# --------------------------------------------------------------------------
# Box state, recorded alongside every number this tool produces.
#
# This module had NO instrumentation at all -- no power, no clock -- while
# producing every prefill and decode figure the docs quote. Its own docstring
# told the OPERATOR to "record the wall-clock window with the numbers"; it
# identified the need and then delegated it to whoever remembered.
#
# The cost of that is not hypothetical and was watched happening on this box: a
# neighbouring project's headline table can no longer be explained, because the
# CSVs behind it recorded clock and no power. Its deltas track CPU clock, and
# WHY the clock was lower is now permanently unknowable. A number without its
# box state is not reproducible, and nobody can go back and add the state later.
# --------------------------------------------------------------------------

# Four fields, one PowerShell launch. A reading that is not available prints as
# an EMPTY field, never as a number: `[math]::Round($null, 1)` is 0, so before
# the guards a failed Get-Counter read came back as "clock 0.0% of base" and
# tripped the low-clock WARNING on a healthy box -- a fabricated sample, which
# the summary cannot tell from a real one. Empty parses to None below, and None
# is "not read", which is the truth.
#
# The two rounded doubles are turned into text with the INVARIANT culture
# before `-f` sees them. `-f` formats a double in the session's current
# culture, so on a comma-decimal Windows (de-DE, fr-FR, ...) 79.2 printed as
# "79,2": the row grew a fifth comma field, box_state_fields() read that as a
# failed query, and every consumer went blind at once -- including
# bench_contention's cool gate, whose on-battery abort reads its power source
# through power_reading() and so never fired (measured with CurrentCulture set
# to de-DE: "True,100,0,79,2"). A string argument passes through `-f`
# untouched, PowerOnline prints "True"/"False" and the charge is an integer in
# every culture, so these two were the only culture-dependent fields.
_INVARIANT = ".ToString([cultureinfo]::InvariantCulture)"
_STATE_PS = (
    "$b = Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus "
    "-ErrorAction SilentlyContinue | Select-Object -First 1; "
    "$c = (Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | "
    "Select-Object -First 1).EstimatedChargeRemaining; "
    "$k = (Get-Counter '\\Processor Information(_Total)\\% Processor "
    "Performance' -ErrorAction SilentlyContinue).CounterSamples.CookedValue; "
    "'{0},{1},{2},{3}' -f $b.PowerOnline, $c, "
    "$(if ($null -eq $b) {''} else "
    "{[math]::Round($b.ChargeRate/1000,1)" + _INVARIANT + "}), "
    "$(if ($null -eq $k) {''} else {[math]::Round($k,1)" + _INVARIANT + "})"
)


def box_state_fields():
    """The sampler's raw fields [power_online, charge_pct, charge_w, clock_pct].

    Strings exactly as PowerShell printed them, with "" for a field the box
    could not supply (no battery class, a counter that failed to read). None
    means the QUERY failed -- off-Windows, a non-zero exit, a timeout -- which
    is a different fact from "the class reported nothing", and the two are
    kept apart so that a transient PowerShell failure cannot masquerade as a
    desktop with no battery. One subprocess for all four, because this sits
    between measurements and the alternative is three launches per sample.
    """
    if sys.platform != "win32":
        return None
    import subprocess
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                              _STATE_PS],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        fields = [f.strip() for f in out.stdout.strip().split(",")]
        return fields if len(fields) == 4 else None
    except Exception:
        return None


def _parse_box_state(fields):
    """(on_ac, charge_pct, charge_watts, clock_pct_of_base) from the raw fields.

    Pure, so the parse can be tested without a device: an empty field is None,
    a numeric one is a float, and the AC flag is a bool only when the class
    actually said "True" or "False".
    """
    if fields is None:
        return None, None, None, None
    try:
        ac, pct, watts, clock = fields

        def _f(v):
            return float(v) if v.strip() else None

        return (ac.strip().lower() == "true" if ac.strip() else None,
                _f(pct), _f(watts), _f(clock))
    except Exception:
        return None, None, None, None


def box_state():
    """(on_ac, charge_pct, charge_watts, clock_pct_of_base). Nones on failure."""
    return _parse_box_state(box_state_fields())


def power_reading():
    """(power_online_raw, charge_pct, charge_watts) from ONE launch.

    The first element is BatteryStatus.PowerOnline exactly as PowerShell
    printed it: "True"/"False", "" without a battery class, None when the
    query itself failed. It is handed over RAW because those last two are
    different facts -- a transient PowerShell failure is not a desktop with no
    battery -- and the parsed bool in box_state() collapses them. The other two
    are the parsed pack figures, None where unread.

    Public because bench_contention's cool gate derives its power source and
    its low-pack advisory from it, instead of carrying a second copy of the WMI
    query -- two copies of the query string is two places for it to drift. One
    call for all three because the gate used to take them as two
    (power_online_raw(), then battery_state()): two launches of the same
    four-field script, the first discarding the pack figures the second went
    back for.

    Magnitudes, never a boolean. Two sessions on this box independently built a
    "charging suspended" flag and both were wrong at their cut point: a <=1 W
    test scored 0/4 on legs that shed 92% and 97% of their draw, and a
    25%-of-opening test scored 2/4 on the same legs. Every threshold mis-sorts
    the legs nearest it, so record the quantity and let a reader draw the line.
    """
    fields = box_state_fields()
    _ac, pct, watts, _clock = _parse_box_state(fields)
    return (None if fields is None else fields[0]), pct, watts


# One reading per measurement, so the run's artifact carries the box's
# trajectory rather than a verdict about it.
BOX_SAMPLES = []


def note_box_state(label):
    """Record the box state for one measurement. Never raises."""
    ac, pct, watts, clock = box_state()
    if pct is None and clock is None:
        return
    BOX_SAMPLES.append({"label": label, "on_ac": ac, "charge_pct": pct,
                        "charge_w": watts, "clock_pct": clock})


def box_state_summary():
    """Lines describing what the box did across the run, or [] if unsampled.

    Reports RANGES rather than a pass/fail, for the same reason power_reading
    returns magnitudes: the reader picks the line. A charging pack below ~25%
    halves prefill on this box (13-20% gives CPU pp512 ~58 against a settled
    130) while decode barely moves, so the same run can be sound for one figure
    and worthless for the other -- which no single verdict can express.
    """
    if not BOX_SAMPLES:
        return []
    out = []
    charges = [s["charge_pct"] for s in BOX_SAMPLES if s["charge_pct"] is not None]
    clocks = [s["clock_pct"] for s in BOX_SAMPLES if s["clock_pct"] is not None]
    draws = [s["charge_w"] for s in BOX_SAMPLES if s["charge_w"] is not None]
    on_ac = [s["on_ac"] for s in BOX_SAMPLES if s["on_ac"] is not None]
    if charges:
        line = "  box: pack %.0f-%.0f%%" % (min(charges), max(charges))
        if draws:
            line += ", draw %.1f-%.1f W" % (min(draws), max(draws))
        # "for part of the run" is a claim that SOME sample saw AC, so a run
        # that never did gets its own wording rather than the weaker one.
        if on_ac and not any(on_ac):
            line += ", ON BATTERY for the whole run"
        elif on_ac and not all(on_ac):
            line += ", ON BATTERY for part of the run"
        out.append(line)
        if min(charges) < 25:
            out.append("  WARNING: pack reached %.0f%%. Below ~25%% this box "
                       "halves PREFILL (pp512 58 against a settled 130) while "
                       "decode holds -- prefill figures here are not a settled "
                       "baseline even on AC." % min(charges))
    if clocks:
        out.append("  clock: %.0f-%.0f%% of base across %d sample(s)"
                   % (min(clocks), max(clocks), len(clocks)))
        if min(clocks) < 80:
            out.append("  WARNING: clock reached %.0f%% of base. Sampled "
                       "BETWEEN measurements, so it brackets them rather than "
                       "describing what happened during one." % min(clocks))
    return out


def _box_state_trailer(summary, accepted, win32=sys.platform == "win32"):
    """The closing lines about box state, given what the run actually did.

    Three different facts, three different lines. A summary means the box was
    sampled and this is what it did. An empty summary with no accepted
    measurement means there was NOTHING to sample beside -- every point was
    skipped, refused or reported raw -- and saying "could not be sampled" there
    blames the sampler for a run that never called it. Only an empty summary
    with accepted measurements on Windows means the sampler itself failed, and
    that is the one case where the numbers carry no record of their conditions.
    """
    if summary:
        return ["", "BOX STATE (sampled between measurements)", *summary]
    if not accepted:
        return ["", "  (no measurements were accepted, so no box state was "
                    "recorded -- there was nothing to record it beside)"]
    if win32:
        return ["", "  (box state could not be sampled -- these numbers carry "
                    "no record of the power or clock conditions they were "
                    "taken under)"]
    return []


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


def post_timed(base, path, payload, timeout):
    """POST and time it. Returns (body, wall), or (None, reason) on failure.

    `wall` is seconds as a float; `reason` is always a string (see _describe).
    Public because it is the HTTP plumbing bench_contention's load generator
    runs on, and a private name invited changing the failure shape without
    anyone noticing that a second tool unpacks it.

    perf_counter, not time.time(): every figure this tool prints is the
    difference of two of these reads, and the wall clock can step (NTP, a
    resume from sleep) between them. bench.py already uses the monotonic clock
    for the same reason, and bench_servers.py times its requests THROUGH this
    function (it also reuses _content_of, resolve_depths, MIN_DECODE_STEPS and
    box_state), so there is one clock and one failure shape across the tools.
    """
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
    except Exception as e:
        return None, _describe(e)
    return body, time.perf_counter() - t0


# The old name. Nothing in this repo uses it any more (bench_contention and
# bench_servers call post_timed); it is kept only so a script outside the repo
# that imported the private spelling keeps working. New code uses post_timed.
_post = post_timed


def _get(base, path, timeout=15):
    """GET and parse JSON, treating an EMPTY 200 body as an empty object.

    Not every server answers /health with JSON -- some return 200 and no body
    at all. Raising there made the caller report "no server", which is the one
    diagnosis guaranteed to send you hunting a process that is running fine.
    A non-empty body that is not JSON still raises, because that is a real
    surprise worth surfacing.
    """
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}


# A connection that was TAKEN and then broken, rather than one that was never
# taken. http.client.RemoteDisconnected is a ConnectionResetError subclass, so
# a peer that simply hung up without answering is in here too. Every one of
# these means something was on the other end -- and says nothing about whether
# it still is, which is why they get their own line in _health_failure.
_TORN_DOWN = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


def _torn_down_connection(err):
    """True when `err` is a connection something broke, wrapped or bare.

    The same server produces both shapes. urllib re-raises whatever goes
    wrong while SENDING the request wrapped in URLError and whatever goes
    wrong waiting for or reading the response bare, so a listener that resets
    on accept comes out wrapped when the reset beats the send and bare when
    it does not -- measured on a loopback listener that accepts and closes
    with SO_LINGER 0: both shapes, same socket, no rule to the split. Read
    identically here, or one server gets two opposite verdicts.
    """
    if isinstance(err, urllib.error.URLError):
        return isinstance(getattr(err, "reason", None), _TORN_DOWN)
    return isinstance(err, _TORN_DOWN)


def _connected_but_unanswered(err):
    """True when something took the connection and gave no usable answer.

    urllib draws a line of its own: everything up to and including sending
    the request is re-raised wrapped in URLError, while whatever goes wrong
    waiting for or reading the RESPONSE comes out bare. That line is NOT the
    one this function wants. Most of the wrapped side -- a refused connect, a
    connect that timed out, a name that does not resolve -- really is "nothing
    took it", but a reset can land during the send as easily as after it, so
    a wrapped _TORN_DOWN reason is a taken connection too (see
    _torn_down_connection). The bare side is all of it: a bare timeout, a
    bare reset or disconnect, a reply that is not HTTP at all, and a 200 whose
    body _get could not parse. A bare ConnectionRefusedError is kept out for
    a caller that hands one in unwrapped: nothing took that connection. An
    allowlist rather than "not a URLError", because a malformed --base raises
    a plain ValueError before any socket is opened, and that is not a listener
    either (JSONDecodeError and UnicodeDecodeError are ValueErrors too, which
    is why they are named rather than caught by their base).
    """
    if isinstance(err, urllib.error.URLError):
        return _torn_down_connection(err)
    if isinstance(err, ConnectionRefusedError):
        return False
    return isinstance(err, (TimeoutError, ConnectionError,
                            http.client.HTTPException,
                            json.JSONDecodeError, UnicodeDecodeError))


def _health_failure(base, err):
    """The exit line for a /health check that raised.

    Four failures that used to print the same "no server ... start
    genie_server.py first":

    * An HTTP error is a server that IS listening and chose to answer
      (genie_server returns 503 with a `state` and `detail` while its engine is
      failing, stalled or wedged), so it is named as up-but-unwell with the
      server's own words.
    * A connection something ACCEPTED and then BROKE before answering -- a
      reset, an abort, a broken pipe, a peer that hung up (see
      _torn_down_connection) -- is the one shape that says nothing about
      whether the thing is still there: a wedged server resetting its
      connections and a server being torn down are the same symptom from
      this side. So this line names both readings and gives the way to tell
      them apart (re-run: a REFUSED second attempt means it has gone), and it
      deliberately does NOT say "a listener IS there". Wrapped or bare, one
      line: urllib's wrapping of this error depends on whether the reset beat
      the send, which is a race, not a difference in the server.
    * A connection that was ACCEPTED and then not usefully answered --
      /health timed out, or the peer spoke something other than HTTP or sent
      a body that is not JSON (see _connected_but_unanswered) -- is a
      listener that engaged with the request, and "start one" is the wrong
      advice for the same reason. The line says what was seen and does not
      guess why: a server still starting, a busy one, a stuck one and a port
      held by something that is not an HTTP server all look alike from this
      side of the socket. It used to be filed under "nothing listening", on
      the strength of a docstring that said every non-HTTP error was exactly
      that.
    * Anything else is nothing listening at all. A genie_server that is still
      LOADING lands here and not above: it binds its port before the load
      and listens only once the model is resident, so until then a connect
      is refused (or times out) like one to a closed port. "Still starting"
      in the case above is some other server's behaviour.

    The launcher advice names the actual launcher and the port it serves on,
    because `python genie_server.py` directly serves on GENIE_PORT (default
    8080) while this tool's --base defaults to the launcher's 8123.
    """
    if isinstance(err, urllib.error.HTTPError):
        body = {}
        try:
            body = json.loads(err.read())
        except Exception:
            pass
        if not isinstance(body, dict):
            body = {}
        state = body.get("state") or body.get("status") or ""
        detail = body.get("detail") or ""
        if not detail and isinstance(body.get("error"), dict):
            detail = body["error"].get("message") or ""
        return ("server at %s is up but not ready: HTTP %d%s%s -- it is "
                "answering, so do not start another; wait for it to recover "
                "or restart it"
                % (base, err.code,
                   " state=%s" % state if state else "",
                   " (%s)" % detail if detail else ""))
    if _connected_but_unanswered(err):
        # Taken, and then either BROKEN or merely unanswered. The two get
        # different advice because they are different evidence, and the split
        # is inside this branch so that "was the connection taken?" is asked
        # in exactly one place.
        if _torn_down_connection(err):
            return ("something at %s accepted the connection and then broke "
                    "it before answering /health (%s) -- that is a server "
                    "holding the port and failing, or one on its way out, and "
                    "this tool cannot tell which from outside: do not read it "
                    "as a healthy listener. Re-run -- a second attempt that "
                    "is REFUSED means it has gone and the port is yours, and "
                    "one that breaks the same way means it is still there and "
                    "its own console or log has the reason"
                    % (base, _describe(err)))
        return ("something at %s accepted the connection but gave /health no "
                "usable answer (%s) -- a listener IS there, so do not start another "
                "on that port. It may still be starting, be busy, or be stuck, "
                "and this tool cannot tell which from outside: re-run in a "
                "moment, and if it keeps happening look at that server's own "
                "console or log" % (base, _describe(err)))
    return ("nothing listening at %s (%s) -- start one: run-genie-server.ps1 "
            "serves on 8123, `python src/genie_server.py` directly serves on "
            "GENIE_PORT (default 8080), so pass --base to match"
            % (base, _describe(err)))


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


def _content_of(body):
    """The generated text in an OpenAI-shaped completion, "" if there is none.

    Reasoning that arrives in a side channel is still generated text for the
    purpose this serves (did the server produce ANYTHING), so it counts.
    """
    try:
        m = (body.get("choices") or [{}])[0].get("message") or {}
    except Exception:
        return ""
    if not isinstance(m, dict):
        return ""
    text = m.get("content") if isinstance(m.get("content"), str) else ""
    for k in ("reasoning_content", "reasoning"):
        if isinstance(m.get(k), str):
            text += m[k]
    return text


def chat(base, model, prompt, max_tokens, timeout):
    """One completion. Returns a dict, or None after printing why it failed.

    The dict carries the server's own token counts and wall time, plus the
    model id it reported and the text it generated; the last two exist so the
    warmup can check the server is fit to measure before a sweep is spent on
    it. A response without a usage block (or with a zero prompt count, which
    is the same thing dressed as a number -- GenieAPIService reports all
    zeros) is refused here by name, because every rate downstream is formed
    from those counts and a zero silently became `prompt=0 ... 0.0 tok/s` rows
    and a decode SKIPPED that blamed the model.
    """
    body, wall = post_timed(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # BOTH cap spellings. genie_server and llama-server take either;
        # geniex serve ignores the legacy `max_tokens` outright and honours
        # only `max_completion_tokens` (README, "Comparing against the
        # official servers"). One spelling meant the "any OpenAI-compatible
        # server" claim failed on one of the two servers this repo compares
        # against, with the cap silently unhonoured.
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        # llama-server reuses a cached prompt prefix by default, which would
        # let the second request of a delta pair skip most of its prefill and
        # break the subtraction (see the module docstring). Off explicitly;
        # genie_server ignores the key.
        "cache_prompt": False,
        # Two spellings of "no thinking", so this works against servers that
        # honour either; a server that knows neither just returns its default
        # behaviour. genie_server also understands Anthropic's `thinking`
        # block, but that belongs to /v1/messages, not to this OpenAI-shaped
        # request, so it is not sent.
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }, timeout)
    if body is None:
        print("  request failed (%s) -- skipping this point" % wall, flush=True)
        return None
    usage = (body.get("usage") if isinstance(body, dict) else None) or {}
    prompt_tokens = usage.get("prompt_tokens") or 0
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        # A 200 carrying an OpenAI `error` object and no completion is the
        # commonest way to land here on the third-party endpoints --base
        # advertises support for: a model still loading, or unloaded. The
        # refusal is right either way, but printing only "no usage" sent the
        # reader after a bug in this tool while the server had already said
        # what was wrong -- and the warmup then exits on "the warmup request
        # failed (reason above)", where the reason above was the wrong one.
        # The sibling probe prints such a body already
        # (probe_server_semantics.ask, "non-OpenAI body: ..."). Appended, not
        # substituted: the counts are still what is missing.
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            err = err.get("message") or json.dumps(err)
        said = ("; the server said: %s" % str(err)[:200]) if err else ""
        print("  server reported no usage (prompt_tokens=%r) -- skipping this "
              "point: the counts here are the server's own, so without them "
              "there is no rate to form%s" % (usage.get("prompt_tokens"), said),
              flush=True)
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "wall": wall,
        "model": body.get("model"),
        "content": _content_of(body),
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
    steps==0 means the model hit EOS before the cap (or produced nothing), so
    there is no window. A non-positive `seconds` with the steps intact is a
    different fact -- the long run came back no slower than the short one,
    which is queueing noise or a cached prefix, never an early stop -- and it
    is handed back as such so the caller can say what actually happened
    instead of blaming EOS for it.
    """
    one = chat(base, model, prompt, 1, timeout)
    if one is None:
        return None
    many = chat(base, model, prompt, 1 + extra, timeout)
    if many is None:
        return None
    steps = many["completion_tokens"] - one["completion_tokens"]
    secs = many["wall"] - one["wall"]
    if steps <= 0:
        return (one, many, 0, 0.0)
    return (one, many, steps, secs)


# Fewest decode steps a delta may rest on. Below this the subtraction is
# measuring per-request overhead rather than decode: connection setup, template
# rendering and the tokenizer round-trip do NOT cancel perfectly between the two
# runs, and dividing their residue by three or four tokens produces a number
# with the shape of a rate and none of the meaning.
#
# Measured 2026-08-27: a prompt whose answer ran to 5 tokens gave a 4-step
# window and reported **0.60 tok/s against a true 17.6** -- off by 29x, printed
# in the same column as a real measurement. The existing `steps <= 0` guard did
# not fire, because 4 is not 0.
#
# This is easy to hit by accident now that the server seeds per process: answer
# LENGTH varies run to run where the shipped fixed seed made it constant. Same
# prompt, three seeds, measured: 105, 53 and 5 tokens -- all three at the same
# ~17.6 tok/s, so it is the WINDOW that moves, never the rate.
#
# The GENIE_ prefix is the server's, kept because docs/GENIE_SERVER.md lists
# every knob in one table and annotates which process reads it; this one is
# read by this tool only (and by bench_contention and bench_servers through
# it). Parsed defensively -- a typo here used to be an import-time traceback.
#
# bench_servers applies this floor AND a rule of its own -- a sample is
# discarded unless the long run produced at least 90% of the --tokens window
# it asked for -- so an accepted sample there has passed both, and one here has
# passed only this.
MIN_DECODE_STEPS = _int_env("GENIE_MIN_DECODE_STEPS", 16)


def _window_problem(steps, secs, hint="raise --tokens or use a prompt that generates"):
    """Why a (steps, seconds) window cannot yield a rate, or None if it can.

    Shared by the decode measurement and the probe that corrects prefill, so
    the two apply ONE floor: the probe used to accept a 4-step window that
    measure_decode would have refused, and its per-step cost was subtracted
    from every prefill row in the run. The three cases are kept apart because
    each wants a different reaction from the reader -- an early stop is the
    prompt, a non-positive delta is the box, a short window is the cap.
    """
    if steps <= 0:
        return "SKIPPED (model stopped before the cap or produced nothing)"
    if secs <= 0:
        return ("SKIPPED (non-positive delta: %d steps in %.2fs -- queueing "
                "noise or a cached prefix, not a rate)" % (steps, secs))
    if steps < MIN_DECODE_STEPS:
        return ("REFUSED: %d-step window (min %d). The model answered before "
                "the cap, so this delta is overhead, not decode -- %s."
                % (steps, MIN_DECODE_STEPS, hint))
    return None


def decode_probe(base, model, timeout, depth=500, steps=None):
    """Seconds per decode step, used to correct the prefill measurements.

    Cheap on purpose (a short generation at a shallow depth). Decode on this
    engine is close to flat with depth, so one figure corrects the whole prefill
    sweep; where it is not flat the correction is small anyway, and the raw wall
    time is printed alongside so nothing is hidden.

    The window it asks for defaults to MIN_DECODE_STEPS -- the same floor
    measure_decode enforces -- and a window that comes back shorter is refused
    under that floor rather than accepted. It used to ask for 8 and accept
    anything above zero, so the one figure subtracted from EVERY prefill row
    could rest on a 4-step window that the decode phase would have thrown out.
    The step count is printed beside the figure for the same reason it is
    printed beside every decode rate.
    """
    if steps is None:
        steps = MIN_DECODE_STEPS
    r = _delta_run(base, model, prompt_of(depth), steps, timeout)
    if r is None:
        return None
    _, many, got, secs = r
    problem = _window_problem(
        got, secs, hint="the prefill figures below will be RAW (uncorrected) and read LOW")
    if problem:
        print("  probe     depth=%-6d %s" % (many["prompt_tokens"], problem), flush=True)
        return None
    per_step = secs / got
    print("  probe     depth=%-6d %3d steps in %6.2fs   %.3f s/token (%.2f t/s)"
          % (many["prompt_tokens"], got, secs, per_step, 1.0 / per_step), flush=True)
    return per_step


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


def _probe_vs_decode(probe_rate, per_depth, tol=1.5):
    """What the measured decode says about the probe that corrected prefill.

    The prefill figures were corrected using the probe's per-step cost, and
    once real decode rates exist the probe can be checked against them -- a
    probe taken during a blip corrupts every prefill number in the run, and
    nothing else would reveal it. Lifted out of main() for the same reason as
    _probe_crosscheck: it is a warning, and it was untested.

    Compared at the SHALLOWEST measured depth, not against the median of every
    sample pooled across depths. The probe runs shallow, and on an engine whose
    decode genuinely falls with depth (the CPU leg this tool advertises, or the
    8192 bundle at 6000 tokens) the pooled median of a deep sweep is a real
    depth effect, not a misbehaving probe -- the pooled comparison fired "the
    probe was not representative" for exactly that. The depth compared at is
    named in the line so the reader can judge how close it is to the probe's.

    1.5, not 2. A real 1.94x understatement slipped through the 2.0 threshold
    on 2026-08-28 -- decode read 9.25 t/s against a true 17.7 and this check
    stayed silent, which is the one moment it exists for. Decode at a fixed
    depth is stable to a few percent run to run, so anything past 1.5x is
    already far outside the noise the tolerance was meant to absorb.
    """
    if not probe_rate:
        return None
    pooled = {}
    for depth, rates in per_depth:
        if rates:
            pooled.setdefault(depth, []).extend(rates)
    if not pooled:
        return None
    depth = min(pooled)
    med = statistics.median(pooled[depth])
    if not med:
        return None
    if med / probe_rate > tol or probe_rate / med > tol:
        return ("  WARNING: the decode probe read %.2f t/s but decode at depth "
                "%d measured %.2f t/s -- the probe was not representative, so "
                "treat the corrected prefill figures above as unreliable and "
                "re-run on a quiet box." % (probe_rate, depth, med))
    return ("  probe %.2f t/s against decode %.2f t/s at depth %d -- consistent, "
            "so the corrected prefill figures above stand."
            % (probe_rate, med, depth))


def measure_prefill(base, model, target, timeout, per_step=0.0):
    r = chat(base, model, prompt_of(target), 1, timeout)
    if r is None:
        return None
    if r["completion_tokens"] > 1:
        # The whole method rests on the cap being honoured: a 1-token request
        # that came back with a generation is prefill PLUS that generation,
        # and subtracting one step from it prints a several-x-low prefill in
        # the normal format with no warning. The warmup checks this once up
        # front; this catches a server that honours it inconsistently.
        print("  prefill   prompt=%-6d REFUSED: the 1-token cap came back with "
              "%d tokens, so this wall is prefill plus a generation -- the "
              "server is not honouring the cap"
              % (r["prompt_tokens"], r["completion_tokens"]), flush=True)
        return None
    raw = r["wall"]
    # Recorded before the RAW branch below, not after it: the sample whose
    # probe looks implausible is the one where the box is most suspect, and it
    # was the one sample that left no box state behind.
    note_box_state("prefill d%d" % r["prompt_tokens"])
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
    problem = _window_problem(steps, secs)
    if problem:
        # Refused rather than reported. A short window is not a slow engine,
        # and the two are indistinguishable once the number is in a table --
        # which is the whole reason this prints the step count beside every
        # rate. Each refusal names its own cause: an early stop, a
        # non-positive delta and a short window want different fixes.
        print("  decode    depth=%-6d %s" % (many["prompt_tokens"], problem), flush=True)
        return None
    rate = steps / secs
    print("  decode    depth=%-6d %3d tokens in %6.2fs   %8.2f tok/s"
          % (many["prompt_tokens"], steps, secs, rate), flush=True)
    note_box_state("decode d%d" % many["prompt_tokens"])
    return rate


def pool_by_depth(per_depth):
    """(shallowest-depth rates, deepest-depth rates) pooled across repeats.

    Lifted out of main() so it can be tested, same reasoning as resolve_depths:
    it decides what the headline verdict is computed FROM, and it was previously
    unreachable from any test.

    The bug it fixes was invisible by construction. main() used to take the
    first and last non-empty GROUPS, which is correct only when the depths run
    once each in ascending order. In the mode this tool actively recommends --
    `--depths 250,3300,250,3300 --decode-every` -- the groups ALTERNATE, so
    first-vs-last compared a single 250 sample against a single 3300 sample and
    discarded the repeats that are the entire reason for interleaving. The
    per-depth table printed all four rows either way, so nothing looked wrong;
    the verdict line just claimed more confidence than it had.

    Pooling by depth VALUE puts every shallow sample on one side and every deep
    one on the other, and sorting picks the true extremes rather than whichever
    happened to be measured first. Returns ([], []) when fewer than two depths
    produced samples -- there is no cross-depth claim to make from one depth.
    """
    pooled = {}
    for depth, rates in per_depth:
        if rates:
            pooled.setdefault(depth, []).extend(rates)
    if len(pooled) < 2:
        return [], []
    order = sorted(pooled)
    return pooled[order[0]], pooled[order[-1]]


def _per_depth_table(per_depth):
    """Lines of the per-depth median table, pooled by depth and sorted.

    Pooled and sorted, not in measured order: the table used to diff each row
    against the previous ROW, and in the interleaved mode the module docstring
    recommends (250, 3300, 250, 3300) every alternation is a ~30% swing, so
    every line carried "<-- step" and the marker meant nothing. Grouping by
    depth value first, exactly as pool_by_depth does for the verdict, makes
    each row one depth and each diff a difference between DEPTHS -- which is
    the only kind of step the table exists to show. [] with fewer than two
    depths, since there is nothing to diff.
    """
    pooled = {}
    for depth, rates in per_depth:
        if rates:
            pooled.setdefault(depth, []).extend(rates)
    if len(pooled) < 2:
        return []
    lines = ["", "  per-depth medians, pooled across repeats (look for PLATEAUS, "
                 "not a smooth slope):"]
    prev = None
    for d in sorted(pooled):
        m = statistics.median(pooled[d])
        # Flag the jumps rather than making the reader diff the column.
        mark = ""
        if prev is not None:
            change = (m - prev) / prev * 100
            mark = "  %+5.1f%%%s" % (change, "  <-- step" if abs(change) >= 8 else "")
        lines.append("    depth %-6d %6.2f t/s (n=%d)%s" % (d, m, len(pooled[d]), mark))
        prev = m
    return lines


def _verdict(shallow, deep, deep_flag="--repeat-deep"):
    """Compare decode at two depths, keeping same-depth noise out of the claim.

    The point of this line is whether cost tracks the COMPILED window or the
    context actually in use -- a statement about the difference BETWEEN depths.
    Pooling both groups and taking max-minus-min folds run-to-run noise at one
    depth into that difference, which is enough to flip the verdict on a noisy
    box while the depths genuinely agree. So compare medians, and report the
    noise as its own number instead of letting it masquerade as a depth effect.

    The noise is REPORTED beside the verdict, not used to flip it: gating on a
    max-minus-min range would let one outlier at n=3 suppress a real effect.
    But a "varies" call whose delta sits inside that noise is a call this run
    cannot actually back, and it says so on the same line rather than leaving
    the reader to notice that the two numbers disagree.
    """
    if not shallow or not deep:
        return
    ms, md = statistics.median(shallow), statistics.median(deep)
    delta = abs(ms - md)
    noise = max(max(shallow) - min(shallow), max(deep) - min(deep))
    print("  shallow median %.2f t/s (n=%d)   deep median %.2f t/s (n=%d)"
          % (ms, len(shallow), md, len(deep)), flush=True)
    flat = delta < 0.25 * min(ms, md)
    if flat:
        call = "flat, so cost tracks the COMPILED window, not the used context"
    else:
        call = "varies with depth on this engine"
        if delta <= noise:
            call += (" (though the delta sits inside the same-depth noise, so "
                     "THIS run cannot tell the two apart -- more repeats would)")
    print("  cross-depth delta %.2f t/s, same-depth noise %.2f t/s -- %s"
          % (delta, noise, call), flush=True)
    if len(deep) == 1:
        print("  (deep depth sampled once -- %s raises that)" % deep_flag,
              flush=True)


DEFAULT_DEPTHS = (500, 1500, 3000, 7000, 12000)


def resolve_depths(spec, limit, ctx, tokens):
    """Depths to probe, given the caller's --depths and the token budget.

    Lifted out of main() so it can be tested without standing up a server: the
    health check runs first, so a bad --depths was previously unreachable from
    any test and only discoverable by a user hitting a traceback.

    Raises ValueError with a message fit to show a user; returns (depths, note)
    where note is a line to print or None. Depths past the budget are DROPPED
    and named -- a silently shortened sweep reads as "measured everything".
    That holds for the DEFAULT sweep too: on the 8192 bundle the 12000 default
    is past the budget, and it used to vanish with no line, which is the same
    silent shortening this docstring promises against.
    """
    if not spec:
        asked = list(DEFAULT_DEPTHS)
        what = "default depth(s)"
    else:
        try:
            asked = [int(d) for d in spec.split(",") if d.strip()]
        except ValueError as e:
            raise ValueError(
                "--depths wants comma-separated integers (%s)" % e) from e
        what = "depth(s)"
    depths = [d for d in asked if 0 < d < limit]
    dropped = [d for d in asked if d not in depths]
    fallback = False
    if not depths:
        if spec:
            raise ValueError("every requested depth exceeds the budget of %d tokens"
                             % limit)
        # Nothing in the default sweep fits: probe the one depth that does
        # rather than refuse, since the caller asked for nothing specific.
        depths, fallback = [min(500, limit)], True
    note = None
    if dropped:
        note = ("  note: dropped %s %s -- past the %d-token budget "
                "(n_ctx %s minus --tokens %d minus margin)"
                % (what, ", ".join(str(d) for d in dropped), limit, ctx, tokens))
        if fallback:
            note += "; probing depth %d instead" % depths[0]
    return depths, note


def _served_model_note(requested, served):
    """The line that files these numbers under the id the SERVER gave.

    The run header used to print the --model flag alone, and nothing consults
    that flag: genie_server and llama-server both ignore the request's `model`
    field and serve whatever they loaded. `run-genie-server.ps1 -Model
    qwen3-8b` advertises `qwen3-8b-npu` on the same default port and the same
    4096 n_ctx as the 4B, so a run against it under the default --model was
    filed as `qwen3-4b-npu` with nothing in the output to say otherwise -- the
    exact "filed under the wrong bundle" failure n_ctx() exists to prevent.
    """
    if not served:
        return ("  (the server did not report a model id; these numbers are "
                "filed under --model %s on trust)" % requested)
    if served == requested:
        return None
    return ("  WARNING: --model %s but the server reports it is serving %s -- "
            "file these numbers under the SERVED id; the request's model field "
            "is not consulted by genie_server or llama-server"
            % (requested, served))


WARMUP_PROMPT = "Count from 1 to 5."
WARMUP_CAP = 24


def _warmup_check(r, cap):
    """Why the warmup answer disqualifies the server, or None if it is fit.

    The warmup result used to be discarded, and nothing later in the run reads
    generated text at all -- so the one server this repo knows to answer
    /health 200 and serve EMPTY completions (a Q8_0 llama-server leg) produced
    a full, legitimate-looking prefill table and a decode diagnosis blaming
    EOS. An ignored cap is the other half: every figure here is a delta between
    two CAPPED runs, and GenieAPIService honours neither spelling, so its
    "deltas" are between two full generations. Both are caught here, once, on
    the cheapest request of the run.
    """
    if r is None:
        return "the warmup request failed (reason above)"
    got = r.get("completion_tokens", 0)
    if got > cap:
        return ("the server ignored the %d-token cap and returned %d tokens. "
                "Every figure here is a delta between two capped runs, so an "
                "unhonoured cap makes them deltas between two full generations "
                "-- GenieAPIService honours neither `max_tokens` spelling (see "
                "README, 'What else serves these bundles, and what this does "
                "differently'); no "
                "delta-based tool in this repo can measure such a server"
                % (cap, got))
    text = (r.get("content") or "").strip()
    if got <= 1 or not text:
        return ("the server answered the warmup with %d token(s) and %d "
                "characters of content. A server that generates nothing still "
                "answers /health 200 and still reports a wall time, so the "
                "prefill table would look measured and every decode row would "
                "blame EOS -- fix the server first (a Q8_0 llama-server leg was "
                "seen doing exactly this)" % (got, len(text)))
    return None


ENV_HELP = """\
environment:
  GENIE_MIN_DECODE_STEPS  fewest decode steps a rate may rest on -- the decode
                          measurement AND the probe that corrects prefill both
                          refuse a shorter window (default 16). Read by this
                          tool only; the GENIE_ prefix is shared with the
                          server so docs/GENIE_SERVER.md can list every knob
                          in one table. A non-integer value is reported at
                          startup and the default used.
"""


def _parser():
    ap = argparse.ArgumentParser(description=__doc__, epilog=ENV_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8123",
                    help="server base URL (default %(default)s)")
    ap.add_argument("--model", default="qwen3-4b-npu",
                    help="model id to request (default %(default)s). The header "
                         "prints the id the server REPORTS beside it; the two "
                         "servers this tool targets ignore the request's field.")
    ap.add_argument("--tokens", type=int, default=200,
                    help="tokens to generate per decode measurement")
    ap.add_argument("--repeat", type=int, default=3,
                    help="decode repetitions at the shallow depth")
    ap.add_argument("--repeat-deep", type=int, default=1,
                    help="decode repetitions at the deep depth; each costs two "
                         "full deep prefills, hence the lower default")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--n-ctx", type=int, dest="n_ctx",
                    help="the served window, when the server has no /props. "
                         "Only a third-party endpoint needs this: without it "
                         "the depth budget falls back to 4096 and every "
                         "deeper depth is dropped with a note, which still "
                         "reads as a short sweep rather than a missing window.")
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
    return ap


def main():
    args = _parser().parse_args()

    base = args.base.rstrip("/")
    try:
        _get(base, "/health")
    except Exception as e:
        sys.exit(_health_failure(base, e))

    ctx = args.n_ctx or n_ctx(base)

    # Warmup BEFORE the header, because its answer is what the header reports
    # the served model from, and because it is the one request whose content
    # is checked: past it, nothing in the run reads generated text.
    print("warmup", flush=True)
    warm = chat(base, args.model, WARMUP_PROMPT, WARMUP_CAP, args.timeout)
    problem = _warmup_check(warm, WARMUP_CAP)
    if problem:
        sys.exit("refusing to benchmark %s: %s" % (base, problem))

    served = warm.get("model")
    print("\nendpoint %s   model=%s (server reports %s)   n_ctx=%s"
          % (base, args.model, served or "no id", ctx if ctx else "unknown"),
          flush=True)
    note = _served_model_note(args.model, served)
    if note:
        print(note, flush=True)
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

    # Measurements that produced a figure. The closing box-state line needs to
    # know whether there was anything to sample beside.
    accepted = 0
    probe_rate = None
    if not args.decode_only:
        print("\nprobe", flush=True)
        per_step = decode_probe(base, args.model, args.timeout,
                                depth=min(500, depths[0]))
        probe_rate = (1.0 / per_step) if per_step else None
        if per_step:
            print("\ndecode probe: %.3f s/token (%.2f t/s), subtracted from each "
                  "prefill below" % (per_step, 1.0 / per_step), flush=True)
        else:
            per_step = 0.0
            print("\ndecode probe failed or was refused; prefill figures still "
                  "include one decode step and therefore read LOW", flush=True)
        print("\nPREFILL (one decode step removed; raw wall also shown)", flush=True)
        for d in depths:
            if measure_prefill(base, args.model, d, args.timeout, per_step) is not None:
                accepted += 1
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
        accepted += len(allr)
        if allr:
            print("\n  decode median %.2f t/s over %d run(s)"
                  % (statistics.median(allr), len(allr)), flush=True)
            # The prefill figures above were corrected using the probe's
            # per-step cost. Now that real decode rates exist, say whether the
            # probe agreed with them, at the depth nearest the probe's own.
            line = _probe_vs_decode(probe_rate, per_depth)
            if line:
                print(line, flush=True)
        if args.decode_every:
            for line in _per_depth_table(per_depth):
                print(line, flush=True)
        shallow, deep = pool_by_depth(per_depth)
        if shallow and deep:
            # --repeat is what raises the count when every depth is measured;
            # --repeat-deep is not consulted in that mode, so naming it there
            # would send the reader to a flag that changes nothing.
            _verdict(shallow, deep,
                     deep_flag="--repeat" if args.decode_every
                     else "--repeat-deep")

    # Last, so it is the thing still on screen when the run ends. These figures
    # are only comparable to another run taken under the same box state, and
    # this is the only place that state is written down.
    for line in _box_state_trailer(box_state_summary(), accepted):
        print(line, flush=True)


if __name__ == "__main__":
    main()
