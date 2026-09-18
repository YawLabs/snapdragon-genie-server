#!/usr/bin/env python3
"""Measure what concurrent NPU + GPU inference costs each engine.

This answers the one open question both `docs/MULTI_ENGINE.md` and
`docs/TYPED_ROUTER_BRIEF.md` refuse to design around: "Does concurrent GPU +
NPU inference actually hold up? Both reach the same memory controller;
contention is measurable but has not been measured here."

Until there is a number, a router has no basis for ORDERING an engine's
fallback list -- and no way to know whether shedding to a second hot engine is
a win at all rather than a way to make both slow.

WHY A SEPARATE SCRIPT FROM bench_endpoint.py

That one measures an engine in isolation and is deliberately serial. Contention
is a property of two engines running AT THE SAME TIME, and the measurement is
worthless unless the windows genuinely overlap -- so the load has to be driven
concurrently and the sampling confined to the overlapped stretch. Everything
else is imported from bench_endpoint rather than reimplemented: the prompts,
the decode-by-subtraction method, the HTTP plumbing and the one PowerShell
box-state query are already validated there, and a second copy would drift.

METHOD, AND ITS LIMITS

  * DECODE is the headline metric. Decode streams the whole model per token, so
    it is bandwidth-bound and it is where shared-bus contention shows. Prefill
    is the weaker signal and is not measured here.

  * Each engine is measured SOLO first. Those are the baselines every later
    ratio is against, so they are only meaningful on a quiet box (below).

  * For the contended pass, a background load generator saturates the OTHER
    engine in a loop while the engine under test is measured. This is the
    point: a naive "fire one request at each and compare" leaves the faster
    engine idle for the tail of the slower one's run, so part of the measured
    window is uncontended and the result understates contention. The generator
    is started first, given a ramp to get in flight, and only then is the
    measurement taken.

  * The generator is shaped like the measurement by default (same depth, same
    token count), which means each of its requests is a full prefill -- which
    is compute-bound -- before its decode steps, which are bus-bound. So the
    peer is bus-bound for only part of the contended window. --load-depth and
    --load-tokens reshape it (a shallow prompt with many tokens keeps the peer
    decoding for nearly the whole window), and the generator's tokens per
    second of leg time is recorded per engine so two runs with different load
    shapes cannot be mistaken for each other in the artifact. A reshaped load
    is sent ONCE to each server in the warmup, as the generator will send it,
    and the run refuses to start if either cannot serve it -- a shape that
    overflows a peer's window would otherwise fail on every generator request
    and leave the contended legs measured against an idle peer.

  * The NPU is SINGLE-FLIGHT: its server serialises behind a lock and answers
    429/529 "server busy" once its small queue fills. The generator is ONE
    sequential client, so against a healthy peer it never sees that -- each of
    its requests is the only one in flight. A 429 here therefore means
    something ELSE was in flight (foreign traffic on this shared box, or a
    permit wedged behind a timed-out request). It is counted as `shed` and
    kept apart from every other failure (`failed`, with the first reason
    printed), because the two say different things: a shed request is time the
    peer spent queued rather than contending, a refused connection is time it
    spent idle. Neither kills the generator, which carries on so the leg stays
    loaded; both are reported afterwards.

  * Reported per engine: solo rate, contended rate, and the ratio. Reported for
    the pair: aggregate throughput while both are hot, against the sum of the
    solo rates. An aggregate BELOW the faster engine's solo rate means routing
    to a second hot engine is a net loss for throughput -- it would still buy
    concurrency and failover, but not speed.

  * A run is only comparable to another run at the same /props n_ctx, which is
    printed and written to the JSON per endpoint, beside the base URL, the
    requested model id and the id the server itself reports. On Genie the
    COMPILED window sets throughput.

THE OPENING LEG IS RE-RUN AT THE END

Every sample is gated on clock recovery before it is taken, and that gate is
structurally blind to what happens next: a leg takes a minute or two, and
sustained load drives this box to 48.9% of base, so a figure can be gated at
entry and decay through its own measurement. So the sweep finishes by re-running
the leg it STARTED with, under the same gate -- an A/A whose only variable is
elapsed time. Opening and closing within `--closing-tol` (default 10%) means
the box held; anything else means the ratios above it are measuring the box,
not contention.

Symmetric on purpose. Slower at the end means decay landed in the numerator
(contended samples are taken after the solo ones they divide) and contention is
overstated. FASTER at the end means the opening sample was the degraded one, so
every baseline the ratios divide by is too low and the retention percentages are
flattered. Both disqualify the run.

This is a different instrument from `drift_note`, which flags a strictly
monotonic decline across 3+ solo samples: one noisy sample out of order hides a
real trend there, and at `--repeat 1` it has nothing to compare. `--no-closing-
recheck` skips it, at the cost of the only number that says whether the run held.

WHAT THE GATE DOES WHEN IT CANNOT GATE

Three outcomes, each recorded in the run's warnings and the JSON, because a
sample whose gate silently did nothing is indistinguishable from a gated one
once it is in a table:

  * On BATTERY the clock cannot recover by waiting, and nothing measured on
    battery is worth keeping -- so the gate ABORTS and the sweep STOPS there.
    Rounds completed before it are reported and the run exits 2.
  * A gate that waits out its limit (300 s) PROCEEDS, and says so: that sample
    was taken on a box that had not recovered.
  * A gate whose counter cannot be read PROCEEDS UNGATED, and says so.

The abort above is the only thing that STOPS a sweep for being on battery, and
it is easy for it never to run: the gate is off (--cool-floor 0), or the clock
never fell below the floor so the power source was never read, or the counter
could not be read at all. A run that finished on battery any of those ways used
to publish `warnings: []` and exit 0. The power source is recorded per round
regardless (`power_samples[].on_ac`), so it is checked once more at the end and
a run taken on battery is warned about in both places -- exit code unchanged,
because only the gate stops a sweep.

A gate that was switched OFF (--cool-floor 0) is not left unjudged either. No
gate runs, so `gate_clock_pct` is [] by construction, and the round-start
clock readings (`power_samples[].clock_pct`) are judged instead -- against 92%
of base, DEFAULT_COOL_FLOOR, the floor the gate itself defaults to. A dip
below it is a warning that begins "UNGATED run (--cool-floor 0)", in the same
two places.

THE QUIET-BOX PRECONDITION IS ENFORCED, NOT SUGGESTED

Every previously recorded number on this hardware was taken on a loaded box and
had to be retracted -- NPU prefill was understated 3.3x that way. So this
refuses to run when free physical memory is below --min-free-gb rather than
printing a caveat nobody reads. Override with --allow-loaded, which stamps
every result LOADED so the output cannot later be mistaken for a baseline.

THE GPU LEG'S -c IS A MEMORY-GATE VARIABLE, NOT ONLY A CAPACITY ONE

Start llama-server with `-c 4096` and its OpenCL KV allocation is **5.76 GB**;
at `-c 1024` the same allocation is **144 MiB**. So the window chosen for the
GPU leg decides whether the pair fits under --min-free-gb at all -- an operator
who picks a window for capacity reasons can find the run refused for memory
reasons, and reach for --allow-loaded to escape a gate that was never really
about a loaded box.

Size the GPU window to the DEPTH being measured, not to the largest prompt the
model could take: a d469 measurement with 120 decode steps fits `-c 1024` with
room to spare. What matters is that the window is IDENTICAL across the legs
being compared -- a contention ratio taken at two different windows is not a
contention ratio.

WHICH PORT IS THE GPU

`run-llama-server.ps1` serves its CPU leg on 8080 (that leg IS typed's local
endpoint) and its GPU leg on 8124. --gpu defaults to 8124 accordingly; the
harness labels whatever answers there "GPU" and cannot tell a CPU leg from a
GPU one by itself, which is why it prints the model id the server REPORTS
beside each label (the launcher aliases the GPU leg `qwen3.5-9b-gpu`) and
writes it to the JSON. A bare run used to default to 8080 and filed a CPU
measurement under "GPU"; the operator relabelled it by hand.

  python src/bench_contention.py
  python src/bench_contention.py --npu http://127.0.0.1:8123 --gpu http://127.0.0.1:8124
  python src/bench_contention.py --load-depth 32 --load-tokens 400   # decode-heavy peer
  python src/bench_contention.py --json out.json

Pure stdlib. Needs both servers already up; it starts nothing. Obeys two
environment variables (see --help): GENIE_LOW_CHARGE_PCT, read here, and
GENIE_MIN_DECODE_STEPS, read by bench_endpoint at import -- the floor under
every decode measurement this tool makes, so a value above --tokens would void
every sample and is refused at startup instead.

Exits 2 when it refuses to run or the sweep was stopped by the gate, 3 when the
run finished but its --json artifact could not go to the path that was asked
for -- it is written beside that path under a timestamped name instead, and the
results are never discarded -- and 0 otherwise.
"""

import argparse
import json
import os
import re
import socket
import statistics
import sys
import tempfile
import threading
import time
import urllib.parse

import bench_endpoint as be


def _float_env(name, default):
    """A float from the environment, or `default` with a line saying why not.

    The float twin of bench_endpoint._int_env, kept here because that module
    only needs integers. Same reason it exists: a typo in the variable used to
    kill this module at IMPORT with a bare `could not convert string to float`
    naming neither the variable nor the form it wanted -- and because the
    test file imports this module at collection, the typo killed the whole
    test run with it.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("[bench] WARNING: %s=%r is not a number; using %r instead."
              % (name, raw, default), flush=True)
        return default


def free_physical_gb():
    """Free physical RAM in GB, or None if it cannot be determined.

    Returning None rather than guessing matters: the precondition below must
    fail loudly on an unknown, not silently pass a box it could not measure.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = MemStatus()
            m.dwLength = ctypes.sizeof(MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            return m.ullAvailPhys / (1024 ** 3)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        return None
    return None


# Per-request cap for the background load generator (see Load.__init__).
LOAD_REQUEST_TIMEOUT_S = 180

# How long stop() waits for the worker past its own request timeout before
# calling it a straggler. The worker only tests the stop flag between requests,
# so it can legitimately be in flight for up to LOAD_REQUEST_TIMEOUT_S after
# stop() is called; anything past that plus this grace is reported.
LOAD_JOIN_GRACE_S = 30

# Pause after a request FAILS. A refused connection comes back in microseconds,
# and a generator that re-posts immediately spins a core the measurement shares
# -- so a dead peer would load the box in a way the "contended" figure then
# attributes to contention.
LOAD_FAILURE_PAUSE_S = 1.0

# Pause after a SHED request (429/529), shorter because a shedding peer is
# alive and the generator wants to be the next request it takes.
#
# This used to be zero, on the reasoning that "a 429 costs the peer a lock
# probe, and re-posting is what keeps its queue full". Both halves were wrong.
# A shed costs the peer a TCP accept, a full read of the depth-sized JSON body
# and a json.loads of it -- genie_server's do_POST does all of that before it
# touches _INFLIGHT -- plus a flushed log line per shed. And the queue it was
# keeping full is not ours: shed_note's own rule is that a lone sequential
# client only sees a 429 when something ELSE holds the peer, so there is
# nothing to hold a place in.
#
# Measured on this box with a loopback harness (genie_server.Handler with
# every permit taken, depth 2000, 5 s): the no-pause loop did 246 requests/s
# at 39% of one core and drew 1234 server log lines; the 0.25 s below did 19
# requests at 2.8% of a core and 19 log lines. That 0.4 of a core lands on the
# host while the engine under test is being TIMED, which pushes the contended
# figure down for a reason that is not peer contention -- and the GPU leg needs
# those cores to dispatch a kernel per token.
#
# Not so long that the peer goes idle between our requests: four a second
# still queues behind anything the peer is serving, and `shed` counts the
# attempts, which is what the leg is judged on.
LOAD_SHED_PAUSE_S = 0.25


def is_backpressure(reason):
    """Whether a post_timed failure reason is the peer shedding (429/529).

    post_timed's reason string starts with "HTTP <code>" when the server
    answered and with the exception's type when it did not (URLError for a
    refused connection, TimeoutError for a request that outran the cap). The
    two are the difference between a peer that was queued and one that was
    idle, which is why they are counted apart.
    """
    return isinstance(reason, str) and reason.startswith(("HTTP 429", "HTTP 529"))


class Load:
    """Saturates an endpoint in a loop until stopped.

    Exists so the engine under test is measured while the other one is
    genuinely busy. One sequential worker, deliberately: the load shape is
    part of what the ratios mean, and a change to it is a change to every
    number this tool has produced.

    Counts three outcomes apart, because they mean three different things for
    the leg they were counted in:

      completed   the peer served the request; `tokens_out` is what it decoded
      shed        the peer answered 429/529. A lone sequential client never
                  gets that from a healthy single-flight server, so it means
                  something else was in flight -- the peer was queued, not
                  contending, for that request
      failed      anything else: refused, reset, timed out, a non-JSON body.
                  The first reason is printed once and kept, because a refused
                  connection is a peer that was NOT loaded, and that decides
                  whether the contended leg measured anything at all

    None of them stops the loop. A generator that died on the first failure
    would silently stop contending halfway through a leg, and the ratio would
    then read as "no contention effect" rather than "no load".
    """

    def __init__(self, base, model, depth, tokens, timeout):
        self.base, self.model = base, model
        self.depth, self.tokens = depth, tokens
        # The generator does NOT inherit the measurement timeout. Its worker
        # only tests the stop flag between requests, so a long per-request
        # timeout makes stop() block for that long -- 1800s by default, which
        # on the single-flight NPU is a real wait behind a queued request, not
        # a theoretical one. It only has to produce load, so a request that
        # overruns this is worth abandoning.
        self.timeout = min(timeout, LOAD_REQUEST_TIMEOUT_S)
        self.join_timeout = self.timeout + LOAD_JOIN_GRACE_S
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self.completed = 0
        self.shed = 0
        self.failed = 0
        self.first_failure = None
        self.tokens_out = 0
        # Wall time from start() to the end of stop(): the denominator of the
        # generator's duty cycle (tokens_out / seconds), which is how a
        # decode-heavy --load-tokens run and a prefill-heavy default one are
        # told apart in the artifact.
        self.seconds = 0.0

    def _count_failure(self, reason):
        if is_backpressure(reason):
            self.shed += 1
            # Paused, like a failure and for the same reason -- a shed comes
            # back nearly as fast as a refused connection, and the spin is
            # CPU the timed leg is paying for. See LOAD_SHED_PAUSE_S for what
            # it cost measured.
            self._stop.wait(LOAD_SHED_PAUSE_S)
            return
        self.failed += 1
        if self.first_failure is None:
            self.first_failure = reason
            print("    (load on %s: a request FAILED -- %s. Counted as a "
                  "failure, not as backpressure; only HTTP 429/529 is that)"
                  % (self.base, reason), flush=True)
        # Wakes early when stop() is called, so the pause never delays a stop.
        self._stop.wait(LOAD_FAILURE_PAUSE_S)

    def _run(self):
        prompt = be.prompt_of(self.depth)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # Both cap spellings, and the prefix cache off, for the reasons
            # bench_endpoint.chat gives: geniex serve ignores the legacy
            # `max_tokens`, and llama-server would otherwise serve every
            # repeat of this identical prompt from its cached prefix -- a
            # generator that skips prefill is not the load the docstring
            # describes.
            "max_tokens": self.tokens,
            "max_completion_tokens": self.tokens,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        while not self._stop.is_set():
            try:
                body, info = be.post_timed(self.base, "/v1/chat/completions",
                                           payload, self.timeout)
                if body is None:
                    # `info` is the failure reason post_timed hands back.
                    # It used to be discarded here as `_wall`, and the
                    # comments then explained that a 429 and a refused
                    # connection were indistinguishable. They never were.
                    self._count_failure(info)
                    continue
                usage = (body.get("usage") if isinstance(body, dict) else None) or {}
                self.completed += 1
                self.tokens_out += int(usage.get("completion_tokens") or 0)
            except Exception as e:
                # A worker that dies takes the load with it, and until this
                # guard nothing would have said so: the leg would have run
                # against an idle peer and published a ratio near 1.0.
                # Counted as a failure so the report can say "no load".
                self._count_failure("%s: %s" % (type(e).__name__, e))

    def start(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.join_timeout)
            if self._thread.is_alive():
                # Not silent. This is the one thing the try/finally around
                # the contended leg exists to prevent -- load outliving its
                # leg -- and returning quietly here would let a straggling
                # request overlap the next solo sample with no record of it.
                # The worker cannot loop again (the flag is set), so at most
                # one request is still in flight; that one is still enough
                # to contaminate a baseline.
                why = ("the load generator on %s was still in flight %ds after "
                       "it was told to stop; its last request may overlap the "
                       "next sample" % (self.base, self.join_timeout))
                GATE_NOTES.append(why)
                print("    (WARNING: %s)" % why, flush=True)
        if self._t0 is not None:
            self.seconds = time.perf_counter() - self._t0


# The two single-counter reads this file keeps of its own, as TEXT, so a test
# can pin them: the PowerShell half cannot run device-free, and the Python half
# below is only three lines of float().
#
# A CookedValue is a double, and PowerShell renders a double in the session's
# CURRENT culture. The bare `.CookedValue` therefore printed
# "72,4370708845929" on a comma-decimal Windows (de-DE, fr-FR, ...) and the
# float() below raised, so BOTH readers returned None -- measured here with
# CurrentCulture set to de-DE, against the same box that gave
# "70.956354241159" under en-US.
#
# That was not a lost reading but a lost GATE. wait_for_cool returns on its
# first poll when the clock reads None, before it ever reaches
# power_limited_note, so on such a box every sample was taken UNGATED and the
# on-battery abort -- the one outcome that stops the sweep -- could not fire at
# all. bench_endpoint's _STATE_PS had the same defect and the same fix; this is
# the rest of it, because that fix left these two reads alone.
#
# `.ToString([cultureinfo]::InvariantCulture)` rather than `-f`, which formats
# in the current culture whatever it is handed (see _STATE_PS, which makes its
# doubles text invariantly before `-f` can). Verified with the real
# powershell.exe under both cultures. A counter that cannot be read still
# prints nothing and still comes back as None: the method call on $null fails
# to stderr, which is the same empty stdout the bare read gave.
_INVARIANT = ".ToString([cultureinfo]::InvariantCulture)"
_CLOCK_PS = ("(Get-Counter '\\Processor Information(_Total)\\% Processor "
             "Performance').CounterSamples.CookedValue" + _INVARIANT)
_BUSY_PS = ("(Get-Counter '\\Processor(_Total)\\% Processor Time')"
            ".CounterSamples.CookedValue" + _INVARIANT)


def cpu_performance_pct():
    """Current clock as a percentage of base, or None.

    A throttle detector, and the reason it is here: this box was observed at
    86.8% while an unrelated export ran. Sustained load on a thin ARM64 laptop
    decays clocks, and a decay that happens to land during the contended half
    of a run is indistinguishable from contention unless it is watched. There
    is no MSAcpi thermal zone on ARM64, so this counter is the available proxy.

    Its own single-counter launch rather than a be.box_state() call, and this
    is the one place the counter path is deliberately written twice:
    wait_for_cool polls it every ten seconds while a box cools and uses
    nothing but the clock, and box_state would run two WMI battery queries
    per poll for values the poll then discards -- which is the cost this
    file's own rule (see power_limited_note) says not to pay. How much those
    queries add to a launch has NOT been timed here; the reason is the
    discarded readings, not a measured saving. The round-start sample in
    paired_sweep, which wants all four values, does go through box_state.

    Writing the counter path twice is the cost of that choice, and it has been
    paid once already: the culture fix that _CLOCK_PS now carries went into
    box_state's script alone, and this reader stayed broken behind it.
    """
    if sys.platform != "win32":
        return None
    import subprocess
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", _CLOCK_PS],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _source_of(raw):
    """'battery' | 'ac' | 'no-battery' | 'unknown', from the raw PowerOnline text.

    Distinguishing the last two matters: a transient PowerShell failure used to
    return the same None as a desktop with no battery, silently downgrading a
    definitive check to a heuristic on a laptop that could have answered.
    """
    if raw is None:
        return "unknown"
    if raw == "":
        return "no-battery"
    return "battery" if raw.lower() != "true" else "ac"


def power_reading():
    """(source, charge_pct, charge_watts) from ONE PowerShell launch.

    `source` is _source_of's label; the pack figures are None where unread.

    The reading comes from bench_endpoint's one WMI query. This module carried
    its own copy of that query for a while, forty lines from a comment saying
    the query string must not exist twice -- and the two copies mapped the
    empty-string case differently. The dependency direction is the one that
    already exists: this module imports bench_endpoint, never the reverse.

    One call for all three, where there used to be a power_source() and a
    battery_state(): each was a full launch of bench_endpoint's four-field
    script, so the gate's AC branch paid for two launches and threw away, from
    the first, the pack figures it then went back for with the second. The
    launch still samples the script's clock counter, which nothing here reads
    -- one discarded field, the price of the query existing exactly once.

    The pack is sampled as MAGNITUDES rather than folded into a boolean, which
    is the lesson from four legs measured across two sessions on this box. A
    counter keyed to "is charging suspended" with an absolute near-zero
    threshold reported 0 suspended samples for legs that shed 92% and 97% of
    their charge draw, because the minima (3.0 W, 1.1 W) cleared a <=1 W test.
    A later threshold at 25% of opening draw scored 2 of the same 4 legs. Every
    cut point mis-sorts the legs nearest it, so record the quantity and let a
    reader pick their own line afterwards.
    """
    raw, charge, watts = be.power_reading()
    return _source_of(raw), charge, watts


def cpu_busy_pct():
    """Box-wide CPU utilisation, or None. The second half of the fingerprint.

    Culture-proof for the same reason as the clock read above, though the
    stakes are lower: this one only feeds power_limited_note's advisory
    fingerprint on a box with no battery, where None downgrades the advice.
    """
    if sys.platform != "win32":
        return None
    import subprocess
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", _BUSY_PS],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# Below this the pack is deep enough into discharge that the system protects
# the charge and starves compute, whether or not AC is connected. Measured on
# this box across two sessions: legs run at 13-20% gave pp512 58.14 / 57.88
# against a settled baseline of 130.20 -- HALF -- while legs at 33% and 41.6%
# came back at 124.53 and 112.58, near baseline and stable. The hazard is the
# depth of discharge, not the act of charging, which is the opposite of what
# both sessions assumed before pooling their legs.
#
# Parsed defensively: a typo here used to be an import-time traceback.
LOW_CHARGE_PCT = _float_env("GENIE_LOW_CHARGE_PCT", 25.0)


def power_limited_note(pct, floor):
    """Why the clock is low, when it is low for a reason waiting cannot fix.

    Returns (message, abort) or (None, False). `abort` says whether waiting is
    provably futile; a message with abort=False is advisory only.

    The power SOURCE is the one definitive signal, so it is the only thing that
    aborts. The low-clock-with-idle-CPU fingerprint is kept as advice but must
    NOT abort, because it is wrong in the case this function is called from:
    this gate runs BEFORE each sample, when the box is legitimately idle, and
    an idle CPU downclocks by design. Treating that as power limiting aborts
    the gate on a perfectly healthy run -- which on a machine reporting no
    battery would mean the gate never works at all. The fingerprint is only
    sound when the CPU is BUSY, which is its own premise: a thermally limited
    box is busy.

    EACH READING COSTS A POWERSHELL LAUNCH, and this runs before every sample.
    So it takes only the launches whose readings it uses: `power_reading`
    always -- one launch for the source (the thing that decides) AND the pack
    state the AC branch wants -- and the CPU-busy figure only in the no-battery
    branch, the one place its value is read. It previously took three launches
    on every call -- including `cpu_busy_pct()` unconditionally, so every AC
    run paid for a number it then discarded -- and then two on AC, the source
    and the pack being separate launches of the same query. The file already
    carried a comment worrying about two subprocess calls here when there were
    two; a third was added without anyone noticing, which is how that kind of
    cost grows.
    """
    if pct is None or pct >= floor:
        return None, False
    src, charge, watts = power_reading()
    if src == "battery":
        return ("clock is %.0f%% of base and this machine is ON BATTERY. That "
                "is power limiting, not heat -- waiting will NOT recover it. "
                "Plug in AC and re-run; nothing measured on battery is worth "
                "keeping." % pct), True
    if src == "unknown":
        return ("clock is %.0f%% of base and the power source could not be "
                "read, so it is not known whether waiting can help." % pct), False
    # AC used to fall straight off the end of this function and return "nothing
    # to report", so a box plugged in but deeply discharged read as clean. That
    # is the state that actually costs you half your prefill, and it is exactly
    # the state an operator reaches after plugging in and starting immediately.
    if src == "ac":
        if charge is not None and charge < LOW_CHARGE_PCT:
            return ("clock is %.0f%% of base, and the pack is at %.0f%%%s. AC "
                    "is connected, but measured on this box a pack below ~%.0f%% "
                    "halves prefill (pp512 58 against a settled 130) while "
                    "decode barely moves -- so this is not a settled-box "
                    "measurement even though it is plugged in. Advisory, not "
                    "fatal: bandwidth-bound work is largely immune. Let it "
                    "charge for a clean baseline."
                    % (pct, charge,
                       " drawing %.0f W" % watts if watts is not None else "",
                       LOW_CHARGE_PCT)), False
        return None, False
    if src == "no-battery":
        busy = cpu_busy_pct()
        if busy is not None and busy < 15.0:
            return ("clock is %.0f%% of base while the CPU is only %.0f%% busy. "
                    "On a box with no battery that is most likely ordinary idle "
                    "downclocking rather than a limit -- but if the clock stays "
                    "low once work starts, check the power budget."
                    % (pct, busy)), False
    return None, False


# Notes raised by the run's instruments -- the cool gate and the load
# generator's stop() -- drained into the run's warnings so they reach the JSON
# as well as the terminal. Every occurrence is recorded; main() collapses
# repeats (dedupe_notes) so a note the gate raises before each of seven legs
# lands in the artifact once, with its count, rather than seven times.
GATE_NOTES = []

# One (on_ac, charge_pct, charge_watts, clock_pct) reading per round, so the
# artifact carries the power TRAJECTORY rather than a verdict about it. A run
# taken while the pack climbed from 20% to 60% is a different run from one
# taken at a steady 95%, and nothing else in the record distinguishes them. The
# clock here is the round-START reading, taken before the gate and after the
# previous round's contended leg, so from round 2 on it shows the dip the gate
# then waits out; it is trajectory, not a verdict, and the warning about
# limited samples keys on the GATED readings instead (see measure()) -- unless
# the run is UNGATED (--cool-floor 0), where nothing waits that dip out, the
# next solo sample is taken AT this reading, and main() judges it against
# DEFAULT_COOL_FLOOR because it is the only clock evidence the run has.
POWER_SAMPLES = []

# How long wait_for_cool waits before giving up and proceeding.
COOL_LIMIT_S = 300

# The floor --cool-floor defaults to, named because it is ALSO the threshold an
# ungated run (--cool-floor 0) is judged against: with the gate off there is
# no operator-chosen floor to hold the round-start readings to, and the one
# the gate would have used by default is the only figure with an argument
# behind it. Not the 95 this judgement once used -- this box idles at ~94%, so
# 95 stamped runs the gate itself would have passed.
DEFAULT_COOL_FLOOR = 92.0

# A number, or a URL matched whole so the numbers INSIDE it are left alone (see
# dedupe_notes). The URL alternative comes first: at an "http" the scan takes
# the whole URL, digits and all, before the number alternative is ever tried.
_NUMBER_OR_URL = re.compile(r"(https?://\S+)|-?\d+(?:\.\d+)?")


def dedupe_notes(notes):
    """The notes with repeats of the same SHAPE collapsed to the first, counted.

    Two notes are the same shape when they differ only in their numbers. The
    gate's AC-low-pack advisory fires before every gated leg with that leg's
    clock and charge interpolated -- up to 2*repeat+1 times a run, then once
    more from main() -- and an artifact that says the same thing eight ways is
    one whose reader skips the ninth warning, which is the one that matters.
    The trajectory those numbers described is in POWER_SAMPLES and BOX_SAMPLES
    anyway; the note only has to say it happened, and how often.

    EXCEPT the numbers in a URL, which are an identity and not a reading. The
    straggler note from Load.stop() names the server its request was still in
    flight on, and at the default --npu/--gpu the PORT is the only thing
    telling the two engines apart -- so wildcarding it collapsed an NPU
    straggler and a GPU one into a single note naming whichever came first,
    "raised 3 times", when which server had the overlap is the whole content
    of that note (an NPU straggler queues the next NPU solo baseline behind it
    on the single-flight server) and is recorded nowhere else in the JSON.
    Repeats on the SAME server still collapse, whatever their other numbers.
    """
    seen = {}
    order = []
    for note in notes:
        key = _NUMBER_OR_URL.sub(lambda m: m.group(1) or "#", note)
        if key in seen:
            seen[key][1] += 1
        else:
            seen[key] = [note, 1]
            order.append(key)
    out = []
    for key in order:
        text, count = seen[key]
        if count > 1:
            text = "%s (raised %d times during this run; the first is shown)" % (text, count)
        out.append(text)
    return out


class GateAborted(Exception):
    """wait_for_cool found waiting provably futile (the box is on battery).

    An exception rather than a flag on the return value, because the right
    response is not "skip this sample" but "stop the sweep": the power source
    does not change between legs, every later gate would abort the same way,
    and the message says nothing measured on battery is worth keeping. The
    return value used to say 'ABORTING THE GATE' and then the sample was taken
    anyway, entered the medians, and the run exited 0.
    """

    def __init__(self, pct, why):
        super().__init__(why)
        self.pct = pct
        self.why = why


def wait_for_cool(floor, limit=COOL_LIMIT_S):
    """Block until the clock recovers to `floor`% of base, or `limit` seconds.

    Returns the reading the gate passed (or gave up) on, None when the counter
    could not be read, and raises GateAborted when waiting is futile. Every
    outcome other than a clean pass is written to GATE_NOTES as well as
    printed, because a gate that did nothing and said nothing is what produced
    the retracted numbers this file keeps mentioning.

    Measured on this box 2026-08-23: sustained GPU decode drove the clock from
    ~94% to 48.9% of base within a few minutes, and a hand-run depth sweep
    taken across that decay produced a clean monotonic 19.65 -> 11.03 t/s that
    looked exactly like a depth effect and was not. Detecting drift after the
    fact (drift_note) tells you the run was wasted; gating on recovery BEFORE
    each sample stops it being wasted. Both are kept -- the gate prevents the
    common case, the detector catches what the gate misses.
    """
    start = time.time()
    pct = None
    first = None
    while time.time() - start < limit:
        pct = cpu_performance_pct()
        if pct is None:
            # Recorded, not merely returned. cpu_performance_pct returns None
            # on any failure -- a 30 s subprocess timeout, an empty counter
            # read -- and this used to return the same silent None, which
            # measure() discarded: a sample taken on a box that never
            # recovered looked identical to a gated one in the terminal AND
            # in the JSON. A fast pass prints nothing either, so the two were
            # indistinguishable by construction.
            why = ("gate skipped: the clock counter could not be read%s, so "
                   "this sample was taken UNGATED and may sit on a box that "
                   "had not recovered"
                   % (" after a %.0f%% reading" % first if first is not None else ""))
            GATE_NOTES.append(why)
            print("    (note: %s)" % why, flush=True)
            return None
        if first is None:
            first = pct
            # Checked ONCE, on the first below-floor reading, rather than every
            # loop: the power source does not change while we spin, and this
            # costs a PowerShell launch (two on a box with no battery). Bailing
            # immediately matters because the alternative is blocking the full
            # `limit` for a recovery that cannot happen -- observed 2026-08-24
            # as ten minutes of silence from a run whose box had been
            # unplugged, which read as a hang.
            why, abort = power_limited_note(pct, floor)
            if why is not None:
                # Recorded, not just printed. A warning that exists only in the
                # terminal is absent from the artifact a consumer reads, which
                # is how a suspect number becomes a clean-looking one
                # downstream -- the same record-vs-reality drift this harness
                # keeps finding elsewhere.
                GATE_NOTES.append(("gate ABORTED: " if abort else "") + why)
                print("    (%s: %s)"
                      % ("ABORTING THE GATE" if abort else "note", why), flush=True)
                if abort:
                    raise GateAborted(pct, why)
        if pct >= floor:
            if time.time() - start > 5:
                print("    (cooled %.0f%% -> %.0f%% after %ds)"
                      % (first, pct, int(time.time() - start)), flush=True)
            return pct
        time.sleep(10)
    # The literal gave-up-on-cooling case. The commit that added GATE_NOTES
    # named exactly this defect ("the results file showed a clean run where the
    # harness had actually given up on cooling") and then fixed only the power
    # branch above; this branch kept printing and never recording.
    why = ("clock still %s%% after %ds; the gate gave up and this sample was "
           "taken on a box that had NOT recovered -- thermally suspect"
           % ("%.0f" % pct if pct is not None else "?", limit))
    GATE_NOTES.append(why)
    print("    (WARNING: %s)" % why, flush=True)
    return pct


def measure(base, model, depth, tokens, timeout, label, cool_floor=None, clocks=None):
    """Decode rate at one depth, printed with its label.

    With a `cool_floor` the sample is gated first, and the reading the gate
    passed (or gave up) on is appended to `clocks`. Those gated readings are
    what the run's clock warning keys on: a round-start reading is taken
    BEFORE the gate, right after the previous round's contended leg, so it
    shows the dip the gate then recovers from on every multi-round run and
    said nothing about the samples. GateAborted propagates.

    WITHOUT a `cool_floor` nothing is read here and `clocks` stays empty, so
    an ungated run has no gated reading to judge; main() falls back to the
    round-start readings in POWER_SAMPLES for that case.
    """
    if cool_floor:
        pct = wait_for_cool(cool_floor)
        if pct is not None and clocks is not None:
            clocks.append(pct)
    print("  [%s]" % label, end=" ", flush=True)
    return be.measure_decode(base, model, depth, tokens, timeout)


def _round_state(i):
    """Sample the box once at the top of round `i` and record it.

    One be.box_state() launch for all four values. This used to be two
    launches -- cpu_performance_pct() and then battery_state(), the second of
    which runs the same counter and discards it -- for readings one launch
    already returns; the file's own rule is that a reading paid for and then
    discarded is how that cost grows.
    """
    ac, charge, watts, clock = be.box_state()
    if charge is None and clock is None:
        return
    POWER_SAMPLES.append({"round": i + 1, "on_ac": ac, "charge_pct": charge,
                          "charge_w": watts, "clock_pct": clock})
    if clock is not None:
        print("  cpu clock %.1f%% of base (round start, before the gate)" % clock,
              flush=True)
    if charge is not None:
        # Three-way, because the flag is None when the class did not say. An
        # unreadable source used to render as BATTERY, which is the one word
        # an operator would act on.
        src = "AC" if ac else ("BATTERY" if ac is not None else "power source unreadable")
        print("  power     %s, pack %.0f%%%s"
              % (src, charge,
                 ", drawing %.1f W" % watts if watts is not None else ""),
              flush=True)


def paired_sweep(engines, a, make_load):
    """Interleave solo and contended samples, and report the PAIRED ratios.

    WHAT "SOLO" MEANS HERE, because it is narrower than the word suggests: the
    other engine's SERVER is still resident, it is merely not generating -- the
    load generator starts for the contended leg only. So this measures
    engine-with-an-idle-peer, not engine-alone. The 25-32% idle-peer penalty
    this docstring used to quantify did NOT survive its controlled re-run --
    the GPU solo row moved 18.20 against 18.47 with an idle poll:true NPU
    resident, which is nothing. The methodological point stands without the
    number: stopping the peer entirely is a different baseline and has to be
    measured deliberately, not inferred from this one.

    The ordering is the whole point. Measuring every solo first and every
    contended second puts all of any thermal decay into the contended half,
    where it is indistinguishable from contention and biases the result the
    same direction every time. Sampling solo and contended ADJACENTLY and
    taking the median of the per-pair ratios cancels drift that is slow
    relative to one pair, which is the shape thermal drift has.

    That is also why the median is taken over RATIOS rather than the ratio
    being taken over medians: the former keeps each solo matched to the
    contended sample nearest it in time, the latter throws that pairing away.
    Only the per-engine "keeps" figure has that property; see main() for what
    the aggregate lines are made from.

    Returns (per_engine, clocks, closing). `clocks` is the gated readings, one
    per gated leg -- so EMPTY on an ungated run (--cool-floor 0), whose clock
    evidence is the round-start readings in POWER_SAMPLES. A GateAborted from any gated leg STOPS the sweep: the rounds
    already taken are returned, and `closing` says where it stopped.
    """
    per_engine = {name: {"solo": [], "contended": [], "ratios": [],
                         "shed": 0, "served": 0, "failed": 0,
                         "first_failure": None,
                         "load_tokens_out": 0, "load_seconds": 0.0}
                  for name, _b, _m in engines}
    clocks = []
    # Which round produced each engine's FIRST solo sample. The closing
    # re-check compares against solo[0], and that is round 1's sample only if
    # round 1 landed one; a skipped opening leg used to let round 2's sample
    # pass as "the leg this run opened with" with nothing saying so.
    first_solo_round = {}

    try:
        for i in range(a.repeat):
            print("\n--- round %d/%d ---" % (i + 1, a.repeat), flush=True)
            _round_state(i)

            for name, base, model in engines:
                # cool_floor is threaded through explicitly. It used to default to
                # None here, which silently disabled the gate this harness's whole
                # method depends on -- the samples were taken across exactly the
                # thermal decay wait_for_cool exists to prevent.
                leg = (i + 1, name)
                solo = measure(base, model, a.depth, a.tokens, a.timeout,
                               "%s solo" % name, cool_floor=a.cool_floor,
                               clocks=clocks)

                other = next(e for e in engines if e[0] != name)
                gen = make_load(other)
                gen.start()
                try:
                    time.sleep(a.ramp)
                    # Deliberately NOT cooled: the other engine is already hot by
                    # design, so waiting for clock recovery here would either never
                    # return or would measure a half-loaded box. The gate belongs on
                    # the SOLO leg, which is the baseline every ratio divides by.
                    cont = measure(base, model, a.depth, a.tokens, a.timeout,
                                   "%s vs %s busy" % (name, other[0]))
                finally:
                    # try/finally because the generator is a live load on a SHARED
                    # box. An interrupt here used to skip stop() and leave a thread
                    # hammering the peer endpoint while main() wrote its results --
                    # load that then lands on whatever the next session starts.
                    gen.stop()

                rec = per_engine[name]
                rec["shed"] += gen.shed
                rec["served"] += gen.completed
                rec["failed"] += gen.failed
                if rec["first_failure"] is None:
                    rec["first_failure"] = gen.first_failure
                rec["load_tokens_out"] += gen.tokens_out
                rec["load_seconds"] += gen.seconds
                if solo is not None:
                    rec["solo"].append(solo)
                    first_solo_round.setdefault(name, i + 1)
                if cont is not None:
                    rec["contended"].append(cont)
                if solo and cont:
                    rec["ratios"].append(cont / solo)
                    print("    pair: %.2f -> %.2f t/s  (keeps %.1f%%)"
                          % (solo, cont, 100 * cont / solo), flush=True)
    except GateAborted as e:
        # Nothing after this point could be kept, so nothing after this point
        # is taken. The generator is never running here: only the SOLO leg is
        # gated, and it is gated before the generator starts.
        print("\n--- SWEEP STOPPED at round %d, %s solo: %s ---"
              % (leg[0], leg[1], e.why), flush=True)
        return per_engine, clocks, {"state": "gate-aborted", "engine": leg[1],
                                    "round": leg[0]}

    # THE CLOSING RE-CHECK. Re-run the leg this sweep OPENED with, last, under
    # the same gate -- an A/A whose only variable is elapsed time.
    #
    # It exists because the two controls already here cannot see this. The cool
    # gate tests the clock BEFORE a sample and says nothing during it, and a
    # leg takes a minute or two on a box that sustained load drives to 48.9% of
    # base -- so every sample can be gated at entry and still decay through its
    # own measurement. drift_note catches the decay afterwards, but only as a
    # strictly monotonic decline over 3+ solo samples: one noisy sample out of
    # order hides a real trend, and at --repeat 1 there is nothing for it to
    # compare at all.
    #
    # This produces a NUMBER instead, from the one comparison that isolates
    # box state: same engine, same depth, same token count, same gate, ~20
    # minutes apart.
    # ALWAYS a dict with a `state`, never None. Three different things used to
    # collapse into one "did not run" line -- the operator disabling it, the
    # closing leg failing, and the opening leg having produced nothing to
    # compare against. Those want opposite responses (respectively: none, look
    # at why it failed, look at why the whole sweep is empty), and a null in
    # the JSON additionally read as "the flag was off".
    if not (a.closing_recheck and engines):
        return per_engine, clocks, {"state": "disabled"}
    name, base, model = engines[0]
    opened_with = per_engine[name]["solo"]
    if not opened_with:
        return per_engine, clocks, {"state": "no-opening-sample",
                                    "engine": name}
    print()
    print("--- closing re-check: %s solo, the leg this run opened with ---"
          % name, flush=True)
    try:
        final = measure(base, model, a.depth, a.tokens, a.timeout,
                        "%s solo (closing)" % name, cool_floor=a.cool_floor,
                        clocks=clocks)
    except GateAborted as e:
        print("\n--- CLOSING RE-CHECK STOPPED: %s ---" % e.why, flush=True)
        return per_engine, clocks, {"state": "gate-aborted", "engine": name,
                                    "round": "closing"}
    if not final:
        return per_engine, clocks, {"state": "failed", "engine": name}
    return per_engine, clocks, {"state": "ok", "engine": name,
                                "first": opened_with[0], "final": final,
                                "first_round": first_solo_round[name],
                                "retained": final / opened_with[0]}


def closing_note(closing, tol_pct=10.0):
    """(message, suspect) comparing the reopened first leg to its first sample.

    SYMMETRIC, and that is not pedantry -- the two directions disqualify a run
    for opposite reasons and the fix differs:

      slower at the end   the box decayed across the run. Every contended
                          sample was taken later than the solo one it is
                          divided by, so the decay sits in the numerator and
                          contention is OVERSTATED.

      faster at the end   the OPENING sample was the degraded one, so every
                          baseline the ratios divide by is too low, and the
                          retention percentages are flattered.

    A tolerance rather than an equality: this box moves a few percent between
    any two samples, and a check that fires on ordinary noise is one the reader
    learns to skip -- the same reasoning as _probe_crosscheck's tol in
    bench_endpoint, which brackets its decode probe the same way.
    """
    state = (closing or {}).get("state")
    if state != "ok":
        why = {
            "disabled": "skipped by --no-closing-recheck",
            "no-opening-sample": "the opening leg produced no sample to "
                                 "compare against, so the whole sweep is "
                                 "suspect for that reason first",
            "failed": "the closing measurement itself failed",
            "gate-aborted": "the gate ABORTED (%s solo, round %s): the box was "
                            "on battery, waiting was futile and the sweep was "
                            "stopped there. Nothing measured on battery is "
                            "worth keeping, and the samples above stand only "
                            "if the box was on AC when they were taken"
                            % ((closing or {}).get("engine"),
                               (closing or {}).get("round")),
        }.get(state, "it did not run")
        # NOT the words the passing branch uses. "Not checked" reading like
        # "checked and fine" is the defect this harness keeps finding in its
        # own instruments.
        return ("note: no closing re-check -- %s. Nothing here says whether "
                "the box held across this sweep." % why), False
    drift = 100.0 * (closing["retained"] - 1.0)
    shape = ("%s solo opened at %.2f t/s and closed at %.2f (%+.1f%%)"
             % (closing["engine"], closing["first"], closing["final"], drift))
    first_round = closing.get("first_round")
    if first_round is not None and first_round != 1:
        # "Opened" is the first SUCCESSFUL solo sample, and that is round 1's
        # only when round 1 landed one. Say so, because an A/A whose "A" is
        # from the middle of the run is a shorter baseline than it looks.
        shape += (" -- NOTE: that opening sample is round %d's; round 1's %s "
                  "solo was skipped, so this brackets less of the run than a "
                  "full sweep would" % (first_round, closing["engine"]))
    if abs(drift) <= tol_pct:
        return ("closing re-check: %s -- within %.0f%%, so the box held and "
                "the ratios above stand." % (shape, tol_pct)), False
    if drift < 0:
        return ("SUSPECT: %s. The box DECAYED across this run. Contended "
                "samples were taken after the solo ones they divide, so that "
                "decay lands in the numerator and contention is overstated. "
                "Cool the box and re-run." % shape), True
    return ("SUSPECT: %s. The box got FASTER, so the OPENING sample was the "
            "degraded one -- every baseline the ratios divide by is too low "
            "and the retention percentages above are flattered. Cool the box "
            "and re-run." % shape), True


def shed_note(name, shed, served, failed=0, reason=None):
    """(message, suspect) about a contended leg's load generator.

    `served`, `shed` and `failed` are the generator's three outcomes for the
    requests it made while `name` was measured (see Load). They are counted
    apart because they decide whether the experiment happened at all: if
    nothing was served, the "contended" leg measured an idle box and the
    ratio comes out near 1.0, which reads as "no contention effect" rather
    than as "no contention".

      failed, none served   the peer was not reachable (or died, or answered
                            garbage) for the whole leg. SUSPECT.
      nothing at all        the generator never got a request off -- its
                            thread died before the first one, or the leg was
                            shorter than one request. SUSPECT for the same
                            reason: no load was applied.
      shed, none served     the peer answered 429 to every request. A lone
                            sequential client never gets that from a healthy
                            single-flight server, so something ELSE held the
                            peer for the whole leg. SUSPECT: the leg measured
                            against an unknown load.
      shed, some served     the peer was queued rather than contending for
                            part of the leg, which UNDERSTATES contention.
      failed, some served   part of the window was idle. Not backpressure --
                            the reason is named so the reader can tell a
                            refused connection from a request that outran the
                            generator's cap, which was still load.
    """
    if not shed and not failed:
        if served == 0:
            return ("SUSPECT: while %s was measured, the load generator "
                    "completed NOTHING and was refused nothing -- it never got "
                    "a request off (its thread died before the first one, or "
                    "the leg was shorter than one request). No load was "
                    "applied, so the contended leg measured an IDLE box and "
                    "any ratio near 1.0 means 'no load', not 'no contention'."
                    % name), True
        return None, False
    if served == 0 and failed:
        return ("SUSPECT: while %s was measured, the load generator got %d "
                "FAILED request(s) (first: %s)%s and ZERO completions. The "
                "peer was not reachable or not answering for the whole leg, "
                "so the contended leg measured an IDLE box and any ratio "
                "near 1.0 means 'no load', not 'no contention'. Check that "
                "the peer was serving before believing this run."
                % (name, failed, reason,
                   " plus %d shed" % shed if shed else "")), True
    if served == 0:
        return ("SUSPECT: while %s was measured, the other engine shed every "
                "one of %d request(s) and served none. A single sequential "
                "client never sees a 429 from a healthy single-flight server, "
                "so something ELSE held the peer for the whole leg (foreign "
                "traffic on this shared box, or a permit wedged behind a "
                "timed-out request). The peer was queued, not contending, "
                "and any ratio near 1.0 means 'no load', not 'no contention'."
                % (name, shed)), True
    parts = []
    if shed:
        parts.append("shed %d request(s): a 429 to a lone sequential client "
                     "means something else was in flight, so for those the "
                     "peer was queued rather than contending, which "
                     "UNDERSTATES contention" % shed)
    if failed:
        parts.append("FAILED %d request(s) (first: %s): not backpressure -- a "
                     "refused connection is an idle peer and understates "
                     "contention; a timeout is a request that outran the "
                     "generator's %ds cap and was still load"
                     % (failed, reason, LOAD_REQUEST_TIMEOUT_S))
    return ("note: while %s was measured, the other engine served %d and %s."
            % (name, served, "; ".join(parts))), False


def drift_note(values, what):
    """Flag a monotonic decline across rounds -- the thermal signature.

    Contention is a step change that appears when the other engine starts and
    vanishes when it stops. Thermal decay is a downward trend across the whole
    run that does not care which engine is busy. If the SOLO samples decline
    monotonically, the box was heating and every ratio in the run is suspect.
    """
    if len(values) < 3:
        return None
    if all(values[i] > values[i + 1] for i in range(len(values) - 1)):
        drop = 100 * (values[0] - values[-1]) / values[0]
        return ("%s declined monotonically across every round "
                "(%.2f -> %.2f, -%.1f%%) -- that is a THERMAL signature, not "
                "contention. Let the box cool and re-run."
                % (what, values[0], values[-1], drop))
    return None


# The pre-flight ping and the warmup are bounded separately from --timeout. The
# warmup used to inherit --timeout (1800 s) for a 4-token request, so a server
# that answered the ping and then wedged held the run for thirty minutes per
# engine before the first round.
PING_TIMEOUT_S = 60
WARMUP_TIMEOUT_S = 300

# The TCP connect asked after a failed ping (see _connect_problem). A refused
# connect comes back in ~2 s on Windows (its SYN retries) and at once
# elsewhere; this only bounds a host that drops the SYN outright.
CONNECT_TIMEOUT_S = 10

# Where each label's server is started, for the line that says to start it.
# The ports are the launchers' own and the flags' defaults.
LAUNCHERS = {"NPU": ("run-genie-server.ps1", 8123, "--npu"),
             "GPU": ("`run-llama-server.ps1 -Leg gpu`", 8124, "--gpu")}


def _connect_problem(base, timeout=CONNECT_TIMEOUT_S):
    """Why nothing took a TCP connection at `base`, or None if something did.

    be.chat returns one None for two opposite facts: nothing took the
    connection (refused, timed out, a host that does not resolve), and
    something took it and gave no usable answer (an HTTP error, a completion
    with no usage block). The ping's refusal used to say "is not answering,
    or reports no usage -- start it first" for both, which is the wrong
    advice for the second and, for the first, for a genie_server still in
    its 11-35 s load: it holds the port and refuses connections until the
    model is resident, the same symptom as nothing running at all.

    Asked only AFTER a failed ping, never as a gate in front of it: a bare
    socket ignores the proxy settings urllib honours, so as a gate it could
    refuse a base the ping itself reaches. main() has already refused a base
    that is not an http(s) URL with a host, so urlsplit here has one.
    """
    parts = urllib.parse.urlsplit(base)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname, port), timeout=timeout).close()
    except OSError as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    return None

# Printed as argparse's epilog, which is NOT %-expanded (argparse only formats
# an epilog that contains "%(prog)"), so a percent sign is written once here.
# Doubled, it rendered as a literal "13-20%% charge" in --help.
#
# BOTH variables this tool obeys, not only the one it reads itself. This used
# to list GENIE_LOW_CHARGE_PCT alone, under a docstring saying it was the one
# variable read -- while bench_endpoint's decode floor, which that module reads
# from the environment at import, sits under the only measurement this tool
# makes. A shell exporting GENIE_MIN_DECODE_STEPS (the documented way to let
# bench_endpoint measure a short window) moved this tool's floor too, and
# nothing here said it could.
ENV_HELP = """\
environment:
  GENIE_LOW_CHARGE_PCT    pack percentage below which a run is flagged as not
                          a settled baseline EVEN ON AC (default 25). Measured
                          on this box: at 13-20% charge, CPU pp512 comes back
                          ~58 against a settled 130 while decode barely moves.
                          Advisory, never fatal. A non-numeric value is
                          reported at startup and the default used.
  GENIE_MIN_DECODE_STEPS  fewest decode steps a rate may rest on (default 16).
                          Read by bench_endpoint, not here, but it governs
                          every sample this tool takes: a decode window
                          shorter than this is REFUSED rather than reported.
                          A --tokens below it could never produce a sample,
                          so that is refused at startup. A non-integer value
                          is reported at startup and the default used.
"""


def _parser():
    ap = argparse.ArgumentParser(epilog=ENV_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npu", default="http://127.0.0.1:8123",
                    help="Genie NPU server base URL (default %(default)s)")
    ap.add_argument("--gpu", default="http://127.0.0.1:8124",
                    help="llama-server GPU leg base URL (default %(default)s, "
                         "where `run-llama-server.ps1 -Leg gpu` serves; 8080 "
                         "is that launcher's CPU leg, `-Leg cpu`). Whatever "
                         "answers here is labelled GPU -- check the model id "
                         "the server reports, printed beside the label")
    ap.add_argument("--npu-model", default="qwen3-4b-npu",
                    help="model id sent to the NPU server (default "
                         "%(default)s)")
    ap.add_argument("--gpu-model", default="default",
                    help="model id sent to the GPU server (default "
                         "%(default)s, which llama-server accepts for whatever "
                         "it has loaded -- so the id it REPORTS is printed and "
                         "written to the JSON as served_model)")
    ap.add_argument("--depth", type=int, default=500,
                    help="context depth for every measurement")
    ap.add_argument("--tokens", type=int, default=120,
                    help="decode steps per measurement")
    ap.add_argument("--load-depth", type=int, default=None,
                    help="context depth of the background load's requests "
                         "(default: --depth). The default shapes the load like "
                         "the measurement, so each of its requests is a full "
                         "prefill before its decode steps and the peer is "
                         "bus-bound for only part of the window; a shallow "
                         "depth with a large --load-tokens keeps it decoding. "
                         "A reshaped load is sent once to each server in the "
                         "warmup, and the run refuses to start if either "
                         "cannot serve it")
    ap.add_argument("--load-tokens", type=int, default=None,
                    help="decode steps per background load request (default: "
                         "--tokens). The generator's tokens per second of leg "
                         "time is recorded per engine so runs with different "
                         "load shapes can be told apart")
    ap.add_argument("--repeat", type=int, default=3,
                    help="rounds; each round takes one solo and one contended "
                         "sample per engine, interleaved (default %(default)s)")
    ap.add_argument("--ramp", type=float, default=8.0,
                    help="seconds to let the background load get in flight "
                         "before the contended measurement starts")
    ap.add_argument("--timeout", type=float, default=1800,
                    help="per-request timeout for the measurements, seconds "
                         "(default %%(default)s). The load generator caps its "
                         "own at %d (as does the warmup's one request in a "
                         "reshaped load's shape) and the pre-flight ping and "
                         "warmup at %d and %d"
                         % (LOAD_REQUEST_TIMEOUT_S, PING_TIMEOUT_S,
                            WARMUP_TIMEOUT_S))
    ap.add_argument("--npu-weights-gb", type=float, default=None,
                    help="weight bytes the NPU streams per token, GB. Enables "
                         "the derived-bandwidth report")
    ap.add_argument("--gpu-weights-gb", type=float, default=None,
                    help="weight bytes the GPU streams per token, GB (the GGUF "
                         "size is a good proxy)")
    ap.add_argument("--peak-bw-gbs", type=float, default=None,
                    help="theoretical bus bandwidth, GB/s, to compare demand "
                         "against (X1E80100 LPDDR5x-8448 x 128-bit = 135)")
    ap.add_argument("--no-closing-recheck", dest="closing_recheck",
                    action="store_false",
                    help="skip re-running the opening leg at the end. That "
                         "re-run is the only check that sees decay DURING a "
                         "sample -- --cool-floor gates before one and says "
                         "nothing after -- so skipping it costs one leg and "
                         "buys back no confidence")
    ap.add_argument("--closing-tol", type=float, default=10.0,
                    help="%% the closing re-check may differ from the opening "
                         "sample before the run is called suspect, in EITHER "
                         "direction (default %(default)s)")
    # The one interpolated sentence is formatted on its own and joined with
    # `+`. Formatting the WHOLE help string spent its `%%` escapes one level
    # early, leaving argparse a bare "% of base" to expand -- which it refuses
    # as "badly formed help string" when the argument is added (3.14) or when
    # --help is rendered (before), so main() died before parsing anything.
    # The last sentence carries a percent sign THROUGH its own formatting, so
    # it is written `%%%%`: one level is spent here, argparse spends the other.
    ap.add_argument("--cool-floor", type=float, default=DEFAULT_COOL_FLOOR,
                    help="wait for the clock to recover to this %% of base "
                         "before each SOLO sample (default %(default)s; 0 "
                         "disables). "
                         + "The wait gives up after %d s and PROCEEDS, "
                           "recording that the sample is suspect. " % COOL_LIMIT_S
                         + "Sustained load drops this box to 48.9%%, and an "
                           "ungated sweep turns that decay into a fake depth "
                           "curve. "
                         + "With the gate off nothing waits a dip out, so the "
                           "round-start clock readings are judged instead, "
                           "against %.0f%%%% of base, and the warning says the "
                           "run was ungated" % DEFAULT_COOL_FLOOR)
    ap.add_argument("--min-free-gb", type=float, default=4.0,
                    help="refuse below this much free physical RAM. This is "
                         "HEADROOM BEYOND the engines under test, which are "
                         "SUPPOSED to be resident: two ~3 GB models leave 6-7 "
                         "GB free on a 31.6 GB box, and that is a healthy "
                         "contention run rather than a loaded one. The old "
                         "default of 8.0 refused the only experiment this "
                         "harness exists to run")
    ap.add_argument("--allow-loaded", action="store_true",
                    help="run anyway on a loaded box; stamps results LOADED")
    ap.add_argument("--json", help="also write the results to this file")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --json if it already exists")
    return ap


def _validate(a, ap):
    """Reject values that would spend the box before failing.

    argparse checks types, not ranges. --repeat 0 used to pay both pings and
    both warmups and then exit 0 reporting both engines incomplete; --ramp -1
    raised from time.sleep AFTER the solo leg, with the generator already
    started. Each exits 2 through ap.error, before the memory gate.

    --npu and --gpu go through bench_endpoint.base_url_problem, the check
    that tool applies to its --base, in the same words. `--npu
    127.0.0.1:8123` (no scheme) used to fail the ping as "unknown url type"
    and then tell the operator to start a server that was already answering
    on that port.
    """
    for flag, base in (("--npu", a.npu), ("--gpu", a.gpu)):
        problem = be.base_url_problem(flag, base)
        if problem:
            ap.error(problem)
    if a.repeat < 1:
        ap.error("--repeat must be >= 1 (got %d): zero rounds would pay both "
                 "pings and both warmups to measure nothing" % a.repeat)
    for flag, v in (("--depth", a.depth), ("--tokens", a.tokens),
                    ("--load-depth", a.load_depth), ("--load-tokens", a.load_tokens)):
        if v is not None and v < 1:
            ap.error("%s must be >= 1 (got %d)" % (flag, v))
    if a.tokens < be.MIN_DECODE_STEPS:
        # The floor is bench_endpoint's, read from GENIE_MIN_DECODE_STEPS at
        # its import. A measurement asks for --tokens decode steps and can
        # come back with fewer, never more, so below the floor EVERY leg is
        # refused -- after its two requests have been paid for -- and the run
        # used to finish the whole sweep that way: both engines "incomplete",
        # exit 0, and a refusal line that gave the floor without naming the
        # variable that set it.
        ap.error("--tokens %d is below the decode floor of %d steps "
                 "(GENIE_MIN_DECODE_STEPS, read by bench_endpoint; default "
                 "16): every measurement would be REFUSED as too short a "
                 "window to be a rate, so the sweep would spend the box and "
                 "report both engines incomplete. Raise --tokens or lower "
                 "the variable" % (a.tokens, be.MIN_DECODE_STEPS))
    if a.ramp < 0:
        ap.error("--ramp must be >= 0 (got %g): a negative sleep would raise "
                 "after the solo leg with the load generator already running"
                 % a.ramp)
    if a.timeout <= 0:
        ap.error("--timeout must be > 0 (got %g)" % a.timeout)
    if a.cool_floor < 0 or a.closing_tol < 0 or a.min_free_gb < 0:
        ap.error("--cool-floor, --closing-tol and --min-free-gb must be >= 0")


def _json_dir_problem(path):
    """Why --json cannot be written, as a sentence, or None if it can be.

    Writability is PROBED -- a file created in the directory and removed
    again -- rather than asked. os.access(W_OK) on Windows reports the
    read-only attribute and answers True for every directory, including one
    an ACL denies and one on a full volume, so the probe is the only answer
    worth having. It costs one file create, once, before a run that costs 20+
    minutes of a shared box.
    """
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        return "--json %s: %s is not a directory" % (path, d)
    try:
        fd, probe = tempfile.mkstemp(dir=d, prefix=".bench_contention-")
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        return "--json %s cannot be written: %s" % (path, exc)
    return None


def _timestamped_beside(path):
    """`path` with a UTC timestamp in front of its extension.

    Beside the path that was asked for, not in a temp directory: the operator
    is looking at the directory they typed, and this has to turn up in it.
    """
    root, ext = os.path.splitext(path)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return "%s.%s%s" % (root, stamp, ext)


def _dump_json(path, record):
    """The one place the record is written, so the intended path and the
    fallback are written identically -- and so a test can make a write fail
    without a read-only directory."""
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def _write_record(path, record, force):
    """Write the run's record, never discarding it. (path_written, note).

    `note` is None only when the record landed on the path that was asked
    for. Anything else is one sentence for the operator AND a non-zero exit,
    because a wrapper that branches on the exit code must not read a fallback
    write as the file it named.

    Both of the outcomes this replaces threw the whole artifact away after
    the box had been spent. A file that appeared DURING the sweep (another
    session writing the same path) printed "NOT writing" and exited 0, so
    nothing downstream could tell that the file it asked for is someone
    else's. Any other OSError -- a directory that went away under the run, a
    full volume, a file another process holds open -- came out of open() as a
    traceback. The startup check catches the common case and cannot catch
    these two: one is a race by definition, and the other only shows up when
    the write is attempted.
    """
    if os.path.exists(path) and not force:
        why = ("%s appeared during the run (another session writing the same "
               "path) and was NOT overwritten" % path)
    else:
        try:
            _dump_json(path, record)
            return path, None
        except OSError as exc:
            why = "%s could not be written (%s)" % (path, exc)
    fallback = _timestamped_beside(path)
    try:
        _dump_json(fallback, record)
    except OSError as exc:
        return None, ("%s, and the fallback %s failed too (%s), so this run's "
                      "numbers exist only in the terminal above"
                      % (why, fallback, exc))
    return fallback, "%s, so this run's results went to %s" % (why, fallback)


def main():
    ap = _parser()
    a = ap.parse_args()
    _validate(a, ap)

    if a.json and os.path.exists(a.json) and not a.force:
        # Checked BEFORE the sweep, not after it. A run here costs 20+ minutes
        # of a shared box, and the refusal used to come at the end -- the
        # operator learned the file would not be written only after spending
        # the box. The file is kept for the same reason as ever: some of these
        # numbers have proved unreproducible, and clobbering a previous result
        # to save a flag is the wrong trade.
        print("REFUSING TO START: --json %s already exists and --force was not "
              "given. Pass --force to overwrite it, or a different --json path."
              % a.json, file=sys.stderr)
        return 2

    if a.json:
        # The DIRECTORY, checked for the same reason as the file above and in
        # the same breath. Only the file-exists case had been moved up front:
        # `--json results\c.json` typed at a prompt with no results\ ran the
        # pings, the warmups and every leg, printed the whole CONTENTION
        # block, and then came out of open() as a bare FileNotFoundError
        # traceback with the artifact still in memory. The sibling tool
        # checks its own --out directory here too (bench_servers, "NOT
        # starting: ... is not a directory").
        problem = _json_dir_problem(a.json)
        if problem:
            print("REFUSING TO START: %s. The results file is the only place "
                  "power_samples, box_samples, gate_clock_pct and the "
                  "per-engine counts exist, so a run that cannot write it is "
                  "not worth 20+ minutes of this box." % problem,
                  file=sys.stderr)
            return 2

    # Cleared per run, not merely appended to. These are module state (one of
    # them bench_endpoint's) and main() copies them into the run's record, so a
    # second run in the same process would inherit the first one's notes and
    # samples and publish them in its JSON -- a warning attached to a run that
    # never earned it, which is the record-vs-reality drift the rest of this
    # harness exists to prevent.
    GATE_NOTES.clear()
    POWER_SAMPLES.clear()
    del be.BOX_SAMPLES[:]

    free = free_physical_gb()
    shown = "unknown" if free is None else "%.2f GB" % free
    print("free physical memory: %s (need >= %.1f GB)" % (shown, a.min_free_gb))
    if free is None and not a.allow_loaded:
        # Its own message. This used to fall into the "box this loaded" refusal
        # below and send the operator to --min-free-gb, which cannot help: no
        # threshold passes an unknown. The gate has nothing to evaluate, and
        # the only honest escape is the one that stamps the output.
        print("\nREFUSING TO RUN: free physical memory could not be read, so "
              "the quiet-box gate cannot be evaluated. --min-free-gb cannot "
              "help here. Fix the reader (GlobalMemoryStatusEx on Windows, "
              "/proc/meminfo elsewhere), or pass --allow-loaded, which stamps "
              "every result LOADED because the box state is unknown.",
              file=sys.stderr)
        return 2
    loaded = free is None or free < a.min_free_gb
    if loaded and not a.allow_loaded:
        print("\nREFUSING TO RUN on a box this loaded.", file=sys.stderr)
        print("Every retracted number on this hardware was measured in exactly "
              "this state -- NPU prefill was understated 3.3x.", file=sys.stderr)
        # The trap this message used to set. --allow-loaded is the WRONG escape
        # for a contention run: the two engines under test are ~3 GB each and
        # are SUPPOSED to be resident, so a legitimate run sits at 6-7 GB free
        # and trips this gate. Sending the operator to --allow-loaded then
        # stamps LOADED on output whose only load is the experiment itself, and
        # that stamp reads as untrustworthy for entirely the wrong reason.
        # Reported by the session that hit it running the poll A/B.
        print("\nIf the only things resident are the two engines under test, "
              "that is a HEALTHY contention run and this gate is set too high "
              "-- lower --min-free-gb (it measures headroom BEYOND the "
              "engines) rather than reaching for --allow-loaded, which stamps "
              "LOADED on results that do not deserve it.", file=sys.stderr)
        print("Use --allow-loaded only when something OTHER than the engines "
              "under test is holding the box.", file=sys.stderr)
        return 2
    if loaded:
        print("!! LOADED BOX -- these are NOT baselines and must not be quoted "
              "as such.", flush=True)

    engines = [("NPU", a.npu, a.npu_model), ("GPU", a.gpu, a.gpu_model)]
    # What each label actually pointed at, for the JSON. The label is hardcoded
    # and the harness cannot tell a CPU leg from a GPU one; the base URL, the
    # window and the id the server REPORTS are what let a reader audit it
    # after the fact -- none of which the artifact used to carry.
    engine_meta = {}
    for name, base, model in engines:
        r = be.chat(base, model, "ping", 1, PING_TIMEOUT_S)
        if r is None:
            # chat() returns None both when nothing took the connection and
            # when something did and answered unusably -- an HTTP error, or no
            # usage (GenieAPIService reports all zeros; such a server was
            # never measurable here, since every rate is formed from the
            # server's own counts). They need opposite advice, so a TCP
            # connect tells them apart (see _connect_problem). A refusal is
            # still not proof of absence: a genie_server mid-load refuses too,
            # so that line names both readings (be.STILL_LOADING) before it
            # says to start one.
            refused = _connect_problem(base)
            if refused:
                launcher, port, flag = LAUNCHERS[name]
                print("\n%s at %s is not answering: nothing listening (%s) -- "
                      "either it is not running, or %s. Otherwise start it "
                      "first: %s serves on %d, so pass %s to match if yours "
                      "is elsewhere."
                      % (name, base, refused, be.STILL_LOADING, launcher,
                         port, flag), file=sys.stderr)
            else:
                print("\n%s at %s took the connection but did not answer the "
                      "ping with a usable completion (the reason is on the "
                      "line above) -- something is serving there, so do not "
                      "start another on that port. A server whose usage "
                      "block is missing or all zeros cannot be measured by "
                      "this tool." % (name, base), file=sys.stderr)
            return 2
        served = r.get("model")
        w = be.n_ctx(base)
        engine_meta[name] = {"base": base, "model": model,
                             "served_model": served, "n_ctx": w}
        print("%s %s  model=%s (server reports %s)  n_ctx=%s"
              % (name, base, model, served or "no id", w if w else "unknown"))

    load_depth = a.load_depth if a.load_depth is not None else a.depth
    load_tokens = a.load_tokens if a.load_tokens is not None else a.tokens
    reshaped = (load_depth, load_tokens) != (a.depth, a.tokens)

    print("\nwarmup", flush=True)
    for name, base, model in engines:
        if be.chat(base, model, be.prompt_of(a.depth), 4,
                   min(a.timeout, WARMUP_TIMEOUT_S)) is None:
            print("\n%s at %s answered the ping but not the warmup (a %d-token "
                  "prompt, %d s allowed) -- not measuring against it."
                  % (name, base, a.depth, min(a.timeout, WARMUP_TIMEOUT_S)),
                  file=sys.stderr)
            return 2

    if reshaped:
        # The load shape is proven servable HERE, not discovered by the sweep.
        # Until --load-depth/--load-tokens existed the load always had the
        # shape the warmup above had just proven; with them, a shape a peer
        # cannot serve -- `--load-depth 3000` against a GPU leg started at the
        # docstring's own `-c 1024`, or the flag's documented use, a shallow
        # depth with `--load-tokens 4096`, against a 4096 window (genie_server
        # answers 400 once prompt + max_tokens + its margin pass n_ctx) --
        # passed the ping and the warmup, ran every round with each generator
        # request failing, printed a VERDICT from legs measured against an
        # idle peer, and exited 0: spend the box, then fail, which is what
        # every other refusal here was moved ahead of the sweep to avoid.
        #
        # The generator's OWN request, not a 4-token probe at the load depth:
        # the cap is half of what a window refuses on, so a short probe would
        # pass exactly the shape the flag invites. And under the generator's
        # own timeout, because a request that overruns it is counted as failed
        # every time it is sent. Every engine gets it, since each is the
        # other's peer. It costs one generator request per server, which the
        # sweep then repeats for the whole of its contended half.
        load_timeout = min(a.timeout, LOAD_REQUEST_TIMEOUT_S)
        for name, base, model in engines:
            print("  load shape on %s: depth %d x %d tokens"
                  % (name, load_depth, load_tokens), flush=True)
            if be.chat(base, model, be.prompt_of(load_depth), load_tokens,
                       load_timeout) is None:
                w = engine_meta[name]["n_ctx"]
                print("\n%s at %s served the warmup but not the background "
                      "load's shape (a %d-token prompt x %d tokens, %d s "
                      "allowed; its window is n_ctx=%s) -- not starting the "
                      "sweep. The generator sends exactly this request for "
                      "the whole of every contended leg, so each one would "
                      "fail the same way and the legs would be measured "
                      "against an idle peer. Reshape --load-depth/"
                      "--load-tokens to fit, or restart the server with a "
                      "window that holds it."
                      % (name, base, load_depth, load_tokens, load_timeout,
                         w if w else "unknown"), file=sys.stderr)
                return 2

    def make_load(other):
        return Load(other[1], other[2], load_depth, load_tokens, a.timeout)

    print("\nPAIRED SWEEP (solo and contended interleaved, per round)",
          flush=True)
    if reshaped:
        print("  background load shaped depth %d x %d tokens (the measurement "
              "is %d x %d)" % (load_depth, load_tokens, a.depth, a.tokens),
              flush=True)
    per_engine, clocks, closing = paired_sweep(engines, a, make_load)
    aborted = closing.get("state") == "gate-aborted"

    solo = {n: statistics.median(r["solo"]) for n, r in per_engine.items() if r["solo"]}
    contended = {n: statistics.median(r["contended"])
                 for n, r in per_engine.items() if r["contended"]}
    ratios = {n: statistics.median(r["ratios"])
              for n, r in per_engine.items() if r["ratios"]}

    print("\n%s" % ("=" * 64))
    print("CONTENTION%s%s" % ("  [LOADED BOX -- NOT A BASELINE]" if loaded else "",
                              "  [SWEEP STOPPED BY THE GATE]" if aborted else ""))
    print("=" * 64)
    for name, _b, _m in engines:
        r = per_engine[name]
        if name in ratios:
            print("  %-4s solo %7.2f   contended %7.2f   keeps %5.1f%%  "
                  "(paired median of %d)"
                  % (name, solo[name], contended[name], 100 * ratios[name],
                     len(r["ratios"])))
        else:
            # Worded by what is actually missing. "every measurement was
            # skipped" used to print here for an engine whose solo samples
            # all landed and whose contended ones were shed -- the commonest
            # asymmetric shape -- and the landed samples were then not shown.
            print("  %-4s incomplete: no round produced BOTH a solo and a "
                  "contended sample (solo landed %d, contended landed %d)"
                  % (name, len(r["solo"]), len(r["contended"])))
        print("       solo samples      %s"
              % (", ".join("%.2f" % v for v in r["solo"]) or "none"))
        print("       contended samples %s"
              % (", ".join("%.2f" % v for v in r["contended"]) or "none"))
        # The generator's duty cycle, so a reader can see how much decode the
        # peer actually did during this engine's contended legs. tokens_out
        # was accumulated for this and then never read.
        secs = r["load_seconds"]
        tps = r["load_tokens_out"] / secs if secs > 0 else None
        r["load_tokens_per_s"] = tps
        print("       peer load         served %d, shed %d, failed %d; %d "
              "tokens over %.0f s%s"
              % (r["served"], r["shed"], r["failed"], r["load_tokens_out"],
                 secs, " = %.1f t/s of leg time" % tps if tps is not None else ""))

    warnings = dedupe_notes(GATE_NOTES)
    note, suspect = closing_note(closing, a.closing_tol)
    print()
    print("  %s" % note)
    if suspect:
        warnings.append(note)
    for name, _b, _m in engines:
        w = drift_note(per_engine[name]["solo"], "%s solo decode" % name)
        if w:
            warnings.append(w)
    # Keyed on the GATED readings and on the floor the gate used, so the two
    # instruments agree on what "limited" means. This used to test the
    # round-start readings against a literal 95 while the gate accepted 92:
    # a run steady at 92-94% passed every gate and was still stamped limited,
    # and from round 2 on the round-start reading is the post-contended dip
    # the gate then waits out, so the stamp landed on essentially every
    # multi-round run. A gated reading below the floor can only mean the gate
    # gave up (or was aborted), which is the case worth a warning.
    if a.cool_floor:
        if clocks and min(clocks) < a.cool_floor:
            warnings.append(
                "a gated SOLO sample was taken at %.1f%% of base, below the "
                "--cool-floor of %.0f%% (gate readings: %s) -- the gate gave "
                "up waiting, so the package was power- or thermally-limited "
                "and part of any measured slowdown is not contention."
                % (min(clocks), a.cool_floor,
                   ", ".join("%.0f" % c for c in clocks)))
    else:
        # The UNGATED run, which the re-keying above left with no clock
        # judgement at all: measure() reads the clock only inside the gate, so
        # `clocks` is empty here by construction and a box sitting at 61% of
        # base wrote `warnings: []` where the round-start test had warned.
        # The argument for dropping the round-start readings -- they show the
        # dip the gate then waits out -- is an argument about a GATED run. With
        # the gate off nothing waits that dip out: the next solo sample is
        # taken AT that reading, which makes it the best evidence this run
        # has, and it is already in POWER_SAMPLES. Judged against the floor
        # the gate would have used by default, and worded as ungated, because
        # "the gate gave up" would be a claim about a gate that never ran.
        starts = [s["clock_pct"] for s in POWER_SAMPLES
                  if s["clock_pct"] is not None]
        if starts and min(starts) < DEFAULT_COOL_FLOOR:
            warnings.append(
                "UNGATED run (--cool-floor 0): the round-start clock dipped "
                "to %.1f%% of base, below the %.0f%% the gate holds a solo "
                "sample to by default (round-start readings: %s). Nothing "
                "waited that out, so solo samples were taken on a package "
                "that was power- or thermally-limited and part of any "
                "measured slowdown is not contention."
                % (min(starts), DEFAULT_COOL_FLOOR,
                   ", ".join("%.0f" % c for c in starts)))
    # The gate ABORT is the only thing that stops a run for being on battery,
    # and three ordinary ways of running leave it unable to fire: --cool-floor
    # 0 runs no gate at all; a clock at or above the floor returns from
    # power_limited_note before power_reading() is ever called; and an
    # unreadable counter returns wait_for_cool on its first None poll, before
    # power_limited_note is consulted. In each of those a run taken entirely on
    # battery finished with `warnings: []` and exit 0 -- a clean-looking
    # artifact for a run this file's own rule ("nothing measured on battery is
    # worth keeping", and the header's promise that the gate ABORTS for it)
    # would have refused. The source was recorded per round from the start and
    # read by nothing but the JSON. `is False` and not falsy: None means the
    # power class did not say, and an unreadable source rendering as BATTERY is
    # the defect _round_state already fixed for its own print.
    battery_rounds = [s["round"] for s in POWER_SAMPLES if s["on_ac"] is False]
    if battery_rounds:
        warnings.append(
            "this run was ON BATTERY for %d of %d round(s) (round %s) and no "
            "gate stopped it. Nothing measured on battery is worth keeping: "
            "the clock cannot recover by waiting, so these samples are "
            "power-limited by an amount nothing here measured. Plug in AC and "
            "re-run."
            % (len(battery_rounds), len(POWER_SAMPLES),
               ", ".join(str(r) for r in battery_rounds)))
    charges = [s["charge_pct"] for s in POWER_SAMPLES if s["charge_pct"] is not None]
    if charges:
        lo, hi = min(charges), max(charges)
        # Two INDEPENDENT checks. The movement warning used to be an `elif`
        # of the low-pack one, so it was suppressed exactly when a run had
        # both problems -- a deeply discharged pack that charged 20+ points
        # during the sweep -- and those describe different defects.
        if lo < LOW_CHARGE_PCT:
            warnings.append(
                "the pack was at %.0f%% during this run (range %.0f-%.0f%%). "
                "Below ~%.0f%% this box halves prefill while decode holds, so "
                "a prefill-sensitive comparison taken here is not a settled "
                "baseline even on AC."
                % (lo, lo, hi, LOW_CHARGE_PCT))
        if hi - lo >= 20:
            # Not about the level but the MOVEMENT: a run spanning 30% to 60%
            # was measured under two different power regimes, and averaging
            # across them hides that as ordinary noise.
            #
            # Reports the EXTREMES the condition actually fired on, not the
            # first and last readings. Those differ whenever the trajectory is
            # not monotonic, and the endpoint version had the message
            # contradicting its own trigger -- "moved 60% -> 62%" printed above
            # a warning raised because the run spanned 40-62%.
            warnings.append(
                "the pack spanned %.0f%%-%.0f%% across this run (opened %.0f%%, "
                "closed %.0f%%). Charge state changed under the measurement, so "
                "legs taken early and late are not strictly comparable."
                % (lo, hi, charges[0], charges[-1]))
    draws = [s["charge_w"] for s in POWER_SAMPLES if s["charge_w"] is not None]
    if draws and max(draws) - min(draws) >= 8:
        # Recorded per round but previously never read, which made it dead
        # weight in the artifact. It needs a frame, because the charge
        # controller moves a LOT on its own: measured on this box at a
        # roughly constant charge level, draw ranged 28-41.9 W across
        # minutes while read-to-read noise was ~1 W. Any percentage anchored
        # to one sample of this is biased by which sample it happened to
        # anchor on, so report the range and let the reader choose.
        warnings.append(
            "charge draw ranged %.1f-%.1f W during this run (opened %.1f). "
            "That is the charge controller moving, not the workload -- do "
            "not anchor a shed percentage to any single reading of it."
            % (min(draws), max(draws), draws[0]))

    bw = {}
    if a.npu_weights_gb or a.gpu_weights_gb:
        # Decode streams the full weight set per token on a dense model, so
        # rate x weight_bytes is the bandwidth that engine is actually pulling.
        # Printing it turns an opaque t/s into something comparable against the
        # bus ceiling -- and if the two engines' combined demand sits far below
        # that ceiling while contention is nonetheless severe, the cause is NOT
        # bandwidth and the shared-bus explanation has to be abandoned.
        print("  %s" % ("-" * 60))
        print("  DERIVED BANDWIDTH (decode rate x weight bytes)")
        weights = {"NPU": a.npu_weights_gb, "GPU": a.gpu_weights_gb}
        for name in ("NPU", "GPU"):
            g = weights.get(name)
            if g and name in solo and name in contended:
                bw[name] = {"solo_gbs": solo[name] * g,
                            "contended_gbs": contended[name] * g}
                print("    %-4s %6.1f GB/s solo   %6.1f GB/s contended  "
                      "(%.2f GB of weights)"
                      % (name, bw[name]["solo_gbs"],
                         bw[name]["contended_gbs"], g))
        if len(bw) == 2:
            tot_solo = sum(v["solo_gbs"] for v in bw.values())
            tot_cont = sum(v["contended_gbs"] for v in bw.values())
            print("    combined demand   %6.1f GB/s (if additive)   "
                  "%6.1f GB/s (actual, both hot)" % (tot_solo, tot_cont))
            if a.peak_bw_gbs:
                print("    bus peak          %6.1f GB/s -- actual combined is "
                      "%.0f%% of peak" % (a.peak_bw_gbs,
                                          100 * tot_cont / a.peak_bw_gbs))
                # Named per engine: the condition is on the WORST engine, and
                # "engines lost" used to print when one lost 20% and the
                # other kept 95%.
                lost = ["%s kept %.0f%%" % (n, 100 * v)
                        for n, v in ratios.items() if v < 0.8]
                if lost and tot_cont < 0.5 * a.peak_bw_gbs:
                    warnings.append(
                        "%s lost >20%% throughput (%s) while the pair together "
                        "used only %.0f%% of peak bandwidth -- the bottleneck "
                        "is NOT the memory bus. Suspect a shared power budget, "
                        "DVFS, or memory-controller latency rather than raw "
                        "bandwidth."
                        % (" and ".join(n for n, v in ratios.items() if v < 0.8),
                           "; ".join(lost), 100 * tot_cont / a.peak_bw_gbs))

    if len(ratios) == 2:
        # Aggregate uses each engine's CONTENDED rate, since that is what the
        # pair actually delivers when both are hot. Comparing it to the faster
        # engine's SOLO rate is the decision: below it, a second hot engine
        # costs throughput and only buys concurrency and failover.
        #
        # UNPAIRED, unlike "keeps" above, and said so on the output. These are
        # medians of the raw solo and contended lists, which paired_sweep's
        # docstring explains "throws the pairing away" -- and there is no
        # paired alternative here, because the two engines' contended rates
        # come from different legs and are never measured simultaneously; the
        # "aggregate" is a construct either way. Slow drift therefore does
        # NOT cancel in these lines the way it does in the per-engine ratio.
        agg = sum(contended.values())
        best_solo = max(solo.values())
        total_solo = sum(solo.values())
        print("  %s" % ("-" * 60))
        print("  aggregate while both hot   %7.2f t/s" % agg)
        print("  sum of solo rates          %7.2f t/s  (additive ideal)"
              % total_solo)
        print("  efficiency vs additive     %7.1f%%" % (100 * agg / total_solo))
        print("  best single engine solo    %7.2f t/s" % best_solo)
        # RATIO WITH A CONDITION, and the condition is easy to violate.
        # Dividing by best_solo is only meaningful when the baseline is
        # INDEPENDENT of whatever is being varied between runs. Measured
        # 2026-08-24: comparing poll:true against poll:false, poll:true scored
        # HIGHER on this line in all three pairs (1.57 / 1.55 / 1.79) while
        # delivering ~20% less absolute throughput -- because poll:true also
        # degrades the single-engine baseline it is divided by, and a smaller
        # denominator flatters the quotient. Aggregate t/s has no such
        # dependence, which is why it is printed first and above.
        print("  speedup vs best engine     %7.2fx  <- compare across runs ONLY"
              % (agg / best_solo))
        print("     if the baseline is independent of what changed between them;")
        print("     otherwise compare aggregate t/s, which has no denominator.")
        print("     (these four lines are medians of the raw sample lists, "
              "UNPAIRED;")
        print("     only the per-engine 'keeps' figure is drift-cancelled.)")
        if agg < best_solo:
            print("\n  VERDICT: running both is SLOWER than the best engine alone.")
            print("  Route to a second engine for concurrency and failover only,")
            print("  never for throughput.")
            # Before believing this, check the flag. A poll:true bundle
            # busy-polls the HTP on ~2.8 cores and starves the OpenCL
            # backend's per-token dispatch, taking about a quarter of the
            # concurrency win (1.70x vs 1.26x over the best single engine). A
            # net-loss verdict on this hardware is far more likely to be that
            # flag than a real hardware limit -- the 0.78x this comment used to
            # cite did not reproduce under a controlled re-run.
            print("  FIRST check QnnHtp/poll in the bundle's genie_config.json:")
            print("  poll:true gives up ~1/4 of the concurrency win; both are gains.")
        else:
            print("\n  VERDICT: two hot engines beat the best single engine by "
                  "%.2fx." % (agg / best_solo))
            # No reference figure printed here. This line used to quote one
            # that the comment twelve lines above had already superseded, and
            # a number in a tool's output outlives its retraction in the docs.
            print("  Two engines, so the additive ceiling is 2x. Reference")
            print("  figures for this hardware live in docs/MULTI_ENGINE.md.")

    for name, _b, _m in engines:
        r = per_engine[name]
        note, suspect = shed_note(name, r["shed"], r["served"], r["failed"],
                                  r["first_failure"])
        if note:
            print("\n  %s" % note)
        if suspect:
            warnings.append(note)

    if warnings:
        print("\n%s" % ("!" * 64))
        for w in warnings:
            print("  WARNING: %s" % w)
        print("!" * 64)

    json_note = None
    if a.json:
        record = {"loaded": loaded, "free_gb": free, "depth": a.depth,
                  "tokens": a.tokens, "repeat": a.repeat,
                  # The settings a reader needs to know whether two
                  # artifacts are comparable at all. None of these used
                  # to be written, and the docstring claimed n_ctx was.
                  "ramp": a.ramp, "timeout": a.timeout,
                  "cool_floor": a.cool_floor, "closing_tol": a.closing_tol,
                  "load_depth": load_depth, "load_tokens": load_tokens,
                  "engines": engine_meta,
                  "solo_median": solo, "contended_median": contended,
                  "paired_ratio_median": ratios, "per_engine": per_engine,
                  # The readings the gate passed each SOLO sample on, one
                  # per gated leg; the round-start clock is in
                  # power_samples.
                  "gate_clock_pct": clocks, "closing_check": closing,
                  "bandwidth": bw,
                  # Magnitudes, not a verdict. A consumer can apply its own
                  # threshold later; it cannot recover a reading the
                  # harness discarded at write time.
                  "power_samples": POWER_SAMPLES,
                  # One reading per accepted measurement, taken by
                  # bench_endpoint right after each: the post-sample
                  # trajectory that power_samples approximates per round.
                  # Populated on every run and never read until now.
                  "box_samples": list(be.BOX_SAMPLES),
                  "warnings": warnings}
        # _write_record decides WHERE, and says so when it is not where the
        # operator asked. It never discards the record: this is the only copy
        # of power_samples, box_samples, gate_clock_pct and the per-engine
        # counts, so the scrollback above is not a substitute for it.
        written, json_note = _write_record(a.json, record, a.force)
        if json_note:
            print("\n  %s. The results above are complete." % json_note)
        if written:
            print("\nwrote %s" % written)
    if aborted:
        # 2 outranks 3: a sweep the gate STOPPED is the more serious fact, and
        # the sentence above says where the file went in either case.
        return 2
    return 3 if json_note else 0


if __name__ == "__main__":
    sys.exit(main())
