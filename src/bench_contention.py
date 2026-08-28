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
else is imported from bench_endpoint rather than reimplemented: the prompts and
the decode-by-subtraction method are already validated there, and a second copy
would drift.

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

  * The NPU is SINGLE-FLIGHT. Its server serialises behind a lock and returns
    429/529 "server busy" once its small queue fills. That is backpressure, not
    an error: the load generator counts it and carries on, because a generator
    that died on the first 429 would silently stop contending halfway through.

  * Reported per engine: solo rate, contended rate, and the ratio. Reported for
    the pair: aggregate throughput while both are hot, against the sum of the
    solo rates. An aggregate BELOW the faster engine's solo rate means routing
    to a second hot engine is a net loss for throughput -- it would still buy
    concurrency and failover, but not speed.

  * A run is only comparable to another run at the same /props n_ctx, which is
    recorded per endpoint. On Genie the COMPILED window sets throughput.

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

  python src/bench_contention.py
  python src/bench_contention.py --npu http://127.0.0.1:8123 --gpu http://127.0.0.1:8080
  python src/bench_contention.py --json out.json

Pure stdlib. Needs both servers already up; it starts nothing.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

import bench_endpoint as be


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


class Load:
    """Saturates an endpoint in a loop until stopped.

    Exists so the engine under test is measured while the other one is
    genuinely busy. Counts 429/529 separately: on the single-flight NPU those
    are the expected steady state once its queue is full, and treating them as
    failures would report a healthy contended run as broken.
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
        self._stop = threading.Event()
        self._thread = None
        self.completed = 0
        self.busy = 0
        self.tokens_out = 0

    def _run(self):
        prompt = be.prompt_of(self.depth)
        while not self._stop.is_set():
            body, _wall = be._post(self.base, "/v1/chat/completions", {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
            }, self.timeout)
            if body is None:
                # _post returns None for any non-2xx, including the 429/529 the
                # NPU raises as backpressure. They are indistinguishable here,
                # so they are counted together and reported as "shed".
                self.busy += 1
                continue
            self.completed += 1
            self.tokens_out += (body.get("usage") or {}).get("completion_tokens", 0)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 30)


def cpu_performance_pct():
    """Current clock as a percentage of base, or None.

    A throttle detector, and the reason it is here: this box was observed at
    86.8% while an unrelated export ran. Sustained load on a thin ARM64 laptop
    decays clocks, and a decay that happens to land during the contended half
    of a run is indistinguishable from contention unless it is watched. There
    is no MSAcpi thermal zone on ARM64, so this counter is the available proxy.
    """
    if sys.platform != "win32":
        return None
    import subprocess
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-Counter '\\Processor Information(_Total)\\% Processor "
             "Performance').CounterSamples.CookedValue"],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def power_source():
    """'battery' | 'ac' | 'no-battery' | 'unknown'.

    Distinguishing the last two matters: a transient PowerShell failure used to
    return the same None as a desktop with no battery, silently downgrading a
    definitive check to a heuristic on a laptop that could have answered.
    """
    v = _power_online_raw()
    if v is None:
        return "unknown"
    if v == "":
        return "no-battery"
    return "battery" if v.lower() != "true" else "ac"


def _power_online_raw():
    """Raw PowerOnline string, "" if the class reports nothing, None on error.

    Kept separate from power_source() so that "no battery" and "query failed"
    stay distinguishable. They were not: the predecessor returned None for
    both, so a transient PowerShell failure silently downgraded a definitive
    check to a heuristic on a laptop that could have answered.
    """
    if sys.platform != "win32":
        return None
    import subprocess
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus "
             "-ErrorAction SilentlyContinue | Select-Object -First 1)"
             ".PowerOnline"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def cpu_busy_pct():
    """Box-wide CPU utilisation, or None. The second half of the fingerprint."""
    if sys.platform != "win32":
        return None
    import subprocess
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-Counter '\\Processor(_Total)\\% Processor Time')"
             ".CounterSamples.CookedValue"],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def battery_state():
    """(on_ac, charge_pct, charge_watts) -- any element None if unreadable.

    Sampled as MAGNITUDES rather than folded into a boolean, which is the
    lesson from four legs measured across two sessions on this box. A counter
    keyed to "is charging suspended" with an absolute near-zero threshold
    reported 0 suspended samples for legs that shed 92% and 97% of their charge
    draw, because the minima (3.0 W, 1.1 W) cleared a <=1 W test. A later
    threshold at 25% of opening draw scored 2 of the same 4 legs. Every cut
    point mis-sorts the legs nearest it, so record the quantity and let a reader
    pick their own line afterwards.
    """
    if sys.platform != "win32":
        return None, None, None
    import subprocess
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "$b = Get-CimInstance -Namespace root\\wmi -ClassName "
             "BatteryStatus -ErrorAction SilentlyContinue | Select-Object "
             "-First 1; $c = (Get-CimInstance Win32_Battery -ErrorAction "
             "SilentlyContinue | Select-Object -First 1)"
             ".EstimatedChargeRemaining; "
             "'{0},{1},{2}' -f $b.PowerOnline, $c, "
             "[math]::Round($b.ChargeRate/1000,1)"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None, None, None
        ac, pct, watts = out.stdout.strip().split(",")
        return (ac.strip().lower() == "true",
                float(pct) if pct.strip() else None,
                float(watts) if watts.strip() else None)
    except Exception:
        return None, None, None


# Below this the pack is deep enough into discharge that the system protects
# the charge and starves compute, whether or not AC is connected. Measured on
# this box across two sessions: legs run at 13-20% gave pp512 58.14 / 57.88
# against a settled baseline of 130.20 -- HALF -- while legs at 33% and 41.6%
# came back at 124.53 and 112.58, near baseline and stable. The hazard is the
# depth of discharge, not the act of charging, which is the opposite of what
# both sessions assumed before pooling their legs.
LOW_CHARGE_PCT = float(os.environ.get("GENIE_LOW_CHARGE_PCT", "25"))


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
    """
    if pct is None or pct >= floor:
        return None, False
    src = power_source()
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
    _ac, charge, watts = battery_state()
    if src == "ac" and charge is not None and charge < LOW_CHARGE_PCT:
        return ("clock is %.0f%% of base, and the pack is at %.0f%%%s. AC is "
                "connected, but measured on this box a pack below ~%.0f%% "
                "halves prefill (pp512 58 against a settled 130) while decode "
                "barely moves -- so this is not a settled-box measurement even "
                "though it is plugged in. Advisory, not fatal: bandwidth-bound "
                "work is largely immune. Let it charge for a clean baseline."
                % (pct, charge,
                   " drawing %.0f W" % watts if watts is not None else "",
                   LOW_CHARGE_PCT)), False
    busy = cpu_busy_pct()
    if src == "no-battery" and busy is not None and busy < 15.0:
        return ("clock is %.0f%% of base while the CPU is only %.0f%% busy. On "
                "a box with no battery that is most likely ordinary idle "
                "downclocking rather than a limit -- but if the clock stays "
                "low once work starts, check the power budget." % (pct, busy)), False
    return None, False


# Notes raised inside wait_for_cool, drained into the run's warnings so they
# reach the JSON as well as the terminal.
GATE_NOTES = []

# One (on_ac, charge_pct, charge_watts) reading per round, so the artifact
# carries the power TRAJECTORY rather than a verdict about it. A run taken while
# the pack climbed from 20% to 60% is a different run from one taken at a steady
# 95%, and nothing else in the record distinguishes them.
POWER_SAMPLES = []


def wait_for_cool(floor, limit=300):
    """Block until the clock recovers to `floor`% of base, or `limit` seconds.

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
            return None
        if first is None:
            first = pct
            # Checked ONCE, on the first below-floor reading, rather than every
            # loop: the power source does not change while we spin, and this
            # costs two subprocess calls. Bailing immediately matters because
            # the alternative is blocking the full `limit` for a recovery that
            # cannot happen -- observed 2026-08-24 as ten minutes of silence
            # from a run whose box had been unplugged, which read as a hang.
            why, abort = power_limited_note(pct, floor)
            if why is not None:
                # Recorded, not just printed. A warning that exists only in the
                # terminal is absent from the artifact a consumer reads, which
                # is how a suspect number becomes a clean-looking one
                # downstream -- the same record-vs-reality drift this harness
                # keeps finding elsewhere.
                GATE_NOTES.append(why)
                print("    (%s: %s)"
                      % ("ABORTING THE GATE" if abort else "note", why), flush=True)
                if abort:
                    return pct
        if pct >= floor:
            if time.time() - start > 5:
                print("    (cooled %.0f%% -> %.0f%% after %ds)"
                      % (first, pct, int(time.time() - start)), flush=True)
            return pct
        time.sleep(10)
    print("    (WARNING: clock still %s%% after %ds, proceeding anyway -- "
          "this sample is thermally suspect)"
          % ("%.0f" % pct if pct is not None else "?", limit), flush=True)
    return pct


def measure(base, model, depth, tokens, timeout, label, cool_floor=None):
    """Decode rate at one depth, printed with its label."""
    if cool_floor:
        wait_for_cool(cool_floor)
    print("  [%s]" % label, end=" ", flush=True)
    return be.measure_decode(base, model, depth, tokens, timeout)


def paired_sweep(engines, a, make_load):
    """Interleave solo and contended samples, and report the PAIRED ratios.

    WHAT "SOLO" MEANS HERE, because it is narrower than the word suggests: the
    other engine's SERVER is still resident, it is merely not generating -- the
    load generator starts for the contended leg only. So this measures
    engine-with-an-idle-peer, not engine-alone. That distinction is not
    academic: an idle genie_server with poll:true was measured costing the GPU
    25-32% of its throughput while answering nothing, so "solo" and "alone" can
    differ by a third. Stopping the peer entirely is a different baseline and
    has to be measured deliberately, not inferred from this one.

    The ordering is the whole point. Measuring every solo first and every
    contended second puts all of any thermal decay into the contended half,
    where it is indistinguishable from contention and biases the result the
    same direction every time. Sampling solo and contended ADJACENTLY and
    taking the median of the per-pair ratios cancels drift that is slow
    relative to one pair, which is the shape thermal drift has.

    That is also why the median is taken over RATIOS rather than the ratio
    being taken over medians: the former keeps each solo matched to the
    contended sample nearest it in time, the latter throws that pairing away.
    """
    per_engine = {name: {"solo": [], "contended": [], "ratios": [],
                         "shed": 0, "served": 0} for name, _b, _m in engines}
    clocks = []

    for i in range(a.repeat):
        print("\n--- round %d/%d ---" % (i + 1, a.repeat), flush=True)
        pct = cpu_performance_pct()
        if pct is not None:
            clocks.append(pct)
            print("  cpu clock %.1f%% of base" % pct, flush=True)
        ac, charge, watts = battery_state()
        if charge is not None:
            POWER_SAMPLES.append({"round": i + 1, "on_ac": ac,
                                  "charge_pct": charge, "charge_w": watts})
            print("  power     %s, pack %.0f%%%s"
                  % ("AC" if ac else "BATTERY", charge,
                     ", drawing %.1f W" % watts if watts is not None else ""),
                  flush=True)

        for name, base, model in engines:
            # cool_floor is threaded through explicitly. It used to default to
            # None here, which silently disabled the gate this harness's whole
            # method depends on -- the samples were taken across exactly the
            # thermal decay wait_for_cool exists to prevent.
            solo = measure(base, model, a.depth, a.tokens, a.timeout,
                           "%s solo" % name, cool_floor=a.cool_floor)

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
            rec["shed"] += gen.busy
            rec["served"] += gen.completed
            if solo is not None:
                rec["solo"].append(solo)
            if cont is not None:
                rec["contended"].append(cont)
            if solo and cont:
                rec["ratios"].append(cont / solo)
                print("    pair: %.2f -> %.2f t/s  (keeps %.1f%%)"
                      % (solo, cont, 100 * cont / solo), flush=True)

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
    final = measure(base, model, a.depth, a.tokens, a.timeout,
                    "%s solo (closing)" % name, cool_floor=a.cool_floor)
    if not final:
        return per_engine, clocks, {"state": "failed", "engine": name}
    return per_engine, clocks, {"state": "ok", "engine": name,
                                "first": opened_with[0], "final": final,
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
        }.get(state, "it did not run")
        # NOT the words the passing branch uses. "Not checked" reading like
        # "checked and fine" is the defect this harness keeps finding in its
        # own instruments.
        return ("note: no closing re-check -- %s. Nothing here says whether "
                "the box held across this sweep." % why), False
    drift = 100.0 * (closing["retained"] - 1.0)
    shape = ("%s solo opened at %.2f t/s and closed at %.2f (%+.1f%%)"
             % (closing["engine"], closing["first"], closing["final"], drift))
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


def shed_note(name, shed, served):
    """(message, suspect) about a contended leg's load generator.

    `shed` counts requests the load generator did not get a completion for.
    That number cannot tell backpressure from a dead endpoint: `_post` returns
    None for a 429 AND for connection-refused, so a peer server that never
    started looks exactly like one too busy to answer. The difference decides
    whether the experiment happened at all -- if nothing connected, the
    "contended" leg measured an idle box and the ratio comes out near 1.0,
    which reads as "no contention effect" rather than as "no contention".

    `served == 0` is the discriminator, and it is the one the counting site
    cannot apply because it only sees one request at a time. A genuinely
    backpressured engine still completes SOME requests between rejections; one
    that completed none across an entire run was most likely never reachable.
    """
    if not shed:
        return None, False
    if served == 0:
        return ("SUSPECT: while %s was measured, the load generator got %d "
                "failure(s) and ZERO completions. A 429 and a refused "
                "connection are indistinguishable here, so this is either a "
                "fully-queued engine or one that was never up -- and if it "
                "was never up, the contended leg measured an IDLE box and any "
                "ratio near 1.0 means 'no load', not 'no contention'. Check "
                "that the peer endpoint was serving before believing this run."
                % (name, shed)), True
    return ("note: while %s was measured, the other engine shed %d request(s) "
            "and served %d. On the single-flight NPU that is expected "
            "backpressure; a shed count near 100%% means it was queued rather "
            "than contending, which UNDERSTATES contention."
            % (name, shed, served)), False


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", default="http://127.0.0.1:8123",
                    help="Genie NPU server base URL (default %(default)s)")
    ap.add_argument("--gpu", default="http://127.0.0.1:8080",
                    help="llama-server GPU base URL (default %(default)s)")
    ap.add_argument("--npu-model", default="qwen3-4b-npu")
    ap.add_argument("--gpu-model", default="default")
    ap.add_argument("--depth", type=int, default=500,
                    help="context depth for every measurement")
    ap.add_argument("--tokens", type=int, default=120,
                    help="decode steps per measurement")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--ramp", type=float, default=8.0,
                    help="seconds to let the background load get in flight "
                         "before the contended measurement starts")
    ap.add_argument("--timeout", type=float, default=1800)
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
    ap.add_argument("--cool-floor", type=float, default=92.0,
                    help="wait for the clock to recover to this %% of base "
                         "before each SOLO sample (0 disables). Sustained load "
                         "drops this box to 48.9%%, and an ungated sweep turns "
                         "that decay into a fake depth curve")
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
    a = ap.parse_args()

    # Cleared per run, not merely appended to. GATE_NOTES is module state and
    # main() copies it into the run's `warnings`, so a second run in the same
    # process would inherit the first one's power notes and publish them in its
    # JSON -- a warning attached to a run that never earned it, which is the
    # record-vs-reality drift the rest of this harness exists to prevent.
    GATE_NOTES.clear()
    POWER_SAMPLES.clear()

    free = free_physical_gb()
    shown = "unknown" if free is None else "%.2f GB" % free
    print("free physical memory: %s (need >= %.1f GB)" % (shown, a.min_free_gb))
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
    for name, base, model in engines:
        w = be.n_ctx(base)
        print("%s %s  model=%s  n_ctx=%s" % (name, base, model, w if w else "unknown"))
        if be.chat(base, model, "ping", 1, 60) is None:
            print("\n%s at %s is not answering -- start it first."
                  % (name, base), file=sys.stderr)
            return 2

    print("\nwarmup", flush=True)
    for _name, base, model in engines:
        be.chat(base, model, be.prompt_of(a.depth), 4, a.timeout)

    def make_load(other):
        return Load(other[1], other[2], a.depth, a.tokens, a.timeout)

    print("\nPAIRED SWEEP (solo and contended interleaved, per round)",
          flush=True)
    per_engine, clocks, closing = paired_sweep(engines, a, make_load)

    solo = {n: statistics.median(r["solo"]) for n, r in per_engine.items() if r["solo"]}
    contended = {n: statistics.median(r["contended"])
                 for n, r in per_engine.items() if r["contended"]}
    ratios = {n: statistics.median(r["ratios"])
              for n, r in per_engine.items() if r["ratios"]}

    print("\n%s" % ("=" * 64))
    print("CONTENTION%s" % ("  [LOADED BOX -- NOT A BASELINE]" if loaded else ""))
    print("=" * 64)
    for name, _b, _m in engines:
        r = per_engine[name]
        if name in ratios:
            print("  %-4s solo %7.2f   contended %7.2f   keeps %5.1f%%  "
                  "(paired median of %d)"
                  % (name, solo[name], contended[name], 100 * ratios[name],
                     len(r["ratios"])))
            print("       solo samples      %s"
                  % ", ".join("%.2f" % v for v in r["solo"]))
            print("       contended samples %s"
                  % ", ".join("%.2f" % v for v in r["contended"]))
        else:
            print("  %-4s incomplete (every measurement was skipped)" % name)

    warnings = list(GATE_NOTES)
    note, suspect = closing_note(closing, a.closing_tol)
    print()
    print("  %s" % note)
    if suspect:
        warnings.append(note)
    for name, _b, _m in engines:
        w = drift_note(per_engine[name]["solo"], "%s solo decode" % name)
        if w:
            warnings.append(w)
    if clocks and min(clocks) < 95:
        warnings.append(
            "cpu clock dipped to %.1f%% of base during the run (samples: %s) -- "
            "the package was power- or thermally-limited, so part of any "
            "measured slowdown is not contention."
            % (min(clocks), ", ".join("%.0f" % c for c in clocks)))
    if POWER_SAMPLES:
        charges = [s["charge_pct"] for s in POWER_SAMPLES]
        lo, hi = min(charges), max(charges)
        if lo < LOW_CHARGE_PCT:
            warnings.append(
                "the pack was at %.0f%% during this run (range %.0f-%.0f%%). "
                "Below ~%.0f%% this box halves prefill while decode holds, so "
                "a prefill-sensitive comparison taken here is not a settled "
                "baseline even on AC."
                % (lo, lo, hi, LOW_CHARGE_PCT))
        elif hi - lo >= 20:
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
                if (ratios and tot_cont < 0.5 * a.peak_bw_gbs
                        and min(ratios.values()) < 0.8):
                    warnings.append(
                        "engines lost >20%% throughput while together using only "
                        "%.0f%% of peak bandwidth -- the bottleneck is NOT the "
                        "memory bus. Suspect a shared power budget, DVFS, or "
                        "memory-controller latency rather than raw bandwidth."
                        % (100 * tot_cont / a.peak_bw_gbs))

    if len(ratios) == 2:
        # Aggregate uses each engine's CONTENDED rate, since that is what the
        # pair actually delivers when both are hot. Comparing it to the faster
        # engine's SOLO rate is the decision: below it, a second hot engine
        # costs throughput and only buys concurrency and failover.
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
        if agg < best_solo:
            print("\n  VERDICT: running both is SLOWER than the best engine alone.")
            print("  Route to a second engine for concurrency and failover only,")
            print("  never for throughput.")
            # Before believing this, check the flag. A poll:true bundle
            # busy-polls the HTP on ~2.8 cores and starves the OpenCL
            # backend's per-token dispatch, which turned a measured 1.45x
            # GAIN into a 0.78x LOSS on 2026-08-24. A net-loss verdict on
            # this hardware is far more likely to be that flag than a real
            # hardware limit.
            print("  FIRST check QnnHtp/poll in the bundle's genie_config.json:")
            print("  poll:true measured 0.78x here, poll:false measured 1.45x.")
        else:
            print("\n  VERDICT: two hot engines beat the best single engine by "
                  "%.2fx." % (agg / best_solo))
            print("  Reference: 1.45x measured on X1E80100 (NPU+GPU, 4B, d469,")
            print("  poll:false). Two engines, so the additive ceiling is 2x.")

    for name, _b, _m in engines:
        r = per_engine[name]
        note, suspect = shed_note(name, r["shed"], r["served"])
        if note:
            print("\n  %s" % note)
        if suspect:
            warnings.append(note)

    if warnings:
        print("\n%s" % ("!" * 64))
        for w in warnings:
            print("  WARNING: %s" % w)
        print("!" * 64)

    if a.json and os.path.exists(a.json) and not a.force:
        # A run here costs 20+ minutes of a shared box, and some of these
        # numbers have turned out to be unreproducible (the contention
        # absolutes could not be obtained twice). Clobbering a previous
        # result to save a flag is the wrong trade, so the file is kept and
        # the fresh numbers are printed above either way.
        print("\n  NOT writing %s: it already exists. The results above are "
              "complete; re-run with --force to overwrite, or pass a different "
              "--json path." % a.json)
    elif a.json:
        with open(a.json, "w") as f:
            json.dump({"loaded": loaded, "free_gb": free, "depth": a.depth,
                       "tokens": a.tokens, "repeat": a.repeat,
                       "solo_median": solo, "contended_median": contended,
                       "paired_ratio_median": ratios, "per_engine": per_engine,
                       "cpu_clock_pct": clocks, "closing_check": closing,
                       "bandwidth": bw,
                       # Magnitudes, not a verdict. A consumer can apply its own
                       # threshold later; it cannot recover a reading the
                       # harness discarded at write time.
                       "power_samples": POWER_SAMPLES,
                       "warnings": warnings}, f, indent=2)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
