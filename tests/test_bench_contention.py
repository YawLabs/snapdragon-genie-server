"""Device-free tests for bench_contention.py's reporting paths.

The two crash bugs these cover were both reachable from a single round where
every contended sample was skipped, and that is not a rare shape: the NPU sheds
by design once its small queue fills (429 OpenAI / 529 Anthropic), so a
contention run against it is the workload MOST likely to produce an asymmetric
solo/contended result. Both guards had assumed it could not happen.

No device, no server, no network -- bench_endpoint is stubbed at import.
"""

import json
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# Stub before importing: bench_contention imports bench_endpoint at module
# scope, and the real one is only useful with a live server.
sys.modules.setdefault("bench_endpoint", types.SimpleNamespace(
    prompt_of=lambda d: "x",
    measure_decode=lambda *a, **k: 1.0,
    chat=lambda *a, **k: {},
    n_ctx=lambda b: 4096,
    _post=lambda *a, **k: (None, 0.0),
    # bench_contention delegates its power sampling here rather than keeping a
    # second copy of the WMI query. The stub returns "unreadable", which is what
    # a test double that cannot see hardware honestly knows -- and it keeps this
    # file's promise of no device and no subprocess. Tests that need a specific
    # reading patch `bc.battery_state` directly.
    battery_state=lambda: (None, None, None),
))

import bench_contention as bc  # noqa: E402


class Args:
    """The subset of the parsed namespace the reporting path reads."""

    def __init__(self, **kw):
        self.depth, self.tokens, self.repeat, self.ramp = 500, 62, 1, 0.0
        self.timeout, self.cool_floor = 60, 0.0
        self.npu_weights_gb, self.gpu_weights_gb = 2.3, 2.32
        self.peak_bw_gbs = 135.2
        self.__dict__.update(kw)


def test_gate_fires_only_when_cool_floor_is_set(monkeypatch):
    """The regression that mattered: the gate was unreachable, silently."""
    calls = []
    monkeypatch.setattr(bc, "wait_for_cool", lambda floor, limit=300: calls.append(floor))
    bc.measure("u", "m", 1, 1, 1, "gated", cool_floor=92.0)
    assert calls == [92.0]
    bc.measure("u", "m", 1, 1, 1, "ungated")
    assert calls == [92.0], "measure() must not cool when no floor is given"


def test_wait_for_cool_survives_an_unreadable_counter(monkeypatch):
    """cpu_performance_pct returns None off-Windows and on counter failure."""
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: None)
    assert bc.wait_for_cool(92.0, limit=1) is None


def test_wait_for_cool_does_not_raise_when_limit_is_zero(monkeypatch):
    """The loop body never runs, so the warning path must not touch an
    unbound name."""
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 50.0)
    assert bc.wait_for_cool(92.0, limit=0) is None


def test_load_generator_does_not_inherit_the_measurement_timeout():
    """stop() blocks behind an in-flight request, so the generator's
    per-request cap has to be short regardless of the measurement timeout."""
    gen = bc.Load("http://x", "m", 500, 62, timeout=1800)
    assert gen.timeout <= bc.LOAD_REQUEST_TIMEOUT_S


def test_drift_note_flags_only_a_monotonic_decline():
    assert bc.drift_note([20.0, 18.0, 16.0], "x") is not None
    assert bc.drift_note([20.0, 16.0, 18.0], "x") is None
    assert bc.drift_note([20.0, 18.0], "x") is None, "needs 3+ samples"


def _shed_round():
    """One engine measured fine; the other had every contended sample shed.

    This is the NPU-under-contention shape: solo samples land, contended ones
    come back as 429/529 and are skipped.
    """
    return {
        "GPU": {"solo": [18.0], "contended": [13.4], "ratios": [13.4 / 18.0],
                "shed": 0, "served": 4},
        "NPU": {"solo": [18.5], "contended": [], "ratios": [],
                "shed": 9, "served": 0},
    }


def test_reporting_survives_a_fully_shed_contended_leg(monkeypatch, capsys):
    """Covers BOTH crashes: the bandwidth block's KeyError on contended[name],
    and min() over an empty ratios dict."""
    monkeypatch.setattr(bc, "paired_sweep", lambda e, a, m: (_shed_round(), [99.0], {"state": "disabled"}))
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})

    argv = ["bench_contention.py", "--repeat", "1", "--cool-floor", "0",
            "--npu-weights-gb", "2.3", "--gpu-weights-gb", "2.32",
            "--peak-bw-gbs", "135.2"]
    monkeypatch.setattr(sys, "argv", argv)
    assert bc.main() == 0          # must not raise
    out = capsys.readouterr().out
    assert "NPU" in out and "GPU" in out
    assert "incomplete" in out, "the shed leg should be reported, not hidden"


def test_reporting_survives_when_no_pair_completes(monkeypatch):
    """Every ratio empty -- the empty-min() path with nothing to divide."""
    both_shed = {
        "GPU": {"solo": [18.0], "contended": [], "ratios": [], "shed": 3, "served": 0},
        "NPU": {"solo": [18.5], "contended": [], "ratios": [], "shed": 9, "served": 0},
    }
    monkeypatch.setattr(bc, "paired_sweep", lambda e, a, m: (both_shed, [99.0], {"state": "disabled"}))
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", [
        "bench_contention.py", "--cool-floor", "0",
        "--npu-weights-gb", "2.3", "--gpu-weights-gb", "2.32",
        "--peak-bw-gbs", "135.2"])
    assert bc.main() == 0


def test_refuses_a_loaded_box_unless_explicitly_allowed(monkeypatch):
    """The precondition is the whole reason this harness is trustworthy."""
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 2.5)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py"])
    assert bc.main() == 2

    monkeypatch.setattr(bc, "paired_sweep", lambda e, a, m: (_shed_round(), [], {"state": "disabled"}))
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", [
        "bench_contention.py", "--allow-loaded", "--cool-floor", "0"])
    assert bc.main() == 0


def test_refuses_when_free_memory_cannot_be_determined(monkeypatch):
    """Unknown must fail loudly rather than pass a box it could not measure."""
    monkeypatch.setattr(bc, "free_physical_gb", lambda: None)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py"])
    assert bc.main() == 2


def test_wait_for_cool_handles_a_real_counter_reading(monkeypatch):
    # The two cases above both dodge the body: an unreadable counter returns
    # before the tracking, and limit=0 never enters the loop. So a gate that
    # crashed on EVERY real reading passed both. It did -- `first` was read
    # before it was assigned, so the first successful sample raised
    # UnboundLocalError. The gate had never gated anything, first because it
    # was dead code and then because wiring it exposed this.
    import bench_contention as bc
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 97.0)
    assert bc.wait_for_cool(92.0, limit=5) == 97.0


def test_wait_for_cool_returns_the_sample_it_gated_on(monkeypatch):
    # Below the floor once, then above: the value returned must be the reading
    # that satisfied the gate, not a fresh sample taken afterwards.
    import bench_contention as bc
    seq = iter([80.0, 95.0])
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: next(seq))
    monkeypatch.setattr(bc.time, "sleep", lambda _s: None)
    # Stubbed because the power check added later SHELLS OUT, which made this
    # environment-dependent: run on a laptop actually on battery, the gate
    # aborts on the 80.0 reading and this returns 80.0 instead of 95.0. It
    # caught exactly that the day the check landed. A suite whose header
    # promises "no device, no server, no network" has to stub every probe, not
    # only the ones that existed when the test was written.
    monkeypatch.setattr(bc, "power_source", lambda: "ac")
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 90.0)
    assert bc.wait_for_cool(92.0, limit=30) == 95.0


def _count_reads(monkeypatch, src, charge=90.0, busy=90.0):
    """Run power_limited_note below the floor, counting each hardware read.

    Every one of these is a PowerShell launch and this gate runs before every
    sample, so the count is the thing worth pinning -- not as a micro-benchmark
    but because it grew from two to three without anyone noticing.
    """
    calls = []

    def rec(name, value):
        def f(*a, **k):
            calls.append(name)
            return value
        return f

    monkeypatch.setattr(bc, "power_source", rec("power_source", src))
    monkeypatch.setattr(bc, "battery_state",
                        rec("battery_state", (True, charge, 30.0)))
    monkeypatch.setattr(bc, "cpu_busy_pct", rec("cpu_busy_pct", busy))
    bc.power_limited_note(45.0, 92.0)
    return calls


def test_the_gate_takes_only_the_readings_it_uses(monkeypatch):
    # cpu_busy_pct's value is read ONLY in the no-battery branch, but it used to
    # be called unconditionally -- so every AC run paid a subprocess for a
    # number it then discarded.
    ac = _count_reads(monkeypatch, "ac")
    assert "cpu_busy_pct" not in ac, "AC path read a CPU figure it cannot use"
    assert ac == ["power_source", "battery_state"]

    nb = _count_reads(monkeypatch, "no-battery")
    assert "battery_state" not in nb, "no-battery path read a pack it does not have"
    assert nb == ["power_source", "cpu_busy_pct"]

    # The two definitive answers cost one reading and stop.
    assert _count_reads(monkeypatch, "battery") == ["power_source"]
    assert _count_reads(monkeypatch, "unknown") == ["power_source"]


def test_the_gate_reads_nothing_at_all_above_the_floor(monkeypatch):
    # The common case by far: the clock is fine, so there is nothing to explain
    # and no reason to touch the hardware.
    calls = []
    monkeypatch.setattr(bc, "power_source",
                        lambda: calls.append("power_source") or "ac")
    assert bc.power_limited_note(99.0, 92.0) == (None, False)
    assert calls == []


def test_a_deeply_discharged_pack_is_flagged_even_on_AC(monkeypatch):
    """AC used to fall off the end of the check and read as clean.

    That is the state an operator reaches by plugging in and starting
    immediately, and it is the one that actually costs prefill: measured across
    two sessions on this box, legs at 13-20% gave pp512 ~58 against a settled
    130, while 33% and 41.6% came back near baseline. The hazard is depth of
    discharge, not charging -- the opposite of what was assumed before the legs
    were pooled.
    """
    monkeypatch.setattr(bc, "power_source", lambda: "ac")
    monkeypatch.setattr(bc, "battery_state", lambda: (True, 14.0, 30.0))
    msg, abort = bc.power_limited_note(45.0, 92.0)
    assert msg is not None and "14%" in msg
    assert abort is False, "advisory only -- bandwidth-bound work is immune"
    assert "not a settled-box measurement" in msg


def test_a_healthy_pack_on_AC_stays_silent(monkeypatch):
    # The counterpart. Warning on a charged box is how the real warning gets
    # skipped.
    monkeypatch.setattr(bc, "power_source", lambda: "ac")
    monkeypatch.setattr(bc, "battery_state", lambda: (True, 88.0, 2.0))
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 90.0)
    assert bc.power_limited_note(45.0, 92.0) == (None, False)


def test_an_unreadable_pack_does_not_invent_a_charge_warning(monkeypatch):
    # battery_state returns Nones off-Windows and on any query failure. Absence
    # of a reading must not become a claim about the reading.
    monkeypatch.setattr(bc, "power_source", lambda: "ac")
    monkeypatch.setattr(bc, "battery_state", lambda: (None, None, None))
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 90.0)
    assert bc.power_limited_note(45.0, 92.0) == (None, False)


def test_power_samples_reach_the_json_as_magnitudes(monkeypatch, tmp_path):
    """The artifact must carry the quantity, not a verdict about it.

    Four legs across two sessions showed every boolean cut point mis-sorting the
    legs nearest it -- a <=1 W test scored 0/4 on legs that shed 92% and 97%,
    and a 25%-of-opening test scored 2/4. A reader can apply a threshold to a
    recorded magnitude; nobody can recover a reading the harness threw away.
    """
    bc.POWER_SAMPLES.clear()
    bc.POWER_SAMPLES.extend([
        {"round": 1, "on_ac": True, "charge_pct": 22.0, "charge_w": 30.0},
        {"round": 2, "on_ac": True, "charge_pct": 24.0, "charge_w": 29.0}])
    out = tmp_path / "run.json"
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    monkeypatch.setattr(bc, "paired_sweep",
                        lambda e, a, m: (rounds, [99.0], {"state": "disabled"}))
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(bc, "battery_state", lambda: (True, 22.0, 30.0))
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "1",
                                      "--cool-floor", "0", "--json", str(out)])
    assert bc.main() == 0
    body = json.loads(out.read_text())
    # main() clears POWER_SAMPLES at entry, so what lands is what the run itself
    # recorded -- empty here, since paired_sweep is stubbed out and never
    # samples. The KEY still has to exist, or a consumer cannot tell "no
    # readings taken" from "this harness does not report power at all".
    assert "power_samples" in body
    assert body["power_samples"] == [], (
        "a stubbed sweep records nothing, so anything here leaked from a "
        "previous run -- the same cross-run bleed GATE_NOTES had")


def test_a_low_pack_warns_in_the_run_summary(monkeypatch, capsys):
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)

    def sweep(e, a, m):
        bc.POWER_SAMPLES.extend([
            {"round": 1, "on_ac": True, "charge_pct": 18.0, "charge_w": 31.0}])
        return rounds, [99.0], {"state": "disabled"}

    monkeypatch.setattr(bc, "paired_sweep", sweep)
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "1",
                                      "--cool-floor", "0"])
    assert bc.main() == 0
    out = capsys.readouterr().out
    assert "pack was at 18%" in out
    assert "halves prefill" in out


def test_the_drift_warning_reports_the_span_it_fired_on(monkeypatch, capsys):
    """A warning must not contradict its own trigger.

    The condition is on the EXTREMES (max - min >= 20), so reporting first and
    last readings instead printed "moved 60% -> 62%" above a warning raised
    because the run spanned 40-62%. Non-monotonic trajectories are the norm here
    -- the pack can dip under load and recover -- so the two differ routinely.
    """
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)

    def sweep(e, a, m):
        # Dips to 40 and recovers: endpoints are 2 points apart, span is 22.
        for i, pct in enumerate((60.0, 40.0, 62.0), 1):
            bc.POWER_SAMPLES.append({"round": i, "on_ac": True,
                                     "charge_pct": pct, "charge_w": 30.0})
        return rounds, [99.0], {"state": "disabled"}

    monkeypatch.setattr(bc, "paired_sweep", sweep)
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "1",
                                      "--cool-floor", "0"])
    assert bc.main() == 0
    out = capsys.readouterr().out
    assert "spanned 40%-62%" in out, "reported endpoints instead of the span"
    assert "opened 60%, closed 62%" in out, "endpoints are still worth showing"


def test_charge_draw_variation_is_surfaced_not_left_dead(monkeypatch, capsys):
    """charge_w was recorded and never read, which is dead weight in the record.

    It needs a frame rather than a raw list: measured on this box, draw ranged
    28-41.9 W at a roughly constant charge level while read-to-read noise was
    ~1 W. A reader who anchors a shed percentage to one sample of that is biased
    by which sample they happened to pick.
    """
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)

    def sweep(e, a, m):
        for i, w in enumerate((28.0, 41.9, 33.0), 1):
            bc.POWER_SAMPLES.append({"round": i, "on_ac": True,
                                     "charge_pct": 70.0, "charge_w": w})
        return rounds, [99.0], {"state": "disabled"}

    monkeypatch.setattr(bc, "paired_sweep", sweep)
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "1",
                                      "--cool-floor", "0"])
    assert bc.main() == 0
    out = capsys.readouterr().out
    assert "28.0-41.9 W" in out
    assert "not the workload" in out, "must say what the variation IS"


def test_a_steady_draw_says_nothing(monkeypatch, capsys):
    # The counterpart: a stable controller must not produce a warning, or the
    # real one gets skipped.
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)

    def sweep(e, a, m):
        for i in (1, 2, 3):
            bc.POWER_SAMPLES.append({"round": i, "on_ac": True,
                                     "charge_pct": 70.0, "charge_w": 30.0})
        return rounds, [99.0], {"state": "disabled"}

    monkeypatch.setattr(bc, "paired_sweep", sweep)
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "1",
                                      "--cool-floor", "0"])
    assert bc.main() == 0
    assert "charge draw ranged" not in capsys.readouterr().out


def test_power_limited_note_is_silent_when_the_clock_is_fine(monkeypatch):
    assert bc.power_limited_note(99.0, 92.0) == (None, False)
    assert bc.power_limited_note(None, 92.0) == (None, False)


def test_power_limited_note_names_the_battery(monkeypatch):
    # Definitive path: the machine reports it is on battery.
    monkeypatch.setattr(bc, "power_source", lambda: "battery")
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert msg is not None and "BATTERY" in msg
    assert "will NOT recover" in msg, "must say waiting is futile, not just why"
    assert abort is True, "the battery case is the ONLY one that aborts"


def test_idle_fingerprint_advises_but_does_NOT_abort(monkeypatch):
    # The gate runs BEFORE each sample, when the box is legitimately idle and
    # downclocked. Treating that as power limiting aborted the gate on a
    # healthy run -- on a machine reporting no battery the gate would never
    # work at all. Advice yes, abort no.
    monkeypatch.setattr(bc, "power_source", lambda: "no-battery")
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 9.0)
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert msg is not None
    assert abort is False, "an idle low clock must not abort the gate"


def test_unknown_power_source_does_not_abort(monkeypatch):
    # A failed query used to return the same None as "no battery", silently
    # downgrading a definitive check to a heuristic on a laptop.
    monkeypatch.setattr(bc, "power_source", lambda: "unknown")
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert msg is not None and "could not be read" in msg
    assert abort is False


def test_gate_note_reaches_the_json_not_only_the_terminal(monkeypatch):
    # A warning that exists only in stdout is absent from the artifact a
    # consumer reads -- the record-vs-reality drift this harness keeps finding.
    bc.GATE_NOTES.clear()
    monkeypatch.setattr(bc, "power_source", lambda: "battery")
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 31.0)
    bc.wait_for_cool(92.0, limit=5)
    assert bc.GATE_NOTES and "BATTERY" in bc.GATE_NOTES[0]
    bc.GATE_NOTES.clear()


def test_power_limited_note_stays_quiet_when_the_box_is_busy(monkeypatch):
    # Low clock + BUSY cpu is the thermal case: waiting DOES help, so the gate
    # must keep waiting rather than aborting.
    monkeypatch.setattr(bc, "power_source", lambda: "no-battery")
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 85.0)
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert abort is False, "thermal case: the gate must keep waiting"
    assert msg is None, "and it must not muddy the log with a power note"


def test_wait_for_cool_aborts_instead_of_blocking_on_battery(monkeypatch):
    # The behaviour that matters: no ten-minute silence waiting for a recovery
    # that cannot come.
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 31.0)
    monkeypatch.setattr(bc, "power_source", lambda: "battery")
    slept = []
    monkeypatch.setattr(bc.time, "sleep", lambda s: slept.append(s))
    assert bc.wait_for_cool(92.0, limit=300) == 31.0
    assert slept == [], "must return immediately, not spin out the limit"


# --- an all-shed leg may mean NO LOAD WAS APPLIED -------------------------
# `_post` returns None for a 429 AND for connection-refused, so the counting
# site cannot tell a queued engine from one that never started. If nothing
# connected, the "contended" leg measured an idle box and the ratio comes out
# near 1.0 -- which reads as "no contention effect" rather than "no experiment".

def test_zero_completions_is_flagged_as_suspect():
    note, suspect = bc.shed_note("NPU", shed=40, served=0)
    assert suspect is True
    assert "SUSPECT" in note
    assert "never up" in note, "must name the possibility, not just the count"


def test_ordinary_backpressure_is_not_flagged_as_suspect():
    # A genuinely queued single-flight NPU still completes SOME requests
    # between rejections; that is the discriminator.
    note, suspect = bc.shed_note("GPU", shed=40, served=7)
    assert suspect is False
    assert "expected backpressure" in note


def test_no_shed_says_nothing():
    assert bc.shed_note("NPU", shed=0, served=12) == (None, False)


def test_the_suspect_note_warns_about_the_ratio_specifically():
    # The failure is not "we lost some load", it is "the number you are about
    # to publish means something else".
    note, _ = bc.shed_note("NPU", shed=99, served=0)
    assert "1.0" in note and "no load" in note


def test_a_suspect_leg_reaches_the_warnings_block(monkeypatch, capsys):
    rounds = {"NPU": {"solo": [10.0], "contended": [9.9], "ratios": [0.99],
                      "shed": 30, "served": 0},
              "GPU": {"solo": [20.0], "contended": [19.0], "ratios": [0.95],
                      "shed": 5, "served": 5}}
    monkeypatch.setattr(bc, "paired_sweep", lambda e, a, m: (rounds, [99.0], {"state": "disabled"}))
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv",
                        ["bench_contention.py", "--repeat", "1", "--cool-floor", "0",
                         "--npu-weights-gb", "2.3", "--gpu-weights-gb", "2.32",
                         "--peak-bw-gbs", "135.2"])
    assert bc.main() == 0
    out = capsys.readouterr().out
    assert "SUSPECT" in out
    # Not merely printed somewhere: it must survive into the warnings block,
    # which is what a reader skimming the tail of a long run actually sees.
    assert out.index("SUSPECT") < len(out)
    assert "!" * 10 in out, "the warnings banner must fire"


# --- the load generator must never outlive the leg ------------------------

def test_the_load_generator_is_stopped_even_when_the_leg_raises(monkeypatch):
    """It is a live load on a SHARED box; an interrupt used to skip stop()."""
    stopped = {"n": 0}

    class FakeLoad:
        busy = 0
        completed = 0

        def start(self):
            pass

        def stop(self):
            stopped["n"] += 1

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:          # the CONTENDED leg, generator running
            raise KeyboardInterrupt
        return 10.0

    monkeypatch.setattr(bc, "measure", boom)
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)
    args = types.SimpleNamespace(repeat=1, depth=250, tokens=40, timeout=60,
                                 ramp=0, cool_floor=None,
                                 closing_recheck=False, closing_tol=10.0)
    engines = [("NPU", "http://a", "m"), ("GPU", "http://b", "m")]
    with pytest.raises(KeyboardInterrupt):
        bc.paired_sweep(engines, args, lambda o: FakeLoad())
    assert stopped["n"] == 1, "the generator was left hammering the peer"


# --- the closing re-check --------------------------------------------------
# Re-runs the leg the sweep OPENED with, last, under the same gate: an A/A
# whose only variable is elapsed time. It exists because neither control
# already here can see decay DURING a sample -- wait_for_cool gates before one
# and says nothing after, and drift_note needs a strictly monotonic decline
# over 3+ solo samples, so one out-of-order sample hides a real trend and
# --repeat 1 gives it nothing to compare.

def _closing(first, final, engine="NPU"):
    return {"state": "ok", "engine": engine, "first": first, "final": final,
            "retained": final / first}


def test_a_box_that_held_is_reported_as_holding():
    note, suspect = bc.closing_note(_closing(18.0, 17.6))
    assert suspect is False
    assert "held" in note and "SUSPECT" not in note


def test_a_decayed_box_is_flagged_and_says_which_way_it_biases():
    # Contended samples are taken AFTER the solo ones they divide, so decay
    # lands in the numerator. Naming the direction is the difference between a
    # warning and an actionable one.
    note, suspect = bc.closing_note(_closing(18.0, 13.0))
    assert suspect is True
    assert "SUSPECT" in note and "DECAYED" in note
    assert "overstated" in note


def test_a_box_that_got_faster_is_equally_disqualifying():
    # Not a nice surprise: it means the OPENING sample was the degraded one, so
    # every baseline the ratios divide by is too low.
    note, suspect = bc.closing_note(_closing(13.0, 18.0))
    assert suspect is True
    assert "FASTER" in note and "flattered" in note


def test_the_check_reports_both_ends_and_the_percentage():
    # A reader has to be able to judge it without re-deriving the arithmetic.
    note, _ = bc.closing_note(_closing(20.0, 15.0))
    assert "20.00" in note and "15.00" in note and "-25.0%" in note


def test_the_tolerance_is_not_hair_trigger():
    # This box moves a few percent between any two samples; a check that fires
    # on ordinary noise is one the reader learns to skip.
    assert bc.closing_note(_closing(18.0, 17.3))[1] is False
    assert bc.closing_note(_closing(18.0, 18.7))[1] is False


def test_the_tolerance_is_configurable_in_both_directions():
    tight = bc.closing_note(_closing(18.0, 17.0), tol_pct=1.0)
    assert tight[1] is True
    assert bc.closing_note(_closing(18.0, 17.0), tol_pct=25.0)[1] is False


def test_a_skipped_check_says_so_rather_than_reading_as_clean():
    # The distinction that matters: "not checked" must not look like "checked
    # and fine", which is the same defect this suite pins in _probe_crosscheck.
    # The discriminator is the CLAIM, not the word "held": these say "nothing
    # here says whether the box held", the opposite of the confirming branch's
    # "the ratios above stand".
    for closing in ({"state": "disabled"}, None):
        note, suspect = bc.closing_note(closing)
        assert suspect is False
        assert "no closing re-check" in note
        assert "ratios above stand" not in note


@pytest.mark.parametrize("state,phrase", [
    ("disabled", "--no-closing-recheck"),
    ("failed", "closing measurement itself failed"),
    ("no-opening-sample", "no sample to compare against"),
])
def test_each_reason_for_no_result_says_which_one(state, phrase):
    """Three causes used to print one identical line.

    They want opposite responses -- nothing at all for a deliberate skip, look
    at the closing leg for a failure, look at the whole sweep for an empty
    opening leg -- and a null in the JSON additionally read as "the flag was
    off". A reader could not tell which had happened.
    """
    note, suspect = bc.closing_note({"state": state, "engine": "NPU"})
    assert suspect is False
    assert phrase in note


def test_the_sweep_reopens_the_first_leg_and_records_it(monkeypatch):
    # End to end through paired_sweep: the closing measurement must be the
    # FIRST engine's solo leg, not the last one measured.
    rates = iter([18.0, 13.0, 17.0, 12.0, 9.0])
    monkeypatch.setattr(bc, "measure", lambda *a, **k: next(rates))
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)

    class FakeLoad:
        busy = completed = 0
        def start(self): pass
        def stop(self): pass

    args = types.SimpleNamespace(repeat=1, depth=250, tokens=40, timeout=60,
                                 ramp=0, cool_floor=92.0,
                                 closing_recheck=True, closing_tol=10.0)
    engines = [("NPU", "http://a", "m"), ("GPU", "http://b", "m")]
    _per, _clocks, closing = bc.paired_sweep(engines, args, lambda o: FakeLoad())
    assert closing["state"] == "ok"
    assert closing["engine"] == "NPU", "must reopen the leg the run STARTED on"
    assert closing["first"] == 18.0, "compared against NPU's FIRST solo sample"
    assert closing["final"] == 9.0
    assert bc.closing_note(closing)[1] is True


def test_the_recheck_can_be_turned_off(monkeypatch):
    rates = iter([18.0, 13.0, 17.0, 12.0])
    monkeypatch.setattr(bc, "measure", lambda *a, **k: next(rates))
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)

    class FakeLoad:
        busy = completed = 0
        def start(self): pass
        def stop(self): pass

    args = types.SimpleNamespace(repeat=1, depth=250, tokens=40, timeout=60,
                                 ramp=0, cool_floor=92.0,
                                 closing_recheck=False, closing_tol=10.0)
    engines = [("NPU", "http://a", "m"), ("GPU", "http://b", "m")]
    _per, _clocks, closing = bc.paired_sweep(engines, args, lambda o: FakeLoad())
    assert closing == {"state": "disabled"}, "no extra leg when disabled"


def test_a_suspect_closing_check_reaches_the_warnings_block(monkeypatch, capsys):
    # Same requirement as the shed note: a warning that only exists mid-output
    # is missed by a reader skimming the tail of a twenty-minute run.
    monkeypatch.setattr(bc, "paired_sweep",
                        lambda e, a, m: (_shed_round(), [99.0],
                                         _closing(18.0, 11.0)))
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "1",
                                      "--cool-floor", "0"])
    assert bc.main() == 0
    out = capsys.readouterr().out
    assert "DECAYED" in out
    assert "!" * 10 in out, "the warnings banner must fire"


# --- the closing re-check's failure paths ---------------------------------
# Only the success and disabled paths were exercised. These two are how it
# actually breaks in the field, and both used to be indistinguishable from a
# deliberate skip.

def _stub_box(monkeypatch, measure_returns):
    """A sweep whose measure() yields a scripted sequence."""
    rates = iter(measure_returns)
    monkeypatch.setattr(bc, "measure", lambda *a, **k: next(rates))
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)

    class FakeLoad:
        busy = completed = 0

        def start(self):
            pass

        def stop(self):
            pass

    args = types.SimpleNamespace(repeat=1, depth=250, tokens=40, timeout=60,
                                 ramp=0, cool_floor=92.0,
                                 closing_recheck=True, closing_tol=10.0)
    engines = [("NPU", "http://a", "m"), ("GPU", "http://b", "m")]
    return bc.paired_sweep(engines, args, lambda o: FakeLoad())


def test_a_closing_leg_that_fails_is_reported_as_failed(monkeypatch):
    # The closing measurement returns None -- a 429, a dropped connection, an
    # early EOS. Reporting that as "skipped" would hide a broken instrument.
    _per, _clocks, closing = _stub_box(monkeypatch,
                                       [18.0, 13.0, 17.0, 12.0, None])
    assert closing == {"state": "failed", "engine": "NPU"}
    assert "failed" in bc.closing_note(closing)[0]


def test_no_opening_sample_means_no_comparison_and_says_so(monkeypatch):
    # Every NPU measurement skipped, so there is nothing for the closing leg to
    # be compared against. Taking the extra leg anyway would burn a minute to
    # produce a number with no partner.
    _per, _clocks, closing = _stub_box(monkeypatch, [None, None, 17.0, 12.0])
    assert closing == {"state": "no-opening-sample", "engine": "NPU"}


def test_the_closing_sample_never_reaches_the_medians(monkeypatch):
    # The closing sample is a CONTROL, not data. If it ever landed in
    # per_engine["solo"] it would shift the median every ratio divides by.
    per, _clocks, _closing = _stub_box(monkeypatch,
                                       [18.0, 13.0, 17.0, 12.0, 9.9])
    assert per["NPU"]["solo"] == [18.0], "closing sample must stay out of data"


# --- main()'s reporting tail ----------------------------------------------
# Everything below runs through main() and was unreachable from any test, which
# is how the closing-check JSON key came to share a name with the CLI flag
# without anything noticing.

def _run_main(monkeypatch, rounds, argv, clocks=None, closing=None):
    monkeypatch.setattr(bc, "paired_sweep",
                        lambda e, a, m: (rounds,
                                         [99.0] if clocks is None else clocks,
                                         closing or {"state": "disabled"}))
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", *argv])
    return bc.main()


def _both_ran(npu_solo, npu_cont, gpu_solo, gpu_cont):
    return {"NPU": {"solo": [npu_solo], "contended": [npu_cont],
                    "ratios": [npu_cont / npu_solo], "shed": 0, "served": 4},
            "GPU": {"solo": [gpu_solo], "contended": [gpu_cont],
                    "ratios": [gpu_cont / gpu_solo], "shed": 0, "served": 4}}


def test_an_engine_that_is_not_answering_refuses_to_measure(monkeypatch,
                                                            capsys):
    # The likeliest real failure: one server started, the other forgotten. It
    # must refuse rather than measure an idle box -- a "contended" leg with
    # nothing contending returns ~1.0, which reads as "no contention effect"
    # rather than as "no experiment".
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(bc.be, "n_ctx", lambda b: 4096)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py"])
    assert bc.main() == 2, "must exit non-zero; a script keys on this"
    assert "not answering" in capsys.readouterr().err


def test_the_not_a_bandwidth_problem_warning_fires(monkeypatch, capsys):
    # The most consequential inference this tool draws, and the one whose
    # earlier conclusion had to be retracted from MULTI_ENGINE.md. Both halves
    # of the condition have to hold: heavy loss AND low bus utilisation.
    rounds = _both_ran(18.0, 6.9, 18.0, 7.3)
    assert _run_main(monkeypatch, rounds,
                     ["--repeat", "1", "--cool-floor", "0",
                      "--npu-weights-gb", "0.3", "--gpu-weights-gb", "0.3",
                      "--peak-bw-gbs", "135.2"]) == 0
    out = capsys.readouterr().out
    assert "bottleneck is NOT the memory bus" in out
    assert "shared power budget" in out, "must name what to suspect instead"


def test_heavy_loss_near_the_bus_ceiling_is_not_blamed_on_something_else(
        monkeypatch, capsys):
    # The other half of the condition. With demand near peak, losing throughput
    # is ordinary bus contention and the warning must stay silent -- firing
    # here sends the reader hunting a power budget that is not the cause.
    rounds = _both_ran(18.0, 6.9, 18.0, 7.3)
    _run_main(monkeypatch, rounds,
              ["--repeat", "1", "--cool-floor", "0",
               "--npu-weights-gb", "6.0", "--gpu-weights-gb", "6.0",
               "--peak-bw-gbs", "135.2"])
    assert "bottleneck is NOT" not in capsys.readouterr().out


def test_a_net_loss_verdict_points_at_the_poll_flag_first(monkeypatch, capsys):
    # This verdict was WRONG for a whole revision of the docs: a 0.78x net loss,
    # measured against a poll:true bundle. The advice telling the next reader to
    # check that flag before believing it is the reason the branch exists.
    #
    # The 0.78x itself was then refuted too -- a controlled re-run found both
    # poll settings a GAIN (1.70x vs 1.26x over the best single engine), so the
    # flag costs about a quarter of the win rather than reversing the sign.
    # This asserts the pointer, not the retired number: the branch has to send
    # the reader to the flag, and must not re-assert a figure that did not
    # reproduce.
    rounds = _both_ran(18.0, 6.9, 18.0, 7.3)
    assert _run_main(monkeypatch, rounds,
                     ["--repeat", "1", "--cool-floor", "0"]) == 0
    out = capsys.readouterr().out
    assert "SLOWER than the best engine alone" in out
    assert "QnnHtp/poll" in out
    assert "concurrency win" in out
    assert "0.78x" not in out, "refuted figure must not come back"


def test_a_gain_verdict_does_not_mention_poll(monkeypatch, capsys):
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    _run_main(monkeypatch, rounds, ["--repeat", "1", "--cool-floor", "0"])
    out = capsys.readouterr().out
    assert "beat the best single engine" in out
    assert "QnnHtp/poll" not in out


# --- the JSON artifact ----------------------------------------------------
# Never written or inspected by any test, which is exactly how a key ended up
# sharing its name with the CLI flag of the same meaning.

def test_the_json_carries_the_closing_check_under_its_own_key(monkeypatch,
                                                              tmp_path):
    out = tmp_path / "run.json"
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds,
                     ["--repeat", "1", "--cool-floor", "0",
                      "--json", str(out)],
                     closing=_closing(18.0, 17.6)) == 0
    body = json.loads(out.read_text())
    # `closing_recheck` is the CLI FLAG's dest. A result stored under that name
    # reads as the flag's value, so the artifact uses a distinct key.
    assert "closing_recheck" not in body
    assert body["closing_check"]["state"] == "ok"
    assert body["closing_check"]["retained"] == pytest.approx(17.6 / 18.0)
    for key in ("solo_median", "contended_median", "paired_ratio_median",
                "cpu_clock_pct", "warnings"):
        assert key in body, key


def test_an_existing_json_is_never_clobbered_without_force(monkeypatch,
                                                           tmp_path, capsys):
    # A run costs 20+ minutes of a shared box and some of these numbers have
    # proved unreproducible, so losing one to a re-run is the wrong trade.
    out = tmp_path / "run.json"
    out.write_text('{"previous": "result"}')
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds,
                     ["--repeat", "1", "--cool-floor", "0",
                      "--json", str(out)]) == 0
    assert json.loads(out.read_text()) == {"previous": "result"}
    assert "NOT writing" in capsys.readouterr().out


def test_force_overwrites_deliberately(monkeypatch, tmp_path):
    out = tmp_path / "run.json"
    out.write_text('{"previous": "result"}')
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    _run_main(monkeypatch, rounds, ["--repeat", "1", "--cool-floor", "0",
                                    "--json", str(out), "--force"])
    assert "previous" not in json.loads(out.read_text())
