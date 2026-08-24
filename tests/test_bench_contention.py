"""Device-free tests for bench_contention.py's reporting paths.

The two crash bugs these cover were both reachable from a single round where
every contended sample was skipped, and that is not a rare shape: the NPU sheds
by design once its small queue fills (429 OpenAI / 529 Anthropic), so a
contention run against it is the workload MOST likely to produce an asymmetric
solo/contended result. Both guards had assumed it could not happen.

No device, no server, no network -- bench_endpoint is stubbed at import.
"""

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
    monkeypatch.setattr(bc, "paired_sweep", lambda e, a, m: (_shed_round(), [99.0]))
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
    monkeypatch.setattr(bc, "paired_sweep", lambda e, a, m: (both_shed, [99.0]))
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

    monkeypatch.setattr(bc, "paired_sweep", lambda e, a, m: (_shed_round(), []))
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
