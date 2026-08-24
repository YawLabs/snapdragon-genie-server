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
