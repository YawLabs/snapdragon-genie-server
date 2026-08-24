"""Tests for the benchmark harness itself.

The server had 78 tests while the tool that produces every number the docs
quote had none -- and a review then found three defects in it, all in pure
functions a test could have pinned. These cover the ones that actually bit,
plus the verdict logic that a bad reading of would invert the headline finding.

Device-free like the rest: no NPU, no bundle, no server. `chat` is stubbed
where a measurement is needed, which is enough because every case here is
arithmetic over its return value.
"""

import importlib.util
import os

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_endpoint", os.path.join(SRC, "bench_endpoint.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def be():
    return _load()


# --- prefill correction ---------------------------------------------------
# A 1-token cap still generates a token, so raw wall is prefill + one step and
# the step is subtracted. The subtraction is only safe when the probe that
# measured the step is credible: a probe taken while something else had the box
# reads far too slow and silently inflates every prefill number in the run.

def _stub_chat(be, wall, prompt_tokens=469):
    be.chat = lambda *a, **k: {"prompt_tokens": prompt_tokens,
                               "completion_tokens": 1, "wall": wall}


def test_prefill_subtracts_a_credible_probe(be):
    _stub_chat(be, wall=0.40)
    rate = be.measure_prefill("b", "m", 469, 1, per_step=0.055)
    assert rate == pytest.approx(469 / (0.40 - 0.055), rel=1e-6)


def test_prefill_refuses_an_implausible_probe(be):
    # Real case: probe read 1.546 s/token against a true 0.127. Subtracting it
    # turned a genuine ~206 tok/s into a reported 549.
    _stub_chat(be, wall=2.40)
    rate = be.measure_prefill("b", "m", 469, 1, per_step=1.546)
    assert rate == pytest.approx(469 / 2.40, rel=1e-6), "should report RAW"
    assert rate < 250, "the inflated 549 tok/s must not come back"


def test_prefill_never_divides_by_the_floor(be):
    # per_step > wall used to hit max(1e-9, ...) and print 4.69e+11 tok/s --
    # a garbage number that reads as a measurement.
    _stub_chat(be, wall=0.21)
    rate = be.measure_prefill("b", "m", 469, 1, per_step=1.546)
    assert rate == pytest.approx(469 / 0.21, rel=1e-6)
    assert rate < 1e5, "must not produce an astronomical rate"


def test_prefill_with_no_probe_reports_raw(be):
    _stub_chat(be, wall=0.40)
    assert be.measure_prefill("b", "m", 469, 1, per_step=0.0) == pytest.approx(469 / 0.40)


def test_prefill_returns_none_when_the_request_failed(be):
    be.chat = lambda *a, **k: None
    assert be.measure_prefill("b", "m", 469, 1, per_step=0.05) is None


# --- --depths resolution --------------------------------------------------
# The health check runs before this, so a bad --depths was unreachable from any
# test and surfaced only as a traceback in a user's terminal.

def test_depths_parses_a_list(be):
    depths, note = be.resolve_depths("250,3300", 4000, 4096, 60)
    assert depths == [250, 3300] and note is None


def test_depths_rejects_non_integers_with_a_message(be):
    with pytest.raises(ValueError) as e:
        be.resolve_depths("500,abc", 4000, 4096, 60)
    assert "comma-separated integers" in str(e.value)


def test_depths_drops_out_of_budget_values_and_says_so(be):
    depths, note = be.resolve_depths("250,9999", 4000, 4096, 60)
    assert depths == [250]
    assert note and "9999" in note, "a silently shortened sweep reads as complete"


def test_depths_rejects_when_nothing_fits(be):
    with pytest.raises(ValueError) as e:
        be.resolve_depths("9999", 4000, 4096, 60)
    assert "budget" in str(e.value)


def test_depths_default_respects_the_budget(be):
    depths, note = be.resolve_depths(None, 2000, 4096, 60)
    assert depths and max(depths) < 2000 and note is None


def test_depths_allows_repeats_for_interleaving(be):
    # Interleaving is how drift is separated from a real depth effect; it is
    # expressed by repeating depths, so dedup here would break the technique.
    depths, _ = be.resolve_depths("250,3300,250,3300", 4000, 4096, 60)
    assert depths == [250, 3300, 250, 3300]


# --- the flat-vs-varies verdict -------------------------------------------
# This line states the repo's headline finding. Computed over the POOLED rates
# it reported same-depth noise as a depth effect and could invert the claim.

def test_verdict_calls_agreeing_depths_flat_despite_noise(be, capsys):
    be._verdict([9.5, 13.2, 12.4], [12.1])   # medians 12.4 vs 12.1
    out = capsys.readouterr().out
    assert "flat" in out and "varies with depth" not in out


def test_verdict_reports_noise_separately_from_depth(be, capsys):
    be._verdict([9.5, 13.2, 12.4], [12.1])
    out = capsys.readouterr().out
    assert "cross-depth delta 0.30" in out
    assert "same-depth noise 3.70" in out


def test_verdict_calls_a_real_depth_effect_varying(be, capsys):
    be._verdict([18.9, 18.5, 18.7], [12.8, 13.0, 13.3])
    assert "varies with depth" in capsys.readouterr().out


def test_verdict_silent_without_both_groups(be, capsys):
    be._verdict([12.0], [])
    assert capsys.readouterr().out == ""


# --- error reporting ------------------------------------------------------

def test_describe_surfaces_the_servers_own_message(be):
    import io as _io
    import urllib.error
    body = b'{"error": {"message": "server busy; NPU is single-flight"}}'
    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {},
                                 _io.BytesIO(body))
    msg = be._describe(err)
    assert "429" in msg and "single-flight" in msg


def test_describe_handles_a_bodyless_error(be):
    import io as _io
    import urllib.error
    err = urllib.error.HTTPError("u", 500, "boom", {}, _io.BytesIO(b""))
    assert "500" in be._describe(err)


def test_prompt_of_is_about_the_requested_size(be):
    p = be.prompt_of(500)
    assert 500 * be.CHARS_PER_TOKEN <= len(p) <= 500 * be.CHARS_PER_TOKEN + 60


# --- the --prefill-only probe cross-check ---------------------------------
# With --prefill-only there is no decode phase to check the probe against, and
# the probe is the ONLY thing shaping the reported numbers. One taken during a
# blip corrupts every figure silently, so the sweep is bracketed by two probes.
# This is a WARNING, and a warning that quietly stops firing is worse than
# none: the run then looks clean precisely when it is not.

def test_crosscheck_accepts_a_box_that_held(be):
    msg = be._probe_crosscheck(0.055, 0.056)
    assert "consistent" in msg
    assert "WARNING" not in msg


def test_crosscheck_warns_when_the_box_drifted(be):
    # 0.055 -> 0.12 s/step is 18.2 vs 8.3 t/s: the corrections above it were
    # computed from a rate the box no longer had.
    msg = be._probe_crosscheck(0.055, 0.12)
    assert "WARNING" in msg
    assert "did not hold" in msg
    assert "Re-run quiet" in msg, "must say what to do about it"


def test_crosscheck_reports_both_rates_so_the_reader_can_judge(be):
    msg = be._probe_crosscheck(0.05, 0.20)
    assert "20.00" in msg and "5.00" in msg


def test_crosscheck_is_symmetric(be):
    # A box that got FASTER mid-sweep is equally disqualifying: it means the
    # opening probe was the contended one, so the corrections were too large.
    slow_then_fast = be._probe_crosscheck(0.20, 0.05)
    fast_then_slow = be._probe_crosscheck(0.05, 0.20)
    assert "WARNING" in slow_then_fast and "WARNING" in fast_then_slow


def test_crosscheck_says_so_when_the_closing_probe_failed(be):
    # Distinct from "checked and fine" -- an unchecked run must not read as a
    # verified one.
    msg = be._probe_crosscheck(0.055, None)
    assert "could not be cross-checked" in msg
    assert "consistent" not in msg


def test_crosscheck_tolerance_is_not_hair_trigger(be):
    # Decode on this engine swings run to run; a threshold that fires on
    # ordinary noise would train the reader to ignore it.
    assert "WARNING" not in be._probe_crosscheck(0.055, 0.075)
