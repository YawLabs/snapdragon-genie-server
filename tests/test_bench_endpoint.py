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


# --- pooling interleaved depths before the verdict ------------------------
# The verdict line states this repo's headline finding, and it was computed
# from the first and last non-empty GROUPS. That is right only when the depths
# run once each in ascending order -- and wrong in exactly the mode the module
# docstring recommends, where depths alternate so that drift cannot masquerade
# as a depth effect. The per-depth table printed every row regardless, so the
# discarded repeats left no trace.

def test_pooling_keeps_every_repeat_of_an_interleaved_sweep(be):
    # --depths 250,3300,250,3300 --decode-every. first-vs-last groups gave
    # [18.9] against [13.3]: n=1 a side, three quarters of the run thrown away.
    per_depth = [(250, [18.9]), (3300, [12.8]), (250, [18.5]), (3300, [13.3])]
    shallow, deep = be.pool_by_depth(per_depth)
    assert sorted(shallow) == [18.5, 18.9]
    assert sorted(deep) == [12.8, 13.3]


def test_pooling_compares_the_extremes_not_the_order_measured(be):
    # The deepest depth is measured FIRST here, so taking the last group would
    # report the middle depth as "deep".
    per_depth = [(3300, [13.0]), (250, [18.7]), (1200, [15.0])]
    shallow, deep = be.pool_by_depth(per_depth)
    assert shallow == [18.7] and deep == [13.0]


def test_pooling_ignores_depths_whose_every_sample_was_skipped(be):
    # A depth can come back empty (429 backpressure, an early EOS); it is not a
    # side of a cross-depth comparison.
    shallow, deep = be.pool_by_depth([(250, [18.5]), (1200, []), (3300, [13.0])])
    assert shallow == [18.5] and deep == [13.0]


def test_pooling_makes_no_claim_from_a_single_depth(be):
    # One depth cannot support a statement ABOUT depth, however many repeats.
    assert be.pool_by_depth([(250, [18.5, 18.9, 18.7])]) == ([], [])
    assert be.pool_by_depth([(250, []), (3300, [])]) == ([], [])


def test_verdict_hint_names_the_flag_that_actually_applies(be, capsys):
    # --decode-every measures every depth with --repeat and never consults
    # --repeat-deep, so naming it there sends the reader to a flag that changes
    # nothing about the run they just did.
    be._verdict([18.5], [13.0], deep_flag="--repeat")
    out = capsys.readouterr().out
    assert "--repeat raises that" in out
    assert "--repeat-deep" not in out


def test_verdict_hint_defaults_to_the_two_depth_flag(be, capsys):
    be._verdict([18.5], [13.0])
    assert "--repeat-deep" in capsys.readouterr().out


# --- the N-minus-1 decode delta -------------------------------------------
# Every decode figure this repo quotes comes out of this subtraction, and the
# prefill correction is derived from it too. The two requests differ only in
# the cap, so prefill, connection setup and template rendering occur in both
# and cancel; returning the TOTALS instead would fold a full prefill into the
# decode rate and understate it badly at depth.

def _run(completion_tokens, wall, prompt_tokens=469):
    return {"prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "wall": wall}


def _stub_chat_seq(be, *results):
    """Stub `chat` with one return value per call, recording the calls.

    _delta_run's whole point is that its two requests come back DIFFERENT, so
    the single-value _stub_chat above cannot express it.
    """
    calls = []

    def fake(base, model, prompt, max_tokens, timeout):
        calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return results[len(calls) - 1]

    be.chat = fake
    return calls


def test_delta_run_returns_the_difference_not_the_totals(be):
    # Deep prefill dominates the 1-token run: 3.00s of the 8.12s total is not
    # decode. Reporting the totals gives 65/8.12 = 8.0 t/s for a bundle that
    # is really doing 12.5.
    _stub_chat_seq(be, _run(1, 3.00), _run(65, 8.12))
    one, many, steps, secs = be._delta_run("b", "m", "p", 64, 1)
    assert steps == 64
    assert secs == pytest.approx(5.12, rel=1e-6)
    assert steps / secs == pytest.approx(12.5, rel=1e-6)
    assert secs < many["wall"], "the totals would report 8.0 t/s, a third low"
    # measure_decode unpacks this positionally and names the depth off `many`.
    assert one["completion_tokens"] == 1 and many["completion_tokens"] == 65


def test_delta_run_sends_one_prompt_at_two_caps(be):
    # The cancellation is only valid because the two requests are identical
    # apart from the cap. A differing prompt would leave a prefill difference
    # in the delta and nothing downstream could tell.
    calls = _stub_chat_seq(be, _run(1, 3.00), _run(65, 8.12))
    be._delta_run("b", "m", "PROMPT", 64, 1)
    assert [c["max_tokens"] for c in calls] == [1, 65]
    assert calls[0]["prompt"] == calls[1]["prompt"] == "PROMPT"


def test_delta_run_returns_none_when_the_first_request_failed(be):
    calls = _stub_chat_seq(be, None, _run(65, 8.12))
    assert be._delta_run("b", "m", "p", 64, 1) is None
    assert len(calls) == 1, "no point paying for the long run once the pair is dead"


def test_delta_run_returns_none_when_the_second_request_failed(be):
    # A 429 on the second leg must skip the point, not produce a delta against
    # a missing run -- the sweep carries on around a gap.
    _stub_chat_seq(be, _run(1, 3.00), None)
    assert be._delta_run("b", "m", "p", 64, 1) is None


def test_delta_run_reports_no_window_when_the_model_stopped_early(be):
    # Both runs hit EOS at one token, so there is no decode window at all. A
    # rate derived from a zero- or one-token difference is noise printed as a
    # measurement.
    _stub_chat_seq(be, _run(1, 3.00), _run(1, 3.05))
    r = be._delta_run("b", "m", "p", 64, 1)
    assert r is not None, "callers distinguish 'no window' from 'request failed'"
    one, many, steps, secs = r
    assert steps == 0 and secs == 0.0
    assert many["prompt_tokens"] == 469, "measure_decode names the depth off this"


def test_delta_run_reports_no_window_when_the_delta_time_is_negative(be):
    # The long run coming back FASTER than the short one is queueing noise, not
    # a measurement; 64 / -0.06 would print -1066 t/s.
    _stub_chat_seq(be, _run(1, 3.00), _run(65, 2.94))
    _, _, steps, secs = be._delta_run("b", "m", "p", 64, 1)
    assert steps == 0 and secs == 0.0


# --- the decode probe that feeds the prefill correction -------------------
# This is where the per_step above comes from. It is subtracted from EVERY
# prefill figure in a run, so a wrong value here inflates the whole table at
# once -- and a failed probe must read as "no correction available" rather
# than as a correction of zero-ish size.

def test_a_window_too_small_to_be_a_rate_is_refused(be, capsys):
    """steps <= 0 was the only guard, and 4 is not 0.

    Measured 2026-08-27: a prompt whose answer ran to 5 tokens gave a 4-step
    window and reported 0.60 tok/s against a true 17.6 -- off by 29x, printed in
    the same column as a real measurement. Per-request overhead does not cancel
    perfectly between the two runs, and dividing its residue by four tokens
    produces something shaped like a rate with none of the meaning.
    """
    _stub_chat_seq(be, _run(1, 0.30), _run(5, 6.98))
    assert be.measure_decode("b", "m", 250, 120, 1) is None
    out = capsys.readouterr().out
    assert "REFUSED" in out and "4-step" in out
    assert "overhead, not decode" in out, "must say WHY, not just that it skipped"


def test_a_full_window_is_still_measured(be, capsys):
    # The counterpart: the floor must not swallow real samples. 104 steps is the
    # shape a healthy run produces at the same prompt and cap.
    _stub_chat_seq(be, _run(1, 0.28), _run(105, 6.17))
    rate = be.measure_decode("b", "m", 250, 120, 1)
    assert rate == pytest.approx(104 / 5.89, rel=1e-6)
    assert "REFUSED" not in capsys.readouterr().out


def test_the_step_floor_is_tunable(be, monkeypatch):
    # A caller deliberately measuring short generations needs a way down; the
    # default protects the common case rather than forbidding the rare one.
    monkeypatch.setattr(be, "MIN_DECODE_STEPS", 2)
    _stub_chat_seq(be, _run(1, 0.30), _run(5, 6.98))
    assert be.measure_decode("b", "m", 250, 120, 1) == pytest.approx(4 / 6.68,
                                                                     rel=1e-6)


# --- box state recorded alongside the numbers -----------------------------
# This module produced every published prefill and decode figure while
# recording no power or clock state at all. Its docstring told the OPERATOR to
# note the conditions, which is the same as not recording them.

def test_box_state_summary_is_silent_with_no_samples(be):
    # Nothing sampled must produce no claim -- absence of a reading is not a
    # reading, and main() distinguishes the two in its own output.
    be.BOX_SAMPLES.clear()
    assert be.box_state_summary() == []


def test_box_state_reports_ranges_not_a_verdict(be):
    # Magnitudes, so a reader applies their own threshold. The same run can be
    # sound for decode and worthless for prefill, which no single verdict says.
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.extend([
        {"label": "a", "on_ac": True, "charge_pct": 62.0, "charge_w": 30.0,
         "clock_pct": 99.0},
        {"label": "b", "on_ac": True, "charge_pct": 64.0, "charge_w": 28.0,
         "clock_pct": 97.0}])
    text = "\n".join(be.box_state_summary())
    assert "pack 62-64%" in text and "draw 28.0-30.0 W" in text
    assert "clock: 97-99% of base" in text
    assert "WARNING" not in text, "a healthy box must not warn"


def test_a_low_pack_warns_about_prefill_specifically(be):
    # The finding is asymmetric: prefill halves, decode holds. A warning that
    # said "results unreliable" would overstate it and get ignored.
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.append({"label": "a", "on_ac": True, "charge_pct": 14.0,
                           "charge_w": 31.0, "clock_pct": 96.0})
    text = "\n".join(be.box_state_summary())
    assert "WARNING" in text and "14%" in text
    assert "PREFILL" in text and "decode holds" in text


def test_running_on_battery_is_called_out(be):
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.extend([
        {"label": "a", "on_ac": True, "charge_pct": 80.0, "charge_w": 0.0,
         "clock_pct": 99.0},
        {"label": "b", "on_ac": False, "charge_pct": 79.0, "charge_w": 0.0,
         "clock_pct": 60.0}])
    assert "ON BATTERY" in "\n".join(be.box_state_summary())


def test_the_clock_warning_says_it_brackets_rather_than_covers(be):
    # Sampled BETWEEN measurements, so it cannot describe what happened during
    # one. Claiming otherwise is the over-read this repo keeps catching.
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.append({"label": "a", "on_ac": True, "charge_pct": 90.0,
                           "charge_w": 0.0, "clock_pct": 58.0})
    text = "\n".join(be.box_state_summary())
    assert "58%" in text and "brackets" in text


def test_an_unreadable_box_records_nothing_rather_than_zeros(be, monkeypatch):
    # box_state returns Nones off-Windows and on any query failure. Recording
    # a row of Nones would put a fake sample in the artifact.
    be.BOX_SAMPLES.clear()
    monkeypatch.setattr(be, "box_state", lambda: (None, None, None, None))
    be.note_box_state("x")
    assert be.BOX_SAMPLES == []


def test_battery_state_delegates_to_the_single_sampler(be, monkeypatch):
    # bench_contention calls this; two copies of the WMI query is two places
    # for it to drift.
    monkeypatch.setattr(be, "box_state", lambda: (True, 55.0, 12.0, 98.0))
    assert be.battery_state() == (True, 55.0, 12.0)


def test_decode_probe_returns_seconds_per_step(be):
    _stub_chat_seq(be, _run(1, 0.30), _run(9, 0.74))
    per_step = be.decode_probe("b", "m", 1, depth=500, steps=8)
    assert per_step == pytest.approx(0.055, rel=1e-6)
    assert per_step < 1, "seconds per step, not the 18.2 t/s reciprocal"


def test_decode_probe_returns_none_when_the_run_failed(be):
    be._delta_run = lambda *a, **k: None
    assert be.decode_probe("b", "m", 1) is None


def test_decode_probe_returns_none_when_there_was_no_decode_window(be):
    # steps==0 is the early-EOS case above. Dividing by it raises, and any
    # number returned here becomes a subtraction against every prefill row.
    be._delta_run = lambda *a, **k: ({}, {}, 0, 0.0)
    assert be.decode_probe("b", "m", 1) is None


def test_decode_probe_value_is_in_the_units_prefill_subtracts(be):
    # The two halves have to agree on orientation: hand measure_prefill the
    # reciprocal (18.2) and PROBE_MAX_SHARE refuses it, so every prefill figure
    # silently reverts to raw and reads low.
    _stub_chat_seq(be, _run(1, 0.30), _run(9, 0.74))
    per_step = be.decode_probe("b", "m", 1, depth=500, steps=8)
    _stub_chat(be, wall=0.40)
    rate = be.measure_prefill("b", "m", 469, 1, per_step=per_step)
    assert rate == pytest.approx(469 / (0.40 - 0.055), rel=1e-6)
