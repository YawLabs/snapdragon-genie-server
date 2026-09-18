"""Tests for the benchmark harness itself.

The server had 78 tests while the tool that produces every number the docs
quote had none -- and a review then found three defects in it, all in pure
functions a test could have pinned. These cover the ones that actually bit,
plus the verdict logic that a bad reading of would invert the headline finding.

Device-free like the rest: no NPU, no bundle, no server, and no subprocess.
`chat` is stubbed where a measurement is needed, which is enough because every
case here is arithmetic over its return value; the HTTP plumbing under it is
exercised against a stubbed urlopen, and the box-state sampler is stubbed in
the fixture so an accepted measurement never launches PowerShell.
"""

import http.client
import importlib.util
import io
import json
import os
import socket
import subprocess
import types
import urllib.error

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_endpoint", os.path.join(SRC, "bench_endpoint.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def be(monkeypatch):
    # Pinned, not inherited. MIN_DECODE_STEPS is read from os.environ at
    # import and this fixture re-executes the module per test, so a developer
    # shell exporting GENIE_MIN_DECODE_STEPS<=4 (the documented way to measure
    # short generations) failed the floor test for a reason that had nothing
    # to do with the code. The tests that exercise the env var itself set it
    # and reload deliberately.
    monkeypatch.delenv("GENIE_MIN_DECODE_STEPS", raising=False)
    mod = _load()
    # Every accepted prefill and decode sample records the box state, and the
    # real sampler launches powershell.exe (WMI plus Get-Counter, 30 s timeout
    # each). Five tests here were doing that for real, which broke the
    # docstring's "device-free" promise and would stretch to minutes on a box
    # with a wedged perf-counter stack. Tests that need a reading stub this
    # to a specific tuple themselves.
    mod.box_state = lambda: (None, None, None, None)
    return mod


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


def test_prefill_refuses_when_the_one_token_cap_was_not_honoured(be, capsys):
    # geniex ignores the legacy cap spelling and GenieAPIService ignores both:
    # the "1-token" leg then runs to EOS, one step is subtracted from a whole
    # generation, and prefill prints several-x low in the normal format.
    be.chat = lambda *a, **k: {"prompt_tokens": 469, "completion_tokens": 125,
                               "wall": 9.0}
    assert be.measure_prefill("b", "m", 469, 1, per_step=0.055) is None
    out = capsys.readouterr().out
    assert "REFUSED" in out and "125 tokens" in out
    assert "not honouring the cap" in out, "must say WHY, not just that it skipped"


def test_prefill_records_box_state_even_when_reported_raw(be):
    # The RAW branch returned before the sampler ran, so the one sample where
    # the box is most suspect was the one that recorded no box state.
    be.box_state = lambda: (True, 55.0, 12.0, 98.0)
    be.BOX_SAMPLES.clear()
    _stub_chat(be, wall=2.40)
    be.measure_prefill("b", "m", 469, 1, per_step=1.546)
    assert [s["label"] for s in be.BOX_SAMPLES] == ["prefill d469"]


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


def test_depths_default_respects_the_budget_and_names_what_it_dropped(be):
    # The default sweep used to drop its out-of-budget depths with note=None,
    # against the docstring two lines above it -- and this test pinned the
    # silence. On the 8192 bundle that is the 12000 default vanishing.
    depths, note = be.resolve_depths(None, 2000, 4096, 60)
    assert depths == [500, 1500]
    assert note and "default" in note
    assert "3000" in note and "7000" in note and "12000" in note


def test_depths_default_is_silent_when_everything_fits(be):
    depths, note = be.resolve_depths(None, 20000, 32768, 60)
    assert depths == list(be.DEFAULT_DEPTHS) and note is None


def test_depths_default_falls_back_to_one_depth_and_says_so(be):
    depths, note = be.resolve_depths(None, 400, 1024, 60)
    assert depths == [400]
    assert note and "probing depth 400 instead" in note


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
    out = capsys.readouterr().out
    assert "varies with depth" in out
    assert "inside the same-depth noise" not in out, "delta 5.7 against noise 0.5 needs no hedge"


def test_verdict_hedges_a_varies_call_that_sits_inside_the_noise(be, capsys):
    # delta 3.9 clears 25% of the smaller median (2.5), so the call is
    # "varies" -- but the same-depth spread is 7.0, and a reader was left to
    # notice that the two numbers on the line disagreed.
    be._verdict([9.0, 16.0, 14.0], [10.0, 10.2, 10.1])
    out = capsys.readouterr().out
    assert "varies with depth" in out
    assert "inside the same-depth noise" in out


def test_verdict_silent_without_both_groups(be, capsys):
    be._verdict([12.0], [])
    assert capsys.readouterr().out == ""


# --- error reporting ------------------------------------------------------

def test_describe_surfaces_the_servers_own_message(be):
    body = b'{"error": {"message": "server busy; NPU is single-flight"}}'
    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {},
                                 io.BytesIO(body))
    msg = be._describe(err)
    assert "429" in msg and "single-flight" in msg


def test_describe_handles_a_bodyless_error(be):
    err = urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b""))
    assert "500" in be._describe(err)


def test_prompt_of_is_about_the_requested_size(be):
    p = be.prompt_of(500)
    assert 500 * be.CHARS_PER_TOKEN <= len(p) <= 500 * be.CHARS_PER_TOKEN + 60


# --- the pre-flight /health exit ------------------------------------------
# Three different failures printed the same "no server ... start
# genie_server.py first": a server answering 503 because its engine is wedged
# IS a server, so is one that took the connection and then said nothing, and
# the advice named a launcher whose default port does not match --base.

def test_health_failure_names_an_unhealthy_server_as_up(be):
    body = b'{"state": "stalled", "detail": "no token for 130s", "status": "stalled"}'
    err = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, io.BytesIO(body))
    msg = be._health_failure("http://127.0.0.1:8123", err)
    assert "up but not ready" in msg and "503" in msg
    assert "stalled" in msg and "no token for 130s" in msg, "the server's own words"
    assert "do not start another" in msg
    assert "nothing listening" not in msg


def test_health_failure_survives_a_503_with_no_usable_body(be):
    err = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, io.BytesIO(b"[1,2]"))
    msg = be._health_failure("http://127.0.0.1:8123", err)
    assert "up but not ready" in msg and "503" in msg


def test_health_failure_reads_the_detail_out_of_an_error_object(be):
    # llama-server's and geniex's 503 shape puts the message under `error`
    # rather than `detail`. It is still the server's own account of itself,
    # and it is what tells the reader whether to wait or to restart.
    err = urllib.error.HTTPError(
        "u", 503, "Service Unavailable", {},
        io.BytesIO(b'{"error": {"message": "model not loaded"}}'))
    msg = be._health_failure("http://127.0.0.1:8123", err)
    assert "up but not ready" in msg and "503" in msg
    assert "model not loaded" in msg, "the server's own words"


def test_health_failure_names_the_launcher_and_both_ports_when_nothing_listens(be):
    err = urllib.error.URLError("[WinError 10061] No connection could be made")
    msg = be._health_failure("http://127.0.0.1:8123", err)
    assert "nothing listening" in msg and "10061" in msg
    assert "run-genie-server.ps1" in msg and "8123" in msg
    assert "GENIE_PORT" in msg and "8080" in msg, "genie_server.py direct serves elsewhere"


# What urllib raises BARE from waiting for or reading the response, for a
# listener that ENGAGED with the request and then gave nothing usable back.
# Each was reproduced against a real loopback socket before being listed here:
# a listener that accepts and never answers (TimeoutError), one that is not an
# HTTP server (BadStatusLine), and a 200 whose body is not JSON. A connection
# that was BROKEN is not in here -- see _TORN_DOWN below, which is a listener
# too but not necessarily one that is still there.
_UNANSWERED = [
    TimeoutError("timed out"),
    http.client.BadStatusLine("SSH-2.0-OpenSSH_9.5"),
    json.JSONDecodeError("Expecting value", "<html>", 0),
]


@pytest.mark.parametrize("err", _UNANSWERED, ids=lambda e: type(e).__name__)
def test_health_failure_names_an_accepted_but_unanswered_connection_as_a_listener(be, err):
    # The two-way split filed all of these under "nothing listening ... start
    # one", which is advice to launch a second server behind one that is
    # already holding the port.
    msg = be._health_failure("http://127.0.0.1:8123", err)
    assert "accepted the connection" in msg and "http://127.0.0.1:8123" in msg
    assert type(err).__name__ in msg, "what was actually seen, in the line"
    assert "do not start another" in msg
    assert "nothing listening" not in msg and "start one" not in msg


def test_the_unanswered_line_does_not_guess_the_cause(be):
    # Still starting, busy, stuck, or not an HTTP server at all: one symptom
    # from outside. The line offers the candidates and claims none of them.
    msg = be._health_failure("http://127.0.0.1:8123", TimeoutError("timed out"))
    assert "cannot tell which" in msg
    for cause in ("starting", "busy", "stuck"):
        assert cause in msg, cause


# A connection something TOOK and then BROKE. urllib wraps this one or leaves
# it bare depending on whether the break beat the send, and that is a race
# rather than a difference in the server: a loopback listener that accepts and
# closes with SO_LINGER 0, hit 40 times, produced both shapes in one run. So
# both shapes have to land on the same line, or one server gets two opposite
# verdicts from one run. The reasons are named individually because only these
# three move a URLError off the "nothing listening" side.
_TORN_DOWN = [
    ConnectionResetError(10054, "An existing connection was forcibly closed"),
    http.client.RemoteDisconnected("Remote end closed connection without response"),
    ConnectionAbortedError(10053, "An established connection was aborted"),
    BrokenPipeError(32, "Broken pipe"),
    urllib.error.URLError(
        ConnectionResetError(10054, "An existing connection was forcibly closed")),
    urllib.error.URLError(
        ConnectionAbortedError(10053, "An established connection was aborted")),
    urllib.error.URLError(BrokenPipeError(32, "Broken pipe")),
]


@pytest.mark.parametrize(
    "err", _TORN_DOWN,
    ids=lambda e: "%s(%s)" % (type(e).__name__, type(e.reason).__name__)
    if isinstance(e, urllib.error.URLError) else type(e).__name__)
def test_a_broken_connection_is_a_taken_one_wrapped_or_bare(be, err):
    # The wrapped half of this used to print "nothing listening at ... --
    # start one" while quoting the reset that contradicts it.
    msg = be._health_failure("http://127.0.0.1:8123", err)
    assert "accepted the connection and then broke it" in msg
    assert "http://127.0.0.1:8123" in msg
    assert type(err).__name__ in msg, "what was actually seen, in the line"
    assert "nothing listening" not in msg and "start one" not in msg


def test_the_broken_connection_line_does_not_promise_a_healthy_listener(be):
    # A reset comes from a wedged server resetting its connections AND from
    # one being torn down -- the same symptom. Telling the reader a listener
    # IS there is wrong half the time, so the line names both readings and
    # gives the test that separates them.
    msg = be._health_failure(
        "http://127.0.0.1:8123",
        ConnectionResetError(10054, "An existing connection was forcibly closed"))
    assert "do not read it as a healthy listener" in msg
    assert "cannot tell which" in msg
    assert "on its way out" in msg, "the dying-server reading, named"
    assert "REFUSED" in msg, "how to tell the two apart: re-run"
    assert "a listener IS there" not in msg
    assert "do not start another" not in msg


def test_a_connection_nothing_accepted_is_still_nothing_listening(be):
    # urllib wraps everything up to the send in URLError, so a CONNECT that
    # timed out is not the bare TimeoutError of an accepted-then-silent one;
    # a refusal took no connection either, wrapped or handed in unwrapped; a
    # name that does not resolve never reached a socket; and a plain
    # ValueError is a malformed --base, raised before any socket was opened.
    for err in (urllib.error.URLError(TimeoutError("timed out")),
                urllib.error.URLError(
                    ConnectionRefusedError(10061, "No connection could be made")),
                urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")),
                ConnectionRefusedError(10061, "No connection could be made"),
                ValueError("unknown url type: '127.0.0.1:8123'")):
        msg = be._health_failure("http://127.0.0.1:8123", err)
        assert "nothing listening" in msg and "accepted" not in msg, err


# --- the HTTP plumbing ----------------------------------------------------
# post_timed is what every timing here is formed from, and it is the plumbing
# bench_contention's load generator runs on. Its (body, wall) | (None, reason)
# contract was pinned by no test in either file, while the sibling's stub
# returned a float where the real failure path returns a string.

class _Resp:
    def __init__(self, raw):
        self.raw = raw

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_urlopen(monkeypatch, be, raw=b"{}", exc=None):
    seen = []

    def fake(req, timeout=None):
        seen.append(req)
        if exc is not None:
            raise exc
        return _Resp(raw)

    monkeypatch.setattr(be.urllib.request, "urlopen", fake)
    return seen


def test_post_timed_returns_the_body_and_a_float_wall(be, monkeypatch):
    _stub_urlopen(monkeypatch, be, raw=b'{"usage": {"prompt_tokens": 3}}')
    body, wall = be.post_timed("http://b", "/v1/chat/completions", {"x": 1}, 5)
    assert body == {"usage": {"prompt_tokens": 3}}
    assert isinstance(wall, float) and wall >= 0


def test_post_timed_returns_none_and_a_reason_string_on_failure(be, monkeypatch):
    body = b'{"error": {"message": "server busy; NPU is single-flight"}}'
    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(body))
    _stub_urlopen(monkeypatch, be, exc=err)
    got, reason = be.post_timed("http://b", "/p", {}, 5)
    assert got is None
    assert isinstance(reason, str), "the failure slot is a reason, never a wall time"
    assert "429" in reason and "single-flight" in reason


def test_post_timed_sends_json_to_the_path_with_the_content_type(be, monkeypatch):
    seen = _stub_urlopen(monkeypatch, be)
    be.post_timed("http://b", "/p", {"a": 1}, 5)
    req = seen[0]
    assert req.full_url == "http://b/p"
    assert json.loads(req.data) == {"a": 1}
    assert req.get_header("Content-type") == "application/json"


def test_post_timed_times_with_the_monotonic_clock(be, monkeypatch):
    # Every figure is a difference of two of these reads. The wall clock can
    # step between them (NTP, resume from sleep). bench.py already uses
    # perf_counter, and bench_servers.py times its requests through THIS
    # function, so pinning the clock here pins it for both HTTP tools.
    _stub_urlopen(monkeypatch, be)
    ticks = iter([100.0, 100.25])

    def boom():
        raise AssertionError("time.time() must not time a measurement")

    monkeypatch.setattr(be, "time", types.SimpleNamespace(
        perf_counter=lambda: next(ticks), time=boom))
    _, wall = be.post_timed("http://b", "/p", {}, 5)
    assert wall == pytest.approx(0.25)


def test_the_old_private_name_still_resolves(be):
    # Nothing in this repo spells it _post any more (bench_contention and
    # bench_servers both call post_timed). The alias is kept for a caller
    # OUTSIDE the repo that imported the old name, and this pins that it is
    # the same function rather than a second copy that could drift.
    assert be._post is be.post_timed


def test_get_treats_an_empty_200_body_as_an_empty_object(be, monkeypatch):
    # 4a1037c: some servers answer /health with 200 and no body at all, and
    # raising there reported "no server" for a process that was running fine.
    _stub_urlopen(monkeypatch, be, raw=b"")
    assert be._get("http://b", "/health") == {}
    _stub_urlopen(monkeypatch, be, raw=b"  \n")
    assert be._get("http://b", "/health") == {}


def test_get_still_raises_on_a_body_that_is_not_json(be, monkeypatch):
    _stub_urlopen(monkeypatch, be, raw=b"<html>gateway</html>")
    with pytest.raises(ValueError):
        be._get("http://b", "/health")


def test_n_ctx_reads_props_and_is_none_without_them(be, monkeypatch):
    _stub_urlopen(monkeypatch, be, raw=b'{"default_generation_settings": {"n_ctx": 8192}}')
    assert be.n_ctx("http://b") == 8192
    _stub_urlopen(monkeypatch, be, exc=urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b"")))
    assert be.n_ctx("http://b") is None


# --- the request itself ---------------------------------------------------
# What chat() sends decides whether the delta method holds on a given server,
# and what it returns is the only thing the pre-flight can check.

def _completion(prompt_tokens=469, completion_tokens=1, content="1, 2, 3, 4, 5.",
                model="qwen3-4b-npu"):
    return {"model": model,
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens}}


def _stub_post(be, body, wall=0.5):
    calls = []

    def fake(base, path, payload, timeout):
        calls.append({"base": base, "path": path, "payload": payload,
                      "timeout": timeout})
        return body, wall

    be.post_timed = fake
    return calls


def test_chat_sends_both_cap_spellings(be):
    # geniex serve ignores the legacy `max_tokens` outright (README); one
    # spelling meant the cap was silently unhonoured on one of the two servers
    # this repo compares against.
    calls = _stub_post(be, _completion())
    be.chat("b", "m", "p", 7, 1)
    payload = calls[0]["payload"]
    assert payload["max_tokens"] == 7
    assert payload["max_completion_tokens"] == 7


def test_chat_switches_prompt_caching_off(be):
    # llama-server reuses a cached prefix by default; the second request of a
    # delta pair would then skip most of its prefill and the subtraction
    # would no longer cancel it.
    calls = _stub_post(be, _completion())
    be.chat("b", "m", "p", 7, 1)
    assert calls[0]["payload"]["cache_prompt"] is False


def test_chat_disables_thinking_in_the_two_spellings_it_claims(be):
    # The comment said "Three spellings" over a payload carrying two.
    calls = _stub_post(be, _completion())
    be.chat("b", "m", "p", 7, 1)
    payload = calls[0]["payload"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"
    assert "thinking" not in payload, "Anthropic's block belongs to /v1/messages"
    assert calls[0]["path"] == "/v1/chat/completions"


def test_chat_returns_the_servers_counts_wall_model_and_text(be):
    _stub_post(be, _completion(prompt_tokens=12, completion_tokens=24,
                               model="qwen3-8b-npu", content="one two"), wall=0.75)
    r = be.chat("b", "m", "p", 24, 1)
    assert r == {"prompt_tokens": 12, "completion_tokens": 24, "wall": 0.75,
                 "model": "qwen3-8b-npu", "content": "one two"}


def test_chat_counts_side_channel_reasoning_as_generated_text(be):
    body = _completion(content="")
    body["choices"][0]["message"]["reasoning_content"] = "hmm"
    _stub_post(be, body)
    assert be.chat("b", "m", "p", 24, 1)["content"] == "hmm"


def test_chat_reads_a_malformed_completion_as_no_text_rather_than_raising(be):
    # The warmup is the pre-flight, so a server with an odd body shape has to
    # reach _warmup_check's "generates nothing" line, not a traceback.
    for choices in ([], [{"message": "not an object"}], [{}], "nonsense"):
        body = _completion()
        body["choices"] = choices
        _stub_post(be, body)
        assert be.chat("b", "m", "p", 24, 1)["content"] == ""


def test_chat_refuses_a_response_without_usage(be, capsys):
    # The counts here are the server's own. Without them the old code produced
    # `prompt=0 ... 0.0 tok/s` prefill rows and a decode SKIPPED blaming EOS.
    body = _completion()
    del body["usage"]
    _stub_post(be, body)
    assert be.chat("b", "m", "p", 1, 1) is None
    out = capsys.readouterr().out
    assert "no usage" in out and "skipping" in out


def test_chat_refuses_all_zero_usage(be, capsys):
    # GenieAPIService reports usage as all zeros (bench_servers.py exists
    # because of it): zeros dressed as counts are still no counts.
    _stub_post(be, _completion(prompt_tokens=0, completion_tokens=0))
    assert be.chat("b", "m", "p", 1, 1) is None
    assert "no usage" in capsys.readouterr().out


def test_chat_refuses_a_200_that_carries_an_error_object_as_no_usage(be, capsys):
    # A still-loading or unloaded model on a third-party OpenAI-compatible
    # endpoint -- what --base advertises support for -- answers 200 with an
    # error envelope and no usage at all. Refused like every other answer
    # without counts, because every rate here is formed from the server's own
    # numbers. The refusal is the pinned behaviour; what the reader is TOLD is
    # the server's own sentence appended to it, because the warmup then exits
    # on "the warmup request failed (reason above)" and the reason above used
    # to be "no usage" alone -- a line that reads like a bug in this tool.
    _stub_post(be, {"error": {"message": "model is still loading"}})
    assert be.chat("b", "m", "p", 1, 1) is None
    out = capsys.readouterr().out
    assert "no usage" in out and "prompt_tokens=None" in out
    assert "model is still loading" in out, "the server already said what was wrong"
    assert be._warmup_check(None, be.WARMUP_CAP) == "the warmup request failed (reason above)"


def test_chat_prints_a_non_dict_error_value_too(be, capsys):
    # `error` as a bare string is what some proxies in front of an OpenAI
    # endpoint send. `.get("message")` on it raised AttributeError out of a
    # print whose whole job is to keep a failure readable.
    _stub_post(be, {"error": "upstream connect error"})
    assert be.chat("b", "m", "p", 1, 1) is None
    assert "upstream connect error" in capsys.readouterr().out


def test_chat_says_nothing_extra_when_there_is_no_error_object(be, capsys):
    # The sentence is appended, never invented: a 200 that simply omits usage
    # gets the "no usage" line and no phantom quote from the server.
    body = _completion()
    del body["usage"]
    _stub_post(be, body)
    assert be.chat("b", "m", "p", 1, 1) is None
    assert "the server said" not in capsys.readouterr().out


def test_chat_returns_none_after_a_failed_request(be, capsys):
    be.post_timed = lambda *a, **k: (None, "HTTP 429 -- server busy")
    assert be.chat("b", "m", "p", 1, 1) is None
    assert "HTTP 429 -- server busy" in capsys.readouterr().out


# --- the warmup as pre-flight ---------------------------------------------
# The warmup result was discarded, and nothing later reads generated text: a
# server answering /health 200 and serving empty completions produced a full,
# legitimate-looking prefill table and a decode diagnosis blaming EOS.

def _warm(completion_tokens=9, content="1, 2, 3, 4, 5."):
    return {"prompt_tokens": 12, "completion_tokens": completion_tokens,
            "wall": 0.3, "model": "m", "content": content}


def test_warmup_check_accepts_a_server_that_generated_under_the_cap(be):
    assert be._warmup_check(_warm(), 24) is None


def test_warmup_check_refuses_an_ignored_cap(be):
    # GenieAPIService: asked for 16, returned 125, under both spellings.
    msg = be._warmup_check(_warm(completion_tokens=125, content="x" * 400), 16)
    assert msg and "ignored the 16-token cap" in msg and "125" in msg


def test_warmup_check_refuses_an_empty_completion(be):
    msg = be._warmup_check(_warm(completion_tokens=1, content=""), 24)
    assert msg and "generates nothing" in msg


def test_warmup_check_refuses_whitespace_as_content(be):
    msg = be._warmup_check(_warm(completion_tokens=3, content="\n\n"), 24)
    assert msg and "generates nothing" in msg


def test_warmup_check_names_a_failed_request(be):
    msg = be._warmup_check(None, 24)
    assert msg and "failed" in msg


def test_served_model_note_is_silent_when_the_ids_agree(be):
    assert be._served_model_note("qwen3-4b-npu", "qwen3-4b-npu") is None


def test_served_model_note_warns_on_a_mismatch(be):
    # run-genie-server.ps1 -Model qwen3-8b serves qwen3-8b-npu on the same
    # port and n_ctx as the 4B; under the default --model the run was filed
    # as qwen3-4b-npu with nothing in the output to say otherwise.
    msg = be._served_model_note("qwen3-4b-npu", "qwen3-8b-npu")
    assert "WARNING" in msg and "qwen3-4b-npu" in msg and "qwen3-8b-npu" in msg
    assert "SERVED" in msg


def test_served_model_note_says_when_the_server_gave_no_id(be):
    msg = be._served_model_note("qwen3-4b-npu", None)
    assert "on trust" in msg and "qwen3-4b-npu" in msg


def test_help_describes_the_env_knob(be):
    # The knob keeps the server's GENIE_ prefix (the docs list every knob in
    # one table), so the tool's own --help has to say it reads it.
    text = be._parser().format_help()
    assert "GENIE_MIN_DECODE_STEPS" in text


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


# --- the probe-vs-decode check after a full run ---------------------------
# Same warning, other end of the run: once real decode rates exist the probe
# that corrected every prefill row can be checked against them. It compared
# against the median of ALL samples pooled across depths, so a deep sweep on an
# engine whose decode genuinely falls with depth fired it for a real effect.

def test_probe_vs_decode_compares_at_the_probes_depth_not_the_pool(be):
    # docs/GENIE_SERVER.md figures: ~18 t/s at 250, 11.5-13 at 3300, 8.1 at
    # 6000. The pooled median is ~11.5 against an 18 probe -- 1.58x -- and
    # the old check called the probe unrepresentative for a genuine depth
    # effect the verdict line then printed the evidence for.
    per_depth = [(250, [18.2, 18.0, 18.4]), (3300, [11.5, 11.3, 11.8]),
                 (6000, [8.1, 8.0, 8.3])]
    line = be._probe_vs_decode(18.0, per_depth)
    assert line and "WARNING" not in line
    assert "depth 250" in line, "must name the depth it compared at"


def test_probe_vs_decode_still_catches_a_probe_taken_during_a_blip(be):
    # 2026-08-28: probe 9.25 t/s against a true 17.7 -- the one moment this
    # check exists for.
    line = be._probe_vs_decode(9.25, [(250, [17.7, 17.6, 17.8]), (3300, [12.0])])
    assert "WARNING" in line and "9.25" in line and "17.7" in line
    assert "re-run" in line, "must say what to do about it"


def test_probe_vs_decode_tolerance_matches_the_crosscheck(be):
    # 1.5x, not 2: a real 1.94x understatement slipped through 2.0.
    assert "WARNING" in be._probe_vs_decode(9.25, [(250, [17.7])])
    assert "WARNING" not in be._probe_vs_decode(15.0, [(250, [17.7])])


def test_probe_vs_decode_is_silent_without_a_probe_or_samples(be):
    assert be._probe_vs_decode(None, [(250, [18.0])]) is None
    assert be._probe_vs_decode(18.0, [(250, []), (3300, [])]) is None


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


# --- the per-depth table --------------------------------------------------
# Same pooling, for the table --decode-every prints. It diffed consecutive rows
# in MEASURED order, so in the interleaved mode every alternation was a ~30%
# swing and every line carried "<-- step".

def _rows(lines):
    return [ln for ln in lines if ln.strip().startswith("depth")]


def test_per_depth_table_pools_repeats_before_diffing(be):
    # docs/GENIE_SERVER.md's own interleaved run: 18.9/18.5 at 250 against
    # 12.8/13.3 at 3300 diffed in order as -32%, +44%, -28% -- three steps
    # for one depth effect.
    per_depth = [(250, [18.9]), (3300, [12.8]), (250, [18.5]), (3300, [13.3])]
    rows = _rows(be._per_depth_table(per_depth))
    assert len(rows) == 2, "one row per depth, not per group"
    assert "<-- step" not in rows[0]
    assert "<-- step" in rows[1] and "(n=2)" in rows[1]


def test_per_depth_table_marks_a_step_and_leaves_a_plateau_alone(be):
    rows = _rows(be._per_depth_table([(500, [18.0]), (1500, [17.8]), (3000, [12.0])]))
    assert "<-- step" not in rows[1], "1% is a plateau"
    assert "<-- step" in rows[2]


def test_per_depth_table_sorts_by_depth_not_measurement_order(be):
    rows = _rows(be._per_depth_table([(3300, [13.0]), (250, [18.7])]))
    assert rows[0].split()[1] == "250" and rows[1].split()[1] == "3300"


def test_per_depth_table_needs_two_depths(be):
    assert be._per_depth_table([(250, [18.0, 18.2])]) == []
    assert be._per_depth_table([(250, []), (3300, [])]) == []


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


def test_delta_run_keeps_the_window_when_the_delta_time_is_negative(be):
    # The long run coming back FASTER than the short one is queueing noise (or
    # a cached prefix), not a measurement; 64 / -0.06 would print -1066 t/s.
    # It used to be collapsed into steps=0, which measure_decode then reported
    # as the model stopping early -- the one diagnosis that is false for a
    # full 64-step window. The window is real; only its timing is not.
    _stub_chat_seq(be, _run(1, 3.00), _run(65, 2.94))
    _, _, steps, secs = be._delta_run("b", "m", "p", 64, 1)
    assert steps == 64
    assert secs == pytest.approx(-0.06)


def test_a_negative_delta_is_named_as_such_not_blamed_on_eos(be, capsys):
    _stub_chat_seq(be, _run(1, 3.00), _run(65, 2.94))
    assert be.measure_decode("b", "m", 250, 64, 1) is None
    out = capsys.readouterr().out
    assert "non-positive delta" in out and "64 steps" in out
    assert "stopped before the cap" not in out
    assert "-1066" not in out


def test_an_early_stop_says_the_model_may_have_produced_nothing(be, capsys):
    # The empty-completion server hits this path too, and "stopped before the
    # cap" alone blamed EOS for a server that generated nothing at all.
    _stub_chat_seq(be, _run(1, 3.00), _run(1, 3.05))
    assert be.measure_decode("b", "m", 250, 64, 1) is None
    assert "stopped before the cap or produced nothing" in capsys.readouterr().out


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


def test_the_step_floor_defaults_to_sixteen(be):
    # The fixture scrubs GENIE_MIN_DECODE_STEPS, so this is the shipped
    # default and not whatever the developer's shell exported.
    assert be.MIN_DECODE_STEPS == 16


def test_the_step_floor_is_tunable_through_the_documented_env_var(monkeypatch):
    # A caller deliberately measuring short generations needs a way down; the
    # default protects the common case rather than forbidding the rare one.
    # Through the env var docs/GENIE_SERVER.md documents, not the module
    # attribute: patching the attribute left the documented pathway exercised
    # by nothing, so a rename of the variable passed the suite.
    monkeypatch.setenv("GENIE_MIN_DECODE_STEPS", "2")
    be = _load()
    be.box_state = lambda: (None, None, None, None)
    _stub_chat_seq(be, _run(1, 0.30), _run(5, 6.98))
    assert be.measure_decode("b", "m", 250, 120, 1) == pytest.approx(4 / 6.68,
                                                                     rel=1e-6)


def test_a_typo_in_the_step_floor_falls_back_with_a_line_naming_it(monkeypatch, capsys):
    # A bare int() here tracebacked at IMPORT with a nameless ValueError --
    # and because bench_contention imports this module at scope, the same
    # typo killed that tool too.
    monkeypatch.setenv("GENIE_MIN_DECODE_STEPS", "sixteen")
    be = _load()
    assert be.MIN_DECODE_STEPS == 16
    out = capsys.readouterr().out
    assert "GENIE_MIN_DECODE_STEPS" in out and "sixteen" in out and "16" in out


def test_decode_probe_returns_seconds_per_step(be):
    _stub_chat_seq(be, _run(1, 0.30), _run(17, 1.18))
    per_step = be.decode_probe("b", "m", 1, depth=500, steps=16)
    assert per_step == pytest.approx(0.055, rel=1e-6)
    assert per_step < 1, "seconds per step, not the 18.2 t/s reciprocal"


def test_decode_probe_asks_for_the_floor_decode_enforces(be):
    # It asked for 8 -- half the floor measure_decode applies -- and accepted
    # any window above zero, so the one figure subtracted from every prefill
    # row could rest on a 4-step window the decode phase would have refused.
    calls = _stub_chat_seq(be, _run(1, 0.30), _run(17, 1.18))
    be.decode_probe("b", "m", 1)
    assert [c["max_tokens"] for c in calls] == [1, 1 + be.MIN_DECODE_STEPS]
    assert be.MIN_DECODE_STEPS >= 16


def test_decode_probe_refuses_a_window_under_the_floor(be, capsys):
    # The 5-token answer the floor's own comment records: 4 steps, accepted
    # silently, subtracted from every prefill row for a ~10% inflation at
    # depth.
    _stub_chat_seq(be, _run(1, 0.30), _run(5, 6.98))
    assert be.decode_probe("b", "m", 1) is None
    out = capsys.readouterr().out
    assert "REFUSED" in out and "4-step" in out
    assert "RAW" in out, "must say what happens to the prefill figures"


def test_decode_probe_prints_its_step_count_beside_the_figure(be, capsys):
    _stub_chat_seq(be, _run(1, 0.30), _run(17, 1.18))
    be.decode_probe("b", "m", 1)
    out = capsys.readouterr().out
    assert "16 steps" in out and "0.055 s/token" in out


def test_decode_probe_refuses_a_non_positive_delta(be, capsys):
    _stub_chat_seq(be, _run(1, 0.30), _run(17, 0.29))
    assert be.decode_probe("b", "m", 1) is None
    assert "non-positive delta" in capsys.readouterr().out


def test_decode_probe_returns_none_when_the_run_failed(be):
    be._delta_run = lambda *a, **k: None
    assert be.decode_probe("b", "m", 1) is None


def test_decode_probe_returns_none_when_there_was_no_decode_window(be):
    # steps==0 is the early-EOS case above. Dividing by it raises, and any
    # number returned here becomes a subtraction against every prefill row.
    be._delta_run = lambda *a, **k: ({"prompt_tokens": 469},
                                     {"prompt_tokens": 469}, 0, 0.0)
    assert be.decode_probe("b", "m", 1) is None


def test_decode_probe_value_is_in_the_units_prefill_subtracts(be):
    # The two halves have to agree on orientation: hand measure_prefill the
    # reciprocal (18.2) and PROBE_MAX_SHARE refuses it, so every prefill figure
    # silently reverts to raw and reads low.
    _stub_chat_seq(be, _run(1, 0.30), _run(17, 1.18))
    per_step = be.decode_probe("b", "m", 1, depth=500, steps=16)
    _stub_chat(be, wall=0.40)
    rate = be.measure_prefill("b", "m", 469, 1, per_step=per_step)
    assert rate == pytest.approx(469 / (0.40 - 0.055), rel=1e-6)


# --- box state recorded alongside the numbers -----------------------------
# This module produced every published prefill and decode figure while
# recording no power or clock state at all. Its docstring told the OPERATOR to
# note the conditions, which is the same as not recording them.

def test_measurements_do_not_shell_out(be, monkeypatch):
    # The docstring promises device-free; an accepted sample calls the sampler,
    # and the sampler launches PowerShell unless the fixture has stubbed it.
    # Recorded rather than raised: the sampler swallows every exception by
    # design, so a raising sentinel would be caught and prove nothing.
    launched = []

    def spy(*a, **k):
        launched.append(a)
        return types.SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(subprocess, "run", spy)
    _stub_chat(be, wall=0.40)
    assert be.measure_prefill("b", "m", 469, 1, per_step=0.055)
    _stub_chat_seq(be, _run(1, 0.28), _run(105, 6.17))
    assert be.measure_decode("b", "m", 250, 120, 1)
    assert launched == [], "a measurement launched a subprocess"


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


def test_running_on_battery_for_part_of_the_run_is_called_out(be):
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.extend([
        {"label": "a", "on_ac": True, "charge_pct": 80.0, "charge_w": 0.0,
         "clock_pct": 99.0},
        {"label": "b", "on_ac": False, "charge_pct": 79.0, "charge_w": 0.0,
         "clock_pct": 60.0}])
    text = "\n".join(be.box_state_summary())
    assert "ON BATTERY for part of the run" in text


def test_running_on_battery_for_the_whole_run_says_so(be):
    # "for part of the run" claims some sample saw AC; an all-battery run got
    # the same line, which understates it.
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.extend([
        {"label": "a", "on_ac": False, "charge_pct": 60.0, "charge_w": -10.0,
         "clock_pct": 99.0},
        {"label": "b", "on_ac": False, "charge_pct": 58.0, "charge_w": -10.0,
         "clock_pct": 98.0}])
    text = "\n".join(be.box_state_summary())
    assert "ON BATTERY for the whole run" in text
    assert "part of the run" not in text


def test_the_clock_warning_says_it_brackets_rather_than_covers(be):
    # Sampled BETWEEN measurements, so it cannot describe what happened during
    # one. Claiming otherwise is the over-read this repo keeps catching.
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.append({"label": "a", "on_ac": True, "charge_pct": 90.0,
                           "charge_w": 0.0, "clock_pct": 58.0})
    text = "\n".join(be.box_state_summary())
    assert "58%" in text and "brackets" in text


def test_a_box_with_no_battery_class_still_reports_its_clock(be):
    # The desktop / mini PC / VM shape of the same sampler: Win32_Battery
    # returns nothing, so the pack is never read, while Get-Counter answers.
    # The clock is the half of the reading that is still trustworthy there,
    # and its sub-80% warning with it.
    be.BOX_SAMPLES.clear()
    be.BOX_SAMPLES.extend([
        {"label": "a", "on_ac": None, "charge_pct": None, "charge_w": None,
         "clock_pct": 74.0},
        {"label": "b", "on_ac": None, "charge_pct": None, "charge_w": None,
         "clock_pct": 96.0}])
    text = "\n".join(be.box_state_summary())
    assert "clock: 74-96% of base across 2 sample(s)" in text
    assert "WARNING: clock reached 74%" in text and "brackets" in text
    assert "pack" not in text and "draw" not in text


def test_an_unreadable_box_records_nothing_rather_than_zeros(be, monkeypatch):
    # box_state returns Nones off-Windows and on any query failure. Recording
    # a row of Nones would put a fake sample in the artifact.
    be.BOX_SAMPLES.clear()
    monkeypatch.setattr(be, "box_state", lambda: (None, None, None, None))
    be.note_box_state("x")
    assert be.BOX_SAMPLES == []


# --- the sampler's parse --------------------------------------------------
# `[math]::Round($null, 1)` is 0, so a failed Get-Counter read used to print as
# "clock 0.0% of base" and fire the low-clock WARNING on a healthy box: a
# fabricated sample the summary cannot tell from a real one. The PowerShell
# half cannot run device-free, so its null guards are pinned by text and the
# Python half is exercised on the row they produce.

def test_the_sampler_script_prints_an_unread_field_as_empty(be):
    assert "$null -eq $k" in be._STATE_PS, "clock: empty when the counter failed"
    assert "$null -eq $b" in be._STATE_PS, "draw: empty when there is no battery class"


def test_the_sampler_script_formats_its_doubles_in_the_invariant_culture(be):
    # `-f` formats a double in the session's CURRENT culture, so on a
    # comma-decimal Windows the clock printed as "79,2": five comma fields,
    # which box_state_fields reads as a failed query. Every consumer went blind
    # together -- bench_contention's on-battery abort among them, which was not
    # culture-proof at HEAD either: HEAD's gate read the clock with the same
    # bare CookedValue and returned at `if pct is None: return None` before it
    # ever asked PowerOnline (HEAD src/bench_contention.py:226-228 and
    # :413-414). Both of its reads are now invariant-formatted too
    # (bench_contention._CLOCK_PS / _BUSY_PS) and the abort fires under
    # CurrentCulture='de-DE'. Measured with
    # CurrentCulture='de-DE': "True,100,0,64,1" before, "True,100,0,63.5"
    # after. Pinned by text for the reason above; what is pinned is that EVERY
    # rounded value is made text invariantly before `-f` can format it.
    invariant = ".ToString([cultureinfo]::InvariantCulture)"
    _head, *rounded = be._STATE_PS.split("[math]::Round(")
    assert len(rounded) == 2, "the draw and the clock"
    for tail in rounded:
        call, _, after = tail.partition(")")
        assert after.startswith(invariant), "Round(%s) is left to -f" % call


def test_parse_box_state_reads_an_empty_field_as_unread_not_zero(be):
    assert be._parse_box_state(["False", "47", "", ""]) == (False, 47.0, None, None)


def test_parse_box_state_reads_a_full_row(be):
    assert be._parse_box_state(["True", "62", "30.2", "97.5"]) == (True, 62.0, 30.2, 97.5)


def test_parse_box_state_reads_a_failed_query_as_all_none(be):
    assert be._parse_box_state(None) == (None, None, None, None)
    assert be._parse_box_state(["", "", "", ""]) == (None, None, None, None)
    assert be._parse_box_state(["garbage"]) == (None, None, None, None)


def test_a_failed_counter_read_does_not_fire_the_clock_warning(be):
    # End to end through the parse: the row a failed Get-Counter now produces
    # records the pack and says nothing about the clock.
    be.box_state = lambda: be._parse_box_state(["False", "47", "", ""])
    be.BOX_SAMPLES.clear()
    be.note_box_state("x")
    text = "\n".join(be.box_state_summary())
    assert "pack 47-47%" in text
    assert "clock" not in text and "WARNING" not in text


def test_box_state_fields_launches_one_powershell_and_splits_its_row(be, monkeypatch):
    launched = []

    def fake_run(cmd, **kw):
        launched.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="True,62,30.2,97.5\r\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(be, "sys", types.SimpleNamespace(platform="win32"))
    assert be.box_state_fields() == ["True", "62", "30.2", "97.5"]
    assert len(launched) == 1 and launched[0][0] == "powershell.exe"
    assert be._STATE_PS in launched[0]


def test_box_state_fields_is_none_when_the_query_fails(be, monkeypatch):
    monkeypatch.setattr(be, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=""))
    assert be.box_state_fields() is None

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("powershell.exe", 30)

    monkeypatch.setattr(subprocess, "run", boom)
    assert be.box_state_fields() is None
    monkeypatch.setattr(be, "sys", types.SimpleNamespace(platform="linux"))
    assert be.box_state_fields() is None


def test_power_reading_keeps_no_battery_apart_from_a_failed_query(be):
    # bench_contention's gate needs exactly this distinction: a transient
    # PowerShell failure must not read as a desktop with no battery. The parsed
    # bool cannot carry it (both are None there), which is why the first
    # element is the RAW text.
    be.box_state_fields = lambda: None
    assert be.power_reading() == (None, None, None)
    be.box_state_fields = lambda: ["", "", "", ""]
    assert be.power_reading() == ("", None, None)
    be.box_state_fields = lambda: ["False", "47", "-10.0", "99.0"]
    assert be.power_reading() == ("False", 47.0, -10.0)


def test_power_reading_is_one_launch_for_the_source_and_the_pack(be, monkeypatch):
    # The point of the function: bench_contention's gate used to take the
    # source and the pack as two calls, each a full launch of the four-field
    # script, the first discarding what the second went back for.
    launched = []

    def fake_run(cmd, **kw):
        launched.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="True,62,30.2,97.5\r\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(be, "sys", types.SimpleNamespace(platform="win32"))
    assert be.power_reading() == ("True", 62.0, 30.2)
    assert len(launched) == 1


# --- the closing box-state line -------------------------------------------
# "(box state could not be sampled ...)" fired whenever nothing was recorded,
# including a run where every point was skipped or refused -- blaming the
# sampler for a run that never called it.

def test_trailer_prints_the_summary_when_there_is_one(be):
    lines = be._box_state_trailer(["  box: pack 60-62%"], accepted=3, win32=True)
    assert "BOX STATE" in lines[1] and lines[-1] == "  box: pack 60-62%"


def test_trailer_does_not_blame_the_sampler_for_a_run_with_nothing_accepted(be):
    text = "\n".join(be._box_state_trailer([], accepted=0, win32=True))
    assert "no measurements were accepted" in text
    assert "could not be sampled" not in text


def test_trailer_names_a_sampler_failure_only_when_there_was_something_to_sample(be):
    text = "\n".join(be._box_state_trailer([], accepted=2, win32=True))
    assert "could not be sampled" in text
    assert be._box_state_trailer([], accepted=2, win32=False) == []


# --- main(), wired end to end against a stubbed server --------------------
# Every helper above is pinned on its own, and none of that says main() CALLS
# them: the warmup result was once computed and discarded, and the header
# printed the flag while the served id sat unread in the same response. These
# run main() over stubs so the wiring has something watching it too.

def _stub_server(be, monkeypatch, argv, served="qwen3-4b-npu", content="1, 2, 3, 4, 5.",
                 fail_after_warmup=False, ctx=8192):
    calls = []

    def fake_chat(base, model, prompt, max_tokens, timeout):
        calls.append(max_tokens)
        if fail_after_warmup and len(calls) > 1:
            return None
        return {"prompt_tokens": max(1, len(prompt) // be.CHARS_PER_TOKEN),
                "completion_tokens": max_tokens,
                "wall": 0.30 + 0.055 * max_tokens,
                "model": served, "content": content}

    be.chat = fake_chat
    be._get = lambda base, path, timeout=15: {}
    be.n_ctx = lambda base: ctx
    monkeypatch.setattr(be.sys, "argv", ["bench_endpoint.py", *argv])
    return calls


def test_main_prints_the_served_model_beside_the_flag(be, monkeypatch, capsys):
    # run-genie-server.ps1 -Model qwen3-8b on the default port, benched under
    # the default --model: the header is what gets copied into the docs.
    _stub_server(be, monkeypatch, ["--prefill-only", "--depths", "250"],
                 served="qwen3-8b-npu")
    be.main()
    out = capsys.readouterr().out
    assert "model=qwen3-4b-npu (server reports qwen3-8b-npu)" in out
    assert "WARNING: --model qwen3-4b-npu" in out
    assert out.index("server reports") < out.index("PREFILL")


def test_main_header_is_quiet_when_the_served_id_matches(be, monkeypatch, capsys):
    _stub_server(be, monkeypatch, ["--prefill-only", "--depths", "250"])
    be.main()
    out = capsys.readouterr().out
    assert "(server reports qwen3-4b-npu)" in out and "WARNING" not in out


def test_main_refuses_a_server_whose_warmup_generated_nothing(be, monkeypatch, capsys):
    # The health-200 / empty-completion server: before the warmup was read it
    # got a full prefill table. Now it gets no table at all.
    calls = _stub_server(be, monkeypatch, ["--prefill-only"], content="")
    with pytest.raises(SystemExit) as e:
        be.main()
    assert "refusing to benchmark" in str(e.value)
    assert "generates nothing" in str(e.value)
    assert calls == [be.WARMUP_CAP], "nothing may be measured after a failed pre-flight"
    assert "PREFILL" not in capsys.readouterr().out


def test_main_names_the_default_depths_it_dropped(be, monkeypatch, capsys):
    # The 8192 bundle with the default --tokens 200: 12000 is past the budget
    # and used to vanish from the sweep with no line.
    _stub_server(be, monkeypatch, ["--prefill-only"], ctx=8192)
    be.main()
    assert "dropped default depth(s) 12000" in capsys.readouterr().out


def test_main_reports_an_unhealthy_server_as_up(be, monkeypatch):
    _stub_server(be, monkeypatch, [])

    def unhealthy(base, path, timeout=15):
        raise urllib.error.HTTPError(
            "u", 503, "Service Unavailable", {},
            io.BytesIO(b'{"state": "wedged", "detail": "engine wedged"}'))

    be._get = unhealthy
    with pytest.raises(SystemExit) as e:
        be.main()
    assert "up but not ready" in str(e.value) and "wedged" in str(e.value)
    assert "nothing listening" not in str(e.value)


def test_main_reports_a_silent_listener_as_a_listener(be, monkeypatch):
    # Through the real _get: the connection is made and the read times out,
    # which urlopen raises bare. "Start one" here is advice to launch a second
    # server behind the one that just took the connection.
    real_get = be._get
    calls = _stub_server(be, monkeypatch, [])
    be._get = real_get
    _stub_urlopen(monkeypatch, be, exc=TimeoutError("timed out"))
    with pytest.raises(SystemExit) as e:
        be.main()
    assert "accepted the connection" in str(e.value)
    assert "nothing listening" not in str(e.value)
    assert calls == [], "nothing is measured behind a failed pre-flight"


def test_main_does_not_blame_the_sampler_when_every_point_failed(be, monkeypatch, capsys):
    # Every request after the warmup fails (a server that starts shedding):
    # nothing was accepted, so nothing was sampled, and the closing line must
    # say that rather than "box state could not be sampled".
    _stub_server(be, monkeypatch, ["--depths", "250"], fail_after_warmup=True)
    be.main()
    out = capsys.readouterr().out
    assert "no measurements were accepted" in out
    assert "could not be sampled" not in out


# --- the decode groups main() assembles -----------------------------------
# pool_by_depth, _per_depth_table, _probe_vs_decode and _verdict are each
# pinned above against a hand-built per_depth literal, and none of that says
# main() BUILDS that literal out of the right depths. It cannot be caught in
# the printed text: a deep group measured at the shallow depth prints the same
# median, the same n and the same "flat, so cost tracks the COMPILED window"
# headline -- this repo's central finding -- so the depths REQUESTED are the
# only witness there is.

def _depths_asked(be):
    """Every depth the run asks a completion at, in order.

    prompt_of() pads `depth` tokens of filler and appends one instruction, so
    the filler is the depth main() chose. The warmup does not go through
    prompt_of and is not counted.
    """
    inner = be.chat
    tail = be.prompt_of(1)[be.CHARS_PER_TOKEN:]   # the instruction, not copied
    asked = []

    def spy(base, model, prompt, max_tokens, timeout):
        if prompt.endswith(tail):
            asked.append((len(prompt) - len(tail)) // be.CHARS_PER_TOKEN)
        return inner(base, model, prompt, max_tokens, timeout)

    be.chat = spy
    return asked


def test_main_measures_the_deep_group_at_the_deepest_depth(be, monkeypatch, capsys):
    # --repeat at the shallowest depth, --repeat-deep at the deepest, two
    # requests per measurement. The deep sample IS the finding: measured at
    # 250 while labelled 3300 the run reads flat by construction.
    _stub_server(be, monkeypatch, ["--decode-only", "--depths", "250,3300",
                                   "--repeat", "2", "--repeat-deep", "1"])
    asked = _depths_asked(be)
    be.main()
    assert asked == [250, 250, 250, 250, 3300, 3300]
    out = capsys.readouterr().out
    assert "shallow median" in out and "deep median" in out


def test_main_measures_every_depth_of_an_interleaved_sweep_under_decode_every(
        be, monkeypatch, capsys):
    # The mode the module docstring recommends -- 250,3300,250,3300 with
    # --decode-every, so drift at one depth cannot masquerade as a depth
    # effect. Every group is measured --repeat times; --repeat-deep is not
    # consulted, which is why the verdict names --repeat there.
    _stub_server(be, monkeypatch,
                 ["--decode-only", "--depths", "250,3300,250,3300",
                  "--decode-every", "--repeat", "2", "--repeat-deep", "7"])
    asked = _depths_asked(be)
    be.main()
    assert asked == [250] * 4 + [3300] * 4 + [250] * 4 + [3300] * 4
    out = capsys.readouterr().out
    rows = [ln for ln in out.splitlines() if ln.strip().startswith("depth ")]
    assert len(rows) == 2, "one pooled row per depth, printed only in this mode"
    assert "(n=4)" in rows[0] and "(n=4)" in rows[1]


def test_main_prints_no_per_depth_table_when_only_two_groups_were_measured(
        be, monkeypatch, capsys):
    _stub_server(be, monkeypatch, ["--decode-only", "--depths", "250,3300",
                                   "--repeat", "1", "--repeat-deep", "1"])
    be.main()
    out = capsys.readouterr().out
    assert [ln for ln in out.splitlines() if ln.strip().startswith("depth ")] == []
    assert "per-depth medians" not in out


# --- the two refusals before any measurement is spent ---------------------
# bench_servers has test_a_non_integer_depth_is_a_sentence_not_a_traceback and
# test_tokens_that_leave_no_room_in_the_window_are_refused over its own
# verbatim copy of these guards; this module, where resolve_depths was lifted
# out of main() precisely so a bad --depths would stop being a traceback, had
# no counterpart at the call site.

def test_main_refuses_tokens_that_leave_no_room_in_the_window(be, monkeypatch, capsys):
    # Without the guard the budget goes negative, resolve_depths falls back to
    # the one depth that "fits", and the run prints a full PREFILL table taken
    # at depth -92064 -- not a cosmetic regression but published numbers.
    calls = _stub_server(be, monkeypatch, ["--tokens", "100000"], ctx=8192)
    with pytest.raises(SystemExit) as e:
        be.main()
    assert "--tokens 100000 leaves no room in an n_ctx=8192 window" in str(e.value)
    assert "lower it" in str(e.value)
    assert calls == [be.WARMUP_CAP], "nothing is measured at a negative depth"
    assert "PREFILL" not in capsys.readouterr().out


def test_main_turns_a_mistyped_depth_into_a_sentence_not_a_traceback(be, monkeypatch):
    calls = _stub_server(be, monkeypatch, ["--depths", "250,abc"])
    with pytest.raises(SystemExit) as e:
        be.main()
    assert "--depths wants comma-separated integers" in str(e.value)
    assert calls == [be.WARMUP_CAP]


def test_main_refuses_depths_that_all_exceed_the_budget_with_its_sentence(be, monkeypatch):
    calls = _stub_server(be, monkeypatch, ["--depths", "9000,12000"], ctx=8192)
    with pytest.raises(SystemExit) as e:
        be.main()
    assert "every requested depth exceeds the budget" in str(e.value)
    assert calls == [be.WARMUP_CAP]
