"""Request mapping, usage accounting, and the streaming contract.

Handlers are driven directly with a fake socket, so nothing here needs the NPU
or a bundle. That also makes the disconnect case testable at all -- staging a
real mid-generation client disconnect is far harder than simulating a write
that fails.
"""

import io
import json
import os
import socket

import pytest

from conftest import StubEngine, Wire

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]
CALL = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'


# --- request mapping -------------------------------------------------------

@pytest.mark.parametrize("req,expected", [
    ({"stop": "END"}, ["END"]),                    # OpenAI, string form
    ({"stop": ["A", "B"]}, ["A", "B"]),            # OpenAI, list form
    ({"stop_sequences": ["X"]}, ["X"]),            # Anthropic
    ({}, None),
    ({"stop": []}, None),                          # empty is "none", not ""
])
def test_stop_sequence_extraction(gs, req, expected):
    assert gs._stop_sequences(req) == expected


def test_tool_turns_default_to_greedy_sampling(gs):
    # Genie owns sampling and offers no grammar hook, so low temperature is the
    # only lever on tool-argument JSON validity.
    assert gs._sampler_params({}, tools_active=True) == {"temp": 0.0, "top-k": 1}


def test_explicit_temperature_beats_the_tool_default(gs):
    # The caller may know better than the default.
    assert gs._sampler_params({"temperature": 0.7}, tools_active=True) == {"temp": 0.7}


def test_sampler_params_absent_when_nothing_asked(gs):
    assert gs._sampler_params({}) is None


@pytest.mark.parametrize("req,expected", [
    ({}, False),                                             # server default: OFF
    ({"chat_template_kwargs": {"enable_thinking": False}}, False),   # Qwen
    ({"reasoning_effort": "none"}, False),                   # OpenAI
    ({"thinking": {"type": "disabled"}}, False),             # Anthropic
    ({"chat_template_kwargs": {"enable_thinking": True}}, True),
    ({"reasoning_effort": "high"}, True),
    ({"thinking": {"type": "enabled"}}, True),
])
def test_thinking_toggle_accepts_every_ecosystem_spelling(gs, req, expected):
    # Three ecosystems disagree; a client should not have to know which one
    # this server speaks. Each spelling has to work in BOTH directions -- a
    # request that asks FOR reasoning must get it now that the default is off,
    # which the old table never checked for two of the three spellings.
    assert gs._wants_thinking(req) is expected


def test_reasoning_is_suppressed_by_default(gs):
    """The default is OFF, and it is load-bearing rather than incidental.

    Measured on this box, the same prompt and the same correct tool call cost
    41s with the reasoning block and 2.4s without -- 10-17x on every agent step,
    with a length that swings run to run, so the old default was unpredictable
    as well as slow. This server exists to be driven by an agent and its own
    docs recommend suppressing it, so the recommended configuration is now the
    one you get without asking.
    """
    assert gs.THINKING_DEFAULT is False
    assert gs._wants_thinking({}) is False


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("yes", True),
    ("0", False), ("false", False), ("", False),
])
def test_the_server_default_is_still_one_env_var_away(gs, monkeypatch, value,
                                                      expected):
    # Flipping a default must not remove the choice. Faithfulness to the model
    # is a legitimate thing to want -- suppression is a prompt prefill, not a
    # filter over the output, so a caller who asks for reasoning gets exactly
    # what the model produced.
    import importlib
    monkeypatch.setenv("GENIE_THINKING", value)
    importlib.reload(gs)
    assert gs.THINKING_DEFAULT is expected
    assert gs._wants_thinking({}) is expected


def test_an_explicit_request_still_beats_the_server_default(gs, monkeypatch):
    # Both directions: the per-request field wins over whatever the server is
    # configured to do, or the three spellings would be decorative.
    import importlib
    monkeypatch.setenv("GENIE_THINKING", "1")
    importlib.reload(gs)
    assert gs._wants_thinking({"reasoning_effort": "none"}) is False
    monkeypatch.setenv("GENIE_THINKING", "0")
    importlib.reload(gs)
    assert gs._wants_thinking({"reasoning_effort": "high"}) is True


def test_suppression_renders_as_a_prefill_not_a_filter(gs):
    # Why flipping the default is not a loss of fidelity: the "off" path adds a
    # CLOSED, empty think block to the prompt so the model resumes after it. It
    # never strips anything the model produced -- that is GENIE_STRIP_THINK, a
    # separate knob that stays off.
    on = gs.TEMPLATE.build([{"role": "user", "content": "hi"}], thinking=True)
    off = gs.TEMPLATE.build([{"role": "user", "content": "hi"}], thinking=False)
    assert off == on + gs._NO_THINK
    assert gs._NO_THINK.startswith("<think>") and "</think>" in gs._NO_THINK
    assert gs.STRIP_THINK is False


@pytest.mark.parametrize("finish,calls,stop,expected", [
    ("stop", [{"name": "f"}], None, "tool_use"),
    ("length", None, None, "max_tokens"),
    ("stop", None, None, "end_turn"),
    ("stop", None, ["X"], "stop_sequence"),
    ("length", None, ["X"], "max_tokens"),      # token cap outranks stop seq
])
def test_anthropic_stop_reason(gs, finish, calls, stop, expected):
    # end_turn after a stop sequence tells the client the model finished on its
    # own when it was actually cut.
    assert gs._anthropic_stop_reason(finish, calls, stop) == expected


def test_anthropic_tool_schema_is_converted_to_openai_shape(gs):
    # Qwen3 was trained on OpenAI-style function schemas inside <tools>.
    out = gs._anthropic_tools([{"name": "f", "description": "d",
                                "input_schema": {"type": "object"}}])
    assert out[0]["function"]["parameters"] == {"type": "object"}


# --- usage accounting ------------------------------------------------------

def test_tool_turn_does_not_report_zero_completion_tokens(gs, handler):
    # Usage was computed from the post-parse remainder, so a turn whose whole
    # output was a tool call billed as free.
    gs.ENGINE = StubEngine(chunks=[CALL])
    h = handler()
    h._complete("prompt", 100, "cid", 0, tools_active=True)
    body = json.loads(h.wfile.text())
    assert body["usage"]["completion_tokens"] == len(CALL) // 4 > 0
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_overhead_absent_when_nothing_was_summarised(gs, handler):
    gs.ENGINE = StubEngine(chunks=["hi"])
    h = handler()
    h._complete("prompt", 100, "cid", 0)
    assert "genie_context_overhead_tokens" not in json.loads(h.wfile.text())["usage"]


def test_overhead_reported_when_summarisation_ran(gs, handler):
    gs.ENGINE = StubEngine(chunks=["hi"])
    h = handler()
    h._complete("prompt", 100, "cid", 0, overhead=33)
    assert json.loads(h.wfile.text())["usage"]["genie_context_overhead_tokens"] == 33


def test_stream_reports_no_usage_unless_asked(gs, handler):
    gs.ENGINE = StubEngine(chunks=["a", "b"])
    h = handler()
    h._stream("prompt", 100, "cid", 0)
    assert not [f for f in h.wfile.sse_frames() if f.get("usage")]
    assert "[DONE]" in h.wfile.text()


def test_stream_usage_frame_has_empty_choices(gs, handler):
    # OpenAI's shape for a usage-only chunk.
    gs.ENGINE = StubEngine(chunks=["a", "b"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=True)
    usage_frames = [f for f in h.wfile.sse_frames() if f.get("usage")]
    assert len(usage_frames) == 1
    assert usage_frames[0]["choices"] == []
    assert h.wfile.text().rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("strip,expected_chars", [
    (True, len(CALL)),                       # think excluded
    (False, len("<think>reasoning</think>") + len(CALL)),
])
def test_streamed_usage_strips_think_like_the_non_streaming_path(
        gs, handler, strip, expected_chars):
    # The same turn must not report different completion_tokens purely because
    # the client chose to stream.
    gs.STRIP_THINK = strip
    gs.ENGINE = StubEngine(chunks=["<think>reasoning</think>", CALL])
    h = handler()
    h._stream("prompt", 100, "cid", 0, tools_active=True, include_usage=True)
    usage = next(f["usage"] for f in h.wfile.sse_frames() if f.get("usage"))
    assert usage["completion_tokens"] == expected_chars // 4


def test_prompt_is_tokenized_once_per_request(gs):
    # _fit encodes the fitted prompt to check the budget and usage encodes the
    # identical string again -- each a native call holding the engine lock.
    gs.ENGINE = StubEngine()
    prompt, dropped, fits, overhead = gs.build_windowed(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}],
        max_tokens=64)
    before = gs.ENGINE.encodes
    gs._tok_count(prompt)
    gs._tok_count(prompt)
    assert gs.ENGINE.encodes == before


def test_token_cache_stays_bounded(gs):
    gs.ENGINE = StubEngine()
    for i in range(50):
        gs._tok_count("string number %d" % i)
    assert len(gs._TOK_CACHE) <= 9


# --- streaming tool calls --------------------------------------------------

def test_streamed_tool_call_is_emitted_whole(gs, handler):
    # A <tool_call> means nothing until it closes; streaming it token by token
    # would hand the client half a call to guess about.
    gs.ENGINE = StubEngine(chunks=["Let me look.", CALL])
    h = handler()
    h._stream("prompt", 100, "cid", 0, tools_active=True)
    frames = h.wfile.sse_frames()
    tool_frames = [f for f in frames
                   if f["choices"] and f["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_frames) == 1
    fn = tool_frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]
    assert fn["name"] == "read_file"
    assert json.loads(fn["arguments"]) == {"path": "a.py"}
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("tools_active", [True, False])
def test_disconnect_aborts_the_generation(gs, handler, tools_active):
    # On a single-flight NPU an abandoned request that runs to max_tokens
    # blocks every other caller. The buffered tools path emits nothing while
    # generating, so it needs a probe write to notice the client left at all.
    gs.ENGINE = StubEngine(chunks=["tok "] * 2000)
    h = handler(fail_after=2)
    h._stream("prompt", 2000, "cid", 0, tools_active=tools_active, include_usage=True)
    assert gs.ENGINE.aborted, "never signalled abort"
    assert gs.ENGINE.yielded < 200, "ran on for %d chunks" % gs.ENGINE.yielded


def test_anthropic_disconnect_aborts_a_buffered_tool_turn(gs, handler):
    gs.ENGINE = StubEngine(chunks=["tok "] * 2000)
    h = handler(fail_after=2)
    h._anthropic_stream("prompt", 2000, "m", "mid", tools_active=True)
    assert gs.ENGINE.aborted
    assert gs.ENGINE.yielded < 200


def test_anthropic_tool_use_block_shape(gs, handler):
    gs.ENGINE = StubEngine(chunks=[CALL])
    h = handler()
    h._anthropic_complete("prompt", 100, "model", "mid", tools_active=True)
    body = json.loads(h.wfile.text())
    assert body["stop_reason"] == "tool_use"
    block = next(b for b in body["content"] if b["type"] == "tool_use")
    assert block["name"] == "read_file" and block["input"] == {"path": "a.py"}
    assert block["id"].startswith("toolu_")


# --- port collision -------------------------------------------------------
# A second server on a port another process is already serving must not start.
# On Windows it silently could: HTTPServer sets allow_reuse_address, which
# there permits binding a LIVE socket rather than just a TIME_WAIT one, so both
# binds succeed and the OLD process keeps answering. That happened while
# benchmarking -- the new bundle loaded and logged a clean startup while every
# request was served by the previous bundle -- and it is the same class of bug
# as a silent CPU fallback: the measurement looks fine and describes the wrong
# thing.

def test_port_in_use_detects_a_live_listener(gs):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert gs.port_in_use(host, port) is True
    finally:
        srv.close()


def test_port_in_use_false_when_nothing_listens(gs):
    # Bind to grab a free port, then close it so the port is known-unused.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    host, port = s.getsockname()
    s.close()
    assert gs.port_in_use(host, port) is False


def test_server_does_not_share_a_live_port_on_windows(gs):
    # The guarantee is per-platform: on Windows SO_REUSEADDR is what allows the
    # hijack, so the class must not set it. On POSIX it stays on, because there
    # it only means "rebind TIME_WAIT" and turning it off makes restarts fail.
    assert gs.Server.allow_reuse_address == (os.name != "nt")


def test_port_in_use_probes_loopback_for_a_wildcard_host(gs):
    # GENIE_HOST=0.0.0.0 is the documented way to expose this server. A
    # wildcard address is not connectable, so checking it directly returned
    # False against a live listener -- exactly in the configuration where the
    # port is most likely to be contended.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    # Backlog > 1: each probe connects and is never accepted, so a backlog of
    # one is filled by the first check and the second gets refused -- which
    # looks exactly like the bug under test.
    srv.listen(8)
    _, port = srv.getsockname()
    try:
        assert gs.port_in_use("0.0.0.0", port) is True
        assert gs.port_in_use("", port) is True
    finally:
        srv.close()


# --- /props is the field a client actually plans against ------------------
# Getting it wrong is not cosmetic: typed sizes its per-turn token budget and
# its compaction threshold from n_ctx, so a wrong window means it never
# suggests /compact and overruns the model instead. This repo has already had
# /props report 4096 while the HTP had allocated 8192 (two servers, one port),
# and a whole table of measurements was filed against the wrong bundle.

def _get(gs, path):
    h = object.__new__(gs.Handler)
    h.path = path
    h.wfile = Wire()
    sent = {}
    h.send_response = lambda code: sent.setdefault("code", code)
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    h.do_GET()
    return sent.get("code"), json.loads(h.wfile.text())


def test_props_reports_the_window_the_bundle_was_compiled_with(gs):
    gs._CONTEXT_SIZE = 8192
    code, body = _get(gs, "/props")
    assert code == 200
    assert body["default_generation_settings"]["n_ctx"] == 8192


def test_props_omits_model_path(gs):
    # typed checks model_path FIRST, so emitting it would take precedence and
    # display the bundle directory -- disagreeing with the name /health and
    # /v1/models already report. One name everywhere beats a detailed one in
    # a single place.
    _code, body = _get(gs, "/props")
    assert "model_path" not in body


def test_props_claims_no_modality_it_does_not_have(gs):
    # Absence reads as text-only, which is the truth for this bundle.
    _code, body = _get(gs, "/props")
    assert "modality" not in body


def test_props_names_the_same_model_as_the_other_endpoints(gs):
    _code, props = _get(gs, "/props")
    _code, health = _get(gs, "/health")
    _code, models = _get(gs, "/v1/models")
    assert props["model_alias"] == health["model"] == models["data"][0]["id"]


def test_an_unknown_path_404s_rather_than_guessing(gs):
    code, body = _get(gs, "/v1/completions")
    assert code == 404
    assert body["error"]["type"] == "invalid_request_error"


# --- the stream's disconnect latch ----------------------------------------
# Once a write raises, every later write must be skipped. Re-raising on a dead
# socket is what handle_one_request exists to suppress, and the usage frame and
# [DONE] are written from a different code path than the token frames.

def test_a_dead_stream_stops_writing_entirely(gs, handler):
    """Counts write ATTEMPTS, not resulting frames.

    Asserting on sse_frames() cannot see this: a write to a dead Wire raises
    and produces no frame whether the latch suppressed it or it was attempted
    and failed. Both look identical in the output, so that assertion passes
    even with the latch removed -- a check whose negative result carries no
    information, which is the exact failure this suite keeps finding
    elsewhere. `writes` is the signal that CAN discriminate: once the client
    is gone the count must stop climbing.
    """
    gs.ENGINE = StubEngine(chunks=["tok"] * 50)
    h = handler(fail_after=1)
    h._stream("prompt", 100, "cid", 0, include_usage=True)
    # One successful write, one that raised and set the latch, nothing after.
    assert h.wfile.writes == 2, (
        "wrote %d times to a socket known to be gone -- the latch is not "
        "holding" % h.wfile.writes)


def test_a_healthy_stream_writes_every_frame(gs, handler):
    # The counterpart: the latch must not suppress on a LIVE connection.
    gs.ENGINE = StubEngine(chunks=["a", "b", "c"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=True)
    assert h.wfile.writes >= 7, "role + 3 tokens + finish + usage + [DONE]"


def test_a_dead_stream_emits_no_usage_frame(gs, handler):
    gs.ENGINE = StubEngine(chunks=["a", "b", "c"])
    h = handler(fail_after=1)
    h._stream("prompt", 100, "cid", 0, include_usage=True)
    assert not [f for f in h.wfile.sse_frames() if f.get("usage")], \
        "usage was written to a socket already known to be gone"


def test_a_dead_stream_emits_no_done_sentinel(gs, handler):
    gs.ENGINE = StubEngine(chunks=["a", "b", "c"])
    h = handler(fail_after=1)
    h._stream("prompt", 100, "cid", 0, include_usage=True)
    assert "[DONE]" not in h.wfile.text()


def test_a_live_stream_still_gets_both(gs, handler):
    # The latch must not be so eager that it fires on a healthy stream.
    gs.ENGINE = StubEngine(chunks=["a", "b"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=True)
    assert [f for f in h.wfile.sse_frames() if f.get("usage")]
    assert "[DONE]" in h.wfile.text()


def test_a_healthy_stream_that_was_not_asked_for_usage_gets_none(gs, handler):
    # Clients that did not opt in must see a byte-identical stream to before.
    gs.ENGINE = StubEngine(chunks=["a", "b"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=False)
    assert not [f for f in h.wfile.sse_frames() if f.get("usage")]
    assert "[DONE]" in h.wfile.text()


# --- the single-flight permit must always come back -----------------------
# A leaked permit is unrecoverable without a restart: the server answers 429
# forever while completely idle, which reads externally as "the NPU is busy"
# and would send the next investigator hunting a contention problem that does
# not exist.

def _post(gs, handler_factory, payload, path="/v1/chat/completions"):
    h = handler_factory()
    h.path = path
    body = json.dumps(payload).encode()
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    h.do_POST()
    return h


def test_the_inflight_permit_is_released_when_the_generator_raises(gs, handler):
    def boom(req):
        raise RuntimeError("engine exploded mid-turn")

    gs.Handler._openai_chat = lambda self, req: boom(req)
    before = gs._INFLIGHT._value
    with pytest.raises(RuntimeError):
        _post(gs, handler, {"messages": [{"role": "user", "content": "hi"}]})
    assert gs._INFLIGHT._value == before, \
        "a leaked permit means 429-forever on an idle server"


def test_the_permit_is_released_on_the_ordinary_path(gs, handler):
    gs.Handler._openai_chat = lambda self, req: None
    before = gs._INFLIGHT._value
    _post(gs, handler, {"messages": [{"role": "user", "content": "hi"}]})
    assert gs._INFLIGHT._value == before


def test_a_shed_request_does_not_release_a_permit_it_never_took(gs, handler):
    # Over-releasing a BoundedSemaphore raises ValueError and would take the
    # server down on the first burst of backpressure.
    gs.Handler._openai_chat = lambda self, req: None
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    h = _post(gs, handler, {"messages": [{"role": "user", "content": "hi"}]})
    assert json.loads(h.wfile.text())["error"]["type"] == "overloaded_error"
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.release()          # must not raise


def test_a_refused_tool_request_never_queues_behind_a_generation(gs, handler):
    # Refusing costs no NPU time, so it is checked BEFORE the lock.
    gs.TOOLS_OK = False
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    try:
        h = _post(gs, handler, {"messages": [{"role": "user", "content": "hi"}],
                                "tools": [{"type": "function"}]})
        body = json.loads(h.wfile.text())
        assert "tool calling is not supported" in body["error"]["message"], \
            "a full queue must not turn a 400 into a 429"
    finally:
        for _ in range(gs.MAX_INFLIGHT):
            gs._INFLIGHT.release()


# --- /props carries what a router cannot otherwise learn -------------------
# n_ctx alone is not enough to choose among an NPU, a GPU and a CPU endpoint,
# and on this engine it is actively misleading: it is the SOFTWARE cap, while
# throughput is set by the compiled window and by whether the bundle carries
# one graph or several. Two bundles reporting the same n_ctx differ 2-3x on
# short prompts, and nothing over HTTP could tell them apart.

def test_props_says_which_engine_is_answering(gs):
    _code, body = _get(gs, "/props")
    assert body["genie"]["engine"] == "npu-hexagon-htp"
    # The constraint a dispatcher has to encode, not discover from a 429.
    assert body["genie"]["single_flight"] is True


def test_props_exposes_the_compiled_graphs_not_just_the_window(gs):
    gs._CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
    _code, body = _get(gs, "/props")
    assert body["genie"]["context_lengths"] == [512, 1024, 2048, 4096]
    assert body["genie"]["multi_length"] is True


def test_props_marks_a_single_length_bundle_as_such(gs):
    # The 2-3x short-prompt difference a router would otherwise attribute to
    # the model or the depth.
    gs._CONTEXT_LENGTHS = [8192]
    _code, body = _get(gs, "/props")
    assert body["genie"]["multi_length"] is False


def test_props_reports_poll_because_it_decides_concurrency(gs):
    # poll:true turns NPU+GPU from a 1.45x gain into a 0.78x loss, so a client
    # deciding whether to run a second engine needs to see it.
    gs._POLL_MATCHES = [(True, "QnnHtp.poll")]
    _code, body = _get(gs, "/props")
    assert body["genie"]["poll"] is True


def test_props_does_not_invent_a_poll_value_it_could_not_read(gs):
    gs._POLL_MATCHES = []
    _code, body = _get(gs, "/props")
    assert body["genie"]["poll"] is None


def test_the_genie_block_is_additive_not_a_replacement(gs):
    # Namespaced so no llama.cpp-shaped field is misreported, and additive so a
    # client that ignores it sees exactly what it saw before.
    _code, body = _get(gs, "/props")
    assert body["default_generation_settings"]["n_ctx"] == gs.read_context_size()
    assert body["model_alias"] == gs.MODEL_ID
    assert "model_path" not in body and "modality" not in body
