"""Request mapping, usage accounting, and the streaming contract.

Handlers are driven directly with a fake socket, so nothing here needs the NPU
or a bundle. That also makes the disconnect case testable at all -- staging a
real mid-generation client disconnect is far harder than simulating a write
that fails.
"""

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
    ({}, True),                                              # server default
    ({"chat_template_kwargs": {"enable_thinking": False}}, False),   # Qwen
    ({"reasoning_effort": "none"}, False),                   # OpenAI
    ({"thinking": {"type": "disabled"}}, False),             # Anthropic
    ({"reasoning_effort": "high"}, True),
])
def test_thinking_toggle_accepts_every_ecosystem_spelling(gs, req, expected):
    # Three ecosystems disagree; a client should not have to know which one
    # this server speaks.
    assert gs._wants_thinking(req) is expected


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
    usage = [f["usage"] for f in h.wfile.sse_frames() if f.get("usage")][0]
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
    block = [b for b in body["content"] if b["type"] == "tool_use"][0]
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
