"""Request mapping, usage accounting, and the streaming contract.

Handlers are driven directly with a fake socket, so nothing here needs the NPU
or a bundle. That also makes the disconnect case testable at all -- staging a
real mid-generation client disconnect is far harder than simulating a write
that fails. Whole requests go through conftest.request, the one helper that
sets path / headers / body on a socketless Handler and keeps the status code.

The last section is the exception: a real Server on a loopback port, for the
three behaviours that only exist on a real connection (the socket timeout, a
closed peer seen as a readable socket, a write that fails only once the reset
is back). Still device-free -- the stub engine is behind it.
"""

import http.client
import json
import os
import socket
import threading
import time

import pytest

from conftest import StubEngine, request

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]
CALL = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
# The same tool in Anthropic's spelling, for the /v1/messages door.
ANTHROPIC_TOOLS = [{"name": "read_file", "description": "Read a file",
                    "input_schema": TOOLS[0]["function"]["parameters"]}]
# A SECOND tool and a second call, so one generation can carry two of them --
# which is what this server's own tool prompt asks for ("You may call one or
# more functions"), what parse_tool_calls' finditer returns, and the shape
# whose ordinals every SDK accumulates streamed calls by.
LIST_DIR = [{"type": "function", "function": {
    "name": "list_dir", "description": "List a directory",
    "parameters": TOOLS[0]["function"]["parameters"]}}]
ANTHROPIC_LIST_DIR = [{"name": "list_dir", "description": "List a directory",
                       "input_schema": TOOLS[0]["function"]["parameters"]}]
CALL2 = '<tool_call>{"name": "list_dir", "arguments": {"path": "src"}}</tool_call>'
HI = [{"role": "user", "content": "hi"}]


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


@pytest.mark.parametrize("strip", [True, False])
@pytest.mark.parametrize("prefilled,chunks", [
    (False, ["<think>reasoning</think>", CALL]),      # a pair: GENIE_STRIP_THINK's
    (True, ["dup answer\n", "</think>\n", CALL]),     # an orphan: stripped always
])
def test_a_streamed_tool_turn_bills_the_raw_generation_like_every_other_path(
        gs, handler, strip, prefilled, chunks):
    """The same turn must not bill differently because the client streamed it.

    This used to be test_streamed_usage_strips_think_like_the_non_streaming_path
    and asserted the STRIPPED count -- parity with a _complete that had since
    stopped stripping before it counted. The OpenAI tool stream was the one
    usage site of five the raw-count fix missed, under a comment saying
    "exactly as _complete does", and this test defended the divergence under a
    name claiming the opposite. What it pins now is the parity itself, against
    the non-streaming answer to the same generation, in both strip settings and
    for both kinds of strip -- the orphan one runs by DEFAULT (thinking is off,
    so every request is prefilled), which is where the undercount was live.
    """
    gs.STRIP_THINK = strip
    raw = "".join(chunks)

    gs.ENGINE = StubEngine(chunks=chunks)
    h = handler()
    h._stream("prompt", 100, "cid", 0, tools_active=True, include_usage=True,
              prefilled=prefilled)
    streamed = next(f["usage"] for f in h.wfile.sse_frames() if f.get("usage"))

    gs.ENGINE = StubEngine(chunks=chunks)
    h2 = handler()
    h2._complete("prompt", 100, "cid", 0, tools_active=True, prefilled=prefilled)
    buffered = json.loads(h2.wfile.text())["usage"]

    assert streamed["completion_tokens"] == len(raw) // 4, \
        "billed %d for a %d-char generation" % (streamed["completion_tokens"], len(raw))
    assert streamed == buffered


def test_prompt_is_tokenized_once_per_request(gs):
    # _fit encodes the fitted prompt to check the budget and usage encodes the
    # identical string again -- each a native call holding the engine lock.
    gs.ENGINE = StubEngine()
    prompt, dropped, fits, overhead = gs.build_windowed(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}],
        max_tokens=64)
    before = gs.ENGINE.encodes
    # ONCE is the claim in the name, so it is asserted: sampling `before` after
    # the build and comparing against it proves only that the two calls below
    # were memoised -- a _fit that encoded the fitted prompt twice passed.
    assert before == 1, "build_windowed made %d tokenizer calls" % before
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


def _run_stream(h, api, **kw):
    """One stream on either API, with each handler's own positional spelling."""
    if api == "openai":
        h._stream("prompt", 2000, "cid", 0, include_usage=True, **kw)
    else:
        h._anthropic_stream("prompt", 2000, "m", "mid", **kw)


# Writes a stream makes BEFORE its query starts: OpenAI's role frame; on the
# Anthropic leg message_start, plus content_block_start and a ping when the
# stream is incremental. One write per frame on both (_Emitter.write).
def _preamble(api, tools_active):
    return 1 if api == "openai" or tools_active else 3


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("tools_active,prefilled,silent", [
    (True, True, True),       # buffered tool turn: nothing goes out at all
    (True, False, True),
    (False, True, True),      # THE PRODUCTION DEFAULT: the orphan gate holds
    (False, False, False),    # reasoning on: the gate is a pass-through
])
def test_disconnect_aborts_the_generation(gs, handler, api, tools_active,
                                          prefilled, silent):
    """An abandoned stream must stop generating, on every path that can be silent.

    On a single-flight NPU a request that runs to max_tokens with nobody
    reading blocks every other caller. A failed write is the only way a stream
    learns its client left, and two paths write NOTHING while they generate: a
    tool turn, buffered until the call closes, and an incremental stream whose
    orphan gate is holding -- which, since thinking is off by default, is every
    ordinary stream this server sends. Both depend on the keep-alive probe.

    This test used to run its plain leg with the default prefilled=False, where
    the gate passes every chunk straight through and the probe branch is never
    reached: deleting that branch left the whole suite green while a default
    stream ran all 2000 chunks into a closed socket. Its comment even said only
    the tools path needed the probe. The Anthropic twin had a different hole --
    its fail_after let the very first keep-alive be the failing write, so it
    passed whatever the cadence was.

    fail_after lets the preamble and ONE more write through, so on a silent
    path the i=0 probe succeeds and the i=8 probe is the write that fails: 9
    chunks, exactly. That pins the cadence too -- `i % 1000` yields 1001.
    """
    gs.ENGINE = StubEngine(chunks=["tok "] * 2000)
    h = handler(fail_after=_preamble(api, tools_active) + 1)
    _run_stream(h, api, tools_active=tools_active, prefilled=prefilled)
    assert gs.ENGINE.aborted, "never signalled abort"
    if silent:
        assert gs.ENGINE.yielded == 9, (
            "the probe runs every 8th chunk while nothing is being written; "
            "this ran for %d" % gs.ENGINE.yielded)
    else:
        # Every chunk is a frame here, so the write after the first one fails.
        assert gs.ENGINE.yielded == 2, gs.ENGINE.yielded


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("tools_active", [True, False])
def test_a_client_gone_before_the_query_starts_costs_nothing(gs, handler, api,
                                                             tools_active):
    # The engine's bare signal_abort() aims at the calling thread's turn, and
    # before query_stream there is none -- so the abort a failed first frame
    # sends lands on nothing. Without the early return the request went on to
    # wait for the lock, prefill the whole prompt and generate a token before
    # the loop's first failed write noticed.
    gs.ENGINE = StubEngine(chunks=["tok "] * 50)
    h = handler(fail_after=0)
    _run_stream(h, api, tools_active=tools_active, prefilled=True)
    assert gs.ENGINE.calls == [], "queried the engine for a client already gone"
    assert h.wfile.writes == 1, "kept writing to a socket known to be dead"


def test_anthropic_tool_use_block_shape(gs, handler):
    gs.ENGINE = StubEngine(chunks=[CALL])
    h = handler()
    h._anthropic_complete("prompt", 100, "model", "mid", tools_active=True)
    body = json.loads(h.wfile.text())
    assert body["stop_reason"] == "tool_use"
    block = next(b for b in body["content"] if b["type"] == "tool_use")
    assert block["name"] == "read_file" and block["input"] == {"path": "a.py"}
    assert block["id"].startswith("toolu_")


# --- the bytes on the wire ---------------------------------------------------
# Everything above reads a stream through conftest.Wire.sse_frames, which
# splitlines() and keeps the `data:` lines. That is exactly the wrong shape for
# four contracts a real client dispatches on -- the frame terminator, the
# `object` discriminator, and the ordinals on parallel tool calls and Anthropic
# content blocks -- so each of them could be broken outright with the whole
# suite green. These read the emitted bytes, or the ordinals, directly.

def _dispatched(raw):
    """The frames a conforming SSE client would dispatch: split on a BLANK line.

    Not splitlines: an SSE event is dispatched by an empty line, and a parser
    that keeps `data:` lines regardless (the suite's own Wire.sse_frames, and
    src/genie_smoke.py's reader) cannot tell a stream of frames from one frame
    that never ends.
    """
    assert raw.endswith(b"\n\n"), "the last frame was never terminated"
    return raw[:-2].split(b"\n\n")


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_every_sse_frame_is_terminated_by_a_blank_line(gs, handler, api):
    # _SSE_GAP is two newlines. With one, every event on both APIs collapses
    # into a single never-dispatched frame -- the whole stream unreadable to
    # any SDK -- and no assertion in this suite moves, because the only parsers
    # here and in genie_smoke read `data:` lines and ignore the terminator.
    gs.ENGINE = StubEngine(chunks=["hello ", "there"])
    h = handler()
    _run_stream(h, api)
    raw = b"".join(h.wfile.chunks)
    frames = _dispatched(raw)
    # One payload line per frame -- `data:` on both APIs, or the OpenAI
    # keep-alive's bare-comment line. Counting them is what catches the
    # collapse: a lost newline leaves the payload lines untouched and the
    # dispatched frames at 1.
    payloads = [ln for ln in raw.split(b"\n")
                if ln.startswith(b"data: ") or ln.startswith(b": ")]
    assert len(frames) == len(payloads) > 4
    for frame in frames:
        assert frame, "an empty frame is a spurious dispatch"
        for line in frame.split(b"\n"):
            assert line.startswith((b"data: ", b"event: ", b":")), frame


@pytest.mark.parametrize("stream", [False, True])
def test_the_openai_envelope_says_which_shape_it_is(gs, handler, stream):
    # Clients and routers dispatch on `object` to tell a chunk from a finished
    # completion. The Anthropic leg has this test already (its message /
    # message_start envelope, further down); the OpenAI leg -- the endpoint
    # typed and every SDK hit first -- had none, so swapping the two
    # discriminators, or deleting the envelope outright, left the suite green.
    gs.ENGINE = StubEngine(chunks=["hello ", "there"])
    before = int(time.time())
    code, body, h = request(gs, handler, "POST", "/v1/chat/completions",
                            {"messages": HI, "stream": stream,
                             "stream_options": {"include_usage": True}})
    assert code == 200
    if stream:
        frames = h.wfile.sse_frames()
        assert [f["object"] for f in frames] == ["chat.completion.chunk"] * len(frames)
        envelopes = {(f["id"], f["created"], f["model"]) for f in frames}
        assert len(envelopes) == 1, "one response, one id: %r" % (envelopes,)
        cmpl_id, created, model = envelopes.pop()
    else:
        assert body["object"] == "chat.completion"
        cmpl_id, created, model = body["id"], body["created"], body["model"]
    assert model == gs.MODEL_ID
    # The id is built from `created`, and every test above passes both by hand
    # -- so this is the only place the construction itself is pinned.
    assert cmpl_id == "chatcmpl-%d" % created
    assert before <= created <= int(time.time())


def test_parallel_tool_calls_keep_their_ordinals_on_the_stream(gs, handler):
    # One generation, two calls: the OpenAI SDKs accumulate streamed tool calls
    # BY index, so a collapsed ordinal merges two calls into one malformed call
    # and the agent runs the wrong thing. Every other tool test here emits a
    # single call, where any ordinal looks right.
    gs.TOOLS_OK = True
    gs.ENGINE = StubEngine(chunks=["Let me look. ", CALL, CALL2])
    code, _body, h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": HI, "tools": TOOLS + LIST_DIR,
                              "stream": True})
    assert code == 200
    deltas = [f["choices"][0]["delta"]["tool_calls"] for f in h.wfile.sse_frames()
              if f["choices"] and f["choices"][0]["delta"].get("tool_calls")]
    assert [len(d) for d in deltas] == [1, 1], "two calls shared one delta frame"
    calls = [d[0] for d in deltas]
    assert [c["index"] for c in calls] == [0, 1]
    assert [c["function"]["name"] for c in calls] == ["read_file", "list_dir"]
    cmpl_id = h.wfile.sse_frames()[0]["id"]
    assert [c["id"] for c in calls] == ["call_%s_0" % cmpl_id,
                                        "call_%s_1" % cmpl_id]


def test_a_text_plus_tool_turn_numbers_its_anthropic_blocks_in_order(gs, handler):
    # The ordinary agent shape: the model says something, then calls. Anthropic
    # carries each piece as its own content block and the SDK's accumulator
    # keys them by index, so collapsed indices or a missing content_block_stop
    # silently lose the text or a call from the assembled message. Nothing in
    # the suite streamed a turn carrying both on this leg.
    gs.TOOLS_OK = True
    gs.ENGINE = StubEngine(chunks=["Let me look. ", CALL, CALL2])
    code, _body, h = request(gs, handler, "POST", "/v1/messages",
                             {"messages": HI,
                              "tools": ANTHROPIC_TOOLS + ANTHROPIC_LIST_DIR,
                              "stream": True})
    assert code == 200
    frames = [f for f in h.wfile.sse_frames() if f["type"].startswith("content_block")]
    shape = [(f["type"], f["index"],
              (f.get("content_block") or f.get("delta") or {}).get("type"))
             for f in frames]
    assert shape == [
        ("content_block_start", 0, "text"),
        ("content_block_delta", 0, "text_delta"),
        ("content_block_stop", 0, None),
        ("content_block_start", 1, "tool_use"),
        ("content_block_delta", 1, "input_json_delta"),
        ("content_block_stop", 1, None),
        ("content_block_start", 2, "tool_use"),
        ("content_block_delta", 2, "input_json_delta"),
        ("content_block_stop", 2, None),
    ]
    starts = [f["content_block"] for f in frames if f["type"] == "content_block_start"]
    assert [b.get("name") for b in starts] == [None, "read_file", "list_dir"]
    msg_id = next(f for f in h.wfile.sse_frames()
                  if f["type"] == "message_start")["message"]["id"]
    assert [b["id"] for b in starts[1:]] == ["toolu_%s_0" % msg_id,
                                             "toolu_%s_1" % msg_id]
    assert [f["delta"]["text"] for f in frames
            if f["type"] == "content_block_delta"
            and f["delta"]["type"] == "text_delta"] == ["Let me look."]


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

def test_props_reports_n_ctx_as_the_software_window_cap(gs, handler):
    # Named for what n_ctx IS. This was "..._the_window_the_bundle_was_compiled
    # _with", the exact claim read_context_size's docstring retracts: the value
    # is dialog.context.size, the SOFTWARE cap, and the fixture's own compiled
    # graphs (_CONTEXT_LENGTHS, topping out at 4096) are not 8192 either. The
    # compiled windows are reported separately, under genie.context_lengths.
    gs._CONTEXT_SIZE = 8192
    code, body, _h = request(gs, handler, "GET", "/props")
    assert code == 200
    assert body["default_generation_settings"]["n_ctx"] == 8192


def test_props_omits_model_path(gs, handler):
    # typed checks model_path FIRST, so emitting it would take precedence and
    # display the bundle directory -- disagreeing with the name /health and
    # /v1/models already report. One name everywhere beats a detailed one in
    # a single place.
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert "model_path" not in body


def test_props_claims_no_modality_it_does_not_have(gs, handler):
    # Absence reads as text-only, which is the truth for this bundle.
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert "modality" not in body


def test_props_names_the_same_model_as_the_other_endpoints(gs, handler):
    _code, props, _h = request(gs, handler, "GET", "/props")
    _code, health, _h = request(gs, handler, "GET", "/health")
    _code, models, _h = request(gs, handler, "GET", "/v1/models")
    assert props["model_alias"] == health["model"] == models["data"][0]["id"]


def test_an_unknown_path_404s_rather_than_guessing(gs, handler):
    code, body, _h = request(gs, handler, "GET", "/v1/completions")
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
    # "Nothing after" covers the usage frame and the [DONE] sentinel too: a
    # dead Wire appends nothing once it raises, so two writes means neither was
    # even attempted. They used to have a test each (..._emits_no_usage_frame,
    # ..._emits_no_done_sentinel), asserting on the OUTPUT -- so, for the reason
    # given above, both stayed green with the latch patched out while this one
    # failed. One behaviour, and this is the assertion that can see it.
    assert h.wfile.writes == 2, (
        "wrote %d times to a socket known to be gone -- the latch is not "
        "holding" % h.wfile.writes)


@pytest.mark.parametrize("prefilled", [True, False])
def test_a_healthy_stream_writes_every_frame(gs, handler, prefilled):
    # The counterpart: the latch must not suppress on a LIVE connection.
    #
    # Asserted on frame KINDS and on the delivered text rather than on a write
    # COUNT, because the count depends on the orphan gate: prefilled=True (the
    # production default) holds the opening and this short reply goes out as
    # ONE content frame; prefilled=False is a pass-through and it is three. A
    # count would encode the hold and say nothing about the latch, which is
    # what this test is for. (It ran only the pass-through leg, under a comment
    # describing the held one.) What must hold either way: every frame kind
    # arrives, and no generated text is dropped on the way.
    gs.ENGINE = StubEngine(chunks=["a", "b", "c"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=True, prefilled=prefilled)
    frames = h.wfile.sse_frames()
    text = "".join(f["choices"][0]["delta"].get("content", "")
                   for f in frames if f.get("choices"))
    assert text == "abc", "generated text lost or reordered: %r" % text
    assert any(f.get("usage") for f in frames), "no usage frame on a live socket"
    assert any(f.get("choices") and f["choices"][0].get("finish_reason")
               for f in frames), "no finish frame on a live socket"
    assert h.wfile.text().rstrip().endswith("[DONE]"), "stream not terminated"


def test_the_gate_keeps_an_orphan_think_out_of_a_stream(gs, handler):
    """The duplicate the gate exists to remove, end to end through _stream.

    The model writes reasoning, closes a block the PREFILL opened, then answers
    -- so the raw generation carries the answer twice. Only the text after the
    orphan close may reach the client.
    """
    gs.ENGINE = StubEngine(chunks=["dup answer\n", "</think>\n", "\nreal answer"])
    h = handler()
    # prefilled=True: the gate only applies when a closed block was actually
    # sent, which is the only case an unmatched close can be ours to drop.
    h._stream("prompt", 100, "cid", 0, include_usage=True, prefilled=True)
    text = "".join(f["choices"][0]["delta"].get("content", "")
                   for f in h.wfile.sse_frames() if f.get("choices"))
    assert text == "real answer", (
        "orphan reasoning reached the client: %r" % text)


def test_an_answer_that_merely_MENTIONS_the_tag_is_not_truncated(gs, handler):
    """The orphan strip must not fire when this request enabled reasoning.

    Without the prefilled gate the strip keys on nothing but an unmatched close,
    so any answer whose first mention of the tag is bare loses everything before
    it -- `The </think> tag closes a reasoning block` came back as `tag closes a
    reasoning block`. A coding agent asking this server about its own
    suppression mechanism hits exactly that string, and the truncation is
    silent. Reasoning ON means no closed block was prefilled, so a closing tag
    can only be the model's own prose.
    """
    answer = "The </think> tag closes a reasoning block."
    gs.ENGINE = StubEngine(chunks=[answer])
    h = handler()
    h._complete("prompt", 100, "cid", 0, prefilled=False)
    assert json.loads(h.wfile.text())["choices"][0]["message"]["content"] == answer

    # AND with prefilled=True, which is the case that actually broke: reasoning
    # is suppressed by DEFAULT, so nearly every real request prefills and the
    # prefilled gate alone would not have saved this. What saves it is that an
    # INLINE mention is not a structural close -- measured live returning "tag
    # closes a reasoning block." before the line anchor landed.
    gs.ENGINE = StubEngine(chunks=[answer])
    h2 = handler()
    h2._complete("prompt", 100, "cid", 0, prefilled=True)
    assert json.loads(h2.wfile.text())["choices"][0]["message"]["content"] == answer


def test_the_same_answer_IS_stripped_when_a_block_was_prefilled(gs, handler):
    # The counterpart, so the guard cannot be "widened" into never stripping.
    gs.ENGINE = StubEngine(chunks=["reasoning\n</think>\n\nthe answer"])
    h = handler()
    h._complete("prompt", 100, "cid", 0, prefilled=True)
    assert json.loads(h.wfile.text())["choices"][0]["message"]["content"] == "the answer"


def test_a_reasoning_enabled_stream_is_not_gated(gs, handler):
    # Same guard on the streaming path: with reasoning ON the gate must pass
    # every chunk through rather than withholding and then dropping a prefix.
    gs.ENGINE = StubEngine(chunks=["The answer.\n", "</think>\n", "\nMore text."])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=True, prefilled=False)
    text = "".join(f["choices"][0]["delta"].get("content", "")
                   for f in h.wfile.sse_frames() if f.get("choices"))
    assert text == "The answer.\n</think>\n\nMore text."


@pytest.mark.parametrize("chunks", [
    # the orphan as observed live: tag alone on its line
    ["A GEMM is a model.\n", "</think>\n", "\nA GEMM is a network."],
    # the hazard: tag lands at the end of a BUFFER, and the next chunk turns out
    # to continue that same line -- so it was an inline mention all along
    ["The\n", "</think>", " tag is how you close it."],
    ["The ", "</think>", " tag closes it."],       # inline, split across chunks
    ["reasoning\n", "</think>"],                   # orphan with nothing after it
    ["<think>r</think>\n\n", "the answer"],        # well-formed pair
    ["just ", "an answer"],                        # no tag at all
])
def test_a_stream_delivers_exactly_what_the_buffered_path_would(gs, chunks):
    """The two paths must decide the same generation the same way.

    They reach the decision differently -- the buffered path sees the whole
    output at once, the gate sees a growing prefix -- and that asymmetry is a
    real trap: end-of-BUFFER is not end-of-OUTPUT, so a permissive anchor made
    the stream strip a tag whose line had not finished yet. Measured, the stream
    emitted "tag is how you close it." for a generation the buffered path kept
    whole. Pinned as an equivalence rather than as two separate expectations,
    because the failure is precisely that they diverge.
    """
    gate = gs._OrphanGate(prefilled=True)
    streamed = "".join(x for x in (gate.feed(c) for c in chunks) if x)
    streamed += gate.flush()
    assert streamed == gs._maybe_strip_think("".join(chunks), True)


def test_a_model_emitted_think_block_survives_the_gate_intact(gs):
    """The INVERSE of the duplicate bug, and the equivalence test cannot see it.

    A model can ignore the prefill and emit its own COMPLETE block. Qwen3's real
    blocks are multi-line, so the close sits at a line start and the anchor
    matches -- the gate then has to notice the matching open before it and pass
    everything through. If that branch regressed, a caller who set
    GENIE_THINKING=1 and explicitly asked for reasoning would have it silently
    deleted, which is the same silent-content-loss failure as the truncation,
    pointed the other way.

    The equivalence test's pair is single-line, so its close is not at a line
    start and the anchor never fires -- that case skips this branch entirely.
    """
    chunks = ["<think>\n", "reasoning\n", "</think>\n", "\nthe answer"]
    gate = gs._OrphanGate(prefilled=True)
    sent = "".join(x for x in (gate.feed(c) for c in chunks) if x) + gate.flush()
    assert sent == "".join(chunks), "a well-formed block was mangled: %r" % sent


def test_a_positive_hold_releases_at_the_cap(gs):
    """The bounded mode, which is the escape hatch the default replaced.

    GENIE_ORPHAN_HOLD_CHARS>0 keeps the old behaviour, and it is DOCUMENTED as
    leaking -- an orphan past the cap goes out. That is precisely why it needs a
    test: it is the setting someone reaches for in an incident, and an untested
    escape hatch already measured failing is the worst thing to find broken then.
    Pinned here is the release itself, not the leak: nothing may be withheld
    forever just because no tag ever arrived.
    """
    gate = gs._OrphanGate(limit=20, prefilled=True)
    out = [x for x in (gate.feed(c) for c in ["x" * 15, "y" * 15, "zzz"]) if x]
    assert "".join(out) + gate.flush() == "x" * 15 + "y" * 15 + "zzz"
    assert gate.open, "the cap must open the gate permanently, not per chunk"


def test_the_seed_is_bounded_to_what_genie_can_parse(gs, monkeypatch):
    """next_seed's only caller is device-gated, so nothing exercised it.

    The int32 bound is load-bearing rather than tidy: Genie parses `seed` into
    an int32_t, so a value past that wraps to something arbitrary instead of
    erroring -- and the symptom is silent non-determinism nobody traces back to
    a seed.
    """
    import importlib
    monkeypatch.delenv("GENIE_SEED", raising=False)
    importlib.reload(gs)
    assert gs.FIXED_SEED is None
    for _ in range(50):
        assert 1 <= gs.next_seed() < 2 ** 31 - 1

    monkeypatch.setenv("GENIE_SEED", "1234")
    importlib.reload(gs)
    # Pinned means pinned: every call, not a fresh draw seeded once.
    assert [gs.next_seed(), gs.next_seed()] == [1234, 1234]


def test_an_anthropic_user_turn_keeps_its_text_beside_a_tool_result(gs):
    # A client that comments on a result ("that failed, try X") sends BOTH
    # blocks in one turn. Dropping the comment reads as the model ignoring the
    # user, and it is the half of the turn that carries the new instruction.
    prompt, _dropped, fits, _overhead = gs._anthropic_to_prompt({"messages": [
        {"role": "user", "content": "run it"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "exit 1"},
            {"type": "text", "text": "that failed, try X"}]}]})
    assert fits
    assert "<tool_response>\nexit 1\n</tool_response>" in prompt
    assert "that failed, try X" in prompt, "the user's own words were dropped"


def test_explicit_top_p_and_top_k_are_mapped(gs):
    # Inert on QAIRT 2.45 (see apply_sampler), so this pins the MAPPING only --
    # which is what has to still be right on the day a runtime honours it.
    assert gs._sampler_params({"top_p": 0.5, "top_k": 7}) == {"top-p": 0.5,
                                                              "top-k": 7}
    # temp 0 from the REQUEST, not from the tool default, still implies greedy.
    assert gs._sampler_params({"temperature": 0}) == {"temp": 0.0, "top-k": 1}


def test_usage_counts_what_the_model_GENERATED_not_what_survived(gs, handler):
    """The house rule, applied to the orphan strip that broke it.

    `_complete` already said "count what the MODEL produced, not what survives
    parsing" for tool blocks. The orphan strip then ran on every request and the
    count was taken AFTER it, so reasoning the model genuinely produced was
    deducted. Measured 2026-08-28: the same prompt and cap took ~5.5s under two
    seeds and reported 51 tokens under one, 103 under the other -- the wall was
    the same because the WORK was the same. Anything dividing tokens by time
    then reads half rate, which is exactly what bench_endpoint did.

    Both paths, because they had drifted apart: streaming counted what it sent,
    non-streaming counted what survived, and neither counted the generation.
    """
    raw = "dup answer\n</think>\nreal answer"
    chunks = ["dup answer\n", "</think>\n", "real answer"]

    gs.ENGINE = StubEngine(chunks=chunks)
    h = handler()
    h._complete("prompt", 100, "cid", 0, prefilled=True)
    body = json.loads(h.wfile.text())
    assert body["choices"][0]["message"]["content"] == "real answer", \
        "the client should still receive only the answer"
    assert body["usage"]["completion_tokens"] == len(raw) // 4, \
        "billed %d for a %d-char generation" % (
            body["usage"]["completion_tokens"], len(raw))

    gs.ENGINE = StubEngine(chunks=chunks)
    h2 = handler()
    h2._stream("prompt", 100, "cid", 0, include_usage=True, prefilled=True)
    usage = next(f["usage"] for f in h2.wfile.sse_frames() if f.get("usage"))
    assert usage["completion_tokens"] == len(raw) // 4, \
        "streaming and non-streaming must bill the same turn identically"


def test_the_anthropic_paths_bill_the_generation_too(gs, handler):
    # Same rule, other API. These drifted independently once already.
    chunks = ["dup answer\n", "</think>\n", "real answer"]
    raw = "".join(chunks)

    gs.ENGINE = StubEngine(chunks=chunks)
    h = handler()
    h._anthropic_complete("prompt", 100, "m", "mid", prefilled=True)
    assert json.loads(h.wfile.text())["usage"]["output_tokens"] == len(raw) // 4

    gs.ENGINE = StubEngine(chunks=chunks)
    h2 = handler()
    h2._anthropic_stream("prompt", 100, "m", "mid", prefilled=True)
    deltas = [f for f in h2.wfile.sse_frames() if f.get("type") == "message_delta"]
    assert deltas and deltas[-1]["usage"]["output_tokens"] == len(raw) // 4


def test_the_gate_releases_a_short_reply_that_never_closes(gs, handler):
    """A reply shorter than the hold must still be delivered, not swallowed.

    Without the end-of-generation flush the gate would turn a rare duplicate
    into a routine EMPTY response, which is a far worse failure than the one it
    is fixing.
    """
    gs.ENGINE = StubEngine(chunks=["short ", "answer"])
    h = handler()
    # prefilled=True so the hold is actually exercised -- with it False the gate
    # is a pass-through and this would pass without testing the flush at all.
    h._stream("prompt", 100, "cid", 0, include_usage=True, prefilled=True)
    text = "".join(f["choices"][0]["delta"].get("content", "")
                   for f in h.wfile.sse_frames() if f.get("choices"))
    assert text == "short answer", "short reply was swallowed: %r" % text


def test_a_live_stream_still_gets_both(gs, handler):
    # The latch must not be so eager that it fires on a healthy stream.
    gs.ENGINE = StubEngine(chunks=["a", "b"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=True)
    assert [f for f in h.wfile.sse_frames() if f.get("usage")]
    assert "[DONE]" in h.wfile.text()


# --- the single-flight permit must always come back -----------------------
# A leaked permit is unrecoverable without a restart: the server answers 429
# forever while completely idle, which reads externally as "the NPU is busy"
# and would send the next investigator hunting a contention problem that does
# not exist.

def test_the_inflight_permit_is_released_when_the_generator_raises(gs, handler):
    """... and the failure is ANSWERED, which it used not to be.

    This asserted pytest.raises(RuntimeError): an exception out of a handler
    propagated out of do_POST, and on a real socket that is a connection closed
    with no response -- the failure the server itself calls the one a client
    cannot tell from the server being dead. The permit half is unchanged; the
    other half is now a 500 in the endpoint's own envelope.
    """
    def boom(req):
        raise RuntimeError("engine exploded mid-turn")

    gs.Handler._openai_chat = lambda self, req: boom(req)
    gs.Handler._anthropic_messages = lambda self, req: boom(req)
    before = gs._INFLIGHT._value
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": HI})
    assert gs._INFLIGHT._value == before, \
        "a leaked permit means 429-forever on an idle server"
    assert code == 500
    assert body == {"error": {"type": "server_error",
                              "message": "RuntimeError: engine exploded mid-turn"}}

    code, body, _h = request(gs, handler, "POST", "/v1/messages", {"messages": HI})
    assert gs._INFLIGHT._value == before
    assert code == 500
    assert body == {"type": "error",
                    "error": {"type": "api_error",
                              "message": "RuntimeError: engine exploded mid-turn"}}


def test_the_permit_is_released_on_the_ordinary_path(gs, handler):
    gs.Handler._openai_chat = lambda self, req: None
    before = gs._INFLIGHT._value
    request(gs, handler, "POST", "/v1/chat/completions", {"messages": HI})
    assert gs._INFLIGHT._value == before


def test_a_shed_request_does_not_release_a_permit_it_never_took(gs, handler):
    # Over-releasing a BoundedSemaphore raises ValueError and would take the
    # server down on the first burst of backpressure.
    gs.Handler._openai_chat = lambda self, req: None
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": HI})
    # The CODE is the contract -- docs/TYPED_ROUTER_BRIEF.md sells 429/529 as
    # the "shed to the next engine" signal and a router keys on it, not on the
    # message. The Anthropic 529 was asserted; this one never was, and turning
    # it into a 503 left the whole suite green.
    assert code == 429
    assert body["error"]["type"] == "overloaded_error"
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.release()          # must not raise


def test_a_refused_tool_request_never_queues_behind_a_generation(gs, handler):
    # Refusing costs no NPU time, so it is checked BEFORE the lock.
    gs.TOOLS_OK = False
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    try:
        code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                                 {"messages": HI, "tools": [{"type": "function"}]})
        assert code == 400
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

def test_props_says_which_engine_is_answering(gs, handler):
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert body["genie"]["engine"] == "npu-hexagon-htp"
    # The constraint a dispatcher has to encode, not discover from a 429.
    assert body["genie"]["single_flight"] is True


def test_props_exposes_the_compiled_graphs_not_just_the_window(gs, handler):
    gs._CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert body["genie"]["context_lengths"] == [512, 1024, 2048, 4096]
    assert body["genie"]["multi_length"] is True


def test_props_marks_a_single_length_bundle_as_such(gs, handler):
    # The 2-3x short-prompt difference a router would otherwise attribute to
    # the model or the depth.
    gs._CONTEXT_LENGTHS = [8192]
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert body["genie"]["multi_length"] is False


def test_props_reports_poll_because_it_decides_concurrency(gs, handler):
    # poll:true turns NPU+GPU from a 1.45x gain into a 0.78x loss, so a client
    # deciding whether to run a second engine needs to see it.
    gs._POLL_MATCHES = [(True, "QnnHtp.poll")]
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert body["genie"]["poll"] is True


def test_props_does_not_invent_a_poll_value_it_could_not_read(gs, handler):
    gs._POLL_MATCHES = []
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert body["genie"]["poll"] is None


def test_the_genie_block_is_additive_not_a_replacement(gs, handler):
    # Namespaced so no llama.cpp-shaped field is misreported, and additive so a
    # client that ignores it sees exactly what it saw before.
    _code, body, _h = request(gs, handler, "GET", "/props")
    assert body["default_generation_settings"]["n_ctx"] == gs.read_context_size()
    assert body["model_alias"] == gs.MODEL_ID
    assert "model_path" not in body and "modality" not in body


# --- POST /v1/messages: the surface typed actually drives ------------------
# Every load-shed branch in do_POST forks on the path, and only the OpenAI leg
# of each was covered. docs/TYPED_ROUTER_BRIEF.md sells 429/529 as THE
# "shed to the next engine" signal, so half of a documented contract was
# running unverified -- and the two error envelopes are not interchangeable:
# OpenAI is {"error": {...}}, Anthropic wraps it as {"type": "error",
# "error": {...}}. A client reading the wrong one sees an empty error.

def test_a_full_queue_sheds_an_anthropic_request_with_529(gs, handler):
    # 529 is the Anthropic spelling of the 429 the OpenAI leg returns. The
    # whole point of the code is that a router moves on to the next engine
    # instead of failing the turn, so the body shape has to be the one that
    # client can parse.
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    try:
        code, body, _h = request(gs, handler, "POST", "/v1/messages",
                                 {"messages": [{"role": "user", "content": "hi"}]})
    finally:
        for _ in range(gs.MAX_INFLIGHT):
            gs._INFLIGHT.release()          # must not raise: never acquired
    assert code == 529
    assert body == {"type": "error",
                    "error": {"type": "overloaded_error",
                              "message": "server busy; NPU is single-flight"}}


def test_a_refused_anthropic_tool_request_uses_the_anthropic_envelope(gs, handler):
    # The 400 that tells typed's probeLocalToolCalls to disable tools for the
    # session. In the OpenAI envelope there is no top-level "type", so an
    # Anthropic client reads it as an unparseable 400 and keeps sending
    # schemas the bundle can never act on.
    gs.TOOLS_OK = False
    code, body, _h = request(gs, handler, "POST", "/v1/messages", {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "read_file", "input_schema": {"type": "object"}}]})
    assert code == 400
    assert body["type"] == "error", "answered in the OpenAI shape"
    assert body["error"]["type"] == "invalid_request_error"
    assert "tool calling is not supported" in body["error"]["message"]
    assert gs.ENGINE.calls == []


def test_a_full_queue_does_not_turn_the_anthropic_400_into_a_529(gs, handler):
    # Refusing costs no NPU time, so the tools check runs BEFORE the permit.
    # Flip the order and a busy server answers 529 -- "come back later" -- to a
    # request that will never work, and a well-behaved router retries it
    # forever against the one engine guaranteed to refuse it.
    gs.TOOLS_OK = False
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    try:
        code, body, _h = request(gs, handler, "POST", "/v1/messages", {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}]})
    finally:
        for _ in range(gs.MAX_INFLIGHT):
            gs._INFLIGHT.release()
    assert code == 400
    assert "tool calling is not supported" in body["error"]["message"]


# --- _anthropic_messages, end to end ---------------------------------------
# The Anthropic handlers were only ever driven directly, so everything the
# entry point itself decides -- validation, the stream/complete fork, what it
# hands the engine -- was untested on the endpoint typed actually uses.

@pytest.mark.parametrize("payload", [{}, {"messages": []}])
def test_anthropic_messages_requires_messages(gs, handler, payload):
    # An empty turn is a client bug, and it has to say so in the envelope the
    # client parses. It also must not reach the NPU: on a single-flight device
    # a request that cannot succeed still costs everyone else the permit.
    code, body, _h = request(gs, handler, "POST", "/v1/messages", payload)
    assert code == 400
    assert body["type"] == "error"
    assert body["error"] == {"type": "invalid_request_error",
                             "message": "messages required"}
    assert gs.ENGINE.calls == []


def test_anthropic_messages_returns_a_well_formed_message(gs, handler):
    # The whole path in one go: request -> prompt -> engine -> message body.
    gs.ENGINE = StubEngine(chunks=["hello"])
    code, body, _h = request(gs, handler, "POST", "/v1/messages", {
        "model": "npu-router-name",
        "system": "You are a coding agent.",
        "messages": [{"role": "user", "content": "hi"}]})
    assert code == 200
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["id"].startswith("msg_")
    # Echoed, not replaced with MODEL_ID: a router fanning out to several
    # engines matches the reply to the request by the model it asked for.
    assert body["model"] == "npu-router-name"
    assert body["usage"]["output_tokens"] == len("hello") // 4
    # ... and the request really reached the engine, system prompt and all.
    assert len(gs.ENGINE.calls) == 1
    assert "You are a coding agent." in gs.ENGINE.calls[0]["prompt"]


def test_the_anthropic_model_echo_is_the_same_on_a_stream(gs, handler):
    # The echo is deliberate (the real Messages API answers with the model the
    # request named, and it is only a routing label -- /health, /props and
    # /v1/models say what is actually served), so it has to hold on both
    # response shapes, and an ABSENT model has to fall back to the truth.
    gs.ENGINE = StubEngine(chunks=["hello"])
    _code, _body, h = request(gs, handler, "POST", "/v1/messages",
                              {"model": "claude-x", "messages": HI, "stream": True})
    start = next(f for f in h.wfile.sse_frames() if f["type"] == "message_start")
    assert start["message"]["model"] == "claude-x"

    _code, body, _h = request(gs, handler, "POST", "/v1/messages", {"messages": HI})
    assert body["model"] == gs.MODEL_ID


@pytest.mark.parametrize("extra,expected", [
    ({"stream": True}, "stream"),
    ({"stream": False}, "complete"),
    ({}, "complete"),                    # absent means non-streaming
])
def test_the_stream_flag_picks_the_anthropic_handler(gs, handler, extra, expected):
    # The two paths write bodies that share nothing -- a run of SSE events
    # against one JSON object -- so dispatching to the wrong one hands the
    # client bytes it cannot parse at all, with a 200 on the front.
    fired = []
    gs.Handler._anthropic_stream = lambda self, *a, **k: fired.append("stream")
    gs.Handler._anthropic_complete = lambda self, *a, **k: fired.append("complete")
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    payload.update(extra)
    # _post, not _post_json: the stubs write no body to parse.
    request(gs, handler, "POST", "/v1/messages", payload)
    assert fired == [expected]


# --- Anthropic content blocks -> ChatML ------------------------------------
# Anthropic carries tool traffic as content BLOCKS (tool_use on assistant
# turns, tool_result on user turns) where ChatML wants assistant tool_calls
# and role="tool" messages. Flatten them to prose and a multi-turn tool
# conversation replays as the model never having seen a result -- so it calls
# the same tool again, forever, one NPU turn at a time.

def test_an_anthropic_system_string_is_folded_into_the_prompt(gs):
    # Anthropic puts the system prompt in a top-level field, not a message, so
    # it reaches the template only if this translation moves it.
    prompt, dropped, fits, overhead = gs._anthropic_to_prompt(
        {"system": "You are a coding agent.",
         "messages": [{"role": "user", "content": "hi"}]})
    assert (dropped, fits, overhead) == (0, True, 0)
    assert "You are a coding agent." in prompt


def test_a_tool_result_block_becomes_a_tool_response_turn(gs):
    # Left as a plain user turn the model reads a tool result as the human
    # talking, which is how an agent loop starts arguing with its own output.
    prompt, _dropped, fits, _overhead = gs._anthropic_to_prompt({"messages": [
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "read_file",
             "input": {"path": "a.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": "print(1)"}]},
    ]})
    assert fits
    assert "<tool_response>\nprint(1)\n</tool_response>" in prompt


def test_a_tool_use_block_becomes_a_tool_call_with_its_name(gs):
    # The name is the whole payload of the replayed call -- an empty or wrong
    # one makes the history describe a tool that was never invoked.
    prompt, _dropped, fits, _overhead = gs._anthropic_to_prompt({"messages": [
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "read_file",
             "input": {"path": "a.py"}}]},
    ]})
    assert fits
    assert "<tool_call>" in prompt
    assert '{"name": "read_file", "arguments": {"path": "a.py"}}' in prompt


def test_anthropic_tools_reach_the_prompt_as_openai_schemas(gs):
    # Qwen3 was trained on OpenAI-shaped function schemas inside <tools>.
    # Anthropic's input_schema spelling is one the model has never seen, and
    # tool-call accuracy is what pays for the difference.
    prompt, _dropped, fits, _overhead = gs._anthropic_to_prompt(
        {"messages": [{"role": "user", "content": "hi"}]},
        tools=[{"name": "read_file", "description": "Read a file",
                "input_schema": {"type": "object",
                                 "properties": {"path": {"type": "string"}}}}])
    assert fits
    assert '"type": "function"' in prompt
    assert '"parameters": {"type": "object"' in prompt
    assert "input_schema" not in prompt, "Anthropic's own spelling reached the model"


# --- do_POST entry point ---------------------------------------------------

def test_a_malformed_body_names_the_parse_problem(gs, handler):
    # "bad request" on its own sends the caller auditing their JSON encoder.
    # The parser's position is what identifies the common cause instead -- a
    # body cut short by a dropped connection.
    code, body, _h = request(gs, handler, "POST", "/v1/messages", b'{"messages": [{"role": "user"')
    assert code == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"].startswith("bad JSON:")
    assert "char" in body["error"]["message"], "a label with no parser detail"
    assert gs.ENGINE.calls == []


def test_a_malformed_body_uses_the_envelope_of_the_endpoint_it_hit(gs, handler):
    """The parse failure happens BEFORE the path fork, and used to ignore it.

    Every /v1/messages error was assumed to be Anthropic-shaped, but do_POST
    forked on path only for the tools refusal and the load shed -- both BELOW
    this failure. So a body cut short by a dropped connection came back to an
    Anthropic client as {"error": {...}} with no top-level "type": a shape it
    cannot parse, at the one moment it needs to be told something.

    Asserting error.type is NOT enough to catch this, which is why it went
    unnoticed: both envelopes carry error.type and error.message, so a check on
    those passes whichever shape is returned. The discriminator is the
    top-level "type": "error" that only the Anthropic envelope has.
    """
    code, body, _h = request(gs, handler, "POST", "/v1/messages", b'{"messages": [')
    assert code == 400
    assert body.get("type") == "error", (
        "an Anthropic client cannot parse an OpenAI-shaped error")
    assert body["error"]["message"].startswith("bad JSON:")

    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions", b'{"messages": [')
    assert code == 400
    assert "type" not in body, "the OpenAI envelope has no top-level type"
    assert body["error"]["message"].startswith("bad JSON:")


def test_the_tools_refusal_keeps_each_api_envelope(gs, handler):
    # The same fork, one layer down, now routed through the shared helper --
    # so a later edit cannot fix one envelope and leave the other behind.
    gs.TOOLS_OK = False
    req = {"messages": [{"role": "user", "content": "hi"}],
           "tools": [{"type": "function"}]}
    _code, anth, _h = request(gs, handler, "POST", "/v1/messages", req)
    assert anth.get("type") == "error"
    _code, oai, _h = request(gs, handler, "POST", "/v1/chat/completions", req)
    assert "type" not in oai


def test_an_unknown_post_path_404s_rather_than_guessing(gs, handler):
    # The GET twin is covered; this one was not. 404 rather than 400 is what
    # tells a probing client the endpoint is absent, not its request bad -- so
    # it stops asking instead of rewriting the payload.
    code, body, _h = request(gs, handler, "POST", "/v1/embeddings", {"input": "x"})
    assert code == 404
    assert body["error"]["type"] == "invalid_request_error"
    assert gs.ENGINE.calls == []
# --- the request door ------------------------------------------------------
# Every guard below runs BEFORE the single-flight semaphore, which is why they
# belong here rather than in a generic input-validation test. The NPU serves
# one generation at a time, so a bad request that gets through does not cost
# one slow response -- it holds the device while every other client waits.

def test_a_negative_max_tokens_is_refused_instead_of_wrapping(gs, handler):
    # GenieDialog_setMaxNumTokens takes a c_uint32, so -1 does not fail: it
    # arrives at the HTP as 4294967295. Nothing downstream caught it either --
    # build_windowed budgets with max(0, max_tokens), so a negative reads as
    # zero there and the overflow 400 never fired. The request was admitted and
    # then pinned the NPU until it walked into the context wall.
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": -1})
    assert code == 400
    assert "max_tokens" in body["error"]["message"]
    assert gs.ENGINE.calls == [], "refused at the door, never reached the NPU"


def test_the_anthropic_leg_refuses_a_negative_max_tokens_in_its_own_envelope(gs, handler):
    # Same guard, other API. The envelope follows the endpoint or a client that
    # cannot parse the error learns nothing at the moment it needs to.
    code, body, _h = request(gs, handler, "POST", "/v1/messages",
                             {"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": -5})
    assert code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert gs.ENGINE.calls == []


def test_max_tokens_past_the_window_keeps_the_number_the_client_sent(gs, handler):
    # NOT clamped to the window. The overflow 400 downstream tells the client to
    # "lower max_tokens", which is only actionable if the figure quoted back is
    # theirs -- clamping first would have rewritten 99999 to 4096 and produced
    # an error that reads as though the server refused its own value.
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 99999})
    assert code == 400
    assert "99999 max_tokens" in body["error"]["message"]
    assert gs.ENGINE.calls == []


def test_an_infinite_max_tokens_answers_instead_of_dropping_the_connection(gs, handler):
    # json.loads accepts Infinity, and int(float("inf")) raises OverflowError
    # rather than ValueError -- so this slipped past the first version of the
    # guard and died in do_POST with no response written.
    code, _body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                              b'{"messages": [{"role": "user", "content": "hi"}], '
                              b'"max_tokens": Infinity}')
    assert code == 400
    assert gs.ENGINE.calls == []


def test_a_negative_content_length_is_refused_rather_than_read(gs, handler):
    # rfile.read(-1) reads to EOF, which on a keep-alive socket never arrives:
    # one such request parks a handler thread for the life of the process, and
    # it sits ahead of the semaphore so it does not even need a permit.
    # (io.BytesIO returns instead of blocking, so this pins the guard, not the
    # hang it prevents -- the hang needs a real socket.)
    code, _body, h = request(gs, handler, "POST", "/v1/chat/completions", b"{}",
                             headers={"Content-Length": "-1"})
    assert code == 400
    assert h.close_connection is True
    assert gs.ENGINE.calls == []


def test_a_non_numeric_content_length_is_refused_too(gs, handler):
    # The third way the header can be wrong, and the only one of the three
    # that had no test: int("abc") raises, and uncaught that is a dropped
    # connection with no response. The connection is closed as well as
    # answered -- with no usable length there is no telling where this body
    # ends and the next request begins.
    for path, anthropic in (("/v1/chat/completions", False), ("/v1/messages", True)):
        code, body, h = request(gs, handler, "POST", path, b"{}",
                                headers={"Content-Length": "abc"})
        assert code == 400
        assert body["error"]["message"] == "bad Content-Length header"
        assert (body.get("type") == "error") is anthropic, "wrong envelope for %s" % path
        assert h.close_connection is True
    assert gs.ENGINE.calls == []


def test_a_chunked_body_is_refused_instead_of_desyncing_the_connection(gs, handler):
    # This server reads exactly Content-Length bytes. A chunked body leaves its
    # frames in the buffer, so the NEXT request on the same connection starts
    # parsing mid-frame -- a failure that surfaces on a request that was fine.
    code, _body, h = request(gs, handler, "POST", "/v1/chat/completions", b"{}",
                             headers={"Transfer-Encoding": "chunked"})
    assert code == 411
    assert h.close_connection is True


def test_a_messages_value_that_is_not_a_list_of_objects_gets_an_error(gs, handler):
    # "hi" and ["hi"] are both truthy, so they cleared the required-check and
    # then died rendering the template -- out of do_POST, no response at all.
    for bad in ("hi", ["hi"], [None], [[]]):
        code, body, _h = request(gs, handler, "POST", "/v1/chat/completions", {"messages": bad})
        assert code == 400, bad
        assert "list of objects" in body["error"]["message"], bad
    code, body, _h = request(gs, handler, "POST", "/v1/messages",
                             {"messages": "hi", "max_tokens": 10})
    assert code == 400
    assert body["type"] == "error"
    assert gs.ENGINE.calls == []


def test_the_modern_max_completion_tokens_spelling_is_honoured_too(gs):
    # OpenAI deprecated max_tokens in favour of max_completion_tokens, so which
    # one arrives depends on the vintage of the caller's SDK. Honouring one and
    # ignoring the other hands a client an unbounded generation on a
    # single-flight NPU for no reason it can see. This server honoured only the
    # legacy spelling until the two official servers were measured for the same
    # thing -- geniex serve has exactly this gap in mirror image.
    assert gs._max_tokens({"max_completion_tokens": 16}) == 16
    # legacy wins when both are sent, since it is the more explicit signal from
    # a client old enough to send it at all
    assert gs._max_tokens({"max_tokens": 8, "max_completion_tokens": 16}) == 8
    # and the validation applies to both spellings, not just the one
    with pytest.raises(ValueError):
        gs._max_tokens({"max_completion_tokens": -1})


def test_a_falsy_max_tokens_still_means_the_default(gs):
    # Absent, 0 and null all meant DEFAULT_MAX_TOKENS before the clamp existed
    # (`req.get(...) or DEFAULT`). A guard that quietly changed that would
    # shorten every reply from a client that sends max_tokens: 0.
    for payload in ({}, {"max_tokens": 0}, {"max_tokens": None}):
        assert gs._max_tokens(payload) == gs.DEFAULT_MAX_TOKENS


def test_a_non_numeric_max_tokens_answers_instead_of_dropping_the_connection(gs, handler):
    # int("abc") used to raise out of do_POST with no response written at all,
    # which a client cannot tell apart from the server being dead.
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": "abc"})
    assert code == 400
    assert gs.ENGINE.calls == []


def test_an_oversized_content_length_is_refused_before_the_body_is_read(gs, handler):
    # MAX_INFLIGHT bounds generations, not bytes, and this read sits ahead of
    # the semaphore -- so an unbounded body never had to queue for anything.
    # The body is nowhere near the declared length: the guard must answer on
    # the HEADER alone, without waiting for bytes that are never coming.
    code, body, h = request(gs, handler, "POST", "/v1/chat/completions", b"{}",
                            headers={"Content-Length": str(gs.MAX_BODY_BYTES + 1)})
    assert code == 413
    assert "too large" in body["error"]["message"]
    assert h.close_connection is True
    assert gs.ENGINE.calls == []


def test_valid_json_that_is_not_an_object_gets_an_error(gs, handler):
    # `[1, 2]` parses fine and then dies on req.get() -- the same silent-drop
    # failure as the non-numeric max_tokens above, one layer earlier.
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions", b"[1, 2]")
    assert code == 400
    assert "object" in body["error"]["message"]
    assert gs.ENGINE.calls == []


def test_the_host_warning_is_quiet_on_loopback_and_loud_off_it(gs):
    # Warn, never refuse -- the same contract as bundle_config_warnings. The
    # env-var table described 0.0.0.0 as a supported way to expose the server
    # with nothing attached about there being no auth behind it.
    for host in ("127.0.0.1", "::1", "localhost"):
        assert gs.host_exposure_warning(host) is None
    for host in ("0.0.0.0", "192.168.1.5"):
        warning = gs.host_exposure_warning(host)
        assert warning.startswith("WARNING:")
        assert "NO authentication" in warning
        assert host in warning


# --- an engine failure, on every response path ------------------------------
# GenieEngine._finish raises on every non-success status, so these branches
# are the ONLY route by which a real driver failure reaches a client -- and not
# one of them had a test. conftest's StubEngine never fails; the one raising
# engine in the suite drives the summariser, never a handler.

class FailingEngine(StubEngine):
    """Generates `chunks`, then fails -- both ways a generation can.

    raises=False is how GenieEngine.query_stream reports it: the worker catches
    the exception and hands it over as res["error"]. raises=True is an engine
    that raises out of the generator instead, which the real one never does on
    purpose and which must still not drop the connection.
    """
    MSG = "GenieDialog_query failed, status=5 (GENIE_STATUS_ERROR_QUERY_FAILED)"

    def __init__(self, chunks=None, raises=False, msg=None):
        super().__init__(chunks=chunks)
        self.raises = raises
        if msg is not None:
            self.MSG = msg

    def query_stream(self, prompt, res, **kw):
        self.calls.append(dict(kw, prompt=prompt))
        for c in self.chunks:
            self.yielded += 1
            yield c
        if self.raises:
            raise RuntimeError(self.MSG)
        res["finish"] = "stop"
        res["error"] = self.MSG


def _chat(path, stream, tools):
    """A request body for either door, with or without tools."""
    body = {"messages": HI, "stream": stream}
    if tools:
        body["tools"] = ANTHROPIC_TOOLS if path == "/v1/messages" else TOOLS
    return body


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("tools", [False, True])
def test_an_engine_failure_is_a_500_on_a_non_streaming_request(gs, handler,
                                                               tools, raises):
    gs.TOOLS_OK = True
    before = gs._INFLIGHT._value
    for path in ("/v1/chat/completions", "/v1/messages"):
        gs.ENGINE = FailingEngine(chunks=["partial "], raises=raises)
        code, body, _h = request(gs, handler, "POST", path, _chat(path, False, tools))
        assert code == 500
        if path == "/v1/messages":
            assert body == {"type": "error", "error": {"type": "api_error",
                                                       "message": FailingEngine.MSG}}
        else:
            assert body == {"error": {"type": "server_error",
                                      "message": FailingEngine.MSG}}
        assert gs._INFLIGHT._value == before, "the failure leaked a permit"


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("tools", [False, True])
def test_a_streamed_engine_failure_is_an_error_frame_not_an_answer(gs, handler,
                                                                   tools, raises):
    """OpenAI leg. The failure used to be assistant CONTENT.

    "[error: ...]" went out as a content delta, followed by finish_reason
    "stop" -- so an agent stored the error string as the model's reply and
    carried on, where the same failure without `stream` was a 500. Now it is
    the 500's body as a data frame, the way llama.cpp's server reports it and
    the shape the OpenAI SDKs raise on: no content carrying the message, no
    finish_reason claiming the turn ended, no usage, and still a [DONE].
    """
    gs.TOOLS_OK = True
    gs.ENGINE = FailingEngine(chunks=["partial "], raises=raises)
    before = gs._INFLIGHT._value
    code, _body, h = request(gs, handler, "POST", "/v1/chat/completions",
                             dict(_chat("/v1/chat/completions", True, tools),
                                  stream_options={"include_usage": True}))
    frames = h.wfile.sse_frames()
    assert code == 200, "the status line had already gone out"
    assert [f for f in frames if "error" in f] == [
        {"error": {"type": "server_error", "message": FailingEngine.MSG}}]
    choices = [f["choices"][0] for f in frames if f.get("choices")]
    assert not [c for c in choices if c["finish_reason"]], \
        "a finish_reason says the turn ended; it did not"
    assert "error" not in "".join(c["delta"].get("content") or "" for c in choices), \
        "the failure was delivered as the model's own words"
    assert not [f for f in frames if f.get("usage")]
    assert h.wfile.text().rstrip().endswith("data: [DONE]"), "stream left open"
    assert gs._INFLIGHT._value == before, "the failure leaked a permit"


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("tools", [False, True])
def test_a_streamed_anthropic_engine_failure_is_an_error_event(gs, handler,
                                                               tools, raises):
    # Same failure, Anthropic's spelling: an `error` event, which its SDKs
    # raise on. The tool stream already did this; the plain stream sent the
    # message as a text_delta and then stop_reason end_turn. No message_delta
    # now on either -- a stop_reason is a claim about how the turn ENDED.
    gs.TOOLS_OK = True
    gs.ENGINE = FailingEngine(chunks=["partial "], raises=raises)
    before = gs._INFLIGHT._value
    code, _body, h = request(gs, handler, "POST", "/v1/messages",
                             _chat("/v1/messages", True, tools))
    frames = h.wfile.sse_frames()
    assert code == 200
    assert [f for f in frames if f["type"] == "error"] == [
        {"type": "error", "error": {"type": "api_error",
                                    "message": FailingEngine.MSG}}]
    assert "event: error\n" in h.wfile.text().replace("\r\n", "\n")
    assert not [f for f in frames if f["type"] == "message_delta"]
    said = "".join(f["delta"].get("text", "") for f in frames
                   if f["type"] == "content_block_delta")
    assert "error" not in said, "the failure was delivered as the model's own words"
    assert frames[-1] == {"type": "message_stop"}, "stream left open"
    assert gs._INFLIGHT._value == before, "the failure leaked a permit"


def test_text_generated_before_a_failure_is_still_delivered(gs, handler):
    # The error frame replaces the LIE, not the output. What the model had
    # produced when the engine failed is real, and on an incremental stream
    # most of it has usually gone out already; the part the orphan gate was
    # still holding follows it rather than vanishing.
    gs.ENGINE = FailingEngine(chunks=["partial ", "answer"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, prefilled=True)
    frames = h.wfile.sse_frames()
    text = "".join(f["choices"][0]["delta"].get("content") or ""
                   for f in frames if f.get("choices"))
    assert text == "partial answer"
    assert "error" in frames[-1], "the error frame comes last, after the text"


class _ShutdownLib:
    """The only Genie calls a shutdown makes. A refused turn reaches none."""

    def GenieDialog_signal(self, dialog, action):
        return 0

    def GenieDialog_free(self, dialog):
        return 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_a_request_that_arrives_during_shutdown_is_shed_not_failed(
        gs, handler, path, stream, capsys):
    # What the client parked behind the engine lock at Ctrl-C actually gets.
    # The REAL engine, because the whole chain is what is under test: shutdown
    # marks it closing, _run_query refuses the turn at the lock,
    # query_stream's worker keeps that refusal apart from an engine failure,
    # and the handler answers it 503 -- shed, not broken. A 500 would tell a
    # router this engine failed the request; a traceback out of do_POST would
    # tell the client nothing at all.
    gs.ENGINE = gs.GenieEngine(_ShutdownLib(), "DIALOG")
    gs.ENGINE.begin_shutdown()              # main()'s finally, first half
    before = gs._INFLIGHT._value
    code, body, h = request(gs, handler, "POST", path,
                            {"messages": HI, "stream": stream})
    if not stream:
        assert code == 503
        err = body["error"]
        assert err["type"] == ("api_error" if path == "/v1/messages"
                               else "server_error")
        assert "shutting down" in err["message"]
        assert (body.get("type") == "error") is (path == "/v1/messages")
    else:
        # The 200 went out before the engine was asked, so all that is left is
        # the failure shape each SDK raises on -- the same one an engine
        # failure gets, and still not a dropped connection.
        assert code == 200
        frames = h.wfile.sse_frames()
        said = [f for f in frames if "error" in f or f.get("type") == "error"]
        assert len(said) == 1 and "shutting down" in json.dumps(said)
        assert "shutting down" not in "".join(
            json.dumps(f) for f in frames if f not in said), (
            "the refusal was delivered as the model's own words")
    assert gs._INFLIGHT._value == before, "the refusal leaked a permit"
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 0, (
        "a turn refused at the door was never a generation to book")
    assert "shutting down" in capsys.readouterr().out, "and it is in the log"


def test_an_engine_failure_mid_stream_leaves_a_line_in_the_log(gs, handler, capsys):
    # The 200 has already gone out, so the per-response line _json prints for
    # a non-2xx is never written for this failure. ONE line, whatever the
    # driver put in its message -- a native error can span several.
    gs.ENGINE = FailingEngine(msg="query failed, status=5\n  backend: QnnHtp\n")
    request(gs, handler, "POST", "/v1/messages", _chat("/v1/messages", True, False))
    assert capsys.readouterr().out == (
        "[genie] engine failure mid-stream POST /v1/messages: "
        "query failed, status=5 backend: QnnHtp\n")


# --- the turn's own verdict, from the door to the wire -----------------------
# test_finish_reason.py drives StubEngine(finish="length") through the six
# handler paths directly and through the door without `stream`. This is the
# remaining leg: a STREAMED request, from a real body to the closing frame.

def test_a_capped_stream_says_so_end_to_end_on_both_apis(gs, handler):
    gs.ENGINE = StubEngine(chunks=["The answer ", "was cut"], finish="length")
    code, _body, h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": HI, "stream": True, "max_tokens": 5})
    reasons = [f["choices"][0]["finish_reason"] for f in h.wfile.sse_frames()
               if f.get("choices")]
    assert code == 200 and reasons[-1] == "length"
    assert gs.ENGINE.calls[-1]["max_tokens"] == 5

    code, _body, h = request(gs, handler, "POST", "/v1/messages",
                             {"messages": HI, "stream": True, "max_tokens": 5,
                              "stop_sequences": ["END"]})
    deltas = [f for f in h.wfile.sse_frames() if f["type"] == "message_delta"]
    # max_tokens, not stop_sequence: the cap outranks a stop sequence that was
    # supplied and never reached.
    assert code == 200 and deltas[-1]["delta"]["stop_reason"] == "max_tokens"


# --- tools, through the door -------------------------------------------------
# Every POST-with-tools test elsewhere runs with TOOLS_OK False -- the module
# default, which the fixture's reload restores -- and so dies at the 400 gate.
# tools_active=True reached the handlers only in tests that pass it by hand, so
# dropping `tools_active=bool(tools)` from an entry point left the suite green
# while a real <tool_call> came back as plain content with finish "stop": the
# accepted-and-dropped degradation this server says it refuses.

@pytest.mark.parametrize("stream", [False, True])
def test_openai_tools_reach_the_engine_and_come_back_as_tool_calls(gs, handler,
                                                                   stream):
    gs.TOOLS_OK = True
    gs.ENGINE = StubEngine(chunks=[CALL])
    code, body, h = request(gs, handler, "POST", "/v1/chat/completions",
                            {"messages": HI, "tools": TOOLS, "stream": stream})
    assert code == 200
    # Greedy BECAUSE tools are active -- the sampler is where the entry point's
    # own tools_active plumbing shows, independent of the response path.
    assert gs.ENGINE.calls[0]["sampler"] == {"temp": 0.0, "top-k": 1}
    assert '"name": "read_file"' in gs.ENGINE.calls[0]["prompt"], "schema never rendered"
    if stream:
        choices = [f["choices"][0] for f in h.wfile.sse_frames() if f.get("choices")]
        calls = [tc for c in choices for tc in c["delta"].get("tool_calls", [])]
        finish = choices[-1]["finish_reason"]
        leaked = "".join(c["delta"].get("content") or "" for c in choices)
    else:
        calls = body["choices"][0]["message"]["tool_calls"]
        finish = body["choices"][0]["finish_reason"]
        leaked = body["choices"][0]["message"]["content"] or ""
    assert [c["function"]["name"] for c in calls] == ["read_file"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}
    assert finish == "tool_calls"
    assert "<tool_call>" not in leaked, "the raw block reached the client as text"


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_tools_reach_the_engine_and_come_back_as_tool_use(gs, handler,
                                                                    stream):
    gs.TOOLS_OK = True
    gs.ENGINE = StubEngine(chunks=[CALL])
    code, body, h = request(gs, handler, "POST", "/v1/messages",
                            {"messages": HI, "tools": ANTHROPIC_TOOLS,
                             "stream": stream})
    assert code == 200
    assert gs.ENGINE.calls[0]["sampler"] == {"temp": 0.0, "top-k": 1}
    if stream:
        frames = h.wfile.sse_frames()
        blocks = [f["content_block"] for f in frames
                  if f["type"] == "content_block_start"]
        args = "".join(f["delta"].get("partial_json", "") for f in frames
                       if f["type"] == "content_block_delta")
        reason = next(f for f in frames
                      if f["type"] == "message_delta")["delta"]["stop_reason"]
    else:
        blocks = body["content"]
        args = json.dumps(blocks[0]["input"])
        reason = body["stop_reason"]
    assert [(b["type"], b.get("name")) for b in blocks] == [("tool_use", "read_file")]
    assert json.loads(args) == {"path": "a.py"}
    assert reason == "tool_use"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("tools,ok", [
    ("ab", False),             # one "function signature" per CHARACTER
    ({"k": 1}, False),         # ... or per key
    (["x"], False),
    ([], True),                # no tools is not a malformed request
    ("real", True),
])
def test_tools_must_be_a_list_of_objects(gs, handler, path, tools, ok):
    # ChatML.build json.dumps whatever it iterates, so the first two rendered a
    # garbage schema and answered 200; on the Anthropic leg a string died in
    # _anthropic_tools on `"a".get` with no response written at all.
    gs.TOOLS_OK = True
    gs.ENGINE = StubEngine(chunks=["hello"])
    if tools == "real":
        tools = ANTHROPIC_TOOLS if path == "/v1/messages" else TOOLS
    code, body, _h = request(gs, handler, "POST", path,
                             {"messages": HI, "tools": tools})
    if ok:
        assert code == 200
        return
    assert code == 400
    assert body["error"] == {"type": "invalid_request_error",
                             "message": "tools must be a list of objects"}
    assert (body.get("type") == "error") is (path == "/v1/messages")
    assert gs.ENGINE.calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_non_ascii_tool_argument_comes_back_as_the_model_wrote_it(gs, handler,
                                                                    stream):
    """... because the client's echo of it has to be a PREFIX of the KV.

    On the OpenAI leg `arguments` is a JSON string, and ChatML.build splices a
    string verbatim when the client sends the turn back as history. json.dumps'
    default would have handed out a backslash-u escape for the e-acute the model
    emitted raw, the echoed turn would no longer start with what the dialog
    holds, and every turn after such a call would re-prefill from scratch.
    """
    cafe = "café.txt"
    generation = ('<tool_call>\n{"name": "read_file", "arguments": {"path": "%s"}}'
                  '\n</tool_call>' % cafe)
    gs.TOOLS_OK = True
    gs.ENGINE = StubEngine(chunks=[generation])
    _code, body, h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": HI, "tools": TOOLS, "stream": stream})
    if stream:
        call = next(tc for f in h.wfile.sse_frames() if f.get("choices")
                    for tc in f["choices"][0]["delta"].get("tool_calls", []))
    else:
        call = body["choices"][0]["message"]["tool_calls"][0]
    arguments = call["function"]["arguments"]
    assert cafe in arguments and "\\u" not in arguments

    # The round trip that matters: what the engine holds is prompt+generation,
    # and the next request renders the echoed call in front of the new turn.
    held = gs.ENGINE.calls[0]["prompt"] + generation
    echo = [*HI,
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": call["id"], "type": "function", "function": call["function"]}]},
            {"role": "tool", "tool_call_id": call["id"], "content": "print(1)"}]
    assert gs.TEMPLATE.build(echo, tools=TOOLS, thinking=False).startswith(held)


# --- refused at the door, in the right envelope ------------------------------

@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_a_lone_surrogate_is_a_400_not_an_engine_failure(gs, handler, path):
    # json.loads accepts "\ud83d" -- what JavaScript emits for a string cut
    # through an emoji -- and the str it makes cannot be encoded for Genie. The
    # engine refuses it, but that surfaced as a 500, or as an error frame on a
    # stream that had already said 200, for what is the client's malformed text.
    raw = ('{"messages": [{"role": "user", "content": "cut \\ud83d here"}],'
           ' "stream": %s}')
    for stream in ("false", "true"):
        code, body, _h = request(gs, handler, "POST", path,
                                 (raw % stream).encode("ascii"))
        assert code == 400, "stream=%s" % stream
        assert body["error"]["type"] == "invalid_request_error"
        assert "not valid Unicode" in body["error"]["message"]
        assert (body.get("type") == "error") is (path == "/v1/messages")
    assert gs.ENGINE.calls == []


_WRONG_TYPED = [("temperature", "hot"), ("top_k", "many"), ("top_p", [0.9]),
                ("stop", 5)]


@pytest.mark.parametrize("path,field,value", [
    *[("/v1/chat/completions", f, v) for f, v in _WRONG_TYPED],
    *[("/v1/messages", f, v) for f, v in _WRONG_TYPED],
    ("/v1/messages", "stop_sequences", 5),
    # An OpenAI field; the Anthropic leg never reads it.
    ("/v1/chat/completions", "stream_options", "yes"),
])
def test_a_wrong_typed_field_is_answered_not_dropped(gs, handler, path, field, value):
    """Valid JSON, wrong type: used to raise out of do_POST with nothing written.

    A closed connection with no response is the failure the server itself
    calls the one a client cannot tell from the server being dead. 400 rather
    than 500, because the request never reached the engine and a router must
    not shed it to the next engine as though this one were broken -- and named,
    because "bad request" alone sends the caller auditing the wrong thing.
    """
    before = gs._INFLIGHT._value
    code, body, _h = request(gs, handler, "POST", path,
                             {"messages": HI, "stream": True, field: value})
    assert code == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert "malformed request" in body["error"]["message"]
    assert (body.get("type") == "error") is (path == "/v1/messages")
    assert gs.ENGINE.calls == [], "refused before any NPU time was spent"
    assert gs._INFLIGHT._value == before


def test_a_failure_after_the_headers_closes_instead_of_answering_twice(gs, handler,
                                                                        capsys):
    # The catch-all answers only while it still can. Once a status line is out
    # a second one would corrupt the response, so the connection is closed --
    # unambiguous, at least -- and the log says why.
    def half(self, req):
        self._json(200, {"ok": True})
        raise RuntimeError("after the fact")

    gs.Handler._openai_chat = half
    code, body, h = request(gs, handler, "POST", "/v1/chat/completions",
                            {"messages": HI})
    assert (code, body) == (200, {"ok": True}), "a second response was written"
    assert h.close_connection is True
    assert "failed mid-response" in capsys.readouterr().out


# --- routing ------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/health?probe=1", "/props?x", "/v1/models?limit=20",
                                  "/health/", "/healthz?ts=1726500000"])
def test_a_query_string_does_not_turn_an_endpoint_into_a_404(gs, handler, path):
    # Routing compared the raw request target, so a health checker's
    # cache-buster made a healthy server answer 404.
    code, _body, _h = request(gs, handler, "GET", path)
    assert code == 200


@pytest.mark.parametrize("method,target", [
    ("GET", "http://[/v1/models"),                 # ValueError: Invalid IPv6 URL
    ("POST", "http://[x]/v1/chat/completions"),    # ... not an IPv4 or IPv6 address
    ("GET", "http://[/health?probe=1"),
])
def test_a_target_urlsplit_refuses_is_a_404_not_a_dropped_connection(
        gs, handler, method, target):
    # urlsplit RAISES for an absolute-form target with a malformed bracketed
    # host, and _route runs first thing in do_GET and do_POST, outside every
    # catch-all. Comparing the raw target had answered these with a 404; the
    # ValueError left through socketserver instead -- a traceback on stderr
    # and a connection closed with no response at all.
    code, body, _h = request(gs, handler, method, target,
                             {"messages": HI} if method == "POST" else None)
    assert code == 404
    assert "error" in body


def test_the_anthropic_beta_query_string_still_reaches_messages(gs, handler):
    # Anthropic's SDK posts to /v1/messages?beta=true for beta features, so
    # this one arrives from a real client -- and an error on it must still be
    # in the Anthropic envelope, which is chosen from the same routed path.
    gs.ENGINE = StubEngine(chunks=["hello"])
    code, body, _h = request(gs, handler, "POST", "/v1/messages?beta=true",
                             {"messages": HI})
    assert code == 200 and body["type"] == "message"
    code, body, _h = request(gs, handler, "POST", "/v1/messages?beta=true", b"{nope")
    assert code == 400 and body.get("type") == "error"


def test_the_retrieve_model_route_resolves_the_one_model_served(gs, handler):
    # Some SDKs GET /v1/models/<id> to validate a model name before the first
    # request. The item is the list endpoint's item, not a second spelling.
    _code, listed, _h = request(gs, handler, "GET", "/v1/models")
    code, body, _h = request(gs, handler, "GET", "/v1/models/" + gs.MODEL_ID)
    assert code == 200
    assert body == listed["data"][0]
    assert body["id"] == gs.MODEL_ID and body["object"] == "model"

    code, body, _h = request(gs, handler, "GET", "/v1/models/gpt-4o")
    assert code == 404
    assert body["error"]["code"] == "model_not_found"
    assert gs.MODEL_ID in body["error"]["message"], "say what IS served"


# --- /health says how good the token counts are ------------------------------

@pytest.mark.parametrize("status,expected", [
    (None, "estimated"),       # before load_engine has run
    (0, "exact"),              # GENIE_STATUS_SUCCESS: a tokenizer is attached
    (2, "estimated"),          # getTokenizer failed: every count is chars/4
])
def test_health_reports_whether_token_counts_are_exact(gs, handler, status, expected):
    # Every usage figure and every window budget this server reports is one or
    # the other, and nothing over HTTP said which. Read from the module-level
    # status rather than the engine, because /health must answer during a wedge
    # (test_supervision pins that it never touches ENGINE).
    gs.TOKENIZER_STATUS = status
    code, body, _h = request(gs, handler, "GET", "/health")
    assert code == 200
    assert body["token_counts"] == expected


# --- one line per request that was not served --------------------------------

def test_every_refusal_leaves_exactly_one_line_in_the_log(gs, handler, capsys):
    # log_message is a no-op, and nothing replaced it for rejections: an
    # overflow 400, a 413 and a load-shed 429/529 left no trace, so a client
    # that was shed all afternoon was invisible in the log of the server that
    # shed it.
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    try:
        request(gs, handler, "POST", "/v1/chat/completions", {"messages": HI})
        request(gs, handler, "POST", "/v1/messages", {"messages": HI})
    finally:
        for _ in range(gs.MAX_INFLIGHT):
            gs._INFLIGHT.release()
    request(gs, handler, "GET", "/nope")
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "[genie] 429 POST /v1/chat/completions: server busy; NPU is single-flight",
        "[genie] 529 POST /v1/messages: server busy; NPU is single-flight",
        "[genie] 404 GET /nope: not found"]


def test_a_served_request_and_a_health_poll_stay_quiet(gs, handler, capsys):
    # No access log: an agent makes hundreds of requests and a line each says
    # nothing. And not /health's 503 either -- that is the ANSWER, polled for
    # as long as a wedge lasts, and the watchdog already announces the state
    # change once; a line per poll would bury it.
    gs.ENGINE = StubEngine(chunks=["hello"])
    request(gs, handler, "POST", "/v1/chat/completions", {"messages": HI})
    gs.HEALTH.begin(0)
    gs.HEALTH.note_stall_signalled(1)
    code, _body, _h = request(gs, handler, "GET", "/health")
    assert code == 503
    assert capsys.readouterr().out == ""


def test_a_log_line_cannot_be_the_reason_a_response_was_not_written(gs, handler,
                                                                    capsys):
    # The message can quote the request. A console in a legacy codepage raises
    # on what it cannot encode, and newlines would split one event over several
    # lines -- so the line is ASCII, single, and bounded.
    code, _body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                              {"messages": HI, "max_tokens": "café\nx" * 200})
    assert code == 400
    out = capsys.readouterr().out
    assert out.isascii() and out.count("\n") == 1
    assert len(out) < 600


# --- a socket that stops moving ------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    # http.server's own default is None: wait forever. A client that sent
    # fewer bytes than its Content-Length, or an idle keep-alive that never
    # sent a request line, parked a handler thread for the life of the process.
    (None, 120),
    ("30", 30),
    ("0", None),               # 0 or less is the old behaviour, on request
    ("-1", None),
    ("soon", 120),             # a typo degrades to the default, like every reader
])
def test_the_socket_timeout_is_one_env_var(gs, monkeypatch, capsys, value, expected):
    import importlib
    if value is None:
        monkeypatch.delenv("GENIE_SOCKET_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("GENIE_SOCKET_TIMEOUT", value)
    importlib.reload(gs)
    assert gs.Handler.timeout == expected
    assert ("GENIE_SOCKET_TIMEOUT" in capsys.readouterr().out) is (value == "soon")


def test_a_body_that_stops_arriving_is_the_client_leaving(gs, handler, capsys):
    # socket.timeout (an alias of TimeoutError since 3.10) out of rfile.read:
    # the client promised bytes and stopped sending them. No response -- a peer
    # that will not finish its own request is not waiting to read one -- but
    # the connection closes and the log says.
    class Stalled:
        def read(self, n):
            raise TimeoutError("timed out")

    h = handler()
    h.command, h.path = "POST", "/v1/chat/completions"
    h.headers = {"Content-Length": "4096"}
    h.rfile = Stalled()
    h.do_POST()
    assert h.close_connection is True
    assert h.wfile.writes == 0
    assert gs.ENGINE.calls == []
    assert "stopped sending its 4096-byte body" in capsys.readouterr().out


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_a_write_that_times_out_is_the_client_leaving(gs, handler, api):
    # A client that stopped READING costs everyone else exactly what one that
    # hung up does. socket.timeout -- TimeoutError -- is an OSError but NOT a
    # ConnectionError, so catching only the latter would let it escape.
    class StopsReading:
        writes = 0

        def write(self, b):
            self.writes += 1
            if self.writes > 1:
                raise TimeoutError("timed out")

        def flush(self):
            pass

    gs.ENGINE = StubEngine(chunks=["tok "] * 2000)
    h = handler()
    h.wfile = StopsReading()
    _run_stream(h, api, tools_active=True)
    assert gs.ENGINE.aborted
    assert gs.ENGINE.yielded == 1, gs.ENGINE.yielded
    assert h.wfile.writes == 2, "wrote again after the timeout"


# --- a NON-streaming client that hangs up --------------------------------------
# It writes nothing until the answer is complete, so no write can fail and an
# abandoned request used to run to its cap holding the single-flight NPU. The
# socket is asked instead: a closed peer is readable with nothing to read.

def _complete_on(h, api, **kw):
    if api == "openai":
        h._complete("prompt", 2000, "cid", 0, **kw)
    else:
        h._anthropic_complete("prompt", 2000, "m", "mid", **kw)


class LeavesMidGeneration(StubEngine):
    """The peer closes its socket once the generation is under way: after the
    first chunk has been handed over, so the i=0 probe still sees it there."""

    def __init__(self, theirs, **kw):
        super().__init__(**kw)
        self.theirs = theirs

    def query_stream(self, prompt, res, **kw):
        for n, c in enumerate(super().query_stream(prompt, res, **kw)):
            if n == 1:
                self.theirs.close()
            yield c


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_a_non_streaming_client_that_left_stops_the_generation(gs, handler, api):
    ours, theirs = socket.socketpair()
    try:
        gs.ENGINE = LeavesMidGeneration(theirs, chunks=["tok "] * 2000)
        h = handler()
        h.connection = ours
        _complete_on(h, api)
        assert gs.ENGINE.aborted, "never signalled abort"
        # There at the i=0 probe, gone by the i=8 one: the same cadence the
        # silent streams are held to.
        assert gs.ENGINE.yielded == 9, "ran on for %d chunks" % gs.ENGINE.yielded
        assert h.wfile.writes == 0, "answered a client that was not there"
        assert h.close_connection is True
    finally:
        ours.close()
        theirs.close()


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_a_non_streaming_client_gone_before_the_query_costs_nothing(gs, handler, api):
    # The non-streaming half of test_a_client_gone_before_the_query_starts_
    # costs_nothing, which only ever drove the two STREAM paths -- where a
    # failed first frame is what notices. A non-streaming response has no
    # first frame, so it went straight to query_stream: the lock wait, a full
    # prefill and one token, all for a client that had closed before the
    # handler got to it (a timeout firing during a summarisation, say). The
    # socket is asked before the generator is even made.
    ours, theirs = socket.socketpair()
    try:
        theirs.close()
        gs.ENGINE = StubEngine(chunks=["tok "] * 50)
        h = handler()
        h.connection = ours
        _complete_on(h, api)
        assert gs.ENGINE.calls == [], "queried the engine for a client already gone"
        assert gs.ENGINE.yielded == 0
        assert h.wfile.writes == 0, "answered a client that was not there"
        assert h.close_connection is True
    finally:
        ours.close()


# --- the handler against the REAL engine ---------------------------------------
# Every test above puts StubEngine behind the handler, and StubEngine's
# query_stream has no close-abort -- so what a disconnect costs the real engine
# was pinned half at a time: signal_abort alone sent [ABORT], close alone sent
# [ABORT], and nothing ran the two in the handler's order.

@pytest.mark.parametrize("shape", ["openai-stream", "anthropic-stream",
                                   "openai-tools", "non-stream"])
def test_a_disconnect_reaches_the_real_engine_as_one_abort(gs, handler, shape):
    from test_engine_kv import ABORT, FakeLib
    release = threading.Event()
    lib = FakeLib(chunks=["tok "] * 12, status=1)
    lib.on_query = lambda: release.wait(timeout=5)   # parked, as a decode is
    gs.ENGINE = gs.GenieEngine(lib, object())
    try:
        if shape == "non-stream":
            h = handler()
            answers = iter([False])                 # there before the query...
            h._client_gone = lambda: next(answers, True)    # ...gone after it
            h._complete("prompt", 2000, "cid", 0)
        else:
            api = shape.split("-")[0]
            tools = shape.endswith("tools")
            h = handler(fail_after=_preamble(api, tools) + 1)
            _run_stream(h, api, tools_active=tools, prefilled=not tools)
        assert lib.signals == [ABORT], (
            "lost() signalled the turn and the generator's close signalled it "
            "again: %r" % (lib.signals,))
    finally:
        release.set()


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("pipelined", [b"", b"POST /v1/chat/completions HTTP/1.1\r\n"])
def test_a_non_streaming_client_that_stayed_gets_its_answer(gs, handler, api,
                                                            pipelined):
    # The counterpart, so the check cannot be "widened" into always-gone. The
    # second case is the one a naive version gets wrong: a readable socket is
    # NOT a closed one -- bytes waiting are a pipelined next request, and the
    # peek must leave them for the read loop that owns them.
    ours, theirs = socket.socketpair()
    try:
        if pipelined:
            theirs.sendall(pipelined)
        gs.ENGINE = StubEngine(chunks=["tok "] * 20)
        h = handler()
        h.connection = ours
        _complete_on(h, api)
        assert not gs.ENGINE.aborted
        assert gs.ENGINE.yielded == 20
        assert "tok tok" in h.wfile.text()
        if pipelined:
            assert ours.recv(len(pipelined), socket.MSG_PEEK) == pipelined, (
                "the probe consumed the next request")
    finally:
        ours.close()
        theirs.close()


# --- the orphan gate's documented off switch ------------------------------------

def test_a_zero_hold_disables_the_gate_even_when_a_block_was_prefilled(gs):
    # GENIE_ORPHAN_HOLD_CHARS=0 is the setting docs/GENIE_SERVER.md recommends
    # to a human watching tokens appear: "streams every chunk as it arrives".
    # Only the default (-1), a positive cap, and the prefilled=False pass-through
    # were ever constructed in a test, so the one documented way to turn the
    # hold OFF was unpinned.
    gate = gs._OrphanGate(limit=0, prefilled=True)
    assert gate.open is True
    assert [gate.feed(c) for c in ["x", "</think>\n", "y"]] == ["x", "</think>\n", "y"]
    assert gate.flush() == ""


# --- over a real socket ---------------------------------------------------------
# Everything above drives a socketless Handler, which is what makes a disconnect
# stageable at all -- but three things only exist on a real connection: the
# timeout reaches the socket through StreamRequestHandler.setup(), a closed
# peer shows up as a READABLE socket rather than as a failed write, and a write
# to one does not fail until the reset has come back. Device-free all the same:
# a Server on a loopback port with the stub engine behind it.

class SlowEngine(StubEngine):
    """StubEngine with a little time between chunks, so a client can leave."""

    def query_stream(self, prompt, res, **kw):
        self.calls.append(dict(kw, prompt=prompt))
        for c in self.chunks:
            time.sleep(0.005)
            self.yielded += 1
            yield c
        res["finish"] = self.finish


@pytest.fixture
def served(gs):
    """(gs, port) with a real Server answering on a daemon thread."""
    srv = gs.Server(("127.0.0.1", 0), gs.Handler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01},
                     daemon=True).start()
    try:
        yield gs, srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _post_bytes(path, payload):
    body = json.dumps(payload).encode("utf-8")
    return (("POST %s HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
             "Content-Length: %d\r\n\r\n" % (path, len(body))).encode("ascii") + body)


def _wait_for(cond, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_an_idle_connection_is_dropped_instead_of_parking_a_thread(served):
    # A keep-alive connection that never sends a request line. With no timeout
    # the handler thread sits in readline() for the life of the process, and
    # ThreadingHTTPServer does not bound how many of those there can be.
    gs, port = served
    gs.Handler.timeout = 0.2
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        assert c.recv(1) == b"", "the server should have closed it"


def test_a_body_cut_short_is_dropped_instead_of_parking_a_thread(served, capsys):
    # Fewer bytes than Content-Length promised: rfile.read(length) waits for
    # the rest, ahead of the single-flight semaphore, so MAX_INFLIGHT never
    # bounded it.
    gs, port = served
    gs.Handler.timeout = 0.2
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                  b"Content-Length: 4096\r\n\r\n{\"messages\": [")
        assert c.recv(65536) == b"", "closed without an answer, not left open"
    assert gs.ENGINE.calls == []
    seen = []

    def logged():
        seen.append(capsys.readouterr().out)
        return "stopped sending its 4096-byte body" in "".join(seen)
    assert _wait_for(logged, 2.0), "dropped silently"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_hanging_up_on_a_non_streaming_request_frees_the_npu(served, path):
    gs, port = served
    gs.ENGINE = SlowEngine(chunks=["tok "] * 1000)     # ~5s if nobody notices
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(_post_bytes(path, {"messages": HI, "max_tokens": 1000}))
        assert _wait_for(lambda: gs.ENGINE.yielded > 0), "never started generating"
    assert _wait_for(lambda: gs.ENGINE.aborted), "ran on with nobody waiting"
    assert gs.ENGINE.yielded < 1000, gs.ENGINE.yielded
    assert _wait_for(lambda: gs._INFLIGHT._value == gs.MAX_INFLIGHT), \
        "the abandoned request kept its permit"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_hanging_up_before_a_non_streaming_query_starts_never_starts_it(served, path):
    # The same over a real connection, with the window a real client falls
    # into: the request is in, the handler is still building the prompt (a
    # summarisation is seconds of NPU time), and the client's timeout closes
    # the socket gracefully. That FIN is all the server will ever hear.
    gs, port = served
    gs.ENGINE = SlowEngine(chunks=["tok "] * 1000)
    building, real = threading.Event(), gs.build_windowed

    def slow_build(*a, **k):
        building.set()
        time.sleep(0.3)
        return real(*a, **k)
    gs.build_windowed = slow_build
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(_post_bytes(path, {"messages": HI, "max_tokens": 1000}))
        assert building.wait(timeout=10), "the request never reached the handler"
    assert _wait_for(lambda: gs._INFLIGHT._value == gs.MAX_INFLIGHT), \
        "the abandoned request kept its permit"
    assert gs.ENGINE.calls == [], "queried the engine for a client already gone"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_hanging_up_on_a_silent_stream_frees_the_npu(served, path):
    # The production default -- thinking off, so the orphan gate holds the
    # whole stream and only the keep-alive probe ever touches the socket.
    gs, port = served
    gs.ENGINE = SlowEngine(chunks=["tok "] * 1000)
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(_post_bytes(path, {"messages": HI, "max_tokens": 1000,
                                     "stream": True}))
        assert c.recv(65536).startswith(b"HTTP/1.1 200"), "no stream opened"
    assert _wait_for(lambda: gs.ENGINE.aborted), "ran on with nobody reading"
    assert gs.ENGINE.yielded < 1000, gs.ENGINE.yielded


def test_a_whole_exchange_over_a_real_connection(served):
    # The counterpart: none of the above may cost a client that stays. One
    # keep-alive connection, a JSON answer and then a second request on it --
    # which the peek that looks for a closed peer must not have eaten into.
    gs, port = served
    gs.ENGINE = StubEngine(chunks=["hello ", "there"])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        for _ in range(2):
            conn.request("POST", "/v1/chat/completions?x=1",
                         body=json.dumps({"messages": HI}),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            body = json.loads(r.read())
            assert r.status == 200
            assert body["choices"][0]["message"]["content"] == "hello there"
        conn.request("POST", "/v1/messages",
                     body=json.dumps({"messages": HI, "stream": True}),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        events = r.read().decode("utf-8")
        assert r.status == 200
        assert "event: message_start" in events and "hello there" in events
        assert events.rstrip().endswith('data: {"type": "message_stop"}')
    finally:
        conn.close()
    assert not gs.ENGINE.aborted


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_the_response_headers_are_the_ones_each_body_needs(served, path):
    """A real client reads the media type before it reads a byte of the body.

    The socketless Handler cannot see this at all -- the `handler` fixture
    no-ops send_header -- so every header below could be deleted with the suite
    green. A strict SSE consumer (EventSource, httpx-sse, a router that
    validates the media type) refuses a stream that is not text/event-stream,
    and an SSE body carries no Content-Length, so without `Connection: close` a
    read-to-EOF client hangs on the keep-alive -- which is the hang
    _sse_headers' own comment says that header is there to prevent.
    """
    gs, port = served
    gs.ENGINE = StubEngine(chunks=["hello ", "there"])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, body=json.dumps({"messages": HI}),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        r.read()
        assert r.status == 200
        assert r.getheader("Content-Type") == "application/json"
        assert r.getheader("Content-Length") is not None
        # Second request on the same connection: the JSON answer above framed
        # itself with a length and left the connection usable.
        conn.request("POST", path,
                     body=json.dumps({"messages": HI, "stream": True}),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        assert r.status == 200
        assert r.getheader("Content-Type") == "text/event-stream"
        assert r.getheader("Cache-Control") == "no-cache"
        assert r.getheader("Connection") == "close"
        assert r.getheader("Content-Length") is None, "an SSE body has no length"
        assert r.read().endswith(b"\n\n"), "the stream did not read to EOF"
    finally:
        conn.close()


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS"])
def test_a_method_this_server_does_not_serve_is_refused_out_loud(served, capsys,
                                                                 method):
    """501 from the stdlib, logged by the send_error override.

    Pinned as it is, including the refusal of `HEAD /health` -- a supervisor or
    load balancer that probes with HEAD reads this server as down, and that is
    a decision to take deliberately rather than by accident. What is a
    documented promise is the line: docs/GENIE_SERVER.md says the log carries
    "the stdlib's own refusals such as 501", and those never pass through
    _json, so this override is the only thing writing them -- and it had never
    been executed by a test at all.
    """
    gs, port = served
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(("%s /health HTTP/1.1\r\nHost: t\r\n\r\n"
                   % method).encode("ascii"))
        raw = b""
        while True:
            chunk = c.recv(65536)
            if not chunk:
                break
            raw += chunk
    head = raw.split(b"\r\n\r\n", 1)[0].decode("ascii")
    assert head.startswith("HTTP/1.1 501 Unsupported method (%r)" % method)
    assert "Content-Type: text/html;charset=utf-8" in head
    assert gs.ENGINE.calls == []
    assert ("[genie] 501 %s /health: Unsupported method (%r)" % (method, method)
            in capsys.readouterr().out)
