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
    #
    # Asserted on frame KINDS and on the delivered text rather than on a write
    # COUNT. The orphan-think gate coalesces the opening of a short stream into
    # one content frame, so a count here would encode the current hold size and
    # break on any change to it -- while saying nothing about the latch, which
    # is what this test is for. What must hold regardless of the hold: every
    # frame kind arrives, and no generated text is dropped on the way.
    gs.ENGINE = StubEngine(chunks=["a", "b", "c"])
    h = handler()
    h._stream("prompt", 100, "cid", 0, include_usage=True)
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


def test_a_malformed_int_env_var_does_not_kill_the_server(gs, monkeypatch,
                                                          capsys):
    # A typo used to raise `invalid literal for int()` at IMPORT, before any
    # startup line printed -- a traceback naming neither the variable nor the
    # form it wanted, on a server whose banner exists to explain itself.
    import importlib
    monkeypatch.setenv("GENIE_SEED", "abc")
    importlib.reload(gs)
    assert gs.FIXED_SEED is None
    assert "GENIE_SEED" in capsys.readouterr().out


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


# --- POST /v1/messages: the surface typed actually drives ------------------
# Every load-shed branch in do_POST forks on the path, and only the OpenAI leg
# of each was covered. docs/TYPED_ROUTER_BRIEF.md sells 429/529 as THE
# "shed to the next engine" signal, so half of a documented contract was
# running unverified -- and the two error envelopes are not interchangeable:
# OpenAI is {"error": {...}}, Anthropic wraps it as {"type": "error",
# "error": {...}}. A client reading the wrong one sees an empty error.

def _post_raw(gs, handler_factory, body, path="/v1/messages"):
    """Like _post above, but keeps the status code and takes RAW bytes.

    The code is the load-shed contract (a router keys on 529, not on the
    message), and raw bytes are the only way to reach the JSON-parse failure.
    """
    h = handler_factory()
    h.path = path
    sent = {}
    h.send_response = lambda code: sent.setdefault("code", code)
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    h.do_POST()
    return sent.get("code"), json.loads(h.wfile.text())


def _post_json(gs, handler_factory, payload, path="/v1/messages"):
    return _post_raw(gs, handler_factory, json.dumps(payload).encode(), path)


def test_a_full_queue_sheds_an_anthropic_request_with_529(gs, handler):
    # 529 is the Anthropic spelling of the 429 the OpenAI leg returns. The
    # whole point of the code is that a router moves on to the next engine
    # instead of failing the turn, so the body shape has to be the one that
    # client can parse.
    for _ in range(gs.MAX_INFLIGHT):
        gs._INFLIGHT.acquire()
    try:
        code, body = _post_json(gs, handler,
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
    code, body = _post_json(gs, handler, {
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
        code, body = _post_json(gs, handler, {
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
    code, body = _post_json(gs, handler, payload)
    assert code == 400
    assert body["type"] == "error"
    assert body["error"] == {"type": "invalid_request_error",
                             "message": "messages required"}
    assert gs.ENGINE.calls == []


def test_anthropic_messages_returns_a_well_formed_message(gs, handler):
    # The whole path in one go: request -> prompt -> engine -> message body.
    gs.ENGINE = StubEngine(chunks=["hello"])
    code, body = _post_json(gs, handler, {
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
    _post(gs, handler, payload, path="/v1/messages")
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
    code, body = _post_raw(gs, handler, b'{"messages": [{"role": "user"')
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
    code, body = _post_raw(gs, handler, b'{"messages": [', path="/v1/messages")
    assert code == 400
    assert body.get("type") == "error", (
        "an Anthropic client cannot parse an OpenAI-shaped error")
    assert body["error"]["message"].startswith("bad JSON:")

    code, body = _post_raw(gs, handler, b'{"messages": [',
                           path="/v1/chat/completions")
    assert code == 400
    assert "type" not in body, "the OpenAI envelope has no top-level type"
    assert body["error"]["message"].startswith("bad JSON:")


def test_the_tools_refusal_keeps_each_api_envelope(gs, handler):
    # The same fork, one layer down, now routed through the shared helper --
    # so a later edit cannot fix one envelope and leave the other behind.
    gs.TOOLS_OK = False
    req = {"messages": [{"role": "user", "content": "hi"}],
           "tools": [{"type": "function"}]}
    _code, anth = _post_json(gs, handler, req, path="/v1/messages")
    assert anth.get("type") == "error"
    _code, oai = _post_json(gs, handler, req, path="/v1/chat/completions")
    assert "type" not in oai


def test_an_unknown_post_path_404s_rather_than_guessing(gs, handler):
    # The GET twin is covered; this one was not. 404 rather than 400 is what
    # tells a probing client the endpoint is absent, not its request bad -- so
    # it stops asking instead of rewriting the payload.
    code, body = _post_json(gs, handler, {"input": "x"}, path="/v1/embeddings")
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
    code, body = _post_json(gs, handler,
                            {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": -1},
                            path="/v1/chat/completions")
    assert code == 400
    assert "max_tokens" in body["error"]["message"]
    assert gs.ENGINE.calls == [], "refused at the door, never reached the NPU"


def test_the_anthropic_leg_refuses_a_negative_max_tokens_in_its_own_envelope(gs, handler):
    # Same guard, other API. The envelope follows the endpoint or a client that
    # cannot parse the error learns nothing at the moment it needs to.
    code, body = _post_json(gs, handler,
                            {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": -5},
                            path="/v1/messages")
    assert code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert gs.ENGINE.calls == []


def test_max_tokens_past_the_window_keeps_the_number_the_client_sent(gs, handler):
    # NOT clamped to the window. The overflow 400 downstream tells the client to
    # "lower max_tokens", which is only actionable if the figure quoted back is
    # theirs -- clamping first would have rewritten 99999 to 4096 and produced
    # an error that reads as though the server refused its own value.
    code, body = _post_json(gs, handler,
                            {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 99999},
                            path="/v1/chat/completions")
    assert code == 400
    assert "99999 max_tokens" in body["error"]["message"]
    assert gs.ENGINE.calls == []


def test_an_infinite_max_tokens_answers_instead_of_dropping_the_connection(gs, handler):
    # json.loads accepts Infinity, and int(float("inf")) raises OverflowError
    # rather than ValueError -- so this slipped past the first version of the
    # guard and died in do_POST with no response written.
    code, _body = _post_raw(gs, handler,
                            b'{"messages": [{"role": "user", "content": "hi"}], '
                            b'"max_tokens": Infinity}',
                            path="/v1/chat/completions")
    assert code == 400
    assert gs.ENGINE.calls == []


def test_a_negative_content_length_is_refused_rather_than_read(gs, handler):
    # rfile.read(-1) reads to EOF, which on a keep-alive socket never arrives:
    # one such request parks a handler thread for the life of the process, and
    # it sits ahead of the semaphore so it does not even need a permit.
    # (io.BytesIO returns instead of blocking, so this pins the guard, not the
    # hang it prevents -- the hang needs a real socket.)
    h = handler()
    h.path = "/v1/chat/completions"
    sent = {}
    h.send_response = lambda code: sent.setdefault("code", code)
    h.headers = {"Content-Length": "-1"}
    h.rfile = io.BytesIO(b"{}")
    h.do_POST()
    assert sent.get("code") == 400
    assert h.close_connection is True
    assert gs.ENGINE.calls == []


def test_a_chunked_body_is_refused_instead_of_desyncing_the_connection(gs, handler):
    # This server reads exactly Content-Length bytes. A chunked body leaves its
    # frames in the buffer, so the NEXT request on the same connection starts
    # parsing mid-frame -- a failure that surfaces on a request that was fine.
    h = handler()
    h.path = "/v1/chat/completions"
    sent = {}
    h.send_response = lambda code: sent.setdefault("code", code)
    h.headers = {"Content-Length": "2", "Transfer-Encoding": "chunked"}
    h.rfile = io.BytesIO(b"{}")
    h.do_POST()
    assert sent.get("code") == 411
    assert h.close_connection is True


def test_a_messages_value_that_is_not_a_list_of_objects_gets_an_error(gs, handler):
    # "hi" and ["hi"] are both truthy, so they cleared the required-check and
    # then died rendering the template -- out of do_POST, no response at all.
    for bad in ("hi", ["hi"], [None], [[]]):
        code, body = _post_json(gs, handler, {"messages": bad},
                                path="/v1/chat/completions")
        assert code == 400, bad
        assert "list of objects" in body["error"]["message"], bad
    code, body = _post_json(gs, handler, {"messages": "hi", "max_tokens": 10},
                            path="/v1/messages")
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
    code, body = _post_json(gs, handler,
                            {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": "abc"},
                            path="/v1/chat/completions")
    assert code == 400
    assert gs.ENGINE.calls == []


def test_an_oversized_content_length_is_refused_before_the_body_is_read(gs, handler):
    # MAX_INFLIGHT bounds generations, not bytes, and this read sits ahead of
    # the semaphore -- so an unbounded body never had to queue for anything.
    h = handler()
    h.path = "/v1/chat/completions"
    sent = {}
    h.send_response = lambda code: sent.setdefault("code", code)
    h.headers = {"Content-Length": str(gs.MAX_BODY_BYTES + 1)}
    # The body is nowhere near the declared length: the guard must answer on
    # the HEADER alone, without waiting for bytes that are never coming.
    h.rfile = io.BytesIO(b"{}")
    h.do_POST()
    assert sent.get("code") == 413
    assert "too large" in json.loads(h.wfile.text())["error"]["message"]
    assert h.close_connection is True
    assert gs.ENGINE.calls == []


def test_valid_json_that_is_not_an_object_gets_an_error(gs, handler):
    # `[1, 2]` parses fine and then dies on req.get() -- the same silent-drop
    # failure as the non-numeric max_tokens above, one layer earlier.
    code, body = _post_raw(gs, handler, b"[1, 2]", path="/v1/chat/completions")
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
