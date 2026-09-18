"""How a turn ENDED, carried from the engine to the client on every path.

The engine reports "length" when a generation ran into its token cap -- the
context window filling, or max_tokens being reached -- and "stop" when the
model finished on its own. Agentic clients key continuation off exactly that
difference: `finish_reason: "length"` / `stop_reason: "max_tokens"` means
"resume from here", and anything else means "done".

StubEngine has always taken a `finish=` argument and no test had ever passed
it, so every handler in the suite only ever saw "stop". Hardcoding "stop" in
any of the six response paths below -- or deleting the line that carries the
engine's verdict out of a stream -- passed all of it.

Device-free, like everything beside it: the engine is conftest's StubEngine,
except in the last section -- a turn the SERVER aborted (shutdown, or the
watchdog on a stall) -- where the real GenieEngine runs over a small fake of
the Genie calls, because who aborted a turn is the engine's to know.
"""

import json

import pytest

from conftest import StubEngine, request

CHUNKS = ["The answer ", "was cut off"]


@pytest.fixture
def cut(gs):
    """A module whose engine reports every turn as ended by the token cap."""
    gs.ENGINE = StubEngine(chunks=CHUNKS, finish="length")
    return gs


# --- /v1/chat/completions ---------------------------------------------------

def test_a_capped_completion_says_length(cut, handler):
    h = handler()
    h._complete("prompt", 100, "cid", 0)
    body = json.loads(h.wfile.text())
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["message"]["content"] == "".join(CHUNKS)


@pytest.mark.parametrize("tools_active", [False, True])
def test_a_capped_stream_says_length(cut, handler, tools_active):
    # Two code paths: plain streaming forwards chunks as they arrive, and a
    # tools-active stream buffers the whole turn to parse calls out of it. Each
    # reads the engine's verdict off the result dict for itself.
    h = handler()
    h._stream("prompt", 100, "cid", 0, tools_active=tools_active)
    reasons = [f["choices"][0]["finish_reason"] for f in h.wfile.sse_frames()
               if f.get("choices")]
    assert [r for r in reasons if r] == ["length"]
    assert reasons[-1] == "length", "on the closing frame, where clients read it"


def test_a_tool_call_still_outranks_the_cap(gs, handler):
    # A turn that produced a usable call is a tool_calls turn however it ended;
    # the cap only matters when there is nothing for the client to act on.
    call = '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    gs.ENGINE = StubEngine(chunks=[call], finish="length")
    h = handler()
    h._stream("prompt", 100, "cid", 0, tools_active=True)
    reasons = [f["choices"][0]["finish_reason"] for f in h.wfile.sse_frames()
               if f.get("choices")]
    assert reasons[-1] == "tool_calls"


# --- /v1/messages -----------------------------------------------------------

def test_a_capped_message_says_max_tokens(cut, handler):
    h = handler()
    h._anthropic_complete("prompt", 100, "some-model", "msg_1")
    body = json.loads(h.wfile.text())
    assert body["stop_reason"] == "max_tokens"


@pytest.mark.parametrize("tools_active", [False, True])
def test_a_capped_message_stream_says_max_tokens(cut, handler, tools_active):
    h = handler()
    h._anthropic_stream("prompt", 100, "some-model", "msg_1",
                        tools_active=tools_active)
    deltas = [f for f in h.wfile.sse_frames() if f.get("type") == "message_delta"]
    assert [d["delta"]["stop_reason"] for d in deltas] == ["max_tokens"]


def test_the_cap_outranks_a_stop_sequence_on_the_wire(cut, handler):
    # With stop sequences supplied, a natural end is reported as stop_sequence
    # (see _anthropic_stop_reason). Running into the cap is not that: the
    # caller's boundary was never reached.
    h = handler()
    h._anthropic_stream("prompt", 100, "some-model", "msg_1", stop=["END"])
    deltas = [f for f in h.wfile.sse_frames() if f.get("type") == "message_delta"]
    assert deltas[0]["delta"]["stop_reason"] == "max_tokens"


# --- end to end, through the door -------------------------------------------
# The same two facts from a real request body, via conftest.request: the one
# helper that sets path / headers / rfile on the fixture's handler and keeps
# the status code.

def test_both_endpoints_report_the_cap_end_to_end(cut, handler):
    msgs = [{"role": "user", "content": "hi"}]
    code, body, h = request(cut, handler, "POST", "/v1/chat/completions",
                            {"messages": msgs, "max_tokens": 5})
    assert code == 200 and body["choices"][0]["finish_reason"] == "length"
    assert cut.ENGINE.calls[-1]["max_tokens"] == 5

    code, body, h = request(cut, handler, "POST", "/v1/messages",
                            {"messages": msgs, "max_tokens": 5})
    assert code == 200 and body["stop_reason"] == "max_tokens"


def test_the_request_helper_covers_the_shapes_the_local_copies_did(gs, handler):
    # GET, with no body and no Content-Length...
    code, body, h = request(gs, handler, "GET", "/health")
    assert code == 200 and body["status"] == "ok" and h.command == "GET"
    # ...RAW bytes, the only way to reach the JSON-parse failure...
    code, body, h = request(gs, handler, "POST", "/v1/chat/completions", b"{not json")
    assert code == 400 and body["error"]["type"] == "invalid_request_error"
    # ...a header the test wants to lie about...
    code, body, h = request(gs, handler, "POST", "/v1/chat/completions", b"{}",
                            headers={"Content-Length": "-1"})
    assert code == 400
    # ...and a response that is not JSON: parsed_body is None, read the wire.
    gs.ENGINE = StubEngine(chunks=["hi"])
    code, body, h = request(gs, handler, "POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "x"}],
                             "stream": True})
    assert code == 200 and body is None
    assert h.wfile.text().rstrip().endswith("data: [DONE]")


# --- a turn the SERVER cut short is not a finish ----------------------------
# Shutdown (Ctrl-C) and the watchdog (a stall) abort a live generation, and
# Genie then returns WARNING_ABORTED, which _finish reports as "stop" -- right
# for a client that left, since nobody reads that answer. For a client that is
# still there it was a 200 carrying text cut wherever the abort landed,
# labelled stop / end_turn (stop_sequence when it sent stop sequences): a
# fragment an agent files as a complete answer, beside the 503 the queued turn
# behind it got. These drive the REAL GenieEngine over a fake of the handful
# of Genie calls one generation makes, so what is under test is the engine's
# account of who aborted the turn AND every response path's use of it.


class CutShort:
    """Genie, for one generation that the server aborts part-way.

    GenieDialog_query hands back two chunks, then runs `during` (the abort,
    from inside the query, as shutdown or the watchdog would land it) and
    returns WARNING_ABORTED -- or SUCCESS when nothing aborted it.
    """

    def __init__(self, during):
        self.during = during
        self.signals = []

    def GenieDialog_reset(self, dialog):
        return 0

    def GenieDialog_setStopSequence(self, dialog, payload):
        return 0

    def GenieDialog_setMaxNumTokens(self, dialog, n):
        return 0

    def GenieDialog_getSampler(self, dialog, out):
        return -1               # the apply is inert on QAIRT 2.45 anyway

    def GenieDialog_query(self, dialog, data, code, cb, udata):
        for chunk in CHUNKS:
            cb(chunk.encode("utf-8"), 2, None)
        self.during()
        return 1 if self.signals else 0

    def GenieDialog_signal(self, dialog, action):
        self.signals.append(action)
        return 0


def _real_engine(gs, by):
    eng = gs.GenieEngine(None, "DIALOG")
    abort = {"shutdown": eng.begin_shutdown,
             "watchdog": lambda: eng.signal_abort(any_turn=True, stalled=True),
             None: lambda: None}[by]
    eng.lib = CutShort(abort)
    gs.ENGINE = eng
    return eng


BODIES = {
    "openai": ("/v1/chat/completions",
               {"messages": [{"role": "user", "content": "hi"}]}),
    "anthropic": ("/v1/messages",
                  {"messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 64, "stop_sequences": ["END"]}),
}
CODES = {"shutdown": 503, "watchdog": 500}


@pytest.mark.parametrize("by", ["shutdown", "watchdog"])
@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_a_buffered_answer_the_server_cut_is_an_error_not_a_finish(
        gs, handler, by, api):
    # Non-streaming: there is still a status to choose, so the fragment is
    # not sent at all -- 503 for shutdown (shed, retry elsewhere: nothing was
    # wrong with the request), 500 for the watchdog (the engine failed it).
    _real_engine(gs, by)
    path, body = BODIES[api]
    code, resp, _h = request(gs, handler, "POST", path, body)
    assert code == CODES[by], resp
    if api == "openai":
        assert "choices" not in resp and resp["error"]["type"] == "server_error"
        msg = resp["error"]["message"]
    else:
        assert "stop_reason" not in resp and resp["type"] == "error"
        assert resp["error"]["type"] == "api_error"
        msg = resp["error"]["message"]
    assert "did not finish" in msg
    assert ("shutting down" if by == "shutdown" else "stalled") in msg


@pytest.mark.parametrize("by", ["shutdown", "watchdog"])
@pytest.mark.parametrize("tools_active", [False, True])
def test_a_stream_the_server_cut_ends_on_an_error_frame_not_a_finish(
        gs, handler, by, tools_active):
    # The 200 has gone out, so the API's own mid-stream error shape is the
    # whole signal: an `error` frame, and NO finish_reason anywhere -- saying
    # how the turn "ended" is the lie. [DONE] still closes it.
    _real_engine(gs, by)
    h = handler()
    h._stream("prompt", 64, "cid", 0, tools_active=tools_active)
    frames = h.wfile.sse_frames()
    reasons = [f["choices"][0]["finish_reason"] for f in frames if f.get("choices")]
    assert [r for r in reasons if r] == [], "no finish on a cut turn: %r" % reasons
    errors = [f["error"] for f in frames if "error" in f]
    assert len(errors) == 1 and "did not finish" in errors[0]["message"]
    assert "error" in frames[-1], "the error is the last word"
    assert h.wfile.text().rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("by", ["shutdown", "watchdog"])
@pytest.mark.parametrize("tools_active", [False, True])
def test_a_message_stream_the_server_cut_ends_on_an_error_event(
        gs, handler, by, tools_active):
    # Anthropic's spelling of the same: an `error` event, no message_delta --
    # so no stop_reason, where this used to say end_turn, or stop_sequence
    # for a request that had sent stop sequences.
    _real_engine(gs, by)
    h = handler()
    h._anthropic_stream("prompt", 64, "some-model", "msg_1",
                        tools_active=tools_active, stop=["END"])
    frames = h.wfile.sse_frames()
    types = [f.get("type") for f in frames]
    assert "message_delta" not in types, "a stop_reason on a cut turn"
    assert types[-2:] == ["error", "message_stop"], types
    assert "did not finish" in frames[-2]["error"]["message"]


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_a_turn_nobody_aborted_still_finishes_normally(gs, handler, api):
    # The control: the same real engine and fake, nothing aborts, and the
    # answer is whole and labelled the way it always was.
    _real_engine(gs, None)
    path, body = BODIES[api]
    code, resp, _h = request(gs, handler, "POST", path, body)
    assert code == 200
    if api == "openai":
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["choices"][0]["message"]["content"] == "".join(CHUNKS)
    else:
        assert resp["stop_reason"] == "stop_sequence"
        assert resp["content"][0]["text"] == "".join(CHUNKS)
