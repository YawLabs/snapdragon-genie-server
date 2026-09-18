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

Device-free, like everything beside it: the engine is conftest's StubEngine.
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
