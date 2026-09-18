"""Shared fixtures. Everything here is device-free -- no NPU, no bundle.

The Genie C API is deliberately NOT mocked. Today's session proved why: the
real API wanted `{"stop-sequence": [...]}` and `{"sampler": {...}}` while the
obvious shapes (a bare array, a bare object) were rejected or silently ignored.
A mock would have accepted the wrong shapes and made the suite agree with a
bug. Anything crossing that boundary belongs in a hardware-gated integration
test instead; these tests cover the pure logic around it.
"""

import importlib
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


class StubEngine:
    """Stands in for GenieEngine, recording what the server asked it to do.

    Token counts are len//4 so assertions can be exact arithmetic rather than
    "some plausible number".
    """

    def __init__(self, chunks=None, finish="stop"):
        self.chunks = list(chunks or [])
        # What the engine reports the turn ended on. "length" is the token cap
        # (CONTEXT_EXCEEDED, or the generation reaching max_tokens) and is the
        # one agentic clients key continuation off -- test_finish_reason.py
        # drives it through all four response paths.
        self.finish = finish
        self.calls = []          # one record per query / query_stream
        self.aborted = False
        self.encodes = 0         # native tokenizer calls, for the memo test
        self.yielded = 0         # chunks actually produced, for the abort test

    def count_tokens(self, text):
        self.encodes += 1
        return len(text) // 4

    def query(self, prompt, on_text, max_tokens=None, stop=None, sampler=None,
              commit=True, internal=False, result=None):
        self.calls.append({"prompt": prompt, "stop": stop, "sampler": sampler,
                           "commit": commit, "max_tokens": max_tokens,
                           "internal": internal})
        for c in self.chunks:
            self.yielded += 1
            on_text(c)
        # As the real one: whether anyone asked for the turn to stop. Nobody
        # aborts a stub, so a test that needs True sets `query_aborted`.
        if result is not None:
            result["aborted"] = getattr(self, "query_aborted", False)
        return self.finish

    def query_stream(self, prompt, res, max_tokens=None, stop=None, sampler=None,
                     commit=True, internal=False):
        # Same signature and the same record as query(): the real
        # GenieEngine.query_stream takes commit= and internal= too, and a stub
        # that is narrower than the thing it stands in for turns a handler
        # passing them into a TypeError only the suite can see.
        self.calls.append({"prompt": prompt, "stop": stop, "sampler": sampler,
                           "commit": commit, "max_tokens": max_tokens,
                           "internal": internal})
        for c in self.chunks:
            self.yielded += 1
            yield c
        res["finish"] = self.finish

    def signal_abort(self, any_turn=False, stalled=False):
        # The real one returns what the abort did (False when no turn was
        # there). The stub has no turns to scope to, so it records the ask
        # and says yes.
        self.aborted = True
        return True


class Wire:
    """Capture what the handler wrote. fail_after simulates a client that left."""

    def __init__(self, fail_after=None):
        self.chunks = []
        self.writes = 0
        self.fail_after = fail_after

    def write(self, b):
        self.writes += 1
        if self.fail_after is not None and self.writes > self.fail_after:
            raise ConnectionError("client gone")
        self.chunks.append(b)

    def flush(self):
        pass

    def text(self):
        return b"".join(self.chunks).decode("utf-8", "replace")

    def sse_frames(self):
        """Every `data:` payload, excluding the [DONE] sentinel.

        splitlines(), so this is structurally BLIND to the frame terminator:
        it cannot tell a stream of frames from one frame that never ends, and
        a server that wrote `\n` where `\n\n` belongs would leave every
        assertion built on this method green. Deliberately not fixed here --
        several tests cut a stream mid-frame on purpose (Wire(fail_after=...),
        the client that left), so a terminator assertion in this method would
        fire on a case the suite is testing. The framing is pinned once,
        against the emitted bytes, by the "bytes on the wire" section of
        test_api.py; read h.wfile.chunks directly for anything else about it.
        """
        out = []
        for line in self.text().splitlines():
            if line.startswith("data: "):
                body = line[6:].strip()
                if body and body != "[DONE]":
                    out.append(json.loads(body))
        return out


@pytest.fixture
def gs():
    """A freshly reloaded genie_server with module state pinned for tests."""
    import genie_server as g
    importlib.reload(g)
    # BEFORE load_chat_template, not after: that reader opens
    # BUNDLE_DIR/metadata.json, and the reload above has just re-read BUNDLE_DIR
    # from the developer's GENIE_BUNDLE_DIR. Every real bundle carries a
    # genie.chat_template, so with the documented export in place the "no
    # bundle" comment on the next line was simply false -- the suite rendered
    # with whatever template the ambient bundle held. "" is the no-bundle path
    # for every reader, this one included.
    g.BUNDLE_DIR = ""
    g.TEMPLATE = g.load_chat_template()   # no bundle -> standard ChatML fallback
    g._CONTEXT_SIZE = 4096                # pin, so no genie_config.json is read
    g._CONTEXT_LENGTHS = [512, 1024, 2048, 4096]   # a multi-length bundle
    g._POLL_MATCHES = [(False, "dialog.engine.backend.QnnHtp.poll")]
    # A correctly configured sampler, so bundle_config_warnings is quiet by
    # default. Pinned for the same reason as the two above: read_sampler opens
    # genie_config.json, and an unpinned fixture would read whatever bundle the
    # developer's GENIE_BUNDLE_DIR points at -- or nothing, and then every test
    # touching the warnings would carry a penalty warning it never asked for.
    g._CONFIG_PRESENT = True   # pinned like the readers below; see config_present
    # The values this repo RECOMMENDS, not Qualcomm's reference. The fixture only
    # needs sampler_penalty_state() == "ok", which both satisfy -- but it used to
    # carry 2.3/0.7/0.8, the exact setting bundle_config_warnings() calls out as
    # corrupting identifiers. A default fixture that models the configuration the
    # code argues against is a quiet contradiction for the next reader.
    g._SAMPLER = {"version": 1, "seed": 42, "temp": 0.8, "top-k": 40,
                  "top-p": 0.95,
                  "token-penalty": {"version": 1, "penalize-last-n": 128,
                                    "repetition-penalty": 1.15,
                                    "presence-penalty": 0.0,
                                    "frequency-penalty": 0.3}}
    g._TOK_CACHE.clear()
    g.STRIP_THINK = False
    # Pinned, not inherited. THINKING_DEFAULT is read from os.environ at import,
    # and this fixture reloads the module -- so a developer who exports
    # GENIE_THINKING=1 in their shell flipped it under the whole suite and two
    # tests failed for a reason that had nothing to do with the code. A suite
    # whose header promises it "runs anywhere" has to pin every ambient input,
    # not only the ones that needed pinning when it was written. The tests that
    # exercise the env var itself reload the module deliberately, which
    # overrides this.
    g.THINKING_DEFAULT = False
    # The same rule, applied to the knobs it had missed -- each one a
    # module-level os.environ read that the reload re-executes, each one the
    # documented default, and each one measured flipping tests when exported:
    # GENIE_ORPHAN_HOLD_CHARS=0 is a setting docs/GENIE_SERVER.md recommends to
    # a human reading the stream, and it failed the orphan-gate streaming tests;
    # GENIE_SUMMARIZE_EVICTED=0 failed every summarisation test;
    # GENIE_WINDOW_MARGIN moves every eviction budget; and the two token caps
    # feed the default max_tokens and summary_token_cap() arithmetic. This repo
    # has no CI, so a developer's shell is the only place the suite ever runs.
    #
    # NOT pinned, deliberately: HOST / PORT / MODEL_ID / MAX_BODY_BYTES /
    # FIXED_SEED and the supervision timeouts. Tests that depend on those
    # either assert against the module's own value or reload under
    # monkeypatch.setenv, and HEALTH / _INFLIGHT are BUILT from theirs at
    # import, so assigning the constant afterwards would pin a number the live
    # object no longer reads.
    g.ORPHAN_HOLD_CHARS = -1
    g.SUMMARIZE_EVICTED = True
    g.WINDOW_MARGIN = 64
    g.DEFAULT_MAX_TOKENS = 512
    g.SUMMARY_MAX_TOKENS = 192
    g.ENGINE = StubEngine()
    return g


@pytest.fixture
def handler(gs):
    """A Handler with no socket. Returns (make, wire_holder).

    Building one via object.__new__ skips BaseHTTPRequestHandler.__init__,
    which would try to serve a real connection.
    """
    def make(fail_after=None):
        h = object.__new__(gs.Handler)
        h.wfile = Wire(fail_after=fail_after)
        h.send_response = lambda *a, **k: None
        h.send_header = lambda *a, **k: None
        h.end_headers = lambda: None
        h.close_connection = False
        return h
    return make


def request(gs, make_handler, method, path, body=None, headers=None):
    """One request through a socketless Handler -> (code, parsed_body, handler).

    The ONE place a test sets path / headers / rfile and keeps the status code.
    The `handler` fixture builds the Handler and its Wire but throws the code
    away (its send_response is a no-op, which is right for the many tests that
    call _complete / _stream directly), so every file that needed the code grew
    its own copy of "override send_response, set the path, dispatch" -- four
    of them, differing only in whether they took bytes or a dict.

    `make_handler` is the `handler` fixture's value. `body` is raw BYTES (the
    only way to reach the JSON-parse failure), or anything else JSON-encoded,
    or None for no body at all. `headers` is merged OVER the computed
    Content-Length, so a test can lie about the length or add
    Transfer-Encoding. `parsed_body` is None when the response is not JSON (an
    SSE stream) -- read it off handler.wfile instead. `code` is None when the
    handler never called send_response.

    `gs` is taken, and unused, as test_api's local helpers took it: a caller
    that holds `handler` necessarily holds `gs`, and the call then reads in
    the same order as the test's fixture list.
    """
    h = make_handler()
    h.command = method
    h.path = path
    raw = b""
    if body is not None:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    h.headers = {"Content-Length": str(len(raw))} if body is not None else {}
    h.headers.update(headers or {})
    h.rfile = io.BytesIO(raw)
    sent = {}
    h.send_response = lambda code, *a, **k: sent.setdefault("code", code)
    getattr(h, "do_" + method)()
    try:
        parsed = json.loads(h.wfile.text())
    except ValueError:
        parsed = None
    return sent.get("code"), parsed, h


def convo(pairs, pad=60, system="You are a coding agent."):
    """A conversation of `pairs` user/assistant exchanges, padded to force eviction."""
    msgs = [{"role": "system", "content": system}]
    for i in range(pairs):
        msgs.append({"role": "user", "content": "q%d %s" % (i, "pad " * pad)})
        msgs.append({"role": "assistant", "content": "a%d %s" % (i, "fill " * pad)})
    msgs.append({"role": "user", "content": "final question"})
    return msgs
