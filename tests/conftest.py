"""Shared fixtures. Everything here is device-free -- no NPU, no bundle.

The Genie C API is deliberately NOT mocked. Today's session proved why: the
real API wanted `{"stop-sequence": [...]}` and `{"sampler": {...}}` while the
obvious shapes (a bare array, a bare object) were rejected or silently ignored.
A mock would have accepted the wrong shapes and made the suite agree with a
bug. Anything crossing that boundary belongs in a hardware-gated integration
test instead; these tests cover the pure logic around it.
"""

import importlib
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
        self.finish = finish
        self.calls = []          # one record per query / query_stream
        self.aborted = False
        self.encodes = 0         # native tokenizer calls, for the memo test
        self.yielded = 0         # chunks actually produced, for the abort test

    def count_tokens(self, text):
        self.encodes += 1
        return len(text) // 4

    def query(self, prompt, on_text, max_tokens=None, stop=None, sampler=None,
              commit=True, internal=False):
        self.calls.append({"prompt": prompt, "stop": stop, "sampler": sampler,
                           "commit": commit, "max_tokens": max_tokens,
                           "internal": internal})
        for c in self.chunks:
            self.yielded += 1
            on_text(c)
        return self.finish

    def query_stream(self, prompt, res, max_tokens=None, stop=None, sampler=None):
        self.calls.append({"prompt": prompt, "stop": stop, "sampler": sampler,
                           "max_tokens": max_tokens})
        for c in self.chunks:
            self.yielded += 1
            yield c
        res["finish"] = self.finish

    def signal_abort(self):
        self.aborted = True


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
        """Every `data:` payload, excluding the [DONE] sentinel."""
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
    g.TEMPLATE = g.load_chat_template()   # no bundle -> standard ChatML fallback
    g._CONTEXT_SIZE = 4096                # pin, so no genie_config.json is read
    g._CONTEXT_LENGTHS = [512, 1024, 2048, 4096]   # a multi-length bundle
    g._POLL_MATCHES = [(False, "dialog.engine.backend.QnnHtp.poll")]
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


def convo(pairs, pad=60, system="You are a coding agent."):
    """A conversation of `pairs` user/assistant exchanges, padded to force eviction."""
    msgs = [{"role": "system", "content": system}]
    for i in range(pairs):
        msgs.append({"role": "user", "content": "q%d %s" % (i, "pad " * pad)})
        msgs.append({"role": "assistant", "content": "a%d %s" % (i, "fill " * pad)})
    msgs.append({"role": "user", "content": "final question"})
    return msgs
