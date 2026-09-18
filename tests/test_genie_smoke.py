"""Tests for the smoke test's verdict.

The smoke test is a hand-run tool and its exit status used to be meaningless:
"OK" printed after whatever came back. Now that the status IS the verdict,
the cases it must get right are covered here with the HTTP layer replaced by
canned responses -- the OpenAI shapes genie_server actually produces, including
the streamed "[error: ...]" frame it uses for an engine failure, and the stream
llama-server actually opens with, since that is the engine a mis-aimed smoke
lands on.

The transport failures are canned as the exception instances urllib and
http.client raise, each one reproduced first against a real loopback socket
(see the comments where they are listed). Device-free and socket-free: nothing
here starts a server or opens a port.
"""

import http.client
import io
import json
import urllib.error
import urllib.request

import pytest

import genie_smoke as gs


# --- the pure verdict ---------------------------------------------------------

def test_a_normal_completion_passes():
    assert gs.verdict("Gravity pulls masses together.", "stop") is None
    assert gs.verdict("Gravity pulls masses together and", "length") is None


def test_empty_content_fails():
    assert gs.verdict("", "stop") == "empty content"
    assert gs.verdict("   \n", "stop") == "empty content"
    assert gs.verdict(None, "stop") == "empty content"


def test_the_engine_error_frame_fails_even_after_partial_content():
    # genie_server appends "\n[error: ...]" after whatever got out, with
    # finish_reason "stop" -- the text is the only signal.
    v = gs.verdict("Gravity is\n[error: QNN_COMMON_ERROR_SYSTEM 1003]", "stop")
    assert v.startswith("engine error frame")
    assert "1003" in v


def test_an_error_frame_at_the_start_fails_too():
    assert gs.verdict("[error: engine failing]", "stop").startswith("engine error frame")


def test_an_unexpected_finish_fails():
    assert gs.verdict("fine", None) == "finish_reason=None"
    assert gs.verdict("fine", "error") == "finish_reason='error'"


def test_ttft_is_never_when_no_content_arrived():
    assert gs.ttft_label(None) == "n/a"
    assert gs.ttft_label(0.256) == "0.26s"


# --- the default target -----------------------------------------------------------

def test_the_default_base_is_the_launcher_port(monkeypatch):
    monkeypatch.delenv("GENIE_PORT", raising=False)
    assert gs.default_base() == "http://127.0.0.1:8123"


def test_genie_port_overrides_the_default(monkeypatch):
    monkeypatch.setenv("GENIE_PORT", "8080")
    assert gs.default_base() == "http://127.0.0.1:8080"


def test_an_empty_genie_port_is_unset_not_a_url_with_no_port(monkeypatch):
    # Set-but-empty is what a shell is left with when something "restores" an
    # unset var by assigning "" to it. genie_server's own _int_env reads that
    # as unset; the smoke built http://127.0.0.1: from it.
    monkeypatch.setenv("GENIE_PORT", "")
    assert gs.default_base() == "http://127.0.0.1:8123"


@pytest.mark.parametrize("raw", ["808O", "abc", "0", "70000", "-1", "80.80"])
def test_a_genie_port_that_is_not_a_port_is_a_note_and_the_default(monkeypatch, capsys, raw):
    # The server accepts the same value: _int_env warns and serves on its own
    # default. Here it went into the URL unparsed and came back out of urllib
    # as http.client.InvalidURL('nonnumeric port: ...') -- an HTTPException, so
    # not even one of main()'s FAIL lines, just 40 lines of traceback.
    monkeypatch.setenv("GENIE_PORT", raw)
    assert gs.default_base() == "http://127.0.0.1:8123"
    assert ("note: GENIE_PORT='%s' is not a port number (1-65535); using 8123." % raw
            in capsys.readouterr().out)


def test_a_padded_genie_port_is_still_that_port(monkeypatch, capsys):
    # Whitespace is what a shell leaves in an exported value; the launcher's
    # [int]::TryParse takes it too.
    monkeypatch.setenv("GENIE_PORT", "  8080  ")
    assert gs.default_base() == "http://127.0.0.1:8080"
    assert capsys.readouterr().out == "", "a readable port is not worth a line"


def test_a_bad_genie_port_never_reaches_urllib(monkeypatch, capsys):
    monkeypatch.setenv("GENIE_PORT", "abc")
    seen = []

    def urlopen(req, timeout=None):
        seen.append(req if isinstance(req, str) else req.full_url)
        raise urllib.error.HTTPError(seen[-1], 503, "down", {}, io.BytesIO(b""))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert gs.main(["smoke"]) == 1
    assert seen == ["http://127.0.0.1:8123/v1/models"]
    assert "note: GENIE_PORT='abc'" in capsys.readouterr().out


# --- the flags --------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["-h", "--help", "/?"])
def test_help_prints_the_usage_and_asks_the_network_nothing(monkeypatch, capsys, flag):
    # `--help` was read as the BASE, so it came back as "FAIL: ValueError
    # during GET /v1/models: unknown url type: '--help/v1/models'" -- from a
    # tool whose usage text existed nowhere but its own source.
    def urlopen(req, timeout=None):
        raise AssertionError("--help went to the network: %r" % (req,))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert gs.main(["smoke", flag]) == 0
    out = capsys.readouterr().out
    assert "Usage: python src/genie_smoke.py" in out
    assert "GENIE_PORT" in out, "the one knob, in the usage"
    assert "== GET /v1/models ==" not in out, "no request was made"


# --- the whole run against canned responses ---------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse(chunks):
    return "".join("data: %s\n\n" % json.dumps(c) for c in chunks) + "data: [DONE]\n\n"


def _chunk(delta, finish=None, model="qwen3-4b-npu"):
    return {"id": "c1", "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _serve(monkeypatch, models=None, completion=None, stream=None, fail=None,
           raw_stream=None):
    """urlopen answering the three requests the smoke makes, in the shapes
    genie_server sends. `fail` = (code, body) raises HTTPError on the first
    POST.

    `is not None` rather than `or` on all three, because a fixture is allowed
    to be falsy: `[]` for /v1/models and `{"choices": []}` for a completion are
    the off-shape 200s a third server on the port can answer with.
    """
    models = models if models is not None else {
        "object": "list",
        "data": [{"id": "qwen3-4b-npu", "object": "model",
                  "owned_by": "qualcomm-genie-npu"}]}
    completion = completion if completion is not None else {
        "id": "c0", "object": "chat.completion", "model": "qwen3-4b-npu",
        "choices": [{"index": 0, "message": {"role": "assistant",
                                             "content": "Gravity pulls masses together."},
                     "finish_reason": "stop"}]}
    stream = stream if stream is not None else [
        _chunk({"role": "assistant"}), _chunk({"content": "Gravity "}),
        _chunk({"content": "pulls."}), _chunk({}, finish="stop")]

    def urlopen(req, timeout=None):
        if isinstance(req, str):
            assert req.endswith("/v1/models")
            return _Resp(json.dumps(models).encode())
        if fail is not None:
            raise urllib.error.HTTPError(req.full_url, fail[0], "err", {},
                                         io.BytesIO(fail[1].encode()))
        body = json.loads(req.data)
        if body.get("stream"):
            # raw_stream is the escape hatch for bodies _sse cannot build,
            # which so far means one that is malformed on purpose.
            return _Resp((raw_stream if raw_stream is not None
                          else _sse(stream)).encode())
        return _Resp(json.dumps(completion).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_a_stream_whose_frames_are_never_terminated_is_a_fail(monkeypatch, capsys):
    # One newline where the SSE frame separator's two belong. This reader
    # keeps `data:` lines and ignores the terminator, so it parsed every frame
    # and reported a clean pass for a stream no SDK can dispatch more than
    # once -- from the tool whose exit status is this repo's verdict on
    # whether a server is fit to use. genie_server's own emission is pinned
    # separately (test_api.py, the bytes-on-the-wire section); this is the
    # instrument that would have to notice.
    chunks = [_chunk({"role": "assistant"}), _chunk({"content": "Gravity "}),
              _chunk({"content": "pulls."}), _chunk({}, finish="stop")]
    collapsed = ("".join("data: %s\n" % json.dumps(c) for c in chunks)
                 + "data: [DONE]\n")
    _serve(monkeypatch, raw_stream=collapsed)
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "not terminated by a blank line" in out
    # The framing is the complaint, not the content: what the server generated
    # still parsed and is still reported, so the operator can see both.
    assert "streamed: Gravity pulls." in out


def test_a_keep_alive_comment_between_frames_is_not_a_missing_terminator(monkeypatch, capsys):
    # `: keep-alive` is its own frame and ends the run of data lines exactly
    # as a blank line does. Counting any two data lines without a blank
    # between them would have failed a perfectly good stream here.
    parts = ["data: %s\n\n" % json.dumps(_chunk({"content": "Gravity "})),
             ": keep-alive\n\n",
             "data: %s\n\n" % json.dumps(_chunk({}, finish="stop")),
             "data: [DONE]\n\n"]
    _serve(monkeypatch, raw_stream="".join(parts))
    assert gs.main(["smoke"]) == 0
    assert "not terminated" not in capsys.readouterr().out


def test_a_healthy_server_is_ok_and_names_its_model(monkeypatch, capsys):
    _serve(monkeypatch)
    assert gs.main(["smoke"]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("OK")
    assert "served: qwen3-4b-npu" in out
    assert out.count("model: qwen3-4b-npu") == 2, "both completions name the engine"
    assert "TTFT 0." in out


# How llama-server opens EVERY chat stream (the fork's server-task.cpp:
# add_delta({"role": "assistant", "content": nullptr})). genie_server's role
# frame has no content key at all, so a fixture without this one is not
# llama-server, whatever model id it carries.
_LLAMA_ROLE_FRAME = {"role": "assistant", "content": None}


def test_the_other_engine_is_visible_on_every_line_that_names_a_model(monkeypatch, capsys):
    # The mix-up this guards against: llama-server on the port the smoke was
    # pointed at. It answers every check, so the id is what gives it away --
    # PROVIDED the run gets as far as printing it. The null content in the
    # role frame used to go into the join as None: a TypeError traceback after
    # the non-streaming half, so the streaming `model:` line never printed.
    llama = "unsloth/Qwen3.5-9B-GGUF:Q4_0"
    _serve(monkeypatch,
           models={"object": "list", "data": [{"id": llama, "object": "model"}]},
           completion={"model": llama,
                       "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]},
           stream=[_chunk(_LLAMA_ROLE_FRAME, model=llama), _chunk({"content": "x"}, model=llama),
                   _chunk({}, finish="stop", model=llama)])
    assert gs.main(["smoke"]) == 0
    out = capsys.readouterr().out
    assert "served: " + llama in out
    assert out.count("model: " + llama) == 2
    assert "streamed: x" in out
    assert "chunks: 1 |" in out, "the role frame is not a content chunk"
    # The id the smoke ASKED for must never be printed as the id that answered.
    assert "qwen3-4b-npu" not in out


@pytest.mark.parametrize("opening", [
    _LLAMA_ROLE_FRAME,                          # llama-server
    {"role": "assistant", "content": ""},       # the OpenAI API itself
])
def test_a_role_frame_with_no_text_is_not_the_first_token(monkeypatch, capsys, opening):
    # `"content" in d` is true for null and for "", so TTFT was stamped at the
    # role frame -- time to the response headers, in effect -- and the frame
    # counted as a chunk. With nothing after it, this stream never produced a
    # token: "n/a" and 0, and the verdict is the ordinary empty-content FAIL.
    _serve(monkeypatch, stream=[_chunk(opening), _chunk({}, finish="stop")])
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "chunks: 0 | TTFT n/a" in out
    assert "FAIL: streaming: empty content" in out


_ERROR_FRAME = {"error": {"message": "QNN 1003", "type": "server_error"}}


@pytest.mark.parametrize("before", [
    [],                                         # failed before the first token
    [{"role": "assistant"}, {"content": "Gravity"}],    # ... or after some
])
def test_a_streamed_engine_error_frame_fails_the_run(monkeypatch, capsys, before):
    # What genie_server sends once the 200 has gone out: the 500's body as a
    # data frame -- NO `choices` key -- and then [DONE], with no finish frame
    # because the turn did not finish. Indexing chunk["choices"] on it was a
    # KeyError traceback from a tool whose exit status is supposed to be the
    # verdict.
    _serve(monkeypatch, stream=[*[_chunk(d) for d in before], _ERROR_FRAME])
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "\nOK" not in out
    assert "FAIL: streaming: engine error frame: QNN 1003" in out
    # The failure is the verdict; what got out before it is not judged as well.
    assert "empty content" not in out and "finish_reason" not in out
    if before:
        assert "streamed: Gravity" in out, "what did arrive is still shown"


def test_an_error_frame_without_a_message_still_fails_the_run(monkeypatch, capsys):
    _serve(monkeypatch, stream=[{"error": {"type": "server_error"}}])
    assert gs.main(["smoke"]) == 1
    assert "FAIL: streaming: engine error frame: unknown" in capsys.readouterr().out


def test_a_usage_frame_with_no_choices_is_skipped_not_indexed(monkeypatch, capsys):
    # stream_options.include_usage ends a stream with `"choices": []`. The
    # smoke does not ask for it, but a server is free to send one, and an
    # empty list is an IndexError where a missing key was a KeyError.
    usage = {"id": "c1", "object": "chat.completion.chunk", "model": "qwen3-4b-npu",
             "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2,
                                      "total_tokens": 11}}
    _serve(monkeypatch, stream=[_chunk({"content": "Gravity pulls."}),
                                _chunk({}, finish="stop"), usage])
    assert gs.main(["smoke"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("OK")


def test_the_legacy_error_content_delta_still_fails_the_run(monkeypatch, capsys):
    # The OLD server shape, kept as a guard for one that predates the error
    # frame above: the failure arrived as a CONTENT delta reading
    # "\n[error: ...]" with finish_reason "stop", so the text was the only
    # signal. genie_server no longer sends this.
    _serve(monkeypatch, stream=[_chunk({"content": "Gravity"}),
                                _chunk({"content": "\n[error: QNN 1003]"}),
                                _chunk({}, finish="stop")])
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "\nOK" not in out
    assert "FAIL: streaming: engine error frame" in out


def test_an_empty_non_streaming_completion_fails_the_run(monkeypatch, capsys):
    _serve(monkeypatch, completion={"model": "qwen3-4b-npu",
                                    "choices": [{"message": {"content": None},
                                                 "finish_reason": "stop"}]})
    assert gs.main(["smoke"]) == 1
    assert "FAIL: non-streaming: empty content" in capsys.readouterr().out


def test_a_stream_with_no_content_reports_ttft_as_never(monkeypatch, capsys):
    _serve(monkeypatch, stream=[_chunk({"role": "assistant"}), _chunk({}, finish="stop")])
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "chunks: 0 | TTFT n/a" in out
    assert "TTFT 0.00s" not in out


def test_an_http_error_prints_the_body_and_exits_non_zero(monkeypatch, capsys):
    # The 500 body is where the Genie exception text goes; a urllib
    # traceback showed the status and threw the body away.
    _serve(monkeypatch, fail=(500, '{"error": {"message": "GenieDialog_query failed 4"}}'))
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "HTTP 500" in out
    assert "GenieDialog_query failed 4" in out
    assert "non-streaming" in out, "must say which request it was"
    assert "\nOK" not in out


def _refused():
    return urllib.error.URLError(ConnectionRefusedError(10061, "actively refused"))


@pytest.mark.parametrize("fails_on, error, stage, named", [
    # Nothing is listening: the commonest way to run a smoke test wrong.
    ("models", _refused, "GET /v1/models", "URLError"),
    # The server went away between the two completions ...
    ("stream", _refused, "streaming POST /v1/chat/completions", "URLError"),
    # ... or stopped answering: urlopen's timeout is a bare OSError subclass,
    # not a URLError, so catching URLError alone would still traceback here.
    ("stream", lambda: TimeoutError("timed out"), "streaming POST /v1/chat/completions",
     "TimeoutError"),
])
def test_an_unreachable_server_is_a_fail_line_not_a_traceback(monkeypatch, capsys, fails_on,
                                                              error, stage, named):
    # Only HTTPError was caught, so "connection refused" left main() as a
    # urllib traceback -- from the tool whose docstring says its failures are
    # FAIL lines. The exit status was already non-zero; what was missing is
    # the line that says which request, in place of forty lines of urllib.
    _serve(monkeypatch)
    answering = urllib.request.urlopen

    def urlopen(req, timeout=None):
        if isinstance(req, str):
            asked = "models"
        else:
            asked = "stream" if json.loads(req.data).get("stream") else "completion"
        if asked == fails_on:
            raise error()
        return answering(req, timeout=timeout)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: %s during %s" % (named, stage) in out
    assert "\nOK" not in out


def test_the_refused_fail_names_the_launcher_port_and_the_bare_one(monkeypatch, capsys):
    # The default moved from 8080 to 8123 with the launcher, so a server run
    # bare (`python src/genie_server.py`, still on GENIE_PORT, default 8080) is
    # now somewhere this tool does not look -- and the FAIL line named no port
    # at all, not even the base it had tried. bench_endpoint names both.
    monkeypatch.delenv("GENIE_PORT", raising=False)

    def urlopen(req, timeout=None):
        raise _refused()
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: URLError during GET /v1/models" in out
    assert "Nothing is listening at http://127.0.0.1:8123" in out
    assert "run-genie-server.ps1 serves on 8123" in out
    assert "GENIE_PORT" in out and "8080" in out, "and where a bare server is"


def test_nothing_listening_is_a_refused_connect_and_nothing_else():
    # "Nothing is listening ... try this other port" is the wrong advice for a
    # server that DID take the connection and then timed out, reset or spoke
    # something that is not HTTP: there is one on that port already.
    assert gs.nothing_listening(_refused()), "urllib wraps the connect phase"
    assert gs.nothing_listening(ConnectionRefusedError(10061, "actively refused"))
    for err in (TimeoutError("timed out"),
                ConnectionResetError(10054, "An existing connection was forcibly closed"),
                urllib.error.URLError(TimeoutError("timed out")),
                http.client.BadStatusLine("SSH-2.0-OpenSSH_9.5"),
                http.client.IncompleteRead(b"x" * 13, 487),
                ValueError("unknown url type: '127.0.0.1:8123'")):
        assert not gs.nothing_listening(err), err


# Neither an OSError nor a ValueError, so `except (OSError, ValueError)` let
# both out as tracebacks. Each was reproduced against a real loopback socket:
# a listener answering "SSH-2.0-OpenSSH_9.5" (BadStatusLine), a
# `Content-Length: 500` followed by 13 bytes and a FIN, and a chunked SSE
# stream cut mid-chunk (IncompleteRead both) -- which is what the llama-server
# this tool is likely to be pointed at by mistake looks like from here when it
# dies mid-generation.
@pytest.mark.parametrize("error, named", [
    (lambda: http.client.BadStatusLine("SSH-2.0-OpenSSH_9.5"), "BadStatusLine"),
    (lambda: http.client.IncompleteRead(b"x" * 13, 487), "IncompleteRead"),
])
@pytest.mark.parametrize("fails_on, stage", [
    ("models", "GET /v1/models"),
    ("stream", "streaming POST /v1/chat/completions"),
])
def test_a_peer_that_is_not_speaking_http_is_a_fail_line_not_a_traceback(
        monkeypatch, capsys, fails_on, stage, error, named):
    _serve(monkeypatch)
    answering = urllib.request.urlopen

    def urlopen(req, timeout=None):
        if isinstance(req, str):
            asked = "models"
        else:
            asked = "stream" if json.loads(req.data).get("stream") else "completion"
        if asked == fails_on:
            raise error()
        return answering(req, timeout=timeout)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: %s during %s" % (named, stage) in out
    assert "\nOK" not in out
    # It answered, badly: this is not the port-advice case.
    assert "Nothing is listening" not in out


class _CutStream:
    """A response body that stops part-way: it hands back the frames it has and
    then raises, the way a chunked SSE stream cut mid-chunk arrives out of
    http.client. The canned failures above all raise from urlopen, i.e. before
    a frame exists; _Resp holds the WHOLE body, so neither can cut a stream
    that has already started delivering."""

    def __init__(self, text, error):
        self.text = text
        self.error = error

    def __iter__(self):
        yield from self.text.encode().splitlines(keepends=True)
        raise self.error


def _cut_after_two_frames(monkeypatch, error):
    """The healthy server, except its stream dies after two content frames."""
    _serve(monkeypatch)
    answering = urllib.request.urlopen

    def urlopen(req, timeout=None):
        if not isinstance(req, str) and json.loads(req.data).get("stream"):
            return _CutStream("".join("data: %s\n\n" % json.dumps(c)
                                      for c in [_chunk({"content": "Gravity "}),
                                                _chunk({"content": "pulls"})]),
                              error)
        return answering(req, timeout=timeout)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


@pytest.mark.parametrize("error, named", [
    # llama-server/cpp-httplib chunking, cut mid-chunk.
    (lambda: http.client.IncompleteRead(b"data: ", 42), "IncompleteRead"),
    # The server killed outright -- Stop-Process -Force in run-llama-server.ps1,
    # taskkill /T /F in bench_servers.py -- so the peer RSTs the open body.
    (lambda: ConnectionResetError(10054, "An existing connection was forcibly closed"),
     "ConnectionResetError"),
])
def test_a_stream_cut_after_it_started_is_a_fail_line_not_a_traceback(
        monkeypatch, capsys, error, named):
    # An engine that dies after the 200 has gone out is this box's documented
    # failure (the QnnHtp.dll fault the supervise loop exists for). The verdict
    # has to survive it: exit 1, the exception and the request named, no "OK".
    _cut_after_two_frames(monkeypatch, error())
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: %s during streaming POST /v1/chat/completions" % named in out
    assert "\nOK" not in out
    assert "Nothing is listening" not in out, "it answered, then stopped"


def test_a_stream_cut_mid_body_drops_what_had_already_arrived(monkeypatch, capsys):
    # Pinning today's behaviour, not blessing it: the streamed text, the chunk
    # count and the TTFT are printed after the loop, so a cut stream loses all
    # three and the FAIL line's exception name is the whole diagnostic -- even
    # though two content frames did arrive and are exactly what an operator
    # chasing a mid-generation death came for. Changing that is a deliberate
    # edit to this test, not a silent one.
    _cut_after_two_frames(monkeypatch, http.client.IncompleteRead(b"data: ", 42))
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "== streaming ==" in out, "it got as far as opening the stream"
    assert "streamed: Gravity pulls" not in out
    assert "chunks: 2" not in out
    assert "TTFT" not in out.split("== streaming ==")[1]


@pytest.mark.parametrize("fixture, stage, named", [
    # /v1/models answering a bare list: `.get("data")` on it is an
    # AttributeError.
    ({"models": []}, "GET /v1/models", "AttributeError"),
    # A 200 carrying the error object genie_server sends WITH a 500 -- a proxy
    # that rewrote the status, say: no `choices` key at all.
    ({"completion": {"error": {"message": "upstream is down"}}},
     "non-streaming POST /v1/chat/completions", "KeyError"),
    # `"choices": []` where the answer belongs: a missing key was a KeyError,
    # an empty list is an IndexError.
    ({"completion": {"model": "qwen3-4b-npu", "choices": []}},
     "non-streaming POST /v1/chat/completions", "IndexError"),
    # A choice whose `message` is a string, not an object: .get() on it is an
    # AttributeError.
    ({"completion": {"model": "qwen3-4b-npu",
                     "choices": [{"message": "Gravity pulls.", "finish_reason": "stop"}]}},
     "non-streaming POST /v1/chat/completions", "AttributeError"),
])
def test_a_200_of_an_unexpected_shape_is_a_fail_line_not_a_traceback(
        monkeypatch, capsys, fixture, stage, named):
    # Neither engine here sends these, so it is a third server on the port or
    # something in front of one -- the mis-aimed run the model id lines exist
    # for, one step further out than a 200 of HTML. Each was a traceback from
    # the tool whose exit status is supposed to be the verdict.
    _serve(monkeypatch, **fixture)
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: unexpected response shape during %s: %s" % (stage, named) in out
    assert "\nOK" not in out


def test_a_body_that_is_not_json_is_a_fail_line_not_a_traceback(monkeypatch, capsys):
    # Something else on the port answering 200 with a page of HTML: the same
    # mis-aimed run the model id lines exist for, one step further from an
    # OpenAI server.
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b"<html>It works!</html>"))
    assert gs.main(["smoke"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: JSONDecodeError during GET /v1/models" in out
    assert "\nOK" not in out


def test_a_bad_finish_reason_fails_the_run(monkeypatch, capsys):
    _serve(monkeypatch, stream=[_chunk({"content": "x"}), _chunk({}, finish="error")])
    assert gs.main(["smoke"]) == 1
    assert "finish_reason='error'" in capsys.readouterr().out


def test_the_base_argument_wins_over_the_default(monkeypatch):
    seen = []

    def urlopen(req, timeout=None):
        seen.append(req if isinstance(req, str) else req.full_url)
        raise urllib.error.HTTPError(seen[-1], 503, "down", {}, io.BytesIO(b""))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    gs.main(["smoke", "http://10.0.0.5:9999"])
    assert seen == ["http://10.0.0.5:9999/v1/models"]
