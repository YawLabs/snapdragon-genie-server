#!/usr/bin/env python3
"""Smoke test for the Genie NPU server. Stdlib only. Exercises /v1/models,
non-streaming and streaming /v1/chat/completions, and prints timing.

Usage: python src/genie_smoke.py [BASE]
       python src/genie_smoke.py -h

BASE defaults to http://127.0.0.1:8123 -- the port run-genie-server.ps1 puts
the server on and where every other tool here looks. GENIE_PORT overrides the
port, for a server started bare on genie_server.py's own default of 8080. An
EMPTY GENIE_PORT is read as unset, as the server reads it, and one that is not
a port number is a note and 8123: unparsed it went into the URL and came back
out of urllib as `nonnumeric port`, forty lines of traceback over a value the
server this tool smokes accepts with a warning of its own. The default used to
be 8080 outright, which is also where run-llama-server.ps1 puts its CPU leg, so
with the documented two-engine stack up a bare run smoked llama-server: the
non-streaming half passed against the wrong engine, and the streaming half
then died in a TypeError, because llama-server opens every stream with a delta
of {"role": "assistant", "content": null} and the null went into the join.
Only a non-empty string is a content delta now -- the null, and the "" the
OpenAI API itself opens with, are neither text nor a first token -- so a run
against llama-server completes and is judged like any other. The model id on
every response is printed for the same reason the default moved: the operator
should see which engine answered, not infer it.

The exit status is the verdict. OK (0) needs each completion to carry content
and a finish_reason of stop or length; one that does not prints FAIL and exits
1. The streamed half is judged on its FRAMING too. An SSE event is dispatched
by a blank line, and the reader below keeps `data:` lines regardless, so it
cannot by itself tell a stream of frames from one frame that never ends: a
server writing one newline where the separator's two belong sends a stream no
SDK can read, and this tool -- the repo's verdict on whether a server is fit --
called it clean. Two `data:` lines with nothing between them are counted and
reported as their own FAIL, beside whatever the content verdict says. A 4xx/5xx prints the status and the server's error body -- the part that
says why; on a 500 that body is where the Genie exception text goes -- and
exits 1, instead of a urllib traceback with the body unread. A server that is
not there (connection refused, a timeout, a reset mid-stream), one that does
not speak HTTP at all, a response that stops mid-body, a 200 whose body is not
JSON, and a 200 of an unexpected SHAPE are FAIL lines naming the request as
well, not tracebacks. A refused connection adds the two ports a server here
is normally on, since the default moved.

A failure AFTER a stream's 200 has gone out cannot be a status, so genie_server
sends the 500's body as a frame of its own -- `data: {"error": {"message": ...,
"type": "server_error"}}`, with no `choices` -- and then [DONE]. That is a FAIL
naming the server's message. It used to arrive as a content delta reading
"\\n[error: ...]" with finish_reason "stop"; that text is still checked for,
only as a guard against a server old enough to send it.
"""
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

MODEL = "qwen3-4b-npu"
PROMPT = "What is gravity? Answer in one short sentence."


def default_base():
    """http://127.0.0.1:<GENIE_PORT>, or the launcher's 8123.

    `or`, not a .get() default: a GENIE_PORT that is set but EMPTY -- what a
    shell is left with when an unset var is "restored" by assigning "" to it
    -- is unset to genie_server (_int_env), and here it built the URL
    http://127.0.0.1: with no port at all.

    A value that is not a port number is a note and the default, the way the
    server degrades one (_int_env warns and uses its own default) and the way
    run-genie-server.ps1 does. Unparsed it went into the URL and came back out
    of urllib as http.client.InvalidURL('nonnumeric port: ...') -- an
    HTTPException, so not even one of the FAIL lines below: 40 lines of
    traceback from the tool, for a value the server it smokes had accepted.
    """
    raw = (os.environ.get("GENIE_PORT") or "").strip()
    port = 8123
    if raw:
        try:
            port = int(raw)
            if not 1 <= port <= 65535:
                raise ValueError(raw)
        except ValueError:
            print("note: GENIE_PORT='%s' is not a port number (1-65535); using 8123." % raw)
            port = 8123
    return "http://127.0.0.1:%d" % port


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        return json.load(r)


def post(base, path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)


def verdict(content, finish):
    """Why a completion is NOT a pass, or None when it is.

    Empty content is a failure the server can deliver as a 200 (README records
    one engine doing exactly that). Any finish other than stop/length means the
    generation did not run to a natural end or the cap.

    The "[error:" line check is a guard for an OLDER genie_server only. That
    one delivered a streamed engine failure as a content delta of
    "\\n[error: ...]" after whatever got out, with finish_reason "stop" --
    indistinguishable from success except by the text, which is why the text is
    checked line by line rather than only at the start. The current server
    sends a frame with an `error` object and no `choices` instead; main()
    reports that one itself and never brings it here.
    """
    content = content or ""
    if not content.strip():
        return "empty content"
    for line in content.splitlines():
        if line.strip().startswith("[error:"):
            return "engine error frame: %s" % line.strip()[:160]
    if finish not in ("stop", "length"):
        return "finish_reason=%r" % (finish,)
    return None


def ttft_label(first):
    """`first` is None when no content delta ever arrived; that is "never",
    not 0.00s."""
    return "n/a" if first is None else "%.2fs" % first


def nothing_listening(err):
    """True when the CONNECT was refused, i.e. nothing holds that port.

    urllib wraps everything up to and including sending the request in
    URLError, so a refused connect arrives as URLError(ConnectionRefusedError);
    one can also be handed in bare. Every other transport failure -- a timeout,
    a reset, a peer that hung up, something that is not an HTTP server -- came
    from a listener that DID take the connection, and "try this other port"
    would be the wrong advice for it.
    """
    return isinstance(err, ConnectionRefusedError) or isinstance(
        getattr(err, "reason", None), ConnectionRefusedError)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) > 1 and argv[1] in ("-h", "--help", "/?"):
        # `--help` was taken as BASE and reported as
        # "unknown url type: '--help/v1/models'", with the usage existing
        # nowhere but this file. The docstring IS the usage; -OO strips it.
        print((__doc__ or "Usage: python src/genie_smoke.py [BASE]").strip())
        return 0
    base = argv[1] if len(argv) > 1 else default_base()
    msgs = [{"role": "user", "content": PROMPT}]
    problems = []
    stage = "GET /v1/models"
    try:
        print("== GET /v1/models ==")
        models = get(base, "/v1/models")
        print(models)
        ids = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
        print("served:", ", ".join(str(i) for i in ids) or "?")

        stage = "non-streaming POST /v1/chat/completions"
        print("\n== non-streaming ==")
        t0 = time.time()
        r = post(base, "/v1/chat/completions",
                 {"model": MODEL, "messages": msgs, "max_tokens": 200})
        obj = json.load(r)
        dt = time.time() - t0
        content = obj["choices"][0]["message"].get("content") or ""
        finish = obj["choices"][0].get("finish_reason")
        print("model:", obj.get("model"))
        print("content:", content[:400])
        print("finish:", finish, "| wall %.1fs" % dt)
        bad = verdict(content, finish)
        if bad:
            problems.append("non-streaming: " + bad)

        stage = "streaming POST /v1/chat/completions"
        print("\n== streaming ==")
        t0 = time.time()
        first = None
        n = 0
        buf = []
        finish = None
        model = None
        stream_error = None
        r = post(base, "/v1/chat/completions",
                 {"model": MODEL, "messages": msgs, "max_tokens": 200, "stream": True})
        unterminated = 0
        prev_was_data = False
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                # A blank line is what DISPATCHES an SSE event, and keeping
                # `data:` lines regardless is blind to it: a server that wrote
                # one newline where the frame separator's two belong sends a
                # stream no SDK can read as more than a single frame that never
                # ends, and this tool -- whose exit status is the verdict on
                # whether the server is fit -- smoked it clean. Any other line
                # (a `: keep-alive` comment, an `event:`) ends the run of data
                # lines the same way a blank one does; two data lines with
                # nothing between them are the collapse.
                prev_was_data = False
                continue
            if prev_was_data:
                unterminated += 1
            prev_was_data = True
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if not isinstance(chunk, dict):
                continue
            # The engine failed after the 200 went out: the server sends the
            # 500's body as a frame with NO `choices`, then [DONE]. Indexing
            # it was a KeyError traceback where a FAIL belonged.
            if isinstance(chunk.get("error"), dict):
                stream_error = str(chunk["error"].get("message") or "unknown")
                break
            model = chunk.get("model", model)
            # A frame with no choice to read -- the include_usage frame's
            # `"choices": []` is the legitimate one -- says nothing about
            # content or finish.
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            d = choice.get("delta") or {}
            # Only a non-empty string is a content delta. llama-server opens
            # every stream with {"role": "assistant", "content": null}, and the
            # OpenAI API itself with "content": ""; `"content" in d` took both
            # for text, so TTFT was stamped at the role frame and the null went
            # into the join below -- a TypeError traceback, and from the one
            # other engine this tool is likely to be pointed at by mistake.
            text = d.get("content")
            if isinstance(text, str) and text:
                if first is None:
                    first = time.time() - t0
                n += 1
                buf.append(text)
        streamed = "".join(buf)
        print("model:", model)
        print("streamed:", streamed[:400])
        print("chunks:", n, "| TTFT", ttft_label(first), "| wall %.1fs" % (time.time() - t0))
        if unterminated:
            # Independent of the verdict below: the frames PARSED fine here, so
            # the content and finish_reason checks still say what the server
            # generated. What is broken is the framing, and it is broken for
            # every client that is not this reader.
            problems.append(
                "streaming: %d SSE frame(s) were not terminated by a blank "
                "line -- a client dispatches an event on that blank line, so "
                "it sees one frame that never ends" % unterminated)
        if stream_error is not None:
            # Not verdict(): the turn did not finish, so "empty content" or
            # "finish_reason=None" would be a second, vaguer description of
            # the failure the server has already named.
            problems.append("streaming: engine error frame: %s" % stream_error[:160])
        else:
            bad = verdict(streamed, finish)
            if bad:
                problems.append("streaming: " + bad)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        print("HTTP %d during %s: %s" % (e.code, stage, body))
        print("\nFAIL: HTTP %d during %s" % (e.code, stage))
        return 1
    except (OSError, ValueError, http.client.HTTPException) as e:
        # After HTTPError, which is a subclass of both URLError and OSError.
        # URLError (connection refused, a name that does not resolve), urlopen's
        # timeout and a reset mid-stream are all OSError; a 200 whose body is
        # not JSON -- something else on the port -- is ValueError. Each left
        # main() as a traceback, which still exits non-zero but buries which
        # request failed under forty lines of urllib.
        #
        # http.client.HTTPException is neither: it is how the client layer
        # reports a peer that is not speaking HTTP (BadStatusLine, e.g. an sshd
        # on the port) and a response that stops mid-body (IncompleteRead --
        # a Content-Length or a chunked stream cut short, which is what a
        # llama-server dying mid-generation looks like from here). Both were
        # tracebacks while this handler's docstring promised FAIL lines, and
        # bench_endpoint has caught them all along.
        print("\nFAIL: %s during %s: %s" % (type(e).__name__, stage, e))
        if nothing_listening(e):
            # Which port to try, in the line that says nothing answered: the
            # default moved from 8080 to 8123 with the launcher, so a bare
            # `python src/genie_server.py` is now somewhere this tool does not
            # look by default. bench_endpoint's equivalent names both ports.
            print("Nothing is listening at %s. run-genie-server.ps1 serves on 8123; "
                  "`python src/genie_server.py` directly serves on GENIE_PORT "
                  "(default 8080). Pass that base as the argument, or set "
                  "GENIE_PORT, to match." % base)
        return 1
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        # A 200 of the wrong SHAPE: a `[]` where the models object belongs, an
        # {"error": {...}} body with no `choices`, a `"choices": []` on a
        # non-streaming completion. Neither engine here sends those, so this is
        # a third server on the port or something in front of one -- the same
        # mis-aimed run the model id lines exist for, one step further out --
        # and each was a traceback from a tool whose status is the verdict.
        print("\nFAIL: unexpected response shape during %s: %s: %s"
              % (stage, type(e).__name__, e))
        return 1

    if problems:
        print("\nFAIL: " + "; ".join(problems))
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
