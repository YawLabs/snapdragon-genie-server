"""Tests for the pure parts of the three server probes.

The probes themselves need two servers and a bundle. What is covered here is
everything that decides what a row SAYS: how the window is read and the three
overflow depths derive from it, how a response becomes a row, and the two
verdict lines that used to be reachable from failures -- "REPLAYS" for three
empty completions and "IGNORED" for a base run that never happened.

The module is imported, not run: its setup (tokenizer, window, argv) lives in
main() so that importing it has no side effects. That is also why the
tokenizer-based builder it uses has its own tests in test_prompt_depth.py.
The command line is covered too -- -h before the bundle is looked for, a BASE
that is not an http(s) URL refused by name, and a server that is not there
stopping the run once, with its URL, before any probe. The one subprocess run
is `--help`, and the one real socket is a loopback port nothing holds; every
other request is a stubbed urlopen, never whatever serves 8123 or 18181 on the
box running the suite.
"""

import http.client
import io
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

import probe_server_semantics as ps


# --- the window and the depths ----------------------------------------------

def _bundle(tmp_path, cfg):
    (tmp_path / "genie_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(tmp_path)


def test_the_window_is_read_from_the_bundle_config(tmp_path):
    b = _bundle(tmp_path, {"dialog": {"context": {"size": 4096}}})
    assert ps.read_window(b) == 4096


def test_a_missing_config_exits_naming_the_path(tmp_path):
    # No fallback: the depths derive from this number, and a default would
    # put back the 8192 literal this replaced.
    with pytest.raises(SystemExit) as e:
        ps.read_window(str(tmp_path))
    assert "genie_config.json" in str(e.value)
    assert str(tmp_path) in str(e.value)


def test_a_config_without_the_key_exits_naming_the_key(tmp_path):
    b = _bundle(tmp_path, {"dialog": {"engine": {}}})
    with pytest.raises(SystemExit) as e:
        ps.read_window(b)
    assert "dialog.context.size" in str(e.value)


def test_a_non_positive_window_is_refused(tmp_path):
    b = _bundle(tmp_path, {"dialog": {"context": {"size": 0}}})
    with pytest.raises(SystemExit):
        ps.read_window(b)


def test_the_depths_straddle_the_window():
    # One inside, one just past, one far past -- for ANY window, which is the
    # property the literal (7000, 9000, 20000) only had at 8192.
    for window in (4096, 8192, 16384):
        inside, past, far = ps.probe_depths(window)
        assert inside < window < past < far, window


def test_the_8192_depths_match_what_the_findings_table_was_measured_at():
    assert ps.probe_depths(8192) == (6963, 9011, 20480)


# --- one response, one row --------------------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, body=None, status=None):
    """urlopen that answers every request with `body` (dict -> 200) or raises
    HTTPError(status) carrying `body` as its text."""
    def urlopen(req, timeout=None):
        if status is not None:
            raise urllib.error.HTTPError(req.full_url, status, "err", {},
                                         io.BytesIO(body.encode()))
        return _Resp(json.dumps(body).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


ARGS = ("http://h", "m", "max_tokens", [{"role": "user", "content": "hi"}])


def test_a_good_response_becomes_a_text_row(monkeypatch):
    _serve(monkeypatch, {"choices": [{"message": {"content": "Saltwick"},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 12}})
    r = ps.ask(*ARGS)
    assert r["text"] == "Saltwick"
    assert r["finish"] == "stop"
    assert r["usage"] == {"prompt_tokens": 12}
    assert r["wall"] >= 0


def test_a_4xx_becomes_an_http_row_with_the_body(monkeypatch):
    _serve(monkeypatch, body='{"error": "prompt of 9011 tokens exceeds"}', status=400)
    r = ps.ask(*ARGS)
    assert r["http"] == 400
    assert "9011" in r["body"]
    assert "text" not in r


def test_a_200_that_is_not_openai_shaped_becomes_an_error_row(monkeypatch):
    # Valid JSON, no "choices". `d["choices"][0]` used to raise KeyError out
    # of ask() here and end the whole three-probe run.
    _serve(monkeypatch, {"error": {"message": "model not loaded"}})
    r = ps.ask(*ARGS)
    assert "error" in r
    assert "non-OpenAI" in r["error"]
    assert "model not loaded" in r["error"]


def test_an_empty_choices_list_is_an_error_row_not_an_index_error(monkeypatch):
    _serve(monkeypatch, {"choices": []})
    assert "error" in ps.ask(*ARGS)


@pytest.mark.parametrize("choice", [
    {"message": "Gravity pulls.", "finish_reason": "stop"},        # a bare string
    {"message": [{"type": "text", "text": "Gravity pulls."}]},     # Anthropic-style blocks
])
def test_a_choice_whose_message_is_not_an_object_is_an_error_row(monkeypatch, choice):
    # "Every failure is a row, never an exception: one bad answer costs one
    # line of output, not the two probes after it" (ask's docstring). This
    # shape reached `m.get("content")` and raised AttributeError out of ask(),
    # killing the run before PROBE 2 and PROBE 3 printed anything. The sibling
    # smoke test cans the identical body (test_genie_smoke.py) because a third
    # server on the port is the mis-aim these positionals exist for.
    _serve(monkeypatch, {"choices": [choice]})
    r = ps.ask(*ARGS)
    assert "error" in r, "an off-shape 200 is a row, not a traceback"
    assert "non-OpenAI" in r["error"]
    assert "Gravity pulls." in r["error"], "the row carries the body it refused"
    assert "text" not in r
    assert r["wall"] >= 0


@pytest.mark.parametrize("content", [
    [{"type": "text", "text": "Gravity pulls."}],   # Anthropic-style blocks
    {"type": "text", "text": "Gravity pulls."},     # one bare block
])
def test_a_message_whose_content_is_not_a_string_is_an_error_row(monkeypatch, content):
    # The same promise as the test above, one layer in: the message IS an
    # object, so `m.get` works, and `text` came back a LIST. Every consumer
    # here treats it as a string -- main() prints `r["text"][:90].replace(...)`
    # -- so the AttributeError landed outside ask(), after the row it was
    # supposed to be. A row, not "": three empty texts make seed_verdict
    # announce "EMPTY -- 3 of 3 completions had no content" about a server
    # that generated words, which is a wrong finding rather than a missing one.
    _serve(monkeypatch, {"choices": [{"message": {"content": content},
                                      "finish_reason": "stop"}]})
    r = ps.ask(*ARGS)
    assert "error" in r, "an off-shape 200 is a row, not a traceback"
    assert "non-OpenAI" in r["error"]
    assert "Gravity pulls." in r["error"], "the row carries the body it refused"
    assert "text" not in r


def test_every_text_row_carries_a_string(monkeypatch):
    # The property main() depends on, stated once: `text` is present only when
    # it is a str, so the slicing and .replace in the printers cannot raise.
    _serve(monkeypatch, {"choices": [{"message": {"content": "Saltwick"}}]})
    assert isinstance(ps.ask(*ARGS)["text"], str)


def test_a_null_content_is_still_an_empty_text_row(monkeypatch):
    # A 200 with no content is a real answer this probe reports on -- it is
    # what seed_verdict's EMPTY line is for -- so null stays a text row of "",
    # not an error row.
    _serve(monkeypatch, {"choices": [{"message": {"content": None},
                                      "finish_reason": "stop"}]})
    r = ps.ask(*ARGS)
    assert r["text"] == ""
    assert "error" not in r


def test_a_choices_object_is_an_error_row_not_a_key_error(monkeypatch):
    # A non-empty dict passes both "is it there" and "is it empty", so the
    # guard walked straight into choices[0] -> KeyError: 0.
    _serve(monkeypatch, {"choices": {"0": {"message": {"content": "x"}}}})
    r = ps.ask(*ARGS)
    assert "error" in r
    assert "non-OpenAI" in r["error"]


def test_one_off_shape_answer_does_not_cost_the_probes_after_it(monkeypatch):
    # The promise is about the RUN, not one call: three asks against the same
    # misshapen server all return, so PROBE 2 and PROBE 3 still get their say.
    _serve(monkeypatch, {"choices": [{"message": "Gravity pulls."}]})
    rows = [ps.ask(*ARGS) for _ in range(3)]
    assert all("error" in r for r in rows)
    assert ps.seed_verdict([r.get("text") for r in rows]) is None


def test_the_cap_key_and_extras_are_sent_as_given(monkeypatch):
    sent = {}

    def urlopen(req, timeout=None):
        sent.update(json.loads(req.data))
        return _Resp(json.dumps({"choices": [{"message": {"content": "x"}}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    ps.ask("http://h", "m", "max_completion_tokens", [], cap=7, extra={"stop": ["four"]})
    assert sent["max_completion_tokens"] == 7
    assert sent["stop"] == ["four"]
    assert "max_tokens" not in sent


# --- PROBE 1: the seed verdict ----------------------------------------------

def test_three_empty_completions_are_reported_as_empty_not_replays():
    v = ps.seed_verdict(["", "", ""])
    assert v.startswith("EMPTY")
    assert "REPLAYS" not in v


def test_a_single_empty_completion_also_blocks_the_verdict():
    assert ps.seed_verdict(["Saltwick", "  ", "Saltwick"]).startswith("EMPTY")


def test_identical_non_empty_completions_are_replays():
    assert "REPLAYS" in ps.seed_verdict(["Saltwick", "Saltwick", "Saltwick"])


def test_varying_completions_are_reseeds():
    assert "RE-SEEDS" in ps.seed_verdict(["Saltwick", "Brinehaven", "Saltwick"])


def test_a_failed_request_gives_no_verdict():
    assert ps.seed_verdict([None, "a", "a"]) is None


# --- PROBE 3: the stop verdict ----------------------------------------------

def test_a_failed_base_run_prints_its_failure_and_skips_the_verdict():
    # base failed (500), the stopped run came back empty. This used to print
    # `no stop : ''` and then IGNORED, comparing "" with "".
    lines = ps.stop_lines({"http": 500, "body": "engine wedged", "wall": 1.0},
                          {"text": "", "finish": "stop", "wall": 1.0})
    joined = "\n".join(lines)
    assert "HTTP 500" in joined
    assert "engine wedged" in joined
    assert "IGNORED" not in joined
    assert "NO VERDICT" in joined


def test_a_failed_stopped_run_prints_its_failure_and_skips_the_verdict():
    lines = ps.stop_lines({"text": "one, two, three, four", "finish": "stop"},
                          {"error": "TimeoutError: timed out"})
    joined = "\n".join(lines)
    assert "timed out" in joined
    assert "NO VERDICT" in joined


def test_stop_honoured():
    lines = ps.stop_lines({"text": "one, two, three, four, five", "finish": "stop"},
                          {"text": "one, two, three, ", "finish": "stop"})
    assert any("HONOURED" in ln for ln in lines)


def test_stop_ignored():
    same = "one, two, three, four, five"
    lines = ps.stop_lines({"text": same, "finish": "stop"}, {"text": same, "finish": "stop"})
    assert any("IGNORED" in ln for ln in lines)


def test_stop_unclear():
    lines = ps.stop_lines({"text": "one, two, three", "finish": "length"},
                          {"text": "one, two", "finish": "length"})
    assert any("UNCLEAR" in ln for ln in lines)


# --- the whole run: the window reaches the banner and the prompts -----------

class _WordTok:
    """One id per word -- enough for prompt_at to cut at a measured size."""

    def encode(self, text, add_special_tokens=False):
        return type("E", (), {"ids": text.split()})

    def decode(self, ids):
        return " ".join(ids)


def _run_main(monkeypatch, tmp_path, window, argv=("probe",), preflight=None):
    bundle = _bundle(tmp_path, {"dialog": {"context": {"size": window}}})
    monkeypatch.setenv("GENIE_BUNDLE_DIR", bundle)
    monkeypatch.setattr(ps, "load_tokenizer", lambda b: _WordTok())
    # Stubbed, never real: a real preflight would GET whatever holds 8123 or
    # 18181 on the box running the suite -- another session's server.
    monkeypatch.setattr(ps, "preflight", lambda base: preflight)
    calls = []

    def ask(base, model, cap_key, messages, **kw):
        calls.append((base, model, cap_key, messages, kw))
        return {"text": "ok", "finish": "stop", "usage": None, "wall": 0.0}
    monkeypatch.setattr(ps, "ask", ask)
    assert ps.main(list(argv)) == 0
    return calls


def test_probe_2_is_sized_and_labelled_from_the_served_bundle(monkeypatch, tmp_path, capsys):
    # The launcher's default 8B tier is a 4096 prebuilt. Against it the old
    # literals put all three rows past the window under a banner that still
    # said 8192; the banner and the prompts must both follow the bundle.
    calls = _run_main(monkeypatch, tmp_path, 4096)
    out = capsys.readouterr().out
    assert "past the 4096-token window" in out
    assert "8192" not in out
    # calls[3:6] are PROBE 2's; the filler is everything before the question.
    sizes = [len(c[3][0]["content"].split("Summarise")[0].split()) for c in calls[3:6]]
    assert sizes == [3481, 4505, 10240] == list(ps.probe_depths(4096))
    for n in sizes:
        assert "~%-6d tok" % n in out


def test_the_positionals_reach_every_request(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, 4096,
                      ("probe", "http://127.0.0.1:8123", "qwen3-4b-npu", "max_tokens"))
    assert len(calls) == 8, "3 seed runs + 3 depths + 2 stop runs"
    assert {c[:3] for c in calls} == {("http://127.0.0.1:8123", "qwen3-4b-npu", "max_tokens")}


def test_with_no_positionals_every_request_goes_to_the_geniex_defaults(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, 4096)
    assert {c[:3] for c in calls} == {(ps.DEFAULT_BASE, ps.DEFAULT_MODEL, ps.DEFAULT_CAP)}


# --- the defaults ---------------------------------------------------------------

def test_the_defaults_are_geniex_serve_as_the_docstring_says():
    assert ps.DEFAULT_BASE == "http://127.0.0.1:18181"
    assert ps.DEFAULT_MODEL == "qualcomm/qwen3-4b-ours"
    assert ps.DEFAULT_CAP == "max_completion_tokens"
    assert "[BASE] [MODEL] [CAP]" in ps.__doc__
    assert "8123" in ps.__doc__ and "qwen3-4b-npu" in ps.__doc__


# --- the command line: usage, refusals, and a server that is not there ------
# `--help` was taken as BASE: with GENIE_BUNDLE_DIR unset it printed the env
# line and exited 1, with it set it ended in a traceback. A dead port cost
# eight identical refused rows over ~20 s, the URL printed nowhere, exit 0.

def _no_bundle_reads(monkeypatch):
    """Unset the bundle and make any look at it fail the test loudly."""
    monkeypatch.delenv("GENIE_BUNDLE_DIR", raising=False)

    def boom(*a, **k):
        raise AssertionError("the bundle was read")
    monkeypatch.setattr(ps, "load_tokenizer", boom)
    monkeypatch.setattr(ps, "read_window", boom)
    monkeypatch.setattr(ps, "preflight", boom)
    monkeypatch.setattr(ps, "ask", boom)


@pytest.mark.parametrize("argv", [
    ["probe", "-h"], ["probe", "--help"], ["probe", "/?"],
    ["probe", "http://127.0.0.1:8123", "--help"],   # asked for after a BASE
])
def test_help_prints_the_usage_and_exits_0_before_the_bundle_is_looked_for(
        monkeypatch, capsys, argv):
    _no_bundle_reads(monkeypatch)
    assert ps.main(argv) == 0
    out = capsys.readouterr().out
    assert "[BASE] [MODEL] [CAP]" in out
    for default in (ps.DEFAULT_BASE, ps.DEFAULT_MODEL, ps.DEFAULT_CAP):
        assert default in out, "each positional's default is in the usage"


def test_help_works_from_the_command_line_with_no_bundle_set():
    # The real entry point, `if __name__ == "__main__"` included: nothing but
    # the usage is printed and the exit status is 0. Exits before any bundle
    # read and before any request, so it is safe on a shared box.
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "src", "probe_server_semantics.py")
    env = {k: v for k, v in os.environ.items() if k != "GENIE_BUNDLE_DIR"}
    r = subprocess.run([sys.executable, script, "--help"], capture_output=True,
                       text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "[BASE] [MODEL] [CAP]" in r.stdout
    assert "GENIE_BUNDLE_DIR to the bundle" not in r.stdout + r.stderr


@pytest.mark.parametrize("base", [
    "127.0.0.1:8123", "localhost:8123", "ftp://127.0.0.1:8123",
    "http://:8123", "http://127.0.0.1:80x",
])
def test_a_base_that_is_not_an_http_url_is_refused_by_name(monkeypatch, base):
    _no_bundle_reads(monkeypatch)
    with pytest.raises(SystemExit) as e:
        ps.main(["probe", base])
    assert repr(base) in str(e.value), "the refusal names what it was given"


def test_an_http_or_https_base_passes_the_check():
    for base in ("http://127.0.0.1:8123", "https://example.test", "HTTP://h:1"):
        assert ps.base_problem(base) is None, base


@pytest.mark.parametrize("argv, bad", [
    (["probe", "--model", "x"], "--model"),          # would have been BASE
    (["probe", "http://h", "--cap"], "--cap"),       # would have been sent as MODEL
])
def test_an_unknown_option_is_refused_not_sent_as_base_or_model(monkeypatch, argv, bad):
    _no_bundle_reads(monkeypatch)
    with pytest.raises(SystemExit) as e:
        ps.main(argv)
    assert "unknown option" in str(e.value) and bad in str(e.value)


def test_a_fourth_positional_is_refused_rather_than_dropped(monkeypatch):
    _no_bundle_reads(monkeypatch)
    with pytest.raises(SystemExit) as e:
        ps.main(["probe", "http://h", "m", "max_tokens", "extra"])
    assert "extra" in str(e.value) and "[BASE] [MODEL] [CAP]" in str(e.value)


def test_a_dead_server_stops_the_run_once_before_any_probe(monkeypatch, tmp_path, capsys):
    said = "cannot reach http://127.0.0.1:18181/v1/models (refused) -- nothing was probed"
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, tmp_path, 4096, preflight=said)
    assert str(e.value) == said, "one line, and a non-zero exit"
    assert "PROBE 1" not in capsys.readouterr().out, "no probe ran"


def test_the_output_names_the_server_it_probes(monkeypatch, tmp_path, capsys):
    _run_main(monkeypatch, tmp_path, 4096,
              ("probe", "http://127.0.0.1:8123", "qwen3-4b-npu", "max_tokens"))
    first = capsys.readouterr().out.splitlines()[0]
    assert "http://127.0.0.1:8123" in first and "qwen3-4b-npu" in first
    assert "max_tokens" in first


def _urlopen_raising(monkeypatch, exc):
    seen = []

    def urlopen(url, timeout=None):
        seen.append(url)
        raise exc
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


def test_preflight_names_the_url_and_the_usual_ports_on_a_refused_connect(monkeypatch):
    refused = ConnectionRefusedError(10061, "No connection could be made")
    seen = _urlopen_raising(monkeypatch, urllib.error.URLError(refused))
    said = ps.preflight("http://127.0.0.1:18181")
    assert seen == ["http://127.0.0.1:18181/v1/models"]
    assert "http://127.0.0.1:18181/v1/models" in said
    assert "Nothing is listening at http://127.0.0.1:18181" in said
    assert "8123" in said and "18181" in said


def test_preflight_stops_on_any_failure_to_connect_but_gives_port_advice_only_on_refusal(
        monkeypatch):
    _urlopen_raising(monkeypatch, urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")))
    said = ps.preflight("http://no-such-host.invalid:8123")
    assert "http://no-such-host.invalid:8123/v1/models" in said
    assert "getaddrinfo failed" in said
    assert "Nothing is listening" not in said


@pytest.mark.parametrize("exc", [
    urllib.error.HTTPError("http://h/v1/models", 404, "Not Found", {}, io.BytesIO(b"")),
    TimeoutError("timed out"),                      # took the connect, slow to answer
    ConnectionResetError(10054, "reset"),           # took it, then hung up
])
def test_preflight_lets_through_anything_that_took_the_connection(monkeypatch, exc):
    _urlopen_raising(monkeypatch, exc)
    assert ps.preflight("http://h") is None
    _urlopen_raising(monkeypatch, http.client.BadStatusLine("SSH-2.0"))
    assert ps.preflight("http://h") is None


def test_preflight_is_none_when_the_server_answers(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=None: _Resp(b'{"data": []}'))
    assert ps.preflight("http://h") is None


def test_preflight_against_a_closed_loopback_port_is_a_named_refusal():
    # A real connect, to a port that was just bound and released, so nothing
    # holds it: what urllib actually raises there is what the check keys on.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    base = "http://127.0.0.1:%d" % port
    said = ps.preflight(base, timeout=10)
    assert said is not None
    assert "%s/v1/models" % base in said
    assert "Nothing is listening at %s" % base in said


@pytest.mark.parametrize("base", [
    "--help",          # no colon at all: Request() itself raises
    "127.0.0.1:8123",  # parses a "127.0.0.1" scheme, so urlopen is what raises
])
def test_ask_turns_a_base_with_no_scheme_into_a_row_not_a_traceback(base):
    # The Request used to be built before the try, so `--help` taken as BASE
    # raised ValueError("unknown url type: '--help/v1/chat/completions'") out
    # of a function promising a row. Neither base reaches the network.
    r = ps.ask(base, "m", "max_tokens", [])
    assert "error" in r and "unknown url type" in r["error"]
    assert r["wall"] >= 0
