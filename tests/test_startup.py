"""What the server decides before it serves a request, run without a device.

Startup is where this server has been quietly wrong the most often -- a port
another process was still holding, a bundle configured to busy-wait, an arch
the box could not drive -- and each fix arrived with a startup line that says
so. This file pins the ones that were still untested: how a GENIE_* variable
that will not parse is handled (every reader, not the two that had a test),
that the bind happens BEFORE the 11-35s model load and can name an address
the machine does not have -- while the LISTEN waits until the model is
resident, so nothing connects to a server that may yet fail to load -- that
shutdown closes the engine rather than only freeing its dialog, that the IPv6
loopback the server lists as recognised can actually be bound, what
probe_tool_support decides from the bundle's files, that --help and a stray
argument are answered before anything is touched, and the Python side of
load_engine itself: the named exit for every setup it cannot use (an unset
variable, a directory that is not a bundle, a file the config names and the
bundle lacks, a Genie.dll that will not load and why), what a failed load
blames, the line a slow load prints, the tokenizer warning and the sampler
baseline, driven through a stand-in for Genie.dll that returns scripted
statuses.

Device-free. The SDK and the bundle are a handful of files under tmp_path,
and the DLL is a recorder -- nothing here asserts what Genie wants, only what
this server does with what it gets back.
"""

import importlib
import json
import os
import re
import socket

import pytest


# --- every config reader degrades instead of refusing to boot -------------

@pytest.fixture
def bare_env(monkeypatch):
    """No GENIE_* variable in the environment but the one the test sets.

    The reader tests below reload the module and then read what it PRINTED,
    and a reload re-reads EVERY GENIE_* variable, not only the one under test.
    With a developer's own exports in place (GENIE_PORT=808O,
    GENIE_MAX_TOKENS=0, GENIE_WINDOW_MARGIN=-5 -- each one a documented
    degradation) every other reader's WARNING landed in the same capture: the
    quiet tests failed on a line about a different variable, and the loud ones
    could pass on one ("808O" in out was satisfied by GENIE_PORT while the test
    was about GENIE_MAX_TOKENS). Scrubbed, so each capture is about one
    variable. The `gs` fixture's pins cannot do this: these tests reload past
    them on purpose.
    """
    for name in list(os.environ):
        if name.upper().startswith("GENIE_"):
            monkeypatch.delenv(name)


READERS = [
    ("GENIE_PORT", "PORT", 8080),
    ("GENIE_MAX_TOKENS", "DEFAULT_MAX_TOKENS", 512),
    ("GENIE_MAX_BODY_BYTES", "MAX_BODY_BYTES", 8 * 1024 * 1024),
    ("GENIE_SOCKET_TIMEOUT", "SOCKET_TIMEOUT_S", 120),
    ("GENIE_WINDOW_MARGIN", "WINDOW_MARGIN", 64),
    ("GENIE_SUMMARY_MAX_TOKENS", "SUMMARY_MAX_TOKENS", 192),
    ("GENIE_FIRST_TOKEN_TIMEOUT", "FIRST_TOKEN_TIMEOUT_S", 300.0),
    ("GENIE_STALL_TIMEOUT", "STALL_TIMEOUT_S", 120.0),
    ("GENIE_WEDGE_GRACE", "WEDGE_GRACE_S", 60.0),
    ("GENIE_FAIL_THRESHOLD", "FAIL_THRESHOLD", 3),
    ("GENIE_MAX_INFLIGHT", "MAX_INFLIGHT", 2),
    ("GENIE_SEED", "FIXED_SEED", None),
    ("GENIE_ORPHAN_HOLD_CHARS", "ORPHAN_HOLD_CHARS", -1),
]


@pytest.mark.parametrize("var,attr,default", READERS)
def test_a_malformed_env_var_degrades_to_its_default_and_says_so(
        bare_env, gs, monkeypatch, capsys, var, attr, default):
    # `808O` -- the typo that used to kill the process at IMPORT with a bare
    # `invalid literal for int()`, before any startup line had printed, on a
    # server whose whole banner exists to explain itself. Every reader, not
    # just GENIE_SEED: _int_env's docstring claimed the rest already degraded
    # while ten of them still called int()/float() bare.
    monkeypatch.setenv(var, "808O")
    importlib.reload(gs)
    assert getattr(gs, attr) == default
    out = capsys.readouterr().out
    assert var in out and "808O" in out, "name the variable AND the value"


def test_no_reader_bypasses_the_degrading_helpers(gs):
    # The guard against the next bare int(os.environ...) landing: there is no
    # such call in the file. A reader added without going through _int_env
    # or _float_env is a typo that kills the process again.
    with open(gs.__file__, encoding="utf-8") as f:
        src = f.read()
    assert not re.search(r"\b(?:int|float)\(\s*os\.environ", src)
    # ...and the on/off flags' version of the same hole: a GENIE_* value
    # compared in place against a hand-picked set of spellings. Every one of
    # the four did that, and every one read some ordinary spelling as the
    # opposite of what was meant -- see _bool_env.
    assert not re.search(
        r'os\.environ\.get\(\s*"GENIE_\w+"[^)]*\)\s*(?:==|!=|in\b|not\s+in\b)',
        src)


FLAGS = [
    ("GENIE_WEDGE_EXIT", "WEDGE_EXIT", True),
    ("GENIE_THINKING", "THINKING_DEFAULT", False),
    ("GENIE_STRIP_THINK", "STRIP_THINK", False),
    ("GENIE_SUMMARIZE_EVICTED", "SUMMARIZE_EVICTED", True),
]


@pytest.mark.parametrize("var,attr,default", FLAGS)
@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("True", True), ("YES", True), ("on", True),
    (" 1", True),
    ("0", False), ("false", False), ("False", False), ("No", False),
    ("off", False), ("OFF", False), ("0 ", False),
])
def test_every_flag_reads_every_usual_spelling_both_ways(
        bare_env, gs, monkeypatch, capsys, var, attr, default, value, expected):
    # Each flag used to parse its own way and each way had a hole that read a
    # spelling as its OPPOSITE, with no line: GENIE_WEDGE_EXIT=False (what
    # PowerShell's `$env:X = $false` stores) left the wedge exit on,
    # GENIE_THINKING=True left reasoning off, GENIE_STRIP_THINK took only
    # "1", and GENIE_SUMMARIZE_EVICTED=false left summarising on.
    monkeypatch.setenv(var, value)
    importlib.reload(gs)
    assert getattr(gs, attr) is expected
    assert "WARNING" not in capsys.readouterr().out, "a spelling it knows"


@pytest.mark.parametrize("var,attr,default", FLAGS)
def test_an_unrecognised_flag_value_degrades_to_the_default_and_says_so(
        bare_env, gs, monkeypatch, capsys, var, attr, default):
    # The _int_env contract, for flags: not a silent reading either way.
    monkeypatch.setenv(var, "disabled")
    importlib.reload(gs)
    assert getattr(gs, attr) is default
    out = capsys.readouterr().out
    assert "WARNING" in out and var in out and "disabled" in out, (
        "name the variable AND the value")


@pytest.mark.parametrize("var,attr,default", FLAGS)
def test_an_unset_or_empty_flag_is_its_default_quietly(
        bare_env, gs, monkeypatch, capsys, var, attr, default):
    monkeypatch.setenv(var, "")
    importlib.reload(gs)
    assert getattr(gs, attr) is default
    monkeypatch.delenv(var)
    importlib.reload(gs)
    assert getattr(gs, attr) is default
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("var,attr,value,default", [
    ("GENIE_MAX_TOKENS", "DEFAULT_MAX_TOKENS", "0", 512),
    ("GENIE_MAX_TOKENS", "DEFAULT_MAX_TOKENS", "-1", 512),
    ("GENIE_WINDOW_MARGIN", "WINDOW_MARGIN", "-5", 64),
])
def test_an_out_of_range_value_degrades_like_a_malformed_one(
        bare_env, gs, monkeypatch, capsys, var, attr, value, default):
    # The values that PARSE and are still not usable. GENIE_MAX_TOKENS=-1
    # reached c_uint32 and wrapped to 4294967295, the unbounded generation
    # _max_tokens exists to refuse, and 0 skipped setMaxNumTokens and left the
    # previous request's cap on the resident dialog; a negative window margin
    # is a budget PAST the window, which build_windowed subtracts without a
    # check. So both are rejected -- but rejected the way a typo is, with a
    # line and the DEFAULT, not clamped to the bound in silence. The clamp
    # read as acceptance: -1 is llama.cpp's "no limit" and this repo runs a
    # llama-server leg beside this one, so an operator got a server whose
    # every uncapped answer was one token long with nothing on screen about it
    # (the banner does not print the effective cap either).
    monkeypatch.setenv(var, value)
    importlib.reload(gs)
    assert getattr(gs, attr) == default, "clamped to the bound, not defaulted"
    out = capsys.readouterr().out
    assert var in out and value in out, "name the variable AND the value"
    assert "WARNING" in out and "must be >=" in out


# --- the two path variables ----------------------------------------------

def test_a_relative_bundle_dir_is_made_absolute(gs, monkeypatch, tmp_path):
    # load_engine chdirs INTO the bundle dir and then joins the same variable
    # again to open genie_config.json -- so a relative path passed the isdir
    # check against the launch cwd and raised FileNotFoundError from the new
    # one, after the point it had been checked.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GENIE_BUNDLE_DIR", "rel_bundle")
    monkeypatch.setenv("GENIE_SDK_DIR", "rel_sdk")
    importlib.reload(gs)
    assert os.path.isabs(gs.BUNDLE_DIR) and os.path.isabs(gs.SDK_DIR)
    assert gs.BUNDLE_DIR == os.path.join(os.getcwd(), "rel_bundle")
    assert gs.SDK_DIR == os.path.join(os.getcwd(), "rel_sdk")


def test_an_unset_path_stays_empty_so_load_engine_can_name_it(gs, monkeypatch):
    monkeypatch.delenv("GENIE_BUNDLE_DIR", raising=False)
    monkeypatch.delenv("GENIE_SDK_DIR", raising=False)
    importlib.reload(gs)
    assert gs.BUNDLE_DIR == "" and gs.SDK_DIR == ""


# --- probe_tool_support: derived from the artifact ------------------------
# Whether a `tools` request is refused with a 400 rests entirely on this.

def test_a_vocab_that_carries_the_tool_token_supports_tools(gs, tmp_path):
    gs.BUNDLE_DIR = str(tmp_path)
    (tmp_path / "added_tokens.json").write_text(
        json.dumps({"<tool_call>": 151657, "</tool_call>": 151658}))
    assert gs.probe_tool_support() is True


def test_the_tokenizer_config_is_the_second_place_looked(gs, tmp_path):
    gs.BUNDLE_DIR = str(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% if tool_call %}<tool_call>{% endif %}"}))
    assert gs.probe_tool_support() is True


def test_a_vocab_without_the_token_does_not(gs, tmp_path):
    gs.BUNDLE_DIR = str(tmp_path)
    (tmp_path / "added_tokens.json").write_text(json.dumps({"<|im_end|>": 1}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"x": 1}))
    assert gs.probe_tool_support() is False


def test_no_bundle_files_at_all_means_no_tools(gs, tmp_path):
    gs.BUNDLE_DIR = str(tmp_path)
    assert gs.probe_tool_support() is False


def test_an_unset_bundle_dir_does_not_read_the_working_directory(
        gs, monkeypatch, tmp_path):
    # os.path.join("", "added_tokens.json") is a path relative to the CWD, so
    # with no bundle dir this used to answer True off a stray file in whatever
    # directory the caller happened to be in -- and True means a `tools`
    # request is accepted for a bundle nobody has looked at. main() cannot get
    # here unset (load_engine exits first); a direct caller can.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "added_tokens.json").write_text(json.dumps({"<tool_call>": 1}))
    gs.BUNDLE_DIR = ""
    assert gs.probe_tool_support() is False


# --- the listening socket ------------------------------------------------

@pytest.mark.parametrize("host,family", [
    ("127.0.0.1", socket.AF_INET), ("0.0.0.0", socket.AF_INET),
    ("localhost", socket.AF_INET), ("::1", socket.AF_INET6),
    ("::", socket.AF_INET6),
])
def test_the_server_picks_its_address_family_from_the_host(gs, host, family):
    # HTTPServer is AF_INET only, while LOOPBACK_HOSTS names "::1" as
    # recognised -- so GENIE_HOST=::1 passed every startup check and died in
    # the bind with a bare gaierror after the model load.
    srv = gs.Server((host, 0), gs.Handler, bind_and_activate=False)
    try:
        assert srv.socket.family == family
    finally:
        srv.server_close()


def test_the_ipv6_loopback_actually_binds(gs):
    if not socket.has_ipv6:
        pytest.skip("no IPv6 on this box")
    srv = gs.Server(("::1", 0), gs.Handler)     # this raised gaierror 11001
    try:
        assert srv.server_address[0] == "::1"
    finally:
        srv.server_close()


# --- main(): bind before load, listen after it, free under the lock -------

class FreeRecorder:
    """The one Genie call main()'s shutdown path makes, recording how."""

    def __init__(self):
        self.engine = None
        self.freed = []              # the handle each free was given
        self.freed_under_lock = None

    def GenieDialog_free(self, dialog):
        self.freed.append(dialog)
        self.freed_under_lock = self.engine.lock.locked()
        return 0


@pytest.fixture
def main_env(gs, monkeypatch):
    """main() with its side effects captured: returns (order, engine).

    port_in_use says free, load_engine is a recorder returning a REAL
    GenieEngine over a FreeRecorder (the shutdown path is the engine's own
    close(), so a stand-in engine would test the stand-in), Server is a
    recorder whose serve_forever raises the KeyboardInterrupt an operator's
    Ctrl-C would, and the watchdog thread is a no-op so nothing outlives the
    test. `engine.scopes` is the any_turn each signal_abort was asked with.
    """
    order = []
    lib = FreeRecorder()
    eng = gs.GenieEngine(lib, "DIALOG", tokenizer="TOKENIZER")
    lib.engine = eng
    eng.scopes = []
    real_abort = eng.signal_abort

    def signal_abort(any_turn=False, stalled=False):
        eng.scopes.append(any_turn)
        return real_abort(any_turn, stalled)
    eng.signal_abort = signal_abort

    class FakeServer:
        def __init__(self, addr, handler, bind_and_activate=True):
            order.append("bind+listen" if bind_and_activate else "socket")

        def server_bind(self):
            order.append("bind")

        def server_activate(self):
            order.append("listen")

        def server_close(self):
            order.append("close")

        def serve_forever(self):
            order.append("serve")
            raise KeyboardInterrupt

    def load():
        order.append("load")
        return eng
    monkeypatch.setattr(gs, "Server", FakeServer)
    monkeypatch.setattr(gs, "load_engine", load)
    monkeypatch.setattr(gs, "port_in_use", lambda h, p: False)
    monkeypatch.setattr(gs, "watchdog", lambda *a, **k: None)
    return order, eng


@pytest.mark.parametrize("flag", ["-h", "--help", "-help", "-?", "/?"])
def test_help_prints_usage_before_anything_is_touched(
        gs, main_env, monkeypatch, capsys, flag):
    # This server used to ignore argv altogether, so `--help` on a box with
    # GENIE_BUNDLE_DIR and GENIE_SDK_DIR set went straight into the 11-35s
    # model load, on an NPU other sessions here benchmark on. Usage, exit 0,
    # and NOTHING before it: no port probe, no bundle read, no bind, no load.
    order, _eng = main_env
    monkeypatch.setattr(gs, "port_in_use",
                        lambda h, p: pytest.fail("probed the port for --help"))
    monkeypatch.setattr(gs, "bundle_config_warnings",
                        lambda: pytest.fail("read the bundle for --help"))
    assert gs.main([flag]) == 0
    assert order == [], "no socket, bind or load for --help"
    out = capsys.readouterr().out
    for must in ("usage:", "GENIE_BUNDLE_DIR", "GENIE_SDK_DIR",
                 "genie_config.json", "docs/GENIE_SERVER.md",
                 "run-genie-server.ps1", "native ARM64"):
        assert must in out, "usage must say %r" % must
    for line in gs.ENDPOINT_LINES:
        assert line in out, "and list the endpoints the banner lists"
    assert "%s:%d" % (gs.HOST, gs.PORT) in out, "and where it would serve"


def test_a_help_flag_anywhere_on_the_line_still_means_help(gs, main_env, capsys):
    order, _eng = main_env
    assert gs.main(["--bogus", "--help"]) == 0
    assert order == [] and "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--port", "8081"], ["serve"], ["-v"]])
def test_an_unknown_argument_is_refused_by_name_before_anything(
        gs, main_env, monkeypatch, capsys, argv):
    # Refused, not ignored: `--port 8081` used to serve on 8080 with nothing
    # on screen to say the flag had done nothing.
    order, _eng = main_env
    monkeypatch.setattr(gs, "port_in_use",
                        lambda h, p: pytest.fail("probed the port"))
    assert gs.main(argv) == 2
    assert order == [], "no socket, bind or load"
    err = capsys.readouterr().err
    for a in argv:
        assert repr(a) in err, "name every argument it refused"
    assert "--help" in err and "GENIE_" in err, "and say what to do instead"


def test_the_script_passes_its_command_line_to_main(gs, tmp_path):
    # End to end, in a real child process: the __main__ block is what hands
    # sys.argv to main(), and a main() that handles --help is no use if the
    # script never gives it the flag. Both variables are SET (to empty dirs),
    # which is the case that used to reach the model load; with them set the
    # old server went on to the bind and exited 1 on the SDK check. No NPU is
    # reachable from here either way -- the dirs hold nothing.
    import subprocess
    import sys
    env = dict(os.environ)
    (tmp_path / "bundle").mkdir()
    (tmp_path / "sdk").mkdir()
    env.update(GENIE_BUNDLE_DIR=str(tmp_path / "bundle"),
               GENIE_SDK_DIR=str(tmp_path / "sdk"),
               GENIE_PORT=str(_free_port()), GENIE_HOST="127.0.0.1")
    r = subprocess.run([sys.executable, gs.__file__, "--help"], env=env,
                       cwd=str(tmp_path), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "usage:" in r.stdout and "GENIE_BUNDLE_DIR" in r.stdout
    both = r.stdout + r.stderr
    assert "SDK lib dir not found" not in both and "archs usable" not in both, (
        "it went on towards the load")
    r = subprocess.run([sys.executable, gs.__file__, "--nope"], env=env,
                       cwd=str(tmp_path), capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2 and "'--nope'" in r.stderr, r.stdout + r.stderr


def test_main_binds_before_it_loads_the_model_and_listens_after(gs, main_env):
    # A bind that cannot succeed should cost nothing, not 11-35s of loading a
    # bundle first -- so the bind is early. The listen is LATE: a port that
    # answers has to mean a model that is resident, which is what every
    # readiness probe written against this server assumes. Constructing the
    # Server the ordinary way does both at once, ahead of the load, and a
    # load that failed seconds in then exited behind a port that had already
    # said "started".
    order, _eng = main_env
    gs.main()
    assert order == ["socket", "bind", "load", "listen", "serve"]


def test_the_reserved_address_is_printed_before_the_load(
        gs, main_env, monkeypatch, capsys):
    # A direct run printed no address until the model was resident, so for
    # the whole 11-35s every client tool here said "nothing listening at ...
    # start one" about a server that WAS starting, with nothing on screen to
    # match that against.
    order, eng = main_env
    seen = {}

    def load():
        seen["before_load"] = capsys.readouterr().out
        order.append("load")
        return eng
    monkeypatch.setattr(gs, "load_engine", load)
    gs.main()
    assert ("port %s:%d is reserved" % (gs.HOST, gs.PORT)) in seen["before_load"]
    assert "refuses connections until the model is resident" in seen["before_load"]


def test_a_bind_failure_exits_by_name_before_the_load(gs, monkeypatch):
    # Over the REAL Server (so not main_env, which replaces it): what is
    # under test includes the socket it made being closed again.
    order, closed = [], []
    monkeypatch.setattr(gs, "port_in_use", lambda h, p: False)
    monkeypatch.setattr(gs, "load_engine", lambda: order.append("load"))

    class Unbindable(gs.Server):
        def server_bind(self):
            raise OSError(10049, "The requested address is not valid in its context")

        def server_close(self):
            closed.append(self)
            super().server_close()
    monkeypatch.setattr(gs, "Server", Unbindable)
    with pytest.raises(SystemExit) as e:
        gs.main()
    msg = str(e.value)
    assert "cannot bind" in msg and "GENIE_HOST" in msg and "10049" in msg
    assert "already holds that port" not in msg, (
        "10049 is an address this machine does not have, not a busy port")
    assert order == [], "no load was paid for"
    assert len(closed) == 1 and closed[0].socket.fileno() == -1, (
        "the socket it made was closed, not left to the collector")


@pytest.mark.parametrize("winerror,text", [
    (10048, "Only one usage of each socket address"),
    (10013, "An attempt was made to access a socket in a way forbidden"),
])
def test_a_taken_port_says_the_holder_can_be_on_another_host(
        gs, monkeypatch, winerror, text):
    # The two errnos an exclusive bind reports a taken port with, and the one
    # thing neither of them says: the other holder may be using a DIFFERENT
    # GENIE_HOST on the same port, which is the collision port_in_use cannot
    # see during a load. 10013 is the worse of the two left bare -- "access
    # forbidden" reads like a firewall or a privileged port, not like the
    # other instance of this server that it actually is.
    monkeypatch.setattr(gs, "port_in_use", lambda h, p: False)
    monkeypatch.setattr(gs, "load_engine", lambda: pytest.fail("no load"))

    class Taken(gs.Server):
        def server_bind(self):
            raise OSError(winerror, text, None, winerror)
    monkeypatch.setattr(gs, "Server", Taken)
    with pytest.raises(SystemExit) as e:
        gs.main()
    msg = str(e.value)
    assert "cannot bind" in msg and text in msg
    assert "already holds that port" in msg and "ANY GENIE_HOST" in msg


def test_a_load_failure_gives_the_port_back(gs, main_env, monkeypatch):
    # load_engine's failures are named sys.exits, and the socket is bound by
    # then. It must not stay bound behind an exit somebody caught.
    order, _eng = main_env

    def load():
        order.append("load")
        raise SystemExit("GenieDialog_create failed, status=-1")
    monkeypatch.setattr(gs, "load_engine", load)
    with pytest.raises(SystemExit) as e:
        gs.main()
    assert "GenieDialog_create" in str(e.value)
    assert order == ["socket", "bind", "load", "close"], "never listened"


def test_a_listen_failure_exits_by_name(gs, main_env, monkeypatch):
    order, _eng = main_env          # gs.Server is main_env's recorder here

    def refuse(self):
        raise OSError(98, "Address already in use")
    monkeypatch.setattr(gs.Server, "server_activate", refuse)
    with pytest.raises(SystemExit) as e:
        gs.main()
    assert "cannot listen" in str(e.value) and "Address already in use" in str(e.value)
    assert order[-1] == "close" and "serve" not in order


def _connects(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def test_nothing_connects_until_the_model_is_resident(gs, monkeypatch):
    # The property the two-step start rests on, against the REAL Server and
    # real sockets because it belongs to the OS: a socket that is bound and
    # not listening turns a connect away exactly as a closed port does. If
    # that ever stopped being true here, bind-early-listen-late would be
    # bind-early-hang-the-client, and this is where it would show.
    seen = {}
    servers = []

    class Spy(gs.Server):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            servers.append(self)

        def serve_forever(self):
            seen["serving"] = _connects(self.server_address[1])
            raise KeyboardInterrupt

    lib = FreeRecorder()
    eng = gs.GenieEngine(lib, "DIALOG")
    lib.engine = eng

    def load():
        seen["loading"] = _connects(servers[0].server_address[1])
        return eng
    monkeypatch.setattr(gs, "Server", Spy)
    monkeypatch.setattr(gs, "HOST", "127.0.0.1")
    monkeypatch.setattr(gs, "PORT", 0)              # any free port
    monkeypatch.setattr(gs, "port_in_use", lambda h, p: False)
    monkeypatch.setattr(gs, "load_engine", load)
    monkeypatch.setattr(gs, "watchdog", lambda *a, **k: None)
    try:
        gs.main()
    finally:
        for s in servers:
            s.server_close()
    assert seen == {"loading": False, "serving": True}


def test_a_second_instance_cannot_take_the_port_during_the_load(gs):
    # The early bind's other job. port_in_use cannot see a server that is
    # still loading (nothing accepts yet), so the BIND has to be what stops
    # the second one -- and on Windows it only does because Server asks for
    # the port exclusively (SO_EXCLUSIVEADDRUSE; leaving SO_REUSEADDR off is
    # not enough, see the next test). On POSIX two such binds both succeed
    # and the loser is turned away at its listen instead (main() exits by
    # name there too).
    first = gs.Server(("127.0.0.1", 0), gs.Handler, bind_and_activate=False)
    second = None
    try:
        first.server_bind()
        port = first.server_address[1]
        assert gs.port_in_use("127.0.0.1", port) is False, "nothing accepts yet"
        second = gs.Server(("127.0.0.1", port), gs.Handler, bind_and_activate=False)
        if os.name == "nt":
            with pytest.raises(OSError):
                second.server_bind()
        else:
            second.server_bind()
            first.server_activate()
            with pytest.raises(OSError):
                second.server_activate()
    finally:
        first.server_close()
        if second is not None:
            second.server_close()


def _free_port():
    """A port nothing holds, by binding one and giving it straight back."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@pytest.mark.skipif(os.name != "nt", reason="SO_EXCLUSIVEADDRUSE is Windows-only")
@pytest.mark.parametrize("first,second", [("0.0.0.0", "127.0.0.1"),
                                          ("127.0.0.1", "0.0.0.0")])
def test_two_instances_on_different_hosts_cannot_share_the_port_either(
        gs, first, second):
    # The gap the test above did not cover: on Windows a wildcard bind and a
    # specific bind coexist on one port unless the bind is EXCLUSIVE, so a
    # second instance on the default GENIE_HOST passed both checks against an
    # instance exposed with the documented GENIE_HOST=0.0.0.0 -- port_in_use
    # sees nothing accepting during the load, and its own bind succeeded --
    # and loaded a second bundle onto the HTP beside the first. Real sockets
    # in both orders, because this belongs to the OS: the two orders fail
    # with DIFFERENT errnos (10013 for the wildcard asking second, 10048 for
    # the specific one), and main()'s exit names both.
    port = _free_port()
    a = gs.Server((first, port), gs.Handler, bind_and_activate=False)
    b = None
    try:
        a.server_bind()             # as main() does, before the model load
        assert gs.port_in_use(first, port) is False, "nothing accepts yet"
        b = gs.Server((second, port), gs.Handler, bind_and_activate=False)
        with pytest.raises(OSError) as e:
            b.server_bind()
        assert e.value.winerror in (10013, 10048), (
            "refused, but not for being taken: %r" % (e.value,))
    finally:
        a.server_close()
        if b is not None:
            b.server_close()


def test_a_restart_right_after_a_clean_exit_still_binds(gs):
    # The other half of an exclusive bind: it must not cost the ordinary
    # restart. SO_EXCLUSIVEADDRUSE is the opposite of SO_REUSEADDR, and
    # SO_REUSEADDR is what POSIX servers set to rebind through TIME_WAIT --
    # so the case to prove is a server that ACCEPTED a connection and then
    # exited cleanly, which is every restart during a bench run. TIME_WAIT
    # applies to that connection, not to the listening socket, so the port
    # comes back: measured here rather than assumed.
    port = _free_port()
    srv = gs.Server(("127.0.0.1", port), gs.Handler, bind_and_activate=False)
    try:
        srv.server_bind()
        srv.server_activate()
        client = socket.create_connection(("127.0.0.1", port), timeout=2)
        conn, _ = srv.socket.accept()
        conn.close()
        client.close()
    finally:
        srv.server_close()
    again = gs.Server(("127.0.0.1", port), gs.Handler, bind_and_activate=False)
    try:
        again.server_bind()         # this is the restart; it must not raise
        again.server_activate()
        assert again.server_address[1] == port
    finally:
        again.server_close()


def test_a_port_out_of_range_exits_by_name_too(gs, monkeypatch):
    # GENIE_PORT=80800 is a perfectly good integer, so _int_env passes it, and
    # bind() raises OverflowError for it -- not an OSError. Against the REAL
    # Server, because which exception that is belongs to the socket module.
    loads = []
    monkeypatch.setattr(gs, "port_in_use", lambda h, p: False)
    monkeypatch.setattr(gs, "load_engine", lambda: loads.append("load"))
    monkeypatch.setattr(gs, "PORT", 80800)
    with pytest.raises(SystemExit) as e:
        gs.main()
    assert "cannot bind" in str(e.value) and "80800" in str(e.value)
    assert loads == [], "no load was paid for"


def test_shutdown_aborts_then_frees_the_dialog_under_the_lock(gs, main_env):
    # Handler and worker threads are daemons, so a generation can still be
    # inside GenieDialog_query when Ctrl-C lands: freeing the handle under it
    # is a fault in a process that is already leaving. Abort first, then free
    # with the lock held.
    _order, eng = main_env
    gs.main()
    # Whoever holds the dialog: the main thread consumes no stream, so a bare
    # signal_abort() from it would be aimed at nothing.
    assert eng.scopes == [True]
    assert eng.lib.freed == ["DIALOG"], "the handle it was created with, once"
    assert eng.lib.freed_under_lock is True
    assert not eng.lock.locked(), "and released afterwards"
    # Through close(), not a bare free: the daemon threads outlive this path,
    # and only a CLOSED engine turns their next native call away (the engine's
    # side of that is pinned in test_engine_kv).
    assert (eng._closed, eng.dialog, eng.tokenizer) == (True, None, None)


def test_shutdown_refuses_new_turns_before_it_aborts_the_live_one(gs, main_env):
    # The ORDER, which is the whole of the fix. An abort frees the engine
    # lock, and the lock goes to the longest waiter -- a request parked in
    # _run_query, not the close() that follows. Marked as closing first, that
    # request raises instead of starting a fresh query; marked after, it can
    # win the gap. The turn's side of this is in test_engine_kv; what is
    # pinned here is that main() takes the two steps in that order.
    _order, eng = main_env
    seen = {}
    wrapped = eng.signal_abort          # main_env's recorder

    def spy(any_turn=False, stalled=False):
        seen["closing"] = eng._closing
        return wrapped(any_turn=any_turn, stalled=stalled)
    eng.signal_abort = spy
    gs.main()
    assert seen == {"closing": True}, "aborted before it stopped admitting turns"
    assert eng.scopes == [True], "and it is still the any_turn abort"


def test_shutdown_skips_the_free_when_the_driver_is_stuck(gs, main_env, capsys):
    # The lock never frees: the abort did not take, the driver is stuck, and
    # a free on a stuck driver can hang too -- the same reasoning as
    # _exit_for_supervisor. Leave it to the OS.
    _order, eng = main_env
    gs.SHUTDOWN_FREE_TIMEOUT_S = 0.01
    eng.lock.acquire()
    try:
        gs.main()
    finally:
        eng.lock.release()
    assert eng.scopes == [True]
    assert eng.lib.freed == [], "the free was never attempted"
    assert (eng._closed, eng.dialog, eng.tokenizer) == (False, "DIALOG", "TOKENIZER"), (
        "and nothing else was touched: the handles are still live ones")
    assert "still inside the driver" in capsys.readouterr().out


# --- load_engine, device-free --------------------------------------------

class FakeFn:
    def __init__(self, name, impl):
        self.name, self.impl = name, impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.impl(self.name, *args)


class FakeDLL:
    """Stands in for C.WinDLL("Genie.dll"): every attribute is a function that
    records its name and returns the scripted status (SUCCESS by default).
    The JSON handed to GenieDialogConfig_createFromJson is kept, because it
    is the only place the seed the dialog is CREATED with can be observed."""

    def __init__(self):
        self.statuses = {}
        self.calls = []
        self.configs = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def impl(fname, *args):
            self.calls.append(fname)
            if fname == "GenieDialogConfig_createFromJson":
                self.configs.append(args[0])
            return self.statuses.get(fname, 0)
        fn = FakeFn(name, impl)
        self.__dict__[name] = fn
        return fn


@pytest.fixture
def fake_sdk(gs, tmp_path, monkeypatch):
    """A loadable fake: one usable Hexagon, a bundle with a config, a DLL."""
    sdk = tmp_path / "sdk"
    (sdk / "lib" / "hexagon-v73" / "unsigned").mkdir(parents=True)
    msvc = sdk / "lib" / "aarch64-windows-msvc"
    msvc.mkdir(parents=True)
    (msvc / "QnnHtpV73Stub.dll").write_bytes(b"")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "genie_config.json").write_text(json.dumps({
        "dialog": {"context": {"size": 4096},
                   "sampler": {"version": 1, "seed": 42, "temp": 0.8}}}))
    gs.SDK_DIR, gs.LIB_DIR, gs.BUNDLE_DIR = str(sdk), str(msvc), str(bundle)
    gs._SAMPLER = None                    # read the config written above
    monkeypatch.delenv("GENIE_HEXAGON_ARCH", raising=False)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))    # restored after
    monkeypatch.setenv("ADSP_LIBRARY_PATH", "")
    monkeypatch.chdir(tmp_path)           # load_engine chdirs into the bundle
    dll = FakeDLL()
    monkeypatch.setattr(gs.C, "WinDLL", lambda path: dll)
    monkeypatch.setattr(gs.os, "add_dll_directory", lambda p: None, raising=False)
    return dll


def test_a_missing_tokenizer_is_announced_not_swallowed(gs, fake_sdk, capsys):
    # Without it every window budget, overflow 400 and usage figure is
    # silently the len//4 estimate -- the degradation this server refuses
    # everywhere else -- and nothing printed a word about it.
    fake_sdk.statuses["GenieDialog_getTokenizer"] = -1
    eng = gs.load_engine()
    assert eng.tokenizer is None
    assert gs.TOKENIZER_STATUS == -1
    out = capsys.readouterr().out
    assert "GenieDialog_getTokenizer failed, status=-1 (ERROR_GENERAL)" in out
    assert "ESTIMATES" in out


def test_a_working_tokenizer_is_reported_and_quiet(gs, fake_sdk, capsys):
    eng = gs.load_engine()
    assert eng.tokenizer is not None
    assert gs.TOKENIZER_STATUS == gs.GENIE_STATUS_SUCCESS
    assert "getTokenizer" not in capsys.readouterr().out


def test_the_restore_baseline_carries_the_seed_the_dialog_was_created_with(gs, fake_sdk):
    # The on-disk block carries the shipped 42; the dialog is created from a
    # config TEXT patched with a fresh seed. A baseline read from disk would
    # restore 42 -- re-arming the fixed-seed replay next_seed() exists to
    # remove, the day GenieSampler_applyConfig starts working.
    gs.FIXED_SEED = None
    eng = gs.load_engine()
    created_with = json.loads(fake_sdk.configs[0])["dialog"]["sampler"]["seed"]
    assert created_with != 42
    assert eng.default_sampler["seed"] == created_with
    assert eng.default_sampler["temp"] == 0.8, "the rest of the block is the bundle's"


def test_a_pinned_seed_is_the_baseline_too(gs, fake_sdk):
    gs.FIXED_SEED = 7
    eng = gs.load_engine()
    assert json.loads(fake_sdk.configs[0])["dialog"]["sampler"]["seed"] == 7
    assert eng.default_sampler["seed"] == 7


def test_an_unpatchable_config_leaves_no_seed_in_the_baseline(gs, fake_sdk, tmp_path, capsys):
    # The seed patch failed, so whatever the dialog was created with is not
    # known here; the baseline must not claim one.
    (tmp_path / "bundle" / "genie_config.json").write_text("not json")
    gs._SAMPLER = None
    eng = gs.load_engine()
    assert "could not set the sampler seed" in capsys.readouterr().out
    assert "seed" not in eng.default_sampler


# --- ...and the three startup failures it exits on ------------------------
# A bundle compiled for another Hexagon or another QAIRT is the most common
# real failure on this box, and each of these exits is what turns it into a
# sentence naming the suspect. None of them can be softened into a warning:
# bench_servers.wait_port decides "started" from "the child is alive and the
# port answers", so an engine returned over a handle Genie refused to make
# would be reported as a healthy server and fault inside the driver on the
# first request instead.

def test_no_hexagon_this_os_can_drive_exits_naming_the_skels_it_found(
        gs, fake_sdk, tmp_path):
    # A skel with no Windows stub beside it is an Android-only Hexagon: QAIRT
    # ships it, and nothing here can reach it. Saying which ones were found is
    # what separates "your SDK is wrong" from "this arch is not drivable".
    (tmp_path / "sdk" / "lib" / "aarch64-windows-msvc"
     / "QnnHtpV73Stub.dll").unlink()
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert "no usable Hexagon" in msg
    assert "v73" in msg, "the skel it found but cannot drive is the clue"
    assert "GENIE_HEXAGON_ARCH" in msg, "and the pin that could be causing it"
    assert fake_sdk.calls == [], "it exits before Genie.dll is even asked"


@pytest.fixture
def dll_loads(gs, fake_sdk, monkeypatch):
    """fake_sdk, with every attempt to load a DLL recorded by path."""
    loads = []

    def load(path):
        loads.append(path)
        return fake_sdk
    monkeypatch.setattr(gs.C, "WinDLL", load)
    return loads


@pytest.mark.parametrize("unset,other", [("GENIE_BUNDLE_DIR", "GENIE_SDK_DIR"),
                                         ("GENIE_SDK_DIR", "GENIE_BUNDLE_DIR")])
def test_one_unset_variable_is_the_only_one_named(gs, fake_sdk, unset, other):
    # Both used to be named whenever either was unset, so an operator who had
    # just set the bundle was told to set it again.
    attr = {"GENIE_BUNDLE_DIR": "BUNDLE_DIR", "GENIE_SDK_DIR": "SDK_DIR"}
    kept = getattr(gs, attr[other])
    setattr(gs, attr[unset], "")
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert msg.startswith("set %s (" % unset), msg
    assert "set %s" % other not in msg and "and %s" % other not in msg
    assert "%s is set (%s)" % (other, kept) in msg, "and what the other one is"


def test_both_unset_are_both_named(gs, fake_sdk):
    gs.BUNDLE_DIR = gs.SDK_DIR = ""
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    assert str(e.value) == ("set GENIE_BUNDLE_DIR (the Genie bundle dir) and "
                            "GENIE_SDK_DIR (the QAIRT 2.45 root) -- see "
                            "docs/GENIE_SERVER.md")


def test_a_bundle_dir_with_no_config_exits_by_name_before_the_dll(
        gs, dll_loads, tmp_path):
    # An existing directory that is not a bundle -- empty, or a half-made
    # one. It used to pass the isdir check, load Genie.dll, and die in a bare
    # FileNotFoundError traceback from the open() of genie_config.json.
    (tmp_path / "bundle" / "genie_config.json").unlink()
    gs._CONFIG_PRESENT = None             # the fixture pins it present
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert msg.startswith("no genie_config.json in %s" % gs.BUNDLE_DIR)
    assert "bundle directory itself" in msg and "not the folder above it" in msg
    assert "Nothing directly inside it" in msg
    assert dll_loads == [], "it exits before Genie.dll is loaded"


def test_the_folder_above_a_bundle_names_the_bundles_inside_it(
        gs, dll_loads, tmp_path):
    # The likeliest first-run shape: `qai-hub-models fetch --extract -o <dir>`
    # puts the bundle in a model-named folder inside <dir>, and the variable
    # gets <dir>. The fix is one folder down, so say which one.
    gs.BUNDLE_DIR = str(tmp_path)
    (tmp_path / "not-a-bundle").mkdir()
    gs._CONFIG_PRESENT = None
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert "no genie_config.json in %s" % tmp_path in msg
    assert "point GENIE_BUNDLE_DIR at one of them: bundle" in msg
    assert "not-a-bundle" not in msg
    assert dll_loads == []


def _write_config(bundle, **dialog):
    cfg = {"dialog": {"context": {"size": 4096},
                      "sampler": {"version": 1, "seed": 42, "temp": 0.8}}}
    cfg["dialog"].update(dialog)
    (bundle / "genie_config.json").write_text(json.dumps(cfg))


def _full_config(bundle):
    _write_config(
        bundle, tokenizer={"version": 1, "path": "tokenizer.json"},
        engine={"backend": {"type": "QnnHtp", "QnnHtp": {"poll": False},
                            "extensions": "htp_backend_ext_config.json"},
                "model": {"type": "binary", "binary": {
                    "ctx-bins": ["part1_of_2.bin", "part2_of_2.bin"]}}})


def test_a_file_the_config_names_and_the_bundle_lacks_exits_by_name(
        gs, dll_loads, tmp_path):
    # An interrupted copy of a multi-GB bundle. GenieDialogConfig_createFromJson
    # does not check the files exist (measured: SUCCESS for a renamed ctx-bin),
    # so this used to reach GenieDialog_create -- whose exit blames an arch or
    # QAIRT mismatch and sends the operator to rebuild a bundle that is only
    # half-copied.
    bundle = tmp_path / "bundle"
    _full_config(bundle)
    (bundle / "part1_of_2.bin").write_bytes(b"")
    (bundle / "tokenizer.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    named = msg.split("names 2 files that are not there: ", 1)[1].split(chr(10))[0]
    assert named == "part2_of_2.bin, htp_backend_ext_config.json", (
        "the missing ones, in config order, and none of the present ones")
    assert "incomplete copy" in msg
    assert dll_loads == []


def test_a_bundle_with_every_file_it_names_goes_on_to_load(gs, dll_loads, tmp_path):
    bundle = tmp_path / "bundle"
    _full_config(bundle)
    for name in ("part1_of_2.bin", "part2_of_2.bin", "tokenizer.json",
                 "htp_backend_ext_config.json"):
        (bundle / name).write_bytes(b"")
    assert gs.bundle_files_missing() == []
    assert gs.load_engine() is not None
    assert len(dll_loads) == 1


def test_an_unreadable_config_is_left_for_genie_to_name(gs, fake_sdk, tmp_path):
    # Not a claim about files in a config nobody could parse: Genie parses it
    # itself and says where the syntax error is.
    (tmp_path / "bundle" / "genie_config.json").write_text("{not json")
    assert gs.bundle_files_missing() == []


def test_an_x64_interpreter_is_told_to_use_an_arm64_one(gs, fake_sdk, monkeypatch):
    # What a direct `python genie_server.py` under an x64 python gets: the
    # launcher refuses one by name, the server used to die in a ctypes
    # traceback ending "%1 is not a valid Win32 application".
    def load(path):
        raise OSError(22, "%1 is not a valid Win32 application", None, 193)
    monkeypatch.setattr(gs.C, "WinDLL", load)
    monkeypatch.setattr(gs.sysconfig, "get_platform", lambda: "win-amd64")
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert "win-amd64" in msg and "native ARM64 Python" in msg
    assert "GENIE_PYTHON" in msg, "and what to point at one under the launcher"
    assert fake_sdk.calls == []


def test_an_sdk_without_genie_dll_names_the_sdk_dir(gs, fake_sdk, monkeypatch):
    def load(path):
        raise FileNotFoundError("Could not find module '%s' (or one of its "
                                "dependencies)." % path)
    monkeypatch.setattr(gs.C, "WinDLL", load)
    monkeypatch.setattr(gs.sysconfig, "get_platform", lambda: "win-arm64")
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert msg.startswith("no Genie.dll at %s" % os.path.join(gs.LIB_DIR, "Genie.dll"))
    assert "GENIE_SDK_DIR (%s)" % gs.SDK_DIR in msg and "QAIRT 2.45 root" in msg


def test_a_genie_dll_whose_dependencies_fail_says_so(gs, fake_sdk, monkeypatch):
    # The file is there, so ctypes' "or one of its dependencies" is the half
    # that is true: a Qnn*.dll missing beside it, or no VC++ runtime.
    with open(os.path.join(gs.LIB_DIR, "Genie.dll"), "wb"):
        pass

    def load(path):
        raise FileNotFoundError("Could not find module '%s' (or one of its "
                                "dependencies)." % path)
    monkeypatch.setattr(gs.C, "WinDLL", load)
    monkeypatch.setattr(gs.sysconfig, "get_platform", lambda: "win-arm64")
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert "one of ITS dependencies" in msg and "Visual C++" in msg
    assert "Qnn*.dll" in msg and gs.SDK_DIR in msg


def test_a_non_arm64_dll_on_an_arm64_python_blames_the_sdk(gs, fake_sdk, monkeypatch):
    def load(path):
        raise OSError(22, "%1 is not a valid Win32 application", None, 193)
    monkeypatch.setattr(gs.C, "WinDLL", load)
    monkeypatch.setattr(gs.sysconfig, "get_platform", lambda: "win-arm64")
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert "not an ARM64 image" in msg and "GENIE_SDK_DIR" in msg
    assert "GENIE_PYTHON" not in msg, "the interpreter is not the problem here"


def test_a_config_genie_refuses_exits_with_the_status_it_gave(gs, fake_sdk):
    # The seed patch above it is allowed to fail quietly -- Genie parses the
    # config itself and gives a better error. This is that error, and it is
    # not a warning: there is no dialog to create from a config that was
    # refused.
    fake_sdk.statuses["GenieDialogConfig_createFromJson"] = -8
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    assert "GenieDialogConfig_createFromJson failed, status=-8" in str(e.value)
    assert "GenieDialog_create" not in fake_sdk.calls


def test_a_bundle_that_will_not_load_names_every_suspect(gs, fake_sdk):
    # A bare status code sends people hunting through their config, so the
    # message names the suspects and prints what this box can offer. ALL of
    # them: it used to name only the bundle built for another Hexagon or
    # QAIRT, whatever the status, and on a box whose HTP is shared or
    # degraded that sent the operator to rebuild a bundle that was fine.
    fake_sdk.statuses["GenieDialog_create"] = -1
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert "GenieDialog_create failed, status=-1 (ERROR_GENERAL)" in msg
    assert "locked to one Hexagon arch AND one QAIRT version" in msg
    assert "held by another process, or degraded" in msg
    assert "Get-Process" in msg and "reboot" in msg, "and what to check"
    assert "v73" in msg, "the archs THIS box has, not a general statement"
    assert gs.BUNDLE_DIR in msg and gs.SDK_DIR in msg, (
        "the two paths whoever reads this has to compare")
    assert "GenieDialog_getTokenizer" not in fake_sdk.calls, (
        "and no engine is built over a dialog Genie refused to make")
    assert msg.index("Hexagon arch AND") < msg.index("held by another"), (
        "an unnamed failure keeps the arch mismatch first")


def test_a_memory_failure_leads_with_memory_not_the_bundle(gs, fake_sdk):
    # GenieDialog.h lists MEM_ALLOC (-3) among GenieDialog_create's returns.
    # Printed bare over advice to rebuild the bundle, it read as an arch
    # mismatch. Which status each cause really produces is unmeasured, so the
    # name reorders the suspects; it does not drop any.
    fake_sdk.statuses["GenieDialog_create"] = -3
    with pytest.raises(SystemExit) as e:
        gs.load_engine()
    msg = str(e.value)
    assert "status=-3 (ERROR_MEM_ALLOC)" in msg
    assert "out of memory" in msg
    assert msg.index("out of memory") < msg.index("Hexagon arch AND")


def _slow_create(fake_sdk, seconds, status=0):
    """GenieDialog_create that takes `seconds` -- a load, stood in for."""
    import time

    def impl(name, *args):
        fake_sdk.calls.append(name)
        time.sleep(seconds)
        return status
    fake_sdk.__dict__["GenieDialog_create"] = FakeFn("GenieDialog_create", impl)


def _watcher_gone(gs):
    import threading
    for t in threading.enumerate():
        if t.name == "genie-load-watch":
            t.join(timeout=2)
            if t.is_alive():
                return False
    return True


def test_a_slow_load_says_so_and_names_what_to_suspect(gs, fake_sdk, capsys):
    # GenieDialog_create is one blocking native call, nothing supervises it,
    # and the port refuses connections throughout -- so a load that never
    # finished looked exactly like one still working: one line, then nothing.
    gs.LOAD_SLOW_AFTER_S, gs.LOAD_SLOW_EVERY_S = 0.05, 30.0
    _slow_create(fake_sdk, 0.5)
    gs.load_engine()
    out = capsys.readouterr().out
    assert out.count("still loading after") == 1, out
    assert "held by another process or degraded" in out
    assert "Get-Process" in out, "and what to check"
    assert out.index("still loading") < out.index("model resident on HTP")
    assert _watcher_gone(gs), "the watcher ends with the load"


def test_a_normal_load_prints_no_slow_line(gs, fake_sdk, capsys):
    gs.load_engine()
    assert "still loading" not in capsys.readouterr().out
    assert _watcher_gone(gs)


def test_the_watcher_ends_with_a_failed_load_too(gs, fake_sdk, capsys):
    gs.LOAD_SLOW_AFTER_S = 0.05
    _slow_create(fake_sdk, 0.2, status=-1)
    with pytest.raises(SystemExit):
        gs.load_engine()
    assert _watcher_gone(gs), "a failed load must not leave it ticking"


def test_the_watcher_is_handed_nothing_that_reaches_the_engine(
        gs, fake_sdk, monkeypatch):
    # The point of doing this from a thread at all is that it cannot make a
    # stuck load worse: it gets the Event and the start time, never the lib,
    # the config handle or the dialog.
    import threading
    made = []
    real = threading.Thread

    def recording(*a, **k):
        made.append(k)
        return real(*a, **k)
    monkeypatch.setattr(gs.threading, "Thread", recording)
    gs.load_engine()
    watch = [k for k in made if k.get("name") == "genie-load-watch"]
    assert len(watch) == 1 and watch[0]["daemon"] is True
    event, t0 = watch[0]["args"]
    assert isinstance(event, threading.Event) and isinstance(t0, float)
    assert event.is_set(), "set once the load returned"


def test_a_status_the_header_does_not_name_is_printed_bare(gs):
    assert gs.status_text(-99) == "-99"
    assert gs.status_text(-6) == "-6 (ERROR_QUERY_FAILED)"
