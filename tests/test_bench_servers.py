"""Tests for the cross-server A/B harness, bench_servers.py.

The measurement needs the NPU, a bundle and two servers. What is covered here
is everything the tool does AROUND the measurement, because that is where it
was dangerous rather than merely wrong: it used to `taskkill /IM geniex.exe`
and force-stop whatever owned both arm ports before pass 1 (a co-tenant's
resident server, killed with no notice on either side), it burned 240 s per
pass on a child that had exited in its first second, and it overwrote its own
results file without asking.

Device-free like the rest: no process is launched, no socket opened, no
PowerShell run and nothing slept on. `subprocess`, `time` and `sys` are
replaced INSIDE the module under test by the World below, the two port
helpers are stubbed where a test is not about them, and the HTTP layer is
either bench_endpoint.post_timed stubbed whole or a stubbed urlopen under the
real one. The tokenizer is a word counter, which is all the arithmetic needs.
The box's process table is the World's too, made up per test: no test lists
or bakes in the processes of the machine it runs on.
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import urllib.error

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    """One token per whitespace-separated word: exact, and needs no bundle."""

    def encode(self, text, add_special_tokens=False):
        return types.SimpleNamespace(ids=text.split())

    def decode(self, ids):
        return " ".join(ids)


@pytest.fixture
def bs(monkeypatch):
    # Pinned, not inherited: MIN_DECODE_STEPS is read from the environment when
    # bench_endpoint is imported, and the floor tests below state it as 16.
    monkeypatch.delenv("GENIE_MIN_DECODE_STEPS", raising=False)
    monkeypatch.syspath_prepend(SRC)
    # A fresh load of the REAL bench_endpoint under its own name, because
    # test_bench_contention installs a SimpleNamespace there at collection
    # time and bench_servers imports the module at scope. The sampler is
    # stubbed: the real one launches PowerShell.
    be = _load("bench_endpoint")
    be.box_state = lambda: (True, 73.0, 12.5, 88.0)
    monkeypatch.setitem(sys.modules, "bench_endpoint", be)
    mod = _load("bench_servers")
    mod.TOK = _Tok()
    return mod


# --- everything outside the process, faked ----------------------------------

class _Proc:
    """A Popen stand-in: alive (returncode None) until something ends it."""

    def __init__(self, world, pid, port, cmd, returncode=None):
        self.world, self.pid, self.port, self.cmd = world, pid, port, cmd
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.end(-15)

    def kill(self):
        self.end(-9)

    def end(self, code):
        if self.returncode is None:
            self.returncode = code
        self.world.listening.discard(self.port)


class World:
    """The box as bench_servers sees it: ports, processes, a clock, two servers.

    A launched child "listens" on its arm's port at once unless the test said
    it dies at startup; `taskkill /PID n` ends child n and frees its port. A
    FOREIGN listener is a port in `owners` (pid) and/or `listening` that no
    child of this run put there -- the thing the tool must never touch.
    """

    def __init__(self, tmp_path):
        self.now = 1000.0
        self.slept = []
        self.listening = set()
        self.owners = {}            # port -> pid of a listener that is not ours
        self.launched = []          # every _Proc, in launch order
        self.ran = []               # every subprocess.run argv
        self.opened = []            # (stdout name, stderr) per Popen
        self.dies_at_start = {}     # port -> (exit code, what the child printed)
        self.never_answers = set()  # ports whose child stays alive and deaf
        self.events = []            # ("launch"|"kill", pid), in order
        self.popen_error = None
        self.on_kill = None         # called after a child is taskkilled
        self.posts = []             # (base, payload) per request
        self.post_hook = None       # may raise, or return a (body, wall) to use
        self.tokenizer_loads = []
        # The box's OTHER processes, as list_processes would return them --
        # made up per test, never a real scan. None is a listing that failed.
        # This run's own children are added by _popen as they launch.
        self.processes = []
        self.process_listings = 0
        self.bundle = tmp_path / "bundle"
        self.bundle.mkdir()
        (self.bundle / "genie_config.json").write_text(
            json.dumps({"dialog": {"context": {"size": 4096}}}), encoding="utf-8")
        self.sdk = tmp_path / "sdk"
        self.sdk.mkdir()
        # main() refuses a geniex.exe that is not there before it launches
        # anything, so the suite supplies one that exists. Pinned rather than
        # inherited: whether GenieX CLI happens to be installed on the box
        # running the tests is not this suite's business.
        self.geniex = tmp_path / "geniex.exe"
        self.geniex.write_bytes(b"")
        self.out = tmp_path / "sweep.json"
        self.subprocess = types.SimpleNamespace(
            Popen=self._popen, run=self._run, STDOUT=subprocess.STDOUT,
            DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired)
        self.time = types.SimpleNamespace(time=lambda: self.now, sleep=self._sleep,
                                          strftime=self._strftime,
                                          gmtime=lambda *a: None)

    def _sleep(self, secs):
        self.slept.append(secs)
        self.now += secs

    def _strftime(self, fmt, _t=None):
        mm, ss = divmod(int(self.now - 1000.0), 60)
        if fmt == "%Y%m%dT%H%M%SZ":     # the fallback file's stamp: a file name
            return "20260916T12%02d%02dZ" % (mm, ss)
        return "2026-09-16 12:%02d:%02d" % (mm, ss)

    def list_processes(self):
        self.process_listings += 1
        if self.processes is None:
            return None
        ours = [{"pid": p.pid, "ppid": 999, "name": os.path.basename(p.cmd[0]),
                 "cmdline": " ".join(p.cmd)}
                for p in self.launched if p.returncode is None]
        return [dict(p) for p in self.processes] + ours

    def _popen(self, cmd, stdout=None, stderr=None, **kw):
        if self.popen_error is not None:
            raise self.popen_error
        env = kw.get("env") or {}
        if "GENIE_PORT" in env:
            port = int(env["GENIE_PORT"])
        else:
            port = int(cmd[cmd.index("--host") + 1].rsplit(":", 1)[1])
        self.opened.append((getattr(stdout, "name", stdout), stderr))
        proc = _Proc(self, 4000 + len(self.launched), port, list(cmd))
        proc.kw = kw
        if port in self.dies_at_start:
            code, said = self.dies_at_start[port]
            stdout.write(said)
            stdout.flush()
            proc.returncode = code
        elif port not in self.never_answers:
            self.listening.add(port)
        self.launched.append(proc)
        self.events.append(("launch", proc.pid))
        return proc

    def _run(self, cmd, **kw):
        self.ran.append(list(cmd))
        if cmd[0] == "taskkill" and "/PID" in cmd:
            pid = int(cmd[cmd.index("/PID") + 1])
            for proc in self.launched:
                if proc.pid == pid:
                    proc.end(1)
            self.events.append(("kill", pid))
            if self.on_kill is not None:
                self.on_kill()
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def post(self, base, path, payload, timeout):
        self.posts.append((base, payload))
        if self.post_hook is not None:
            got = self.post_hook(base, payload)
            if got is not None:
                return got
        cap = payload["max_completion_tokens"]
        body = {"choices": [{"message": {"content": "w " * cap}}],
                "usage": {"prompt_tokens": 263, "completion_tokens": cap}}
        return body, 0.30 + 0.05 * cap

    def load_tokenizer(self, bundle_dir):
        self.tokenizer_loads.append(bundle_dir)
        return _Tok()

    def kills(self):
        """Every command run that could end a process."""
        return [c for c in self.ran
                if c[0] == "taskkill" or any("Stop-Process" in str(x) for x in c)]


@pytest.fixture
def world(bs, monkeypatch, tmp_path):
    w = World(tmp_path)
    monkeypatch.setattr(bs, "subprocess", w.subprocess)
    monkeypatch.setattr(bs, "time", w.time)
    monkeypatch.setattr(bs, "sys", types.SimpleNamespace(
        platform="win32", executable="python.exe", exit=sys.exit))
    monkeypatch.setattr(bs, "port_open", lambda port: port in w.listening)
    monkeypatch.setattr(bs, "port_owner", lambda port: w.owners.get(port))
    monkeypatch.setattr(bs, "load_tokenizer", w.load_tokenizer)
    monkeypatch.setattr(bs, "list_processes", w.list_processes)
    monkeypatch.setattr(bs.be, "post_timed", w.post)
    monkeypatch.setattr(bs, "BUNDLE_DIR", str(w.bundle))
    monkeypatch.setattr(bs, "SDK_DIR", str(w.sdk))
    monkeypatch.setattr(bs, "GENIEX", str(w.geniex))
    return w


def _main(bs, monkeypatch, world, *argv, out=True):
    args = ["bench_servers.py", *argv]
    if out:
        args += ["--out", str(world.out)]
    monkeypatch.setattr(sys, "argv", args)
    return bs.main()


def _result(world):
    return json.loads(world.out.read_text(encoding="utf-8"))


ARMS = {"ours": {"port": 8123, "model": "a"}, "geniex": {"port": 18181, "model": "b"}}


# --- the environment is checked before anything is spent --------------------
# GENIE_SDK_DIR was read at import and never looked at, while the `ours` arm
# exits at load without it -- into DEVNULL, so the failure surfaced 240 s
# later, per pass, as a reasonless "FAILED to start".

def test_an_unset_sdk_dir_is_refused_by_name(bs, tmp_path):
    problem = bs.env_problem(str(tmp_path), "")
    assert "GENIE_SDK_DIR" in problem and "ours" in problem


def test_an_unset_bundle_dir_is_refused_first(bs):
    assert "GENIE_BUNDLE_DIR" in bs.env_problem("", "")


def test_an_sdk_dir_that_is_not_a_directory_is_refused_with_the_path(bs, tmp_path):
    missing = str(tmp_path / "no-such-sdk")
    problem = bs.env_problem(str(tmp_path), missing)
    assert "GENIE_SDK_DIR" in problem and missing in problem


def test_a_complete_environment_is_no_problem(bs, tmp_path):
    assert bs.env_problem(str(tmp_path), str(tmp_path)) is None


def test_main_refuses_without_the_sdk_before_anything_is_launched(bs, world, monkeypatch):
    monkeypatch.setattr(bs, "SDK_DIR", "")
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world)
    assert "GENIE_SDK_DIR" in str(e.value)
    assert world.launched == [] and world.ran == [] and world.tokenizer_loads == []
    assert not world.out.exists()


# The `ours` arm's prerequisites were checked up front and geniex's were not,
# so a box with no GenieX CLI ran the WHOLE sweep one-armed: every geniex
# launch "FAILED to start -- skipping", the table "geniex  no data", and then
# "complete" with rc 0. Same shape for --tokens under the decode floor, which
# this change newly applies to every sample: 0 rows, "complete", rc 0.

def test_main_refuses_a_missing_geniex_before_anything_is_launched(
        bs, world, monkeypatch, tmp_path):
    missing = str(tmp_path / "no-such-geniex.exe")
    monkeypatch.setattr(bs, "GENIEX", missing)
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world)
    assert missing in str(e.value) and "GENIEX_EXE" in str(e.value)
    assert "geniex" in str(e.value), "the arm it names, not just a path"
    assert world.launched == [] and world.ran == [] and world.tokenizer_loads == []
    assert not world.out.exists(), "nothing of the box was spent"


def test_a_geniex_found_on_the_path_is_not_refused(bs, world, monkeypatch):
    # GENIEX_EXE may be a bare name: Popen resolves one on PATH, so isfile
    # alone would refuse a perfectly good install.
    monkeypatch.setattr(bs, "GENIEX", "geniex.exe")
    monkeypatch.setattr(bs, "shutil", types.SimpleNamespace(
        which=lambda name: str(world.geniex) if name == "geniex.exe" else None))
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 0
    assert [p.cmd[0] for p in world.launched][1] == "geniex.exe"


def test_main_refuses_tokens_under_the_decode_floor_before_anything_is_launched(
        bs, world, monkeypatch):
    # sample() refuses every window under the floor, so --tokens 12 measures
    # nothing -- after paying two requests per depth per arm-run for it.
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world, "--tokens", "12")
    said = str(e.value)
    assert "--tokens 12" in said and "floor of 16 steps" in said
    assert "GENIE_MIN_DECODE_STEPS" in said, "the variable that sets it"
    assert "REFUSED" in said and "spend the box" in said
    assert world.launched == [] and world.tokenizer_loads == []
    assert not world.out.exists()


def test_tokens_at_the_floor_exactly_are_accepted(bs, world, monkeypatch):
    # The refusal is strictly below the floor: a window OF the floor is the
    # shortest one sample() accepts, and refusing it here would contradict it.
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250",
                 "--tokens", str(bs.be.MIN_DECODE_STEPS)) == 0
    assert len(_result(world)["rows"]) == 2


# --- --out is never overwritten silently, and is refused BEFORE the sweep ---

def test_main_refuses_an_existing_out_before_the_sweep(bs, world, monkeypatch):
    world.out.write_text("twenty minutes of someone's box", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world)
    assert str(world.out) in str(e.value) and "--force" in str(e.value)
    assert world.out.read_text(encoding="utf-8") == "twenty minutes of someone's box"
    assert world.launched == [] and world.posts == [] and world.tokenizer_loads == []


def test_force_overwrites_an_existing_out(bs, world, monkeypatch):
    world.out.write_text("stale", encoding="utf-8")
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250", "--force") == 0
    assert _result(world)["tool"] == "bench_servers"


def test_main_refuses_an_out_directory_that_does_not_exist(bs, world, monkeypatch, tmp_path):
    # The arm logs are opened there at the first launch; found out now, not
    # after a model load.
    missing = tmp_path / "nowhere" / "sweep.json"
    monkeypatch.setattr(sys, "argv", ["bench_servers.py", "--out", str(missing)])
    with pytest.raises(SystemExit) as e:
        bs.main()
    assert "not a directory" in str(e.value) and str(tmp_path / "nowhere") in str(e.value)
    assert world.launched == []


def test_an_empty_out_writes_no_results_and_logs_to_the_cwd(bs, world, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250",
                 "--out", "", out=False) == 0
    assert not world.out.exists()
    assert (tmp_path / "bench_servers-ours.log").exists()


# --- co-tenants: this run stops what it started and nothing else ------------

def test_a_foreign_listener_is_refused_with_its_pid_and_port(bs, monkeypatch):
    monkeypatch.setattr(bs, "port_owner", lambda port: 4242 if port == 8123 else None)
    monkeypatch.setattr(bs, "port_open", lambda port: port == 8123)
    problem = bs.foreign_listener(ARMS)
    assert "4242" in problem and "8123" in problem
    assert "--ours-port" in problem, "the way out is another port, never a kill"


def test_the_refusal_names_the_arm_whose_port_is_taken(bs, monkeypatch):
    monkeypatch.setattr(bs, "port_owner", lambda port: 5150 if port == 18181 else None)
    monkeypatch.setattr(bs, "port_open", lambda port: port == 18181)
    problem = bs.foreign_listener(ARMS)
    assert "5150" in problem and "18181" in problem and "--geniex-port" in problem


def test_a_listener_whose_owner_cannot_be_named_is_still_refused(bs, monkeypatch):
    # Off-Windows, or PowerShell failed: port_owner is None, the port answers.
    # On Windows a second bind SUCCEEDS and the old process keeps answering,
    # so starting behind it would measure someone else's server.
    monkeypatch.setattr(bs, "port_owner", lambda port: None)
    monkeypatch.setattr(bs, "port_open", lambda port: port == 8123)
    problem = bs.foreign_listener(ARMS)
    assert "8123" in problem and "could not be identified" in problem


def test_free_ports_are_no_problem(bs, monkeypatch):
    monkeypatch.setattr(bs, "port_owner", lambda port: None)
    monkeypatch.setattr(bs, "port_open", lambda port: False)
    assert bs.foreign_listener(ARMS) is None


def test_main_refuses_a_foreign_listener_before_pass_1_and_kills_nothing(
        bs, world, monkeypatch):
    # The resident genie_server on the launcher's default 8123: the README's
    # normal state, and what the old stop_all force-stopped before pass 1.
    world.owners[8123] = 4242
    world.listening.add(8123)
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world)
    assert "4242" in str(e.value) and "8123" in str(e.value)
    assert world.launched == [], "nothing may start behind someone else's listener"
    assert world.ran == [], "and nothing at all may be run against it"
    assert 8123 in world.listening, "the co-tenant is still serving"


def test_a_listener_that_appears_mid_run_ends_the_run_without_a_kill(
        bs, world, monkeypatch, capsys):
    def someone_starts_geniex():
        world.owners[18181] = 5150
        world.listening.add(18181)
    world.on_kill = None
    # It appears while `ours` is being measured, i.e. before the second arm-run.
    world.post_hook = lambda base, payload: someone_starts_geniex()
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    out = capsys.readouterr().out
    assert "5150" in out and "18181" in out
    ours = world.launched[0]
    assert world.kills() == [["taskkill", "/PID", str(ours.pid), "/T", "/F"]]
    assert len(world.launched) == 1, "the geniex arm was never started behind it"
    assert 18181 in world.listening
    got = _result(world)
    assert got["outcome"].startswith("refused:") and "5150" in got["outcome"]
    assert [r["arm"] for r in got["rows"]] == ["ours"], "what was measured is kept"


def test_stop_all_stops_only_the_children_this_run_started(bs, world, capsys):
    children = {}
    child, problem = bs.start("ours", ARMS["ours"], str(world.bundle), children)
    assert problem is None
    assert children == {"ours": child}, "start() registers what it launched"
    world.listening.add(18181)          # someone else's geniex, left alone
    bs.stop_all(children)
    pid = child["proc"].pid
    assert world.ran == [["taskkill", "/PID", str(pid), "/T", "/F"]]
    flat = " ".join(" ".join(c) for c in world.ran)
    assert "/IM" not in flat and "Stop-Process" not in flat and "geniex" not in flat
    assert children == {}, "emptied, so the next stop_all has nothing of ours to find"
    assert 18181 in world.listening
    assert "stopping ours (pid %d)" % pid in capsys.readouterr().out


def test_stop_all_with_no_children_runs_nothing_and_does_not_settle(bs, world):
    bs.stop_all({})
    assert world.ran == [] and world.slept == []


def test_stop_all_leaves_an_already_dead_child_unkilled(bs, world):
    world.dies_at_start[8123] = (1, "boom\n")
    child, problem = bs.start("ours", ARMS["ours"], str(world.bundle), {})
    assert problem is not None
    children = {"ours": child}
    bs.stop_all(children)
    assert world.ran == [] and children == {}


def test_a_port_that_still_answers_after_our_child_is_gone_is_reported_not_killed(
        bs, world, capsys):
    child, _ = bs.start("ours", ARMS["ours"], str(world.bundle), {})
    world.on_kill = lambda: world.listening.add(8123)   # a second process had it too
    bs.stop_all({"ours": child})
    assert "still answers" in capsys.readouterr().out
    assert len(world.kills()) == 1, "only our own pid was ever killed"
    assert 8123 in world.listening


def test_a_child_that_will_not_die_is_reported_and_stopping_does_not_raise(bs, world, capsys):
    # stop_all runs from main()'s `finally`: an exception here would replace
    # the results file with a traceback.
    child, _ = bs.start("ours", ARMS["ours"], str(world.bundle), {})

    def wait(timeout=None):
        raise subprocess.TimeoutExpired("python.exe", timeout)
    child["proc"].wait = wait
    bs.stop_all({"ours": child})
    assert "pid %d did not exit" % child["proc"].pid in capsys.readouterr().out


def test_off_windows_the_child_is_ended_through_its_handle(bs, world, monkeypatch):
    child, _ = bs.start("ours", ARMS["ours"], str(world.bundle), {})
    monkeypatch.setattr(bs, "sys", types.SimpleNamespace(platform="linux"))
    bs.stop_all({"ours": child})
    assert child["proc"].terminated and world.ran == []


def test_port_owner_reads_the_listening_pid_and_runs_nothing_that_stops_a_process(
        bs, monkeypatch):
    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="4242\r\n", stderr="")

    monkeypatch.setattr(bs, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(bs, "subprocess", types.SimpleNamespace(run=fake_run))
    assert bs.port_owner(8123) == 4242
    script = ran[0][-1]
    assert "-LocalPort 8123" in script and "Listen" in script
    assert "Stop-Process" not in script and "taskkill" not in script


def test_port_owner_is_none_when_nothing_listens_or_the_query_fails(bs, monkeypatch):
    monkeypatch.setattr(bs, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(bs, "subprocess", types.SimpleNamespace(
        run=lambda cmd, **kw: types.SimpleNamespace(returncode=0, stdout="\r\n", stderr="")))
    assert bs.port_owner(8123) is None

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired("powershell.exe", 30)
    monkeypatch.setattr(bs, "subprocess", types.SimpleNamespace(run=boom))
    assert bs.port_owner(8123) is None


def test_port_owner_is_none_off_windows_without_running_anything(bs, monkeypatch):
    def boom(cmd, **kw):
        raise AssertionError("no PowerShell off Windows")
    monkeypatch.setattr(bs, "sys", types.SimpleNamespace(platform="linux"))
    monkeypatch.setattr(bs, "subprocess", types.SimpleNamespace(run=boom))
    assert bs.port_owner(8123) is None


def test_port_open_is_a_connect_to_loopback(bs, monkeypatch):
    seen = []

    def connect(addr, timeout=None):
        seen.append(addr)
        return types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(bs, "socket", types.SimpleNamespace(create_connection=connect))
    assert bs.port_open(8123) is True and seen == [("127.0.0.1", 8123)]

    def refused(addr, timeout=None):
        raise ConnectionRefusedError("nobody home")
    monkeypatch.setattr(bs, "socket", types.SimpleNamespace(create_connection=refused))
    assert bs.port_open(8123) is False


# --- a child that dies fails fast, with its exit code and its own words -----

def _log(tmp_path, lines):
    path = tmp_path / "bench_servers-ours.log"
    path.write_text("".join("%s\n" % ln for ln in lines), encoding="utf-8")
    return str(path)


def test_wait_port_fails_at_once_when_the_child_has_exited(bs, world, tmp_path):
    log = _log(tmp_path, ["loading", "set GENIE_SDK_DIR to the QAIRT SDK root"])
    dead = _Proc(world, 4001, 8123, ["python"], returncode=3)
    problem = bs.wait_port(8123, proc=dead, log=log)
    assert "code 3" in problem and "8123" in problem
    assert "set GENIE_SDK_DIR to the QAIRT SDK root" in problem, "the child's own reason"
    assert world.slept == [], "not one sleep, let alone the 240 s this used to cost"


def test_a_dead_child_is_a_failure_even_when_the_port_answers(bs, world, tmp_path):
    # Its bind lost to someone else's listener and it exited: the port answers,
    # but not with OUR server. Polling before probing is what tells them apart.
    world.listening.add(8123)
    dead = _Proc(world, 4001, 8123, ["python"], returncode=1)
    assert "code 1" in bs.wait_port(8123, proc=dead, log=None)


def test_a_child_that_dies_during_the_wait_is_caught_on_the_next_turn(bs, world):
    proc = _Proc(world, 4001, 8123, ["python"])
    polls = []

    def poll():
        polls.append(1)
        if len(polls) == 3:
            proc.returncode = 7
        return proc.returncode
    proc.poll = poll
    problem = bs.wait_port(8123, proc=proc)
    assert "code 7" in problem
    assert sum(world.slept) < 10, "two 2 s turns, not the timeout"


def test_wait_port_is_none_once_the_port_answers(bs, world):
    proc = _Proc(world, 4001, 8123, ["python"])
    real_sleep = world.time.sleep

    def sleep(secs):
        real_sleep(secs)
        if len(world.slept) == 2:
            world.listening.add(8123)
    world.time.sleep = sleep
    assert bs.wait_port(8123, proc=proc) is None
    assert len(world.slept) == 2


def test_wait_port_times_out_saying_the_process_is_still_alive(bs, world, tmp_path):
    log = _log(tmp_path, ["still loading the bundle"])
    proc = _Proc(world, 4001, 8123, ["python"])
    problem = bs.wait_port(8123, proc=proc, log=log)
    assert "after %d s" % bs.START_SECS in problem and "still alive" in problem
    assert "still loading the bundle" in problem
    assert sum(world.slept) >= bs.START_SECS


def test_log_tail_is_the_last_lines_and_says_when_there_are_none(bs, tmp_path):
    log = _log(tmp_path, ["line %d" % i for i in range(1, 21)])
    tail = bs.log_tail(log, lines=3)
    assert "line 18" in tail and "line 20" in tail and "line 17" not in tail
    assert log in tail
    empty = tmp_path / "empty.log"
    empty.write_text("", encoding="utf-8")
    assert "is empty" in bs.log_tail(str(empty))
    assert bs.log_tail(None) == ""
    assert bs.log_tail(str(tmp_path / "never-written.log")) == ""


# What a child writes is not ours to choose. geniex is a Go binary whose
# failure messages are emoji-prefixed UTF-8; a redirected Python child writes
# the ANSI code page, so its em-dash is the single byte 0x97.
_WARNING_SIGN, _CHECK = chr(0x26A0), chr(0x2713)


def _piped_stdout():
    """stdout as it is under `| Tee-Object` or an agent's shell: cp1252, strict."""
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", errors="strict")


def test_log_tail_is_printable_whatever_stdout_can_encode(bs, tmp_path):
    path = tmp_path / "bench_servers-geniex.log"
    path.write_bytes(("%s Oops. Runtime failed to load\n" % _WARNING_SIGN).encode("utf-8")
                     + b"no usable Hexagon " + bytes([0x97]) + b" giving up\n")
    tail = bs.log_tail(str(path))
    lines = tail.split(str(path), 1)[1]
    assert lines.isascii(), "every line escaped, whatever the child wrote"
    assert "u26a0 Oops. Runtime failed to load" in lines, "escaped, not dropped"
    assert "no usable Hexagon" in lines and "x97 giving up" in lines, (
        "a byte that is not UTF-8 is shown as the byte, not as U+FFFD")
    raw, piped = _piped_stdout()
    print("   FAILED to start -- skipping: exited with code 1%s" % tail, file=piped, flush=True)
    assert b"Oops. Runtime failed to load" in raw.getvalue()


def test_a_log_stdout_cannot_encode_does_not_end_the_sweep(bs, world, monkeypatch):
    # The failed-start line embeds the log tail and is printed inside main()'s
    # loop-wide try: on a piped cp1252 stdout one emoji raised
    # UnicodeEncodeError, `except Exception` took it for a bug, and the sweep
    # ended with outcome "error: UnicodeEncodeError", no failed_starts entry
    # and the other arm never run. bs.sys is the World's namespace and has no
    # stdout, so forgiving_stdout is a no-op here: the tail stands on its own.
    raw, piped = _piped_stdout()
    monkeypatch.setattr(sys, "stdout", piped)
    world.dies_at_start[18181] = (1, "%s Oops. Model failed to load %s\n"
                                  % (_WARNING_SIGN, _CHECK))
    # rc 1 and "incomplete" because geniex, this sweep's only geniex run,
    # produced nothing -- what is under test is that the run did not end in
    # "error: UnicodeEncodeError" with the OTHER arm never measured.
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    piped.flush()
    assert b"FAILED to start -- skipping: exited with code 1" in raw.getvalue()
    assert b"Oops. Model failed to load" in raw.getvalue()
    got = _result(world)
    assert got["outcome"] == "incomplete: no rows for geniex"
    assert [f["arm"] for f in got["failed_starts"]] == ["geniex"]
    assert [r["arm"] for r in got["rows"]] == ["ours"], "the other arm was still measured"


def test_a_server_message_stdout_cannot_encode_does_not_end_the_sweep(
        bs, world, monkeypatch):
    # The same hazard by another door: a request's failure reason carries the
    # SERVER's words into the "warmup failed" and SKIP lines. main() makes
    # stdout escape what it cannot encode before anything is printed.
    raw, piped = _piped_stdout()
    monkeypatch.setattr(sys, "stdout", piped)
    monkeypatch.setattr(bs.sys, "stdout", piped, raising=False)

    def geniex_is_unwell(base, payload):
        if base.endswith(":18181"):
            return None, "HTTP 500 -- %s Oops. Model failed to load" % _WARNING_SIGN
        return None
    world.post_hook = geniex_is_unwell
    # rc 1 for the same reason as above: geniex answered nothing usable at any
    # depth. The point here is the escaping, and that `ours` was measured.
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    piped.flush()
    assert b"warmup failed: request failed: HTTP 500" in raw.getvalue()
    assert b"u26a0 Oops. Model failed to load" in raw.getvalue(), "escaped, still said"
    got = _result(world)
    assert got["outcome"] == "incomplete: no rows for geniex"
    assert [r["arm"] for r in got["rows"]] == ["ours"]


def test_forgiving_stdout_leaves_a_stream_it_cannot_reconfigure_alone(bs, monkeypatch):
    # A capture object, an IDE's stream, or (in these tests) no stdout on the
    # module's sys at all: never a reason to stop before the run has begun.
    monkeypatch.setattr(bs, "sys", types.SimpleNamespace())
    bs.forgiving_stdout()
    monkeypatch.setattr(bs, "sys", types.SimpleNamespace(stdout=object()))
    bs.forgiving_stdout()


def test_start_keeps_the_childs_output_in_a_per_arm_log(bs, world, tmp_path):
    child, problem = bs.start("ours", ARMS["ours"], str(tmp_path), {})
    assert problem is None
    log = os.path.join(str(tmp_path), "bench_servers-ours.log")
    assert child["log"] == log and child["port"] == 8123
    assert world.opened == [(log, subprocess.STDOUT)], "stderr rides with stdout into the log"
    assert subprocess.DEVNULL not in world.opened[0]
    proc = child["proc"]
    assert proc.cmd == ["python.exe", os.path.join("src", "genie_server.py")]
    assert proc.kw["cwd"] == bs.REPO and proc.kw["env"]["GENIE_PORT"] == "8123"
    with open(log, encoding="utf-8") as f:
        header = f.read()
    assert "genie_server.py" in header and "===" in header


def test_each_arm_has_its_own_log_and_geniex_is_told_its_port(bs, world, tmp_path):
    child, _ = bs.start("geniex", ARMS["geniex"], str(tmp_path), {})
    assert child["log"].endswith("bench_servers-geniex.log")
    cmd = child["proc"].cmd
    assert cmd[0] == bs.GENIEX and cmd[cmd.index("--host") + 1] == "127.0.0.1:18181"


def test_a_relaunch_appends_to_the_arms_log(bs, world, tmp_path):
    world.dies_at_start[8123] = (1, "first launch said this\n")
    bs.start("ours", ARMS["ours"], str(tmp_path), {})
    bs.start("ours", ARMS["ours"], str(tmp_path), {})
    text = (tmp_path / "bench_servers-ours.log").read_text(encoding="utf-8")
    assert text.count("first launch said this") == 2 and text.count("===") == 4


def test_start_reports_a_launch_failure_instead_of_raising(bs, world, tmp_path):
    # A missing geniex.exe used to traceback out of main() with nothing written.
    world.popen_error = FileNotFoundError(2, "The system cannot find the file specified")
    child, problem = bs.start("geniex", ARMS["geniex"], str(tmp_path), {})
    assert child is None
    assert "could not launch" in problem and bs.GENIEX in problem


def test_start_hands_back_a_child_whose_port_never_answered(bs, world, tmp_path):
    # So the caller stops what it started instead of leaving it on the Hexagon.
    world.dies_at_start[8123] = (3, "no usable Hexagon\n")
    child, problem = bs.start("ours", ARMS["ours"], str(tmp_path), {})
    assert child is not None and child["proc"].returncode == 3
    assert "code 3" in problem and "no usable Hexagon" in problem


def test_main_skips_an_arm_that_died_at_startup_and_says_why(bs, world, monkeypatch, capsys):
    world.dies_at_start[8123] = (1, "set GENIE_SDK_DIR to the QAIRT SDK root\n")
    # rc 1: one pass, so the arm that died has no rows at all (see the
    # no-rows tests below). The skip itself is what is under test.
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    out = capsys.readouterr().out
    assert "FAILED to start -- skipping: exited with code 1" in out
    assert "set GENIE_SDK_DIR to the QAIRT SDK root" in out
    assert sum(world.slept) < bs.START_SECS, "a dead child must not cost the port wait"
    assert [r["arm"] for r in _result(world)["rows"]] == ["geniex"], "the other arm still ran"


def test_an_arm_that_is_alive_but_deaf_is_stopped_before_the_next_one_starts(
        bs, world, monkeypatch, capsys):
    # The port wait ran out with the process still up -- loaded onto the
    # Hexagon, perhaps, and not serving. It is still OURS, and single-flight
    # means the next arm must not start until it is gone.
    world.never_answers.add(8123)
    # rc 1 for the arm with no rows; the stop-before-the-next-start is the
    # subject.
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    out = capsys.readouterr().out
    assert "FAILED to start -- skipping: nothing listening on port 8123 after 240 s" in out
    ours, geniex = world.launched
    assert ours.returncode is not None
    assert world.events.index(("kill", ours.pid)) < world.events.index(("launch", geniex.pid))


def test_a_failed_start_is_in_the_artifact_not_only_on_the_screen(bs, world, monkeypatch):
    # `outcome` says how the LOOP ended and, when the loop ran to its end with
    # an arm empty, that too -- but neither is a REASON. Each failed arm-run
    # carries its own into the file, or a reader has the gap and no cause.
    world.dies_at_start[8123] = (1, "set GENIE_SDK_DIR to the QAIRT SDK root\n")
    _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250")
    got = _result(world)
    assert got["outcome"] == "incomplete: no rows for ours"
    (failed,) = got["failed_starts"]
    assert (failed["arm"], failed["run"], failed["pass"]) == ("ours", 1, 1)
    assert "exited with code 1" in failed["reason"]


# --- a child that dies AFTER its port answered ------------------------------
# An answering port is a TCP connect, not a loaded model: geniex binds first
# and loads on its first request, and genie_server spent one pass of this repo
# LISTENING ahead of its load too (it still binds ahead of it, but listens only
# once the model is resident). wait_port's fail-fast cannot see that death.
# What it looked like: "warmup failed: ConnectionResetError", a SKIP per depth,
# then "wrote sweep.json (1 samples, complete)" with failed_starts [] and
# exit 0 -- the complete-file-with-an-arm-missing-and-no-reason that
# failed_starts exists to prevent.

def _dies_on_request(world, arm, nth, code, said):
    """A post_hook: `arm`'s server exits on its `nth` request, having written
    `said` to its log, and the request fails as a real one does."""
    port = ARMS[arm]["port"]
    seen = []

    def hook(base, payload):
        if not base.endswith(":%d" % port):
            return None
        seen.append(1)
        if len(seen) < nth:
            return None
        proc = [p for p in world.launched if p.port == port][-1]
        if proc.returncode is None:
            with open(os.path.join(str(world.out.parent), "bench_servers-%s.log" % arm),
                      "a", encoding="utf-8") as f:
                f.write(said + "\n")
            proc.end(code)
        return None, "ConnectionResetError: [WinError 10054] forcibly closed"
    return hook


def test_death_of_is_none_while_the_child_runs_and_costs_at_most_the_grace(bs, world):
    proc = _Proc(world, 4001, 8123, ["python"])
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        raise subprocess.TimeoutExpired("python.exe", timeout)
    proc.wait = wait
    assert bs.death_of({"proc": proc, "log": None, "port": 8123}) is None
    assert waits == [bs.DEATH_GRACE_SECS], "bounded: never a wait with no timeout"


def test_death_of_gives_the_exit_code_the_port_and_the_childs_own_words(bs, world, tmp_path):
    log = _log(tmp_path, ["loading", "GenieDialog_create failed, status=1"])
    dead = _Proc(world, 4001, 8123, ["python"], returncode=1)
    problem = bs.death_of({"proc": dead, "log": log, "port": 8123})
    assert "exited with code 1" in problem and "after port 8123 had answered" in problem
    assert "GenieDialog_create failed, status=1" in problem


def test_death_of_waits_out_a_child_that_is_still_on_its_way_down(bs, world):
    # Measured on this box: a dying server's connections are reset BEFORE its
    # handle reports the exit -- poll() was still None straight after the reset
    # 40 times of 40. An immediate poll() files a dead arm as a live one.
    proc = _Proc(world, 4001, 8123, ["python"])

    def wait(timeout=None):
        proc.returncode = 1             # it finishes exiting inside the grace
        return 1
    proc.wait = wait
    assert proc.poll() is None, "not yet dead when first asked"
    assert "exited with code 1" in bs.death_of({"proc": proc, "log": None, "port": 8123})


def test_an_arm_that_dies_in_its_load_after_the_port_answered_is_a_failed_start(
        bs, world, monkeypatch, capsys):
    world.post_hook = _dies_on_request(world, "ours", 1, 1,
                                       "GenieDialog_create failed, status=1")
    # rc 1: `ours` died in its only run, so it has no rows.
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250,1500") == 1
    out = capsys.readouterr().out
    assert "warmup failed: request failed: ConnectionResetError" in out
    assert "FAILED to start -- skipping: exited with code 1 after port 8123" in out
    assert "GenieDialog_create failed, status=1" in out, "the reason, not only the reset"
    to_ours = [p for base, p in world.posts if base.endswith(":8123")]
    assert len(to_ours) == 1, "the warmup only: a dead arm's depths are not walked"
    assert "SKIP" not in out
    got = _result(world)
    (failed,) = got["failed_starts"]
    assert (failed["arm"], failed["run"], failed["pass"]) == ("ours", 1, 1)
    assert "exited with code 1" in failed["reason"]
    assert "GenieDialog_create failed" in failed["reason"]
    assert got["died_mid_run"] == []
    assert sorted({r["arm"] for r in got["rows"]}) == ["geniex"], "the other arm still ran"
    assert world.kills() == [["taskkill", "/PID", str(world.launched[1].pid), "/T", "/F"]], (
        "the dead arm is not killed again, and geniex is still stopped at the end")


def test_a_failed_warmup_with_the_child_still_up_goes_on_to_the_depths(
        bs, world, monkeypatch, capsys):
    # Not every failed warmup is a death: the server may simply have refused
    # that one request, and a sample then says why in its own SKIP line.
    def first_request_fails(base, payload):
        if base.endswith(":8123") and len(world.posts) == 1:
            return None, "HTTP 500 -- busy"
        return None
    world.post_hook = first_request_fails
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 0
    assert "warmup failed: request failed: HTTP 500" in capsys.readouterr().out
    got = _result(world)
    assert got["failed_starts"] == [] and got["died_mid_run"] == []
    assert sorted(r["arm"] for r in got["rows"]) == ["geniex", "ours"]


def test_an_arm_that_dies_mid_run_is_never_a_complete_arm_run(
        bs, world, monkeypatch, capsys):
    # Warmup, d250 x2, then the first d1500 request finds the server gone.
    world.post_hook = _dies_on_request(world, "ours", 4, 3, "QNN: HTP session lost")
    assert _main(bs, monkeypatch, world, "--passes", "1",
                 "--depths", "250,1500,3000") == 0
    out = capsys.readouterr().out
    assert "ours DIED mid-run -- skipping its remaining depths: exited with code 3" in out
    assert "QNN: HTP session lost" in out
    to_ours = [p for base, p in world.posts if base.endswith(":8123")]
    assert len(to_ours) == 4, "d3000 is never asked of a server that is gone"
    got = _result(world)
    (died,) = got["died_mid_run"]
    assert (died["arm"], died["run"], died["pass"], died["depth"]) == ("ours", 1, 1, 1500)
    assert "exited with code 3" in died["reason"] and "HTP session lost" in died["reason"]
    assert got["failed_starts"] == [], "it started; it is listed as what it was"
    assert [(r["arm"], r["depth"]) for r in got["rows"] if r["arm"] == "ours"] == [
        ("ours", 250)], "what it measured before it died is kept"
    assert len([r for r in got["rows"] if r["arm"] == "geniex"]) == 3
    # `outcome` is the loop's, so the closing line carries the gap.
    assert got["outcome"] == "complete"
    closing = out[out.index("=== medians"):]
    assert "NOT every arm-run ran to its end: of 2, 1 died mid-run (ours)" in closing


def test_a_failed_sample_with_the_child_still_up_is_only_a_skip(bs, world, monkeypatch, capsys):
    def one_bad_request(base, payload):
        if base.endswith(":8123") and len(world.posts) == 2:
            return None, "HTTP 500 -- busy"
        return None
    world.post_hook = one_bad_request
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250,1500") == 0
    out = capsys.readouterr().out
    assert "SKIP" in out and "DIED" not in out and "NOT every arm-run" not in out
    got = _result(world)
    assert got["died_mid_run"] == []
    assert [(r["arm"], r["depth"]) for r in got["rows"] if r["arm"] == "ours"] == [
        ("ours", 1500)], "the next depth is still measured"


def test_the_closing_line_counts_what_did_not_run_to_its_end_and_is_silent_otherwise(bs):
    assert bs.incomplete_runs([], [], 6) == ""
    line = bs.incomplete_runs([{"arm": "geniex"}, {"arm": "geniex"}], [{"arm": "ours"}], 6)
    assert "of 6, 2 failed to start (geniex) and 1 died mid-run (ours)" in line
    assert "1 failed to start (ours)" in bs.incomplete_runs([{"arm": "ours"}], [], 2)


def test_a_failed_start_is_said_again_beside_the_table(bs, world, monkeypatch, capsys):
    world.dies_at_start[8123] = (1, "set GENIE_SDK_DIR to the QAIRT SDK root\n")
    _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250")
    out = capsys.readouterr().out
    closing = out[out.index("=== medians"):]
    assert "ours    no data" in closing
    assert "NOT every arm-run ran to its end: of 2, 1 failed to start (ours)" in closing


def test_a_clean_run_has_no_gap_line_and_an_empty_died_mid_run(bs, world, monkeypatch, capsys):
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 0
    out = capsys.readouterr().out
    assert "NOT every arm-run" not in out and "NOT an A/B" not in out
    got = _result(world)
    assert got["failed_starts"] == [] and got["died_mid_run"] == []


# --- an arm with NO rows is not an A/B, however complete the loop -----------
# The loop running to its end used to be the whole of `outcome`, so a sweep
# with one side empty wrote "complete" and returned 0 -- which a wrapper
# script branching on the exit code takes for a finished comparison.

def test_arms_with_no_rows_names_the_empty_ones_in_run_order(bs):
    arms = {"ours": {}, "geniex": {}}
    assert bs.arms_with_no_rows([], arms) == ["ours", "geniex"]
    assert bs.arms_with_no_rows([{"arm": "geniex"}], arms) == ["ours"]
    assert bs.arms_with_no_rows([{"arm": "ours"}], arms) == ["geniex"]
    assert bs.arms_with_no_rows([{"arm": "ours"}, {"arm": "geniex"}], arms) == []


def test_a_sweep_that_left_an_arm_empty_is_not_complete_and_exits_non_zero(
        bs, world, monkeypatch, capsys):
    world.dies_at_start[18181] = (1, "no usable Hexagon\n")
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    closing = capsys.readouterr().out
    closing = closing[closing.index("=== medians"):]
    assert "NOT an A/B: `geniex` produced no rows at all" in closing
    assert "nothing to compare" in closing and "non-zero" in closing
    got = _result(world)
    assert got["outcome"] == "incomplete: no rows for geniex"
    assert "complete" != got["outcome"]
    assert [r["arm"] for r in got["rows"]] == ["ours"]


def test_an_arm_empty_with_no_failed_run_at_all_is_still_caught(bs, world, monkeypatch, capsys):
    # Nothing failed to start and nothing died: every SAMPLE was refused (a
    # server that ignores the cap, so the 1-token and the N-token request come
    # back the same length and the delta is 0 steps). incomplete_runs() has
    # nothing to say about this one, which is why it is not the same check.
    def ignores_the_cap(base, payload):
        return ({"choices": [{"message": {"content": "w"}}],
                 "usage": {"prompt_tokens": 263, "completion_tokens": 1}}, 0.30)
    world.post_hook = ignores_the_cap
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    out = capsys.readouterr().out
    assert "early stop (0 of 120 steps)" in out
    assert "NOT every arm-run" not in out, "no arm-run ended early"
    assert "NOT an A/B: `ours` and `geniex` produced no rows at all" in out
    got = _result(world)
    assert got["rows"] == [] and got["failed_starts"] == []
    assert got["outcome"] == "incomplete: no rows for ours, geniex"


def test_an_arm_that_lost_only_some_of_its_runs_is_still_a_measured_arm(
        bs, world, monkeypatch, capsys):
    # The deliberate part, kept: one failed arm-run of two is a gap in the
    # table, not a missing side of the comparison. It is said on the screen
    # and in `failed_starts`, and the run still ends complete and returns 0.
    real_popen = world.subprocess.Popen

    def dies_on_its_first_launch_only(cmd, **kw):
        proc = real_popen(cmd, **kw)
        if proc.port == 18181:
            world.dies_at_start.pop(18181, None)
        return proc
    world.dies_at_start[18181] = (1, "transient: the HTP was still busy\n")
    monkeypatch.setattr(world.subprocess, "Popen", dies_on_its_first_launch_only)
    assert _main(bs, monkeypatch, world, "--passes", "2", "--depths", "250") == 0
    out = capsys.readouterr().out
    assert "NOT every arm-run ran to its end: of 4, 1 failed to start (geniex)" in out
    assert "NOT an A/B" not in out, "geniex was measured, just not in every run"
    got = _result(world)
    assert got["outcome"] == "complete"
    assert sorted({r["arm"] for r in got["rows"]}) == ["geniex", "ours"]


# --- try/finally: no arm of ours is left holding the Hexagon ----------------

def test_the_child_is_registered_before_its_port_is_waited_on(bs, world, monkeypatch, tmp_path):
    # The wait is up to START_SECS long and an exception out of it never
    # reaches start()'s `return`, so a handle that is only handed back is a
    # handle main()'s `finally` never sees.
    children, seen = {}, []

    def wait_port(port, secs=None, proc=None, log=None):
        seen.append(dict(children))
        raise KeyboardInterrupt
    monkeypatch.setattr(bs, "wait_port", wait_port)
    with pytest.raises(KeyboardInterrupt):
        bs.start("ours", ARMS["ours"], str(tmp_path), children)
    (during,) = seen
    assert during["ours"]["proc"] is world.launched[0], "already there when the wait began"
    assert children["ours"]["port"] == 8123 and children["ours"]["log"].endswith("-ours.log")


def test_a_launch_that_failed_registers_nothing(bs, world, tmp_path):
    world.popen_error = FileNotFoundError(2, "The system cannot find the file specified")
    children = {}
    bs.start("geniex", ARMS["geniex"], str(tmp_path), children)
    assert children == {}, "stop_all would otherwise reach for a proc that never existed"


def test_a_ctrl_c_during_the_port_wait_still_stops_the_child_just_launched(
        bs, world, monkeypatch, capsys):
    # Alive and deaf is the case a user DOES interrupt: 240 s of nothing. The
    # screen said "stopping the arms" and nothing was stopped -- the child kept
    # the Hexagon and port 8123, which the next run refuses as "pid N, which
    # this run did not start".
    world.never_answers.add(8123)
    real_sleep = world.time.sleep

    def sleep(secs):
        real_sleep(secs)
        if world.launched and secs == 2:        # wait_port's own 2 s turn
            raise KeyboardInterrupt
    world.time.sleep = sleep
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    (ours,) = world.launched
    assert ours.returncode is not None, "the child whose port was being waited on"
    assert world.kills() == [["taskkill", "/PID", str(ours.pid), "/T", "/F"]]
    assert _result(world)["outcome"] == "interrupted"
    assert "stopping ours (pid %d)" % ours.pid in capsys.readouterr().out


def test_a_ctrl_c_mid_run_stops_the_arm_and_still_writes_the_rows(
        bs, world, monkeypatch, capsys):
    def interrupt(base, payload):
        if len(world.posts) == 4:       # warmup, d250 x2, then the first d1500 request
            raise KeyboardInterrupt
    world.post_hook = interrupt
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250,1500") == 1
    ours = world.launched[0]
    assert ours.returncode is not None, "the arm this run started was stopped"
    assert world.kills() == [["taskkill", "/PID", str(ours.pid), "/T", "/F"]]
    assert world.listening == set()
    got = _result(world)
    assert got["outcome"] == "interrupted"
    assert [(r["arm"], r["depth"]) for r in got["rows"]] == [("ours", 250)]
    assert "interrupted" in capsys.readouterr().out


def test_a_bug_mid_run_still_stops_the_arm_and_writes_the_rows(
        bs, world, monkeypatch, capsys):
    real = bs.box_sample
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 3:             # run header, the d250 row, then this
            raise RuntimeError("sampler fell over")
        return real()
    monkeypatch.setattr(bs, "box_sample", flaky)
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250,1500") == 1
    assert world.launched[0].returncode is not None and world.listening == set()
    got = _result(world)
    assert got["outcome"] == "error: RuntimeError: sampler fell over"
    assert [(r["arm"], r["depth"]) for r in got["rows"]] == [("ours", 250)]
    captured = capsys.readouterr()
    assert "sampler fell over" in captured.err, "the traceback is still shown"


# --- one request shape, shared with bench_endpoint --------------------------

def _bench_endpoint_payload(bs, monkeypatch, model, prompt, cap):
    seen = {}

    def capture(base, path, payload, timeout):
        seen.update(path=path, payload=payload)
        return None, "captured"
    monkeypatch.setattr(bs.be, "post_timed", capture)
    bs.be.chat("http://127.0.0.1:8123", model, prompt, cap, 5)
    return seen


def test_the_request_is_bench_endpoints_plus_an_explicit_stream_false(bs, monkeypatch, capsys):
    # The two HTTP tools measured decode under different settings: this one
    # pinned nothing about thinking. Held against the OTHER tool's live body,
    # so a change there fails here instead of quietly splitting the columns.
    theirs = _bench_endpoint_payload(bs, monkeypatch, "m", "hello", 121)
    mine = bs.request_body("m", "hello", 121)
    assert mine.pop("stream") is False
    assert mine == theirs["payload"]
    assert theirs["path"] == "/v1/chat/completions"


def test_thinking_is_pinned_off_in_both_spellings_and_both_caps_are_sent(bs):
    body = bs.request_body("m", "hello", 121)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "none"
    assert body["cache_prompt"] is False
    assert body["max_tokens"] == 121 and body["max_completion_tokens"] == 121


def test_the_docstring_states_the_precedence_genie_server_actually_has(bs, gs):
    # request_body justifies sending both spellings by what each server does
    # with them, and said genie_server "prefers the modern one". It is the
    # other way round, and has been since before this tool sent both: held
    # against the server's own resolver so that a flip THERE fails here.
    assert gs._max_tokens({"max_tokens": 8, "max_completion_tokens": 16}) == 8
    assert gs._max_tokens({"max_tokens": 0, "max_completion_tokens": 16}) == 16
    doc = " ".join(bs.request_body.__doc__.split())
    assert "a legacy `max_tokens` that is set wins" in doc
    assert "absent or 0" in doc
    assert "prefers the modern" not in doc


class _Resp:
    def __init__(self, raw):
        self.raw = raw

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_urlopen(monkeypatch, bs, body=None, exc=None):
    seen = []

    def fake(req, timeout=None):
        seen.append((req, timeout))
        if exc is not None:
            raise exc
        return _Resp(json.dumps(body).encode("utf-8"))
    monkeypatch.setattr(bs.be.urllib.request, "urlopen", fake)
    return seen


def test_chat_goes_through_bench_endpoints_plumbing_to_the_arms_port(bs, monkeypatch):
    seen = _stub_urlopen(monkeypatch, bs, {
        "choices": [{"message": {"content": "one two three"}}],
        "usage": {"prompt_tokens": 257, "completion_tokens": 3}})
    wall, tokens, reported = bs.chat(ARMS["geniex"], "hi", 16, 33)
    assert isinstance(wall, float) and wall >= 0
    assert tokens == 3 and reported == 257
    req, timeout = seen[0]
    assert req.full_url == "http://127.0.0.1:18181/v1/chat/completions" and timeout == 33
    assert json.loads(req.data) == bs.request_body("b", "hi", 16)


def test_tokens_are_counted_locally_not_taken_from_usage(bs, monkeypatch):
    # One instrument for both arms: a usage block is each server's own
    # convention, so a server that mis-reports it changes nothing here.
    _stub_urlopen(monkeypatch, bs, {
        "choices": [{"message": {"content": "one two three"}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 9000}})
    assert bs.chat(ARMS["ours"], "hi", 16, 5)[1] == 3


def test_reasoning_in_a_side_channel_is_still_counted(bs, monkeypatch):
    _stub_urlopen(monkeypatch, bs, {"choices": [{"message": {
        "content": "four", "reasoning_content": " one two three"}}]})
    assert bs.chat(ARMS["ours"], "hi", 16, 5)[1] == 4


def test_prompt_tokens_is_reported_only_when_the_server_gave_a_positive_one(bs, monkeypatch):
    for usage, want in (({"prompt_tokens": 0}, None), ({}, None), (None, None),
                        ({"prompt_tokens": "257"}, None), ({"prompt_tokens": 257}, 257)):
        _stub_urlopen(monkeypatch, bs, {"choices": [{"message": {"content": "x"}}],
                                        "usage": usage})
        assert bs.chat(ARMS["ours"], "hi", 1, 5)[2] == want, usage


def test_a_body_that_is_not_a_completion_counts_nothing_and_does_not_raise(bs, monkeypatch):
    _stub_urlopen(monkeypatch, bs, ["not", "an", "object"])
    wall, tokens, reported = bs.chat(ARMS["ours"], "hi", 1, 5)
    assert tokens == 0 and reported is None


def test_a_failed_request_raises_with_the_servers_own_message(bs, monkeypatch):
    err = urllib.error.HTTPError(
        "u", 429, "Too Many Requests", {},
        io.BytesIO(b'{"error": {"message": "server busy; NPU is single-flight"}}'))
    _stub_urlopen(monkeypatch, bs, exc=err)
    with pytest.raises(RuntimeError) as e:
        bs.chat(ARMS["ours"], "hi", 1, 5)
    assert "429" in str(e.value) and "single-flight" in str(e.value)


# --- the acceptance rule: 90% of N, AND bench_endpoint's absolute floor -----

def _stub_chat(bs, monkeypatch, short, long, reported=None):
    """chat() returning (wall, tokens) `short` for the 1-cap call, `long` after."""
    def fake(arm, prompt, cap, timeout):
        wall, tokens = short if cap == 1 else long
        return wall, tokens, reported
    monkeypatch.setattr(bs, "chat", fake)


def test_a_full_window_is_a_rate_with_its_steps_seconds_and_prompt_tokens(bs, monkeypatch):
    _stub_chat(bs, monkeypatch, (0.5, 1), (6.5, 121), reported=263)
    rate, note, detail = bs.sample(ARMS["ours"], 250, 120, 5)
    assert rate == pytest.approx(20.0) and note == "120/6.00s"
    assert detail == {"steps": 120, "secs": 6.0, "prompt_tokens": 263}


def test_ninety_percent_of_the_window_is_accepted_and_one_step_less_is_not(bs, monkeypatch):
    # The docstring used to say "really produced N more tokens"; the code said
    # 0.9. The number is now a named constant and the docstring states it.
    assert bs.EARLY_STOP_TOLERANCE == 0.9
    _stub_chat(bs, monkeypatch, (0.5, 1), (5.9, 109))
    assert bs.sample(ARMS["ours"], 250, 120, 5)[0] == pytest.approx(108 / 5.4)
    _stub_chat(bs, monkeypatch, (0.5, 1), (5.9, 108))
    rate, note, detail = bs.sample(ARMS["ours"], 250, 120, 5)
    assert rate is None and note == "early stop (107 of 120 steps)"
    assert detail["steps"] == 107, "a refused sample still says what it saw"


def test_a_window_under_bench_endpoints_floor_is_refused_whatever_the_fraction(
        bs, monkeypatch):
    # --tokens 10 and all 10 produced: 100% of N, and still overhead divided
    # by a handful of tokens. The floor is bench_endpoint's, not a copy of it.
    assert bs.be.MIN_DECODE_STEPS == 16
    _stub_chat(bs, monkeypatch, (0.5, 1), (1.0, 11))
    rate, note, _ = bs.sample(ARMS["ours"], 250, 10, 5)
    assert rate is None and "16-step floor" in note and "GENIE_MIN_DECODE_STEPS" in note
    monkeypatch.setattr(bs.be, "MIN_DECODE_STEPS", 4)
    assert bs.sample(ARMS["ours"], 250, 10, 5)[0] == pytest.approx(20.0)


def test_a_non_positive_delta_is_refused_not_divided(bs, monkeypatch):
    _stub_chat(bs, monkeypatch, (6.5, 1), (6.5, 121))
    rate, note, _ = bs.sample(ARMS["ours"], 250, 120, 5)
    assert rate is None and "non-positive" in note


# --- one clock instrument, under the key the other tools use ----------------

def test_box_sample_is_bench_endpoints_sampler_under_bench_endpoints_keys(bs):
    bs.be.note_box_state("a row from the other tool")
    theirs = dict(bs.be.BOX_SAMPLES[-1])
    theirs.pop("label")
    assert bs.box_sample() == theirs
    assert bs.box_sample()["clock_pct"] == 88.0
    assert "clock" not in bs.box_sample(), "the old key, from the old instrument"


def test_an_unread_clock_is_none_and_prints_unreadable_never_minus_one(bs):
    bs.be.box_state = lambda: (None, None, None, None)
    box = bs.box_sample()
    assert box["clock_pct"] is None
    assert bs.clock_text(box) == "clock unreadable"
    assert bs.clock_text({"clock_pct": 87.6}) == "clock 88% of base"


# --- depths: bench_endpoint's resolver against the bundle's window ----------

def test_the_window_is_read_from_the_bundles_config(bs, world, tmp_path):
    assert bs.bundle_n_ctx(str(world.bundle)) == 4096
    assert bs.bundle_n_ctx(str(tmp_path / "no-bundle")) is None
    (world.bundle / "genie_config.json").write_text(
        json.dumps({"dialog": {"context": {"size": 0}}}), encoding="utf-8")
    assert bs.bundle_n_ctx(str(world.bundle)) is None


def test_a_non_integer_depth_is_a_sentence_not_a_traceback(bs, world, monkeypatch):
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world, "--depths", "250,abc")
    assert "comma-separated integers" in str(e.value)
    assert world.launched == []


def test_a_depth_past_the_budget_is_dropped_with_a_note(bs, world, monkeypatch, capsys):
    # 4096 window - 120 tokens - 256 margin = 3720: bench_endpoint's rule.
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250,5000") == 0
    out = capsys.readouterr().out
    assert "dropped" in out and "5000" in out and "3720" in out
    assert _result(world)["depths"] == [250]


def test_every_depth_past_the_budget_is_refused_before_a_launch(bs, world, monkeypatch):
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world, "--depths", "5000,6000")
    assert "exceeds the budget" in str(e.value) and world.launched == []


def test_n_ctx_overrides_the_bundles_window_and_is_recorded_as_the_source(
        bs, world, monkeypatch):
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "5000",
                 "--n-ctx", "8192") == 0
    got = _result(world)
    assert got["depths"] == [5000] and got["n_ctx"] == 8192
    assert got["n_ctx_source"] == "--n-ctx"


def test_a_bundle_with_no_config_assumes_4096_and_says_so(bs, world, monkeypatch, capsys):
    (world.bundle / "genie_config.json").unlink()
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 0
    assert "assumes a 4096 window" in capsys.readouterr().out
    got = _result(world)
    assert got["n_ctx"] is None and got["n_ctx_source"] == "assumed 4096"


def test_tokens_that_leave_no_room_in_the_window_are_refused(bs, world, monkeypatch):
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world, "--tokens", "4000")
    assert "--tokens 4000" in str(e.value) and world.launched == []


# --- the prompt builder is the shared one ------------------------------------

def test_prompt_at_is_prompt_depths_builder_over_this_tools_tokenizer(bs):
    import prompt_depth
    assert bs.prompt_at(250) == prompt_depth.prompt_at(bs.TOK, 250)
    assert bs.ntok(bs.prompt_at(250)) == 250
    assert bs.prompt_at(250).startswith("The measurement below concerns memory bandwidth")


# --- CLI spellings -----------------------------------------------------------

def test_repeat_is_passes_and_depth_is_depths(bs):
    # bench_endpoint and bench_contention say --repeat; bench_contention says
    # --depth. The docs quote --passes/--depths for this tool, so both work.
    new = bs._parser().parse_args(["--repeat", "5", "--depth", "64,128"])
    old = bs._parser().parse_args(["--passes", "5", "--depths", "64,128"])
    assert new.passes == old.passes == 5
    assert new.depths == old.depths == "64,128"


def test_both_spellings_are_declared_not_left_to_argparse_abbreviation(bs):
    # `--depth` would parse anyway, as an unambiguous prefix of `--depths`.
    # Declared, it shows in --help and survives the day a --depth-something
    # flag makes the prefix ambiguous.
    declared = bs._parser()._option_string_actions
    for flag in ("--passes", "--repeat", "--depths", "--depth"):
        assert flag in declared, flag


def test_the_defaults_are_the_ones_the_readme_table_was_measured_at(bs):
    a = bs._parser().parse_args([])
    assert (a.passes, a.tokens, a.depths) == (3, 120, "250,1500,3000")
    assert (a.ours_port, a.geniex_port, a.timeout) == (8123, 18181, 900)
    assert a.out == "sweep-results.json" and a.force is False


def test_the_help_renders(bs):
    # %-signs in help text are argparse format strings; a stray one is a
    # traceback on --help and nowhere else.
    assert "--repeat" in bs._parser().format_help()


# --- the artifact ------------------------------------------------------------

ROW_KEYS = {"arm", "pass", "run", "depth", "rate", "steps", "secs", "prompt_tokens",
            "on_ac", "charge_pct", "charge_w", "clock_pct"}


def test_a_full_run_writes_everything_needed_to_tell_two_sweeps_apart(
        bs, world, monkeypatch):
    assert _main(bs, monkeypatch, world, "--passes", "2", "--depths", "250,1500",
                 "--tokens", "20", "--ours-port", "8200") == 0
    got = _result(world)
    assert got["tool"] == "bench_servers" and got["outcome"] == "complete"
    assert got["arms"] == {"ours": {"port": 8200, "model": "qwen3-4b-npu"},
                           "geniex": {"port": 18181, "model": "qualcomm/qwen3-4b-ours"}}
    assert got["bundle_dir"] == str(world.bundle)
    assert got["passes"] == 2
    assert got["order"] == ["ours", "geniex", "geniex", "ours"]
    assert got["depths"] == [250, 1500] and got["tokens"] == 20 and got["timeout"] == 900
    assert got["n_ctx"] == 4096 and got["n_ctx_source"] == "genie_config.json"
    assert got["acceptance"] == {"min_fraction_of_cap": 0.9, "min_steps": 16}
    assert got["cap_spellings"] == ["max_tokens", "max_completion_tokens"]
    assert got["request_settings"] == {
        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none", "stream": False}
    assert got["started"] < got["finished"], "two timestamps, not one"
    assert "Processor Performance" in got["clock_instrument"]
    assert got["logs"] == {
        name: os.path.join(str(world.out.parent), "bench_servers-%s.log" % name)
        for name in ("ours", "geniex")}
    assert set(got) >= {"depth_means", "token_counter", "rows"}


def test_every_row_carries_its_pass_its_run_its_window_and_the_box_state(
        bs, world, monkeypatch):
    _main(bs, monkeypatch, world, "--passes", "2", "--depths", "250,1500", "--tokens", "20")
    rows = _result(world)["rows"]
    assert len(rows) == 8, "2 passes x 2 arms x 2 depths"
    assert all(set(r) == ROW_KEYS for r in rows)
    first = rows[0]
    assert first["rate"] == pytest.approx(20.0) and first["steps"] == 20
    assert first["prompt_tokens"] == 263 and first["clock_pct"] == 88.0
    assert first["on_ac"] is True and first["charge_pct"] == 73.0


def test_pass_counts_ab_pairs_and_run_counts_arm_runs(bs, world, monkeypatch, capsys):
    # `--passes 3` printed "[pass 6/6]" and stored pass=1..6, while the flag
    # and the README's "three passes each" count pairs.
    _main(bs, monkeypatch, world, "--passes", "2", "--depths", "250", "--tokens", "20")
    rows = _result(world)["rows"]
    assert [(r["arm"], r["pass"], r["run"]) for r in rows] == [
        ("ours", 1, 1), ("geniex", 1, 2), ("geniex", 2, 3), ("ours", 2, 4)]
    out = capsys.readouterr().out
    assert "[run 4/4, pass 2/2] ours" in out and "pass 4" not in out


def test_the_servers_own_prompt_count_is_printed_beside_the_target_depth(
        bs, world, monkeypatch, capsys):
    # "d250" is 250 untemplated tokens; the server counted the templated
    # prompt. Shown together so neither is read as the other.
    _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250", "--tokens", "20")
    assert "server says prompt=263" in capsys.readouterr().out


def test_every_arm_gets_the_same_request_apart_from_its_model(bs, world, monkeypatch):
    _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250", "--tokens", "20")
    by_base = {}
    for base, payload in world.posts:
        body = dict(payload)
        body.pop("model")
        by_base.setdefault(base, []).append(body)
    ours, geniex = by_base["http://127.0.0.1:8123"], by_base["http://127.0.0.1:18181"]
    assert ours == geniex and len(ours) == 3, "warmup, then the 1-cap and the 1+N call"
    assert [b["max_completion_tokens"] for b in ours] == [16, 1, 21]


def test_the_run_ends_with_nothing_of_ours_left_running(bs, world, monkeypatch):
    _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250", "--tokens", "20")
    assert len(world.launched) == 2
    assert all(p.returncode is not None for p in world.launched)
    assert world.listening == set()
    assert sorted(c[2] for c in world.kills()) == sorted(str(p.pid) for p in world.launched)


# --- the closing A/B comparison ---------------------------------------------
# The medians table and the OVERLAP/disjoint line under it ARE the cross-server
# result the README quotes, and unlike bench_endpoint's _verdict -- lifted out
# of main() precisely so it could be tested -- they are still inline. Nothing
# reached them: the World answers every request in 0.30 + 0.05s a token, so
# both arms measure exactly 20.00 t/s, every range is a point, and "are
# disjoint" was never produced anywhere in this suite.

def _rates(world, ours, geniex):
    """Give each arm its own decode rate, one per sample, cycling.

    A sample is a 1-cap request followed by a (1+N)-cap one and the rate is
    the delta between them, so the cycle advances on the 1-cap request and
    both halves of a sample share a rate. The warmup asks for 16 and is
    discarded.
    """
    wanted = {"ours": ours, "geniex": geniex}
    nth = {"ours": -1, "geniex": -1}

    def hook(base, payload):
        arm = "ours" if base.endswith(":8123") else "geniex"
        cap = payload["max_completion_tokens"]
        if cap == 1:
            nth[arm] += 1
        rate = wanted[arm][max(0, nth[arm]) % len(wanted[arm])]
        body = {"choices": [{"message": {"content": "w " * cap}}],
                "usage": {"prompt_tokens": 263, "completion_tokens": cap}}
        return body, 0.30 + cap / rate

    world.post_hook = hook


def _closing(capsys):
    out = capsys.readouterr().out
    return out[out.index("=== medians"):]


def test_the_closing_table_gives_each_arm_its_median_and_its_full_range(
        bs, world, monkeypatch, capsys):
    # The median is the statistic, and the bracket beside it is the whole
    # spread with the n it rests on -- so a reader can see that two medians
    # 1 t/s apart came out of ranges that cover each other.
    _rates(world, ours=[18.0, 24.0], geniex=[19.0, 21.0])
    assert _main(bs, monkeypatch, world, "--passes", "2", "--depths", "250",
                 "--tokens", "20") == 0
    closing = _closing(capsys)
    assert "ours     21.00 [18.00-24.00 n=2]" in closing, "median, not min or max"
    assert "geniex   20.00 [19.00-21.00 n=2]" in closing


def test_two_arms_whose_ranges_meet_are_called_indistinguishable(
        bs, world, monkeypatch, capsys):
    # 21.00 against 20.00 is a 5% "win" that the two ranges say nothing about.
    # Quoting the ratio of the medians is the read this line exists to refuse.
    _rates(world, ours=[18.0, 24.0], geniex=[19.0, 21.0])
    _main(bs, monkeypatch, world, "--passes", "2", "--depths", "250", "--tokens", "20")
    assert "d250   ranges OVERLAP -- indistinguishable" in _closing(capsys)


def test_two_arms_whose_ranges_do_not_meet_are_called_disjoint(
        bs, world, monkeypatch, capsys):
    # Every sample of one arm faster than every sample of the other: the one
    # case where the tool may say the two differ.
    _rates(world, ours=[50.0, 60.0], geniex=[19.0, 21.0])
    _main(bs, monkeypatch, world, "--passes", "2", "--depths", "250", "--tokens", "20")
    closing = _closing(capsys)
    assert "d250   ranges are disjoint" in closing
    assert "OVERLAP" not in closing
    assert "ours     55.00 [50.00-60.00 n=2]" in closing


def test_the_comparison_is_per_depth_and_each_depth_is_called_on_its_own(
        bs, world, monkeypatch, capsys):
    # The finding the README quotes is depth-by-depth: at 250 and 1500 the two
    # overlap and at 3000 they part. One verdict over the pooled sweep would
    # say neither.
    def hook(base, payload):
        cap = payload["max_completion_tokens"]
        deep = payload["messages"][0]["content"].count(" ") > 2000
        rate = 50.0 if (base.endswith(":8123") and deep) else 20.0
        body = {"choices": [{"message": {"content": "w " * cap}}],
                "usage": {"prompt_tokens": 263, "completion_tokens": cap}}
        return body, 0.30 + cap / rate
    world.post_hook = hook
    _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250,3000",
          "--tokens", "20")
    closing = _closing(capsys)
    assert "d250   ranges OVERLAP -- indistinguishable" in closing
    assert "d3000  ranges are disjoint" in closing


def test_a_depth_only_one_arm_measured_gets_no_verdict_at_all(
        bs, world, monkeypatch, capsys):
    # An arm with no rows at a depth is not the other side of a comparison,
    # and a one-sided "OVERLAP" would read as a measured agreement.
    world.dies_at_start[18181] = (1, "no usable Hexagon\n")
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250",
                 "--tokens", "20") == 1
    closing = _closing(capsys)
    assert "geniex  no data" in closing
    assert "ranges" not in closing


# --- the results file at the END of the sweep --------------------------------
# The startup refusal only sees what is there when the run begins. A file that
# appeared during the twenty minutes -- another run writing the same default
# sweep-results.json -- was overwritten without a word, and an OSError at the
# write was a traceback with no record anywhere. bench_contention._write_record
# already had the shape these pin: never over a file it was not given, a
# timestamped file beside it instead, where it went on the screen, exit 1.

FALLBACK_GLOB = "sweep.2026*.json"


def _someone_writes_out(world, text, on_post=1):
    """A post_hook: on the `on_post`-th request, another writer puts `text` at --out."""
    count = []

    def hook(base, payload):
        count.append(1)
        if len(count) == on_post:
            world.out.write_text(text, encoding="utf-8")
    world.post_hook = hook


@pytest.mark.parametrize("force", [False, True])
def test_a_file_that_appears_during_the_sweep_is_not_overwritten(
        bs, world, monkeypatch, capsys, force):
    # --force is consent to replace what was there at the START; nothing was.
    _someone_writes_out(world, "other-session-results")
    argv = ["--passes", "1", "--depths", "250"] + (["--force"] if force else [])
    assert _main(bs, monkeypatch, world, *argv) == 1, \
        "a complete sweep whose record is not at --out is not a clean exit"
    assert world.out.read_text(encoding="utf-8") == "other-session-results"
    fallbacks = list(world.out.parent.glob(FALLBACK_GLOB))
    assert len(fallbacks) == 1, "the record went beside it, once"
    got = json.loads(fallbacks[0].read_text(encoding="utf-8"))
    assert got["tool"] == "bench_servers" and got["outcome"] == "complete"
    assert len(got["rows"]) == 2
    out = capsys.readouterr().out
    assert "appeared during the sweep" in out and "NOT overwritten" in out
    assert "wrote %s" % fallbacks[0] in out, "where it went, in the line that says so"


def test_force_leaves_alone_a_file_replaced_during_the_sweep(bs, world, monkeypatch, capsys):
    world.out.write_text("stale", encoding="utf-8")
    _someone_writes_out(world, "a newer run's results, not the stale file")
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250", "--force") == 1
    assert world.out.read_text(encoding="utf-8") == "a newer run's results, not the stale file"
    assert len(list(world.out.parent.glob(FALLBACK_GLOB))) == 1
    assert "was replaced during the sweep" in capsys.readouterr().out


def test_force_still_replaces_the_file_it_was_given_and_nothing_else(bs, world, monkeypatch):
    world.out.write_text("stale", encoding="utf-8")
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250", "--force") == 0
    assert _result(world)["tool"] == "bench_servers"
    assert list(world.out.parent.glob(FALLBACK_GLOB)) == []


def test_an_out_that_became_a_directory_falls_back_without_a_traceback(
        bs, world, monkeypatch, capsys):
    # The refuter's run 3: PermissionError out of the write, no record at all.
    count = []

    def hook(base, payload):
        count.append(1)
        if len(count) == 1:
            world.out.mkdir()
    world.post_hook = hook
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    fallbacks = list(world.out.parent.glob(FALLBACK_GLOB))
    assert len(fallbacks) == 1
    assert json.loads(fallbacks[0].read_text(encoding="utf-8"))["tool"] == "bench_servers"
    assert "Traceback" not in capsys.readouterr().err


def _failing_dump(bs, monkeypatch, fail):
    """_dump_json that raises for any path `fail` says so for."""
    real = bs._dump_json

    def dump(path, record, mode):
        if fail(path):
            raise PermissionError(13, "Permission denied", path)
        return real(path, record, mode)
    monkeypatch.setattr(bs, "_dump_json", dump)


def test_an_oserror_writing_out_falls_back_and_says_why(bs, world, monkeypatch, capsys):
    _failing_dump(bs, monkeypatch, lambda path: path == str(world.out))
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    assert not world.out.exists()
    fallbacks = list(world.out.parent.glob(FALLBACK_GLOB))
    assert len(fallbacks) == 1
    out = capsys.readouterr().out
    assert "could not be written" in out and "Permission denied" in out
    assert str(fallbacks[0]) in out


def test_when_the_fallback_fails_too_the_run_says_the_numbers_are_only_on_screen(
        bs, world, monkeypatch, capsys):
    _failing_dump(bs, monkeypatch, lambda path: True)
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    out = capsys.readouterr().out
    assert "failed too" in out and "only in the terminal" in out
    assert "\nwrote " not in out, "no claim of a file that was not written"


def test_a_file_appearing_between_the_check_and_the_open_is_not_overwritten(
        bs, world, monkeypatch):
    # The check says "nothing there" and the file lands before the open: the
    # exclusive create is what catches it, not the check.
    world.out.write_text("landed in the gap", encoding="utf-8")
    monkeypatch.setattr(bs, "out_state", lambda path: None)
    written, note = bs.write_results(str(world.out), {"tool": "bench_servers"}, None)
    assert world.out.read_text(encoding="utf-8") == "landed in the gap"
    assert written != str(world.out) and "NOT overwritten" in note
    with open(written, encoding="utf-8") as f:
        assert json.load(f) == {"tool": "bench_servers"}


def test_the_fallback_never_replaces_a_file_either(bs, world, monkeypatch):
    # Two runs falling back in the same second: the second one's fallback is
    # created exclusively too, so it fails loudly rather than replacing.
    world.out.write_text("someone's", encoding="utf-8")
    monkeypatch.setattr(bs, "_timestamped_beside", lambda path: path + ".fallback")
    (world.out.parent / "sweep.json.fallback").write_text("first fallback", encoding="utf-8")
    written, note = bs.write_results(str(world.out), {"x": 1}, None)
    assert written is None and "failed too" in note
    assert (world.out.parent / "sweep.json.fallback").read_text(encoding="utf-8") == "first fallback"


def test_an_out_that_is_a_directory_is_refused_at_startup_even_with_force(
        bs, world, monkeypatch):
    world.out.mkdir()
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world, "--force")
    assert "is a directory" in str(e.value) and str(world.out) in str(e.value)
    assert world.launched == [] and world.tokenizer_loads == []


# --- other NPU servers on the box, on ANY port -------------------------------
# foreign_listener looks at this run's two ports only. A genie_server on 8124,
# a GenieAPIService on 8910 or a geniex on any port but 18181 went unseen, and
# the sweep loaded a second ~3 GB bundle onto the Hexagon beside it. Every
# process table here is made up; none is a scan of a real box.

BS = chr(92)


def _win(*parts):
    return BS.join(parts)


PY = _win("C:", "Python312", "python.exe")
GENIE_SERVER_CMD = '"%s" %s' % (PY, _win("C:", "work", "repo", "src", "genie_server.py"))
RUN_PATH_CMD = "python -c \"import runpy; runpy.run_path('src/genie_server.py')\""


def _proc(pid, name, cmdline="", ppid=1):
    return {"pid": pid, "ppid": ppid, "name": name, "cmdline": cmdline}


@pytest.mark.parametrize("name, cmdline, kind", [
    ("python.exe", GENIE_SERVER_CMD, "genie_server"),
    ("python.exe", "python src/genie_server.py", "genie_server"),
    ("python3.12.exe", "python3.12 -u -X utf8 src/genie_server.py", "genie_server"),
    ("pythonw.exe", '"%s" "%s"' % (PY, _win("C:", "My Repo", "src", "genie_server.py")),
     "genie_server"),
    ("geniex.exe", '"%s" serve --host 127.0.0.1:18199' % _win("C:", "GenieX CLI", "geniex.exe"),
     "geniex"),
    ("GENIEX.EXE", "", "geniex"),
    ("GenieAPIService.exe", "GenieAPIService.exe -c config.json", "GenieAPIService"),
    # A server run through a module that runs the script it is handed.
    ("python.exe", "python -m pdb src/genie_server.py", "genie_server"),
    ("python.exe", "python -m cProfile -o out.prof src/genie_server.py", "genie_server"),
    # Not servers: the command line only MENTIONS genie_server.
    ("python.exe", "python -m pytest tests/test_startup.py -k genie_server", None),
    ("python.exe", "python -m genie_server", None),
    ("python.exe", RUN_PATH_CMD, None),
    ("python.exe", 'python -c "import sys; print(sys.argv)" src/genie_server.py', None),
    ("python.exe", "python", None),                     # a REPL
    ("bash.exe", 'bash -c "python src/genie_server.py"', None),
    ("python.exe", "python src/bench_servers.py --ours-port 8200", None),
    ("py.exe", "py -3 src/genie_server.py", None),       # its python child is the one
    # An unterminated quote runs to the end of the line, as Windows reads it.
    ("python.exe", 'python "C:/my repo/src/genie_server.py', "genie_server"),
    # An apostrophe is part of a path, not a quote (shlex took it for one):
    # read as a quote, the two below join into one argument ending log.txt.
    ("python.exe", "python %s --log %s" % (
        _win("C:", "Users", "O'Brien", "src", "genie_server.py"),
        _win("C:", "Users", "O'Brien", "log.txt")), "genie_server"),
    ("explorer.exe", "", None),
])
def test_npu_server_kind_goes_by_image_and_script_not_by_words(bs, name, cmdline, kind):
    assert bs.npu_server_kind(name, cmdline) == kind


def test_a_geniex_under_another_file_name_is_still_geniex(bs):
    exe = _win("C:", "tools", "geniex-0.5.exe")
    assert bs.npu_server_kind("geniex-0.5.exe", exe + " serve", geniex=exe) == "geniex"
    assert bs.npu_server_kind("geniex-0.5.exe", exe + " serve") is None


def test_other_npu_servers_leaves_out_this_run_and_everything_below_it(bs):
    procs = [
        _proc(100, "python.exe", "python src/bench_servers.py"),        # this run
        _proc(4000, "python.exe", "python src/genie_server.py", ppid=100),
        _proc(4001, "geniex.exe", "geniex serve", ppid=4000),           # its child
        _proc(4500, "python.exe", "python src/genie_server.py", ppid=4400),  # orphan of 4400
        _proc(6100, "python.exe", GENIE_SERVER_CMD, ppid=1),
        _proc(5150, "GenieAPIService.exe", "GenieAPIService.exe", ppid=6100),
        _proc(7000, "notepad.exe", "notepad", ppid=1),
    ]
    found = bs.other_npu_servers(procs, {100, 4400})
    assert [(p["pid"], p["kind"]) for p in found] == [(5150, "GenieAPIService"),
                                                     (6100, "genie_server")]


def test_a_parent_loop_in_the_table_does_not_hang_the_walk(bs):
    procs = [_proc(1, "geniex.exe", ppid=2), _proc(2, "explorer.exe", ppid=1)]
    assert [p["pid"] for p in bs.other_npu_servers(procs, {99})] == [1]


def _stub_listing(bs, monkeypatch, stdout="", returncode=0, platform="win32", raises=None):
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        if raises is not None:
            raise raises
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(bs, "subprocess", types.SimpleNamespace(
        run=run, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(bs, "sys", types.SimpleNamespace(platform=platform))
    return ran


def test_list_processes_reads_the_cim_table_and_stops_nothing(bs, monkeypatch):
    rows = [{"ProcessId": 6100, "ParentProcessId": 1, "Name": "python.exe",
             "CommandLine": GENIE_SERVER_CMD},
            {"ProcessId": 4, "ParentProcessId": None, "Name": "System", "CommandLine": None}]
    ran = _stub_listing(bs, monkeypatch, stdout=json.dumps(rows))
    assert bs.list_processes() == [
        {"pid": 6100, "ppid": 1, "name": "python.exe", "cmdline": GENIE_SERVER_CMD},
        {"pid": 4, "ppid": 0, "name": "System", "cmdline": ""}]
    script = " ".join(ran[0])
    assert "Get-CimInstance" in script and "Win32_Process" in script
    assert "Stop-Process" not in script and "taskkill" not in script


def test_list_processes_takes_a_one_process_table_as_a_list(bs, monkeypatch):
    one = {"ProcessId": 7, "ParentProcessId": 1, "Name": "geniex.exe", "CommandLine": "g"}
    _stub_listing(bs, monkeypatch, stdout=json.dumps(one))
    assert [p["pid"] for p in bs.list_processes()] == [7]


@pytest.mark.parametrize("how", [
    {"stdout": "not json"},
    {"stdout": "", "returncode": 1},
    {"raises": subprocess.TimeoutExpired("powershell.exe", 60)},
    {"raises": FileNotFoundError(2, "no powershell")},
])
def test_list_processes_is_none_when_it_could_not_look(bs, monkeypatch, how):
    _stub_listing(bs, monkeypatch, **how)
    assert bs.list_processes() is None


def test_list_processes_is_none_off_windows_without_running_anything(bs, monkeypatch):
    ran = _stub_listing(bs, monkeypatch, platform="linux")
    assert bs.list_processes() is None and ran == []


def test_main_refuses_another_npu_server_on_another_port_before_any_launch(
        bs, world, monkeypatch):
    world.processes = [_proc(6100, "python.exe", GENIE_SERVER_CMD),
                       _proc(6200, "geniex.exe", "geniex.exe serve --host 127.0.0.1:18199")]
    with pytest.raises(SystemExit) as e:
        _main(bs, monkeypatch, world)
    said = str(e.value)
    assert said.startswith("NOT starting")
    assert "pid 6100" in said and "genie_server" in said
    assert "pid 6200" in said and "geniex" in said
    assert "--allow-other-npu-servers" in said, "the deliberate way on"
    assert world.launched == [], "no bundle loaded beside them"
    assert world.kills() == [], "and nothing of theirs touched"
    assert not world.out.exists()


def test_another_npu_server_that_appears_mid_sweep_ends_it_as_refused(
        bs, world, monkeypatch, capsys):
    def someone_starts_one(base, payload):
        if not world.processes:
            world.processes.append(_proc(5150, "GenieAPIService.exe", "GenieAPIService.exe"))
    world.post_hook = someone_starts_one
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 1
    assert len(world.launched) == 1, "the second arm was never loaded beside it"
    got = _result(world)
    assert got["outcome"].startswith("refused:")
    assert "5150" in got["outcome"] and "GenieAPIService" in got["outcome"]
    assert [r["arm"] for r in got["rows"]] == ["ours"], "what was measured is kept"
    assert world.kills() == [["taskkill", "/PID", str(world.launched[0].pid), "/T", "/F"]]


def test_the_flag_runs_beside_them_and_the_file_lists_who_was_there(
        bs, world, monkeypatch, capsys):
    world.processes = [_proc(6200, "geniex.exe", "geniex.exe serve --host 127.0.0.1:18199")]
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250",
                 "--allow-other-npu-servers") == 0
    assert len(world.launched) == 2
    assert "WARNING: running beside 1 other NPU server" in capsys.readouterr().out
    other = _result(world)["other_npu_servers"]
    assert other["allowed"] is True and other["unscanned_runs"] == []
    assert [(p["pid"], p["kind"], p["run"]) for p in other["seen"]] == [
        (6200, "geniex", 0), (6200, "geniex", 1), (6200, "geniex", 2)]


def test_this_runs_own_leftovers_are_not_taken_for_someone_elses(bs, world, monkeypatch):
    # A grandchild of the `ours` arm (a venv launcher's python child, say)
    # that outlived the tree kill still descends from a pid this run started.
    def leftover(base, payload):
        if not world.processes:
            world.processes.append(_proc(4999, "python.exe", "python src/genie_server.py",
                                         ppid=world.launched[0].pid))
    world.post_hook = leftover
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 0
    assert _result(world)["outcome"] == "complete"


def test_a_process_listing_that_fails_is_said_and_recorded_not_taken_for_none(
        bs, world, monkeypatch, capsys):
    world.processes = None
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 0
    assert "were NOT looked for" in capsys.readouterr().out
    assert _result(world)["other_npu_servers"]["unscanned_runs"] == [0, 1, 2]


def test_a_clean_box_is_looked_at_before_every_start_and_recorded_as_such(
        bs, world, monkeypatch):
    assert _main(bs, monkeypatch, world, "--passes", "1", "--depths", "250") == 0
    assert world.process_listings == 3, "at startup and before each of the two starts"
    assert _result(world)["other_npu_servers"] == {
        "allowed": False, "seen": [], "unscanned_runs": []}
