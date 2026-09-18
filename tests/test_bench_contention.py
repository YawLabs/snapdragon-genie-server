"""Device-free tests for bench_contention.py.

This file started as cover for two crash bugs, both reachable from a single
round where every contended sample was skipped -- and that is not a rare shape:
the NPU sheds by design once its small queue fills (429 OpenAI / 529
Anthropic), so a contention run against it is the workload MOST likely to
produce an asymmetric solo/contended result. Both guards had assumed it could
not happen. It has since grown to pin the load generator, the gate, the
headline arithmetic and the JSON artifact.

No device, no server, no network and NO SUBPROCESS. bench_endpoint is replaced
by a stub for the duration of bench_contention's import, every hardware probe
is stubbed by an autouse fixture, and that same fixture fails any test that
reaches `subprocess` anyway -- so the promise is checked, not merely made.
"""

import builtins
import ctypes
import importlib.util
import io
import json
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
_MISSING = object()


def _endpoint_stub():
    """What bench_contention uses from bench_endpoint, and nothing else.

    Exactly the surface, so a name bench_contention stops using (it spelled
    `post_timed` as `_post` for a while) or starts using shows up as a failure
    in test_the_stub_is_the_surface_bench_contention_uses rather than as a stub
    that quietly answers for a function nobody calls. Every probe returns
    "unreadable", which is what a test double that cannot see hardware
    honestly knows; tests that need a reading patch it in. The failure shape
    of post_timed is (None, reason-STRING), as the real one's is.
    """
    return types.SimpleNamespace(
        prompt_of=lambda d: "x",
        measure_decode=lambda *a, **k: 1.0,
        chat=lambda *a, **k: {"prompt_tokens": 1, "completion_tokens": 1,
                              "wall": 0.1, "model": "stub-model",
                              "content": "ok"},
        n_ctx=lambda b: 4096,
        post_timed=lambda *a, **k: (None, "stubbed"),
        box_state=lambda: (None, None, None, None),
        power_reading=lambda: (None, None, None),
        BOX_SAMPLES=[],
        MIN_DECODE_STEPS=16,
    )


def _load():
    """bench_contention, executed from its file with the stub as bench_endpoint.

    The stub sits in sys.modules only while the module body runs, and whatever
    was there before is put back. This used to be a module-scope
    `sys.modules.setdefault(...)` that was never removed: a process-wide
    SimpleNamespace that every later `import bench_endpoint` in the session
    received, that silently YIELDED if the real module had been imported first
    (so which file pytest collected first decided what this one tested), and
    that two sibling test files had to load around.
    """
    saved = sys.modules.get("bench_endpoint", _MISSING)
    sys.modules["bench_endpoint"] = _endpoint_stub()
    try:
        spec = importlib.util.spec_from_file_location(
            "bench_contention", SRC / "bench_contention.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is _MISSING:
            del sys.modules["bench_endpoint"]
        else:
            sys.modules["bench_endpoint"] = saved
    return mod


bc = _load()

# The real probes, kept so they can be tested against a faked subprocess. The
# autouse fixture below replaces the module's own names with stubs.
_REAL_CPU_PERFORMANCE_PCT = bc.cpu_performance_pct
_REAL_CPU_BUSY_PCT = bc.cpu_busy_pct


@pytest.fixture(autouse=True)
def hardware(monkeypatch):
    """Stub every hardware probe, and fail the test if one launches anyway.

    Six tests here used to call the real paired_sweep with
    cpu_performance_pct unpatched, so on Windows each of them launched
    PowerShell Get-Counter (2-7 s apiece, 35 s for the six) under a header
    promising "no device". Nothing asserted on the reading, so nothing failed
    -- which is the state in which a platform-dependent flake waits for the
    first test that does. Stubbing per test is a rule somebody has to remember;
    this is the rule applied for them, plus a tripwire for the probe that does
    not exist yet.

    Yields the list of attempted launches, for the one test that wants to see
    the tripwire work.
    """
    launches = []

    def refuse(*a, **k):
        launches.append(a[0] if a else k.get("args"))
        raise AssertionError("the suite tried to launch a subprocess: %r"
                             % (launches[-1],))

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 99.0)
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 50.0)
    # Read from the environment at import, so pinned: a developer shell that
    # exports GENIE_LOW_CHARGE_PCT would otherwise move the low-pack tests.
    monkeypatch.setattr(bc, "LOW_CHARGE_PCT", 25.0)
    # Module state that main() drains into the artifact. Cleared on both sides
    # so no test inherits another's notes or leaves its own behind.
    for store in (bc.GATE_NOTES, bc.POWER_SAMPLES, bc.be.BOX_SAMPLES):
        del store[:]
    yield launches
    for store in (bc.GATE_NOTES, bc.POWER_SAMPLES, bc.be.BOX_SAMPLES):
        del store[:]
    assert launches == [], "a hardware probe was left unstubbed: %r" % launches


def _fake_clock(monkeypatch):
    """Give bench_contention a clock that moves only when it is slept on.

    Scoped to the module's own `time` name rather than patched onto the real
    time module, so nothing outside the code under test sees it. Returns the
    list of sleeps taken.
    """
    now = {"t": 1000.0}
    slept = []

    def sleep(s):
        slept.append(s)
        now["t"] += s

    monkeypatch.setattr(bc, "time", types.SimpleNamespace(
        time=lambda: now["t"], perf_counter=lambda: now["t"], sleep=sleep))
    return slept


def _flat(text):
    """Collapse runs of whitespace, so an assertion can quote a printed line
    without reproducing its column padding."""
    return " ".join(text.split())


# --- the import itself -------------------------------------------------------

def test_the_stub_does_not_outlive_the_import():
    # The old setdefault left the SimpleNamespace in sys.modules for the whole
    # session. Whatever is there now, it is not this file's stub.
    assert sys.modules.get("bench_endpoint") is not bc.be
    assert isinstance(bc.be, types.SimpleNamespace)


def test_the_stub_is_the_surface_bench_contention_uses():
    """The stub and the code agree on which names cross the module boundary,
    and the real bench_endpoint defines every one of them.

    A stub is a claim about another module. This one had drifted twice: it
    carried `_post` returning a float where the real failure path returns a
    reason string, and it lacked box_state / BOX_SAMPLES and the power reading
    once bench_contention began delegating to them.
    """
    source = (SRC / "bench_contention.py").read_text(encoding="utf-8")
    used = set(re.findall(r"\bbe\.([A-Za-z_]\w*)", source))
    assert used == set(vars(_endpoint_stub()))
    real = (SRC / "bench_endpoint.py").read_text(encoding="utf-8")
    for name in sorted(used):
        assert re.search(r"^(def %s\(|%s = )" % (name, name), real, re.M), (
            "bench_endpoint no longer defines %s" % name)


def test_a_typo_in_the_env_var_does_not_kill_the_import(monkeypatch, capsys):
    # float(os.environ[...]) at module scope: a typo was an import-time
    # ValueError naming neither the variable nor the form it wanted, and since
    # this file imports the module at collection it took the test run with it.
    monkeypatch.setenv("GENIE_LOW_CHARGE_PCT", "twenty")
    fresh = _load()
    assert fresh.LOW_CHARGE_PCT == 25.0
    out = capsys.readouterr().out
    assert "GENIE_LOW_CHARGE_PCT" in out and "'twenty'" in out


def test_the_env_var_is_honoured_when_it_is_a_number(monkeypatch, capsys):
    monkeypatch.setenv("GENIE_LOW_CHARGE_PCT", " 32.5 ")
    assert _load().LOW_CHARGE_PCT == 32.5
    monkeypatch.setenv("GENIE_LOW_CHARGE_PCT", "")
    assert _load().LOW_CHARGE_PCT == 25.0
    assert "WARNING" not in capsys.readouterr().out


# --- the probes --------------------------------------------------------------

def test_an_unstubbed_probe_is_caught_rather_than_launched(hardware, monkeypatch):
    """The tripwire itself. The real clock probe, on a platform it runs on,
    reaches subprocess.run -- and the fixture records the attempt instead of
    letting PowerShell start. The probe swallows the refusal (it returns None
    on ANY exception), which is exactly why the fixture checks its own list at
    teardown rather than relying on the exception surfacing."""
    monkeypatch.setattr(bc, "sys", types.SimpleNamespace(platform="win32"))
    assert _REAL_CPU_PERFORMANCE_PCT() is None
    assert _REAL_CPU_BUSY_PCT() is None
    assert [cmd[0] for cmd in hardware] == ["powershell.exe", "powershell.exe"]
    assert "Processor Performance" in hardware[0][-1]
    assert "Processor Time" in hardware[1][-1]
    del hardware[:]            # seen and accounted for


def test_the_probes_parse_the_first_line_and_launch_nothing_off_windows(
        monkeypatch):
    monkeypatch.setattr(bc, "sys", types.SimpleNamespace(platform="linux"))
    assert _REAL_CPU_PERFORMANCE_PCT() is None      # and the tripwire is quiet
    assert _REAL_CPU_BUSY_PCT() is None
    monkeypatch.setattr(bc, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="93.5\r\n12.0\r\n", returncode=0))
    assert _REAL_CPU_PERFORMANCE_PCT() == 93.5
    assert _REAL_CPU_BUSY_PCT() == 93.5


# The PowerShell half of each probe cannot run device-free, so it is pinned by
# TEXT, the way bench_endpoint pins _STATE_PS. What is pinned is the one thing
# that made both readers return None on a comma-decimal Windows: every
# CookedValue is turned into a string in the INVARIANT culture before
# PowerShell can render the double in the session's own.

def test_both_counter_probes_format_their_double_in_the_invariant_culture():
    """A bare `.CookedValue` prints "72,4370708845929" on a comma-decimal
    Windows (measured with the real powershell.exe under CurrentCulture
    de-DE, against "70.956354241159" from the same box under en-US), so
    float() raised and the reading was lost. Losing the CLOCK loses the GATE:
    wait_for_cool returns on its first None poll, before power_limited_note,
    so every sample went UNGATED and the on-battery abort could not fire.
    _STATE_PS was fixed for this and these two reads were left behind."""
    invariant = ".ToString([cultureinfo]::InvariantCulture)"
    for name, script in (("clock", bc._CLOCK_PS), ("busy", bc._BUSY_PS)):
        _head, *reads = script.split(".CookedValue")
        assert len(reads) == 1, "%s: one counter read per probe" % name
        assert reads[0].startswith(invariant), (
            "%s: the CookedValue is left to the current culture" % name)


def test_each_probe_launches_the_script_that_is_pinned(monkeypatch, hardware):
    """The pin above is only worth having if these are the strings that run:
    an inline copy in the function body would drift from the constant the test
    reads. Both probes go through the tripwire, which records the command."""
    monkeypatch.setattr(bc, "sys", types.SimpleNamespace(platform="win32"))
    assert _REAL_CPU_PERFORMANCE_PCT() is None
    assert _REAL_CPU_BUSY_PCT() is None
    assert [cmd[-1] for cmd in hardware] == [bc._CLOCK_PS, bc._BUSY_PS]
    assert "Processor Information" in bc._CLOCK_PS      # clock, not utilisation
    assert "% Processor Time" in bc._BUSY_PS
    del hardware[:]            # seen and accounted for


def test_a_comma_decimal_reading_is_a_lost_reading_not_a_repaired_one(
        monkeypatch):
    """Why the fix is in the PowerShell and not in the float(). A decimal
    comma cannot be swapped back safely in Python -- "1,234" is a thousands
    separator in one culture and a fraction in another -- so a reading that
    arrives that way is None, and the script has to be the thing that never
    produces it."""
    monkeypatch.setattr(bc, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="72,4370708845929\r\n", returncode=0))
    assert _REAL_CPU_PERFORMANCE_PCT() is None
    assert _REAL_CPU_BUSY_PCT() is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="", returncode=1))
    assert _REAL_CPU_PERFORMANCE_PCT() is None, "an empty read is None, not 0"


@pytest.mark.parametrize("raw,expected", [
    (None, "unknown"),          # the QUERY failed
    ("", "no-battery"),         # the class answered with nothing: a desktop
    ("True", "ac"),
    ("true", "ac"),
    ("False", "battery"),
])
def test_power_reading_maps_the_raw_source(monkeypatch, raw, expected):
    """None and "" are the pair that matters. A transient PowerShell failure
    used to return the same value as a desktop with no battery, silently
    downgrading a definitive check to a heuristic on a laptop that could have
    answered -- and every test stubbed the mapped label itself, so collapsing
    the two left the suite green. The pack figures ride along untouched: they
    are bench_endpoint's reading, from the same launch."""
    monkeypatch.setattr(bc.be, "power_reading", lambda: (raw, 61.0, 14.5))
    assert bc.power_reading() == (expected, 61.0, 14.5)


# --- the memory reader -------------------------------------------------------
# free_physical_gb is the SOLE input to the quiet-box precondition, and every
# test that reaches main() replaces it with a lambda, so its body ran nowhere
# in the suite: a wrong field or a wrong divisor reads too HIGH and silently
# passes a loaded box, which is the exact state every retracted number on this
# hardware was measured in and the failure --min-free-gb exists to prevent.
#
# Covered the way the counter probes are. The OS call is the half that cannot
# run device-free, so it is faked, and what is pinned is the STRUCT this file
# hands it and the arithmetic done on what comes back. MEMORYSTATUSEX is
# declared below from Win32's own documentation rather than read off the
# module, and the module's buffer is written THROUGH that declaration: a field
# given the wrong width in bench_contention shifts every offset after it, so
# the module then reads back a different field than the one this test wrote.

class _MEMORYSTATUSEX(ctypes.Structure):
    """MEMORYSTATUSEX, as Win32 documents it. 64 bytes; ullAvailPhys at 16."""

    _fields_ = [("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def _memory_status(monkeypatch, avail_bytes, total_bytes=31 * 1024 ** 3, ok=1):
    """free_physical_gb's win32 branch over a faked GlobalMemoryStatusEx.

    Returns (result, dwLength-as-the-module-set-it). Win32 refuses the call
    outright when dwLength is not sizeof(MEMORYSTATUSEX), so that field is
    part of the contract rather than a detail.
    """
    seen = {}

    def fill(ref):
        m = ctypes.cast(ref, ctypes.POINTER(_MEMORYSTATUSEX)).contents
        seen["dwLength"] = m.dwLength
        m.ullTotalPhys = total_bytes
        m.ullAvailPhys = avail_bytes
        return ok

    monkeypatch.setattr(bc, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes.windll.kernel32, "GlobalMemoryStatusEx", fill)
    return bc.free_physical_gb(), seen.get("dwLength")


def _meminfo(monkeypatch, text):
    """free_physical_gb's /proc branch over `text`, or over an open() that
    raises when `text` is None. builtins.open is put back before the caller
    asserts anything: pytest opens source files to build a failure report, and
    a still-faked open turns a red test into a confusing one."""
    monkeypatch.setattr(bc, "sys", types.SimpleNamespace(platform="linux"))

    def fake_open(*a, **k):
        if text is None:
            raise OSError("no /proc on this box")
        return io.StringIO(text)

    real, builtins.open = builtins.open, fake_open
    try:
        return bc.free_physical_gb()
    finally:
        builtins.open = real


@pytest.mark.skipif(sys.platform != "win32",
                    reason="GlobalMemoryStatusEx exists only on Windows")
def test_the_memory_reader_reports_what_is_AVAILABLE_not_what_is_installed():
    """ullTotalPhys reads 31.6 GB on this box whatever is resident, so a
    fully loaded box would clear the default --min-free-gb 4.0 and the run
    would proceed -- the one failure the gate exists to prevent. The two
    fields carry different values here, so no single number satisfies both."""
    with pytest.MonkeyPatch.context() as mp:
        got, dw_length = _memory_status(mp, avail_bytes=8 * 1024 ** 3,
                                        total_bytes=31 * 1024 ** 3)
    assert got == 8.0
    assert dw_length == ctypes.sizeof(_MEMORYSTATUSEX) == 64, (
        "the struct this file declares is not the one Win32 was told about")


@pytest.mark.skipif(sys.platform != "win32",
                    reason="GlobalMemoryStatusEx exists only on Windows")
def test_the_memory_reader_converts_bytes_to_GB_and_not_to_MB():
    """The divisor is the whole conversion: 1024**2 reads 1024x HIGH and
    passes any box at all, 1024**4 reads low and refuses every one. Not a
    round number of GB, so a divisor off by a factor shows as a wrong figure
    rather than a plausible one."""
    with pytest.MonkeyPatch.context() as mp:
        got, _ = _memory_status(mp, avail_bytes=9_632_235_520)
    assert got == pytest.approx(8.97, abs=0.005)


@pytest.mark.skipif(sys.platform != "win32",
                    reason="GlobalMemoryStatusEx exists only on Windows")
def test_a_refused_memory_query_is_unknown_rather_than_a_number():
    """None, not 0.0 and not whatever the uninitialised buffer held. main()
    refuses an unknown outright and says --min-free-gb cannot help, which is
    only true if an unreadable box never arrives here as a figure."""
    with pytest.MonkeyPatch.context() as mp:
        got, _ = _memory_status(mp, avail_bytes=8 * 1024 ** 3, ok=0)
    assert got is None


def test_meminfo_is_read_for_MemAvailable_and_converted_from_kB(monkeypatch):
    """The /proc half. MemAvailable, not MemFree: MemFree excludes reclaimable
    cache and reads far LOW, which refuses a healthy box -- the opposite
    failure, the same broken gate. kB to GB is 1024**2, and the field is not
    the first line of the file."""
    got = _meminfo(monkeypatch, "MemTotal:       32115240 kB\n"
                                "MemFree:          812344 kB\n"
                                "MemAvailable:    9408512 kB\n"
                                "Buffers:          123456 kB\n")
    assert got == pytest.approx(9408512 / 1024 ** 2)
    assert got == pytest.approx(8.97, abs=0.005)


@pytest.mark.parametrize("text", [
    "MemTotal:       32115240 kB\nMemFree:          812344 kB\n",   # no field
    None,                                                           # no /proc
])
def test_a_meminfo_that_cannot_answer_is_unknown_rather_than_zero(monkeypatch,
                                                                  text):
    """Same rule as the Windows branch, and the reason the docstring gives for
    returning None: the precondition must fail loudly on an unknown rather
    than silently pass a box it could not measure."""
    assert _meminfo(monkeypatch, text) is None


# --- the load generator ------------------------------------------------------
# Load._run, start() and stop() were executed by no test: the one real Load
# only had its .timeout read, and every sweep test substituted a no-op fake.
# So the counting, the stop-flag loop and the join were all unpinned -- on the
# component whose misbehaviour (no load, or load that outlives its leg) makes
# a twenty-minute shared-box run publish a wrong ratio.

class _Flag:
    """The worker's stop flag, with a budget of looks.

    A worker that never reaches the scripted post -- it spelled the helper's
    name wrong, say, and its own guard swallowed the AttributeError -- is never
    stopped by the script and would spin here for ever. Past the budget the
    flag reads as set, so that defect is a red test rather than a hung suite.
    It records the pauses asked of it instead of sleeping them.
    """

    def __init__(self, budget):
        self.flag, self.looks, self.budget, self.waits = False, 0, budget, []

    def is_set(self):
        self.looks += 1
        return self.flag or self.looks > self.budget

    def set(self):
        self.flag = True

    def wait(self, seconds):
        self.waits.append(seconds)


def _script(monkeypatch, gen, outcomes):
    """Drive gen._run() through `outcomes`, one per request, then stop it.

    Each outcome is what post_timed returns -- (body, wall) or (None, reason)
    -- or an exception for it to raise. Returns the calls it received.
    """
    calls = []
    gen._stop = _Flag(budget=len(outcomes) + 3)

    def post(base, path, payload, timeout):
        calls.append((base, path, payload, timeout))
        outcome = outcomes[len(calls) - 1]
        if len(calls) == len(outcomes):
            gen._stop.set()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(bc.be, "post_timed", post)
    gen._run()
    assert gen._stop.flag, (
        "the worker never got through the script: %d of %d requests reached "
        "post_timed" % (len(calls), len(outcomes)))
    return calls


def _served(tokens):
    return {"usage": {"completion_tokens": tokens}}, 0.5


def test_load_generator_does_not_inherit_the_measurement_timeout():
    """stop() blocks behind an in-flight request, so the generator's
    per-request cap has to be short regardless of the measurement timeout."""
    gen = bc.Load("http://x", "m", 500, 62, timeout=1800)
    assert gen.timeout == bc.LOAD_REQUEST_TIMEOUT_S
    assert gen.join_timeout == bc.LOAD_REQUEST_TIMEOUT_S + bc.LOAD_JOIN_GRACE_S
    assert bc.Load("http://x", "m", 500, 62, timeout=20).timeout == 20


def test_the_generator_counts_served_shed_and_failed_apart(monkeypatch):
    """The three outcomes mean three different things for the leg, and the
    reason that tells them apart used to be thrown away as `_wall` under a
    comment explaining that a 429 and a refused connection were
    indistinguishable."""
    gen = bc.Load("http://peer", "m", 500, 62, timeout=60)
    calls = _script(monkeypatch, gen, [
        _served(62),
        (None, "HTTP 429 -- server busy; NPU is single-flight"),
        _served(60),
        (None, "URLError: <urlopen error [WinError 10061] refused>"),
        (None, "HTTP 529 -- overloaded"),
        (None, "TimeoutError: timed out"),
    ])
    assert len(calls) == 6, "no outcome may end the loop; only stop() does"
    assert gen.completed == 2
    assert gen.shed == 2
    assert gen.failed == 2
    assert gen.tokens_out == 122
    assert gen.first_failure.startswith("URLError"), "the FIRST reason is kept"


def test_the_generator_posts_the_measurements_request_shape(monkeypatch):
    gen = bc.Load("http://peer", "the-model", 500, 62, timeout=60)
    monkeypatch.setattr(bc.be, "prompt_of", lambda d: "prompt of %d" % d)
    (base, path, payload, timeout), = _script(monkeypatch, gen, [_served(62)])
    assert (base, path, timeout) == ("http://peer", "/v1/chat/completions", 60)
    assert payload["model"] == "the-model"
    assert payload["messages"] == [{"role": "user", "content": "prompt of 500"}]
    # Both cap spellings: geniex serve ignores the legacy one outright.
    assert payload["max_tokens"] == payload["max_completion_tokens"] == 62
    # Off, or llama-server serves every repeat of this identical prompt from
    # its cached prefix and the generator stops doing the prefill its
    # docstring says it does.
    assert payload["cache_prompt"] is False


def _real_bench_endpoint():
    """The REAL bench_endpoint, loaded from src the way _load loads the module
    under test. `bc.be` is the stub, which cannot say what be.chat actually
    sends -- and be.chat's body is the one the reshaped-load warmup proves
    servable, so it is the only thing worth holding the generator's copy
    against. Loaded under its own name, so sys.modules is untouched."""
    spec = importlib.util.spec_from_file_location("bench_endpoint_real",
                                                  SRC / "bench_endpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_generator_sends_the_body_the_warmup_proved_servable(monkeypatch):
    """The reshaped-load pre-flight's whole claim is that it sends the request
    "as the generator will send it" -- but it calls be.chat and Load._run
    builds its OWN copy of the body, and nothing held the two together. Held
    against be.chat's LIVE payload, the way tests/test_bench_servers.py holds
    bs.request_body for the sibling tool, so a change on either side fails
    here instead of splitting them quietly.

    The two no-thinking keys are what the shape test above stops short of, and
    they are the live exposure: THINKING_DEFAULT is off in this repo's server,
    but llama-server and geniex follow Qwen3's own template, which defaults
    thinking ON. Drop them from the generator and a peer that answered the
    warmup reshapes every generator request, so each contended leg is measured
    against a thinking peer the pre-flight never proved."""
    real = _real_bench_endpoint()
    seen = {}

    def capture(base, path, payload, timeout):
        seen.update(base=base, path=path, payload=payload, timeout=timeout)
        return None, "captured"

    monkeypatch.setattr(real, "post_timed", capture)
    real.chat("http://peer", "the-model", "prompt of 500", 62, 60)

    gen = bc.Load("http://peer", "the-model", 500, 62, timeout=60)
    monkeypatch.setattr(bc.be, "prompt_of", lambda d: "prompt of %d" % d)
    (base, path, payload, timeout), = _script(monkeypatch, gen, [_served(62)])
    assert (base, path, timeout) == (seen["base"], seen["path"], seen["timeout"])
    assert payload == seen["payload"], (
        "the generator's body has drifted from the one the warmup sends")
    # Stated as well as compared: both bodies moving together would keep the
    # equality above green while changing what the peer decodes.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"


def test_only_the_first_failure_is_announced(monkeypatch, capsys):
    gen = bc.Load("http://peer", "m", 500, 62, timeout=60)
    _script(monkeypatch, gen, [(None, "URLError: refused"),
                               (None, "HTTP 500 -- boom"),
                               (None, "HTTP 429 -- busy")])
    out = capsys.readouterr().out
    assert out.count("FAILED") == 1
    assert "URLError: refused" in out and "HTTP 500" not in out
    assert "not as backpressure" in out
    assert gen.first_failure == "URLError: refused"
    assert (gen.failed, gen.shed) == (2, 1)


def test_a_shed_request_is_never_announced_as_a_failure(monkeypatch, capsys):
    gen = bc.Load("http://peer", "m", 500, 62, timeout=60)
    _script(monkeypatch, gen, [(None, "HTTP 429 -- busy")] * 3)
    assert capsys.readouterr().out == ""
    assert gen.first_failure is None and (gen.shed, gen.failed) == (3, 0)


def test_a_worker_exception_is_counted_not_fatal(monkeypatch):
    """A worker that dies takes the load with it, and the leg then runs against
    an idle peer and publishes a ratio near 1.0. post_timed catches its own
    errors, so this is the guard for everything around it."""
    gen = bc.Load("http://peer", "m", 500, 62, timeout=60)
    calls = _script(monkeypatch, gen, [RuntimeError("boom"), _served(5)])
    assert len(calls) == 2, "the loop survived the exception"
    assert (gen.completed, gen.failed) == (1, 1)
    assert gen.first_failure == "RuntimeError: boom"


def test_a_served_body_that_is_not_an_object_still_counts(monkeypatch):
    # Valid JSON that is not a dict (a list, a number) has no usage block. It
    # was served; it must not raise AttributeError out of the worker.
    gen = bc.Load("http://peer", "m", 500, 62, timeout=60)
    _script(monkeypatch, gen, [([1, 2], 0.1), ({"usage": None}, 0.1)])
    assert (gen.completed, gen.failed, gen.tokens_out) == (2, 0, 0)


def test_a_failure_and_a_shed_both_pause_the_generator(monkeypatch):
    """Neither kind of unserved request may be re-posted at once, because both
    come back in about the time a TCP round trip takes and the spin burns a
    core the timed leg shares.

    This test used to assert the opposite for a shed -- one wait, for the
    failure only -- on the rationale that "a 429 costs the peer a lock probe"
    and that re-posting keeps its queue full. Measured (loopback harness,
    every permit taken, depth 2000, 5 s): 246 requests/s at 39% of one core,
    1234 server log lines; 19 requests at 2.8% with the pause. A shed is a
    full body read and json parse on the peer before it declines, and
    shed_note's own rule says a lone client only gets a 429 when SOMEBODY
    ELSE holds the peer -- so there was no queue of ours to keep full.

    A shed still pauses for less than a failure: the peer is alive, and we
    want to be the next request it accepts."""
    gen = bc.Load("http://peer", "m", 500, 62, timeout=60)
    _script(monkeypatch, gen, [(None, "HTTP 429 -- busy"),
                               (None, "URLError: refused"), _served(3)])
    # Recorded by _Flag, which is what proves the pause is STOP-AWARE: a
    # time.sleep would leave this list empty (and would be added to every
    # stop(), which the worker can only reach between requests).
    assert gen._stop.waits == [bc.LOAD_SHED_PAUSE_S, bc.LOAD_FAILURE_PAUSE_S]
    assert 0 < bc.LOAD_SHED_PAUSE_S < bc.LOAD_FAILURE_PAUSE_S


def test_is_backpressure_is_429_and_529_only():
    assert bc.is_backpressure("HTTP 429 -- server busy")
    assert bc.is_backpressure("HTTP 529")
    assert not bc.is_backpressure("HTTP 500 -- boom")
    assert not bc.is_backpressure("HTTP 400 -- prompt too long")
    assert not bc.is_backpressure("URLError: refused")
    assert not bc.is_backpressure(None)
    assert not bc.is_backpressure(0.25), "a wall time is not a reason"


def test_start_and_stop_run_the_worker_and_join_it(monkeypatch):
    """The real thread: start() puts load on, stop() takes it off and waits.
    `seconds` is the denominator of the recorded duty cycle."""
    first = threading.Event()

    def post(*a):
        first.set()
        time.sleep(0.001)
        return _served(2)

    monkeypatch.setattr(bc.be, "post_timed", post)
    gen = bc.Load("http://peer", "m", 8, 2, timeout=5)
    gen.join_timeout = 3        # so a broken stop() fails this test in seconds
    gen.start()
    try:
        assert first.wait(5), "the worker never posted"
    finally:
        gen.stop()
        alive = gen._thread.is_alive()
        gen._stop.set()         # whatever stop() did, never leave it running
    assert not alive, "stop() returned with the generator still running"
    served = gen.completed
    assert served >= 1 and gen.tokens_out == 2 * served
    assert gen.seconds > 0
    time.sleep(0.02)
    assert gen.completed == served, "load continued after stop()"
    assert bc.GATE_NOTES == []


def test_a_generator_that_outlives_its_join_is_recorded(capsys):
    """Load outliving its leg is the one thing the try/finally around the
    contended leg exists to prevent, and stop() used to return silently when
    the join timed out -- a straggling request overlapping the next solo
    baseline with no record of it."""
    gen = bc.Load("http://peer", "m", 8, 2, timeout=5)

    class Stuck:
        waited = None

        def join(self, timeout=None):
            self.waited = timeout

        def is_alive(self):
            return True

    gen._thread = Stuck()
    gen.stop()
    assert gen._stop.is_set()
    assert gen._thread.waited == 5 + bc.LOAD_JOIN_GRACE_S
    assert len(bc.GATE_NOTES) == 1
    assert "still in flight" in bc.GATE_NOTES[0] and "http://peer" in bc.GATE_NOTES[0]
    assert "WARNING" in capsys.readouterr().out


def test_stop_before_start_is_harmless():
    gen = bc.Load("http://peer", "m", 8, 2, timeout=5)
    gen.stop()
    assert gen.seconds == 0.0 and bc.GATE_NOTES == []


# --- the cool gate -----------------------------------------------------------

def test_gate_fires_only_when_cool_floor_is_set(monkeypatch):
    """The regression that mattered: the gate was unreachable, silently."""
    calls = []
    monkeypatch.setattr(bc, "wait_for_cool", lambda floor: calls.append(floor))
    bc.measure("u", "m", 1, 1, 1, "gated", cool_floor=92.0)
    assert calls == [92.0]
    bc.measure("u", "m", 1, 1, 1, "ungated")
    bc.measure("u", "m", 1, 1, 1, "ungated", cool_floor=0.0)
    assert calls == [92.0], "measure() must not cool when no floor is given"


def test_measure_records_the_reading_the_gate_passed_on(monkeypatch):
    """These are the readings the run's clock warning keys on. It used to key
    on a round-START reading taken before the gate ran -- from round 2 on, the
    dip left by the previous contended leg, which the gate then waits out --
    while measure() called wait_for_cool as a bare statement and threw the
    recovered reading away."""
    readings = iter([97.0, None, 88.0])
    monkeypatch.setattr(bc, "wait_for_cool", lambda floor: next(readings))
    monkeypatch.setattr(bc.be, "measure_decode", lambda *a: 12.5)
    clocks = []
    for _ in range(3):
        assert bc.measure("u", "m", 1, 1, 1, "x", cool_floor=92.0,
                          clocks=clocks) == 12.5
    assert clocks == [97.0, 88.0], "an unreadable gate records no reading"
    bc.measure("u", "m", 1, 1, 1, "x", clocks=clocks)
    assert clocks == [97.0, 88.0], "an ungated sample records none either"


def test_measure_passes_the_request_straight_through(monkeypatch):
    seen = []
    monkeypatch.setattr(bc.be, "measure_decode",
                        lambda *a: seen.append(a) or 7.0)
    assert bc.measure("http://u", "m", 500, 120, 1800, "NPU solo") == 7.0
    assert seen == [("http://u", "m", 500, 120, 1800)]


def test_an_unreadable_counter_ungates_the_sample_and_says_so(monkeypatch,
                                                             capsys):
    """cpu_performance_pct returns None off-Windows and on ANY failure -- a 30 s
    subprocess timeout, an empty counter read. wait_for_cool returned the same
    silent None, measure() discarded it, and a fast gate pass prints nothing
    either: a sample taken on a box that never recovered was indistinguishable
    from a gated one, in the terminal and in the JSON."""
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: None)
    assert bc.wait_for_cool(92.0, limit=1) is None
    assert len(bc.GATE_NOTES) == 1
    assert "could not be read" in bc.GATE_NOTES[0]
    assert "UNGATED" in bc.GATE_NOTES[0]
    assert "UNGATED" in capsys.readouterr().out


def test_a_counter_lost_mid_wait_names_the_reading_it_had(monkeypatch):
    # The worse case: the box WAS below the floor, and then the counter went.
    _fake_clock(monkeypatch)
    readings = iter([70.0, None])
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: next(readings))
    monkeypatch.setattr(bc, "power_reading", lambda: ("ac", 90.0, 5.0))
    assert bc.wait_for_cool(92.0, limit=300) is None
    assert "after a 70% reading" in bc.GATE_NOTES[0]


def test_a_gate_that_gives_up_is_recorded_not_only_printed(monkeypatch, capsys):
    """The literal gave-up-on-cooling case. The commit that added GATE_NOTES
    named exactly this defect ("the results file showed a clean run where the
    harness had actually given up on cooling") and then fixed only the power
    branch; the timeout kept printing and never recording."""
    slept = _fake_clock(monkeypatch)
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 61.0)
    monkeypatch.setattr(bc, "power_reading", lambda: ("ac", 90.0, 5.0))
    assert bc.wait_for_cool(92.0, limit=300) == 61.0
    assert sum(slept) == 300, "it waited out the whole limit first"
    assert len(bc.GATE_NOTES) == 1
    assert "clock still 61% after 300s" in bc.GATE_NOTES[0]
    assert "NOT recovered" in bc.GATE_NOTES[0]
    assert "WARNING" in capsys.readouterr().out


def test_wait_for_cool_does_not_raise_when_limit_is_zero(monkeypatch):
    """The loop body never runs, so the warning path must not touch an
    unbound name -- and giving up without a single reading is still giving
    up."""
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 50.0)
    assert bc.wait_for_cool(92.0, limit=0) is None
    assert "clock still ?% after 0s" in bc.GATE_NOTES[0]


def test_wait_for_cool_handles_a_real_counter_reading(monkeypatch):
    # The two cases above both dodge the body: an unreadable counter returns
    # before the tracking, and limit=0 never enters the loop. So a gate that
    # crashed on EVERY real reading passed both. It did -- `first` was read
    # before it was assigned, so the first successful sample raised
    # UnboundLocalError. The gate had never gated anything, first because it
    # was dead code and then because wiring it exposed this.
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 97.0)
    assert bc.wait_for_cool(92.0, limit=5) == 97.0
    assert bc.GATE_NOTES == [], "a clean pass is the one outcome with no note"


def test_wait_for_cool_returns_the_sample_it_gated_on(monkeypatch):
    # Below the floor once, then above: the value returned must be the reading
    # that satisfied the gate, not a fresh sample taken afterwards.
    slept = _fake_clock(monkeypatch)
    seq = iter([80.0, 95.0])
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: next(seq))
    # Stubbed because the power check added later SHELLS OUT, which made this
    # environment-dependent: run on a laptop actually on battery, the gate
    # aborts on the 80.0 reading instead of waiting for the 95.0. It caught
    # exactly that the day the check landed. The AC branch reads the PACK (not
    # the CPU-busy figure this test stubbed for a while, which that branch
    # never calls), so the pack is what has to be stubbed.
    monkeypatch.setattr(bc, "power_reading", lambda: ("ac", 90.0, 5.0))
    assert bc.wait_for_cool(92.0, limit=30) == 95.0
    assert slept == [10]
    assert bc.GATE_NOTES == []


def _count_reads(monkeypatch, src, charge=90.0, busy=90.0):
    """Run power_limited_note below the floor, counting each hardware read.

    Every one of these is a PowerShell launch and this gate runs before every
    sample, so the count is the thing worth pinning -- not as a micro-benchmark
    but because it grew from two to three without anyone noticing.
    """
    calls = []

    def rec(name, value):
        def f(*a, **k):
            calls.append(name)
            return value
        return f

    monkeypatch.setattr(bc, "power_reading",
                        rec("power_reading", (src, charge, 30.0)))
    monkeypatch.setattr(bc, "cpu_busy_pct", rec("cpu_busy_pct", busy))
    bc.power_limited_note(45.0, 92.0)
    return calls


def test_the_gate_takes_only_the_readings_it_uses(monkeypatch):
    # cpu_busy_pct's value is read ONLY in the no-battery branch, but it used to
    # be called unconditionally -- so every AC run paid a subprocess for a
    # number it then discarded.
    ac = _count_reads(monkeypatch, "ac")
    assert "cpu_busy_pct" not in ac, "AC path read a CPU figure it cannot use"
    # ONE launch, including with a low pack to report: the source and the pack
    # used to be two launches of the same query, the first discarding the pack
    # figures the second went back for.
    assert ac == ["power_reading"]
    assert _count_reads(monkeypatch, "ac", charge=14.0) == ["power_reading"]

    assert _count_reads(monkeypatch, "no-battery") == ["power_reading", "cpu_busy_pct"]

    # The two definitive answers cost one reading and stop.
    assert _count_reads(monkeypatch, "battery") == ["power_reading"]
    assert _count_reads(monkeypatch, "unknown") == ["power_reading"]


def test_the_AC_advisory_costs_one_launch_at_the_module_boundary(monkeypatch):
    # The same count taken where the cost actually is: every call that crosses
    # into bench_endpoint here is a PowerShell launch. Through the real
    # power_reading and the real mapping, not a stub of either.
    launches = []
    monkeypatch.setattr(bc.be, "power_reading",
                        lambda: launches.append(1) or ("True", 14.0, 30.0))
    msg, abort = bc.power_limited_note(45.0, 92.0)
    assert "pack is at 14% drawing 30 W" in msg and abort is False
    assert launches == [1]


def test_the_gate_reads_nothing_at_all_above_the_floor(monkeypatch):
    # The common case by far: the clock is fine, so there is nothing to explain
    # and no reason to touch the hardware.
    calls = []
    monkeypatch.setattr(bc, "power_reading",
                        lambda: calls.append("power_reading") or ("ac", 90.0, 5.0))
    assert bc.power_limited_note(99.0, 92.0) == (None, False)
    assert calls == []


def test_a_deeply_discharged_pack_is_flagged_even_on_AC(monkeypatch):
    """AC used to fall off the end of the check and read as clean.

    That is the state an operator reaches by plugging in and starting
    immediately, and it is the one that actually costs prefill: measured across
    two sessions on this box, legs at 13-20% gave pp512 ~58 against a settled
    130, while 33% and 41.6% came back near baseline. The hazard is depth of
    discharge, not charging -- the opposite of what was assumed before the legs
    were pooled.
    """
    monkeypatch.setattr(bc, "power_reading", lambda: ("ac", 14.0, 30.0))
    msg, abort = bc.power_limited_note(45.0, 92.0)
    assert msg is not None and "14%" in msg
    assert abort is False, "advisory only -- bandwidth-bound work is immune"
    assert "not a settled-box measurement" in msg


def test_a_healthy_pack_on_AC_stays_silent(monkeypatch):
    # The counterpart. Warning on a charged box is how the real warning gets
    # skipped.
    monkeypatch.setattr(bc, "power_reading", lambda: ("ac", 88.0, 2.0))
    assert bc.power_limited_note(45.0, 92.0) == (None, False)


def test_an_unreadable_pack_does_not_invent_a_charge_warning(monkeypatch):
    # The pack figures are None whenever the class did not supply them. Absence
    # of a reading must not become a claim about the reading.
    monkeypatch.setattr(bc, "power_reading", lambda: ("ac", None, None))
    assert bc.power_limited_note(45.0, 92.0) == (None, False)


def test_power_limited_note_is_silent_when_the_clock_is_fine():
    assert bc.power_limited_note(99.0, 92.0) == (None, False)
    assert bc.power_limited_note(None, 92.0) == (None, False)


def test_power_limited_note_names_the_battery(monkeypatch):
    # Definitive path: the machine reports it is on battery.
    monkeypatch.setattr(bc, "power_reading",
                        lambda: ("battery", None, None))
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert msg is not None and "BATTERY" in msg
    assert "will NOT recover" in msg, "must say waiting is futile, not just why"
    assert abort is True, "the battery case is the ONLY one that aborts"


def test_idle_fingerprint_advises_but_does_NOT_abort(monkeypatch):
    # The gate runs BEFORE each sample, when the box is legitimately idle and
    # downclocked. Treating that as power limiting aborted the gate on a
    # healthy run -- on a machine reporting no battery the gate would never
    # work at all. Advice yes, abort no.
    monkeypatch.setattr(bc, "power_reading",
                        lambda: ("no-battery", None, None))
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 9.0)
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert msg is not None
    assert abort is False, "an idle low clock must not abort the gate"


def test_unknown_power_source_does_not_abort(monkeypatch):
    # A failed query used to return the same None as "no battery", silently
    # downgrading a definitive check to a heuristic on a laptop.
    monkeypatch.setattr(bc, "power_reading",
                        lambda: ("unknown", None, None))
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert msg is not None and "could not be read" in msg
    assert abort is False


def test_power_limited_note_stays_quiet_when_the_box_is_busy(monkeypatch):
    # Low clock + BUSY cpu is the thermal case: waiting DOES help, so the gate
    # must keep waiting rather than aborting.
    monkeypatch.setattr(bc, "power_reading",
                        lambda: ("no-battery", None, None))
    monkeypatch.setattr(bc, "cpu_busy_pct", lambda: 85.0)
    msg, abort = bc.power_limited_note(31.0, 92.0)
    assert abort is False, "thermal case: the gate must keep waiting"
    assert msg is None, "and it must not muddy the log with a power note"


def test_wait_for_cool_aborts_instead_of_blocking_on_battery(monkeypatch):
    """No ten-minute silence waiting for a recovery that cannot come -- and no
    return value either. The gate used to say 'ABORTING THE GATE' and hand the
    reading back, whereupon the sample was taken anyway, entered the medians,
    and the run exited 0 under a message saying nothing measured on battery is
    worth keeping."""
    slept = _fake_clock(monkeypatch)
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: 31.0)
    monkeypatch.setattr(bc, "power_reading",
                        lambda: ("battery", None, None))
    with pytest.raises(bc.GateAborted) as e:
        bc.wait_for_cool(92.0, limit=300)
    assert e.value.pct == 31.0 and "ON BATTERY" in e.value.why
    assert slept == [], "must stop immediately, not spin out the limit"
    assert len(bc.GATE_NOTES) == 1
    assert bc.GATE_NOTES[0].startswith("gate ABORTED: ")


def test_an_advisory_note_is_recorded_and_the_gate_keeps_waiting(monkeypatch):
    _fake_clock(monkeypatch)
    seq = iter([45.0, 96.0])
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: next(seq))
    monkeypatch.setattr(bc, "power_reading", lambda: ("ac", 14.0, 30.0))
    assert bc.wait_for_cool(92.0, limit=300) == 96.0
    assert len(bc.GATE_NOTES) == 1 and "pack is at 14%" in bc.GATE_NOTES[0]
    assert not bc.GATE_NOTES[0].startswith("gate ABORTED")


def test_repeated_notes_collapse_to_the_first_with_a_count():
    """The AC-low-pack advisory fires before every gated leg -- 2*repeat+1
    times a run -- with that leg's numbers interpolated, so the copies are
    equivalent without being identical. An artifact that says one thing seven
    ways teaches its reader to skip the eighth warning."""
    notes = ["clock is 45% of base, and the pack is at 14% drawing 30 W.",
             "the load generator on http://a was still in flight 210s after",
             "clock is 51% of base, and the pack is at 16% drawing 28 W.",
             "clock is 60.5% of base, and the pack is at 19% drawing 31 W."]
    out = bc.dedupe_notes(notes)
    assert len(out) == 2
    assert out[0].startswith(notes[0]), "the FIRST occurrence is the one shown"
    assert "raised 3 times" in out[0]
    assert out[1] == notes[1], "a note raised once is left exactly as it was"
    assert bc.dedupe_notes([]) == []


def _straggler_notes(*stragglers):
    """The notes the REAL Load.stop() raises for generators that outlived
    their join, one per (base, request timeout), so the text under test is
    whatever stop() writes today rather than a copy of it."""
    class Stuck:
        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    for base, timeout in stragglers:
        gen = bc.Load(base, "m", 8, 2, timeout=timeout)
        gen._thread = Stuck()
        gen.stop()
    notes = list(bc.GATE_NOTES)
    del bc.GATE_NOTES[:]
    return notes


def test_stragglers_on_different_servers_are_not_collapsed_into_one():
    """The shape key replaced EVERY number, the port included, and at the
    default --npu/--gpu the port is all that tells the engines apart. One GPU
    straggler and two NPU ones came out as a single note naming 8124 "raised 3
    times" -- and which server had the overlapping request is the whole
    content of that note: an NPU straggler queues the next NPU solo baseline
    behind it on the single-flight server."""
    npu, gpu = "http://127.0.0.1:8123", "http://127.0.0.1:8124"
    notes = _straggler_notes((gpu, 180), (npu, 180), (npu, 180))
    assert len(notes) == 3
    out = bc.dedupe_notes(notes)
    assert len(out) == 2, "one note per SERVER, not one for the pair"
    assert gpu in out[0] and "raised" not in out[0]
    assert npu in out[1] and "raised 2 times" in out[1]


def test_a_url_shields_only_its_own_numbers():
    """The counterpart: numbers OUTSIDE the URL are still readings, so repeats
    on one server collapse whatever their other figures say -- here the join
    timeout, which differs when --timeout is below the generator's own cap."""
    out = bc.dedupe_notes(_straggler_notes(("http://127.0.0.1:8123", 180),
                                           ("http://127.0.0.1:8123", 45)))
    assert len(out) == 1 and "raised 2 times" in out[0]
    assert "still in flight 210s" in out[0], "the first is the one shown"


def test_drift_note_flags_only_a_monotonic_decline():
    note = bc.drift_note([20.0, 18.0, 16.0], "x")
    assert note is not None and "20.00 -> 16.00, -20.0%" in note
    assert bc.drift_note([20.0, 16.0, 18.0], "x") is None
    assert bc.drift_note([20.0, 18.0], "x") is None, "needs 3+ samples"


# --- paired_sweep ------------------------------------------------------------

ENGINES = [("NPU", "http://a", "npu-model"), ("GPU", "http://b", "gpu-model")]


class FakeLoad:
    """A generator that reports a fixed leg: what paired_sweep reads off one."""
    completed, shed, failed = 4, 0, 0
    first_failure = None
    tokens_out, seconds = 0, 0.0

    def start(self):
        pass

    def stop(self):
        pass


def _sweep_args(**kw):
    a = types.SimpleNamespace(repeat=1, depth=250, tokens=40, timeout=60,
                              ramp=0, cool_floor=92.0,
                              closing_recheck=True, closing_tol=10.0)
    a.__dict__.update(kw)
    return a


def _sweep(monkeypatch, measure_returns, make_load=None, **kw):
    """The real paired_sweep over a scripted sequence of measure() results.

    The order measure() is called in is the sweep's order: per round, per
    engine, solo then contended; then the closing leg.
    """
    rates = iter(measure_returns)
    monkeypatch.setattr(bc, "measure", lambda *a, **k: next(rates))
    return bc.paired_sweep(ENGINES, _sweep_args(**kw),
                           make_load or (lambda other: FakeLoad()))


def test_the_ratios_are_paired_per_round_not_formed_from_the_lists(
        monkeypatch, capsys):
    """The headline arithmetic. A ratio exists only for a round that produced
    BOTH samples, and it divides that round's contended rate by THAT ROUND's
    solo rate -- which is what cancels slow drift. GPU's round 1 contended
    sample is skipped here, so its one ratio must be round 2's pair (21/28),
    not 21 over the first solo sample."""
    per, _clocks, _closing = _sweep(
        monkeypatch,
        [20.0, 10.0, 30.0, None,        # round 1: NPU solo/cont, GPU solo/cont
         18.0, 12.0, 28.0, 21.0],       # round 2
        repeat=2, closing_recheck=False)
    assert per["NPU"]["solo"] == [20.0, 18.0]
    assert per["NPU"]["contended"] == [10.0, 12.0]
    assert per["NPU"]["ratios"] == [10.0 / 20.0, 12.0 / 18.0]
    assert per["GPU"]["solo"] == [30.0, 28.0]
    assert per["GPU"]["contended"] == [21.0]
    assert per["GPU"]["ratios"] == [21.0 / 28.0]
    out = _flat(capsys.readouterr().out)
    assert "pair: 20.00 -> 10.00 t/s (keeps 50.0%)" in out
    assert "pair: 28.00 -> 21.00 t/s (keeps 75.0%)" in out
    assert out.count("pair:") == 3, "no pair line for the skipped sample"


def test_each_leg_is_measured_as_labelled_and_only_solo_is_gated(monkeypatch):
    seen = []

    def measure(base, model, depth, tokens, timeout, label, cool_floor=None,
                clocks=None):
        seen.append((base, model, depth, tokens, timeout, label, cool_floor))
        return 10.0

    monkeypatch.setattr(bc, "measure", measure)
    loaded = []

    def make_load(other):
        loaded.append(other[0])
        return FakeLoad()

    bc.paired_sweep(ENGINES, _sweep_args(), make_load)
    assert seen == [
        ("http://a", "npu-model", 250, 40, 60, "NPU solo", 92.0),
        ("http://a", "npu-model", 250, 40, 60, "NPU vs GPU busy", None),
        ("http://b", "gpu-model", 250, 40, 60, "GPU solo", 92.0),
        ("http://b", "gpu-model", 250, 40, 60, "GPU vs NPU busy", None),
        ("http://a", "npu-model", 250, 40, 60, "NPU solo (closing)", 92.0),
    ]
    assert loaded == ["GPU", "NPU"], "the load goes on the OTHER engine"


def test_the_generators_counts_accumulate_per_measured_engine(monkeypatch):
    class Leg(FakeLoad):
        completed, shed, failed = 3, 2, 1
        first_failure = "URLError: refused"
        tokens_out, seconds = 40, 2.5

    per, _c, _cl = _sweep(monkeypatch, [10.0] * 8, repeat=2,
                          closing_recheck=False, make_load=lambda o: Leg())
    rec = per["NPU"]
    assert (rec["served"], rec["shed"], rec["failed"]) == (6, 4, 2)
    assert rec["first_failure"] == "URLError: refused"
    assert (rec["load_tokens_out"], rec["load_seconds"]) == (80, 5.0)


def test_one_box_reading_per_round_and_no_second_launch(monkeypatch):
    """Each round used to launch PowerShell twice -- the clock counter, then
    the battery query, which runs that same counter and discards it -- for
    readings one launch already returns."""
    reads = []
    monkeypatch.setattr(bc.be, "box_state",
                        lambda: reads.append(1) or (True, 80.0, 12.0, 96.5))
    for probe in ("cpu_performance_pct", "power_reading"):
        monkeypatch.setattr(bc, probe, lambda probe=probe: pytest.fail(
            "%s read during the round-start sample" % probe))
    _sweep(monkeypatch, [10.0] * 8, repeat=2, closing_recheck=False)
    assert len(reads) == 2
    assert bc.POWER_SAMPLES == [
        {"round": 1, "on_ac": True, "charge_pct": 80.0, "charge_w": 12.0,
         "clock_pct": 96.5},
        {"round": 2, "on_ac": True, "charge_pct": 80.0, "charge_w": 12.0,
         "clock_pct": 96.5}]


@pytest.mark.parametrize("on_ac,word", [
    (True, "AC"), (False, "BATTERY"), (None, "power source unreadable")])
def test_an_unreadable_power_source_is_not_rendered_as_battery(
        monkeypatch, capsys, on_ac, word):
    # The flag is None when the class did not say. `"AC" if ac else "BATTERY"`
    # turned that into the one word an operator would act on.
    monkeypatch.setattr(bc.be, "box_state", lambda: (on_ac, 57.0, None, 88.4))
    bc._round_state(0)
    out = _flat(capsys.readouterr().out)
    assert "power %s, pack 57%%" % word in out
    assert ("BATTERY" in out) == (on_ac is False)
    assert "cpu clock 88.4% of base (round start, before the gate)" in out
    assert bc.POWER_SAMPLES[0]["on_ac"] is on_ac


def test_a_round_with_nothing_readable_records_nothing(capsys):
    bc._round_state(0)          # the stub's box_state is all None
    assert bc.POWER_SAMPLES == []
    assert "power" not in capsys.readouterr().out


def test_the_load_generator_is_stopped_even_when_the_leg_raises(monkeypatch):
    """It is a live load on a SHARED box; an interrupt used to skip stop()."""
    events = []

    class Watched(FakeLoad):
        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:          # the CONTENDED leg, generator running
            raise KeyboardInterrupt
        return 10.0

    monkeypatch.setattr(bc, "measure", boom)
    with pytest.raises(KeyboardInterrupt):
        bc.paired_sweep(ENGINES, _sweep_args(closing_recheck=False),
                        lambda o: Watched())
    assert events == ["start", "stop"], "the generator was left hammering the peer"


def _battery_after(monkeypatch, good_gates):
    """The real measure() and the real gate, on a box that goes to battery
    after `good_gates` clean readings. Returns the legs actually measured."""
    readings = iter([99.0] * good_gates + [31.0] * 9)
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: next(readings))
    monkeypatch.setattr(bc, "power_reading",
                        lambda: ("battery", None, None))
    measured = []
    monkeypatch.setattr(bc.be, "measure_decode",
                        lambda base, *a: measured.append(base) or 10.0)
    return measured


def test_a_gate_abort_stops_the_sweep_and_takes_no_further_sample(monkeypatch,
                                                                  capsys):
    """'Nothing measured on battery is worth keeping' -- so nothing more is
    measured. The power source does not change between legs, so every later
    gate would abort the same way."""
    measured = _battery_after(monkeypatch, good_gates=2)
    started = []

    def make_load(other):
        started.append(other[0])
        return FakeLoad()

    per, clocks, closing = bc.paired_sweep(
        ENGINES, _sweep_args(repeat=3), make_load)
    assert closing == {"state": "gate-aborted", "engine": "NPU", "round": 2}
    assert measured == ["http://a", "http://a", "http://b", "http://b"], (
        "round 1 stands; round 2's NPU solo was never taken")
    assert started == ["GPU", "NPU"], "no generator was started after the abort"
    assert per["NPU"]["solo"] == [10.0] and per["GPU"]["ratios"] == [1.0]
    assert clocks == [99.0, 99.0], "only readings a gate PASSED on are recorded"
    assert "SWEEP STOPPED at round 2, NPU solo" in capsys.readouterr().out


def test_a_gate_abort_on_the_closing_leg_is_its_own_state(monkeypatch, capsys):
    measured = _battery_after(monkeypatch, good_gates=2)
    _per, _clocks, closing = bc.paired_sweep(ENGINES, _sweep_args(),
                                             lambda o: FakeLoad())
    assert closing == {"state": "gate-aborted", "engine": "NPU",
                       "round": "closing"}
    assert len(measured) == 4, "the closing sample was not taken"
    assert "CLOSING RE-CHECK STOPPED" in capsys.readouterr().out


# --- the closing re-check --------------------------------------------------
# Re-runs the leg the sweep OPENED with, last, under the same gate: an A/A
# whose only variable is elapsed time. It exists because neither control
# already here can see decay DURING a sample -- wait_for_cool gates before one
# and says nothing after, and drift_note needs a strictly monotonic decline
# over 3+ solo samples, so one out-of-order sample hides a real trend and
# --repeat 1 gives it nothing to compare.

def _closing(first, final, engine="NPU", first_round=1):
    return {"state": "ok", "engine": engine, "first": first, "final": final,
            "first_round": first_round, "retained": final / first}


def test_a_box_that_held_is_reported_as_holding():
    note, suspect = bc.closing_note(_closing(18.0, 17.6))
    assert suspect is False
    assert "held" in note and "SUSPECT" not in note
    assert "round" not in note, "round 1 IS the opening; nothing to qualify"


def test_a_decayed_box_is_flagged_and_says_which_way_it_biases():
    # Contended samples are taken AFTER the solo ones they divide, so decay
    # lands in the numerator. Naming the direction is the difference between a
    # warning and an actionable one.
    note, suspect = bc.closing_note(_closing(18.0, 13.0))
    assert suspect is True
    assert "SUSPECT" in note and "DECAYED" in note
    assert "overstated" in note


def test_a_box_that_got_faster_is_equally_disqualifying():
    # Not a nice surprise: it means the OPENING sample was the degraded one, so
    # every baseline the ratios divide by is too low.
    note, suspect = bc.closing_note(_closing(13.0, 18.0))
    assert suspect is True
    assert "FASTER" in note and "flattered" in note


def test_the_check_reports_both_ends_and_the_percentage():
    # A reader has to be able to judge it without re-deriving the arithmetic.
    note, _ = bc.closing_note(_closing(20.0, 15.0))
    assert "20.00" in note and "15.00" in note and "-25.0%" in note


def test_the_tolerance_is_not_hair_trigger():
    # This box moves a few percent between any two samples; a check that fires
    # on ordinary noise is one the reader learns to skip.
    assert bc.closing_note(_closing(18.0, 17.3))[1] is False
    assert bc.closing_note(_closing(18.0, 18.7))[1] is False


def test_the_tolerance_is_configurable_in_both_directions():
    tight = bc.closing_note(_closing(18.0, 17.0), tol_pct=1.0)
    assert tight[1] is True
    assert bc.closing_note(_closing(18.0, 17.0), tol_pct=25.0)[1] is False


def test_a_skipped_check_says_so_rather_than_reading_as_clean():
    # The distinction that matters: "not checked" must not look like "checked
    # and fine", which is the same defect this suite pins in _probe_crosscheck.
    # The discriminator is the CLAIM, not the word "held": these say "nothing
    # here says whether the box held", the opposite of the confirming branch's
    # "the ratios above stand".
    for closing in ({"state": "disabled"}, None):
        note, suspect = bc.closing_note(closing)
        assert suspect is False
        assert "no closing re-check" in note
        assert "ratios above stand" not in note


@pytest.mark.parametrize("state,phrase", [
    ("disabled", "--no-closing-recheck"),
    ("failed", "closing measurement itself failed"),
    ("no-opening-sample", "no sample to compare against"),
    ("gate-aborted", "the gate ABORTED (NPU solo, round 2)"),
])
def test_each_reason_for_no_result_says_which_one(state, phrase):
    """Several causes used to print one identical line.

    They want opposite responses -- nothing at all for a deliberate skip, look
    at the closing leg for a failure, look at the whole sweep for an empty
    opening leg, plug the box in for an abort -- and a null in the JSON
    additionally read as "the flag was off". A reader could not tell which had
    happened.
    """
    note, suspect = bc.closing_note({"state": state, "engine": "NPU",
                                     "round": 2})
    assert suspect is False
    assert phrase in note
    assert "ratios above stand" not in note


def test_the_sweep_reopens_the_first_leg_and_records_it(monkeypatch):
    # End to end through paired_sweep: the closing measurement must be the
    # FIRST engine's solo leg, not the last one measured.
    _per, _clocks, closing = _sweep(monkeypatch, [18.0, 13.0, 17.0, 12.0, 9.0])
    assert closing == {"state": "ok", "engine": "NPU", "first": 18.0,
                       "final": 9.0, "first_round": 1, "retained": 0.5}
    assert bc.closing_note(closing)[1] is True


def test_an_opening_sample_from_a_later_round_is_labelled_as_such(monkeypatch):
    """`first` is the first SUCCESSFUL solo sample. When round 1's was skipped
    that is round 2's, and the A/A then brackets less of the run than its
    wording ("the leg this run opened with") says."""
    _per, _clocks, closing = _sweep(
        monkeypatch,
        [None, 10.0, 30.0, 20.0,        # round 1: the NPU solo leg is skipped
         18.0, 12.0, 28.0, 21.0,        # round 2
         17.5],                         # closing
        repeat=2)
    assert closing["first"] == 18.0 and closing["first_round"] == 2
    note, suspect = bc.closing_note(closing)
    assert suspect is False
    assert "that opening sample is round 2's" in note
    assert "round 1's NPU solo was skipped" in note


def test_the_recheck_can_be_turned_off(monkeypatch):
    _per, _clocks, closing = _sweep(monkeypatch, [18.0, 13.0, 17.0, 12.0],
                                    closing_recheck=False)
    assert closing == {"state": "disabled"}, "no extra leg when disabled"


# --- the closing re-check's failure paths ---------------------------------
# Only the success and disabled paths were exercised. These two are how it
# actually breaks in the field, and both used to be indistinguishable from a
# deliberate skip.

def test_a_closing_leg_that_fails_is_reported_as_failed(monkeypatch):
    # The closing measurement returns None -- a 429, a dropped connection, an
    # early EOS. Reporting that as "skipped" would hide a broken instrument.
    _per, _clocks, closing = _sweep(monkeypatch, [18.0, 13.0, 17.0, 12.0, None])
    assert closing == {"state": "failed", "engine": "NPU"}
    assert "failed" in bc.closing_note(closing)[0]


def test_no_opening_sample_means_no_comparison_and_says_so(monkeypatch):
    # Every NPU measurement skipped, so there is nothing for the closing leg to
    # be compared against. Taking the extra leg anyway would burn a minute to
    # produce a number with no partner.
    _per, _clocks, closing = _sweep(monkeypatch, [None, None, 17.0, 12.0])
    assert closing == {"state": "no-opening-sample", "engine": "NPU"}


def test_the_closing_sample_never_reaches_the_medians(monkeypatch):
    # The closing sample is a CONTROL, not data. If it ever landed in
    # per_engine["solo"] it would shift the median every ratio divides by.
    per, _clocks, _closing = _sweep(monkeypatch, [18.0, 13.0, 17.0, 12.0, 9.9])
    assert per["NPU"]["solo"] == [18.0], "closing sample must stay out of data"


# --- what the load generator's counts say about the leg ----------------------
# If nothing was served, the "contended" leg measured an idle box and the ratio
# comes out near 1.0 -- which reads as "no contention effect" rather than "no
# experiment". These notes used to be built on the claim that a 429 and a
# refused connection could not be told apart, and called every failure
# "expected backpressure". Neither was true: post_timed hands back the reason,
# and a lone sequential client never sees a 429 from a healthy single-flight
# server at all.

def test_an_all_shed_leg_is_suspect_and_says_what_a_429_means_here():
    note, suspect = bc.shed_note("NPU", shed=40, served=0)
    assert suspect is True
    assert note.startswith("SUSPECT")
    assert "shed every one of 40" in note
    assert "something ELSE held the peer" in note


def test_an_unreachable_peer_is_suspect_and_names_the_reason():
    note, suspect = bc.shed_note("NPU", shed=0, served=0, failed=12,
                                 reason="URLError: refused")
    assert suspect is True
    assert "12 FAILED request(s) (first: URLError: refused)" in note
    assert "ZERO completions" in note and "not reachable" in note
    both, _ = bc.shed_note("NPU", shed=3, served=0, failed=12, reason="x")
    assert "plus 3 shed" in both


def test_a_generator_that_did_nothing_at_all_is_suspect():
    """shed == 0 used to return "nothing to say" before served was looked at,
    so a generator whose thread died before its first request -- no load
    whatever -- published a ~1.0 ratio with no warning."""
    note, suspect = bc.shed_note("NPU", shed=0, served=0)
    assert suspect is True
    assert "completed NOTHING" in note and "No load was applied" in note


def test_a_clean_leg_says_nothing():
    assert bc.shed_note("NPU", shed=0, served=12) == (None, False)


def test_a_partly_shed_leg_is_a_note_not_called_expected():
    note, suspect = bc.shed_note("GPU", shed=40, served=7)
    assert suspect is False
    assert note.startswith("note:") and "served 7" in note
    assert "shed 40 request(s)" in note and "UNDERSTATES contention" in note
    assert "expected" not in note, (
        "a 429 to a lone sequential client is never the expected steady state")


def test_failures_beside_completions_are_not_called_backpressure():
    note, suspect = bc.shed_note("GPU", shed=0, served=7, failed=3,
                                 reason="TimeoutError: timed out")
    assert suspect is False
    assert "FAILED 3 request(s) (first: TimeoutError: timed out)" in note
    assert "not backpressure" in note
    assert "shed" not in note, "nothing was shed, so nothing says so"


@pytest.mark.parametrize("counts", [
    {"shed": 99, "served": 0},
    {"shed": 0, "served": 0},
    {"shed": 0, "served": 0, "failed": 9, "reason": "URLError: refused"},
])
def test_every_suspect_note_warns_about_the_ratio_specifically(counts):
    # The failure is not "we lost some load", it is "the number you are about
    # to publish means something else".
    note, suspect = bc.shed_note("NPU", **counts)
    assert suspect is True
    assert "1.0" in note and "'no load'" in note


# --- the command line ----------------------------------------------------------

def test_the_help_renders_and_says_what_it_has_to():
    """Rendered, because a help string is a format string and argparse only
    finds out when it expands one: --cool-floor's was %-formatted once too
    often, which left a bare "% of base" -- a ValueError from add_argument on
    3.14 and from --help before it, so main() died before parsing anything."""
    # Colour codes stripped: 3.14 can colour help, and FORCE_COLOR turns that
    # on even under capture.
    text = re.sub(r"\x1b\[[0-9;]*m", "", bc._parser().format_help())
    flat = _flat(text)
    assert "%%" not in text, "an escape reached the reader unexpanded"
    # --gpu names both launcher legs, so the port cannot be guessed wrong.
    assert "http://127.0.0.1:8124" in flat
    assert "-Leg gpu" in flat and "8080" in flat and "CPU leg" in flat
    # The gate's give-up limit, which no option exposes.
    assert "gives up after %d s and PROCEEDS" % bc.COOL_LIMIT_S in flat
    # --repeat and --timeout had no help at all.
    assert "rounds; each round takes one solo and one contended" in flat
    assert "per-request timeout for the measurements" in flat
    assert "GENIE_LOW_CHARGE_PCT" in flat and "13-20% charge" in flat


def test_every_environment_variable_the_tool_obeys_is_named():
    """The docstring said "Reads one environment variable" and --help listed
    that one, while bench_endpoint's GENIE_MIN_DECODE_STEPS -- read at ITS
    import -- floors the only measurement this tool makes. A shell that exports
    it to let bench_endpoint measure short windows moved this tool's floor
    too, under documentation saying nothing of the kind was read."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", bc._parser().format_help())
    for where, said in (("--help", text), ("the module docstring", bc.__doc__)):
        for var in ("GENIE_LOW_CHARGE_PCT", "GENIE_MIN_DECODE_STEPS"):
            assert var in said, "%s does not name %s" % (where, var)
    assert "Reads one environment variable" not in bc.__doc__
    # The default quoted in --help is the stub's, which is the real module's.
    assert "rest on (default %d)" % bc.be.MIN_DECODE_STEPS in _flat(text)


def test_the_defaults_are_the_launchers_ports_and_todays_load_shape():
    a = bc._parser().parse_args([])
    assert a.npu == "http://127.0.0.1:8123"
    # 8080 is run-llama-server.ps1's CPU leg. Defaulting --gpu to it filed a
    # CPU measurement under "GPU", and the operator relabelled it by hand.
    assert a.gpu == "http://127.0.0.1:8124"
    assert (a.load_depth, a.load_tokens) == (None, None)
    assert (a.depth, a.tokens, a.repeat, a.cool_floor) == (500, 120, 3, 92.0)


@pytest.mark.parametrize("argv,names", [
    (["--repeat", "0"], "--repeat"),
    (["--ramp", "-1"], "--ramp"),
    (["--depth", "0"], "--depth"),
    (["--tokens", "0"], "--tokens"),
    # One under bench_endpoint's decode floor: no leg could ever be accepted.
    (["--tokens", "15"], "GENIE_MIN_DECODE_STEPS"),
    (["--load-depth", "0"], "--load-depth"),
    (["--load-tokens", "-5"], "--load-tokens"),
    (["--timeout", "0"], "--timeout"),
    (["--cool-floor", "-1"], "--cool-floor"),
])
def test_a_value_that_would_waste_the_box_is_refused_first(monkeypatch, capsys,
                                                           argv, names):
    """argparse checks types, not ranges. --repeat 0 paid both pings and both
    warmups and then exited 0 with both engines "incomplete"; --ramp -1 raised
    from time.sleep after the solo leg, with the generator already started."""
    touched = []
    monkeypatch.setattr(bc, "free_physical_gb",
                        lambda: touched.append("memory") or 32.0)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: touched.append("ping"))
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", *argv])
    with pytest.raises(SystemExit) as e:
        bc.main()
    assert e.value.code == 2
    assert touched == [], "refused AFTER touching the box"
    assert names in capsys.readouterr().err


def test_the_decode_floor_is_checked_against_tokens_before_the_box_is_spent(
        monkeypatch, capsys):
    """GENIE_MIN_DECODE_STEPS=200 with the default --tokens 120 used to run the
    whole sweep: every leg printed "REFUSED: 119-step window (min 200)" without
    naming the variable, both engines ended "incomplete", and the run exited 0.
    A measurement can return fewer steps than --tokens and never more, so the
    refusal is provable before the first ping."""
    touched = []
    monkeypatch.setattr(bc, "free_physical_gb",
                        lambda: touched.append("memory") or 32.0)
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: touched.append("ping"))
    monkeypatch.setattr(bc.be, "MIN_DECODE_STEPS", 200)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py"])
    with pytest.raises(SystemExit) as e:
        bc.main()
    assert e.value.code == 2 and touched == []
    err = _flat(capsys.readouterr().err)
    assert "--tokens 120 is below the decode floor of 200 steps" in err
    assert "GENIE_MIN_DECODE_STEPS" in err
    # AT the floor is allowed: a full window is exactly enough.
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: {"model": "m"})
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5),
                     [*QUICK, "--tokens", "200"]) == 0


# --- main()'s reporting tail ----------------------------------------------
# Everything below runs through main() and was unreachable from any test, which
# is how the closing-check JSON key came to share a name with the CLI flag
# without anything noticing.

def _rec(solo, contended, shed=0, served=4, failed=0, first_failure=None,
         tokens_out=0, seconds=0.0):
    """One engine's record, as paired_sweep builds it. Ratios are paired by
    position, which is only right for fixtures whose lists line up (or whose
    contended list is empty -- hence strict=False)."""
    return {"solo": list(solo), "contended": list(contended),
            "ratios": [c / s for s, c in zip(solo, contended, strict=False)],
            "shed": shed, "served": served, "failed": failed,
            "first_failure": first_failure,
            "load_tokens_out": tokens_out, "load_seconds": seconds}


def _both_ran(npu_solo, npu_cont, gpu_solo, gpu_cont):
    return {"NPU": _rec([npu_solo], [npu_cont]),
            "GPU": _rec([gpu_solo], [gpu_cont])}


def _shed_round():
    """One engine measured fine; the other had every contended sample shed.

    This is the NPU-under-contention shape: solo samples land, contended ones
    come back as 429/529 and are skipped.
    """
    return {"GPU": _rec([18.0], [13.4]),
            "NPU": _rec([18.5], [], shed=9, served=0)}


def _run_main(monkeypatch, rounds, argv, clocks=None, closing=None, sweep=None,
              free=32.0):
    """main() with the box, both servers and (by default) the sweep stubbed.

    `sweep` replaces the default stub when a test needs the sweep to DO
    something -- record a sample, raise a gate note -- the way the real one
    would, between main() clearing its module state and draining it.
    """
    def default(engines, a, make_load):
        return (rounds, [99.0] if clocks is None else clocks,
                closing or {"state": "disabled"})

    monkeypatch.setattr(bc, "paired_sweep", sweep or default)
    monkeypatch.setattr(bc, "free_physical_gb", lambda: free)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", *argv])
    return bc.main()


QUICK = ["--repeat", "1", "--cool-floor", "0"]


def test_reporting_survives_a_fully_shed_contended_leg(monkeypatch, capsys):
    """Covers BOTH crashes: the bandwidth block's KeyError on contended[name],
    and min() over an empty ratios dict."""
    assert _run_main(monkeypatch, _shed_round(),
                     [*QUICK, "--npu-weights-gb", "2.3", "--gpu-weights-gb",
                              "2.32", "--peak-bw-gbs", "135.2"]) == 0
    out = _flat(capsys.readouterr().out)
    assert "GPU solo 18.00 contended 13.40 keeps 74.4%" in out
    # Worded by what is missing. This engine's solo sample LANDED; the line
    # used to read "incomplete (every measurement was skipped)" and then not
    # show it.
    assert ("NPU incomplete: no round produced BOTH a solo and a contended "
            "sample (solo landed 1, contended landed 0)") in out
    assert "solo samples 18.50" in out
    assert "contended samples none" in out
    assert "every measurement was skipped" not in out


def test_reporting_survives_when_no_pair_completes(monkeypatch, capsys):
    """Every ratio empty -- the empty-min() path with nothing to divide."""
    both_shed = {"GPU": _rec([18.0], [], shed=3, served=0),
                 "NPU": _rec([18.5], [], shed=9, served=0)}
    assert _run_main(monkeypatch, both_shed,
                     ["--cool-floor", "0", "--npu-weights-gb", "2.3",
                      "--gpu-weights-gb", "2.32", "--peak-bw-gbs",
                      "135.2"]) == 0
    out = capsys.readouterr().out
    assert out.count("incomplete") == 2
    assert "VERDICT" not in out, "no pair, so no verdict to give"


def test_refuses_a_loaded_box_unless_explicitly_allowed(monkeypatch, capsys):
    """The precondition is the whole reason this harness is trustworthy."""
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 2.5)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py"])
    assert bc.main() == 2
    assert "REFUSING TO RUN on a box this loaded" in capsys.readouterr().err

    assert _run_main(monkeypatch, _shed_round(),
                     ["--allow-loaded", "--cool-floor", "0"], free=2.5) == 0
    out = capsys.readouterr().out
    assert "!! LOADED BOX" in out, "said before the sweep is spent"
    assert "[LOADED BOX -- NOT A BASELINE]" in out, "and stamped on the result"


def test_refuses_when_free_memory_cannot_be_determined(monkeypatch, capsys):
    """Unknown must fail loudly rather than pass a box it could not measure --
    and say THAT, not send the operator to a threshold. The refusal used to
    read "a box this loaded ... lower --min-free-gb", which cannot help: no
    threshold passes an unknown."""
    monkeypatch.setattr(bc, "free_physical_gb", lambda: None)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py"])
    assert bc.main() == 2
    err = capsys.readouterr().err
    assert "could not be read" in err and "cannot be evaluated" in err
    assert "--min-free-gb cannot help" in err and "--allow-loaded" in err
    assert "a box this loaded" not in err


def test_an_unknown_box_may_run_only_under_the_loaded_stamp(monkeypatch,
                                                           tmp_path, capsys):
    # The only honest escape from an unreadable gate is the one that stamps
    # the output, because the box state really is unknown.
    out = tmp_path / "run.json"
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5),
                     [*QUICK, "--allow-loaded", "--json", str(out)],
                     free=None) == 0
    assert "[LOADED BOX -- NOT A BASELINE]" in capsys.readouterr().out
    body = json.loads(out.read_text())
    assert body["loaded"] is True and body["free_gb"] is None


def test_an_engine_that_is_not_answering_refuses_to_measure(monkeypatch,
                                                            capsys):
    # The likeliest real failure: one server started, the other forgotten. It
    # must refuse rather than measure an idle box -- a "contended" leg with
    # nothing contending returns ~1.0, which reads as "no contention effect"
    # rather than as "no experiment".
    swept = []
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: None)
    rc = _run_main(monkeypatch, {}, [],
                   sweep=lambda *a: swept.append(1))
    assert rc == 2, "must exit non-zero; a script keys on this"
    err = capsys.readouterr().err
    assert "NPU at http://127.0.0.1:8123 is not answering" in err
    # chat() also returns None for a server that answers with no usage block.
    assert "reports no usage" in err
    assert swept == []


def test_the_preflight_is_bounded_separately_from_the_measurement(monkeypatch,
                                                                  capsys):
    """The warmup inherited --timeout (1800 s) for a 4-token request, so a
    server that answered the ping and then wedged held the run for thirty
    minutes per engine before the first round."""
    asked = []

    def chat(base, model, prompt, max_tokens, timeout):
        asked.append((base, max_tokens, timeout))
        return {"model": "m"}

    monkeypatch.setattr(bc.be, "chat", chat)
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds, QUICK) == 0
    npu, gpu = "http://127.0.0.1:8123", "http://127.0.0.1:8124"
    assert asked == [(npu, 1, 60), (gpu, 1, 60), (npu, 4, 300), (gpu, 4, 300)]
    del asked[:]
    # A --timeout shorter than the warmup bound still wins.
    assert _run_main(monkeypatch, rounds, [*QUICK, "--timeout", "45"]) == 0
    assert [t for _b, _n, t in asked] == [60, 60, 45, 45]


def test_a_server_that_answers_the_ping_but_not_the_warmup_is_refused(
        monkeypatch, capsys):
    swept = []
    monkeypatch.setattr(
        bc.be, "chat",
        lambda base, model, prompt, max_tokens, timeout:
            {"model": "m"} if max_tokens == 1 else None)
    rc = _run_main(monkeypatch, {}, QUICK, sweep=lambda *a: swept.append(1))
    assert rc == 2 and swept == []
    err = capsys.readouterr().err
    assert "answered the ping but not the warmup" in err and "300 s" in err


def _recorded_chat(monkeypatch, refuse=lambda base, depth, max_tokens: False):
    """be.chat recording (base, prompt depth, max_tokens, timeout) per request,
    and answering None for whatever `refuse` picks -- a peer's overflow 400.
    prompt_of is made to carry its depth so the record can show which SHAPE
    each request had; the stub's returns "x" for every depth."""
    asked = []
    monkeypatch.setattr(bc.be, "prompt_of", lambda d: d)

    def chat(base, model, prompt, max_tokens, timeout):
        asked.append((base, prompt, max_tokens, timeout))
        return None if refuse(base, prompt, max_tokens) else {"model": "m"}

    monkeypatch.setattr(bc.be, "chat", chat)
    return asked


def test_a_reshaped_load_is_sent_to_each_server_before_the_sweep(monkeypatch,
                                                                 capsys):
    """The warmup only ever sent --depth, which WAS the load's shape until
    --load-depth/--load-tokens existed. With them the first request in the
    load's shape was the generator's own, inside round 1's contended leg.
    Sent as the generator sends it: the FULL token cap (a window refuses on
    prompt + cap, so a 4-token probe passes the flag's documented use, a
    shallow depth with a large cap) under the generator's own timeout."""
    npu, gpu = "http://127.0.0.1:8123", "http://127.0.0.1:8124"
    asked = _recorded_chat(monkeypatch)
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--load-depth", "32", "--load-tokens", "400"]) == 0
    assert asked == [
        (npu, "ping", 1, 60), (gpu, "ping", 1, 60),
        (npu, 500, 4, 300), (gpu, 500, 4, 300),
        # Both, because each engine is the other's peer.
        (npu, 32, 400, bc.LOAD_REQUEST_TIMEOUT_S),
        (gpu, 32, 400, bc.LOAD_REQUEST_TIMEOUT_S),
    ]
    assert "load shape on GPU: depth 32 x 400 tokens" in capsys.readouterr().out
    # One flag is enough to make the shape differ, and a --timeout below the
    # generator's cap is the timeout the generator would really use.
    del asked[:]
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--load-tokens", "400", "--timeout", "45"]) == 0
    assert asked[4:] == [(npu, 500, 400, 45), (gpu, 500, 400, 45)]
    # A load spelled out but shaped like the measurement was already proven
    # by the warmup: nothing extra is paid for.
    del asked[:]
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--load-depth", "500", "--load-tokens", "120"]) == 0
    assert len(asked) == 4


def test_a_load_shape_a_peer_cannot_serve_is_refused_before_the_sweep(
        monkeypatch, capsys):
    """`--load-depth 3000` against a GPU leg started at the docstring's own
    `-c 1024`: the ping passed, the depth-500 warmup passed, all 13 legs ran
    with every generator request answered 400, the report printed a VERDICT
    from legs measured against an idle peer, and the run exited 0."""
    gpu = "http://127.0.0.1:8124"
    swept = []
    asked = _recorded_chat(
        monkeypatch, refuse=lambda base, depth, cap: base == gpu and depth == 3000)
    monkeypatch.setattr(bc.be, "n_ctx",
                        lambda base: 1024 if base == gpu else 4096)
    rc = _run_main(monkeypatch, {}, [*QUICK, "--load-depth", "3000"],
                   sweep=lambda *a: swept.append(1))
    assert rc == 2 and swept == [], "refused BEFORE the sweep, not after it"
    # The NPU served the shape; the refusal is about the peer that could not.
    assert [(b, d) for b, d, _c, _t in asked[4:]] == [
        ("http://127.0.0.1:8123", 3000), (gpu, 3000)]
    err = _flat(capsys.readouterr().err)
    assert ("GPU at %s served the warmup but not the background load's shape"
            % gpu) in err
    assert "a 3000-token prompt x 120 tokens, 180 s allowed" in err
    assert "n_ctx=1024" in err, "the window is what the operator has to fix"
    assert "--load-depth/--load-tokens" in err


def test_the_flags_documented_use_is_what_the_load_warmup_catches(monkeypatch,
                                                                 capsys):
    """A shallow depth with a large cap, against a server that refuses on
    prompt + max_tokens over its window the way genie_server does. A probe
    that sent the load's DEPTH with a 4-token cap is served, which is why the
    warmup sends the generator's own request instead."""
    swept = []
    # The ping's prompt is a word, not a depth; only shaped requests overflow.
    _recorded_chat(monkeypatch,
                   refuse=lambda base, depth, cap: (isinstance(depth, int)
                                                    and depth + cap + 64 > 4096))
    rc = _run_main(monkeypatch, {},
                   [*QUICK, "--load-depth", "32", "--load-tokens", "4096"],
                   sweep=lambda *a: swept.append(1))
    assert rc == 2 and swept == []
    assert "a 32-token prompt x 4096 tokens" in _flat(capsys.readouterr().err)


def test_each_label_is_printed_beside_what_the_server_says_it_is(monkeypatch,
                                                                 capsys):
    """The label is hardcoded and the harness cannot tell a CPU leg from a GPU
    one, so what answered at each URL is printed beside it: the launcher
    aliases its GPU leg, and a CPU leg answering on the --gpu URL shows here
    as the wrong id."""
    ids = {"http://127.0.0.1:8123": "qwen3-4b-npu",
           "http://127.0.0.1:8124": "qwen3.5-9b-gpu"}
    monkeypatch.setattr(bc.be, "chat",
                        lambda base, *a, **k: {"model": ids[base]})
    monkeypatch.setattr(bc.be, "n_ctx",
                        lambda base: 1024 if base.endswith("8124") else None)
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5), QUICK) == 0
    out = _flat(capsys.readouterr().out)
    assert ("GPU http://127.0.0.1:8124 model=default (server reports "
            "qwen3.5-9b-gpu) n_ctx=1024") in out
    assert ("NPU http://127.0.0.1:8123 model=qwen3-4b-npu (server reports "
            "qwen3-4b-npu) n_ctx=unknown") in out


# --- the headline arithmetic -------------------------------------------------
# Every fixture above is single-sample, where a median is the identity and a
# ratio of medians equals a median of ratios. So the suite pinned the plumbing
# (states, phrases, JSON keys) and none of the numbers the tool exists to
# print: inverting the ratio, swapping median for mean, max for min, dropping
# a 100x, or dividing by the wrong baseline each left it green.

def _three_rounds():
    """Lists chosen so that every shortcut gives a different number.

    NPU   solo 10 20 40 (median 20, mean 23.3)   contended 8 10 36 (median 10,
          mean 18)   ratios .8 .5 .9 -> median .8, mean .733, and the ratio
          of the medians is .5
    GPU   solo 30 50 31 (median 31)   contended 15 20 28 (median 20)
          ratios .5 .4 .903 -> median .5
    """
    return {"NPU": _rec([10.0, 20.0, 40.0], [8.0, 10.0, 36.0]),
            "GPU": _rec([30.0, 50.0, 31.0], [15.0, 20.0, 28.0])}


def test_the_printed_figures_are_medians_and_the_keeps_figure_is_paired(
        monkeypatch, capsys, tmp_path):
    out_json = tmp_path / "run.json"
    assert _run_main(monkeypatch, _three_rounds(),
                     ["--cool-floor", "0", "--json", str(out_json)]) == 0
    out = _flat(capsys.readouterr().out)
    assert ("NPU solo 20.00 contended 10.00 keeps 80.0% (paired median of 3)"
            in out)
    assert ("GPU solo 31.00 contended 20.00 keeps 50.0% (paired median of 3)"
            in out)
    assert "solo samples 10.00, 20.00, 40.00" in out
    assert "contended samples 15.00, 20.00, 28.00" in out
    # The pair: contended medians summed, against the solo medians summed and
    # against the BEST solo median.
    assert "aggregate while both hot 30.00 t/s" in out
    assert "sum of solo rates 51.00 t/s" in out
    assert "efficiency vs additive 58.8%" in out
    assert "best single engine solo 31.00 t/s" in out
    assert "speedup vs best engine 0.97x" in out
    assert "SLOWER than the best engine alone" in out
    # Said on the output, because only "keeps" is drift-cancelled.
    assert "UNPAIRED" in out
    body = json.loads(out_json.read_text())
    assert body["solo_median"] == {"NPU": 20.0, "GPU": 31.0}
    assert body["contended_median"] == {"NPU": 10.0, "GPU": 20.0}
    assert body["paired_ratio_median"] == {"NPU": pytest.approx(0.8),
                                           "GPU": pytest.approx(0.5)}


def test_the_generators_duty_cycle_is_reported_per_engine(monkeypatch, capsys,
                                                          tmp_path):
    """tokens_out was accumulated per completion and read by nothing. It is
    what tells a decode-heavy --load-tokens run from the default in the
    artifact: tokens the peer decoded per second of contended-leg time."""
    rounds = {"NPU": _rec([18.0], [13.4], served=6, shed=1, failed=2,
                          first_failure="URLError: refused",
                          tokens_out=720, seconds=90.0),
              "GPU": _rec([18.0], [13.5], served=0, tokens_out=0, seconds=0.0)}
    out_json = tmp_path / "run.json"
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--json", str(out_json)]) == 0
    out = _flat(capsys.readouterr().out)
    assert ("peer load served 6, shed 1, failed 2; 720 tokens over 90 s = 8.0 "
            "t/s of leg time") in out
    assert "peer load served 0, shed 0, failed 0; 0 tokens over 0 s" in out
    per = json.loads(out_json.read_text())["per_engine"]
    assert per["NPU"]["load_tokens_per_s"] == 8.0
    assert per["GPU"]["load_tokens_per_s"] is None, "no leg time, no rate"
    assert per["NPU"]["first_failure"] == "URLError: refused"


def _built_load(monkeypatch, argv):
    """The Load main() hands paired_sweep, for one peer."""
    built = []

    def sweep(engines, a, make_load):
        built.append(make_load(("GPU", "http://peer", "peer-model")))
        return _both_ran(18.0, 13.4, 18.0, 13.5), [], {"state": "disabled"}

    assert _run_main(monkeypatch, None, [*QUICK, *argv], sweep=sweep) == 0
    return built[0]


def test_the_load_is_shaped_like_the_measurement_by_default(monkeypatch,
                                                           capsys):
    gen = _built_load(monkeypatch, ["--depth", "300", "--tokens", "50"])
    assert isinstance(gen, bc.Load)
    assert (gen.base, gen.model) == ("http://peer", "peer-model")
    assert (gen.depth, gen.tokens) == (300, 50), "today's behaviour, unchanged"
    assert gen.timeout == bc.LOAD_REQUEST_TIMEOUT_S
    assert "background load shaped" not in capsys.readouterr().out


def test_the_load_can_be_made_decode_heavy_and_the_run_says_so(monkeypatch,
                                                              capsys, tmp_path):
    out_json = tmp_path / "run.json"
    gen = _built_load(monkeypatch, ["--load-depth", "32", "--load-tokens", "400",
                                    "--json", str(out_json)])
    assert (gen.depth, gen.tokens) == (32, 400)
    assert ("background load shaped depth 32 x 400 tokens (the measurement is "
            "500 x 120)") in _flat(capsys.readouterr().out)
    body = json.loads(out_json.read_text())
    assert (body["load_depth"], body["load_tokens"]) == (32, 400)
    assert (body["depth"], body["tokens"]) == (500, 120)


# --- the warnings block ------------------------------------------------------

def _warnings_block(out):
    """The banner-fenced block at the end of the report, or "" if none."""
    fence = "!" * 64
    if fence not in out:
        return ""
    return out[out.index(fence):out.rindex(fence)]


def test_a_suspect_leg_reaches_the_warnings_block(monkeypatch, capsys):
    rounds = {"NPU": _rec([10.0], [9.9], shed=30, served=0),
              "GPU": _rec([20.0], [19.0], shed=5, served=5)}
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--npu-weights-gb", "2.3", "--gpu-weights-gb",
                              "2.32", "--peak-bw-gbs", "135.2"]) == 0
    out = capsys.readouterr().out
    # Not merely printed somewhere: it must survive INTO the warnings block,
    # which is what a reader skimming the tail of a long run actually sees.
    # (This used to assert `out.index("SUSPECT") < len(out)`, which cannot
    # fail, and the note is also printed BEFORE the banner.)
    block = _warnings_block(out)
    assert "WARNING: SUSPECT: while NPU was measured" in block
    # The GPU leg was partly shed: a note, printed, but not a warning.
    assert "note: while GPU was measured" in out
    assert "while GPU was measured" not in block


def test_a_suspect_closing_check_reaches_the_warnings_block(monkeypatch, capsys):
    # Same requirement as the shed note: a warning that only exists mid-output
    # is missed by a reader skimming the tail of a twenty-minute run.
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5), QUICK,
                     closing=_closing(18.0, 11.0)) == 0
    block = _warnings_block(capsys.readouterr().out)
    assert "WARNING: SUSPECT: NPU solo opened at 18.00" in block
    assert "DECAYED" in block


def test_a_monotonic_solo_decline_reaches_the_warnings_block(monkeypatch,
                                                             capsys, tmp_path):
    """warnings[] is drift_note's ONLY outlet -- unlike the closing check and
    the shed note, it is printed nowhere else -- so a broken wire here loses
    the thermal signature from the terminal AND the artifact with nothing left
    to notice it. Neither instrument that overlaps covers the loss: the
    closing re-check sees only the engine that OPENS the sweep and dies under
    --no-closing-recheck, and the gate reads the clock, not the samples.

    Sustained load drives this box to 48.9% of base, so a monotonic decline
    across every round is the EXPECTED failure of a twenty-minute run."""
    rounds = {"NPU": _rec([20.0, 18.0, 16.0], [10.0, 9.0, 8.0]),
              "GPU": _rec([30.0, 29.0, 28.0], [15.0, 14.5, 14.0])}
    out_json = tmp_path / "run.json"
    assert _run_main(monkeypatch, rounds, [*QUICK, "--json", str(out_json)]) == 0
    body = json.loads(out_json.read_text())
    drift = [w for w in body["warnings"] if "declined monotonically" in w]
    assert len(drift) == 2, body["warnings"]
    assert drift[0].startswith("NPU solo decode declined monotonically across "
                               "every round (20.00 -> 16.00, -20.0%)")
    assert drift[1].startswith("GPU solo decode declined monotonically across "
                               "every round (30.00 -> 28.00, -6.7%)")
    assert "THERMAL signature" in drift[0]
    out = capsys.readouterr().out
    assert drift[0] in _warnings_block(out), "in the terminal block too"
    assert out.count("declined monotonically") == 2, (
        "two engines, one line each: the block is the note's only outlet, so "
        "any other count means it is being printed somewhere else as well")


def test_a_clean_run_has_no_warnings_block(monkeypatch, capsys):
    # The counterpart to everything in this section: warnings on a clean run
    # are how the real ones get skipped.
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5), QUICK,
                     closing=_closing(18.0, 17.6)) == 0
    out = capsys.readouterr().out
    assert _warnings_block(out) == "" and "WARNING" not in out


@pytest.mark.parametrize("clocks,floor,warns", [
    ([93.0, 94.0, 92.0], "92", False),   # every gate PASSED at the floor
    ([97.0, 80.0, 99.0], "92", True),    # one gate gave up at 80%
    ([93.0, 94.0], "95", True),          # the same readings under a higher floor
    # Gate off: a GATED reading is not what gets judged. Production cannot
    # even produce this list (measure() reads the clock only inside the gate),
    # so this row pins the branch and nothing else -- what an ungated run IS
    # judged on is in the tests below, over the real sweep.
    ([40.0], "0", False),
    ([], "92", False),
])
def test_the_clock_warning_agrees_with_the_gate_about_what_limited_means(
        monkeypatch, capsys, clocks, floor, warns):
    """This tested round-START readings against a literal 95 while the gate
    accepted 92. This box idles at ~94%, so a run whose every sample the gate
    passed was still stamped "power- or thermally-limited" -- and from round 2
    on the round-start reading is the dip the previous contended leg left,
    which the gate then waits out, so the stamp landed on essentially every
    multi-round run."""
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5),
                     ["--repeat", "1", "--cool-floor", floor],
                     clocks=clocks) == 0
    block = _flat(_warnings_block(capsys.readouterr().out))
    assert ("power- or thermally-limited" in block) is warns
    if warns:
        assert "taken at %.1f%% of base" % min(clocks) in block
        assert "below the --cool-floor of %s%%" % floor in block


def _real_sweep_main(monkeypatch, tmp_path, argv, round_start, gate=99.0,
                     on_ac=True):
    """main() over the REAL paired_sweep and the REAL measure(), with the box
    reading `round_start`% of base at the top of each round and `gate`% to the
    cool gate. Only the servers and the generator are stubbed, so `clocks` and
    POWER_SAMPLES hold what production would put in them -- which a stubbed
    sweep handed a `clocks` list cannot show. Returns (exit code, JSON body).

    `on_ac` is the power source box_state reports each round: True, False for
    a box running on its pack, or None for a class that did not say. It is a
    parameter because every other caller here runs on AC, which is the one
    state the battery checks cannot be seen in."""
    out_json = tmp_path / "run.json"
    monkeypatch.setattr(bc.be, "box_state",
                        lambda: (on_ac, 80.0, 12.0, round_start))
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: gate)
    monkeypatch.setattr(bc, "Load", lambda *a, **k: FakeLoad())
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--ramp", "0",
                                      "--json", str(out_json), *argv])
    rc = bc.main()
    return rc, json.loads(out_json.read_text())


def test_an_ungated_run_on_a_throttled_box_still_gets_a_clock_warning(
        monkeypatch, capsys, tmp_path):
    """Re-keying the clock warning onto the GATED readings left --cool-floor 0
    with no clock judgement at all: measure() reads the clock only inside the
    gate, so `clocks` is empty by construction and a box steady at 61% of base
    wrote `warnings: []`. The round-start readings ARE the evidence on an
    ungated run -- nothing waits that reading out, the next solo sample is
    taken at it -- so they are judged, against the gate's default floor, and
    the warning says the run was ungated rather than that a gate gave up."""
    rc, body = _real_sweep_main(monkeypatch, tmp_path,
                                ["--cool-floor", "0", "--repeat", "3"], 61.0)
    assert rc == 0
    assert body["gate_clock_pct"] == [], "no gate ran, so no gated reading"
    assert [s["clock_pct"] for s in body["power_samples"]] == [61.0] * 3
    clock = [w for w in body["warnings"] if "of base" in w]
    assert len(clock) == 1, body["warnings"]
    said = clock[0]
    assert said.startswith("UNGATED run (--cool-floor 0)")
    assert "dipped to 61.0% of base, below the 92% the gate holds" in said
    assert "(round-start readings: 61, 61, 61)" in said
    assert "power- or thermally-limited" in said
    assert "gave up" not in said, "no gate ran, so none gave up"
    # And in the terminal block, which is what the operator reads.
    assert "WARNING: UNGATED run" in _warnings_block(capsys.readouterr().out)


@pytest.mark.parametrize("argv,round_start,warns", [
    (["--cool-floor", "0"], 91.9, True),
    # AT the threshold is not below it: the gate itself passes a 92.
    (["--cool-floor", "0"], 92.0, False),
    (["--cool-floor", "0"], None, False),    # unreadable: nothing to judge
    # GATED, and every gate passed at 99%: the round-start dip is the one the
    # gate waited out, which is why that reading stopped being judged. The
    # ungated fallback must not bring the old stamp back for gated runs.
    (["--cool-floor", "92"], 61.0, False),
])
def test_the_round_start_clock_is_judged_only_when_nothing_gated_it(
        monkeypatch, capsys, tmp_path, argv, round_start, warns):
    rc, body = _real_sweep_main(monkeypatch, tmp_path,
                                [*argv, "--repeat", "2"], round_start)
    assert rc == 0
    clock = [w for w in body["warnings"] if "thermally-limited" in w]
    assert bool(clock) is warns, body["warnings"]
    if warns:
        assert "UNGATED" in clock[0]
        assert "below the %.0f%%" % bc.DEFAULT_COOL_FLOOR in clock[0]


def test_the_ungated_threshold_is_the_gates_own_default():
    """One number, not two: the floor an ungated run is judged against is the
    one --cool-floor defaults to, so the help, the gate and the warning cannot
    drift apart the way the literal 95 drifted from the gate's 92."""
    assert bc._parser().parse_args([]).cool_floor == bc.DEFAULT_COOL_FLOOR
    flat = _flat(re.sub(r"\x1b\[[0-9;]*m", "", bc._parser().format_help()))
    assert ("judged instead, against %.0f%% of base" % bc.DEFAULT_COOL_FLOOR
            in flat)


# --- the power SOURCE, judged when no gate judged it -------------------------
# The gate's ABORT is the only thing that STOPS a sweep for being on battery,
# and it is easy for it never to run. The module header promises "nothing
# measured on battery is worth keeping -- so the gate ABORTS and the sweep
# STOPS there", and under that promise a run taken entirely on battery
# published `warnings: []` and exit 0 in three ordinary shapes. on_ac was
# recorded per round the whole time and read by nothing but the JSON.

@pytest.mark.parametrize("argv,gate", [
    # No gate runs at all, so the abort cannot fire.
    (["--cool-floor", "0"], 99.0),
    # Gated, but every reading is ABOVE the floor: power_limited_note returns
    # before power_reading() is called, so the source is never consulted.
    (["--cool-floor", "92"], 99.0),
    # Gated, but the counter is unreadable: wait_for_cool returns on its first
    # None poll, before power_limited_note.
    (["--cool-floor", "92"], None),
])
def test_a_run_finished_on_battery_says_so_even_when_no_gate_caught_it(
        monkeypatch, capsys, tmp_path, argv, gate):
    rc, body = _real_sweep_main(monkeypatch, tmp_path,
                                [*argv, "--repeat", "2"], 96.0, gate=gate,
                                on_ac=False)
    assert [s["on_ac"] for s in body["power_samples"]] == [False, False]
    said = [w for w in body["warnings"] if "ON BATTERY" in w]
    assert len(said) == 1, body["warnings"]
    assert said[0].startswith("this run was ON BATTERY for 2 of 2 round(s) "
                              "(round 1, 2) and no gate stopped it")
    assert "nothing measured on battery is worth keeping" in said[0].lower()
    # And in the terminal block, which is what the operator reads.
    assert "WARNING: this run was ON BATTERY" in _warnings_block(
        capsys.readouterr().out)
    # Exit code unchanged: only the GATE stops a sweep, and no gate did.
    assert rc == 0


@pytest.mark.parametrize("on_ac", [True, None])
def test_a_source_that_did_not_say_battery_is_not_warned_about_as_one(
        monkeypatch, tmp_path, on_ac):
    """None is "the class did not say", not "on battery" -- the same
    three-way the round-start print was fixed for. A run on AC is the control:
    nothing else about this box earns a warning, so the block stays empty."""
    rc, body = _real_sweep_main(monkeypatch, tmp_path,
                                ["--cool-floor", "0", "--repeat", "2"], 96.0,
                                on_ac=on_ac)
    assert [s["on_ac"] for s in body["power_samples"]] == [on_ac, on_ac]
    assert (rc, body["warnings"]) == (0, [])


def _with_power_samples(samples):
    """A sweep stub that records per-round samples the way the real one does:
    after main() has cleared the list, before it drains it."""
    def sweep(engines, a, make_load):
        for i, (pct, watts) in enumerate(samples, 1):
            bc.POWER_SAMPLES.append({"round": i, "on_ac": True,
                                     "charge_pct": pct, "charge_w": watts,
                                     "clock_pct": 96.0})
        return _both_ran(18.0, 13.4, 18.0, 13.5), [99.0], {"state": "disabled"}
    return sweep


def test_a_low_pack_warns_in_the_run_summary(monkeypatch, capsys):
    assert _run_main(monkeypatch, None, QUICK,
                     sweep=_with_power_samples([(18.0, 31.0)])) == 0
    block = _warnings_block(capsys.readouterr().out)
    assert "pack was at 18%" in block
    assert "halves prefill" in block


def test_the_drift_warning_reports_the_span_it_fired_on(monkeypatch, capsys):
    """A warning must not contradict its own trigger.

    The condition is on the EXTREMES (max - min >= 20), so reporting first and
    last readings instead printed "moved 60% -> 62%" above a warning raised
    because the run spanned 40-62%. Non-monotonic trajectories are the norm here
    -- the pack can dip under load and recover -- so the two differ routinely.
    """
    # Dips to 40 and recovers: endpoints are 2 points apart, span is 22.
    samples = [(60.0, 30.0), (40.0, 30.0), (62.0, 30.0)]
    assert _run_main(monkeypatch, None, QUICK,
                     sweep=_with_power_samples(samples)) == 0
    block = _warnings_block(capsys.readouterr().out)
    assert "spanned 40%-62%" in block, "reported endpoints instead of the span"
    assert "opened 60%, closed 62%" in block, "endpoints are still worth showing"
    assert "pack was at" not in block, "40% is not a low pack"


def test_a_low_pack_that_also_moved_gets_both_warnings(monkeypatch, capsys):
    """The movement warning was an `elif` of the low-pack one, so it was
    suppressed exactly when a run had both problems -- a deeply discharged
    pack that charged 40 points during the sweep. They are different defects:
    one is about the level, the other about two power regimes being averaged."""
    samples = [(18.0, 30.0), (40.0, 30.0), (60.0, 30.0)]
    assert _run_main(monkeypatch, None, QUICK,
                     sweep=_with_power_samples(samples)) == 0
    block = _warnings_block(capsys.readouterr().out)
    assert "pack was at 18% during this run (range 18-60%)" in block
    assert "spanned 18%-60%" in block


def test_charge_draw_variation_is_surfaced_not_left_dead(monkeypatch, capsys):
    """charge_w was recorded and never read, which is dead weight in the record.

    It needs a frame rather than a raw list: measured on this box, draw ranged
    28-41.9 W at a roughly constant charge level while read-to-read noise was
    ~1 W. A reader who anchors a shed percentage to one sample of that is biased
    by which sample they happened to pick.
    """
    samples = [(70.0, 28.0), (70.0, 41.9), (70.0, 33.0)]
    assert _run_main(monkeypatch, None, QUICK,
                     sweep=_with_power_samples(samples)) == 0
    block = _warnings_block(capsys.readouterr().out)
    assert "28.0-41.9 W" in block
    assert "not the workload" in block, "must say what the variation IS"


def test_a_steady_draw_says_nothing(monkeypatch, capsys):
    # The counterpart: a stable controller must not produce a warning, or the
    # real one gets skipped.
    samples = [(70.0, 30.0)] * 3
    assert _run_main(monkeypatch, None, QUICK,
                     sweep=_with_power_samples(samples)) == 0
    assert "charge draw ranged" not in capsys.readouterr().out


def test_the_not_a_bandwidth_problem_warning_fires(monkeypatch, capsys):
    # The most consequential inference this tool draws, and the one whose
    # earlier conclusion had to be retracted from MULTI_ENGINE.md. Both halves
    # of the condition have to hold: heavy loss AND low bus utilisation.
    rounds = _both_ran(18.0, 6.9, 18.0, 7.3)
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--npu-weights-gb", "0.3", "--gpu-weights-gb",
                              "0.3", "--peak-bw-gbs", "135.2"]) == 0
    out = _flat(capsys.readouterr().out)
    assert "bottleneck is NOT the memory bus" in out
    assert "shared power budget" in out, "must name what to suspect instead"
    assert "NPU and GPU lost >20% throughput (NPU kept 38%; GPU kept 41%)" in out
    # decode rate x weight bytes, per engine and summed.
    assert "NPU 5.4 GB/s solo 2.1 GB/s contended (0.30 GB of weights)" in out
    assert ("combined demand 10.8 GB/s (if additive) 4.3 GB/s (actual, both "
            "hot)") in out
    assert "actual combined is 3% of peak" in out


def test_the_warning_names_the_engine_that_lost_not_engines(monkeypatch,
                                                            capsys):
    # The condition is on the WORST engine. "engines lost >20%" printed when
    # one lost 62% and the other kept 94%.
    rounds = _both_ran(18.0, 6.9, 18.0, 17.0)
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--npu-weights-gb", "0.3", "--gpu-weights-gb",
                              "0.3", "--peak-bw-gbs", "135.2"]) == 0
    block = _flat(_warnings_block(capsys.readouterr().out))
    assert "WARNING: NPU lost >20% throughput (NPU kept 38%)" in block
    assert "GPU kept" not in block and "engines lost" not in block


def test_heavy_loss_near_the_bus_ceiling_is_not_blamed_on_something_else(
        monkeypatch, capsys):
    # The other half of the condition. With demand near peak, losing throughput
    # is ordinary bus contention and the warning must stay silent -- firing
    # here sends the reader hunting a power budget that is not the cause.
    rounds = _both_ran(18.0, 6.9, 18.0, 7.3)
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--npu-weights-gb", "6.0", "--gpu-weights-gb",
                              "6.0", "--peak-bw-gbs", "135.2"]) == 0
    assert "bottleneck is NOT" not in capsys.readouterr().out


# The three flags _validate does NOT range-check. Pinned AS THEY ARE: what an
# operator typo produces today is recorded, so adding the check is a change
# with a failing test to update rather than a silent one.

def test_the_weight_and_peak_flags_are_not_range_checked(monkeypatch):
    """Every other numeric flag goes through _validate's ap.error before the
    box is touched. These three are checked nowhere, so a sign typo is
    accepted and the run proceeds -- held here against the refusal list so the
    two cannot silently disagree about which flags are guarded."""
    ap = bc._parser()
    a = ap.parse_args(["--npu-weights-gb", "-2.3", "--gpu-weights-gb", "0",
                       "--peak-bw-gbs", "-135.2"])
    bc._validate(a, ap)                  # ap.error would SystemExit here
    assert (a.npu_weights_gb, a.gpu_weights_gb, a.peak_bw_gbs) == (-2.3, 0.0,
                                                                   -135.2)
    # The contrast, on the same call: a sibling flag given the same kind of
    # value is refused before anything is spent.
    with pytest.raises(SystemExit):
        bc._validate(ap.parse_args(["--cool-floor", "-1"]), ap)


def test_a_negative_weight_prints_negative_bandwidth_and_still_exits_zero(
        monkeypatch, capsys):
    """--npu-weights-gb -2.3 is a single mistyped minus, and it turns on the
    tool's most consequential inference -- the one whose earlier conclusion
    had to be retracted from MULTI_ENGINE.md -- with full confidence and exit
    0: the negative total trivially satisfies `tot_cont < 0.5 * peak`, so the
    bottleneck-is-NOT-the-memory-bus warning fires. The printed rows are
    visibly absurd (a negative GB/s, and an 'actual' ABOVE its own additive
    ideal), which is the only thing standing between this and a believed
    number."""
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--npu-weights-gb", "-2.3", "--gpu-weights-gb",
                              "2.32", "--peak-bw-gbs", "135.2"]) == 0
    out = _flat(capsys.readouterr().out)
    assert "NPU -41.4 GB/s solo -30.8 GB/s contended (-2.30 GB of weights)" in out
    assert ("combined demand 0.4 GB/s (if additive) 0.5 GB/s (actual, both "
            "hot)") in out, "the 'actual' is ABOVE the additive ideal"
    assert "bus peak 135.2 GB/s -- actual combined is 0% of peak" in out
    assert "WARNING: NPU and GPU lost >20% throughput" in out
    assert "bottleneck is NOT the memory bus" in out


def test_a_zero_weight_drops_its_engine_with_no_line_saying_why(monkeypatch,
                                                                capsys):
    """The silent half, and the dangerous one: 0 is falsy, so that engine gets
    no row, `len(bw) == 2` fails and the combined demand, the bus-peak line
    and the warning all vanish -- a shorter report that looks like a complete
    one. Both weights 0 skips the block outright, because its guard is
    `if a.npu_weights_gb or a.gpu_weights_gb`."""
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--npu-weights-gb", "0", "--gpu-weights-gb",
                              "2.32", "--peak-bw-gbs", "135.2"]) == 0
    out = _flat(capsys.readouterr().out)
    assert "DERIVED BANDWIDTH" in out
    assert "GPU 41.8 GB/s solo" in out
    assert out.count("GB of weights") == 1, (
        "one row, and no line anywhere saying the other engine was dropped")
    assert "combined demand" not in out and "bus peak" not in out
    assert "bottleneck is NOT" not in out
    capsys.readouterr()
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5),
                     [*QUICK, "--npu-weights-gb", "0", "--gpu-weights-gb", "0",
                              "--peak-bw-gbs", "135.2"]) == 0
    assert "DERIVED BANDWIDTH" not in capsys.readouterr().out


def test_a_net_loss_verdict_points_at_the_poll_flag_first(monkeypatch, capsys):
    # This verdict was WRONG for a whole revision of the docs: a 0.78x net loss,
    # measured against a poll:true bundle. The advice telling the next reader to
    # check that flag before believing it is the reason the branch exists.
    #
    # The 0.78x itself was then refuted too -- a controlled re-run found both
    # poll settings a GAIN (1.70x vs 1.26x over the best single engine), so the
    # flag costs about a quarter of the win rather than reversing the sign.
    # This asserts the pointer, not the retired number: the branch has to send
    # the reader to the flag, and must not re-assert a figure that did not
    # reproduce.
    rounds = _both_ran(18.0, 6.9, 18.0, 7.3)
    assert _run_main(monkeypatch, rounds, QUICK) == 0
    out = capsys.readouterr().out
    assert "SLOWER than the best engine alone" in out
    assert "QnnHtp/poll" in out
    assert "concurrency win" in out
    assert "0.78x" not in out, "refuted figure must not come back"


def test_a_gain_verdict_does_not_mention_poll(monkeypatch, capsys):
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds, QUICK) == 0
    out = capsys.readouterr().out
    assert "two hot engines beat the best single engine by 1.49x" in out
    assert "QnnHtp/poll" not in out
    # No reference figure in the tool's output: it quoted 1.45x for the
    # poll:false configuration while the comment twelve lines above it cited
    # 1.70x for the same one, and a number printed by a tool outlives its
    # retraction in the docs. The docs are where the figures live.
    assert "1.45x" not in out and "1.70x" not in out
    assert "docs/MULTI_ENGINE.md" in out


# --- the JSON artifact ----------------------------------------------------
# Never written or inspected by any test, which is exactly how a key ended up
# sharing its name with the CLI flag of the same meaning.

def test_the_json_carries_the_closing_check_under_its_own_key(monkeypatch,
                                                              tmp_path):
    out = tmp_path / "run.json"
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds, [*QUICK, "--json", str(out)],
                     closing=_closing(18.0, 17.6), clocks=[97.0, 96.0]) == 0
    body = json.loads(out.read_text())
    # `closing_recheck` is the CLI FLAG's dest. A result stored under that name
    # reads as the flag's value, so the artifact uses a distinct key.
    assert "closing_recheck" not in body
    assert body["closing_check"]["state"] == "ok"
    assert body["closing_check"]["retained"] == pytest.approx(17.6 / 18.0)
    for key in ("solo_median", "contended_median", "paired_ratio_median",
                "per_engine", "bandwidth", "warnings"):
        assert key in body, key
    # The readings each SOLO sample was gated on. Renamed from cpu_clock_pct
    # when its contents changed: that key held round-start readings taken
    # BEFORE the gate, which now live in power_samples[].clock_pct.
    assert body["gate_clock_pct"] == [97.0, 96.0]
    assert "cpu_clock_pct" not in body


def test_the_json_records_what_the_run_was_taken_under(monkeypatch, tmp_path):
    """Two artifacts are comparable only at the same window, endpoints and
    settings, and the docstring's own advice is to change the GPU leg's -c
    between runs to clear the memory gate. None of this used to be written --
    the docstring claimed n_ctx was -- and the label "GPU" could not be audited
    afterwards, which is how a CPU measurement filed under it went unnoticed."""
    monkeypatch.setattr(bc.be, "chat", lambda base, *a, **k: {
        "model": "qwen3.5-9b-gpu" if base.endswith(":9001") else None})
    monkeypatch.setattr(bc.be, "n_ctx",
                        lambda base: 1024 if base.endswith(":9001") else None)
    out = tmp_path / "run.json"
    assert _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5),
                     ["--repeat", "2", "--gpu", "http://h:9001", "--gpu-model",
                      "g", "--depth", "300", "--tokens", "50", "--ramp", "2.5",
                      "--timeout", "900", "--cool-floor", "90", "--closing-tol",
                      "7.5", "--json", str(out)]) == 0
    body = json.loads(out.read_text())
    assert body["engines"] == {
        "NPU": {"base": "http://127.0.0.1:8123", "model": "qwen3-4b-npu",
                "served_model": None, "n_ctx": None},
        "GPU": {"base": "http://h:9001", "model": "g",
                "served_model": "qwen3.5-9b-gpu", "n_ctx": 1024}}
    settings = {k: body[k] for k in ("depth", "tokens", "repeat", "ramp",
                                     "timeout", "cool_floor", "closing_tol",
                                     "load_depth", "load_tokens", "loaded",
                                     "free_gb")}
    assert settings == {"depth": 300, "tokens": 50, "repeat": 2, "ramp": 2.5,
                        "timeout": 900.0, "cool_floor": 90.0,
                        "closing_tol": 7.5, "load_depth": 300,
                        "load_tokens": 50, "loaded": False, "free_gb": 32.0}


def test_power_samples_reach_the_json_as_magnitudes(monkeypatch, tmp_path):
    """The artifact must carry the quantity, not a verdict about it.

    Four legs across two sessions showed every boolean cut point mis-sorting the
    legs nearest it -- a <=1 W test scored 0/4 on legs that shed 92% and 97%,
    and a 25%-of-opening test scored 2/4. A reader can apply a threshold to a
    recorded magnitude; nobody can recover a reading the harness threw away.

    This test asserted `power_samples == []` for a long time: its seed was
    cleared by main() and its stubbed sweep recorded nothing, so no test ever
    put a magnitude INTO the JSON, and replacing the list with booleans, or
    with dicts missing charge_w, left the suite green.
    """
    # A previous run's leftovers, which main() must clear rather than publish.
    bc.POWER_SAMPLES.append({"round": 9, "on_ac": False, "charge_pct": 1.0,
                             "charge_w": 0.0, "clock_pct": 1.0})
    out = tmp_path / "run.json"
    assert _run_main(monkeypatch, None, [*QUICK, "--json", str(out)],
                     sweep=_with_power_samples([(62.0, 30.5),
                                                (64.0, 1.1)])) == 0
    assert json.loads(out.read_text())["power_samples"] == [
        {"round": 1, "on_ac": True, "charge_pct": 62.0, "charge_w": 30.5,
         "clock_pct": 96.0},
        {"round": 2, "on_ac": True, "charge_pct": 64.0, "charge_w": 1.1,
         "clock_pct": 96.0}]


def test_the_per_sample_box_trajectory_reaches_the_json(monkeypatch, tmp_path):
    """bench_endpoint records the box state after every accepted measurement --
    a PowerShell launch per sample -- and this tool never read it, though it
    holds the post-sample clock the contention JSON lacked."""
    sample = {"label": "d500", "on_ac": True, "charge_pct": 80.0,
              "charge_w": 9.5, "clock_pct": 71.2}
    bc.be.BOX_SAMPLES.append({"label": "left over from a previous run"})

    def sweep(engines, a, make_load):
        assert bc.be.BOX_SAMPLES == [], "main() must clear it before the sweep"
        bc.be.BOX_SAMPLES.append(sample)
        return _both_ran(18.0, 13.4, 18.0, 13.5), [99.0], {"state": "disabled"}

    out = tmp_path / "run.json"
    assert _run_main(monkeypatch, None, [*QUICK, "--json", str(out)],
                     sweep=sweep) == 0
    assert json.loads(out.read_text())["box_samples"] == [sample]


def test_gate_note_reaches_the_json_not_only_the_terminal(monkeypatch, tmp_path,
                                                         capsys):
    """A warning that exists only in stdout is absent from the artifact a
    consumer reads -- the record-vs-reality drift this harness keeps finding.

    End to end, because the pieces were only ever tested apart: this used to
    call wait_for_cool and look at the module-level list, while every test that
    went through main() stubbed the sweep and ran ungated -- so `warnings` was
    only ever built from an empty GATE_NOTES, and moving the clear() below the
    sweep or dropping the drain would have passed the suite. Here the REAL
    sweep runs the REAL gate, whose counter cannot be read, before each of its
    three gated legs (NPU solo, GPU solo, closing)."""
    monkeypatch.setattr(bc, "cpu_performance_pct", lambda: None)
    monkeypatch.setattr(bc.be, "measure_decode", lambda *a: 10.0)
    monkeypatch.setattr(bc, "Load", lambda *a: FakeLoad())
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    bc.GATE_NOTES.append("left over from a previous run")
    out = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "1",
                                      "--ramp", "0", "--json", str(out)])
    assert bc.main() == 0
    body = json.loads(out.read_text())
    ungated = [w for w in body["warnings"] if "UNGATED" in w]
    assert len(ungated) == 1, "one entry for the three legs, not three"
    assert "raised 3 times during this run" in ungated[0]
    assert not any("left over" in w for w in body["warnings"])
    assert body["gate_clock_pct"] == [], "no gate passed on a reading"
    assert body["closing_check"]["state"] == "ok"
    assert "WARNING: gate skipped" in _warnings_block(capsys.readouterr().out)


def test_a_gate_abort_exits_2_and_the_artifact_says_why(monkeypatch, tmp_path,
                                                        capsys):
    """'ABORTING THE GATE' used to abort only the WAIT: the sample was taken,
    entered solo_median and the ratios, and the run exited 0."""
    measured = _battery_after(monkeypatch, good_gates=2)
    monkeypatch.setattr(bc, "Load", lambda *a: FakeLoad())
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    out = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "3",
                                      "--ramp", "0", "--json", str(out)])
    assert bc.main() == 2, "a script keys on this"
    assert len(measured) == 4, "round 1 only"
    text = capsys.readouterr().out
    assert "[SWEEP STOPPED BY THE GATE]" in text
    assert "NPU solo 10.00 contended 10.00 keeps 100.0%" in _flat(text), (
        "the rounds completed before the abort are still reported")
    body = json.loads(out.read_text())
    assert body["closing_check"] == {"state": "gate-aborted", "engine": "NPU",
                                     "round": 2}
    assert any(w.startswith("gate ABORTED: ") for w in body["warnings"])
    assert body["per_engine"]["NPU"]["solo"] == [10.0]


def test_a_sweep_aborted_at_its_first_gate_still_runs_the_shed_notes(
        monkeypatch, tmp_path, capsys):
    """Pinned AS IT IS, not as it should be -- read the assertions as a
    record, not as an endorsement.

    The gate aborts at round 1's FIRST solo, so no leg was measured and no
    generator was ever started. main() runs shed_note for every engine
    unconditionally, so both engines -- `solo: []`, `contended: []`, served 0
    -- get the note for a generator that "completed NOTHING ... while NPU was
    measured", a claim that is false on its face for an engine that was not.

    Recorded rather than suppressed because nothing here is a wrong NUMBER:
    the true line (the gate ABORT) is FIRST in the block, the header still
    says [SWEEP STOPPED BY THE GATE] and the exit code is still 2, so this is
    diagnostic noise on the run that matters most rather than a corrupted
    measurement. Skipping the note for an engine with no samples is the
    obvious change; this test is what would go red when someone makes it, so
    it is made deliberately.
    """
    measured = _battery_after(monkeypatch, good_gates=0)
    monkeypatch.setattr(bc, "Load", lambda *a: FakeLoad())
    monkeypatch.setattr(bc, "free_physical_gb", lambda: 32.0)
    out = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["bench_contention.py", "--repeat", "3",
                                      "--ramp", "0", "--json", str(out)])
    assert bc.main() == 2
    assert measured == [], "the abort came before the first measurement"
    body = json.loads(out.read_text())
    assert body["per_engine"]["NPU"] == body["per_engine"]["GPU"], (
        "both engines are equally empty")
    assert body["per_engine"]["NPU"]["solo"] == []
    assert body["per_engine"]["NPU"]["contended"] == []
    assert body["warnings"][0].startswith("gate ABORTED: "), (
        "the one true line leads the block")
    claimed = [w for w in body["warnings"] if "completed NOTHING" in w]
    assert len(claimed) == 2
    assert claimed[0].startswith("SUSPECT: while NPU was measured")
    assert claimed[1].startswith("SUSPECT: while GPU was measured")
    block = _warnings_block(capsys.readouterr().out)
    assert "WARNING: SUSPECT: while GPU was measured" in block
    # The new on-battery check adds nothing here: box_state could not read the
    # source, so POWER_SAMPLES is empty and the abort is the only witness.
    assert body["power_samples"] == []
    assert not any("ON BATTERY for" in w for w in body["warnings"])


def test_an_existing_json_is_refused_before_the_box_is_spent(monkeypatch,
                                                            tmp_path, capsys):
    # A run costs 20+ minutes of a shared box and some of these numbers have
    # proved unreproducible, so losing one to a re-run is the wrong trade --
    # and so is learning that the file will not be written only AFTER the
    # sweep, which is when this refusal used to come.
    out = tmp_path / "run.json"
    out.write_text('{"previous": "result"}')
    touched = []
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: touched.append("ping"))
    rc = _run_main(monkeypatch, {}, [*QUICK, "--json", str(out)],
                   sweep=lambda *a: touched.append("sweep"))
    assert rc == 2
    assert touched == [], "refused after touching the servers"
    assert json.loads(out.read_text()) == {"previous": "result"}
    err = capsys.readouterr().err
    assert "REFUSING TO START" in err and "--force" in err


def test_a_json_that_appears_during_the_run_goes_to_a_fallback_name(
        monkeypatch, tmp_path, capsys):
    """Another session writing the same path while this one swept.

    The rule used to be "NOT writing ..., exit 0", which threw this run's
    artifact away: power_samples, box_samples, gate_clock_pct and the
    per-engine counts exist ONLY in the file, so 20+ minutes of a shared box
    went with it, and a wrapper reading the exit code could not tell that the
    file it named holds somebody else's run. So: the named file is still never
    overwritten, the record lands beside it under a timestamp, one sentence
    names both, and the exit code is non-zero.
    """
    out = tmp_path / "run.json"

    def sweep(engines, a, make_load):
        out.write_text('{"someone": "else"}')
        return _both_ran(18.0, 13.4, 18.0, 13.5), [99.0], {"state": "disabled"}

    assert _run_main(monkeypatch, None, [*QUICK, "--json", str(out)],
                     sweep=sweep) == 3
    assert json.loads(out.read_text()) == {"someone": "else"}
    beside = [p for p in tmp_path.iterdir() if p.name != "run.json"]
    assert len(beside) == 1, "one fallback, beside the path that was asked for"
    assert beside[0].name.startswith("run.") and beside[0].suffix == ".json"
    assert json.loads(beside[0].read_text())["solo_median"]["NPU"] == 18.0
    text = capsys.readouterr().out
    assert "appeared during the run" in text and "NOT overwritten" in text
    assert beside[0].name in text, "the sentence names where it went"
    assert "VERDICT" in text, "the results are still reported in full"


def test_a_missing_json_directory_is_refused_before_the_box_is_spent(
        monkeypatch, tmp_path, capsys):
    """`--json results\\c.json` with no results\\ -- an operator typing a
    subdirectory that is not there. This used to run the pings, the warmups
    and every leg, print the whole CONTENTION block, and then come out of
    open() as a bare FileNotFoundError with the record still in memory."""
    out = tmp_path / "no-such-dir" / "c.json"
    touched = []
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: touched.append("ping"))
    rc = _run_main(monkeypatch, {}, [*QUICK, "--json", str(out)],
                   sweep=lambda *a: touched.append("sweep"))
    assert rc == 2
    assert touched == [], "refused after touching the servers"
    err = capsys.readouterr().err
    assert "REFUSING TO START" in err and "not a directory" in err


def test_an_unwritable_json_directory_is_refused_at_startup(monkeypatch,
                                                            tmp_path, capsys):
    """The directory exists and still cannot be written -- a denying ACL, a
    full volume. Probed with a real file create, because os.access(W_OK) on
    Windows answers True for every directory. The probe is stubbed here to
    raise what such a directory raises; leaving it out is what made the
    failure arrive at the END of the run instead."""
    out = tmp_path / "c.json"
    touched = []
    monkeypatch.setattr(bc.be, "chat", lambda *a, **k: touched.append("ping"))

    def refuse(**kw):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(bc.tempfile, "mkstemp", refuse)
    rc = _run_main(monkeypatch, {}, [*QUICK, "--json", str(out)],
                   sweep=lambda *a: touched.append("sweep"))
    assert rc == 2
    assert touched == []
    err = capsys.readouterr().err
    assert "REFUSING TO START" in err and "cannot be written" in err
    assert "Permission denied" in err, "the reason the OS gave"


def test_the_startup_probe_leaves_the_json_directory_as_it_found_it(
        monkeypatch, tmp_path):
    """The write test is a file created and removed. A probe that leaked one
    would litter the operator's results directory once per run."""
    out = tmp_path / "c.json"
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds, [*QUICK, "--json", str(out)]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_a_write_that_fails_at_the_end_falls_back_rather_than_traceback(
        monkeypatch, tmp_path, capsys):
    """The directory passed at startup and the write failed anyway -- it went
    away under the run, the volume filled, another process holds the file
    open. open() used to raise straight out of main() after the box had been
    spent, so the exit code said "crashed" and the record was gone."""
    out = tmp_path / "c.json"
    real_dump = bc._dump_json
    tried = []

    def dump(path, record):
        tried.append(path)
        if len(tried) == 1:
            raise OSError(28, "No space left on device")
        real_dump(path, record)

    monkeypatch.setattr(bc, "_dump_json", dump)
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds, [*QUICK, "--json", str(out)]) == 3
    assert tried[0] == str(out) and tried[1] != str(out)
    assert json.loads(open(tried[1]).read())["solo_median"]["NPU"] == 18.0
    text = capsys.readouterr().out
    assert "could not be written" in text and "No space left" in text
    assert tried[1] in text, "the sentence names where it went"
    assert "VERDICT" in text, "the results are still reported in full"


def test_a_fallback_that_fails_too_says_so_and_still_exits_non_zero(
        monkeypatch, tmp_path, capsys):
    """Nothing left to write to. The one outcome where the terminal IS the
    artifact, and it has to say that rather than print "wrote" or exit 0."""
    out = tmp_path / "c.json"

    def dump(path, record):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(bc, "_dump_json", dump)
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds, [*QUICK, "--json", str(out)]) == 3
    assert not out.exists()
    text = capsys.readouterr().out
    assert "the fallback" in text and "only in the terminal above" in text
    assert "wrote " not in text


def test_a_gate_abort_outranks_a_fallback_write(monkeypatch, tmp_path, capsys):
    """Both facts are true and the exit code can only carry one. A stopped
    sweep is the more serious of the two, and the sentence above the exit says
    where the file went either way."""
    out = tmp_path / "c.json"
    real_dump = bc._dump_json

    def dump(path, record):
        if path == str(out):
            raise OSError(28, "No space left on device")
        real_dump(path, record)

    monkeypatch.setattr(bc, "_dump_json", dump)
    rc = _run_main(monkeypatch, _both_ran(18.0, 13.4, 18.0, 13.5),
                   [*QUICK, "--json", str(out)],
                   closing={"state": "gate-aborted", "engine": "NPU",
                            "round": 2})
    assert rc == 2, "the gate abort, not the fallback"
    text = capsys.readouterr().out
    assert "could not be written" in text, "and the file is still accounted for"


def test_force_overwrites_deliberately(monkeypatch, tmp_path):
    out = tmp_path / "run.json"
    out.write_text('{"previous": "result"}')
    rounds = _both_ran(18.0, 13.4, 18.0, 13.5)
    assert _run_main(monkeypatch, rounds,
                     [*QUICK, "--json", str(out), "--force"]) == 0
    body = json.loads(out.read_text())
    assert "previous" not in body and body["solo_median"]["NPU"] == 18.0
