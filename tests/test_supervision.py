"""Tests for wedge detection and the escalation that follows it.

A wedged HTP cannot be reproduced in a test -- it needs the device, and it is
by definition the state where the device stops answering. That is the argument
FOR these tests rather than against them: the failure is unreproducible, so the
only thing that can be checked ahead of time is the decision made about it. If
the escalation logic is wrong it will be wrong at the exact moment nobody is
able to debug it.

The distinction under test throughout is slow vs stopped. A generation running
for ten minutes is not a wedge -- 2000 tokens at the slowest measured 3.3 t/s
is exactly that -- and killing it would be a self-inflicted outage. What
separates the two is not elapsed time but whether tokens are still arriving.

Clocks are passed in rather than read, so nothing here sleeps.
"""

import pytest


@pytest.fixture
def H(gs):
    """Tight thresholds; the units are arbitrary since time is injected."""
    return gs.EngineHealth(first_token_timeout=100, stall_timeout=50,
                           grace=30, fail_threshold=3)


# --- idle and healthy -----------------------------------------------------

def test_an_idle_engine_is_ok(H):
    assert H.assess(1000)[0] == "ok"


def test_a_generation_in_flight_is_ok_before_the_first_token(H):
    H.begin(1000)
    assert H.assess(1050)[0] == "ok", "prefill at depth is legitimately slow"


def test_a_long_generation_that_keeps_producing_is_never_wedged(H):
    # The case that must not be killed: ten minutes of real work. Every step is
    # well inside stall_timeout, so no amount of TOTAL elapsed time matters.
    H.begin(0)
    now = 0
    for _ in range(600):
        now += 1
        H.progress(now)
        assert H.assess(now)[0] == "ok"
    assert now == 600, "a ten-minute generation stayed healthy throughout"


def test_progress_resets_the_stall_clock(H):
    H.begin(0)
    H.progress(10)
    assert H.assess(55)[0] == "ok", "45s since the last token, limit 50"
    H.progress(56)
    assert H.assess(100)[0] == "ok", "the new token restarted the clock"


# --- stalled --------------------------------------------------------------

def test_no_first_token_past_the_limit_is_a_stall(H):
    H.begin(1000)
    state, detail = H.assess(1101)
    assert state == "stalled"
    assert "first token" in detail


def test_a_gap_between_tokens_past_the_limit_is_a_stall(H):
    H.begin(0)
    H.progress(10)
    state, detail = H.assess(61)
    assert state == "stalled"
    assert "further token" in detail
    assert "1 token" in detail, "how far it got is what makes it diagnosable"


def test_the_same_gap_means_different_things_before_and_after_a_token(H):
    # Prefill legitimately takes far longer than one decode step, so the two
    # limits are not interchangeable: 60s of silence is normal while waiting on
    # prefill (limit 100) and a stall once tokens have started (limit 50).
    H.begin(0)
    assert H.assess(60)[0] == "ok", "60s with no first token yet is fine"

    H.progress(1)
    assert H.assess(61)[0] == "stalled", "60s BETWEEN tokens is not fine"


# --- wedged ---------------------------------------------------------------

def test_a_stall_becomes_wedged_only_after_the_abort_had_time_to_work(H):
    H.begin(0)
    H.note_stall_signalled(101)
    assert H.assess(120)[0] == "stalled", "still inside the grace period"
    state, detail = H.assess(132)
    assert state == "wedged"
    assert "did not take" in detail


def test_a_stall_never_escalates_if_no_abort_was_attempted(H):
    # Escalation is a consequence of the cheap remedy failing, not of time
    # passing. Exiting without having tried the abort would be a self-inflicted
    # outage on a device that might have recovered.
    H.begin(0)
    assert H.assess(10_000)[0] == "stalled"


def test_recovery_clears_everything(H):
    H.begin(0)
    H.note_stall_signalled(101)
    assert H.assess(200)[0] == "wedged"
    H.end(ok=True, now=201)
    assert H.assess(202)[0] == "ok", "a completed generation is not a wedge"


# --- hard failures --------------------------------------------------------

def test_consecutive_failures_are_reported_as_failing(H):
    for i in range(3):
        H.begin(i)
        H.end(ok=False, now=i)
    state, detail = H.assess(10)
    assert state == "failing"
    assert "3 consecutive" in detail


def test_one_success_clears_the_failure_streak(H):
    for i in range(2):
        H.begin(i)
        H.end(ok=False, now=i)
    H.begin(5)
    H.end(ok=True, now=5)
    assert H.assess(10)[0] == "ok"


def test_a_couple_of_failures_are_not_an_outage(H):
    # A 400-shaped rejection or a context overflow is an ordinary event; only a
    # RUN of them says the engine itself has stopped working.
    H.begin(0)
    H.end(ok=False, now=1)
    assert H.assess(2)[0] == "ok"


# --- what /health publishes -----------------------------------------------

def test_snapshot_carries_enough_to_diagnose_without_the_logs(H):
    H.begin(0)
    H.progress(1)
    H.progress(2)
    snap = H.snapshot(3)
    assert snap["state"] == "ok"
    assert snap["generating"] is True
    assert snap["tokens_in_flight"] == 2


def test_snapshot_reports_the_wedge(H):
    H.begin(0)
    H.note_stall_signalled(101)
    snap = H.snapshot(200)
    assert snap["state"] == "wedged" and snap["detail"]


# --- the endpoint ---------------------------------------------------------

def _health(gs):
    import json

    from conftest import Wire
    h = object.__new__(gs.Handler)
    h.path = "/health"
    h.wfile = Wire()
    sent = {}
    h.send_response = lambda c: sent.setdefault("code", c)
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    h.do_GET()
    return sent["code"], json.loads(h.wfile.text())


def test_health_is_200_when_the_engine_is_fine(gs):
    code, body = _health(gs)
    assert code == 200 and body["status"] == "ok"


def test_health_503s_when_the_engine_cannot_serve(gs):
    # The whole point. The old handler returned 200 unconditionally, so it
    # could not fail -- a supervisor reading it would have been told everything
    # was fine for as long as the wedge lasted.
    gs.HEALTH.begin(0)
    gs.HEALTH.note_stall_signalled(1)
    import time as _t
    real, _t.time = _t.time, lambda: 10_000
    try:
        code, body = _health(gs)
    finally:
        _t.time = real
    assert code == 503
    assert body["status"] == "wedged"
    assert body["detail"], "503 without a reason is not actionable"


def test_health_never_touches_the_engine(gs):
    """What lets health be reported DURING a wedge at all.

    The stuck thread holds the engine lock, so a handler that reached for the
    engine -- for the lock, or for anything guarded by it -- would hang exactly
    when it is needed. Rather than assert that indirectly with a held lock,
    this makes ANY access to the engine an immediate error.
    """
    class Untouchable:
        def __getattribute__(self, name):
            raise AssertionError(
                "/health touched the engine (.%s); during a wedge that is the "
                "one object it must not reach for" % name)

    gs.ENGINE = Untouchable()
    code, body = _health(gs)
    assert code == 200 and body["model"]


# --- the watchdog's escalation --------------------------------------------

class FakeEngine:
    def __init__(self, raises=False):
        self.aborts = 0
        self.raises = raises
        self.lock = None

    def signal_abort(self):
        self.aborts += 1
        if self.raises:
            raise RuntimeError("driver refused the signal")


def test_the_watchdog_signals_an_abort_on_a_stall(gs, capsys):
    eng = FakeEngine()
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=10_000,
                        fail_threshold=3)
    h.begin(0)
    gs.watchdog(eng, h, interval=0, iterations=1)
    assert eng.aborts == 1
    assert "STALL" in capsys.readouterr().out


def test_the_watchdog_exits_when_the_abort_did_not_take(gs, capsys):
    eng = FakeEngine()
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=-1,
                        fail_threshold=3)
    h.begin(0)
    h.note_stall_signalled(0)
    seen = {}
    gs.watchdog(eng, h, interval=0, iterations=1,
                on_wedge=lambda d: seen.setdefault("detail", d))
    assert seen, "a wedge must escalate, not be logged and ignored"
    out = capsys.readouterr().out
    assert "WEDGED" in out
    assert "supervisor" in out, "the operator needs to know what happens next"


def test_a_driver_that_refuses_the_abort_does_not_kill_the_watchdog(gs, capsys):
    # If the signal itself throws, the watchdog must survive to escalate --
    # losing it here would leave the wedge undetected forever.
    eng = FakeEngine(raises=True)
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=10_000,
                        fail_threshold=3)
    h.begin(0)
    gs.watchdog(eng, h, interval=0, iterations=1)
    assert "abort signal failed" in capsys.readouterr().out


def test_the_watchdog_leaves_a_healthy_engine_alone(gs, capsys):
    eng = FakeEngine()
    h = gs.EngineHealth()
    gs.watchdog(eng, h, interval=0, iterations=3)
    assert eng.aborts == 0
    assert capsys.readouterr().out == ""


def test_the_watchdog_reports_a_failing_engine_without_exiting(gs, capsys):
    # Repeated hard errors are worth surfacing, but the engine is answering --
    # restarting on them would turn a bad bundle into a crash loop.
    eng = FakeEngine()
    h = gs.EngineHealth(fail_threshold=2)
    for i in range(2):
        h.begin(i)
        h.end(ok=False, now=i)
    gs.watchdog(eng, h, interval=0, iterations=1,
                on_wedge=lambda d: pytest.fail("must not escalate on failures"))
    assert "UNHEALTHY" in capsys.readouterr().out
    assert eng.aborts == 0


def test_the_token_count_does_not_outlive_the_generation(H):
    # Observed live: /health reported tokens_in_flight=54 with generating=false,
    # describing work that had already finished. A diagnostic field that
    # survives the state it describes is the drift this endpoint exists to end.
    H.begin(0)
    H.progress(1)
    H.progress(2)
    H.end(ok=True, now=3)
    snap = H.snapshot(4)
    assert snap["generating"] is False
    assert snap["tokens_in_flight"] == 0


def test_the_stall_message_reads_as_english(H):
    # It goes in front of an operator mid-incident; "no another token" does not.
    H.begin(0)
    H.progress(1)
    _state, detail = H.assess(100)
    assert "further token" in detail
    assert "another token" not in detail


# --- internal work is supervised but not counted as served -----------------
# Summarising evicted turns is a real generation on the same device and can
# wedge it exactly as a client request can, so it keeps full stall and failure
# supervision. What it is not is traffic anyone asked for.

def test_an_internal_call_is_not_counted_as_a_generation(H):
    H.begin(0)
    H.end(ok=True, now=1, counted=False)
    assert H.snapshot(2)["generations"] == 0, \
        "/health would report more work served than any client requested"


def test_an_internal_call_still_clears_the_in_flight_state(H):
    # Not counting it must not mean not closing it out -- a `started` left set
    # would make the next assess() read the finished call as a stall.
    H.begin(0)
    H.progress(1)
    H.end(ok=True, now=2, counted=False)
    snap = H.snapshot(10_000)
    assert snap["state"] == "ok" and snap["generating"] is False
    assert snap["tokens_in_flight"] == 0


def test_a_failing_internal_call_still_counts_against_the_engine(H):
    # The failure streak is about the DEVICE, not about who asked. A summariser
    # failing three times running is the engine returning errors.
    for i in range(3):
        H.begin(i)
        H.end(ok=False, now=i, counted=False)
    assert H.assess(10)[0] == "failing"
    assert H.snapshot(10)["generations"] == 0


def test_client_traffic_is_still_counted(H):
    H.begin(0)
    H.end(ok=True, now=1)
    assert H.snapshot(2)["generations"] == 1


def test_summarisation_marks_itself_internal(gs):
    from conftest import StubEngine, convo
    gs._CONTEXT_SIZE = 500
    gs.ENGINE = StubEngine(chunks=["- a note"])
    gs.build_windowed(convo(20), max_tokens=64)
    assert [c["internal"] for c in gs.ENGINE.calls] == [True]


# --- the steady states announce once ---------------------------------------
# `failing` persists until a generation succeeds and `wedged` persists forever
# under GENIE_WEDGE_EXIT=0, so printing them every interval reprinted a
# multi-line stanza every five seconds for as long as the outage lasted --
# burying the first occurrence, which carries the original cause, under
# thousands of identical copies of itself.

class ScriptedHealth:
    """Returns a fixed sequence of states, so nothing here depends on a clock."""

    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def assess(self, now):
        state = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return state, "scripted %s" % state

    def note_stall_signalled(self, now):
        pass


def test_a_persistent_wedge_is_announced_once(gs, capsys):
    eng = FakeEngine()
    gs.watchdog(eng, ScriptedHealth(["wedged"] * 5), interval=0, iterations=5,
                on_wedge=lambda d: None)
    out = capsys.readouterr().out
    assert out.count("WEDGED") == 1, \
        "reprinted the stanza %d times -- the first one carries the cause" % \
        out.count("WEDGED")


def test_a_persistent_failing_engine_is_announced_once(gs, capsys):
    eng = FakeEngine()
    gs.watchdog(eng, ScriptedHealth(["failing"] * 5), interval=0, iterations=5,
                on_wedge=lambda d: pytest.fail("must not escalate on failures"))
    assert capsys.readouterr().out.count("UNHEALTHY") == 1


def test_recovery_re_arms_the_announcement(gs, capsys):
    # A second, genuinely new episode is new information and must be printed.
    eng = FakeEngine()
    gs.watchdog(eng, ScriptedHealth(["failing", "failing", "ok", "failing"]),
                interval=0, iterations=4,
                on_wedge=lambda d: pytest.fail("must not escalate"))
    assert capsys.readouterr().out.count("UNHEALTHY") == 2


def test_a_stall_is_not_latched_because_each_line_marks_a_retry(gs, capsys):
    # Deliberately unlike the two above: every STALL line corresponds to a
    # fresh abort signal, and the state is bounded by the grace period rather
    # than open-ended.
    eng = FakeEngine()
    gs.watchdog(eng, ScriptedHealth(["stalled"] * 3), interval=0, iterations=3)
    assert capsys.readouterr().out.count("STALL") == 3
    assert eng.aborts == 3, "each announcement is a real re-signal"
