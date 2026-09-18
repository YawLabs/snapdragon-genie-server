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

from conftest import request


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
    # Not `assert now == 600` -- the loop increments it exactly 600 times, so
    # that could never fail. What CAN: every token must have been counted.
    assert H.snapshot(now)["tokens_in_flight"] == 600, (
        "a ten-minute generation stayed healthy throughout")


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


def test_the_grace_clock_runs_from_the_first_abort_not_the_latest(H):
    # The watchdog re-signals a stall on every pass, so note_stall_signalled
    # is called again and again through one episode. Keeping the LATEST
    # timestamp would restart the grace period each time: `now -
    # stall_signalled_at` would never grow past `grace`, assess() would answer
    # "stalled" forever, and the wedged engine would answer 503 for as long as
    # anyone left it up rather than being replaced.
    H.begin(0)
    H.note_stall_signalled(101)
    H.note_stall_signalled(120)
    H.note_stall_signalled(131)
    state, detail = H.assess(132)
    assert state == "wedged", "the clock still runs from the abort at 101"
    assert "31s ago" in detail, "and says how long the FIRST abort has had"


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
    H.end(ok=True)
    assert H.assess(202)[0] == "ok", "a completed generation is not a wedge"


# --- hard failures --------------------------------------------------------

def test_consecutive_failures_are_reported_as_failing(H):
    for i in range(3):
        H.begin(i)
        H.end(ok=False)
    state, detail = H.assess(10)
    assert state == "failing"
    assert "3 consecutive" in detail


def test_one_success_clears_the_failure_streak(H):
    for i in range(2):
        H.begin(i)
        H.end(ok=False)
    H.begin(5)
    H.end(ok=True)
    assert H.assess(10)[0] == "ok"


def test_a_couple_of_failures_are_not_an_outage(H):
    # A 400-shaped rejection or a context overflow is an ordinary event; only a
    # RUN of them says the engine itself has stopped working.
    H.begin(0)
    H.end(ok=False)
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

def _health(gs, handler):
    """GET /health -> (status code, body), through conftest's request().

    This used to build the socketless Handler and its Wire by hand, then to
    set the path and capture the code by hand -- a near-copy of what test_api
    carried four of. request() is the one place that does it now.
    """
    code, body, _h = request(gs, handler, "GET", "/health")
    return code, body


def test_health_is_200_when_the_engine_is_fine(gs, handler):
    code, body = _health(gs, handler)
    assert code == 200 and body["status"] == "ok"


def test_health_503s_when_the_engine_cannot_serve(gs, handler):
    # The whole point. The old handler returned 200 unconditionally, so it
    # could not fail -- a supervisor reading it would have been told everything
    # was fine for as long as the wedge lasted.
    gs.HEALTH.begin(0)
    gs.HEALTH.note_stall_signalled(1)
    import time as _t
    real, _t.time = _t.time, lambda: 10_000
    try:
        code, body = _health(gs, handler)
    finally:
        _t.time = real
    assert code == 503
    assert body["status"] == "wedged"
    assert body["detail"], "503 without a reason is not actionable"


def test_health_never_touches_the_engine(gs, handler):
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
    code, body = _health(gs, handler)
    assert code == 200 and body["model"]


# --- the watchdog's escalation --------------------------------------------

class FakeEngine:
    """Records the aborts the watchdog asks for. The signature is the real
    one on purpose: a looser fake swallowed a keyword the engine did not take
    as "abort signal failed", and the test for THAT line passed for the wrong
    reason."""

    def __init__(self, raises=False, sent=None):
        self.aborts = 0
        self.scopes = []         # the (any_turn, stalled) each abort was asked with
        self.raises = raises
        self.sent = sent         # what signal_abort answers
        self.lock = None

    def signal_abort(self, any_turn=False, stalled=False):
        self.aborts += 1
        self.scopes.append((any_turn, stalled))
        if self.raises:
            raise RuntimeError("driver refused the signal")
        return self.sent


def test_the_watchdog_signals_an_abort_on_a_stall(gs, capsys):
    eng = FakeEngine()
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=10_000,
                        fail_threshold=3)
    h.begin(0)
    gs.watchdog(eng, h, interval=0, iterations=1)
    assert eng.aborts == 1
    # A bare signal_abort() means "the stream THIS thread is consuming", and
    # the watchdog consumes none -- so it has to ask for whoever holds the
    # dialog, or a stall is never signalled at all. (test_engine_kv runs the
    # same thing against the real engine.) And it says the abort is FOR a
    # stall, which is what keeps the aborted turn from being booked as an
    # ordinary finish.
    assert eng.scopes == [(True, True)]
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


class _Ticks:
    """A clock read off a script, for the watchdog's two uses of `time`.

    Stands in for the module's `time`, not for the code under test: watchdog
    calls time.sleep(interval) and time.time() and nothing else, so a real
    multi-pass escalation can be walked through without the test sleeping.
    """

    def __init__(self, ticks):
        self.ticks = list(ticks)
        self.slept = []

    def time(self):
        return self.ticks.pop(0) if len(self.ticks) > 1 else self.ticks[0]

    def sleep(self, seconds):
        self.slept.append(seconds)


def test_the_watchdog_escalates_although_it_re_signals_on_every_pass(
        gs, capsys, monkeypatch):
    # The same guard as above, through the real watchdog: the escalation is
    # the ONLY production path to "wedged", and it takes many passes to get
    # there (WEDGE_GRACE_S is 60 against a 5s interval, so about twelve).
    # Every one of them re-signals the stall and re-notes it, and if that
    # restarted the grace clock the loop below would signal forever and the
    # supervisor would wait for an exit that never comes.
    eng = FakeEngine(sent=gs.ABORT_SIGNALLED)
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=30,
                        fail_threshold=3)
    h.begin(0)
    clock = _Ticks([10, 20, 30, 40, 50])
    monkeypatch.setattr(gs, "time", clock)
    seen = {}
    gs.watchdog(eng, h, interval=5, iterations=5,
                on_wedge=lambda d: seen.setdefault("detail", d))
    assert eng.aborts == 4, "it re-signalled the stall on every pass but the last"
    assert clock.slept == [5] * 5, "and waited its interval between them"
    assert seen, "and still escalated: the grace clock ran from the FIRST abort"
    assert "did not take" in seen["detail"]
    assert "WEDGED" in capsys.readouterr().out


def test_a_driver_that_refuses_the_abort_does_not_kill_the_watchdog(gs, capsys):
    # If the signal itself throws, the watchdog must survive to escalate --
    # losing it here would leave the wedge undetected forever.
    eng = FakeEngine(raises=True)
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=10_000,
                        fail_threshold=3)
    h.begin(0)
    gs.watchdog(eng, h, interval=0, iterations=1)
    assert "abort signal failed: driver refused the signal" in capsys.readouterr().out


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
        h.end(ok=False)
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
    H.end(ok=True)
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
    H.end(ok=True, counted=False)
    assert H.snapshot(2)["generations"] == 0, \
        "/health would report more work served than any client requested"


def test_an_internal_call_still_clears_the_in_flight_state(H):
    # Not counting it must not mean not closing it out -- a `started` left set
    # would make the next assess() read the finished call as a stall.
    H.begin(0)
    H.progress(1)
    H.end(ok=True, counted=False)
    snap = H.snapshot(10_000)
    assert snap["state"] == "ok" and snap["generating"] is False
    assert snap["tokens_in_flight"] == 0


def test_a_failing_internal_call_still_counts_against_the_engine(H):
    # The failure streak is about the DEVICE, not about who asked. A summariser
    # failing three times running is the engine returning errors.
    for i in range(3):
        H.begin(i)
        H.end(ok=False, counted=False)
    assert H.assess(10)[0] == "failing"
    assert H.snapshot(10)["generations"] == 0


def test_client_traffic_is_still_counted(H):
    H.begin(0)
    H.end(ok=True)
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

    def note_stall_signalled(self, now, native=True):
        self.native = native


def test_a_persistent_wedge_is_announced_once(gs, capsys):
    eng = FakeEngine()
    health = ScriptedHealth(["wedged"] * 5)
    gs.watchdog(eng, health, interval=0, iterations=5, on_wedge=lambda d: None)
    # All five looks happened: "printed once" is only a finding about the
    # latch if the loop was still running to print it again.
    assert health.calls == 5
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


def test_a_stall_with_nothing_to_abort_says_so(gs, capsys):
    # signal_abort is scoped to a generation in flight and reports when there
    # was none. A stall inside a host-side call (the tokenizer) has nothing an
    # ABORT can reach, and "signalling abort" alone would claim a signal that
    # was never sent.
    health = ScriptedHealth(["stalled"])
    gs.watchdog(FakeEngine(sent=False), health, interval=0, iterations=1)
    out = capsys.readouterr().out
    assert "STALL" in out and "nothing in flight to abort" in out
    assert health.native is False, "and the grace clock is told none went out"


def test_a_stall_before_the_query_says_no_abort_was_sent(gs, capsys):
    # HEALTH.begin runs at lock acquisition, so a hang in one of a turn's
    # PRE-QUERY native calls (the reset, the stop sequences, the sampler, the
    # token cap) is a stall -- and the engine sends no native signal outside
    # GenieDialog_query. The turn is flagged only, signal_abort said True
    # either way, and the log read "signalling abort" and then "an abort was
    # signalled 60s ago and did not take" about a driver that was never asked.
    health = ScriptedHealth(["stalled"])
    gs.watchdog(FakeEngine(sent=gs.ABORT_FLAGGED), health, interval=0, iterations=1)
    out = capsys.readouterr().out
    assert "no native ABORT was sent" in out and "BEFORE its query" in out
    assert "nothing in flight" not in out, "a turn IS in flight; it is not querying"
    assert health.native is False


def test_only_a_signalled_stall_says_the_abort_was_signalled(gs, capsys):
    # The ordinary case: the turn was inside its query and the ABORT went, so
    # "signalling abort" is simply true and the clock is a native one. It is
    # said HERE, after the attempt, and not in the STALL line -- that line is
    # printed before the attempt (signal_abort can block on a wedged driver)
    # and so cannot know yet whether anything was sent. The two cases above
    # are the proof that matters: neither of them may carry the phrase.
    health = ScriptedHealth(["stalled"])
    gs.watchdog(FakeEngine(sent=gs.ABORT_SIGNALLED), health, interval=0, iterations=1)
    out = capsys.readouterr().out
    assert out.count("[genie]") == 2, out
    assert "STALL" in out and "signalling abort" in out
    assert health.native is True


@pytest.mark.parametrize("sent", [False, "flagged"])
def test_a_stall_nobody_signalled_does_not_claim_an_abort_went(gs, capsys, sent):
    # The STALL line used to end "-- signalling abort" unconditionally, so the
    # log said a signal had gone out and then, on the next line, that none
    # had. The correction stays; the claim it corrects is gone.
    sent = gs.ABORT_FLAGGED if sent == "flagged" else False
    gs.watchdog(FakeEngine(sent=sent), ScriptedHealth(["stalled"]),
                interval=0, iterations=1)
    out = capsys.readouterr().out
    assert "STALL" in out
    assert "signalling abort" not in out, out


def test_a_refused_signal_still_starts_the_grace_clock(gs, capsys):
    # The note moved to AFTER the attempt so it can record what the attempt
    # was. An attempt that raised must still start the clock, or a driver that
    # throws on every signal is never escalated.
    eng = FakeEngine(raises=True)
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=5,
                        fail_threshold=3)
    h.begin(0)
    gs.watchdog(eng, h, interval=0, iterations=1)
    assert h.stall_signalled_at is not None


def test_a_wedge_nobody_could_signal_does_not_say_the_abort_was_ignored(H):
    # "an abort was signalled Ns ago and did not take" is the sentence an
    # operator files against the driver. When no GenieDialog_signal went out
    # -- the stall is outside the query -- it must not be said.
    H.begin(0)
    H.note_stall_signalled(101, native=False)
    state, detail = H.assess(132)
    assert state == "wedged", "the escalation itself is unchanged"
    assert "did not take" not in detail
    assert "No ABORT was sent" in detail
    # ...and a fresh generation starts from the default again.
    H.end(ok=False)
    H.begin(200)
    H.note_stall_signalled(301)
    assert "did not take" in H.assess(400)[1]


# --- GENIE_WEDGE_EXIT=0: stay up, keep reporting 503 ------------------------

def test_with_exit_disabled_the_watchdog_keeps_watching(gs, capsys, monkeypatch):
    # The stay-up mode the banner advertises. Every other wedge test injects
    # on_wedge, so the production path through _exit_for_supervisor returning
    # None was never executed. os._exit is patched to RAISE for the duration:
    # if _exit_for_supervisor ever stopped honouring WEDGE_EXIT, this fails
    # visibly instead of taking the test runner down with exit 75.
    monkeypatch.setattr(gs.os, "_exit", lambda code: (_ for _ in ()).throw(
        AssertionError("os._exit(%d) with GENIE_WEDGE_EXIT=0" % code)))
    gs.WEDGE_EXIT = False
    health = ScriptedHealth(["wedged"] * 3)
    result = gs.watchdog(FakeEngine(), health, interval=0, iterations=3)
    assert result is None
    assert health.calls == 3, "declining the exit must not end the loop"
    out = capsys.readouterr().out
    assert out.count("WEDGED") == 1, "announced once, then kept watching"
    assert "GENIE_WEDGE_EXIT=0" in out


def test_with_exit_enabled_the_supervisor_exit_is_what_runs(gs, monkeypatch):
    # The other branch of the same function, with the exit intercepted.
    codes = []
    monkeypatch.setattr(gs.os, "_exit", codes.append)
    gs.WEDGE_EXIT = True
    gs._exit_for_supervisor("detail")
    assert codes == [gs.EXIT_WEDGED]


def test_with_exit_disabled_the_wedge_stanza_does_not_promise_a_restart(gs, capsys):
    # The stanza is announced ONCE, so it is the whole record of the event --
    # and under GENIE_WEDGE_EXIT=0 it said "Exiting 75 so a supervisor
    # restarts a clean process" about a process that then stayed up forever.
    # Whoever sets the variable sets it because nothing supervises this
    # process, so that sentence told the one person who has to act that
    # somebody else would.
    gs.WEDGE_EXIT = False
    gs.watchdog(FakeEngine(), ScriptedHealth(["wedged"]), interval=0,
                iterations=1, on_wedge=lambda d: None)
    out = capsys.readouterr().out
    assert "WEDGED" in out
    assert "Exiting %d" % gs.EXIT_WEDGED not in out, out
    assert "supervisor restarts" not in out
    assert "restart it by hand" in out and "GENIE_WEDGE_EXIT=0" in out
    assert "503" in out, "and what it answers meanwhile"


def test_with_exit_enabled_the_wedge_stanza_still_names_the_exit(gs, capsys):
    # The other half: the default configuration's wording is unchanged.
    gs.WEDGE_EXIT = True
    gs.watchdog(FakeEngine(), ScriptedHealth(["wedged"]), interval=0,
                iterations=1, on_wedge=lambda d: None)
    out = capsys.readouterr().out
    assert "Exiting %d" % gs.EXIT_WEDGED in out
    assert "supervisor restarts a clean process" in out
    assert "restart it by hand" not in out


@pytest.mark.parametrize("sent", [False, "flagged"])
def test_with_exit_disabled_a_stall_says_no_exit_is_coming_either(gs, capsys, sent):
    # The two stall lines that end on "what can clear it". Both promised "only
    # the exit below can clear it" whatever GENIE_WEDGE_EXIT said, and under 0
    # there is no exit below: the stall line and the WEDGED stanza were the
    # only two announcements of the outage and both pointed at a restart that
    # was not coming.
    sent = gs.ABORT_FLAGGED if sent == "flagged" else False
    gs.WEDGE_EXIT = False
    gs.watchdog(FakeEngine(sent=sent), ScriptedHealth(["stalled"]),
                interval=0, iterations=1)
    out = capsys.readouterr().out
    assert "only the exit below" not in out, out
    assert "GENIE_WEDGE_EXIT=0" in out and "restart it by hand" in out
    assert "503" in out, "and what it answers meanwhile"


# --- host-side native calls are supervised on their own clock -------------
# The tokenizer encode that sizes every request is a call into the same
# driver and can wedge the same way, but it is not a generation: it must not
# show as generating, must not be counted, and must not touch the failure
# streak (a successful encode clearing three failed generations would report
# a failing engine as recovered).

def test_a_native_call_within_the_limit_is_ok_and_not_generating(H):
    H.native_begin(0, "GenieTokenizer_encode")
    assert H.assess(50)[0] == "ok"
    snap = H.snapshot(50)
    assert snap["generating"] is False
    assert snap["generations"] == 0


def test_a_native_call_past_the_first_token_limit_is_a_stall(H):
    H.native_begin(0, "GenieTokenizer_encode")
    state, detail = H.assess(101)
    assert state == "stalled"
    assert "GenieTokenizer_encode" in detail, "name the call that is stuck"


def test_a_native_stall_escalates_to_wedged_like_any_other(H):
    H.native_begin(0, "GenieTokenizer_encode")
    H.note_stall_signalled(101)
    assert H.assess(120)[0] == "stalled", "still inside the grace period"
    assert H.assess(132)[0] == "wedged"


def test_a_native_call_returning_clears_the_stall(H):
    H.native_begin(0, "GenieTokenizer_encode")
    assert H.assess(101)[0] == "stalled"
    H.native_end()
    assert H.assess(102)[0] == "ok"


def test_a_native_call_does_not_touch_the_failure_streak(H):
    for i in range(3):
        H.begin(i)
        H.end(ok=False)
    assert H.assess(10)[0] == "failing"
    H.native_begin(11, "GenieTokenizer_encode")
    H.native_end()
    assert H.assess(12)[0] == "failing", "an encode succeeding is not a recovery"
    assert H.snapshot(12)["consecutive_failures"] == 3


def test_a_failing_engine_still_reports_failing_during_a_native_call(H):
    for i in range(3):
        H.begin(i)
        H.end(ok=False)
    H.native_begin(10, "GenieTokenizer_encode")
    assert H.assess(11)[0] == "failing"


def test_the_engine_supervises_its_tokenizer_calls(gs):
    # count_tokens is the one caller; the clock must open and close around it.
    seen = {}

    class Lib:
        def GenieTokenizer_encode(self, tok, data, acb, tokptr, ntok):
            seen["during"] = gs.HEALTH.native_since is not None
            ntok._obj.value = 3
            return 0
    eng = gs.GenieEngine(Lib(), object(), tokenizer=object())
    assert eng.count_tokens("abc") == 3
    assert seen == {"during": True}
    assert gs.HEALTH.native_since is None, "closed out afterwards"


# --- end() takes no timestamp ---------------------------------------------

def test_end_does_not_accept_a_timestamp_it_would_ignore(H):
    # It used to take `now`, every caller passed one, and nothing read it. A
    # parameter that does nothing is a claim the signature makes and the body
    # does not keep.
    H.begin(0)
    with pytest.raises(TypeError):
        H.end(ok=True, now=1)
