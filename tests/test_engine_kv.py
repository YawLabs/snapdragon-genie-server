"""Tests for the KV-reuse decision -- the engine's one silent-wrong-answer path
-- and for the one code path every generation now runs through.

Everything else in this server fails loudly: a bad request 400s, a dead socket
aborts, an oversized prompt is refused by name. `_plan` is the exception. It
decides whether to CONTINUE the dialog's resident KV or reset it, and a wrong
"continue" produces a fluent answer drawn from a conversation that never
happened -- no error, no log line, nothing to notice until someone reads a
reply that references a turn they did not send.

The template half of this invariant is already pinned in test_prompt.py (a
second render must extend the first byte-for-byte). This file pins the half
that acts on it, which was untested: the engine can be handed a correct
prefix and still get the decision wrong.

The second half of this file drives the REAL query() / query_stream() through
`_run_query` against a FakeLib that answers the handful of Genie calls the
engine makes. That is not the C API being mocked for shape -- conftest.py says
why that would be wrong, and the fake accepts anything: where a payload is
asserted it is to see WHICH one the engine chose to send (clear or set,
override or baseline), in the shape the server's own comments record from
hardware, not to bless the shape. What is under test is the Python-side state
machine around the calls: what gets recorded in the KV, what the health
accounting books, whose generation an abort lands on, and what finish reason
a capped generation reports. Those decisions used to live
twice (query and query_stream's worker were copies) and could drift with the
whole suite green, because the only engine any handler test ever saw was
conftest's StubEngine, which records its kwargs and acts on none of them.

Device-free. `_plan` needs only `lib.GenieDialog_reset`, `_commit` is pure
arithmetic over strings, and the query path needs the FakeLib below -- all of
it runs against the REAL methods; nothing here is a stand-in for the code
under test.
"""

import gc
import json
import threading
import weakref

import pytest

ABORT = 0x01           # GENIE_DIALOG_ACTION_ABORT


class FakeLib:
    """The Genie calls the engine makes, recorded; SUCCESS unless told otherwise.

    `chunks` are handed to the query callback one per call, as bytes -- that
    is how Genie delivers them, and the engine's decode is part of what is
    under test. `on_query` runs INSIDE GenieDialog_query, which is how a test
    stands in for the handler thread calling signal_abort mid-generation.
    """

    def __init__(self, chunks=(), status=0, ntok=None):
        self.resets = 0
        self.calls = []             # every Genie call, in order, by name
        self.stop_payloads = []     # decoded setStopSequence payloads
        self.max_tokens = []        # every cap set on the dialog
        self.signals = []           # GenieDialog_signal actions
        self.sent = []              # bytes handed to GenieDialog_query
        self.chunks = list(chunks)
        self.status = status        # what GenieDialog_query returns
        self.stop_status = 0        # what setStopSequence returns
        self.reset_status = 0       # what GenieDialog_reset returns
        self.max_status = 0         # what setMaxNumTokens returns
        self.ntok = ntok            # GenieTokenizer_encode's answer; None = fail
        self.sampler_ok = False     # getSampler fails unless a test opts in
        self.sampler_payloads = []  # decoded GenieSamplerConfig payloads
        self.freed_configs = 0
        self.freed = []             # the handle each GenieDialog_free was given
        self.on_query = None
        self.on_reset = None
        self.on_stop = None
        self.queries = 0

    def GenieDialog_reset(self, dialog):
        self.resets += 1
        self.calls.append("reset")
        if self.on_reset:
            self.on_reset()
        return self.reset_status

    def GenieDialog_setStopSequence(self, dialog, payload):
        self.calls.append("setStop")
        self.stop_payloads.append(json.loads(payload.decode("utf-8")))
        if self.on_stop:
            self.on_stop()
        return self.stop_status

    def GenieDialog_setMaxNumTokens(self, dialog, n):
        self.calls.append("setMax")
        self.max_tokens.append(n.value)
        return self.max_status

    def GenieDialog_getSampler(self, dialog, out):
        # apply_sampler bails on a failed getSampler, which is the default
        # here: the apply is inert on QAIRT 2.45 anyway (its docstring). The
        # sampler tests opt in to see what WOULD be sent.
        self.calls.append("getSampler")
        return 0 if self.sampler_ok else -1

    def GenieSamplerConfig_createFromJson(self, payload, out):
        self.sampler_payloads.append(json.loads(payload.decode("utf-8")))
        return 0

    def GenieSampler_applyConfig(self, sampler, handle):
        self.calls.append("applySampler")
        return 0

    def GenieSamplerConfig_free(self, handle):
        self.freed_configs += 1
        return 0

    def GenieDialog_query(self, dialog, data, code, cb, udata):
        self.calls.append("query")
        self.queries += 1
        self.sent.append(data)
        for c in self.chunks:
            # str for readability, bytes where the split matters -- and None,
            # which is what ctypes hands the callback for a NULL const char*.
            cb(c.encode("utf-8") if isinstance(c, str) else c, 2, None)
        if self.on_query:
            self.on_query()
        return self.status

    def GenieDialog_signal(self, dialog, action):
        self.signals.append(action)
        return 0

    def GenieTokenizer_encode(self, tok, data, acb, tokptr, ntok_out):
        self.calls.append("encode")
        if self.ntok is None:
            return -1
        ntok_out._obj.value = self.ntok
        return 0

    def GenieDialog_free(self, dialog):
        self.calls.append("free")
        self.freed.append(dialog)
        return 0


@pytest.fixture
def eng(gs):
    return gs.GenieEngine(FakeLib(), object(), tokenizer=None)


def run(eng, prompt="USER: hi\n", **kw):
    """query() with the chunks collected. Returns (finish, chunks)."""
    out = []
    finish = eng.query(prompt, out.append, **kw)
    return finish, out


def park_a_generation(eng, prompt="OTHER: hi\n", **kw):
    """ANOTHER request's generation, parked inside GenieDialog_query on its
    own thread. Returns a callable that lets it finish and gives its finish.

    This is the bystander in every scoping test below: the turn that holds
    the dialog while somebody else's abort goes off.
    """
    started, release = threading.Event(), threading.Event()
    got = {}

    def inside_the_query():
        started.set()
        release.wait(timeout=5)
    eng.lib.on_query = inside_the_query

    def body():
        got["finish"] = eng.query(prompt, lambda t: None, **kw)
    t = threading.Thread(target=body, daemon=True)
    t.start()
    assert started.wait(timeout=5), "the bystander never reached its query"

    def let_it_finish():
        release.set()
        t.join(timeout=5)
        assert not t.is_alive()
        return got["finish"]
    return let_it_finish


# --- _plan: continue or reset --------------------------------------------

def test_a_strict_extension_reuses_the_kv(eng):
    # The whole point: send only the new suffix, keep the resident prefill.
    eng._committed = "SYSTEM\nUSER: hi\nASSISTANT: hello\n"
    text, reused = eng._plan(eng._committed + "USER: more\nASSISTANT: ")
    assert reused is True
    assert text == "USER: more\nASSISTANT: ", "must send ONLY the new suffix"
    assert eng.lib.resets == 0, "reuse must not reset the dialog"


def test_one_changed_byte_forces_a_full_reset(eng):
    # A near-match is NOT good enough. If the client edited history, the window
    # evicted a turn, or the echoed assistant turn differs by a character, then
    # resuming would answer from a history that never happened.
    eng._committed = "USER: what is 2+2\nASSISTANT: four\n"
    text, reused = eng._plan("USER: what is 2+2\nASSISTANT: five\nUSER: again\n")
    assert reused is False
    assert text == "USER: what is 2+2\nASSISTANT: five\nUSER: again\n"
    assert eng.lib.resets == 1


def test_an_identical_prompt_resets_rather_than_sending_nothing(eng):
    # The boundary in the condition (`len(prompt) > len(c)`), reachable from an
    # ordinary client retry. Reusing here would send a zero-token suffix, which
    # is the shape most likely to wedge rather than error.
    eng._committed = "USER: hi\n"
    text, reused = eng._plan("USER: hi\n")
    assert reused is False
    assert text == "USER: hi\n"
    assert eng.lib.resets == 1


def test_a_shorter_prompt_resets(eng):
    # History was truncated: the resident KV holds MORE than the client now
    # claims, so continuing would answer with turns the client dropped.
    eng._committed = "USER: a\nUSER: b\nUSER: c\n"
    _text, reused = eng._plan("USER: a\n")
    assert reused is False
    assert eng.lib.resets == 1


def test_nothing_committed_means_reset(eng):
    # Cold start, and the state every failure path returns to.
    assert eng._committed is None
    _text, reused = eng._plan("USER: hi\n")
    assert reused is False
    assert eng.lib.resets == 1


def test_a_prefix_that_is_not_at_the_start_does_not_count(eng):
    # startswith, not "contains" -- shared text mid-prompt is not a resumable
    # prefix and treating it as one would resume at the wrong offset.
    eng._committed = "ASSISTANT: hello\n"
    _text, reused = eng._plan("USER: say hello\nASSISTANT: hello\n")
    assert reused is False


def test_a_reset_drops_the_record_at_the_reset(eng):
    # From the reset on, the KV is empty. Leaving the old prefix recorded until
    # _commit overwrote it meant a throw in between (a driver error, a config
    # rejected) left the PREVIOUS conversation recorded against an empty KV,
    # and its next turn prefilled only the new suffix -- no system prompt, no
    # history -- and answered fluently from nothing.
    eng._committed = "USER: old\n"
    eng._plan("USER: unrelated\n")
    assert eng._committed is None


# --- _commit: what the KV is recorded as holding --------------------------

def test_commit_records_exactly_what_was_sent_and_returned(eng):
    # Not a byte more. Appending a turn terminator that was never sent would
    # claim the KV holds a byte it may not, and every later continuation would
    # resume one token out of step.
    eng._commit("PROMPT", "GENERATED", True)
    assert eng._committed == "PROMPTGENERATED"


def test_a_failed_generation_drops_the_record(eng):
    # On failure the resident state is UNKNOWN. Guessing poisons every
    # subsequent continuation, so the next turn must re-prefill.
    eng._committed = "old"
    eng._commit("PROMPT", "partial", False)
    assert eng._committed is None


def test_a_dropped_record_forces_the_next_plan_to_reset(eng):
    # The two halves together: a failure must actually cost the reuse.
    eng._commit("USER: hi\n", "hello", True)
    _t, reused = eng._plan("USER: hi\nhelloUSER: more\n")
    assert reused is True

    eng._commit("USER: hi\nhelloUSER: more\n", "x", False)
    _t, reused = eng._plan("USER: hi\nhelloUSER: more\nxUSER: again\n")
    assert reused is False, "a failed turn must force a re-prefill"


# --- status codes -> finish_reason ----------------------------------------
# Pinned against QAIRT 2.45's GenieCommon.h, because the constants were wrong
# and nothing here noticed: CONTEXT_EXCEEDED was declared as 1, which is the
# value of ABORTED. The two failures are each other's mirror image, and both
# are invisible without hardware -- which is exactly why the mapping is worth
# a device-free test even though the calls it decodes are not.

def test_a_context_full_generation_reports_length_rather_than_raising(gs):
    # GENIE_STATUS_WARNING_CONTEXT_EXCEEDED is 4. Declared as 1, a real
    # context-full generation fell through to the raise and surfaced as a 500
    # on a request that had in fact produced a complete, usable answer.
    assert gs.GENIE_STATUS_WARNING_CONTEXT_EXCEEDED == 4
    assert gs.GenieEngine._finish(4) == "length"


def test_an_aborted_generation_is_not_an_error(gs):
    # ABORTED is 1, and it is this server's OWN signal_abort landing after the
    # client hung up -- so it reports like any other early stop. Under the old
    # constants it returned "length", claiming the model hit the context wall
    # when the truth was that nobody was listening any more.
    assert gs.GENIE_STATUS_WARNING_ABORTED == 1
    assert gs.GenieEngine._finish(1) == "stop"


def test_a_real_error_status_still_raises(gs):
    # The guard the two fixes above must not have widened: a genuine failure
    # (-6 is GENIE_STATUS_ERROR_QUERY_FAILED) still raises rather than being
    # decoded into a plausible finish_reason.
    assert gs.GenieEngine._finish(gs.GENIE_STATUS_SUCCESS) == "stop"
    with pytest.raises(RuntimeError, match="status=-6"):
        gs.GenieEngine._finish(-6)


@pytest.mark.parametrize("status,name", [(2, "WARNING_BOUND_HANDLE"),
                                         (3, "WARNING_PAUSED")])
def test_the_other_two_warnings_raise_by_name(gs, status, name):
    # Declared beside ABORTED and CONTEXT_EXCEEDED but never a way a query
    # ends here (nothing sends PAUSE, nothing frees a handle mid-query), so
    # they are not decoded into a finish -- and the raise says WHICH warning,
    # where "status=2" alone sent the reader to the header.
    with pytest.raises(RuntimeError, match=name):
        gs.GenieEngine._finish(status)


def test_the_finished_set_is_exactly_what_finish_decodes(gs):
    # One set drives both _finish and the health verdict. If the two ever
    # disagree again, an abort goes back to counting as an engine failure.
    decoded = {s for s in range(-20, 10)
               if not _raises(gs.GenieEngine._finish, s)}
    assert decoded == set(gs.GENIE_FINISHED_STATUSES) == {0, 1, 4}


def _raises(fn, *a):
    try:
        fn(*a)
    except RuntimeError:
        return True
    return False


# --- the unified query path: one choreography, exercised for real ---------

def test_query_makes_the_calls_in_order_and_commits(eng, gs):
    eng.lib.chunks = ["hel", "lo"]
    finish, out = run(eng, "USER: hi\n", max_tokens=32)
    assert finish == "stop"
    assert out == ["hel", "lo"]
    assert eng.lib.calls == ["reset", "setMax", "query"]
    assert eng.lib.max_tokens == [32]
    assert eng.lib.sent == [b"USER: hi\n"]
    assert eng._committed == "USER: hi\nhello"


def test_the_stream_path_is_the_same_choreography(eng, gs):
    # query_stream's worker used to be a line-for-line copy of query() and
    # the copies had drifted. Now it is the same _run_query: same calls, same
    # record, same finish -- proven by running both and comparing.
    eng.lib.chunks = ["a", "b"]
    res = {}
    got = list(eng.query_stream("USER: hi\n", res, max_tokens=32))
    assert got == ["a", "b"] and res == {"finish": "stop"}
    assert eng._committed == "USER: hi\nab"
    assert eng.lib.calls == ["reset", "setMax", "query"]


def test_a_second_turn_that_extends_the_first_sends_only_the_suffix(eng):
    eng.lib.chunks = ["hello"]
    run(eng, "USER: hi\n")
    eng.lib.chunks = ["more"]
    run(eng, "USER: hi\nhelloUSER: again\n")
    assert eng.lib.resets == 1, "the second turn must continue, not reset"
    assert eng.lib.sent[-1] == b"USER: again\n"
    assert eng._committed == "USER: hi\nhelloUSER: again\nmore"


def test_the_cap_is_always_applied_even_when_none_was_asked_for(eng, gs):
    # The cap lives on the RESIDENT dialog. A turn that skipped
    # setMaxNumTokens ran under whatever the previous turn had set; with
    # nothing asked for, the server default is set explicitly instead.
    run(eng, max_tokens=None)
    assert eng.lib.max_tokens == [gs.DEFAULT_MAX_TOKENS]


def test_a_consumer_that_raises_does_not_break_the_generation(eng):
    eng.lib.chunks = ["a", "b"]

    def bad(_t):
        raise RuntimeError("consumer bug")
    assert eng.query("p", bad) == "stop"
    assert eng._committed == "pab", "the generation completed and was recorded"


# --- what the health accounting books ------------------------------------

def test_a_client_generation_is_counted_and_clears_the_streak(eng, gs):
    gs.HEALTH.consecutive_failures = 2
    run(eng)
    snap = gs.HEALTH.snapshot(0)
    assert snap["generations"] == 1
    assert snap["consecutive_failures"] == 0
    assert snap["generating"] is False, "end() must have closed it out"


def test_an_internal_call_is_supervised_but_not_counted(eng, gs):
    # The real consumer of `internal`: HEALTH.end(counted=not internal). The
    # StubEngine tests in test_supervision only check the kwarg was passed.
    eng.lib.chunks = ["- note"]
    run(eng, "summarise this", internal=True, commit=False)
    assert gs.HEALTH.snapshot(0)["generations"] == 0
    assert eng.lib.queries == 1, "not counted is not the same as not run"


def test_commit_false_leaves_the_record_unknown_even_on_success(eng):
    # The real consumer of `commit=False`: an internal summarisation leaves
    # the dialog holding text that is NOT the caller's conversation. Losing
    # this branch (the obvious casualty of a unify-the-two refactor) leaves a
    # pre-summarisation prefix that _plan would byte-match against a dialog
    # that was reset to hold the summary prompt.
    eng.lib.chunks = ["hello"]
    run(eng, "USER: hi\n")
    assert eng._committed == "USER: hi\nhello"
    eng.lib.chunks = ["- note"]
    run(eng, "summarise", commit=False, internal=True)
    assert eng._committed is None
    _t, reused = eng._plan("USER: hi\nhelloUSER: more\n")
    assert reused is False, "the summary prompt is what the KV holds now"


def test_the_stream_path_honours_commit_and_internal_too(eng, gs):
    res = {}
    list(eng.query_stream("summarise", res, commit=False, internal=True))
    assert res["finish"] == "stop"
    assert eng._committed is None
    assert gs.HEALTH.snapshot(0)["generations"] == 0


@pytest.mark.parametrize("status,finish", [(1, "stop"), (4, "length")])
def test_an_abort_or_a_full_window_is_not_a_failure(eng, gs, status, finish):
    # Both used to be booked against consecutive_failures because the health
    # call compared against SUCCESS alone, while _finish decoded them as
    # ordinary finishes: three stop-button presses in a row flipped /health
    # to 503 "failing" on an engine that had done nothing wrong.
    eng.lib.status = status
    gs.HEALTH.consecutive_failures = 2
    assert run(eng)[0] == finish
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 0
    assert eng._committed is None, "the tail is unrecorded either way"


def test_an_error_status_raises_and_counts_as_a_failure(eng, gs):
    eng.lib.status = -6
    with pytest.raises(RuntimeError, match="status=-6"):
        run(eng)
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 1
    assert eng._committed is None


def test_a_throw_inside_the_query_counts_as_a_failure(eng, gs):
    # `status` used to be pre-set to SUCCESS before the try, so a throw inside
    # it reached the finally as a successful, counted generation -- resetting
    # the failure streak on the very call that had just failed.
    def boom():
        raise OSError("driver fault on the calling thread")
    eng.lib.on_query = boom
    gs.HEALTH.consecutive_failures = 2
    with pytest.raises(OSError):
        run(eng)
    snap = gs.HEALTH.snapshot(0)
    assert snap["consecutive_failures"] == 3, "a throw is not a success"
    assert snap["generating"] is False, "and it must still be closed out"


def test_health_begins_before_the_first_native_call(eng, gs):
    # begin() used to run after reset / setStopSequence / getSampler /
    # setMaxNumTokens, so a wedge inside any of them left started=None and
    # /health said ok forever while every request 429d behind the lock.
    seen = {}
    eng.lib.on_stop = lambda: seen.setdefault(
        "stop", gs.HEALTH.snapshot(0)["generating"])
    eng.lib.on_reset = lambda: seen.setdefault(
        "reset", gs.HEALTH.snapshot(0)["generating"])
    run(eng, stop=["X"])
    assert seen == {"stop": True, "reset": True}
    assert eng.lib.calls.index("setStop") < eng.lib.calls.index("query")


# --- a throw must not leave a stale record --------------------------------

def test_a_throw_after_the_reset_drops_the_record(eng):
    # The sync path had no exception path at all: _plan reset the dialog,
    # the query raised, and the caller's 500 left the previous conversation's
    # prefix recorded against an EMPTY KV -- its next turn was then served as
    # a suffix-only prefill with no system prompt and no history.
    eng.lib.chunks = ["hello"]
    run(eng, "USER: hi\n")
    assert eng._committed == "USER: hi\nhello"

    def boom():
        raise OSError("mid-query")
    eng.lib.on_query = boom
    with pytest.raises(OSError):
        run(eng, "USER: something else\n")
    assert eng._committed is None
    eng.lib.on_query = None
    _t, reused = eng._plan("USER: hi\nhelloUSER: more\n")
    assert reused is False, "the old prefix must not be resumable"


def test_a_throw_on_the_reuse_path_drops_the_record(eng):
    # No reset happens on a continuation, so _plan's reset-time clear does not
    # apply -- this is the case the except path exists for: the KV now holds
    # the old prefix plus however much of the suffix was prefilled before the
    # throw, which is a state nothing can describe.
    eng.lib.chunks = ["hello"]
    run(eng, "USER: hi\n")

    def boom():
        raise OSError("mid-prefill")
    eng.lib.on_query = boom
    with pytest.raises(OSError):
        run(eng, "USER: hi\nhelloUSER: more\n")
    assert eng.lib.resets == 1, "this WAS the reuse path"
    assert eng._committed is None


def test_a_prompt_that_cannot_be_encoded_fails_before_touching_the_engine(eng, gs):
    # A lone surrogate survives json.loads and count_tokens swallows the
    # encode error while sizing, so the first thing to notice used to be the
    # encode INSIDE the locked section, after the reset. Now it is refused
    # before any native call, and the engine's state is exactly as it was.
    eng._committed = "USER: hi\nhello"
    with pytest.raises(UnicodeEncodeError):
        run(eng, "USER: \ud83d\n")
    assert eng.lib.calls == [], "no native call was made"
    assert eng._committed == "USER: hi\nhello", "the record is still accurate"
    snap = gs.HEALTH.snapshot(0)
    assert snap["generations"] == 0 and snap["generating"] is False


def test_the_stream_path_reports_a_throw_as_an_error_and_drops_the_record(eng, gs):
    eng._committed = "old"
    eng.lib.status = -6
    res = {}
    got = list(eng.query_stream("p", res))
    assert got == []
    assert res["finish"] == "stop" and "status=-6" in res["error"]
    assert eng._committed is None
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 1


# --- stop sequences: the dirty latch, and a rejected set ------------------

def test_stop_sequences_are_cleared_on_the_next_request_that_has_none(eng):
    # The docstring's contract: "must be called on EVERY request". The dialog
    # is resident, so one caller's stop sequences would truncate the next
    # caller's output unless the next request clears them -- with [""], the
    # idle value the SDK's own configs carry.
    run(eng, stop=["X"])
    run(eng, stop=None)
    assert eng.lib.stop_payloads == [{"stop-sequence": ["X"]},
                                     {"stop-sequence": [""]}]
    run(eng, stop=None)
    assert len(eng.lib.stop_payloads) == 2, "nothing set, nothing to clear"


def test_a_rejected_stop_sequence_set_fails_the_request_visibly(eng, gs):
    # Both callers used to discard the status: a rejected set ran the
    # generation without the stops the caller asked for, silently.
    eng.lib.stop_status = -8
    with pytest.raises(RuntimeError, match=r"setStopSequence failed, status=-8 \(ERROR_JSON_SCHEMA\)"):
        run(eng, stop=["X"])
    assert eng.lib.queries == 0, "no generation without the caller's stops"
    assert eng._stop_dirty is False
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 1


def test_a_rejected_clear_names_the_leak_and_keeps_retrying(eng):
    run(eng, stop=["X"])
    eng.lib.stop_status = -8
    with pytest.raises(RuntimeError, match="still armed"):
        run(eng, stop=None)
    assert eng._stop_dirty is True, "still armed, so the next request must retry"
    eng.lib.stop_status = 0
    run(eng, stop=None)
    assert eng.lib.stop_payloads[-1] == {"stop-sequence": [""]}


# --- the other two dialog-state calls, and the statuses they discarded ----
# set_stop_sequences above raises on a rejected set because the dialog is
# RESIDENT: a call that did not take leaves the turn running against state
# nobody in it chose. The reset and the token cap are the same dialog and the
# same hazard, and both used to throw their status away.

def test_a_rejected_reset_does_not_prefill_onto_the_last_conversation(eng, gs):
    # The one silent wrong answer this file exists to prevent, reached from
    # the other side: a refused reset leaves the previous conversation in the
    # KV while the engine prefills the new prompt on top of it and records
    # prompt+generated as resident. The next turn's byte-prefix check then
    # passes and the model answers from a history that never happened.
    eng.lib.chunks = ["hello"]
    run(eng, "USER: hi\n")
    eng.lib.reset_status = -1
    with pytest.raises(RuntimeError, match=r"GenieDialog_reset failed, status=-1 \(ERROR_GENERAL\)"):
        run(eng, "USER: something else\n")
    assert eng.lib.queries == 1, "no generation onto a KV that was not cleared"
    assert eng._committed is None, "and nothing is recorded as resident"
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 1


def test_a_rejected_token_cap_does_not_run_under_the_last_requests_cap(eng, gs):
    # "Always a cap, and always applied" is what the comment beside it claims;
    # discarding the status was the one way that could be false. The cap lives
    # on the resident dialog, and `capped` is measured against the REQUESTED
    # cap -- so a turn held at the previous request's limit came back short
    # reporting "stop", which an agentic client reads as a complete answer.
    run(eng, "USER: hi\n", max_tokens=8)
    eng.lib.max_status = -1
    with pytest.raises(RuntimeError,
                       match=r"setMaxNumTokens\(64\) failed, status=-1 \(ERROR_GENERAL\)"):
        run(eng, "USER: and now a long one\n", max_tokens=64)
    assert eng.lib.queries == 1, "no generation under a cap nobody chose"
    assert eng._committed is None
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 1


def test_a_rejected_cap_leaves_a_kv_the_dropped_record_agrees_with(eng):
    # The cap is set AFTER _plan, so by the time the refusal lands the reset
    # has already run: the KV is empty, and the record dropped on the way out
    # says so rather than guessing. The next turn re-prefills from scratch.
    eng.lib.max_status = -1
    with pytest.raises(RuntimeError, match="setMaxNumTokens"):
        run(eng, "USER: hi\n")
    assert eng.lib.calls == ["reset", "setMax"], "and no query went out"
    assert eng._committed is None
    eng.lib.max_status = 0
    assert run(eng, "USER: hi\n")[0] == "stop", "and the next turn serves"
    assert eng.lib.resets == 2, "having re-prefilled, as the dropped record said"


# --- the sampler: a per-request override, and the baseline it returns to --
# Inert on QAIRT 2.45 (apply_sampler's docstring), and kept because the hazard
# it guards is real the day that changes: the dialog is resident, so an
# override that is never undone becomes every later caller's sampler.

def test_a_per_request_sampler_rides_on_the_baseline_and_is_restored(eng):
    eng.lib.sampler_ok = True
    eng.default_sampler = {"version": 1, "seed": 7, "temp": 0.8}
    run(eng, "tool turn", sampler={"temp": 0.0})
    assert eng.lib.calls.index("applySampler") < eng.lib.calls.index("query")
    run(eng, "chat turn", sampler=None)
    run(eng, "another", sampler=None)
    assert [p["sampler"] for p in eng.lib.sampler_payloads] == [
        {"version": 1, "seed": 7, "temp": 0.0},     # the override, ON the baseline
        {"version": 1, "seed": 7, "temp": 0.8},     # undone for the next caller
    ], "and nothing more once it is clean"
    assert eng.lib.freed_configs == 2, "every config handle made is freed"


def test_the_restore_sends_the_seed_the_dialog_was_created_with(eng):
    # What load_engine's baseline is FOR (test_startup pins that it carries
    # the created-with seed): this is the path that would send it. A baseline
    # read from disk made the restore re-apply the bundle's fixed 42.
    eng.lib.sampler_ok = True
    eng.default_sampler = {"version": 1, "seed": 123456, "temp": 0.8}
    run(eng, "a", sampler={"temp": 0.0})
    run(eng, "b", sampler=None)
    assert [p["sampler"]["seed"] for p in eng.lib.sampler_payloads] == [123456, 123456]


# --- a stop-sequence hit leaves tokens the record cannot carry -----------

def test_a_stop_sequence_request_that_ended_early_is_not_reusable(eng):
    # Genie STRIPS the matched text from what it hands back, but the tokens
    # that began the match were fed and sit in the KV. Recording
    # prompt + generated would pass the next byte-prefix check against a KV
    # holding a few tokens more, and the continuation would resume out of
    # step. Whether a sequence fired is not observable, so the record goes
    # whenever one could have.
    eng.lib.chunks = ["answer"]
    run(eng, "USER: hi\n", stop=["\nObservation:"], max_tokens=64)
    assert eng._committed is None


def test_a_stop_sequence_request_that_ran_to_its_cap_is_reusable(eng):
    # A generation that reached its cap could not have hit a stop sequence,
    # so prompt + generated is exact there.
    eng.lib.chunks = ["a", "b"]
    run(eng, "USER: hi\n", stop=["X"], max_tokens=2)
    assert eng._committed == "USER: hi\nab"


def test_a_recount_alone_does_not_make_a_stop_sequence_record_reusable(eng):
    # The re-encode is the tokenizer's split of the text, not the model's, and
    # it can run high. That is good enough to call a finish "length" and not
    # good enough to claim what the KV holds: only the callback count, which
    # can only run low, proves the cap was reached.
    eng.tokenizer = object()
    eng.lib.ntok = 3
    eng.lib.chunks = ["ab", "c"]            # two callbacks; the recount says 3
    finish, _out = run(eng, "USER: hi\n", stop=["X"], max_tokens=3)
    assert finish == "length"
    assert eng._committed is None


def test_a_request_without_stop_sequences_is_still_reusable(eng):
    eng.lib.chunks = ["answer"]
    run(eng, "USER: hi\n", max_tokens=64)
    assert eng._committed == "USER: hi\nanswer"


# --- finish "length" at the cap -----------------------------------------

def test_reaching_the_cap_reports_length(eng):
    # Genie reports SUCCESS at the cap -- a normal sentence-end -- so every
    # capped generation reported "stop" and a client could not tell a cut
    # answer from a complete one. One callback per token is the count.
    eng.lib.chunks = ["a", "b", "c"]
    assert run(eng, max_tokens=3)[0] == "length"


def test_stopping_short_of_the_cap_reports_stop(eng):
    eng.lib.chunks = ["a", "b"]
    assert run(eng, max_tokens=3)[0] == "stop"


def test_the_tokenizer_count_is_used_when_it_is_higher(eng):
    # The callback count can run low (a token whose bytes are a partial
    # character may not get a callback of its own); the re-encode can run
    # low too (the model's split can be longer than the canonical one). The
    # larger of the two is the better estimate.
    eng.tokenizer = object()
    eng.lib.ntok = 3
    eng.lib.chunks = ["ab", "c"]            # two callbacks, three tokens
    assert run(eng, max_tokens=3)[0] == "length"
    assert "encode" in eng.lib.calls


def test_the_recount_is_skipped_when_it_cannot_change_the_answer(eng):
    # One native call per generation is not free under a single-flight lock:
    # the callbacks already reached the cap, or the status already decided.
    eng.tokenizer = object()
    eng.lib.ntok = 99
    eng.lib.chunks = ["a", "b"]
    assert run(eng, max_tokens=2)[0] == "length"
    eng.lib.status = 4
    assert run(eng, "again", max_tokens=64)[0] == "length"
    assert "encode" not in eng.lib.calls


def test_the_recount_is_supervised_like_any_other_native_call(eng, gs):
    # It runs AFTER the generation has been closed out, still under the engine
    # lock. Unsupervised, a wedge in it left /health at ok with every request
    # 429ing behind the lock -- the hole begin-before-native-calls closed,
    # reopened one call later.
    seen = {}
    real = eng.lib.GenieTokenizer_encode

    def watching(*a):
        seen["native"] = gs.HEALTH.native_what
        seen["generating"] = gs.HEALTH.snapshot(0)["generating"]
        return real(*a)
    eng.lib.GenieTokenizer_encode = watching
    eng.tokenizer = object()
    eng.lib.ntok = 1
    eng.lib.chunks = ["a"]
    run(eng, max_tokens=8)
    assert seen == {"native": "GenieTokenizer_encode", "generating": False}
    assert gs.HEALTH.native_since is None, "and closed out afterwards"


def test_a_capped_abort_is_still_a_stop(eng):
    # "length" is for a generation that RAN into its cap. An abort that
    # happens to land at the cap did not.
    eng.lib.status = 1
    eng.lib.chunks = ["a", "b"]
    assert run(eng, max_tokens=2)[0] == "stop"


def test_a_full_window_is_length_regardless_of_the_count(eng):
    eng.lib.status = 4
    eng.lib.chunks = ["a"]
    assert run(eng, max_tokens=64)[0] == "length"


# --- abort scoping --------------------------------------------------------

def test_an_abort_with_nothing_in_flight_is_a_no_op(eng):
    # A handler's post-loop write failing after the worker released the lock
    # used to send GenieDialog_signal(ABORT) to an idle dialog. Nothing in
    # flight: nothing sent, nothing dropped.
    eng.lib.chunks = ["hello"]
    run(eng, "USER: hi\n")
    assert eng.signal_abort() is False
    assert eng.signal_abort(any_turn=True) is False
    assert eng.lib.signals == []
    assert eng._committed == "USER: hi\nhello", "a finished turn's record stands"


def test_a_finished_streams_abort_does_not_land_on_the_next_request(eng):
    # THE case. The native signal is engine-global: this thread's stream has
    # ended and released the engine lock, another request is already inside
    # its query, and then this handler's post-loop write fails. That abort
    # used to cut the OTHER client's answer short as an ordinary "stop".
    eng.lib.chunks = ["mine"]
    assert list(eng.query_stream("MINE: hi\n", {})) == ["mine"]
    eng.lib.chunks = ["theirs"]
    let_it_finish = park_a_generation(eng)
    assert eng.signal_abort() is False, "my turn is over; there is nothing of mine"
    assert eng.lib.signals == [], "the bystander must not be signalled"
    assert let_it_finish() == "stop"
    assert eng._committed == "OTHER: hi\ntheirs", "and its record stands"


def test_a_thread_that_never_had_a_turn_cannot_abort_anyone(eng):
    # A stream whose FIRST frame fails to write signals before it has called
    # query_stream at all. While another request is generating, that signal
    # must go nowhere.
    let_it_finish = park_a_generation(eng)
    assert eng.signal_abort() is False
    assert eng.lib.signals == []
    let_it_finish()
    assert eng._committed is not None


def test_any_turn_aborts_whoever_holds_the_dialog(eng, gs):
    # The watchdog and shutdown consume no stream, so a bare signal_abort()
    # from them is aimed at nothing; they say any_turn.
    eng.lib.chunks = ["cut"]
    let_it_finish = park_a_generation(eng)
    assert eng.signal_abort(any_turn=True) == gs.ABORT_SIGNALLED
    assert eng.lib.signals == [ABORT]
    let_it_finish()
    assert eng._committed is None, "an aborted turn's tail is unrecorded"


def test_the_watchdog_reaches_a_generation_it_is_not_consuming(eng, gs):
    # The real watchdog against the real engine, because the fakes in
    # test_supervision accept anything: had the watchdog kept calling a bare
    # signal_abort(), a stall would never have been signalled again.
    class Stalled:
        def assess(self, now):
            return "stalled", "scripted"

        def note_stall_signalled(self, now, native=True):
            pass
    let_it_finish = park_a_generation(eng)
    gs.watchdog(eng, Stalled(), interval=0, iterations=1)
    assert eng.lib.signals == [ABORT]
    let_it_finish()


def test_a_turn_the_watchdog_aborted_for_stalling_is_a_failed_generation(eng, gs):
    # ABORTED is a finished status, for the sake of a client's stop button.
    # The watchdog's abort rode the same path: a generation that stalled past
    # its limit and ended only because it was cut was booked as a healthy
    # finish, which RESET the failure streak. A device that stalled on every
    # turn but honoured each abort never reported `failing`, and two real
    # errors followed by one stall read as recovered.
    for _ in range(gs.HEALTH.fail_threshold - 1):
        gs.HEALTH.end(ok=False, counted=False)
    eng.lib.status = 1                      # Genie honoured the abort
    eng.lib.on_query = lambda: eng.signal_abort(any_turn=True, stalled=True)
    assert run(eng, "p")[0] == "stop", "the client still gets an ordinary finish"
    snap = gs.HEALTH.snapshot(0)
    assert snap["consecutive_failures"] == gs.HEALTH.fail_threshold
    assert snap["state"] == "failing"
    assert snap["generations"] == 1, "counted as served traffic all the same"


@pytest.mark.parametrize("kw", [{}, {"any_turn": True}])
def test_an_abort_nobody_sent_for_a_stall_is_still_not_a_failure(eng, gs, kw):
    # The other side of the line: a client's disconnect (bare) and shutdown
    # (any_turn, but not for a stall) stay ordinary finishes. Three presses of
    # a stop button must not flip /health to failing.
    gs.HEALTH.end(ok=False)
    eng.lib.status = 1
    eng.lib.on_query = lambda: eng.signal_abort(**kw)
    run(eng, "p")
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 0


def test_the_real_watchdog_books_the_stall_it_aborted(eng, gs):
    # The same through the real watchdog, so the keyword it passes and the
    # flag the engine reads cannot drift apart behind two green tests.
    class Stalled:
        def assess(self, now):
            return "stalled", "scripted"

        def note_stall_signalled(self, now, native=True):
            pass
    eng.lib.status = 1
    let_it_finish = park_a_generation(eng)
    gs.watchdog(eng, Stalled(), interval=0, iterations=1)
    assert let_it_finish() == "stop"
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 1


def test_a_disconnect_is_one_native_abort_not_two(eng, gs):
    # What Handler._run really does when a client leaves mid-generation, in
    # its order and on its thread: _Emitter.lost() calls signal_abort(), the
    # loop breaks, and the finally closes the generator -- whose close aborts
    # the same turn again, microseconds later. Each half was tested alone and
    # each sent [ABORT]; together they sent it twice, the second just as the
    # first was making the query return. A signal landing after that return
    # is a signal at an idle dialog -- the case this engine refuses to create
    # anywhere else.
    release = threading.Event()
    eng.lib.on_query = lambda: release.wait(timeout=5)
    eng.lib.chunks = ["first"]
    eng.lib.status = 1
    gen = eng.query_stream("p", {})
    assert next(gen) == "first"
    assert eng.signal_abort() == gs.ABORT_SIGNALLED      # _Emitter.lost()
    gen.close()                                          # _run's finally
    assert eng.lib.signals == [ABORT], "one disconnect, one native signal"
    # A repeat from the consumer still REPORTS the turn as signalled.
    assert eng.signal_abort() == gs.ABORT_SIGNALLED
    assert eng.lib.signals == [ABORT]
    release.set()


def test_the_watchdog_may_signal_a_turn_again(eng, gs):
    # The exemption: each STALL line is a fresh signal at a query that has
    # not honoured the last one, and the once-per-turn rule must not mute it
    # -- not even for a turn its own consumer has already aborted.
    eng.lib.status = 1
    let_it_finish = park_a_generation(eng)
    assert eng.signal_abort(any_turn=True, stalled=True) == gs.ABORT_SIGNALLED
    assert eng.signal_abort(any_turn=True, stalled=True) == gs.ABORT_SIGNALLED
    assert eng.lib.signals == [ABORT, ABORT]
    let_it_finish()


def test_a_stall_before_the_query_is_flagged_and_reported_as_unsignalled(
        eng, gs, capsys):
    # The real watchdog, the real engine, a hang inside setStopSequence: a
    # turn holds the dialog and is not querying. No native signal goes out
    # (test_an_abort_before_the_query_begins_... says why), and the log must
    # not claim one did -- first as "signalling abort" with nothing after it,
    # then as "an abort was signalled ... and did not take".
    started, release = threading.Event(), threading.Event()

    def stuck():
        started.set()
        release.wait(timeout=5)
    eng.lib.on_stop = stuck
    got = {}
    t = threading.Thread(
        target=lambda: got.setdefault("finish", run(eng, "p", stop=["X"])[0]),
        daemon=True)
    t.start()
    assert started.wait(timeout=5)
    h = gs.EngineHealth(first_token_timeout=0, stall_timeout=0, grace=-1,
                        fail_threshold=3)
    h.begin(0)
    detail = gs.watchdog(eng, h, interval=0, iterations=2, on_wedge=lambda d: d)
    out = capsys.readouterr().out
    assert "no native ABORT was sent" in out
    assert "did not take" not in out and "did not take" not in detail
    assert "No ABORT was sent" in detail
    release.set()
    t.join(timeout=5)
    assert got == {"finish": "stop"}
    assert eng.lib.signals == [] and eng.lib.queries == 0
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 1, (
        "a turn the watchdog had to flag for stalling is a failed one")


def test_an_aborted_turn_does_not_poison_the_next_one(eng):
    # The flag lives on the turn, so there is no stale one to inherit: the
    # turn after an aborted turn commits like any other.
    eng.lib.on_query = lambda: eng.signal_abort()
    run(eng, "p")
    assert eng._committed is None
    eng.lib.on_query = None
    eng.lib.chunks = ["G2"]
    run(eng, "P2")
    assert eng._committed == "P2G2", "the next turn must be allowed to commit"


def test_an_abort_mid_generation_is_sent_and_drops_the_record(eng):
    eng.lib.on_query = lambda: eng.lib.signals.append(("sent", eng.signal_abort()))
    eng.lib.chunks = ["hel"]
    eng.lib.status = 1                      # Genie honoured it
    assert run(eng, "p")[0] == "stop"
    assert eng.lib.signals == [ABORT, ("sent", "signalled")]
    assert eng._committed is None


def test_an_abort_that_lands_after_the_query_finished_still_drops_the_record(eng):
    # The race the flag exists for: the signal arrives inside the in-flight
    # window, but Genie had already produced its last token, so the query
    # returns SUCCESS. The worker must not re-commit over the abort.
    eng.lib.on_query = lambda: eng.signal_abort()
    eng.lib.chunks = ["done"]
    eng.lib.status = 0
    run(eng, "p")
    assert eng._committed is None


def test_an_abort_before_the_query_begins_is_honoured_without_starting_it(eng, gs):
    # A stream whose FIRST frame failed to write signalled an abort before the
    # worker had entered GenieDialog_query. That signal used to be spent on an
    # idle dialog and the generation then ran to its cap with nobody reading.
    # Inside the turn's window but before the query, the abort is honoured by
    # not starting the query at all.
    seen = {}
    eng.lib.on_stop = lambda: seen.setdefault("flagged", eng.signal_abort())
    finish, out = run(eng, "p", stop=["X"])
    assert seen == {"flagged": gs.ABORT_FLAGGED}, "marked, and NOT signalled"
    assert finish == "stop" and out == []
    assert eng.lib.queries == 0, "a generation nobody will read must not start"
    assert eng.lib.signals == [], (
        "no native signal at a dialog that is not inside a query: whether it "
        "would stick -- and cut the NEXT request short -- is not knowable")
    assert eng._committed is None
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 0, "not a failure"


def test_a_stream_consumer_that_leaves_early_aborts_the_generation(eng, gs):
    # The handler breaks out of its loop on a dead socket. That closes the
    # generator, and the close must reach the worker still inside the query
    # -- otherwise the worker holds the single-flight NPU to the cap.
    started, release, finished = (threading.Event(), threading.Event(),
                                  threading.Event())

    def blocking_query():
        started.set()
        release.wait(timeout=5)
        finished.set()
    eng.lib.on_query = blocking_query
    eng.lib.chunks = ["first"]
    eng.lib.status = 1
    for chunk in eng.query_stream("p", {}):     # a temporary, as the handler's is
        assert chunk == "first"
        break
    assert started.wait(timeout=5)
    assert eng.lib.signals == [ABORT], "leaving the loop must abort the worker"
    release.set()
    assert finished.wait(timeout=5)


def test_closing_the_stream_from_another_thread_still_aborts_its_turn(eng):
    # A generator can be finalised on a thread that never iterated it, so the
    # close aborts its turn BY REFERENCE rather than through whichever thread
    # happens to run the finally.
    release = threading.Event()
    eng.lib.on_query = lambda: release.wait(timeout=5)
    eng.lib.chunks = ["first"]
    eng.lib.status = 1
    box = {}

    def consumer():
        box["gen"] = eng.query_stream("p", {})
        box["first"] = next(box["gen"])
    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    t.join(timeout=5)
    assert box["first"] == "first"
    box["gen"].close()                      # on THIS thread, which has no turn
    assert eng.lib.signals == [ABORT]
    release.set()


def test_a_stream_consumed_to_the_end_sends_no_abort(eng):
    eng.lib.chunks = ["a"]
    res = {}
    list(eng.query_stream("p", res))
    assert res["finish"] == "stop"
    assert eng.lib.signals == []


# --- who cut the turn short: the stream says when it was the SERVER ---------
# _finish reports ABORTED as "stop", and a server abort -- shutdown, or the
# watchdog on a stall -- reached the client as exactly that: a fragment
# labelled as a finished answer. query_stream now says who aborted it, and
# Handler._run turns that into an error (test_finish_reason drives the
# response paths). A client's own abort must not be dressed up the same way.

def test_a_turn_shutdown_aborted_is_reported_as_the_servers_abort(eng, gs):
    eng.lib.chunks = ["hello"]
    eng.lib.status = 1                          # ABORTED, as Genie returns it
    eng.lib.on_query = eng.begin_shutdown       # Ctrl-C lands mid-decode
    res = {}
    assert list(eng.query_stream("p", res)) == ["hello"]
    assert res["aborted_by"] == "shutdown" and res["finish"] == "stop"
    assert res["error"] == gs.SERVER_ABORTS["shutdown"]
    assert "closing" not in res, "not the queued turn refused at the door"
    assert eng.lib.signals == [ABORT]


def test_a_turn_the_watchdog_aborted_is_reported_as_the_servers_abort(eng, gs):
    eng.lib.chunks = ["hello"]
    eng.lib.status = 1
    eng.lib.on_query = lambda: eng.signal_abort(any_turn=True, stalled=True)
    res = {}
    list(eng.query_stream("p", res))
    assert res["aborted_by"] == "watchdog"
    assert res["error"] == gs.SERVER_ABORTS["watchdog"]


def test_a_stall_abort_that_landed_as_the_query_finished_is_still_reported(eng, gs):
    # SUCCESS after the abort: whether the generation was cut cannot be told
    # (the KV record drops it for the same reason), so the client is not told
    # it is whole either.
    eng.lib.chunks = ["hello"]
    eng.lib.on_query = lambda: eng.signal_abort(any_turn=True, stalled=True)
    res = {}
    list(eng.query_stream("p", res))
    assert res["aborted_by"] == "watchdog"


def test_a_consumers_own_abort_is_not_the_servers(eng, gs):
    # A client leaving aborts its turn through the consumer's bare
    # signal_abort. That is not the server cutting an answer short -- the
    # handler returns on `em.gone` without answering -- so nothing here may
    # read as one.
    eng.lib.chunks = ["hello", "world"]
    eng.lib.status = 1
    signalled = threading.Event()
    real_signal = eng.lib.GenieDialog_signal

    def signal(dialog, action):
        signalled.set()
        return real_signal(dialog, action)
    eng.lib.GenieDialog_signal = signal
    # Held inside the query until the consumer's abort arrives, so it lands
    # on a live query rather than on a turn that has already returned.
    eng.lib.on_query = lambda: signalled.wait(timeout=5)
    got, res = [], {}
    for chunk in eng.query_stream("p", res):
        got.append(chunk)
        if len(got) == 1:
            eng.signal_abort()                  # MINE, from the consumer
    assert eng.lib.signals == [ABORT]
    assert "aborted_by" not in res and "error" not in res
    assert res["finish"] == "stop"


def test_a_turn_marked_for_shutdown_that_finished_first_is_whole(eng, gs):
    # The mark alone decides nothing: begin_shutdown marks the live turn and
    # then aborts it, and a turn that ends between the two was not cut.
    eng.lib.chunks = ["hello"]

    def mark_only():
        with eng._abort_lock:
            eng._live.shutdown = True
    eng.lib.on_query = mark_only
    res = {}
    list(eng.query_stream("p", res))
    assert "aborted_by" not in res and "error" not in res
    assert res["finish"] == "stop"


def test_begin_shutdown_marks_only_the_turn_it_aborts(eng, gs):
    # No live turn: nothing to mark, nothing signalled.
    assert eng.begin_shutdown() is False
    assert eng._closing is True


def test_a_signal_that_faults_in_the_driver_does_not_escape_the_abort(eng, gs):
    # _abort runs from query_stream's `finally` on every client disconnect and
    # from _Emitter.lost(). A driver fault inside GenieDialog_signal arrives
    # here as ctypes' "OSError: exception: access violation" -- the same
    # 0xC0000005 the shutdown path anticipates -- and letting it out would
    # replace the generator's real exception on the way out, turning an
    # ordinary hang-up into an unhandled failure on a request nobody was
    # reading any more.
    def faulting_signal(dialog, action):
        raise OSError("exception: access violation reading 0x0")
    eng.lib.GenieDialog_signal = faulting_signal
    eng.lib.chunks = ["hello"]
    let_it_finish = park_a_generation(eng)
    assert eng.signal_abort(any_turn=True) == gs.ABORT_SIGNALLED, (
        "the signal went out as far as this side can tell")
    assert let_it_finish() == "stop", "and the turn it aimed at still ended"


# --- close(): after the free, nothing reaches Genie ----------------------
# Shutdown frees the dialog under the engine lock, and that used to be the
# whole of it. But handler and worker threads are daemons: they outlive the
# free for as long as the interpreter takes to leave, and each of them was one
# lock acquisition away from another native call on the freed handle.

@pytest.fixture
def closing(gs):
    """A real engine with a tokenizer, installed as the server's ENGINE."""
    lib = FakeLib(chunks=["hello"], ntok=99)
    gs.ENGINE = gs.GenieEngine(lib, "DIALOG", tokenizer="TOKENIZER")
    return gs.ENGINE


def test_close_frees_the_dialog_once_under_the_lock_and_drops_the_handles(closing):
    eng = closing
    seen = {}
    real_free = eng.lib.GenieDialog_free

    def free(dialog):
        seen["locked"] = eng.lock.locked()
        seen["handles_gone_first"] = (eng.dialog, eng.tokenizer, eng._closed)
        return real_free(dialog)
    eng.lib.GenieDialog_free = free
    assert eng.close() is True
    assert eng.lib.freed == ["DIALOG"], "the handle it was made with"
    assert seen == {"locked": True, "handles_gone_first": (None, None, True)}
    assert not eng.lock.locked()
    assert eng.close() is True, "closing twice is not an error"
    assert eng.lib.freed == ["DIALOG"], "...and is not a double free"


def test_close_gives_up_without_touching_anything_when_the_lock_is_held(closing):
    eng = closing
    eng.lock.acquire()                  # a generation still inside the driver
    try:
        assert eng.close(timeout=0.01) is False
    finally:
        eng.lock.release()
    assert eng.lib.calls == []
    assert (eng._closed, eng.dialog, eng.tokenizer) == (False, "DIALOG", "TOKENIZER")


def test_a_free_that_faults_in_the_driver_still_finishes_the_shutdown(closing):
    # main()'s finally calls begin_shutdown() and close() bare, and the
    # comment there anticipates exactly this: a 0xC0000005 out of the driver,
    # which ctypes raises as an OSError. Letting it out of close() would take
    # the shutdown with it -- and it is the LAST thing main() does, so the
    # process would leave through an unhandled exception instead of a clean
    # exit. The handles are nulled before the free, so the free is the only
    # part that can be lost.
    eng = closing

    def faulting_free(dialog):
        eng.lib.calls.append("free")
        raise OSError("exception: access violation reading 0x0")
    eng.lib.GenieDialog_free = faulting_free
    assert eng.close(timeout=5) is True, "the shutdown still completed"
    assert eng.lib.calls == ["free"], "and the free WAS attempted"
    assert (eng._closed, eng.dialog, eng.tokenizer) == (True, None, None)
    assert not eng.lock.locked(), "the engine lock is released regardless"
    with pytest.raises(RuntimeError, match="engine is closed"):
        run(eng, "QUEUED: hi\n")


def test_nothing_reaches_genie_after_the_shutdown_free(closing, gs):
    # main()'s shutdown, in order, with a generation in flight as it is when
    # Ctrl-C lands -- and then what the daemon threads go on to do with the
    # lock the free has just released. What this does NOT cover is a request
    # that was ALREADY waiting for the lock when shutdown began, which is the
    # ordinary loaded state (GENIE_MAX_INFLIGHT is 2): every "queued" request
    # below is issued after close() has returned. The test after this one is
    # that case.
    eng = closing
    eng.lib.status = 1
    let_it_finish = park_a_generation(eng)
    assert eng.signal_abort(any_turn=True) == gs.ABORT_SIGNALLED
    assert let_it_finish() == "stop"
    assert eng.close(timeout=5) is True
    assert eng.lib.calls[-1] == "free"
    # 1. The handler whose generation that abort just ended counts its usage:
    #    GenieTokenizer_encode, on the tokenizer of the freed dialog. It gets
    #    the documented fallback instead -- the len//4 estimate, not ntok.
    assert eng.count_tokens("twelve chars") is None
    assert gs._tok_count("twelve chars") == 3
    # 2. A request that was queued behind the lock runs its turn: a reset and
    #    a query on the freed dialog. It raises by name, on both paths.
    with pytest.raises(RuntimeError, match="engine is closed"):
        run(eng, "QUEUED: hi\n")
    res = {}
    assert list(eng.query_stream("QUEUED: hi\n", res)) == []
    assert "engine is closed" in res["error"]
    # 3. The watchdog is a daemon too.
    assert eng.signal_abort(any_turn=True) is False
    assert eng.lib.calls[-1] == "free", (
        "a native call after GenieDialog_free: %r" % eng.lib.calls)
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 0, (
        "a turn refused at the door was never a generation to book")


class _WatchedLock:
    """The engine lock, plus an Event that fires when a thread BLOCKS on it.

    Not "is about to take it": the non-blocking attempt below can only fail
    while somebody else holds the lock, so the event means the caller is
    genuinely parked -- which is the state the shutdown race needs and the
    one a sleep can only guess at. Everything else delegates, because close()
    calls acquire(timeout=)/release() directly and other tests read locked().
    """

    def __init__(self, lock, parked):
        self._lock, self._parked = lock, parked

    def acquire(self, blocking=True, timeout=-1):
        if self._lock.acquire(False):
            return True
        if not blocking:
            return False
        self._parked.set()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return True

    def __exit__(self, *exc):
        self.release()


def test_a_turn_parked_on_the_lock_when_shutdown_begins_never_starts(closing, gs):
    # The loaded Ctrl-C: A is inside its query and B is already waiting for
    # the engine lock. The abort that ends A hands the lock to the LONGEST
    # WAITER, which is B and not close() -- so B used to reset the dialog and
    # begin a fresh GenieDialog_query, close() gave up after its 5s timeout,
    # main() printed "a generation is still inside the driver" about a driver
    # that was working perfectly, and the process left with a generation
    # running inside Genie and the dialog never freed (5 of 5 against the real
    # engine). B was never flagged, so nothing aborted it either.
    eng = closing
    eng.lib.status = 1
    let_it_finish = park_a_generation(eng)          # A, inside its query
    parked = threading.Event()
    eng.lock = _WatchedLock(eng.lock, parked)
    queries_before = eng.lib.queries
    err = {}

    def queued():
        try:
            run(eng, "QUEUED: hi\n")
        except BaseException as e:      # recorded, not swallowed
            err["raised"] = e
    b = threading.Thread(target=queued, daemon=True)
    b.start()
    assert parked.wait(timeout=5), "B never reached the engine lock"
    # main()'s shutdown, in that order.
    assert eng.begin_shutdown() == gs.ABORT_SIGNALLED
    assert let_it_finish() == "stop"
    assert eng.close(timeout=5) is True, (
        "close() lost the lock to the queued turn and gave up")
    b.join(timeout=5)
    assert not b.is_alive()
    assert isinstance(err.get("raised"), gs.EngineClosing), (
        "the parked turn ran instead of being refused: %r" % (err,))
    assert "shutting down" in str(err["raised"])
    assert eng.lib.queries == queries_before, "it started a second generation"
    assert eng.lib.calls[-1] == "free", (
        "a native call after GenieDialog_free: %r" % eng.lib.calls)
    assert gs.HEALTH.snapshot(0)["consecutive_failures"] == 0, (
        "a turn refused at the door was never a generation to book")


def test_a_bare_close_refuses_a_parked_turn_too(closing, gs):
    # begin_shutdown is the caller with a live turn to abort; close() on its
    # own still must not hand the lock it is waiting for to a turn that would
    # query a dialog about to be freed. Here the lock is held by nothing more
    # than a bystander thread, as it is while a tokenizer count runs.
    eng = closing
    parked = threading.Event()
    eng.lock = _WatchedLock(eng.lock, parked)
    eng.lock.acquire()                      # somebody else holds the engine
    err, done = {}, threading.Event()

    def queued():
        try:
            run(eng, "QUEUED: hi\n")
        except BaseException as e:      # recorded, not swallowed
            err["raised"] = e
        done.set()
    threading.Thread(target=queued, daemon=True).start()
    assert parked.wait(timeout=5), "the turn never reached the engine lock"
    assert eng.close(timeout=0.01) is False, "the lock was held"
    eng.lock.release()
    assert done.wait(timeout=5)
    assert isinstance(err.get("raised"), gs.EngineClosing)
    assert eng.lib.queries == 0, "it started a generation during shutdown"
    assert eng.close(timeout=5) is True, "and the retry gets the lock"


def test_a_count_that_was_already_past_the_door_still_makes_no_call(closing):
    # count_tokens checks for a tokenizer BEFORE it takes the lock, so a
    # handler can pass that check, wait out the free on the lock, and arrive
    # inside with the handle gone. _encode_count is where it arrives.
    eng = closing
    assert eng.close() is True
    with eng.lock:
        assert eng._encode_count("twelve chars") is None
    assert "encode" not in eng.lib.calls


def test_a_closed_engine_sends_no_signal_whatever_its_state_says(closing, gs):
    # Not reachable through the public paths -- a turn cannot be querying
    # once close() has had the lock -- which is exactly why it is pinned from
    # the inside: the signal is the one native call made WITHOUT the engine
    # lock, so it is the one that must not rest on that argument alone.
    eng = closing
    assert eng.close() is True
    turn = gs._Turn()
    eng._live, eng._querying = turn, True
    assert eng._abort(turn) is False
    assert eng.signal_abort(any_turn=True) is False
    assert eng.lib.signals == []


# --- what comes back is decoded as a stream, not per callback ------------

def test_a_multibyte_character_split_across_callbacks_survives(eng):
    # A byte-level BPE token can be a partial UTF-8 sequence. Decoding each
    # callback on its own turned one euro sign into two U+FFFD -- in the
    # stream and in the KV record.
    eng.lib.chunks = [b"\xe2\x82", b"\xac", b"!"]
    _f, out = run(eng, "p")
    assert "".join(out) == "€!"
    assert "�" not in "".join(out)
    assert eng._committed == "p€!"


def test_a_dangling_partial_sequence_is_flushed_as_a_replacement(eng):
    # The generation ended mid-character (a cap landing inside a multi-token
    # emoji). The tail is replaced, once, and emitted so the stream and the
    # record agree.
    eng.lib.chunks = [b"ok", b"\xe2\x82"]
    _f, out = run(eng, "p")
    assert "".join(out) == "ok�"
    assert eng._committed == "pok�"


def test_a_partial_callback_still_counts_as_progress(eng, gs):
    # Held bytes are still a token that arrived; the stall clock must see it.
    ticks = []
    gs.HEALTH.progress = lambda now: ticks.append(now)
    eng.lib.chunks = [b"\xe2\x82", b"\xac"]
    run(eng, "p")
    assert len(ticks) == 2


@pytest.mark.parametrize("nothing,name", [(b"", "empty"), (None, "NULL")],
                         ids=["empty", "NULL"])
def test_a_callback_carrying_nothing_is_neither_a_token_nor_progress(
        eng, gs, nothing, name):
    # The sibling guard above (a HELD partial sequence) deliberately DOES
    # count. This one must not: an empty or NULL response is the driver
    # calling back with no token at all. Counted, it inflates callbacks[] --
    # the count the code calls the CERTAIN cap signal -- so a generation that
    # stopped normally reports finish_reason "length" and an agentic client
    # continues an answer that was already complete; and the HEALTH.progress
    # it would tick resets the stall clock, so a driver calling back with
    # nothing would look like one producing tokens and never be declared
    # stalled.
    seen = {}
    eng.lib.on_query = lambda: seen.setdefault("tokens", gs.HEALTH.tokens)
    eng.lib.chunks = [nothing, "hi", nothing]
    finish, out = run(eng, "p", max_tokens=2)
    assert out == ["hi"], "the %s callback emitted nothing" % name
    assert seen == {"tokens": 1}, "and was not progress on the stall clock"
    assert finish == "stop", "one token back out of two is not the cap"
    assert eng._committed == "phi"


# --- _encode_count: the ctypes contract, and the swallow above it ---------
# The tokenizer encode is the one place this server hands Genie a buffer of
# its own rather than a handle, and the one native call every request makes
# (window fit, usage, the cap check). Both halves below are Python-side
# obligations the driver cannot enforce from its side of the boundary.

def test_the_encode_buffer_outlives_the_callback_that_allocated_it(
        eng, gs, monkeypatch):
    # Genie asks for the buffer through ALLOC_CALLBACK and goes on writing
    # token ids into it AFTER the callback has returned, so the only thing
    # keeping it alive is the engine's own `held` list. Drop that and CPython
    # frees the buffer the moment the callback returns while the driver still
    # holds the pointer -- a 0xC0000005 in the WER log of a process that is
    # otherwise healthy, which is the manufactured driver-fault evidence this
    # repo's docs and its close() are written to avoid.
    made = []
    real_buffer = gs.C.create_string_buffer

    def spy(size):
        b = real_buffer(size)
        made.append((size, weakref.ref(b)))
        return b
    monkeypatch.setattr(gs.C, "create_string_buffer", spy)

    seen = {}

    def encoding(tok, data, acb, tokptr, ntok_out):
        eng.lib.calls.append("encode")
        out = gs.C.c_char_p()
        acb(0, gs.C.byref(out))     # a driver that asks for nothing at all
        acb(16, gs.C.byref(out))
        gc.collect()                # ...and has not finished writing yet
        seen["sizes"] = [size for size, _ in made]
        seen["alive"] = [ref() is not None for _, ref in made]
        seen["pointer"] = gs.C.cast(out, gs.C.c_void_p).value is not None
        ntok_out._obj.value = 7
        return 0
    eng.lib.GenieTokenizer_encode = encoding
    eng.tokenizer = object()

    assert eng.count_tokens("some text") == 7
    assert seen["sizes"] == [1, 16], "an ask for 0 bytes still gets one byte"
    assert seen["pointer"] is True, "the driver was handed the buffer"
    assert seen["alive"] == [True, True], (
        "freed while the driver is still writing into it")


def test_a_count_of_text_that_cannot_be_encoded_is_none_rather_than_a_raise(
        eng, gs):
    # _tok_count promises never to raise, and build_windowed sizes every
    # prompt through it BEFORE _unsendable can answer 400 -- so on a live
    # server every lone-surrogate request (what JS emits for a sliced emoji,
    # and likelier now that tool arguments render with ensure_ascii=False)
    # goes through this swallow. Without it the named 400 degrades to
    # do_POST's catch-all reporting a UnicodeEncodeError over a rendered
    # prompt, which names neither the offending text nor what to do about it.
    eng.tokenizer = object()
    eng.lib.ntok = 99
    assert eng.count_tokens("cut \ud83d here") is None
    assert "encode" not in eng.lib.calls, "and the text never reached Genie"
    assert gs.HEALTH.native_since is None, "the native clock was closed out"
    assert eng.count_tokens("plain text") == 99, "an encodable count still works"
