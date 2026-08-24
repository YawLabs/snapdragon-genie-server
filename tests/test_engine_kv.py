"""Tests for the KV-reuse decision -- the engine's one silent-wrong-answer path.

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

Device-free. `_plan` needs only `lib.GenieDialog_reset`, and `_commit` is pure
arithmetic over strings, so both run against the REAL methods -- nothing here
is a stand-in for the code under test.
"""

import pytest


class FakeLib:
    """Records resets. The only Genie call `_plan` makes."""

    def __init__(self):
        self.resets = 0

    def GenieDialog_reset(self, dialog):
        self.resets += 1
        return 0


@pytest.fixture
def eng(gs):
    e = object.__new__(gs.GenieEngine)
    e.lib = FakeLib()
    e.dialog = object()
    e._committed = None
    e._aborted = False
    return e


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


def test_an_aborted_generation_drops_the_record_even_on_success(eng):
    # The race this flag exists for: the client disconnects, signal_abort fires
    # on the handler thread, and the worker then finishes and calls _commit
    # with ok=True. Committing there would re-arm reuse against a generation
    # that was cut short.
    eng._committed = "old"
    eng._aborted = True
    eng._commit("PROMPT", "cut short", True)
    assert eng._committed is None


def test_the_abort_flag_is_consumed_not_left_set(eng):
    # A stale abort must not poison the NEXT turn as well.
    eng._aborted = True
    eng._commit("P", "G", True)
    assert eng._aborted is False
    eng._commit("P2", "G2", True)
    assert eng._committed == "P2G2", "the next turn must be allowed to commit"


def test_a_dropped_record_forces_the_next_plan_to_reset(eng):
    # The two halves together: a failure must actually cost the reuse.
    eng._commit("USER: hi\n", "hello", True)
    _t, reused = eng._plan("USER: hi\nhelloUSER: more\n")
    assert reused is True

    eng._commit("USER: hi\nhelloUSER: more\n", "x", False)
    _t, reused = eng._plan("USER: hi\nhelloUSER: more\nxUSER: again\n")
    assert reused is False, "a failed turn must force a re-prefill"
