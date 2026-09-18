"""Context-window eviction and the summarisation that rides on it.

Overflow is a hard GenieDialog_query failure, not a truncation, and Genie has
no sliding-window mode -- so everything here is what stands between a long
conversation and a 500.

Tests that go through build_windowed pin a small n_ctx explicitly, because
build_windowed derives its budget from the window and a modest conversation
fits comfortably inside the real 4096.
"""

import importlib
import json
import random
import re

import pytest

from conftest import StubEngine, convo, request

SMALL = 500


def _unit_start(rest):
    """Index of the last message that is neither a tool result nor a system
    message (0 if none): where the turn being answered starts."""
    return max((i for i, m in enumerate(rest)
                if m.get("role") not in ("tool", "system")), default=0)


def linear_fit(g, messages, tools, thinking, budget):
    """The pre-bisection implementation, kept as the equivalence oracle.

    One message at a time from the front, written the slow obvious way on
    purpose. It stops when what is left is the current UNIT -- the last
    non-tool turn and the tool results after it -- and it never leaves a tool
    result at the head: both used to be `len(rest) > 1`, the same floor _fit
    had, which is how the two agreed with each other about keeping a lone
    trailing tool result.

    The system turn is the LEADING run of system messages, spelled out here
    rather than borrowed from the module, so the oracle is a second opinion
    on that too. A later one is a turn, and is never left at the head either.
    """
    lead = 0
    while lead < len(messages) and messages[lead].get("role") == "system":
        lead += 1
    sysm, rest = messages[:lead], messages[lead:]
    ev = []
    while True:
        p = g.TEMPLATE.build(sysm + rest, tools=tools, thinking=thinking)
        if g._tok_count(p) <= budget:
            return p, rest, ev, True
        if _unit_start(rest) == 0:
            return p, rest, ev, False
        ev.append(rest.pop(0))
        # Safe without a length check: the unit start was >= 1 before the pop,
        # so a message that is neither is still in there to stop on.
        while rest[0].get("role") in ("tool", "system"):
            ev.append(rest.pop(0))


# --- eviction --------------------------------------------------------------

def test_short_conversation_is_untouched(gs):
    p, kept, ev, fits = gs._fit(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        None, True, 4096)
    assert fits and ev == []


def test_system_turn_and_tools_survive_eviction(gs):
    tools = [{"type": "function", "function": {"name": "read_file",
              "description": "Read", "parameters": {"type": "object"}}}]
    p, kept, ev, fits = gs._fit(convo(20), tools, True, 400)
    assert fits and ev
    # Dropping these is how an agent forgets it has tools -- which reads as the
    # model getting dumber rather than as context loss.
    assert "You are a coding agent." in p
    assert "# Tools" in p and "read_file" in p
    assert "final question" in p and "q0 " not in p


def test_bisection_matches_the_linear_scan(gs):
    # The bisection replaced a linear rescan for speed. Speed is worthless if
    # it changes which turns survive.
    rng = random.Random(11)
    for _ in range(25):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(rng.randint(4, 40)):
            msgs.append({"role": "user",
                         "content": "q%d %s" % (i, "pad " * rng.randint(5, 50))})
            msgs.append({"role": "assistant",
                         "content": "a%d %s" % (i, "fil " * rng.randint(5, 50))})
            if rng.random() < 0.3:
                msgs.append({"role": "tool",
                             "content": "tool result %d %s" % (i, "z " * 20)})
            # A system message in the middle of the conversation -- a per-turn
            # reminder, a narrator's event -- is a turn like the others.
            if rng.random() < 0.3:
                msgs.append({"role": "system",
                             "content": "reminder %d %s" % (i, "r " * rng.randint(1, 60))})
        # Half of them end the way an AGENT's request ends: on the results of
        # the calls the last assistant turn made. Every conversation here used
        # to end on a user turn, so the tail rule was compared against nothing.
        if rng.random() < 0.5:
            msgs.append({"role": "user", "content": "final"})
        else:
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": "f", "arguments": {}}}]})
            for j in range(rng.randint(1, 3)):
                msgs.append({"role": "tool",
                             "content": "last result %d %s" % (j, "y " * rng.randint(5, 400))})
        budget = rng.choice([60, 300, 800, 1500, 3900])
        assert gs._fit(list(msgs), None, True, budget) == \
            linear_fit(gs, list(msgs), None, True, budget)


def test_bisection_is_cheaper_than_the_linear_scan(gs):
    msgs = convo(60)
    gs.ENGINE = StubEngine()
    gs._TOK_CACHE.clear()
    gs._fit(list(msgs), None, True, 3900)
    bisect = gs.ENGINE.encodes
    gs.ENGINE = StubEngine()
    gs._TOK_CACHE.clear()
    linear_fit(gs, list(msgs), None, True, 3900)
    linear = gs.ENGINE.encodes
    # Each encode is a native call holding the engine lock over the whole
    # prompt, and _fit runs twice per request when summarising.
    assert bisect * 3 < linear, (bisect, linear)


def test_the_token_cache_survives_a_clear_between_check_and_read(gs):
    # Handlers are concurrent threads, and the cache is cleared WHOLESALE by
    # whichever of them inserts the tenth entry. `if text in cache: return
    # cache[text]` is two lookups with a gap between them, and a clear landing
    # in that gap is a KeyError out of a function documented "Never raises".
    # do_POST's catch-all answers it, wrongly: a 400 "malformed request" for a
    # request with nothing wrong with it, a 500 for a finished answer, or a
    # stream closed mid-response. This dict makes the interleaving
    # deterministic: the other thread's clear lands exactly between the
    # membership test and the read.
    class ClearedBetweenCheckAndRead(dict):
        def __contains__(self, key):
            hit = dict.__contains__(self, key)
            self.clear()
            return hit

    gs._TOK_CACHE = ClearedBetweenCheckAndRead({"abcdefgh": 2})
    assert gs._tok_count("abcdefgh") == 2


def test_the_cache_lock_is_not_held_across_the_engine_call(gs):
    # count_tokens takes the ENGINE lock, which another request can hold for a
    # whole generation. Waiting there with the cache lock held would park every
    # cache HIT -- a dict read -- behind somebody else's decode.
    seen = []

    class Watching(StubEngine):
        def count_tokens(self, text):
            seen.append(gs._TOK_CACHE_LOCK.locked())
            return StubEngine.count_tokens(self, text)

    gs.ENGINE = Watching()
    assert gs._tok_count("x" * 40) == 10
    assert seen == [False]
    assert gs._tok_count("x" * 40) == 10 and seen == [False], "second is a hit"


def test_tool_results_are_never_orphaned(gs):
    # A token-level evictor inside Genie could not guarantee this; evicting at
    # message boundaries is the whole reason it happens server-side.
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(15):
        msgs.append({"role": "user", "content": "q%d %s" % (i, "x" * 60)})
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "f", "arguments": {}}}]})
        msgs.append({"role": "tool", "content": "result %d %s" % (i, "z" * 60)})
    msgs.append({"role": "user", "content": "last"})
    p, kept, ev, fits = gs._fit(msgs, None, True, SMALL)
    assert fits and ev
    _assert_no_orphaned_results(p)


def _assert_no_orphaned_results(p):
    # Outside the loop, and first: the check below lives INSIDE a loop over the
    # tags, so a prompt with no tool tags at all -- eviction having stripped
    # every one -- sailed through it without a single assertion running.
    assert "<tool_call>" in p and "<tool_response>" in p, "nothing to check"
    depth = 0
    for m in re.finditer(r"<(tool_call|tool_response)>", p):
        if m.group(1) == "tool_call":
            depth += 1
        else:
            assert depth > 0, "tool_response with no preceding tool_call"
            depth -= 1


def _call(*names):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": n, "arguments": {}}} for n in names]}


def test_a_trailing_tool_result_is_never_kept_without_its_call(gs):
    # Where the guarantee used to break: an agent's request ENDS on tool
    # results, and the evictor floored at the last MESSAGE rather than at the
    # turn that made the call. With two parallel results and room for only the
    # second, it kept that one alone and reported fits=True -- the handlers
    # check nothing else, so the client got a 200 over a prompt that opened on
    # a bare <tool_response>, the call and the sibling result silently gone.
    msgs = [{"role": "user", "content": "read both files"},
            _call("read_a", "read_b"),
            {"role": "tool", "content": "A" * 8000},      # 2000 tokens
            {"role": "tool", "content": "B" * 6000}]      # 1500 tokens
    for budget in (1600, 2000, 3000, 3500):
        p, kept, ev, fits = gs._fit(list(msgs), None, True, budget)
        assert not fits, "only an orphaned result fits at %d" % budget
        # What is reported back is the whole unit, call first -- so the 400's
        # token count describes the prompt that would actually be needed.
        assert kept == msgs[1:] and ev == msgs[:1]
        assert p.index("<tool_call>") < p.index("<tool_response>")


def test_the_current_tool_unit_survives_eviction_whole(gs):
    # The same shape when it DOES fit: older turns go, and the call stays
    # attached to every one of its results.
    msgs = [*convo(20)[:-1], _call("read_a", "read_b"),
            {"role": "tool", "content": "result A"},
            {"role": "tool", "content": "result B"}]
    p, kept, ev, fits = gs._fit(list(msgs), None, True, 400)
    assert fits and ev
    assert kept[-3:] == msgs[-3:]
    assert "result A" in p and "result B" in p and "read_a" in p
    _assert_no_orphaned_results(p)


def test_a_conversation_that_is_one_tool_unit_has_nothing_to_evict(gs):
    # The `last == 0` guard with more than one message left: [call, result,
    # result] is a single unit however many messages it spans, so an oversized
    # one is unfittable -- not "drop the call and keep a result".
    msgs = [_call("f"), {"role": "tool", "content": "R" * 8000},
            {"role": "tool", "content": "S" * 40}]
    p, kept, ev, fits = gs._fit(list(msgs), None, True, 400)
    assert not fits and kept == msgs and ev == []


def test_single_oversized_message_reports_unfittable(gs):
    p, kept, ev, fits = gs._fit(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "W" * 20000}],
        None, True, 400)
    assert not fits          # caller owes the client a 400, not a doomed query


def test_a_long_conversation_ending_oversized_reports_unfittable(gs):
    # The other way to be unfittable: plenty to evict, but evicting ALL of it
    # still does not fit. The bisection finds no k that works, and a `fits`
    # defaulted to True there would send a prompt over the compiled window --
    # a hard GenieDialog_query failure rather than the 400 the client is owed.
    tools = [{"type": "function", "function": {"name": "read_file",
              "description": "Read", "parameters": {"type": "object"}}}]
    msgs = convo(20)
    msgs[-1] = {"role": "user", "content": "W" * 8000}
    p, kept, ev, fits = gs._fit(msgs, tools, True, 400)
    assert not fits
    # Reached with turns still on the table, not via the len(rest) <= 1 guard:
    # everything but the current turn was dropped and it STILL did not fit.
    assert kept == [msgs[-1]] and len(ev) == len(msgs) - 2


def test_overflow_message_names_the_numbers(gs):
    # Names the real numbers, not just "too big" -- the client needs to know
    # what to shrink.
    # 99, not 64: WINDOW_MARGIN is 64 and the message always ends "(margin
    # 64)", so asserting "64" passed whether or not max_tokens was in it.
    msg = gs._overflow_msg("W" * 8000, 99)
    assert "n_ctx=%d" % gs.read_context_size() in msg
    assert "2000 prompt tokens + 99 max_tokens" in msg
    assert "(margin %d)" % gs.WINDOW_MARGIN in msg


def test_the_anthropic_leg_refuses_a_prompt_that_cannot_be_made_to_fit(gs, handler):
    # The OpenAI twin of this is covered; this leg is the one the typed router
    # drives, and its `fits` handling is a SECOND copy of the decision --
    # _anthropic_to_prompt passes build_windowed's 4-tuple straight through,
    # pinned by a comment and nothing else. Without the refusal the request
    # goes to the NPU as a query over a prompt longer than the compiled
    # window, which is a hard GenieDialog_query failure, not a truncation.
    code, body, _h = request(gs, handler, "POST", "/v1/messages",
                             {"messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 99999})
    assert code == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    # The numbers, in this endpoint's own envelope: a client that cannot parse
    # the error learns nothing at the moment it most needs to.
    assert "n_ctx=%d" % gs.read_context_size() in body["error"]["message"]
    assert "99999 max_tokens" in body["error"]["message"]
    assert gs.ENGINE.calls == [], "refused, never sent"


def test_an_eviction_says_so_on_both_legs(gs, handler, capsys):
    # Eviction is a real loss of information, and this print is the only thing
    # standing between that and silence. Both legs, because each calls it from
    # its own code: the OpenAI half could keep working while the Anthropic
    # half went quiet, and nothing else would notice.
    gs._CONTEXT_SIZE = SMALL
    for path, extra in (("/v1/chat/completions", {}), ("/v1/messages", {})):
        gs.ENGINE = StubEngine(chunks=["- earlier turns"])
        body = {"messages": convo(20), "max_tokens": 16, **extra}
        code, _b, _h = request(gs, handler, "POST", path, body)
        out = capsys.readouterr().out
        assert code == 200, (path, code)
        assert re.search(r"dropped \d+ oldest message", out), (path, out)
        assert "n_ctx=%d" % SMALL in out, (path, out)


def test_a_developer_turn_is_anchored_against_eviction_like_a_system_one(gs):
    # The eviction half of the decision pinned in test_prompt.py. Rendered as
    # an ordinary user turn, the agent's own instructions were the FIRST thing
    # thrown overboard: the model got dumber as the conversation grew, with no
    # error and no log line saying which turn went -- the exact outcome _fit's
    # anchoring exists to prevent. Byte-for-byte against the same conversation
    # spelled "system", because three places decide this (_leading_system, the
    # renderer, and _fit's unit boundary) and one of them lagging is how the
    # two spellings would quietly disagree.
    def spelled(role):
        # The instruction at the front, and -- the shape that makes the middle
        # of the conversation matter -- a reminder in the same spelling after
        # every exchange, which is what a framework injecting per-turn
        # instructions actually sends.
        out = []
        for m in convo(20):
            out.append({"role": role, "content": "You are a coding agent."}
                       if m.get("role") == "system" else m)
            if m.get("role") == "assistant":
                out.append({"role": role, "content": "Reminder: answer in French."})
        return out

    def windowed(msgs):
        gs.ENGINE = StubEngine(chunks=["- earlier turns"])
        return gs.build_windowed(list(msgs), max_tokens=64)

    gs._CONTEXT_SIZE = SMALL
    dev, sysm = spelled("developer"), spelled("system")
    pd, dropped, fd, _overhead = windowed(dev)
    assert fd and dropped, "the fixture has to actually evict for this to mean anything"
    assert "You are a coding agent." in pd
    assert "q0 " not in pd, "ordinary turns went first, as they should"
    # Byte for byte against the same conversation spelled "system", because
    # three places decide this -- _leading_system, the renderer, and _fit's
    # unit boundary -- and one of them lagging is how two spellings of one
    # role quietly disagree. `overhead` is left out of the comparison: the
    # second build reuses the retained note rather than paying for a second
    # summarisation, which is the note cache working, not a difference here.
    ps, ds, fs, _o = windowed(sysm)
    assert (pd, dropped, fd) == (ps, ds, fs)
    # Across budgets, because WHICH turn the cut lands on is what decides
    # whether the unit boundary is consulted at all. One left at the HEAD of
    # what is kept is no longer a later message: rendered after the system
    # turn it is folded INTO it, behind the retained note where _split_note
    # never finds it again, and anchored on the second pass.
    for budget in (60, 120, 300, 600, 900, 1500):
        pd, keptd, evd, fd = gs._fit(list(dev), None, True, budget)
        ps, _kepts, evs, fs = gs._fit(list(sysm), None, True, budget)
        assert (pd, fd, len(evd)) == (ps, fs, len(evs)), budget
        assert keptd[0].get("role") not in gs.SYSTEM_ROLES, budget


# --- summarisation ---------------------------------------------------------

def test_evicted_turns_are_summarised_into_the_system_turn(gs):
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["- user asked about the parser\n- token ZX-4417"])
    p, dropped, fits, overhead = gs.build_windowed(convo(20), max_tokens=64)
    assert fits and dropped and gs.SUMMARY_MARKER in p and "ZX-4417" in p
    # It rides in the system turn because that is the one thing eviction never
    # touches; anywhere else it would itself be evicted.
    assert p.index(gs.SUMMARY_MARKER) < p.index("<|im_start|>user")
    assert "You are a coding agent." in p          # base prompt not clobbered


def test_notes_do_not_stack(gs):
    # Self-amplifying failure: stacked notes consume the window they exist to
    # protect, and nothing errors.
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["- new fact"])
    prior = [{"role": "system",
              "content": "You are a coding agent.\n\n%s\n- old fact" % gs.SUMMARY_MARKER}]
    p, dropped, fits, overhead = gs.build_windowed(prior + convo(20)[1:], max_tokens=64)
    assert p.count(gs.SUMMARY_MARKER) == 1
    assert "old fact" not in p and "new fact" in p
    # "Not in the output" is only half of replace-don't-stack, and the half
    # that also holds when the old note is simply thrown away -- _apply_note
    # discards it whatever the summariser was fed. The contract is that it is
    # RE-SUMMARISED together with the newly evicted turns, so it has to have
    # reached the summariser.
    assert "- old fact" in gs.ENGINE.calls[0]["prompt"]
    assert "You are a coding agent." in p          # the base is not the note


def grown(msgs, n=1):
    """`msgs` as a stateless client resends it `n` exchanges later: the same
    history verbatim, plus the assistant's answer and the next question."""
    out = list(msgs)
    for i in range(n):
        out.append({"role": "assistant", "content": "answer%d %s" % (i, "fill " * 60)})
        out.append({"role": "user", "content": "followup%d %s" % (i, "pad " * 60)})
    return out


def test_the_previous_note_is_carried_forward_for_a_stateless_client(gs):
    # The documented mechanism -- "a later eviction re-summarises the previous
    # note together with the newly evicted turns" -- could not happen for any
    # real client. The note goes into the PROMPT; no response returns it, and
    # clients are told to resend their history verbatim, so the next request
    # carried no marker and was summarised from scratch, from the last few
    # thousand characters of what it evicted. A fact stated before that tail
    # was gone for good. Every earlier test of the prior-note branch PLANTED
    # the marker in the request, which is the one thing a client never sends.
    gs._CONTEXT_SIZE = SMALL
    first = convo(20)
    gs.ENGINE = StubEngine(chunks=["- token ZX-4417 lives in parser.py"])
    gs.build_windowed(first, max_tokens=64)
    evicted_first = gs._NOTE_STATE[1]

    gs.ENGINE = StubEngine(chunks=["- ZX-4417 in parser.py; followups started"])
    p, dropped, fits, overhead = gs.build_windowed(grown(first), max_tokens=64)
    assert fits and dropped > evicted_first
    asked = gs.ENGINE.calls[0]["prompt"]
    assert "- token ZX-4417 lives in parser.py" in asked, "the prior note"
    # ...together with ONLY what is newly on its way out. q0 was accounted for
    # by the first note; feeding it again is the from-scratch behaviour.
    assert "q0 " not in asked and "q18 " in asked
    assert p.count(gs.SUMMARY_MARKER) == 1
    assert "followups started" in p and "lives in parser.py" not in p
    # The record moved on to what THIS note covers. (<= dropped, not ==: the
    # re-fit with the note in place may drop a further turn, which no note
    # covers yet -- the next request finds it among its newly evicted ones.)
    assert evicted_first < gs._NOTE_STATE[1] <= dropped
    assert gs._NOTE_STATE[2] == "- ZX-4417 in parser.py; followups started"


def test_an_unchanged_eviction_reuses_the_note_without_an_npu_call(gs):
    # Same turns on their way out, same note: there is nothing new to
    # summarise. Re-summarising anyway spent an NPU call AND reset the dialog
    # (commit=False), so the request it was helping re-prefilled from nothing.
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["- a note"])
    first = gs.build_windowed(convo(20), max_tokens=64)
    gs.ENGINE = StubEngine(chunks=["- must not be asked for"])
    again = gs.build_windowed(convo(20), max_tokens=64)
    assert gs.ENGINE.calls == []
    assert again[0] == first[0], "byte-identical, so the resident KV still matches"
    assert again[3] == 0, "no NPU time was spent, so none is billed"


def test_another_conversation_never_inherits_the_note(gs):
    # Keyed by CONTENT -- the digest of the very turns the note summarises --
    # so a match cannot hand one conversation another's facts.
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["- secret from conversation A"])
    gs.build_windowed(convo(20), max_tokens=64)

    gs.ENGINE = StubEngine(chunks=["- facts of conversation B"])
    other = convo(20, system="You are a different agent.")
    other[1] = {"role": "user", "content": "a different opening " + "pad " * 60}
    p, dropped, fits, overhead = gs.build_windowed(other, max_tokens=64)
    assert "conversation A" not in gs.ENGINE.calls[0]["prompt"]
    assert "conversation A" not in p and "conversation B" in p


def test_a_changed_upstream_note_is_not_answered_from_the_stored_one(gs, capsys):
    # The stored note ABSORBED whatever note the client's own system turn
    # carried (a second genie_server in front of this one), so that note is
    # part of what the stored one stands for. If the upstream note changes, the
    # stored one is a summary of something the client no longer says.
    gs._CONTEXT_SIZE = 700

    def with_upstream(note):
        return convo(20, system="You are a coding agent.\n\n%s\n%s"
                     % (gs.SUMMARY_MARKER, note))

    gs.ENGINE = StubEngine(chunks=["- merged with upstream A"])
    gs.build_windowed(with_upstream("- upstream A"), max_tokens=64)
    gs.ENGINE = StubEngine(chunks=["- merged with upstream B"])
    p, dropped, fits, overhead = gs.build_windowed(
        with_upstream("- upstream B"), max_tokens=64)
    asked = gs.ENGINE.calls[0]["prompt"]
    assert "- upstream B" in asked and "upstream A" not in asked
    assert "merged with upstream B" in p and "upstream A" not in p
    # From scratch both times: nothing stored stands for these turns under the
    # new upstream note, so ALL of them were summarised again, not zero.
    said = [ln for ln in capsys.readouterr().out.splitlines() if "summarised" in ln]
    assert len(said) == 2 and all(
        "summarised %d evicted" % gs._NOTE_STATE[1] in ln for ln in said)


def test_a_failed_resummary_keeps_the_note_it_already_had(gs):
    # A summary must never break a request -- and with a note in hand, failing
    # to extend it is not a reason to throw it away. The turns it covers are
    # still gone from the prompt and it is still true of them.
    gs._CONTEXT_SIZE = SMALL
    first = convo(20)
    gs.ENGINE = StubEngine(chunks=["- the first note"])
    gs.build_windowed(first, max_tokens=64)
    state = gs._NOTE_STATE

    class Boom(StubEngine):
        def query(self, *a, **k):
            raise RuntimeError("npu wedged")

    gs.ENGINE = Boom()
    p, dropped, fits, overhead = gs.build_windowed(grown(first), max_tokens=64)
    assert fits and "- the first note" in p and overhead == 0
    # ...and the record is NOT advanced: it still covers only what the first
    # note covers, so the next request tries the uncovered turns again.
    assert gs._NOTE_STATE == state


def test_a_summary_cut_short_by_an_abort_is_not_remembered(gs, capsys):
    # The watchdog aborts a stalled summarisation like any other generation,
    # and the engine reports ABORTED as an ordinary "stop" with whatever text
    # had come back. While a note lasted one request that cost one prompt a
    # poor note. Now the note is REMEMBERED as standing for the turns it was
    # made from: they are never offered to the summariser again, and the
    # fragment rides forward as `prior` for the life of the conversation.
    gs._CONTEXT_SIZE = SMALL
    first = convo(20)
    gs.ENGINE = StubEngine(chunks=["- user asked", " about pars"])
    gs.ENGINE.query_aborted = True
    p, dropped, fits, overhead = gs.build_windowed(first, max_tokens=64)
    assert fits and dropped, "a summary never breaks the request"
    assert "about pars" not in p and gs.SUMMARY_MARKER not in p
    assert overhead == 0
    assert gs._NOTE_STATE is None, "nothing is remembered on the fragment's behalf"
    assert "cut short by an abort" in capsys.readouterr().out
    # The next request, with a healthy summariser: the SAME turns are offered
    # again, from the transcript, with no fragment in front of them.
    gs.ENGINE = StubEngine(chunks=["- a whole note"])
    p, _dropped, fits, _overhead = gs.build_windowed(grown(first), max_tokens=64)
    assert fits and "- a whole note" in p
    assert "about pars" not in gs.ENGINE.calls[0]["prompt"], "fed forward as prior"
    # EVERY evicted turn went to the summariser, not just the two new ones:
    # the count in the log line is evicted-less-covered, and nothing was
    # covered.
    assert ("summarised %d evicted" % gs._NOTE_STATE[1]) in capsys.readouterr().out


def test_an_aborted_resummary_keeps_the_note_it_already_had(gs):
    # The same with a note in hand: an aborted roll-up is a failed one, so
    # the good note stays and the record is not advanced -- exactly what
    # test_a_failed_resummary_... pins for a raise.
    gs._CONTEXT_SIZE = SMALL
    first = convo(20)
    gs.ENGINE = StubEngine(chunks=["- the first note"])
    gs.build_windowed(first, max_tokens=64)
    state = gs._NOTE_STATE
    gs.ENGINE = StubEngine(chunks=["- the first note, and a fragm"])
    gs.ENGINE.query_aborted = True
    p, _dropped, fits, _overhead = gs.build_windowed(grown(first), max_tokens=64)
    assert fits and "- the first note" in p and "a fragm" not in p
    assert gs._NOTE_STATE == state


def test_the_real_engine_says_when_a_summary_was_aborted(gs):
    # The stub above only says what it is told to. This is the real engine
    # with the watchdog's own call landing mid-generation: Genie returns
    # ABORTED, the finish is an ordinary "stop", text HAS come back -- and
    # `aborted` is the one thing that tells the two apart.
    from test_engine_kv import FakeLib
    lib = FakeLib(chunks=["- user asked", " about pars"], status=1)
    gs.ENGINE = gs.GenieEngine(lib, object())
    lib.on_query = lambda: gs.ENGINE.signal_abort(any_turn=True, stalled=True)
    assert gs._summarize_turns([{"role": "user", "content": "real content " * 30}]) \
        == (None, 0)
    assert lib.queries == 1 and lib.signals == [0x01], "it ran, and was cut"
    # ...and an unaborted one is still a note.
    lib.on_query, lib.status = None, 0
    note, _tokens = gs._summarize_turns(
        [{"role": "user", "content": "real content " * 30}])
    assert note == "- user asked about pars"
    # query() reports it on a raise as well: the flag is the turn's, not the
    # return value's.
    res = {}
    lib.status = -1
    lib.on_query = lambda: gs.ENGINE.signal_abort(any_turn=True)
    with pytest.raises(RuntimeError):
        gs.ENGINE.query("p", lambda t: None, result=res)
    assert res == {"aborted": True}


def test_a_system_prompt_that_quotes_the_marker_keeps_its_tail(gs):
    # The marker used to be the bare prose "[earlier context]", split on
    # wherever it appeared: a system prompt that merely QUOTED it lost
    # everything after the quote on its first eviction, cut out as a "previous
    # note" and fed to the summariser. A note is only a note as the LAST block
    # of the system turn, the marker on a line of its own.
    gs._CONTEXT_SIZE = 700
    # The hardest quote to tell from a note: the marker ENDS its line, so what
    # follows it looks exactly like a note body. What it does not do is START
    # the line, which everything _apply_note writes does.
    quoting = ("You maintain genie_server. Its retained notes are headed %s\n"
               "Never delete CHANGELOG.md." % gs.SUMMARY_MARKER)
    assert gs._prior_note([{"role": "system", "content": quoting}]) == ""
    gs.ENGINE = StubEngine(chunks=["- a fact"])
    p, dropped, fits, overhead = gs.build_windowed(
        convo(20, system=quoting), max_tokens=64)
    assert fits and dropped
    assert "Never delete CHANGELOG.md." in p, "the tail of the client's prompt"
    assert "CHANGELOG" not in gs.ENGINE.calls[0]["prompt"], "fed as a prior note"
    assert p.count(gs.SUMMARY_MARKER) == 2         # the quote, and the real note
    assert p.index("Never delete") < p.rindex(gs.SUMMARY_MARKER) < p.index("- a fact")


def test_only_the_last_block_of_the_system_turn_is_a_note(gs):
    m = gs.SUMMARY_MARKER
    # The shape _apply_note writes, and the only one honoured.
    assert gs._split_note("base\n\n%s\n- fact" % m) == ("base", "- fact")
    assert gs._split_note("%s\n- fact" % m) == ("", "- fact")
    # Mid-line is prose ABOUT the marker, not a note.
    assert gs._split_note("see %s\n- not a note" % m) == ("see %s\n- not a note" % m, "")
    # A quote early on AND a real note at the end: the last one is the note.
    both = "quoting %s here\nrule\n\n%s\n- fact" % (m, m)
    assert gs._split_note(both) == ("quoting %s here\nrule" % m, "- fact")
    assert gs._split_note("plain") == ("plain", "")
    assert gs._split_note("") == ("", "")


def test_a_client_with_no_system_prompt_keeps_the_default_one(gs):
    # The renderer falls back to the template's default_system only when the
    # system text is EMPTY, and a system turn holding a note is not. So the
    # first eviction swapped "You are a helpful AI assistant." for the note
    # alone -- the model lost its system prompt exactly when the conversation
    # got long. The old test called _apply_note and never rendered the result.
    gs._CONTEXT_SIZE = SMALL
    default = gs.TEMPLATE.default_system
    assert default, "the fallback template has one"
    for msgs in (convo(20)[1:],                                        # none
                 [{"role": "system", "content": ""}, *convo(20)[1:]]):   # empty
        gs.ENGINE = StubEngine(chunks=["- a fact"])
        assert default in gs.TEMPLATE.build(msgs), "before eviction"
        p, dropped, fits, overhead = gs.build_windowed(msgs, max_tokens=64)
        assert fits and dropped and "- a fact" in p
        assert default in p, "and after it"
        assert p.index(default) < p.index(gs.SUMMARY_MARKER)


def test_the_note_lands_after_every_leading_system_message(gs):
    # Folded into the FIRST of two system messages it would sit mid-turn once
    # the renderer joined them -- where _split_note, which only looks at the
    # end, would never find it again, and notes would stack after all.
    gs._CONTEXT_SIZE = 700
    msgs = convo(20)
    msgs.insert(1, {"role": "system", "content": "Reminder: answer in French."})
    gs.ENGINE = StubEngine(chunks=["- a fact"])
    p, dropped, fits, overhead = gs.build_windowed(msgs, max_tokens=64)
    assert fits and dropped
    assert (p.index("You are a coding agent.") < p.index("answer in French.")
            < p.index(gs.SUMMARY_MARKER) < p.index("- a fact")
            < p.index("<|im_start|>user"))
    assert p.count("<|im_start|>system") == 1
    merged = gs._apply_note(msgs[:2], "n1")
    assert [m["role"] for m in merged] == ["system"]
    assert gs._prior_note(merged) == "n1"
    assert gs._prior_note(gs._apply_note(merged, "n2")) == "n2"      # replaced


# --- a system message that is NOT at the front is a turn ------------------
# The system turn is what eviction anchors, and it is the LEADING system
# messages only. Every role=system message used to be hoisted into it, which
# cost nothing while the renderer discarded all but the first -- and became
# unbounded, unevictable growth the day their words reached the prompt.

def reminded(exchanges, chars=400):
    """The client shape that broke: a system reminder kept in the history
    after every exchange."""
    msgs = [{"role": "system", "content": "You are a coding agent."}]
    for i in range(exchanges):
        msgs.append({"role": "user", "content": "q%d" % i})
        msgs.append({"role": "assistant", "content": "a%d" % i})
        msgs.append({"role": "system", "content": "reminder %d: %s" % (i, "r" * chars)})
    msgs.append({"role": "user", "content": "final"})
    return msgs


@pytest.mark.parametrize("summarize", [False, True])
def test_accumulating_system_messages_are_evicted_like_any_other_turn(gs, summarize):
    # Measured against the version that anchored them all, at this window:
    # 18 exchanges kept 3 of 37 real turns (the reminders had the rest of the
    # window), and from 20 on NOTHING fitted -- a 400 telling the client to
    # "send a shorter message", repeated on every later request, because the
    # client resends the same history and the reminders were all unevictable.
    gs._CONTEXT_SIZE = 2048
    gs.ENGINE = StubEngine(chunks=["- a fact"])
    for exchanges in (20, 25, 200):
        gs._NOTE_STATE = None
        msgs = reminded(exchanges)
        p, dropped, fits, _overhead = gs.build_windowed(
            msgs, max_tokens=64, summarize=summarize)
        assert fits, "%d exchanges is a 400 forever" % exchanges
        assert gs._tok_count(p) <= 2048 - 64 - gs.WINDOW_MARGIN
        assert 0 < dropped < len(msgs) - 2, "evicted some, not everything"
        assert "You are a coding agent." in p, "the system turn is still anchored"
        assert "reminder 0:" not in p, "the oldest reminder went with its turns"
        last = exchanges - 1
        assert "q%d" % last in p and "reminder %d:" % last in p and "final" in p
        # In order, in place, each as a block of its own.
        assert (p.index("q%d<" % last) < p.index("reminder %d:" % last)
                < p.index("final"))
        assert "<|im_start|>system\nreminder %d: " % last in p


def test_an_evicted_system_message_reaches_the_summariser(gs):
    # "Never silent": what leaves the window is condensed, and a reminder is
    # not exempt from that because of its role.
    gs._CONTEXT_SIZE = 2048
    gs.ENGINE = StubEngine(chunks=["- a fact"])
    _p, dropped, _fits, _overhead = gs.build_windowed(reminded(25), max_tokens=64)
    assert dropped
    # The transcript keeps the TAIL of what was evicted, so it is the last
    # evicted reminder that is sure to be in it, under its own role.
    assert re.search(r"^system: reminder \d+: r+$", gs.ENGINE.calls[0]["prompt"], re.M)


def test_what_is_kept_never_opens_on_a_system_message(gs):
    # One left at the head of what survives would be rendered straight after
    # the system turn -- part of the leading run again: folded in BEHIND the
    # note (which is only found as the last thing there) and anchored on the
    # second pass. It goes with the turns it sat among.
    msgs = reminded(30)
    for budget in range(250, 1900, 50):
        gs._TOK_CACHE.clear()
        p, kept, ev, fits = gs._fit(list(msgs), None, True, budget)
        if not ev:
            continue
        assert kept[0]["role"] != "system", budget
        assert ev + kept == msgs[1:], "every message is kept or evicted, in order"
    gs._CONTEXT_SIZE = 2048
    gs.ENGINE = StubEngine(chunks=["- a fact"])
    p, dropped, fits, _overhead = gs.build_windowed(msgs, max_tokens=64)
    assert fits and dropped
    turn = p[:p.index(gs.TEMPLATE.sys_suf)]
    assert turn.endswith(gs.SUMMARY_MARKER + "\n- a fact"), "the note is LAST in it"


def test_a_trailing_reminder_does_not_make_the_question_evictable(gs):
    # A client that appends its reminder AFTER the newest turn ends every
    # request on a system message. The unit that is never evicted has to
    # start at the user's turn, not at the reminder about it.
    msgs = convo(20)
    msgs.append({"role": "system", "content": "Reminder: answer in French."})
    p, kept, ev, fits = gs._fit(list(msgs), None, True, 120)
    assert fits and ev
    assert kept == msgs[-2:], "the question and its reminder, and nothing else"
    assert p.index("final question") < p.index("answer in French.")
    # ...and when even that does not fit, it is reported whole.
    msgs[-2] = {"role": "user", "content": "W" * 8000}
    p, kept, ev, fits = gs._fit(list(msgs), None, True, 120)
    assert not fits and kept == msgs[-2:]


def test_summarisation_cost_is_reported(gs):
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["- a note"])
    p, dropped, fits, overhead = gs.build_windowed(convo(20), max_tokens=64)
    # Spending NPU time invisibly is the same failure as reporting zero tokens
    # for a tool turn.
    assert overhead == len("- a note") // 4 > 0


def test_no_eviction_means_no_npu_call_and_no_cost(gs):
    gs.ENGINE = StubEngine(chunks=["should not run"])
    p, dropped, fits, overhead = gs.build_windowed(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        max_tokens=32)
    assert (dropped, overhead, gs.ENGINE.calls) == (0, 0, [])


def test_summariser_failure_degrades_to_plain_eviction(gs):
    gs._CONTEXT_SIZE = SMALL

    class Boom(StubEngine):
        def query(self, *a, **k):
            raise RuntimeError("npu wedged")

    gs.ENGINE = Boom()
    p, dropped, fits, overhead = gs.build_windowed(convo(20), max_tokens=64)
    # A summary is a nice-to-have; it must never break the request.
    assert fits and gs.SUMMARY_MARKER not in p and overhead == 0


def test_a_note_too_big_to_fit_degrades_to_plain_eviction(gs):
    # The other half of that guarantee, and the harder half: the summariser
    # SUCCEEDS, but the note pushes the render back over budget. Returning the
    # second pass's unfitting prompt would overflow the window on a request
    # that was already fitting before we tried to help it.
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["should not run"])
    plain, plain_dropped, _, _ = gs.build_windowed(
        convo(20), max_tokens=64, summarize=False)

    gs.ENGINE = StubEngine(chunks=["- " + "note " * 900])
    p, dropped, fits, overhead = gs.build_windowed(convo(20), max_tokens=64)
    assert fits and gs.SUMMARY_MARKER not in p
    assert (p, dropped) == (plain, plain_dropped)   # the plain-eviction result
    # The NPU call really happened, so reporting it as free would hide real
    # decode time from the caller -- the same failure as billing zero for a
    # tool turn.
    assert gs.ENGINE.calls and overhead > 0


def test_summarisation_does_not_claim_the_resident_kv(gs):
    # It leaves text in the dialog that is NOT the caller's conversation, so
    # recording it as the resident prefix would be a false claim.
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["- note"])
    gs.build_windowed(convo(20), max_tokens=64)
    assert [c["commit"] for c in gs.ENGINE.calls] == [False]


def test_summarisation_can_be_disabled(gs):
    gs._CONTEXT_SIZE = SMALL
    gs.ENGINE = StubEngine(chunks=["- note"])
    p, dropped, fits, overhead = gs.build_windowed(
        convo(20), max_tokens=64, summarize=False)
    assert fits and dropped and gs.SUMMARY_MARKER not in p and gs.ENGINE.calls == []


def test_the_note_creates_a_system_turn_when_there_is_none(gs):
    """A client that sends no system prompt is entirely ordinary.

    The note rides in the system turn because that is the one thing eviction
    never touches -- so with no system turn to fold into, one has to be made.
    Every fixture here supplies one (convo() always does), so this branch had
    never run: the summary would have been spent on an NPU call and then
    dropped on the floor.
    """
    out = gs._apply_note([{"role": "user", "content": "hi"}], "the note")
    assert out[0]["role"] == "system"
    assert gs.SUMMARY_MARKER in out[0]["content"] and "the note" in out[0]["content"]
    # ...and the original turns survive after it, in order.
    assert [m["role"] for m in out] == ["system", "user"]
    # The new turn OPENS with the template's default system prompt: the
    # renderer only supplies that default when the system text is empty, which
    # this turn no longer is. See the rendered version of this further down.
    assert out[0]["content"] == "%s\n\n%s\nthe note" % (
        gs.TEMPLATE.default_system, gs.SUMMARY_MARKER)


def test_summarisation_degrades_when_the_note_comes_back_empty(gs):
    """The summariser SUCCEEDING with useless output, not raising.

    The docs promise a summary is never allowed to break a request, and that
    promise was only tested for the exception path. A model that returns
    whitespace -- or a think block that strips to nothing -- takes a different
    route: no error, no note, and the caller must fall back to plain eviction.
    """
    gs.ENGINE = StubEngine(chunks=["   \n  "])
    assert gs._summarize_turns([{"role": "user", "content": "real content"}]) == (None, 0)

    # The think block the docstring promises, which whitespace never was: only
    # a real <think>...</think> reaches the _THINK_RE strip, and a note that is
    # nothing BUT reasoning has to come out as no note at all -- it would ride
    # in the system turn and be re-read on every later request.
    gs.ENGINE = StubEngine(chunks=["<think>only ", "reasoning</think>\n"])
    assert gs._summarize_turns([{"role": "user", "content": "real content"}]) == (None, 0)
    assert len(gs.ENGINE.calls) == 1, "the call was made; its output was empty"
    # ...and a note that FOLLOWS the reasoning survives without it.
    gs.ENGINE = StubEngine(chunks=["<think>hmm</think>\n- the fact"])
    assert gs._summarize_turns([{"role": "user", "content": "real content"}]) == (
        "- the fact", len("- the fact") // 4)

    # And the other early exit: nothing worth summarising in the first place,
    # which must not spend an NPU call at all.
    gs.ENGINE = StubEngine(chunks=["a note"])
    assert gs._summarize_turns([{"role": "user", "content": "   "}]) == (None, 0)
    assert gs.ENGINE.calls == [], "paid for a summary of nothing"


def test_an_evicted_tool_call_is_named_in_the_transcript(gs):
    """Evicting a tool-using conversation is the COMMON case for an agent.

    Without the annotation the note says the assistant went silent exactly where
    it acted, so the model re-reads a file it already read -- which is the
    behaviour summarisation exists to prevent.
    """
    t = gs._transcript([{"role": "assistant", "content": "looking",
                         "tool_calls": [{"function": {"name": "read_file"}}]}])
    assert t == "assistant: looking [called read_file]"

    # The shape that actually dominates: a turn whose whole output WAS the call,
    # so content is empty and the annotation is the only thing left to keep.
    t2 = gs._transcript([{"role": "assistant", "content": "",
                          "tool_calls": [{"function": {"name": "ls"}}]}])
    assert t2 == "assistant: [called ls]"


# --- the summarisation call has to fit the window too ---------------------

@pytest.mark.parametrize("ctx,expected", [
    (8192, 6000),      # the shipped windows: unchanged, so the recall
    (4096, 6000),      # measurements in the docs still describe them
    (1024, 1536),
    (512, 768),
])
def test_the_transcript_cap_follows_the_window(gs, ctx, expected):
    # It was a literal 6000 characters -- ~1500 tokens, larger than the whole
    # window of a bundle compiled at 1024, where the summarisation prompt
    # overflowed on EVERY eviction: a failed NPU call, paid for, that HEALTH
    # counts toward "failing".
    gs._CONTEXT_SIZE = ctx
    assert gs._transcript_cap() == expected
    long = [{"role": "user", "content": "x" * 20000}]
    assert len(gs._transcript(long)) == expected
    assert gs._transcript(long).endswith("x"), "the TAIL is what is kept"
    assert gs._transcript(long, 0) == "", "text[-0:] is the whole string"


def test_the_summarisation_prompt_is_measured_before_it_is_sent(gs):
    # A character cap is a guess -- dense text runs to a token per character --
    # so the prompt actually built is counted against the window less the
    # note's own cap and the margin, and the transcript halved until it fits.
    gs._CONTEXT_SIZE = 400
    budget = 400 - gs.summary_token_cap() - gs.WINDOW_MARGIN
    msgs = [{"role": "user", "content": "w" * 5000}]
    gs.ENGINE = StubEngine(chunks=["- note"])
    # A prior note big enough that the first-guess transcript does not fit.
    note, cost = gs._summarize_turns(msgs, prior="p" * 500)
    assert note == "- note"
    sent = gs.ENGINE.calls[0]["prompt"]
    assert gs._tok_count(sent) <= budget
    assert "w" * gs._MIN_TRANSCRIPT_CHARS in sent and "w" * gs._transcript_cap() not in sent


def test_a_summarisation_prompt_that_cannot_fit_is_never_sent(gs, capsys):
    gs._CONTEXT_SIZE = 400
    gs.ENGINE = StubEngine(chunks=["- note"])
    msgs = [{"role": "user", "content": "w" * 5000}]
    assert gs._summarize_turns(msgs, prior="p" * 4000) == (None, 0)
    assert gs.ENGINE.calls == [], "a doomed call is still a paid-for call"
    assert "NOT summarised" in capsys.readouterr().out, "never silent"


def test_an_unparseable_metadata_claims_nothing_and_does_not_raise(gs, tmp_path):
    # The sibling reader for genie_config.json has exactly this test; without it
    # here, a corrupt metadata.json takes a different and unverified route. It
    # runs at startup, so raising would turn a bad bundle file into a server
    # that will not boot.
    (tmp_path / "metadata.json").write_text('{"genie": {"context_lengths"',
                                            encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONTEXT_LENGTHS = None
    assert gs.read_context_lengths() == []
    # [] rather than None: cached as "asked and got nothing", so the broken file
    # is not re-opened and re-parsed on every request that reaches /props.
    assert gs._CONTEXT_LENGTHS == []


def test_a_metadata_without_a_genie_block_claims_nothing(gs, tmp_path):
    # Valid JSON, no genie key -- a bundle from a tool that does not write one.
    # Must read as "unknown", never as single-length, which would put a
    # SINGLE-length warning in front of a bundle nobody can characterise.
    (tmp_path / "metadata.json").write_text('{"other": 1}', encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONTEXT_LENGTHS = None
    gs._POLL_MATCHES = [(False, "QnnHtp.poll")]
    assert gs.read_context_lengths() == []
    assert gs.bundle_config_warnings() == []


# --- the chat template the bundle itself carries ---------------------------
# The production prompt path, and until these ran, nothing exercised it: the
# `gs` fixture pins BUNDLE_DIR="" before calling load_chat_template, so all of
# the suite renders with the generic Qwen fallback. Every prompt byte a real
# operator sends comes from the branch below instead.

_TEMPLATE_FILE = {"system_prefix": "[S]", "system_suffix": "[/S]",
                  "user_prefix": "[U]", "user_suffix": "[/U]",
                  "assistant_prefix": "[A]", "assistant_suffix": "[/A]",
                  "default_system_prompt": "Bundle default."}


def test_the_bundles_own_chat_template_is_what_renders(gs, tmp_path, capsys):
    # A bundle's delimiters are its own. Serving generic ChatML to a model
    # trained on something else is wrong in a way no request can report: every
    # answer is a little worse and every response is a 200.
    (tmp_path / "metadata.json").write_text(
        json.dumps({"genie": {"chat_template": _TEMPLATE_FILE}}), encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    t = gs.load_chat_template()
    assert t.build([{"role": "user", "content": "hi"}], thinking=True) == \
        "[S]Bundle default.[/S][U]hi[/U][A]"
    assert capsys.readouterr().out == "", "a bundle that works is not news"


@pytest.mark.parametrize("broken,why", [
    ('{"genie": {"chat_template"', "truncated mid-write"),
    ('{"genie": {"chat_template": {"system_prefix": "[S]"}}}', "missing keys"),
    ('{"genie": {"chat_template": "{%- for m in messages %}"}}', "the Jinja string"),
    ('{"genie": "nope"}', "genie is not a block"),
    ('[]', "not even an object"),
])
def test_an_unusable_chat_template_falls_back_and_says_so(gs, tmp_path, capsys,
                                                          broken, why):
    # Every shape a hand-edited or half-written metadata.json actually takes,
    # each one measured raising out of here: JSONDecodeError, KeyError,
    # TypeError, AttributeError. This runs at startup, from main(), before
    # anything is listening -- so raising turns a bad bundle file into a server
    # that will not boot, which is the argument its sibling readers were given
    # (read_context_lengths has this same test, one section up) and it applies
    # here too.
    (tmp_path / "metadata.json").write_text(broken, encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    assert gs.load_chat_template().usr_pre == "<|im_start|>user\n", why
    # ...and unlike those siblings, it SAYS so. The substitution is otherwise
    # invisible: generic ChatML renders every bundle plausibly, so the wrong
    # delimiters are served forever with nothing in the log to explain why the
    # answers are slightly off.
    out = capsys.readouterr().out
    assert "WARNING" in out and str(tmp_path / "metadata.json") in out, why


def test_a_bundle_with_no_chat_template_falls_back_quietly(gs, tmp_path, capsys):
    # Not a failure -- the documented fallback. A warning here would put a
    # scary line in front of every start on a bundle whose metadata simply does
    # not carry the key, which is how a warning that does matter gets ignored.
    meta = tmp_path / "metadata.json"
    gs.BUNDLE_DIR = str(tmp_path)
    for quiet in ('{"genie": {"context_lengths": [4096]}}',   # no chat_template
                  '{"other": 1}',                             # no genie block
                  '{"genie": null}'):                         # read as absent, like
        meta.write_text(quiet, encoding="utf-8")              # read_context_lengths
        assert gs.load_chat_template().usr_pre == "<|im_start|>user\n", quiet
        assert capsys.readouterr().out == "", quiet
    # ...and with no metadata.json in the bundle dir at all.
    meta.unlink()
    assert gs.load_chat_template().usr_pre == "<|im_start|>user\n"
    assert capsys.readouterr().out == ""


# --- config clamps ---------------------------------------------------------

@pytest.mark.parametrize("ctx,requested,expected", [
    (4096, 192, 192),      # default: unclamped
    (4096, 10000, 512),    # absurd request bounded by the window
    (512, 192, 64),
    (128, 192, 32),        # floor keeps the note useful on a tiny bundle
])
def test_summary_note_is_clamped_to_the_window(gs, ctx, requested, expected):
    gs._CONTEXT_SIZE = ctx
    gs.SUMMARY_MAX_TOKENS = requested
    assert gs.summary_token_cap() == expected


def test_inflight_cap_is_floored_and_never_disabled(gs, monkeypatch):
    # "0 disables the cap" bought no concurrency on a single-flight NPU -- it
    # let unbounded threads park on the engine lock.
    for value in ("0", "-5", "1", "3"):
        monkeypatch.setenv("GENIE_MAX_INFLIGHT", value)
        importlib.reload(gs)
        assert gs.MAX_INFLIGHT == max(1, int(value))
        assert gs._INFLIGHT is not None


def test_context_size_is_read_once(gs, tmp_path, monkeypatch):
    # It sits on the request path; re-reading and JSON-parsing a file per
    # request is blocking I/O for a value that cannot change while loaded.
    cfg = tmp_path / "genie_config.json"
    cfg.write_text('{"dialog": {"context": {"size": 8192}}}', encoding="utf-8")
    monkeypatch.setenv("GENIE_BUNDLE_DIR", str(tmp_path))
    importlib.reload(gs)
    assert gs.read_context_size() == 8192
    cfg.unlink()                                  # file gone
    assert gs.read_context_size() == 8192         # still served from cache


def test_context_size_falls_back_when_unreadable(gs, tmp_path, monkeypatch):
    monkeypatch.setenv("GENIE_BUNDLE_DIR", str(tmp_path))   # no config at all
    importlib.reload(gs)
    # /props answering with a default beats failing to start over a field it
    # only needs for a metadata endpoint.
    assert gs.read_context_size() == 4096


# --- bundle configuration --------------------------------------------------
# The two settings that decide most of this server's throughput were the two
# nothing read. `poll: true` ships as the default, busy-waits on ~2.7 host
# cores while idle and costs up to 36% of decode; a single-length bundle runs
# every token against its whole compiled window and is 2-3x slower on short
# prompts than a multi-length one at the SAME n_ctx. Both were left to whoever
# remembered the docs, in a server that otherwise derives and asserts every
# fact it depends on -- placement, port, Hexagon arch.

def test_poll_true_is_called_out_with_what_it_costs(gs):
    gs._POLL_MATCHES = [(True, "dialog.engine.backend.QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [512, 4096]
    warnings = gs.bundle_config_warnings()
    assert len(warnings) == 1
    w = warnings[0]
    assert "WARNING" in w
    assert "dialog.engine.backend.QnnHtp.poll" in w, "name the key to edit"
    # What it COSTS is the point -- a warning that only says "this is wrong"
    # gets skipped. The concurrency figure this used to assert (1.45x turning
    # into a 0.78x net loss) was refuted: both poll settings are a gain, and
    # the flag gives away about a quarter of the win. Assert the cost is named,
    # and that the retired number stays retired.
    assert "36%" in w, "name the decode cost"
    assert "concurrency" in w, "name the second cost"
    assert "0.78x" not in w, "refuted figure must not come back"


def test_a_correctly_configured_bundle_says_nothing(gs):
    # A startup warning that fires on a healthy bundle trains the reader to
    # skip it, which is how the real one goes unread.
    gs._POLL_MATCHES = [(False, "QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
    assert gs.bundle_config_warnings() == []


def test_a_missing_poll_key_is_a_note_not_a_warning(gs):
    # Absent is not the same as false: the shipped default is true, but we did
    # not read it here and must not claim we did.
    gs._POLL_MATCHES = []
    gs._CONTEXT_LENGTHS = [512, 4096]
    out = gs.bundle_config_warnings()
    assert len(out) == 1 and out[0].startswith("note:")
    assert "WARNING" not in out[0]


def test_a_single_length_bundle_is_flagged(gs):
    gs._POLL_MATCHES = [(False, "QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [8192]
    out = gs.bundle_config_warnings()
    assert len(out) == 1 and "SINGLE-length" in out[0]
    assert "--context-lengths" in out[0], "must say how to fix it"


def test_an_unreadable_bundle_claims_nothing_about_its_graphs(gs):
    # metadata.json missing -> [] -> no claim either way. Guessing "multi"
    # would be the same silent-degradation this server refuses elsewhere.
    gs._POLL_MATCHES = [(False, "QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = []
    assert gs.bundle_config_warnings() == []


def test_both_problems_are_reported_together(gs):
    # One restart should surface everything wrong, not the first thing wrong.
    gs._POLL_MATCHES = [(True, "QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [16384]
    assert len(gs.bundle_config_warnings()) == 2


def test_the_poll_key_is_found_wherever_the_sdk_nests_it(gs):
    # Searched rather than addressed by a fixed path: the QnnHtp block has moved
    # between QAIRT releases and this server supports more than one, so a path
    # that is right for 2.45 and absent on the next reads as "not set" -- the
    # wrong answer for a flag whose default is the expensive one.
    cfg = {"dialog": {"engine": {"backend": {"QnnHtp": {"poll": True}}}}}
    assert gs._find_all(cfg, "poll") == [
        (True, "dialog.engine.backend.QnnHtp.poll")]


def test_a_nested_false_is_found_and_not_mistaken_for_absent(gs):
    # The discriminating case for any "did we find it" guard: `false` is a
    # value, not a miss. Reading it as absent would report a correctly
    # configured bundle as unconfigured and warn about nothing.
    cfg = {"dialog": {"engine": {"backend": {"QnnHtp": {"poll": False}}}}}
    value, where = gs._pick_poll(gs._find_all(cfg, "poll"))
    assert value is False
    assert where == "dialog.engine.backend.QnnHtp.poll"


def test_key_search_reports_absence_rather_than_a_default(gs):
    assert gs._find_all({"dialog": {"engine": {}}}, "poll") == []
    assert gs._pick_poll([]) == (None, None)


def test_key_search_descends_through_lists(gs):
    cfg = {"dialog": {"engine": {"backends": [{"type": "cpu"},
                                              {"type": "QnnHtp", "poll": True}]}}}
    matches = gs._find_all(cfg, "poll")
    assert len(matches) == 1
    assert matches[0][0] is True and "[1]" in matches[0][1]


def test_the_qnnhtp_copy_wins_over_a_shallower_one(gs):
    # Depth-first-first-match preferred whichever came first in insertion order,
    # which on this config is the nested one regardless of which is correct.
    # The BLOCK is always called QnnHtp; only its nesting moves.
    cfg = {"poll": False, "dialog": {"backend": {"QnnHtp": {"poll": True}}}}
    assert gs._pick_poll(gs._find_all(cfg, "poll")) == (
        True, "dialog.backend.QnnHtp.poll")


def test_the_shallowest_wins_when_nothing_is_qnnhtp_qualified(gs):
    cfg = {"poll": True, "a": {"b": {"poll": False}}}
    assert gs._pick_poll(gs._find_all(cfg, "poll")) == (True, "poll")


def test_conflicting_poll_keys_are_reported_not_silently_resolved(gs):
    # The picker is a heuristic. One that resolves a real conflict without
    # saying so is how a wrong value reaches /props looking authoritative.
    gs._POLL_MATCHES = [(True, "poll"), (False, "dialog.QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [512, 4096]
    out = gs.bundle_config_warnings()
    assert any("conflicting values" in w for w in out)
    assert any("dialog.QnnHtp.poll" in w for w in out), "name where each was found"


def test_agreeing_duplicate_poll_keys_are_not_noise(gs):
    # Same value twice is odd but harmless; warning on it trains the reader to
    # skip the warning that matters.
    gs._POLL_MATCHES = [(False, "poll"), (False, "dialog.QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [512, 4096]
    assert gs.bundle_config_warnings() == []


# --- poll truthiness, not identity ----------------------------------------
# `poll is True` was identity-strict, so every truthy non-True encoding fell
# through it AND through the `is None` branch and produced no warning at all --
# silently accepting the one setting this check exists to catch. JSON `true`
# parses to Python True, but 1 and "true" are valid config and both busy-wait.

@pytest.mark.parametrize("value", [True, 1, "true", "yes"])
def test_every_truthy_poll_encoding_warns(gs, value):
    gs._POLL_MATCHES = [(value, "QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [512, 4096]
    out = gs.bundle_config_warnings()
    assert len(out) == 1 and out[0].startswith("WARNING")
    assert "2.7 host" in out[0], "must say what it costs, not just that it is set"


def test_the_warning_reports_the_value_as_written(gs):
    # It used to assert "= true" regardless. Naming the actual value is what
    # lets the reader find it in the file.
    gs._POLL_MATCHES = [(1, "QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [512, 4096]
    assert "QnnHtp.poll = 1" in gs.bundle_config_warnings()[0]


@pytest.mark.parametrize("value", [False, 0])
def test_falsy_poll_encodings_stay_silent(gs, value):
    # The correctly-configured case. Warning here would train the reader to
    # skip the warning that matters.
    gs._POLL_MATCHES = [(value, "QnnHtp.poll")]
    gs._CONTEXT_LENGTHS = [512, 4096]
    assert gs.bundle_config_warnings() == []


# --- context_lengths must be a LIST ---------------------------------------
# A bare string is iterable, so "8192" was read element-wise into [8, 1, 9, 2]:
# a single-length bundle reported as MULTI-length with four invented graph
# lengths, in the startup banner and in /props.genie.context_lengths. A router
# reading that concludes the bundle is fast on short prompts when it is 2-3x
# slower -- precisely the misreport the field was added to prevent.

@pytest.mark.parametrize("bad", ["8192", 8192, {"a": 1}, None])
def test_a_non_list_context_lengths_claims_nothing(gs, tmp_path, bad):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"genie": {"context_lengths": bad}}), encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONTEXT_LENGTHS = None
    assert gs.read_context_lengths() == []


def test_a_real_list_is_read_normally(gs, tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"genie": {"context_lengths": [512, 1024, 8192]}}),
        encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONTEXT_LENGTHS = None
    assert gs.read_context_lengths() == [512, 1024, 8192]


def test_poll_is_found_in_a_real_genie_config(gs, tmp_path):
    # Everything above pins _POLL_MATCHES by hand, so the search had never
    # actually run against a file. A read that silently returns nothing reports
    # a busy-waiting bundle as unconfigured -- 2.7 idle cores and up to 36% of
    # decode, unremarked.
    (tmp_path / "genie_config.json").write_text(json.dumps({
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {"version": 1, "size": 4096, "n-vocab": 151936},
            "sampler": {"version": 1, "temp": 0.8},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {"version": 1, "type": "QnnHtp",
                            "QnnHtp": {"version": 1, "spill-fill-bufsize": 0,
                                       "use-mmap": True, "poll": True,
                                       "pos-id-dim": 64}},
                "model": {"version": 1, "type": "binary"},
            },
        }}), encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._POLL_MATCHES = None
    assert gs.read_poll_setting() == (
        True, "dialog.engine.backend.QnnHtp.poll")


def test_an_unparseable_genie_config_claims_nothing_and_does_not_raise(gs, tmp_path):
    # This runs during startup. Raising on a malformed bundle file would turn a
    # performance note into a server that will not boot -- the opposite of
    # warn-never-refuse.
    (tmp_path / "genie_config.json").write_text(
        '{"dialog": {"engine": ', encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._POLL_MATCHES = None
    assert gs.read_poll_setting() == (None, None)
    assert gs._POLL_MATCHES == []      # absent, not "unread" -- see the cache


def test_a_missing_genie_config_claims_nothing(gs, tmp_path):
    # Absent is not false: the shipped default is true, but we did not read it
    # and must not report a bundle as correctly configured on that basis.
    gs.BUNDLE_DIR = str(tmp_path)               # no config at all
    gs._POLL_MATCHES = None
    assert gs.read_poll_setting() == (None, None)
    assert gs.bundle_config_warnings()[0].startswith("note:")


def test_the_poll_read_is_cached(gs, tmp_path):
    # It is called per warning line and from /props; re-opening and JSON-parsing
    # the bundle config each time is blocking I/O for a value that cannot change
    # while the bundle is loaded.
    cfg = tmp_path / "genie_config.json"
    cfg.write_text('{"dialog": {"QnnHtp": {"poll": true}}}', encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._POLL_MATCHES = None
    assert gs.read_poll_setting() == (True, "dialog.QnnHtp.poll")
    cfg.unlink()                                            # file gone
    assert gs.read_poll_setting() == (True, "dialog.QnnHtp.poll")


def test_a_bogus_length_list_does_not_claim_multi_length(gs, tmp_path):
    # The consequence that actually reaches a router, asserted end to end.
    (tmp_path / "metadata.json").write_text(
        json.dumps({"genie": {"context_lengths": "8192"}}), encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONTEXT_LENGTHS = None
    gs._POLL_MATCHES = [(False, "QnnHtp.poll")]
    assert len(gs.read_context_lengths()) != 4, "must not invent four graphs"
    assert gs.bundle_config_warnings() == [], "unreadable is not single-length"


# --- the sampler's repetition penalty --------------------------------------
# Genie's token-penalty block is OPTIONAL and every field in it defaults to 0,
# so a bundle without it samples at temp 0.8 with nothing suppressing a loop.
# Qualcomm's reference config for this stack sets it; the AI Hub export path
# emits the same sampler WITHOUT it. The symptom is not a crash or an error --
# the model answers normally and then repeats one paragraph to max_tokens --
# so nothing about it is visible until someone reads the output.

def _sampler(**pen):
    s = {"version": 1, "seed": 42, "temp": 0.8, "top-k": 40, "top-p": 0.95}
    if pen:
        s["token-penalty"] = dict({"version": 1}, **pen)
    return s


def test_a_sampler_with_no_penalty_block_is_flagged(gs):
    gs._SAMPLER = _sampler()
    out = [w for w in gs.bundle_config_warnings() if "token-penalty" in w]
    assert len(out) == 1 and out[0].startswith("WARNING")
    assert "penalize-last-n" in out[0], "must give the block to paste, not a hint"
    assert "per request" in out[0], "sampling binds at create -- say so"


def test_penalties_applied_to_an_empty_window_are_flagged(gs):
    # The subtle one. Someone sets repetition-penalty, restarts, sees no change
    # and concludes the knob is broken -- when penalize-last-n=0 means it was
    # read and then applied to nothing.
    gs._SAMPLER = _sampler(**{"penalize-last-n": 0, "repetition-penalty": 2.3})
    out = [w for w in gs.bundle_config_warnings() if "penalize-last-n" in w]
    assert len(out) == 1 and "empty window" in out[0]


def test_a_window_with_every_penalty_zero_is_flagged(gs):
    gs._SAMPLER = _sampler(**{"penalize-last-n": 64, "repetition-penalty": 0.0,
                              "presence-penalty": 0.0, "frequency-penalty": 0.0})
    assert [w for w in gs.bundle_config_warnings() if "every penalty" in w]


def test_a_correctly_penalised_sampler_is_silent(gs):
    # The fixture default. A warning that fires on a good bundle trains the
    # reader to skip the one that matters.
    assert gs.bundle_config_warnings() == []


def test_an_unreadable_config_makes_no_claim_about_the_sampler(gs):
    # {} is "could not open it", which the poll note already reports. Saying it
    # twice in different words reads as two separate problems.
    gs._SAMPLER = {}
    gs._POLL_MATCHES = []
    assert not [w for w in gs.bundle_config_warnings() if "token-penalty" in w]


@pytest.mark.parametrize("pen,expected", [
    (None, "absent"),
    ({"penalize-last-n": 0, "repetition-penalty": 2.3}, "no-window"),
    ({"repetition-penalty": 2.3}, "no-window"),           # key omitted == 0
    ({"penalize-last-n": 64}, "all-zero"),
    ({"penalize-last-n": 64, "repetition-penalty": 0, "presence-penalty": 0,
      "frequency-penalty": 0}, "all-zero"),
    ({"penalize-last-n": 64, "repetition-penalty": 2.3}, "ok"),
    ({"penalize-last-n": 64, "presence-penalty": 0.7}, "ok"),   # any one is enough
    ({"penalize-last-n": 64, "frequency-penalty": 0.8}, "ok"),
])
def test_penalty_state_classification(gs, pen, expected):
    s = _sampler(**pen) if pen is not None else _sampler()
    assert gs.sampler_penalty_state(s) == expected


def test_a_junk_penalty_value_does_not_crash_startup(gs):
    # A malformed config must not take the server down before it can say what
    # is wrong with it -- float("abc") would raise inside the check itself.
    gs._SAMPLER = _sampler(**{"penalize-last-n": "sixty-four",
                              "repetition-penalty": None})
    assert gs.sampler_penalty_state(gs._SAMPLER) == "no-window"
    assert gs.bundle_config_warnings()


def test_a_non_dict_penalty_is_treated_as_absent(gs):
    assert gs.sampler_penalty_state({"token-penalty": "yes"}) == "absent"
    assert gs.sampler_penalty_state({}) == "absent"
    assert gs.sampler_penalty_state(None) == "absent"


def test_the_restore_baseline_survives_an_unreadable_config(gs):
    # read_default_sampler now reads THROUGH read_sampler, so the two cannot
    # disagree -- but its fallback has to stay non-empty: it is what a
    # per-request override is restored TO, and {} would restore nothing.
    gs._SAMPLER = {}
    assert gs.read_default_sampler() == {"version": 1}
    gs._SAMPLER = _sampler(**{"penalize-last-n": 64, "repetition-penalty": 2.3})
    assert gs.read_default_sampler()["temp"] == 0.8


# --- a bundle that is not there is not a misconfigured bundle -------------
# The first line a new user saw running the server directly was a note about
# `poll` in a genie_config.json they did not have, printed in front of the real
# error naming the env vars to set. It reads as "your bundle is misconfigured"
# when the answer is "you have not pointed me at one". read_sampler already
# refuses to make a claim about a file it could not open; this is the same rule
# applied to the config as a whole.

def test_no_bundle_config_means_no_warnings_at_all(gs, tmp_path):
    gs.BUNDLE_DIR = str(tmp_path)          # a real dir, but no genie_config.json
    gs._CONFIG_PRESENT = None
    gs._POLL_MATCHES = None
    gs._SAMPLER = None
    gs._CONTEXT_LENGTHS = None
    assert gs.config_present() is False
    assert gs.bundle_config_warnings() == [], (
        "a note about a file that does not exist is noise in front of the real "
        "error")


def test_an_unset_bundle_dir_is_silent_too(gs, tmp_path, monkeypatch):
    # The literal first-run case: nothing exported at all -- run from a
    # directory that HAS a genie_config.json, because that is the only place
    # the difference shows. os.path.join("", "genie_config.json") is the bare
    # filename, i.e. a path relative to the working directory, so this test
    # used to pass only because pytest's CWD happened not to contain one; from
    # a directory that did, the server read a stray config, believed it, and
    # warned about it in front of the "set GENIE_BUNDLE_DIR" exit.
    (tmp_path / "genie_config.json").write_text(json.dumps({
        "dialog": {"context": {"size": 1234},
                   "engine": {"backend": {"QnnHtp": {"poll": True}}},
                   "sampler": {"version": 1, "temp": 0.8}}}), encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "genie": {"context_lengths": [8192],
                  "chat_template": {"system_prefix": "[S]", "system_suffix": "[/S]",
                                    "user_prefix": "[U]", "user_suffix": "[/U]",
                                    "assistant_prefix": "[A]",
                                    "assistant_suffix": "[/A]"}}}),
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    gs.BUNDLE_DIR = ""
    gs._CONFIG_PRESENT = None
    gs._POLL_MATCHES = None
    gs._SAMPLER = None
    gs._CONTEXT_LENGTHS = None
    gs._CONTEXT_SIZE = None
    assert gs.config_present() is False
    assert gs.bundle_config_warnings() == []
    # Every reader, not only the presence check: each of them joined its
    # filename onto the same empty string.
    assert gs.read_poll_setting() == (None, None)
    assert gs.read_sampler() == {}
    assert gs.read_context_lengths() == []
    assert gs.read_context_size() == 4096          # the default, not the 1234
    assert gs.config_parse_error() == ""
    assert gs.load_chat_template().usr_pre == "<|im_start|>user\n"   # the fallback


def test_a_config_that_IS_present_still_warns(gs, tmp_path):
    # The check must not have silenced the real thing. A bundle that exists and
    # is misconfigured is exactly what these warnings are for.
    cfg = tmp_path / "genie_config.json"
    cfg.write_text(json.dumps({
        "dialog": {"engine": {"backend": {"QnnHtp": {"poll": True}}},
                   "sampler": {"version": 1, "temp": 0.8}}}), encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONFIG_PRESENT = None
    gs._POLL_MATCHES = None
    gs._SAMPLER = None
    gs._CONTEXT_LENGTHS = []
    assert gs.config_present() is True
    out = gs.bundle_config_warnings()
    assert any("poll" in w and "busy-waits" in w for w in out)
    assert any("token-penalty" in w for w in out)


def test_a_present_but_corrupt_config_still_warns(gs, tmp_path):
    # PRESENT and unparseable is a different case from ABSENT: the operator has
    # a bundle and something is wrong with it, so a warning is owed -- and it
    # has to be the TRUE one. This used to assert only that something was
    # printed, and what was printed was "note: no `poll` key found in
    # genie_config.json": a claim about the contents of a file that had never
    # been read, pointing the operator at a missing key when the problem is a
    # syntax error.
    (tmp_path / "genie_config.json").write_text("{not json", encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONFIG_PRESENT = None
    gs._POLL_MATCHES = None
    gs._SAMPLER = None
    gs._CONTEXT_LENGTHS = []
    assert gs.config_present() is True
    out = gs.bundle_config_warnings()
    assert len(out) == 1 and out[0].startswith("WARNING")
    assert "could not parse genie_config.json" in out[0]
    assert "line 1 column 2" in out[0], "the parser's position IS the fix"
    assert "SKIPPED" in out[0], "an unrun check must not read as a passed one"
    assert not any("no `poll` key" in w for w in out), (
        "nothing can be said about the keys of a file that did not parse")


def test_a_corrupt_config_does_not_hide_the_bundle_shape(gs, tmp_path):
    # context_lengths lives in metadata.json, a different file. An unparseable
    # genie_config.json skips the checks that READ genie_config.json and no
    # others -- one restart should still surface everything that can be known.
    (tmp_path / "genie_config.json").write_text("{not json", encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONFIG_PRESENT = None
    gs._POLL_MATCHES = None
    gs._SAMPLER = None
    gs._CONTEXT_LENGTHS = [8192]
    out = gs.bundle_config_warnings()
    assert len(out) == 2
    assert "could not parse" in out[0] and "SINGLE-length" in out[1]


def test_a_parseable_or_absent_config_has_no_parse_error(gs, tmp_path):
    # "" for both: absence is config_present()'s case, and reporting it here as
    # well would put two warnings in front of one missing file.
    gs.BUNDLE_DIR = str(tmp_path)
    assert gs.config_parse_error() == ""                       # absent
    cfg = tmp_path / "genie_config.json"
    cfg.write_text('{"dialog": {}}', encoding="utf-8")
    gs._CONFIG_PARSE_ERROR = None
    assert gs.config_parse_error() == ""                       # parses
    # Cached like every reader beside it: it sits in front of the startup
    # warnings, and the answer cannot change while the bundle is loaded.
    cfg.write_text("{not json", encoding="utf-8")
    assert gs.config_parse_error() == ""
    gs._CONFIG_PARSE_ERROR = None
    assert "line 1 column 2" in gs.config_parse_error()
    # Not-UTF-8 is "there and not JSON" as well, and is a ValueError too.
    cfg.write_bytes(b'{"dialog": "\xff\xfe"}')
    gs._CONFIG_PARSE_ERROR = None
    assert gs.config_parse_error() != ""


def test_config_presence_is_cached_like_the_other_readers(gs, tmp_path):
    # It sits in front of every warning at startup; re-statting per call would
    # be filesystem I/O for a value that cannot change while the server runs.
    cfg = tmp_path / "genie_config.json"
    cfg.write_text("{}", encoding="utf-8")
    gs.BUNDLE_DIR = str(tmp_path)
    gs._CONFIG_PRESENT = None
    assert gs.config_present() is True
    cfg.unlink()
    assert gs.config_present() is True, "served from cache after the first stat"
