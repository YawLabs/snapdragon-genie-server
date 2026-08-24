"""Context-window eviction and the summarisation that rides on it.

Overflow is a hard GenieDialog_query failure, not a truncation, and Genie has
no sliding-window mode -- so everything here is what stands between a long
conversation and a 500.

Tests that go through build_windowed pin a small n_ctx explicitly, because
build_windowed derives its budget from the window and a modest conversation
fits comfortably inside the real 4096.
"""

import importlib
import random
import re

import pytest

from conftest import StubEngine, convo

SMALL = 500


def linear_fit(g, messages, tools, thinking, budget):
    """The pre-bisection implementation, kept as the equivalence oracle."""
    sysm = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    ev = []
    while True:
        p = g.TEMPLATE.build(sysm + rest, tools=tools, thinking=thinking)
        if g._tok_count(p) <= budget:
            return p, rest, ev, True
        if len(rest) <= 1:
            return p, rest, ev, False
        ev.append(rest.pop(0))
        while len(rest) > 1 and rest[0].get("role") == "tool":
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
        msgs.append({"role": "user", "content": "final"})
        budget = rng.choice([300, 800, 1500, 3900])
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
    assert fits
    depth = 0
    for m in re.finditer(r"<(tool_call|tool_response)>", p):
        if m.group(1) == "tool_call":
            depth += 1
        else:
            assert depth > 0, "tool_response with no preceding tool_call"
            depth -= 1


def test_single_oversized_message_reports_unfittable(gs):
    p, kept, ev, fits = gs._fit(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "W" * 20000}],
        None, True, 400)
    assert not fits          # caller owes the client a 400, not a doomed query


def test_overflow_message_names_the_numbers(gs):
    # Names the real numbers, not just "too big" -- the client needs to know
    # what to shrink.
    msg = gs._overflow_msg("W" * 8000, 64)
    assert str(gs.read_context_size()) in msg and "64" in msg


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
