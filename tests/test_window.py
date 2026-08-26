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
# cores while idle and costs up to 55% of decode; a single-length bundle runs
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
    assert "1.45x" in w and "0.78x" in w, "the concurrency reversal is the point"


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
    # a busy-waiting bundle as unconfigured -- 2.7 idle cores and up to 55% of
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
