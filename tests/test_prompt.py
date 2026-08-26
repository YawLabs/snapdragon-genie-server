"""Prompt rendering and tool-call parsing.

Every case here corresponds to something that actually broke or was actually
ambiguous, not to a hypothetical.
"""

import json

import pytest

TEXT_BLOCK = [{"type": "text", "text": "hello"}]


# --- content flattening ----------------------------------------------------
# `content` may be a LIST of blocks on both APIs -- OpenAI SDKs emit that shape
# by default. It broke the renderer once (TypeError, dropped connection), was
# fixed there, and then broke again in the summarisation helpers because the
# fix stopped at the first consumer. These cover every consumer at once.

@pytest.mark.parametrize("role", ["user", "system", "assistant", "tool"])
def test_array_content_renders_for_every_role(gs, role):
    msgs = [{"role": "user", "content": "seed"}, {"role": role, "content": TEXT_BLOCK}]
    assert "hello" in gs.TEMPLATE.build(msgs)


def test_content_flattener_handles_every_shape(gs):
    f = gs._content_text
    assert f("plain") == "plain"
    assert f(None) == ""
    assert f(TEXT_BLOCK) == "hello"
    assert f([{"type": "image"}, {"type": "text", "text": "a"}]) == "a"   # image ignored
    assert f([{"type": "tool_result", "content": TEXT_BLOCK}]) == "hello"  # nested


def test_content_flattener_accepts_bare_strings_in_the_list(gs):
    # Some clients put BARE STRINGS in the list instead of text blocks. They
    # are not dicts, so a flattener that only understands blocks drops them
    # silently -- the model is then asked to answer a prompt the user's words
    # are missing from, which reads as a bad answer rather than as a bug.
    f = gs._content_text
    assert f(["one ", "two"]) == "one two"
    # A MIXED list must concatenate in the order the client sent it, not
    # grouped by kind: reordering scrambles the sentence.
    assert f(["A", {"type": "text", "text": "B"}, "C"]) == "ABC"
    # ...and the ignored shapes stay ignored with a bare string beside them --
    # the string branch must not become "stringify anything".
    assert f(["A", {"type": "image"}, "B"]) == "AB"
    assert f([{"type": "tool_result", "content": ["x", "y"]}]) == "xy"  # via recursion


def test_anthropic_text_delegates_to_the_shared_flattener(gs):
    # One implementation, so the two endpoints cannot drift apart again.
    assert gs._anthropic_text(TEXT_BLOCK) == gs._content_text(TEXT_BLOCK)


def test_summarisation_helpers_accept_array_content(gs):
    # These raised AttributeError ('list' has no 'strip' / 'split') until the
    # flattener reached them.
    assert gs._transcript([{"role": "user", "content": TEXT_BLOCK}]).endswith("hello")
    out = gs._apply_note([{"role": "system", "content": TEXT_BLOCK}], "note")
    assert "hello" in out[0]["content"] and "note" in out[0]["content"]


def test_prior_note_found_when_system_content_is_a_list(gs):
    # `MARKER in [block, ...]` is a LIST MEMBERSHIP test: it returns False
    # instead of raising, so the prior note goes unfound and notes stack.
    sysmsg = [{"role": "system",
               "content": [{"type": "text",
                            "text": "base %s\n- old fact" % gs.SUMMARY_MARKER}]}]
    assert gs._prior_note(sysmsg) == "- old fact"


# --- tool-call parsing -----------------------------------------------------

CALL_JSON = '{"name": "read_file", "arguments": {"path": "c.yaml"}}'
EXPECTED = [{"name": "read_file", "arguments": {"path": "c.yaml"}}]


@pytest.mark.parametrize("tag", ["tool_call", "function_call", "tool_use"])
def test_every_wrapper_the_model_actually_emits(gs, tag):
    # <tool_call> is the trained tag, but with thinking suppressed the model
    # improvises <function_call>; all three were observed on real hardware.
    text, calls = gs.parse_tool_calls("<%s>\n%s\n</%s>" % (tag, CALL_JSON, tag))
    assert calls == EXPECTED and text == ""


def test_bare_json_call_is_accepted(gs):
    assert gs.parse_tool_calls(CALL_JSON)[1] == EXPECTED


def test_text_and_call_are_split(gs):
    text, calls = gs.parse_tool_calls("Let me look.\n<tool_call>%s</tool_call>" % CALL_JSON)
    assert text == "Let me look." and calls == EXPECTED


# OpenAI's actual wire format: `arguments` arrives as a STRING containing JSON,
# not as an object. Built with json.dumps so the escaping is unambiguous.
STR_ARGS_CALL = json.dumps({"name": "read_file", "arguments": '{"path": "c.yaml"}'})


def test_string_arguments_are_parsed_into_an_object(gs):
    # Handing the string straight back makes every client json.loads it a
    # second time. The wrapped path and the bare-blob path do this parse in
    # SEPARATE code, so one can be fixed and the other left behind -- both are
    # asserted against the same EXPECTED as the object form, which is the
    # actual contract: a client cannot tell the two wire shapes apart.
    assert gs.parse_tool_calls("<tool_call>%s</tool_call>" % STR_ARGS_CALL)[1] == EXPECTED
    assert gs.parse_tool_calls(STR_ARGS_CALL)[1] == EXPECTED          # bare, no tags
    assert gs._bare_tool_calls(STR_ARGS_CALL) == EXPECTED             # and the helper itself


def test_unparseable_string_arguments_pass_through_raw(gs):
    # Pins what the bare `except: pass` actually does -- `arguments` comes back
    # as the RAW STRING, so a client doing args["path"] gets TypeError, not a
    # KeyError it could handle. Both paths behave the same. If that is ever
    # tightened (to {}, or to refusing the call) this is the test that says so.
    junk = json.dumps({"name": "read_file", "arguments": "path=c.yaml"})
    raw = [{"name": "read_file", "arguments": "path=c.yaml"}]
    assert gs.parse_tool_calls("<tool_call>%s</tool_call>" % junk)[1] == raw
    assert gs._bare_tool_calls(junk) == raw


def test_empty_string_arguments_stay_an_empty_string(gs):
    # The pass-through case a WELL-FORMED client can produce, not model junk: a
    # zero-argument call rendered as "". json.loads("") raises, so it falls
    # through the same except and the caller gets "" where {} is what the call
    # means -- unlike an omitted `arguments`, which the wrapped path defaults.
    empty = json.dumps({"name": "now", "arguments": ""})
    assert gs.parse_tool_calls("<tool_call>%s</tool_call>" % empty)[1] == [
        {"name": "now", "arguments": ""}]
    assert gs.parse_tool_calls('<tool_call>{"name": "now"}</tool_call>')[1] == [
        {"name": "now", "arguments": {}}]


@pytest.mark.parametrize("payload", [
    '{"answer": 42}',                                   # ordinary JSON answer
    '{"name": "bob"}',                                  # name alone is not a call
    'Here is some JSON: %s' % CALL_JSON,                # prose around it
    '[1, 2, 3]',
    'plain text',
    '<tool_call>{not json}</tool_call>',                # malformed payload
    '<tool_call>%s</function_call>' % CALL_JSON,        # mismatched open/close
])
def test_things_that_must_not_parse_as_calls(gs, payload):
    # The bare-JSON path is a deliberate loosening. Without pinned negatives a
    # future widening silently turns ordinary answers into tool calls.
    assert gs.parse_tool_calls(payload)[1] == []


def test_malformed_call_stays_visible(gs):
    # Dropping it would leave the caller with a mystery empty response.
    text, calls = gs.parse_tool_calls("oops <tool_call>{not json}</tool_call>")
    assert calls == [] and "<tool_call>" in text


# --- tool round-trip -------------------------------------------------------

def test_tool_conversation_renders_in_template_shape(gs):
    tools = [{"type": "function", "function": {
        "name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "read main.py"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": '{"path": "main.py"}'}}]},
        {"role": "tool", "content": "print('hi')"},
        {"role": "tool", "content": "second result"},
    ]
    # thinking passed EXPLICITLY: this test is about the tool round-trip
    # shape, not the server's reasoning policy, and leaving it implicit made
    # it silently depend on that policy -- it broke the day the default
    # flipped.
    p = gs.TEMPLATE.build(msgs, tools=tools, thinking=True)
    assert "# Tools" in p and '"name": "read_file"' in p
    assert "<tool_call>" in p and "<tool_response>" in p
    # Consecutive tool results share ONE user turn, per the bundle's template.
    assert p.count("<|im_start|>user") == 2
    # The template's own prefix, not a copy of it -- a duplicated literal
    # here is a second place to update when the bundle's template changes.
    assert p.endswith(gs.TEMPLATE.asst_pre)

    # And the shape the server ACTUALLY ships now: the same open assistant
    # turn, with the closed think block prefilled after it.
    off = gs.TEMPLATE.build(msgs, tools=tools, thinking=False)
    assert off.endswith(gs.TEMPLATE.asst_pre + gs._NO_THINK)
    # Twice, not once: the prefill goes in front of every assistant turn,
    # HISTORY included, because that is what was actually sent when those
    # turns were generated. Rendering history without it would diverge from
    # the dialog's resident KV by exactly those bytes and silently defeat
    # reuse -- see the note in ChatML.build.
    assert off.count(gs._NO_THINK) == 2, (
        "one for the assistant history turn, one for the open turn")
    assert p.count(gs._NO_THINK) == 0


def test_the_render_default_follows_the_server_policy(gs):
    """Omitting `thinking` means "whatever the server does", not `True`.

    These two defaults disagreed for a while: the signatures said True while
    THINKING_DEFAULT said False. No production caller was affected -- both
    handlers pass it explicitly -- but every eviction test omitted it, so the
    suite was exercising a prompt shape the server would never emit.
    """
    msgs = [{"role": "user", "content": "hi"}]
    gs.THINKING_DEFAULT = False
    assert gs.TEMPLATE.build(msgs) == gs.TEMPLATE.build(msgs, thinking=False)
    gs.THINKING_DEFAULT = True
    assert gs.TEMPLATE.build(msgs) == gs.TEMPLATE.build(msgs, thinking=True)


def test_build_windowed_default_follows_the_server_policy(gs):
    # Same contract one layer up, where the handlers actually call in.
    msgs = [{"role": "user", "content": "hi"}]
    for default in (False, True):
        gs.THINKING_DEFAULT = default
        implicit = gs.build_windowed(msgs, max_tokens=32)
        explicit = gs.build_windowed(msgs, thinking=default, max_tokens=32)
        assert implicit[0] == explicit[0]


def test_thinking_off_prefills_a_closed_think_block(gs):
    on = gs.TEMPLATE.build([{"role": "user", "content": "hi"}], thinking=True)
    off = gs.TEMPLATE.build([{"role": "user", "content": "hi"}], thinking=False)
    assert off == on + "<think>\n\n</think>\n\n"


def test_history_renders_the_way_it_was_generated(gs):
    # With thinking suppressed the prefill IS in the dialog's KV, because we
    # sent it. Re-rendering history without it diverges from resident state by
    # exactly those bytes and silently defeats KV reuse.
    first = [{"role": "user", "content": "hi"}]
    committed = gs.TEMPLATE.build(first, thinking=False) + "Hello."
    second = [*first,
              {"role": "assistant", "content": "Hello."},
              {"role": "user", "content": "again"}]
    assert gs.TEMPLATE.build(second, thinking=False).startswith(committed)
