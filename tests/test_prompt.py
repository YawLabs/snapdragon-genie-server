"""Prompt rendering and tool-call parsing.

Every case here corresponds to something that actually broke or was actually
ambiguous, not to a hypothetical.
"""

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
    p = gs.TEMPLATE.build(msgs, tools=tools)
    assert "# Tools" in p and '"name": "read_file"' in p
    assert "<tool_call>" in p and "<tool_response>" in p
    # Consecutive tool results share ONE user turn, per the bundle's template.
    assert p.count("<|im_start|>user") == 2
    assert p.endswith("<|im_start|>assistant\n")


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
    second = first + [{"role": "assistant", "content": "Hello."},
                      {"role": "user", "content": "again"}]
    assert gs.TEMPLATE.build(second, thinking=False).startswith(committed)
