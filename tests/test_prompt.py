"""Prompt rendering and tool-call parsing.

Every case here corresponds to something that actually broke or was actually
ambiguous, not to a hypothetical.
"""

import json

import pytest

from conftest import request

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


# Every shape `content` arrives in, with what it flattens to. ONE list, because
# two functions are held to it: the flattener itself, and the Anthropic wrapper
# that claims to be nothing more than a call to it.
CONTENT_SHAPES = [
    ("plain", "plain"),
    (None, ""),
    (TEXT_BLOCK, "hello"),
    ([{"type": "image"}, {"type": "text", "text": "a"}], "a"),        # image ignored
    ([{"type": "tool_result", "content": TEXT_BLOCK}], "hello"),      # nested
    (["one ", "two"], "one two"),                                     # bare strings
    (["A", {"type": "text", "text": "B"}, "C"], "ABC"),               # mixed, in order
    (["A", {"type": "image"}, "B"], "AB"),
    ([{"type": "tool_result", "content": ["x", "y"]}], "xy"),         # via recursion
    ([{"type": "tool_use", "name": "f", "input": {}}], ""),           # not text
]


def test_content_flattener_handles_every_shape(gs):
    for shape, expected in CONTENT_SHAPES:
        assert gs._content_text(shape) == expected, shape


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


def test_a_bare_content_block_is_not_an_empty_turn(gs):
    # One wrapper short of the list above: the block sent as `content` ITSELF.
    # Clients write that by hand, and a tool_result's own content arrives that
    # way. It used to fall through to "" -- the question answered 200 over a
    # prompt it was missing from, which reads as the model being stupid rather
    # than as the server having dropped it.
    f = gs._content_text
    assert f({"type": "text", "text": "what is 2+2"}) == "what is 2+2"
    # Exactly the one-element list it means, so the two spellings cannot
    # disagree: a block that contributes nothing inside a list still
    # contributes nothing on its own.
    for block in ({"type": "text", "text": "hello"},
                  {"type": "image"},
                  {"type": "tool_use", "name": "f", "input": {}},
                  {"type": "tool_result", "content": TEXT_BLOCK},
                  {"nothing": "recognised"}):
        assert f(block) == f([block]), block
    # Through the renderer, including the role a tool result lands on: the
    # file body used to vanish while the <tool_call> that asked for it stayed,
    # which reads to a client as a tool that returned nothing.
    p = gs.TEMPLATE.build([{"role": "user", "content": {"type": "text", "text": "2+2"}},
                           {"role": "tool", "content": {"type": "text",
                                                        "text": "print('hi')"}}])
    assert "<|im_start|>user\n2+2<|im_end|>" in p
    assert "<tool_response>\nprint('hi')\n</tool_response>" in p
    # And the Anthropic wrapper, which claims to be nothing but this function
    # -- an Anthropic top-level `system` is routinely written as a bare block.
    assert gs._anthropic_text({"type": "text", "text": "be brief"}) == "be brief"


def test_content_that_cannot_be_rendered_refuses_instead_of_emptying_the_turn(gs, handler):
    # A number or a bool has no words in it and no honest text to stand in for
    # them: returning "" asks the model a blank question and answers 200, and
    # str(content) is the "stringify anything" rule the list branch already
    # refuses. Raising is what this file does with every other unusable field,
    # and do_POST's catch-all turns it into a 400 that names the type -- an
    # answer the client can act on.
    for value in (42, True, 3.5):
        with pytest.raises(TypeError, match="content"):
            gs._content_text(value)
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": 42}]})
    assert code == 400 and "content" in body["error"]["message"]
    assert gs.ENGINE.calls == [], "refused while the prompt was built, never sent"


def test_a_text_block_whose_text_is_not_a_string_refuses_on_both_legs(gs, handler):
    # "".join raises on a None, and every _content_text call site runs during
    # prompt build -- before a byte of response -- so this surfaces as a 400
    # and not as the dropped connection the docstring above is about.
    bad = [{"type": "text", "text": None}]
    with pytest.raises(TypeError):
        gs._content_text(bad)
    code, body, _h = request(gs, handler, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": bad}]})
    assert code == 400 and "TypeError" in body["error"]["message"]
    code, body, _h = request(gs, handler, "POST", "/v1/messages",
                             {"messages": [{"role": "user", "content": bad}],
                              "max_tokens": 16})
    assert code == 400 and body["type"] == "error"
    # BOTH spellings of the join, because there are two of them and only one
    # is the shared flattener: a message carrying TOOL blocks is built by
    # _anthropic_to_prompt's own inline join and returns before it ever
    # reaches _content_text, so hardening either one alone would leave the
    # other exactly where it was. That is the drift
    # test_anthropic_text_delegates_to_the_shared_flattener was written to
    # prevent and cannot reach, since it exercises _anthropic_text instead.
    with_result = [{"type": "tool_result", "content": "file body"},
                   {"type": "text", "text": None}]
    code, body, _h = request(gs, handler, "POST", "/v1/messages",
                             {"messages": [{"role": "user", "content": with_result}],
                              "max_tokens": 16})
    assert code == 400 and body["type"] == "error"
    assert gs.ENGINE.calls == [], "nothing generated over a prompt that never built"


def test_anthropic_text_delegates_to_the_shared_flattener(gs, monkeypatch):
    # One implementation, so the two endpoints cannot drift apart again. This
    # used to compare the two on a single text block, which is the one shape
    # any reimplementation gets right -- a second flattener that only knew
    # {"type": "text"} passed, and "delegates" was a name rather than a check.
    # Every shape, including the ones that broke once already...
    for shape, expected in CONTENT_SHAPES:
        assert gs._anthropic_text(shape) == expected, shape
    # ...and then the claim itself: it CALLS the shared one, and hands back
    # whatever that returns.
    seen = []
    monkeypatch.setattr(gs, "_content_text",
                        lambda content: seen.append(content) or "from the shared one")
    assert gs._anthropic_text(TEXT_BLOCK) == "from the shared one"
    assert seen == [TEXT_BLOCK]


def test_summarisation_helpers_accept_array_content(gs):
    # These raised AttributeError ('list' has no 'strip' / 'split') until the
    # flattener reached them.
    assert gs._transcript([{"role": "user", "content": TEXT_BLOCK}]).endswith("hello")
    out = gs._apply_note([{"role": "system", "content": TEXT_BLOCK}], "note")
    assert "hello" in out[0]["content"] and "note" in out[0]["content"]


def test_prior_note_found_when_system_content_is_a_list(gs):
    # `MARKER in [block, ...]` is a LIST MEMBERSHIP test: it returns False
    # instead of raising, so the prior note goes unfound and notes stack. The
    # note is in the shape _apply_note writes -- marker on its own line, last
    # in the turn -- because that is the only shape that IS a note.
    sysmsg = [{"role": "system",
               "content": [{"type": "text",
                            "text": "base\n\n%s\n- old fact" % gs.SUMMARY_MARKER}]}]
    assert gs._prior_note(sysmsg) == "- old fact"
    # Round trip: what _apply_note writes, _prior_note reads back -- through a
    # list-of-blocks system message too.
    written = gs._apply_note([{"role": "system", "content": TEXT_BLOCK}], "- a fact")
    assert gs._prior_note(written) == "- a fact"
    assert written[0]["content"].startswith("hello")


# --- more than one system message -----------------------------------------
# The renderer took the FIRST system message and skipped every one after it
# as "already folded into the system turn" -- so a second one never reached
# the model, with no log line and no error. Real clients send them: a trailing
# system reminder appended each turn, a framework's per-turn injection.
#
# Where one lands depends on where it SAT. The system messages a conversation
# opens with are the system turn -- one block, anchored against eviction. One
# that comes later is a turn: its own block, at its position, as the bundle's
# Jinja renders it. (Folding those in as well made the one unevictable turn
# grow with the conversation; tests/test_window.py has that half.)
#
# The expected prompts below are LITERALS, not rebuilt from TEMPLATE's fields:
# the engine reuses its KV on a byte-exact prefix match, so a renderer change
# that moves one byte of an ordinary conversation silently re-prefills every
# turn of every client. They were taken from the renderer as it stood before
# later system messages were given their own blocks.

_TOOLS_F = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
_PREAMBLE_F = (
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within <tools></tools> "
    "XML tags:\n<tools>\n"
    '{"type": "function", "function": {"name": "f", "parameters": {}}}\n'
    "</tools>\n\nFor each function call, return a json object with function "
    "name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')


def test_a_single_system_conversation_renders_byte_for_byte_as_before(gs):
    msgs = [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "read main.py"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": '{"path": "main.py"}'}}]},
            {"role": "tool", "content": "print('hi')"},
            {"role": "assistant", "content": "It prints hi."},
            {"role": "user", "content": "thanks"}]
    assert gs.TEMPLATE.build(msgs, thinking=False) == (
        "<|im_start|>system\nYou are a coding agent.<|im_end|>\n"
        "<|im_start|>user\nread main.py<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n"
        '{"name": "f", "arguments": {"path": "main.py"}}\n</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\nprint('hi')\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nIt prints hi.<|im_end|>\n"
        "<|im_start|>user\nthanks<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert gs.TEMPLATE.build(msgs, tools=_TOOLS_F, thinking=True) == (
        "<|im_start|>system\nYou are a coding agent.\n\n" + _PREAMBLE_F + "<|im_end|>\n"
        "<|im_start|>user\nread main.py<|im_end|>\n"
        "<|im_start|>assistant\n<tool_call>\n"
        '{"name": "f", "arguments": {"path": "main.py"}}\n</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\nprint('hi')\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\nIt prints hi.<|im_end|>\n"
        "<|im_start|>user\nthanks<|im_end|>\n"
        "<|im_start|>assistant\n")


def test_leading_system_messages_are_one_system_turn(gs):
    msgs = [{"role": "system", "content": "You are a coding agent."},
            {"role": "system", "content": ""},                  # contributes nothing
            {"role": "system", "content": [{"type": "text", "text": "Answer in French."}]},
            {"role": "user", "content": "hi"}]
    # One system turn, in order, a blank line between.
    assert gs.TEMPLATE.build(msgs, thinking=False) == (
        "<|im_start|>system\nYou are a coding agent.\n\nAnswer in French.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # The same with tools, where the system text shares its turn with the
    # tool preamble.
    assert gs.TEMPLATE.build(msgs, tools=_TOOLS_F, thinking=True) == (
        "<|im_start|>system\nYou are a coding agent.\n\nAnswer in French.\n\n"
        + _PREAMBLE_F + "<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n")
    assert gs._system_text(msgs) == "You are a coding agent.\n\nAnswer in French."


def test_a_later_system_message_is_its_own_block_where_it_sat(gs):
    msgs = [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": [{"type": "text", "text": "Answer in French."}]},
            {"role": "system", "content": ""},                  # renders nothing
            {"role": "user", "content": "again"}]
    assert gs.TEMPLATE.build(msgs, thinking=True) == (
        "<|im_start|>system\nYou are a coding agent.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nhello<|im_end|>\n"
        "<|im_start|>system\nAnswer in French.<|im_end|>\n"
        "<|im_start|>user\nagain<|im_end|>\n"
        "<|im_start|>assistant\n")
    # It is NOT part of the system turn: not for the renderer, and not for
    # the helpers that decide where a retained note goes and is found again.
    assert gs._system_text(msgs) == "You are a coding agent."
    noted = gs._apply_note(msgs, "- a fact")
    assert [m["role"] for m in noted] == ["system", "user", "assistant",
                                          "system", "system", "user"]
    assert noted[0]["content"].endswith("\n- a fact") and noted[1:] == msgs[1:]
    assert gs._prior_note(noted) == "- a fact"
    # With no system message at the front, the default prompt still opens the
    # conversation and the later one still sits where it was sent.
    p = gs.TEMPLATE.build(msgs[1:], thinking=True)
    assert p.startswith("<|im_start|>system\n" + gs.TEMPLATE.default_system + "<|im_end|>\n")
    assert "hello<|im_end|>\n<|im_start|>system\nAnswer in French.<|im_end|>\n" in p


def test_a_growing_conversation_with_reminders_still_extends_byte_for_byte(gs):
    # The KV-reuse invariant, for the client shape this exists for: the next
    # request's render must START WITH the previous one plus what the model
    # said. Folded into the system turn, each new reminder changed the FIRST
    # block of the prompt and every turn re-prefilled from zero.
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "q0"},
            {"role": "system", "content": "reminder 0"}]
    first = gs.TEMPLATE.build(msgs, thinking=False)
    msgs += [{"role": "assistant", "content": "a0"},
             {"role": "user", "content": "q1"},
             {"role": "system", "content": "reminder 1"}]
    assert gs.TEMPLATE.build(msgs, thinking=False).startswith(first + "a0")


def test_the_default_system_prompt_only_fills_an_empty_system_turn(gs):
    user = [{"role": "user", "content": "hi"}]
    default = gs.TEMPLATE.default_system
    assert default in gs.TEMPLATE.build(user)
    assert default in gs.TEMPLATE.build([{"role": "system", "content": ""}, *user])
    assert default not in gs.TEMPLATE.build([{"role": "system", "content": "mine"}, *user])


def test_a_developer_message_is_the_system_turn_it_means(gs):
    # OpenAI's current spelling of the system role, emitted by its SDKs. The
    # unknown-role fall-through rendered one as a USER turn, and _system_text
    # then found no system message at all -- so the template's default_system
    # was injected in FRONT of the agent's own instructions, contradicting
    # them with a prompt nobody sent.
    msgs = [{"role": "developer", "content": "You are a coding agent."},
            {"role": "user", "content": "hi"}]
    assert gs.TEMPLATE.build(msgs, thinking=True) == (
        "<|im_start|>system\nYou are a coding agent.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n")
    assert gs.TEMPLATE.default_system not in gs.TEMPLATE.build(msgs)
    assert gs._system_text(msgs) == "You are a coding agent."
    # The two spellings are one role, byte for byte -- with the tool preamble
    # sharing the turn too, which is where a second definition would show.
    spelled_system = [{"role": "system", "content": "You are a coding agent."}, *msgs[1:]]
    for tools in (None, _TOOLS_F):
        assert gs.TEMPLATE.build(msgs, tools=tools) == \
            gs.TEMPLATE.build(spelled_system, tools=tools)
    # And a later one is a turn where it sat, exactly as a later system
    # message is: its own block, evictable with the turns around it.
    later = [{"role": "user", "content": "hi"},
             {"role": "developer", "content": "Answer in French."},
             {"role": "user", "content": "again"}]
    assert "hi<|im_end|>\n<|im_start|>system\nAnswer in French.<|im_end|>\n" in \
        gs.TEMPLATE.build(later, thinking=True)


def test_a_role_nobody_recognises_is_still_a_turn(gs):
    # The other half of that decision, and deliberate: "developer" is the
    # system role because OpenAI says it is, but an unknown role is not
    # something to guess at. Rendering it as a user turn keeps the words in
    # the prompt -- dropping it would be the silent loss this file refuses --
    # and it is not read as an instruction, so the default system prompt
    # still applies.
    msgs = [{"role": "narrator", "content": "the door creaks"},
            {"role": "user", "content": "hi"}]
    p = gs.TEMPLATE.build(msgs, thinking=True)
    assert "<|im_start|>user\nthe door creaks<|im_end|>\n" in p
    assert p.startswith("<|im_start|>system\n" + gs.TEMPLATE.default_system)
    assert gs._system_text(msgs) == ""


# --- `tools` must be a list of objects -------------------------------------
# It was only ever tested for truthiness. A string is iterated by CHARACTER and
# a dict by KEY, and each element is json.dumps'd into the <tools> preamble as
# a "function signature": `"tools": "ab"` answered 200 over a prompt offering
# the model the functions "a" and "b".

@pytest.mark.parametrize("tools,ok", [
    ([], True),                                          # "no tools" is a list
    ([{"type": "function", "function": {"name": "f"}}], True),
    ("ab", False),
    ({"k": 1, "z": 2}, False),
    (["read_file"], False),                              # a list, of the wrong thing
    ([{"name": "f"}, None], False),
    (7, False),
    (True, False),
    (None, False),
])
def test_only_a_list_of_objects_is_a_tool_list(gs, tools, ok):
    assert gs._is_tool_list(tools) is ok


def test_what_an_unchecked_tools_value_renders_as(gs):
    # Why the check above has to happen at the door: the renderer itself has no
    # opinion. This is the garbage, pinned so nobody mistakes build() for the
    # guard.
    p = gs.TEMPLATE.build([{"role": "user", "content": "hi"}], tools="ab")
    assert '<tools>\n"a"\n"b"\n</tools>' in p


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


def test_arguments_that_do_not_resolve_to_an_object_are_not_a_call(gs):
    # Was: the raw string came back as `arguments`, so a client doing
    # args["path"] got TypeError rather than a KeyError it could handle. There
    # is no honest dict to hand back here -- coercing to {} would invent an
    # argument-free call the model never made -- so the block is treated like
    # any other malformed one and LEFT VISIBLE, where a human can see what the
    # model actually emitted. Both parse paths agree.
    junk = json.dumps({"name": "read_file", "arguments": "path=c.yaml"})
    text, calls = gs.parse_tool_calls("<tool_call>%s</tool_call>" % junk)
    assert calls == []
    assert "path=c.yaml" in text, "swallowing it leaves a mystery empty reply"
    assert gs._bare_tool_calls(junk) == []


def test_valid_json_that_is_not_an_object_is_not_a_call(gs, monkeypatch):
    # The subtler half: these PARSE fine and still cannot be an argument set.
    # `arguments` reaching a client as the int 123 breaks the OpenAI schema and
    # produces an Anthropic tool_use block whose input is not an object.
    #
    # BOTH spellings, because they take different branches of _tool_arguments:
    # a JSON STRING holding a scalar ("123") is parsed and then refused for not
    # being an object, and a BARE scalar (123) is refused before any parse.
    # Only the first was ever exercised -- while the comment above described
    # the second. Bodies are built with json.dumps: the old hand-escaped
    # '"\"hello\""' was not the JSON it looked like (Python had already eaten
    # the backslashes), so that case failed the BODY parse and never reached
    # _tool_arguments at all.
    reached = []
    real = gs._tool_arguments
    monkeypatch.setattr(gs, "_tool_arguments",
                        lambda raw: reached.append(raw) or real(raw))
    scalars = (123, [1, 2], True, "hello")
    for value in scalars:
        for arguments in (json.dumps(value), value):     # "123", then 123
            body = json.dumps({"name": "n", "arguments": arguments})
            wrapped = "<tool_call>%s</tool_call>" % body
            text, calls = gs.parse_tool_calls(wrapped)
            assert calls == [], body
            assert body in text, "left visible, like any other unusable block"
            assert gs.parse_tool_calls(body)[1] == [], body      # the bare-blob path
    # Every one of them got as far as the argument check, on both parse paths --
    # which is what "these PARSE fine" means.
    assert len(reached) == len(scalars) * 2 * 2
    # JSON null is the exception, and deliberately: it is a third spelling of
    # "no arguments", beside "" and an omitted field.
    null = json.dumps({"name": "n", "arguments": None})
    assert gs.parse_tool_calls("<tool_call>%s</tool_call>" % null)[1] == [
        {"name": "n", "arguments": {}}]
    # ...while the STRING "null" is a scalar like any other.
    quoted = json.dumps({"name": "n", "arguments": "null"})
    assert gs.parse_tool_calls("<tool_call>%s</tool_call>" % quoted)[1] == []


def test_a_zero_argument_call_means_no_arguments(gs):
    # A WELL-FORMED client shape, not model junk: arguments rendered as "".
    # json.loads("") raises, so it used to fall through to the raw string and
    # the caller got "" where {} is what the call means -- inconsistent with an
    # OMITTED arguments three lines away, which already defaulted to {}. The
    # two spellings of "no arguments" now agree.
    empty = json.dumps({"name": "now", "arguments": ""})
    assert gs.parse_tool_calls("<tool_call>%s</tool_call>" % empty)[1] == [
        {"name": "now", "arguments": {}}]
    assert gs.parse_tool_calls('<tool_call>{"name": "now"}</tool_call>')[1] == [
        {"name": "now", "arguments": {}}]
    assert gs._bare_tool_calls(empty) == [{"name": "now", "arguments": {}}]


def test_arguments_are_always_a_dict_when_a_call_is_returned(gs):
    # The contract the downstream consumers need, stated once: _complete
    # json.dumps this and _anthropic_complete puts it in tool_use.input, where
    # Anthropic requires an object. Anything that cannot be one is not a call.
    # Bodies built with json.dumps rather than literal escapes -- a hand-written
    # backslash here is one layer of quoting away from silently testing
    # something else.
    for payload in ({"name": "n"},
                    {"name": "n", "arguments": ""},
                    {"name": "n", "arguments": {"a": 1}},
                    {"name": "n", "arguments": json.dumps({"a": 1})}):
        body = json.dumps(payload)
        calls = gs.parse_tool_calls("<tool_call>%s</tool_call>" % body)[1]
        assert calls and isinstance(calls[0]["arguments"], dict), body


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


def test_a_truncated_bare_call_stays_visible_text(gs):
    # The commonest malformed thing this parser sees on real hardware: a BARE
    # call -- the shape thinking-suppressed Qwen3 emits -- cut off by
    # max_tokens. Unlike the wrapped cases above, the visible text starts with
    # { or [, so it reaches json.loads and raises there. The guard is what
    # stands between an ordinary truncated answer and an exception thrown on a
    # COMPLETED generation: a 500 on the non-stream path, and on a stream a
    # connection closed after the 200 headers had already gone out.
    for payload in ('{"name": "ls", "arguments": {"path": "/a',
                    '[{"name": "ls", "arguments": {}}',
                    '{"name": "ls", "arguments": {},}'):     # trailing comma
        text, calls = gs.parse_tool_calls(payload)
        assert calls == [], payload
        assert text == payload, "left visible, like any other unusable block"
        assert gs._bare_tool_calls(payload) == [], payload


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


def test_the_no_think_prefill_is_the_templates_exact_bytes(gs):
    # The ONE place these bytes are written out, because they are the bundle
    # template's and a drifted character here defeats the suppression without
    # erroring. Everything else -- test_api's "renders as a prefill" among them
    # -- says gs._NO_THINK, for the reason given in the tool round-trip test
    # above: a copied literal is a second place to update. This used to be a
    # copy of that test_api case with the constant spelled out.
    assert gs._NO_THINK == "<think>\n\n</think>\n\n"


# --- non-ASCII survives the render ----------------------------------------
# json.dumps escapes non-ASCII by default. For tool ARGUMENTS that defeats KV
# reuse outright: the model emitted raw UTF-8, the engine recorded exactly
# those bytes as resident, and a history render that spells the same character
# as a backslash-u escape no longer starts with them -- so every turn after a
# tool call with an accented path re-prefilled the whole conversation.

CAFE = "caf\u00e9.txt"          # escaped HERE so this file stays ASCII


def test_tool_arguments_render_as_the_bytes_the_model_emitted(gs):
    first = [{"role": "user", "content": "read it"}]
    # What the model generates and the engine commits, byte for byte:
    generated = '<tool_call>\n{"name": "read_file", "arguments": {"path": "%s"}}\n</tool_call>' % CAFE
    committed = gs.TEMPLATE.build(first, thinking=False) + generated
    # What the client sends back. parse_tool_calls hands `arguments` over as a
    # dict, and the Anthropic leg returns and receives it as one.
    text, calls = gs.parse_tool_calls(generated)
    assert calls == [{"name": "read_file", "arguments": {"path": CAFE}}]
    second = [*first,
              {"role": "assistant", "content": text,
               "tool_calls": [{"function": calls[0]}]},
              {"role": "tool", "content": "file contents"}]
    p = gs.TEMPLATE.build(second, thinking=False)
    assert CAFE in p and "\\u00e9" not in p
    assert p.startswith(committed), "or the byte-prefix check refuses reuse"


def test_a_call_only_turn_renders_with_no_newline_before_the_call(gs):
    # The template puts a newline in front of a call only when the turn's own
    # content, or an earlier call, precedes it. The render tested `if body:`,
    # and with thinking off `body` already holds the prefilled think block --
    # never empty -- so the commonest agent turn there is, one that is nothing
    # but a call, came back with a newline the model had not emitted. That one
    # byte failed the prefix check after EVERY tool call, ASCII or not.
    call = '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    first = [{"role": "user", "content": "list it"}]
    tc = [{"function": {"name": "ls", "arguments": {}}}]
    for thinking in (False, True):
        committed = gs.TEMPLATE.build(first, thinking=thinking) + call
        echoed = [*first, {"role": "assistant", "content": "", "tool_calls": tc},
                  {"role": "tool", "content": "a.txt"}]
        assert gs.TEMPLATE.build(echoed, thinking=thinking).startswith(committed)
        # null content is the other way clients spell "nothing but a call".
        echoed[1] = {"role": "assistant", "content": None, "tool_calls": tc}
        assert gs.TEMPLATE.build(echoed, thinking=thinking).startswith(committed)
    # The newline that IS the template's: after content, and between calls.
    said = [*first, {"role": "assistant", "content": "Looking.", "tool_calls": tc * 2}]
    p = gs.TEMPLATE.build(said, thinking=False)
    assert gs._NO_THINK + "Looking.\n" + call + "\n" + call + gs.TEMPLATE.asst_suf in p


def test_a_string_arguments_value_is_spliced_verbatim(gs):
    # As the bundle's template does. So on the OpenAI leg, where `arguments`
    # travels as a JSON string, reuse is only as exact as the string the
    # RESPONSE handed out -- the render must not re-spell it either way.
    for spelled in ('{"path": "%s"}' % CAFE, '{"path": "caf\\u00e9.txt"}',
                    '{"path":"x"}'):
        msgs = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "f", "arguments": spelled}}]}]
        assert '"arguments": ' + spelled + "}" in gs.TEMPLATE.build(msgs)


def test_tool_schemas_render_unescaped_too(gs):
    # Not a reuse question -- the same schema renders the same way every time
    # -- but the model reads this text, and the template's tojson does not
    # escape it: a description in any non-Latin script arrived as a wall of
    # backslash-u sequences.
    tools = [{"type": "function", "function": {
        "name": "read_file", "description": "Lit un fichier \u2014 par exemple %s" % CAFE}}]
    p = gs.TEMPLATE.build([{"role": "user", "content": "hi"}], tools=tools)
    assert "par exemple %s" % CAFE in p and "\\u" not in p


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
