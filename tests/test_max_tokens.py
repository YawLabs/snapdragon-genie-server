"""How a request's output cap is resolved, before anything is generated.

_max_tokens sits at the door of both APIs. It has two jobs: refuse what
GenieDialog_setMaxNumTokens would silently misread (a negative wraps to
4294967295 in a c_uint32; neither downstream guard notices), and decide which
of two spellings -- OpenAI's deprecated `max_tokens` and its replacement
`max_completion_tokens` -- a request is actually using. The tests in
test_api.py pin the shape of the 400; this file pins the resolution itself,
including the two places it used to be inconsistent with its own docstring.
"""

import pytest


def test_absent_and_null_mean_the_default(gs):
    assert gs._max_tokens({}) == gs.DEFAULT_MAX_TOKENS
    assert gs._max_tokens({"max_tokens": None}) == gs.DEFAULT_MAX_TOKENS


def test_zero_means_the_default_however_it_is_spelled(gs):
    # A numeric 0 kept the default while the string "0" answered 400, because
    # the zero check ran before the parse. Nothing a client sends should be
    # treated differently for being a JSON string of the same digits when the
    # function accepts strings at all.
    for zero in (0, 0.0, "0", False):
        assert gs._max_tokens({"max_tokens": zero}) == gs.DEFAULT_MAX_TOKENS, zero


def test_a_zero_legacy_cap_defers_to_the_modern_spelling(gs):
    # {"max_tokens": 0, "max_completion_tokens": 16} returned the default: the
    # legacy field was present, so the fallback never ran, and then its 0 was
    # read as "use the default" -- the 0 hid the 16.
    assert gs._max_tokens({"max_tokens": 0, "max_completion_tokens": 16}) == 16
    assert gs._max_tokens({"max_tokens": "0", "max_completion_tokens": 16}) == 16
    assert gs._max_tokens({"max_tokens": None, "max_completion_tokens": 16}) == 16


def test_a_set_legacy_cap_wins_over_the_modern_spelling(gs):
    # The more explicit signal, from a client old enough to send it at all.
    assert gs._max_tokens({"max_tokens": 8, "max_completion_tokens": 16}) == 8


@pytest.mark.parametrize("bad", [-1, "-1", "abc", True, [1], {"n": 1},
                                 float("inf"), float("nan")])
def test_anything_that_is_not_a_positive_integer_is_refused(gs, bad):
    with pytest.raises(ValueError):
        gs._max_tokens({"max_tokens": bad})


def test_the_refusal_names_the_field_the_client_sent(gs):
    with pytest.raises(ValueError, match="max_completion_tokens"):
        gs._max_tokens({"max_completion_tokens": -1})
    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        gs._max_tokens({"max_tokens": -1})


def test_a_float_is_truncated_as_int_does(gs):
    assert gs._max_tokens({"max_tokens": 2.5}) == 2


def test_the_cap_is_not_clamped_to_the_window(gs):
    # The docstring used to say "clamped to the window" while the code did
    # not clamp -- and the code was right: the fits check downstream refuses
    # an oversized cap with a 400 that quotes the CLIENT's number back, which
    # only works if the number was not rewritten here first.
    assert gs._max_tokens({"max_tokens": 10 ** 6}) == 10 ** 6
