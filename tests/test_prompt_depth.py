"""Tests for the shared prompt-at-depth builder.

The bundle tokenizer is not here (no bundle, and `tokenizers` is optional), so
the builder is driven with a whitespace tokenizer that has the two methods it
uses. That is enough to pin the ALGORITHM -- repeat, top up, cut at exactly
`depth` ids -- which is what the three tools used to disagree on. Whether a
real BPE tokenizer re-encodes the cut to the same count was checked by hand
against the Qwen3 4B bundle when this module was written (it does, at every
depth tried) and is recorded in the builder's docstring, not asserted here.
"""

import sys
import types

import pytest

import prompt_depth as pd


class WordTokenizer:
    """One id per whitespace-separated word; decode joins them with spaces."""

    def __init__(self):
        self.vocab = {}
        self.words = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False, "the builder must never add specials"
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
            ids.append(self.vocab[w])
        return types.SimpleNamespace(ids=ids)

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)


@pytest.fixture
def tok():
    return WordTokenizer()


@pytest.mark.parametrize("depth", [1, 7, 64, 1000])
def test_the_prompt_measures_exactly_depth_tokens(tok, depth):
    assert pd.ntok(tok, pd.prompt_at(tok, depth)) == depth


def test_a_depth_below_one_unit_still_cuts_to_size(tok):
    # depth // ntok(unit) is 0 here; the max(1, ...) floor keeps one unit to
    # cut from instead of an empty body that the top-up loop then grows.
    unit_len = pd.ntok(tok, pd.FILLER_UNIT)
    assert unit_len > 3
    assert pd.ntok(tok, pd.prompt_at(tok, 3)) == 3


def test_the_prompt_is_made_of_the_filler_unit(tok):
    p = pd.prompt_at(tok, 40)
    assert p.startswith(pd.FILLER_UNIT.split()[0])
    assert set(p.split()) <= set(pd.FILLER_UNIT.split())


def test_a_caller_may_supply_its_own_unit(tok):
    p = pd.prompt_at(tok, 5, unit="alpha beta gamma ")
    assert p.split() == ["alpha", "beta", "gamma", "alpha", "beta"]


def test_the_filler_is_bench_servers_sentence_verbatim():
    # Switching bench_servers.py over must change none of its prompts: its
    # numbers are the README's findings table.
    assert pd.FILLER_UNIT == ("The measurement below concerns memory bandwidth on a mobile "
                              "accelerator, and this paragraph repeats to reach a target depth. ")


# --- load_tokenizer: every pre-request failure is a sentence, not a trace ----

def _fake_tokenizers(monkeypatch, opened):
    mod = types.ModuleType("tokenizers")

    class Tokenizer:
        @staticmethod
        def from_file(path):
            opened.append(path)
            return "TOK"
    mod.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", mod)


def test_a_bundle_dir_without_tokenizer_json_exits_naming_the_path(monkeypatch, tmp_path):
    # The directory exists (a geniex cache copy, a path one level off) but the
    # file does not. This used to reach the Rust side and come back as a bare
    # "os error 2" that named neither the file nor the variable.
    opened = []
    _fake_tokenizers(monkeypatch, opened)
    with pytest.raises(SystemExit) as e:
        pd.load_tokenizer(str(tmp_path))
    assert "tokenizer.json" in str(e.value)
    assert str(tmp_path) in str(e.value)
    assert "GENIE_BUNDLE_DIR" in str(e.value)
    assert opened == [], "must not hand a missing path to the library"


def test_an_unset_bundle_dir_exits_with_the_variable_name(monkeypatch, tmp_path):
    # Run from a directory that HOLDS a tokenizer.json: os.path.join("", name)
    # is just `name`, so without its own check an unset variable quietly loads
    # whatever tokenizer the cwd happens to have and sizes every prompt with it.
    opened = []
    _fake_tokenizers(monkeypatch, opened)
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        pd.load_tokenizer("")
    assert str(e.value).startswith("set GENIE_BUNDLE_DIR")
    assert opened == [], "an unset variable must not fall through to the cwd"


def test_a_missing_package_exits_with_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenizers", None)   # import -> ImportError
    with pytest.raises(SystemExit) as e:
        pd.load_tokenizer("anything")
    assert "pip install tokenizers" in str(e.value)


def test_a_present_tokenizer_json_is_opened(monkeypatch, tmp_path):
    opened = []
    _fake_tokenizers(monkeypatch, opened)
    (tmp_path / "tokenizer.json").write_text("{}")
    assert pd.load_tokenizer(str(tmp_path)) == "TOK"
    assert opened == [str(tmp_path / "tokenizer.json")]
