"""One prompt-at-depth builder for the tokenizer-based tools.

Three tools used to carry their own. bench_servers.py `prompt_at` and
probe_server_semantics.py `filler` were the same three-line algorithm under two
names with two filler sentences; bench_endpoint.py `prompt_of` sizes by a
chars/4 estimate over a third. So "depth N" named a different prompt in each
tool, and a row measured in one could not be read against the same depth in
another. The two tokenizer-based tools build from here now. bench_endpoint.py
keeps its estimate on purpose: it runs against any OpenAI-compatible server
with no bundle to borrow a tokenizer from, and says so at its CHARS_PER_TOKEN.

The tokenizer is a parameter rather than a module global so one builder serves
a tool that binds a tokenizer once at startup and a test that hands in a stub.
Stdlib only at import; `tokenizers` is imported inside load_tokenizer so a tool
can fail with a sentence instead of a traceback when it is missing.
"""
import os
import sys

# bench_servers.py's sentence, byte-identical, so switching that tool over here
# changes none of its prompts -- its numbers are the README's findings table.
FILLER_UNIT = ("The measurement below concerns memory bandwidth on a mobile "
               "accelerator, and this paragraph repeats to reach a target depth. ")


def ntok(tok, text):
    """Token count of `text` under the bundle's own tokenizer, no specials."""
    return len(tok.encode(text, add_special_tokens=False).ids)


def prompt_at(tok, depth, unit=FILLER_UNIT):
    """A prompt of `depth` tokens, measured rather than estimated.

    Repeats `unit` until its tokenized length reaches `depth`, then decodes
    the first `depth` ids -- so the size is the tokenizer's count of what was
    sent, not a characters-per-token guess. Checked against the Qwen3 4B
    bundle's tokenizer: the result re-encodes to exactly `depth` at every
    size tried (1 through 9011). The servers still report their own
    prompt_tokens, and the tools print that beside the target rather than
    assume the two agree -- the chat template adds its own tokens on top.
    """
    depth = int(depth)
    body = unit * max(1, depth // max(1, ntok(tok, unit)))
    while ntok(tok, body) < depth:
        body += unit
    return tok.decode(tok.encode(body, add_special_tokens=False).ids[:depth])


def load_tokenizer(bundle_dir):
    """The bundle's tokenizer.json as a `tokenizers.Tokenizer`, or a named exit.

    Everything a tool can hit before its first request is a sys.exit that says
    what to do, not a traceback out of a library: the package missing,
    GENIE_BUNDLE_DIR unset, or the directory existing but holding no
    tokenizer.json (a geniex cache copy, a path one level off). That last one
    used to reach Tokenizer.from_file unchecked and came back as a bare
    `Exception: The system cannot find the file specified. (os error 2)` from
    the Rust side, which names neither the file nor the variable to fix.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("this tool needs `pip install tokenizers` -- it sizes prompts "
                 "with the bundle's own tokenizer rather than an estimate")
    if not bundle_dir:
        sys.exit("set GENIE_BUNDLE_DIR to the bundle the server under test is serving")
    path = os.path.join(bundle_dir, "tokenizer.json")
    if not os.path.isfile(path):
        sys.exit("no tokenizer.json at %s -- GENIE_BUNDLE_DIR must be the bundle "
                 "directory itself, the one holding genie_config.json" % path)
    return Tokenizer.from_file(path)
