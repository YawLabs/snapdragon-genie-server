#!/usr/bin/env python3
"""
OpenAI-compatible HTTP server for a Qualcomm Genie NPU LLM bundle
(Snapdragon, Hexagon HTP). Targets the Windows-on-Snapdragon Hexagons, and
that set is DERIVED at startup rather than hardcoded: an arch counts only if
the SDK ships both its skel and its Windows stub, so a future Hexagon works
without editing this file and the Android-only archs are excluded with a
reason. On QAIRT 2.45 it comes out v68, v73 and v81.

Loads the Genie context-binary bundle ONCE via the Genie C API (ctypes ->
Genie.dll) so the model stays resident on the HTP; every /v1/chat/completions
request reuses it (no 11-35s bundle load that genie-t2t-run.exe would repeat per
invocation -- measured on the 8192 multi bundle; docs/GENIE_SERVER.md, Run).

Pure Python stdlib -- no pip dependencies. Must run on a native ARM64 (aarch64)
Python, because Genie.dll and its Qnn* deps are aarch64-windows-msvc.

Configured through GENIE_* environment variables. The table of them -- each
one's default and what it decides -- is the "Environment" section of
docs/GENIE_SERVER.md, and it is deliberately not repeated here: a second copy
is how this docstring came to list nine of the twenty-one and to promise
"sensible defaults" for the two that have none. GENIE_BUNDLE_DIR and
GENIE_SDK_DIR are REQUIRED -- load_engine exits at startup naming them when
either is unset -- and run-genie-server.ps1 derives both from GENIE_NPU_ROOT,
so the normal launch path never sets them by hand. Every reader degrades on a
malformed value (a warning line naming the variable, then the default) rather
than refusing to boot; see _int_env.
"""

import codecs
import ctypes as C
import hashlib
import json
import os
import queue
import random
import re
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
def _int_env(name, default, minimum=None):
    """An int from the environment, or `default` with a line saying why not.

    EVERY config reader here goes through this, _float_env or _path_env, and
    that is the contract: a typo in any GENIE_* variable degrades to the
    default with a line naming the variable and the value it rejected, never
    a refusal to boot. It held for exactly two readers for a while (GENIE_SEED
    and GENIE_ORPHAN_HOLD_CHARS) while this docstring claimed the rest already
    behaved -- GENIE_PORT=808O killed the process at IMPORT with a bare
    `invalid literal for int()`, before any startup line had printed, on a
    server whose whole startup banner exists to explain itself.

    `minimum` extends the same contract to a value that PARSES and is still
    not usable, because "rejected" has to mean the same thing either way. A
    silent clamp reads as acceptance: GENIE_MAX_TOKENS=-1 is llama.cpp's
    spelling of "no limit" and this repo runs llama-server legs beside this
    one, so an operator carried that habit over, got -1 floored to 1 with no
    line anywhere, and every default-capped completion came back one token
    long with finish_reason "length". The default is what an out-of-range
    value degrades to, not the bound: 512 is a cap somebody might have
    wanted, 1 is not. (The two floors that are NOT wired up this way,
    GENIE_FAIL_THRESHOLD and GENIE_MAX_INFLIGHT, predate this and are
    deliberate -- 0 there is a documented "disable the cap" that the NPU's
    single-flight makes meaningless, and their tests pin the clamp.)
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print("[genie] WARNING: %s=%r is not an integer; using %r instead."
              % (name, raw, default), flush=True)
        return default
    if minimum is not None and value < minimum:
        print("[genie] WARNING: %s=%r must be >= %d; using %r instead."
              % (name, raw, minimum, default), flush=True)
        return default
    return value


def _float_env(name, default):
    """_int_env for the supervision timeouts: seconds, and may be fractional."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("[genie] WARNING: %s=%r is not a number; using %r instead."
              % (name, raw, default), flush=True)
        return default


def _path_env(name):
    """A directory from the environment, made absolute; "" when unset.

    Absolute because load_engine chdirs INTO the bundle dir (Genie resolves
    the config's relative ctx-bin and tokenizer paths against CWD) and then
    opens genie_config.json by joining the same variable again -- so a
    relative GENIE_BUNDLE_DIR passed the isdir check against the launch cwd
    and raised a bare FileNotFoundError from the new one, after the point it
    had been checked. os.add_dll_directory wants an absolute GENIE_SDK_DIR for
    its own reasons. The launcher only ever produces absolute paths; this
    makes a hand-set relative one behave the same.
    """
    raw = os.environ.get(name, "")
    return os.path.abspath(raw) if raw else ""


# The bundle (context binaries + config + tokenizer) and the QAIRT 2.45 runtime
# are large external artifacts that do NOT live in this repo, and these two
# have NO default: load_engine exits at startup naming both when either is
# empty. run-genie-server.ps1 derives them from GENIE_NPU_ROOT -- the directory
# holding bundles/ and qairt/, defaulting to ../genie-npu beside this repo --
# picking the bundle by its -Model name and the newest qairt/* for the SDK, so
# the normal launch path never sets these by hand. Set them directly only when
# running genie_server.py without the launcher; see docs/GENIE_SERVER.md.
BUNDLE_DIR = _path_env("GENIE_BUNDLE_DIR")
SDK_DIR = _path_env("GENIE_SDK_DIR")
HOST = os.environ.get("GENIE_HOST", "127.0.0.1")
PORT = _int_env("GENIE_PORT", 8080)
MODEL_ID = os.environ.get("GENIE_MODEL_ID", "qwen3-4b-npu")
# Must be >= 1, because this one bypassed the check the per-request value gets
# at the door (_max_tokens): GENIE_MAX_TOKENS=-1 reached c_uint32 and wrapped
# to 4294967295 -- the unbounded generation that guard exists to refuse -- and
# 0 skipped setMaxNumTokens altogether, leaving whatever cap the previous
# request had set on the resident dialog. Rejected LOUDLY, back to 512, rather
# than clamped: a clamp to 1 answered every uncapped request with one token
# and no line on screen said why (-1 is llama.cpp's "no limit"), and the
# startup banner does not print the effective default cap either.
DEFAULT_MAX_TOKENS = _int_env("GENIE_MAX_TOKENS", 512, minimum=1)
# Ceiling on a request body, checked BEFORE the read. 8 MB is orders of
# magnitude above any legitimate prompt at n_ctx 16384; the point is that
# Content-Length was previously trusted and read in full, ahead of the
# single-flight semaphore, so MAX_INFLIGHT did not bound it.
MAX_BODY_BYTES = _int_env("GENIE_MAX_BODY_BYTES", 8 * 1024 * 1024)
# Seconds any ONE socket read or write may block before the connection is
# dropped; 0 or less waits forever, which is what http.server does left alone.
# Nothing bounded these before: a client that sent fewer bytes than its
# Content-Length, or opened a keep-alive connection and never sent a request
# line, parked a handler thread for the life of the process -- ahead of the
# single-flight semaphore, so MAX_INFLIGHT did not bound it, and
# ThreadingHTTPServer does not bound its threads either. It is NOT a cap on how
# long a generation may run: a handler never blocks on the socket while it
# waits for the engine, and a stream's writes complete as fast as the client
# reads. What it bounds there is a client that stopped READING, which is
# handled as one that left (see _Emitter.write).
SOCKET_TIMEOUT_S = _int_env("GENIE_SOCKET_TIMEOUT", 120)
STRIP_THINK = os.environ.get("GENIE_STRIP_THINK", "0") == "1"
# Qwen3 is a reasoning model: left alone it emits a <think> block before every
# answer. Measured on this box, a single tool-calling turn spent ~280 of its
# 300 output tokens thinking -- 41s against 2.4s for the same prompt and the
# same correct call. The bundle's own template supports suppressing it by
# PREFILLING a closed, empty think block, so this exposes that as a knob.
#
# DEFAULT IS OFF, changed deliberately. This used to default ON, on the
# reasoning that faithfulness to the model is the honest default and agent
# clients could opt out. Two things make that the wrong trade HERE. The cost is
# not a tax on quality, it is 10-17x on every agent step, and its length swings
# run to run -- so the default was not merely slow but unpredictable, which is
# the property a human actually notices. And this server exists to be driven by
# an agent: the docs recommend turning thinking off for agentic use, so shipping
# the opposite made the recommended configuration the one nobody got by default.
# A default should be the thing the primary caller wants.
#
# Faithfulness is still one env var or one request field away, and NOTHING here
# is lossy -- suppression is a prompt prefill, not a filter over the output, so
# a caller that asks for reasoning gets exactly what the model produces.
THINKING_DEFAULT = os.environ.get("GENIE_THINKING", "0") in ("1", "true", "yes")
# Headroom left between the rendered prompt and the compiled window, so a
# generation has somewhere to go. Genie hard-errors (status=4) on overflow --
# it does not truncate -- so the margin is what stands between a long session
# and a 500. Must be >= 0: a negative margin is not more headroom, it is a
# budget past the window, and build_windowed subtracts this without a check.
# Rejected loudly back to 64, like the token cap above and for the same
# reason -- a clamp to 0 leaves a generation no room at all and says nothing.
WINDOW_MARGIN = _int_env("GENIE_WINDOW_MARGIN", 64, minimum=0)
# Pin the sampler seed for reproducible output; unset means a fresh seed per
# PROCESS (a per-request seed is inert on QAIRT 2.45 -- see next_seed()), which
# is the default and the thing that stops every server start giving the same
# answer to a given prompt.
FIXED_SEED = _int_env("GENIE_SEED", None)


def next_seed():
    """The sampler seed to load this process with.

    Why this is needed at all: Genie re-seeds its RNG from the config's `seed`
    on every GenieDialog_reset, and _plan calls reset on every request that does
    not continue the resident KV. The AI Hub bundles ship `"seed": 42`, so each fresh prompt
    replays the SAME pseudo-random stream and the model walks an identical
    sampling trajectory. Identical prompt in, byte-identical answer out --
    measured here three times running, and again across separate server
    processes.

    That is not a cosmetic determinism note. It is why a prompt that lands on a
    repetitive answer lands there EVERY time, and why re-asking never escapes
    it: at temp 0.8 the model is nominally sampling, but the dice are reset
    before every roll. No repetition penalty can fix that -- the problem is not
    which tokens are penalised, it is that the same draw is taken every time.

    Why it is applied HERE, at create, and not per request: a per-request
    GenieSamplerConfig apply was tried and is inert on QAIRT 2.45 (three
    identical generations with a fresh seed on each) -- see apply_sampler.
    Sampling binds at GenieDialog_create, so the config text is the only thing
    that can carry it.

    And `"seed": -1` in the bundle does NOT work either, though it looks like
    it should: the constructor reads -1 as "seed from the clock", but reset()
    re-seeds with `_seed` unconditionally, so -1 casts to a fixed uint32 and
    every generation after the first is deterministic again.

    LIMIT, because this is a real one: the seed varies per PROCESS, not per
    request. Within one server run an identical prompt still replays its
    identical answer. Fixing that needs a QAIRT that honours a post-create
    sampler apply.

    GENIE_SEED pins it when reproducibility is what you want (comparing two
    bundles, bisecting a bad generation). The benchmarks do not need it --
    throughput does not depend on the seed.
    """
    if FIXED_SEED is not None:
        return FIXED_SEED
    # Bounded to int32: Genie parses `seed` into an int32_t, so a larger value
    # would wrap to something arbitrary rather than be rejected.
    return random.randrange(1, 2 ** 31 - 1)


# Plain eviction drops the oldest turns outright, so the agent forgets it
# already read a file and reads it again -- burning the window a second time on
# information it had. Summarising the turns on their way out keeps the facts and
# discards only the tokens. Costs one extra NPU call, and ONLY when eviction was
# going to happen anyway (i.e. the alternative was losing the content).
SUMMARIZE_EVICTED = os.environ.get("GENIE_SUMMARIZE_EVICTED", "1") != "0"
SUMMARY_MAX_TOKENS = _int_env("GENIE_SUMMARY_MAX_TOKENS", 192)


def summary_token_cap():
    """SUMMARY_MAX_TOKENS, clamped so the note cannot crowd out the window.

    The note is retained context, so on a small-context bundle a large setting
    makes it a meaningful fraction of n_ctx. build_windowed already re-fits
    afterwards and falls back to plain eviction if the note does not fit -- but
    that fails LATE, after the summarisation call has already been paid for.
    An eighth of the window is a cheap early bound; the floor of 32 keeps the
    note useful on a tiny bundle rather than clamping it to nothing.
    """
    return max(32, min(SUMMARY_MAX_TOKENS, read_context_size() // 8))
# Heading of the retained note inside the system turn. The model reads it, so
# it says what the block under it is. It is ALSO how a note is recognised in a
# system turn a client sends back (_split_note), and that is why it is this
# specific: it used to be the bare prose "[earlier context]", matched anywhere
# in the text, so a system prompt that merely QUOTED those two words lost
# everything after them on its first eviction -- cut out as a "previous note"
# and fed to the summariser. Named, versioned, and honoured only on a line of
# its own with nothing but the note between it and the end of the turn.
#
# It is NOT how this server finds its OWN previous note. No response carries
# the note, so a stateless client resending its history never sends the marker
# back; that carry-forward is _NOTE_STATE's job, server-side.
SUMMARY_MARKER = "[genie_server note v1: summary of earlier turns]"


def _bundle_file(name):
    """BUNDLE_DIR/name, or None when no bundle dir is configured.

    None rather than a joined path, because os.path.join("", name) is just
    `name` -- a path relative to the CURRENT DIRECTORY. With GENIE_BUNDLE_DIR
    unset, every reader below therefore opened ./genie_config.json or
    ./metadata.json from wherever the server was started, and main() runs
    bundle_config_warnings() BEFORE load_engine's "set GENIE_BUNDLE_DIR" exit:
    a stray config in the working directory was read, believed, and warned
    about as if it were the bundle's. An unset bundle dir has no files in it.
    """
    return os.path.join(BUNDLE_DIR, name) if BUNDLE_DIR else None


def _load_bundle_json(name):
    """Parsed BUNDLE_DIR/name. Raises what open() and json.load raise.

    OSError means the file is not there -- FileNotFoundError when no bundle dir
    is set at all, see _bundle_file. ValueError means it is there and is not
    JSON, which covers JSONDecodeError and UnicodeDecodeError alike. Every
    reader below catches both and returns its own could-not-read value;
    config_parse_error() is the one place that tells them apart.
    """
    path = _bundle_file(name)
    if path is None:
        raise FileNotFoundError("no bundle dir set, so no %s" % name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_CONTEXT_SIZE = None


def read_context_size(default=4096):
    """`dialog.context.size` from genie_config.json -- the SOFTWARE window cap.

    Precisely NOT "the window this bundle was compiled at", which this
    docstring used to claim. The two can differ: measured here, setting this to
    1024 against a 4096-compiled bundle left the HTP allocation byte-identical
    and decode unchanged, because the compiled window is fixed at export time
    (`--context-lengths`) and this key only lowers the ceiling the evictor works
    against. So it is the right number for the eviction budget and for what a
    client may send, and the WRONG number to predict latency from -- that
    belongs to the compiled window, which read_context_lengths() reports.

    Read rather than hardcoded, for the same reason llama.cpp reports n_ctx at
    /props instead of publishing a constant: the value belongs to the bundle,
    and a different bundle (or a recompile at another length) makes any literal
    here quietly wrong. A client that plans against the wrong window does not
    error -- it silently overruns the model, which is the failure this endpoint
    exists to prevent.

    Falls back to `default` when the config is missing or malformed: /props
    answering with a slightly stale number is far better than the server
    failing to start over a field it only needs for a metadata endpoint.
    """
    # Cached after the first read: the value cannot change while the bundle is
    # loaded, and this sits on the request path (build_windowed every request,
    # again on eviction and on /props) -- re-opening and JSON-parsing the config
    # per request is blocking file I/O for a constant.
    global _CONTEXT_SIZE
    if _CONTEXT_SIZE is not None:
        return _CONTEXT_SIZE
    try:
        cfg = _load_bundle_json("genie_config.json")
        size = int(cfg["dialog"]["context"]["size"])
        _CONTEXT_SIZE = size if size > 0 else default
    except Exception:
        _CONTEXT_SIZE = default
    return _CONTEXT_SIZE


_CONTEXT_LENGTHS = None
# Every `poll` found in the config; None until read, [] when there are none.
# Deliberately ONE global rather than this plus a cached pick: two that must be
# written together is an invariant nothing enforces, and the failure is silent
# -- a pre-set pick with no matches makes the conflict check below evaluate to
# empty and report nothing. _pick_poll is pure and runs on a 0-2 element list,
# so deriving it per call costs nothing worth keeping a second global for.
_POLL_MATCHES = None
_SAMPLER = None
_CONFIG_PRESENT = None
# Why genie_config.json would not parse: None until checked, "" when it parsed
# or is not there. Its OWN cached check rather than a flag the readers set on
# their way past, for the reason given above _POLL_MATCHES: a flag that three
# readers must each remember to write is an invariant nothing enforces, and the
# first reader to be called with a warm cache would leave it unset.
_CONFIG_PARSE_ERROR = None


def _find_all(obj, key, path=""):
    """Every (value, dotted-path) for `key`, in document order.

    Searched rather than addressed by a fixed path because the QnnHtp block's
    nesting has moved between QAIRT releases and this server deliberately
    supports more than one. A hardcoded path that is right for 2.45 and absent
    on the next SDK would read as "the flag is not set" -- the wrong answer for
    a flag whose shipped default is the expensive one.

    ALL of them rather than the first, because "first" means depth-first in
    insertion order, which prefers a NESTED match over a shallower one: on
    {"a": {"poll": true}, "poll": false} it returned a.poll. Harmless on every
    real genie_config.json, which has one -- and silently wrong on one with
    two, in a value that drives both a startup warning and /props. Collect them
    and let the caller disambiguate, so a config we cannot read confidently
    says so instead of picking.
    """
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = "%s.%s" % (path, k) if path else k
            if k == key:
                out.append((v, here))
            else:
                out.extend(_find_all(v, key, here))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_find_all(v, key, "%s[%d]" % (path, i)))
    return out


def _pick_poll(matches):
    """The (value, path) that is most likely the backend's, or (None, None).

    A QnnHtp-qualified path wins. The BLOCK is called QnnHtp in every QAIRT
    that ships one; what moves between releases is where it sits, which is
    exactly why the search is by key rather than by path. Failing that, the
    shallowest match -- a buried key is less likely to be the backend's than a
    top-level one -- and document order breaks a remaining tie, since sorted()
    is stable.
    """
    if not matches:
        return (None, None)
    qualified = [m for m in matches if "qnnhtp" in m[1].lower()]
    return sorted(qualified or matches, key=lambda m: m[1].count("."))[0]


def read_poll_setting():
    """The bundle's QnnHtp `poll` flag as (value, where). (None, None) if absent.

    Checked at startup rather than left to a doc line, because it is the single
    most consequential thing about a bundle and it ships in the wrong state.
    `"poll": true` busy-waits: measured here, a server that has answered nothing
    but /health burns 270% CPU -- 2.7 cores -- while completely idle, and it
    costs up to 36% of decode on top. It also takes about a quarter of the
    win from running this engine beside a GPU one (1.70x against 1.26x over the
    best single engine), because the OpenCL backend needs those same host cores
    to dispatch a kernel per token. An earlier revision called that a 0.78x NET
    LOSS; a controlled re-run refuted it -- both settings are a gain, and the
    retraction is in MULTI_ENGINE.md.

    Nearly every retracted number in docs/ traces back to this flag being true
    and nobody noticing. Noticing is cheap; the docs are the record of what not
    noticing costs.
    """
    global _POLL_MATCHES
    if _POLL_MATCHES is None:
        try:
            _POLL_MATCHES = _find_all(_load_bundle_json("genie_config.json"),
                                      "poll")
        except Exception:
            # Missing and unparseable both land here as "no matches", which is
            # right for THIS reader's callers (/props reports poll: null) and
            # wrong as the basis of a claim about the file --
            # bundle_config_warnings asks config_parse_error() before it says
            # "no `poll` key found" about a file nobody managed to read.
            _POLL_MATCHES = []
    return _pick_poll(_POLL_MATCHES)


def read_context_lengths():
    """`genie.context_lengths` from metadata.json -- the graphs inside the bundle.

    Not a record of what the model COULD be exported at: confirmed with
    `qnn-context-binary-utility`, a bundle carries one prefill and one decode
    graph per compiled length and runs each token against the smallest that
    fits. A single-length bundle has one pair and so runs every token against
    its whole window. Measured 2-3x on short prompts at the SAME n_ctx, for
    +3.8% bundle size and zero extra HTP memory.

    Read off the artifact because nothing else can show it: two bundles of the
    same window are byte-identical in metadata.json apart from this list, so
    /props cannot distinguish them and neither can a latency measurement taken
    at one depth.
    """
    global _CONTEXT_LENGTHS
    if _CONTEXT_LENGTHS is None:
        try:
            v = (_load_bundle_json("metadata.json").get("genie")
                 or {}).get("context_lengths")
            # isinstance, not truthiness: a bare string is iterable, so
            # "8192" yielded [8, 1, 9, 2] -- a bundle reported as multi-length
            # with four invented graph lengths, which is exactly the misreport
            # this field was added to prevent. Anything that is not a list
            # claims nothing.
            _CONTEXT_LENGTHS = [int(x) for x in v] if isinstance(v, list) else []
        except Exception:
            _CONTEXT_LENGTHS = []
    return _CONTEXT_LENGTHS


def config_present():
    """Is there a bundle config here AT ALL?

    Distinct from "the config has no poll key", and the distinction is the
    whole point. With no bundle dir -- unset env, a typo, a fresh clone --
    every reader below returns its could-not-read value, and the checks then
    report a bundle that is missing rather than misconfigured. The first line a
    new user saw was a note about `poll` in a genie_config.json they do not
    have, sitting in front of the real error naming the env vars to set.

    read_sampler already refuses to make a claim about a file it could not
    open; this applies the same rule to the config as a whole. A config that is
    PRESENT and corrupt still warns -- but about being unparseable, which is
    the true and actionable thing to say; see config_parse_error.

    An UNSET bundle dir is absent by definition and is never stat'ed: joined
    with "" the path is the bare filename, i.e. a genie_config.json in the
    current directory, which is not this server's bundle however real it is.
    """
    global _CONFIG_PRESENT
    if _CONFIG_PRESENT is None:
        path = _bundle_file("genie_config.json")
        _CONFIG_PRESENT = bool(path) and os.path.isfile(path)
    return _CONFIG_PRESENT


def config_parse_error():
    """Why genie_config.json would not parse; "" if it parsed or is not there.

    The readers around this one fold "missing" and "not JSON" into a single
    could-not-read value, and for their own callers that is right. It stopped
    being right in bundle_config_warnings: a file holding `{not json` came out
    as "note: no `poll` key found in genie_config.json" -- a statement about
    the CONTENTS of a file that was never successfully read, sending the
    operator to add a key to a config whose actual problem is a syntax error
    one line up. The parser's own message is returned because it carries the
    line and column, which is the whole fix.

    Absent is "" and not an error: config_present() already owns that case.
    """
    global _CONFIG_PARSE_ERROR
    if _CONFIG_PARSE_ERROR is None:
        try:
            _load_bundle_json("genie_config.json")
            _CONFIG_PARSE_ERROR = ""
        except ValueError as e:
            _CONFIG_PARSE_ERROR = str(e) or e.__class__.__name__
        except Exception:
            _CONFIG_PARSE_ERROR = ""
    return _CONFIG_PARSE_ERROR


def read_sampler():
    """The bundle's `dialog.sampler` block, or {} if it cannot be read.

    {} means "no answer", never "empty sampler" -- the caller has to tell those
    apart, because warning about a missing penalty on a config we failed to
    open would be a claim about a file we never read.
    """
    global _SAMPLER
    if _SAMPLER is None:
        try:
            _SAMPLER = dict(
                _load_bundle_json("genie_config.json")["dialog"]["sampler"])
        except Exception:
            _SAMPLER = {}
    return _SAMPLER


def _as_number(v):
    """v as a float, or 0.0 if it is not one. A junk value is not a penalty."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def sampler_penalty_state(sampler):
    """Is this sampler's repetition penalty actually going to do anything?

    Genie applies four `token-penalty` fields, and the block is OPTIONAL: with
    it absent every one defaults to 0 (measured here, by serving the same
    bundle with the block present and absent), which means no
    repetition suppression at all. Qualcomm's own reference config for this
    stack sets it; the AI Hub export path emits the same sampler WITHOUT it, so
    a bundle arrives sampling at temp 0.8 with nothing holding it back. What
    that looks like from the outside is a model that answers normally for a
    while and then emits the same paragraph until it hits max_tokens.

    Three ways to get nothing, which is why this returns a state rather than a
    bool -- they need different advice:

      absent     no token-penalty block at all; every field defaults to 0.
      no-window  `penalize-last-n` is 0, so the penalties below it are read
                 and then applied to an empty window. This is the one worth
                 catching: someone sets repetition-penalty, restarts, sees no
                 change, and concludes the knob does not work.
      all-zero   a window, but every penalty in it is 0.
    """
    pen = (sampler or {}).get("token-penalty")
    if not isinstance(pen, dict):
        return "absent"
    if _as_number(pen.get("penalize-last-n")) <= 0:
        return "no-window"
    if not any(_as_number(pen.get(k)) for k in
               ("repetition-penalty", "presence-penalty", "frequency-penalty")):
        return "all-zero"
    return "ok"


def _poll_warnings():
    """bundle_config_warnings' lines about the QnnHtp `poll` flag."""
    out = []
    poll, where = read_poll_setting()
    # Say so rather than pick silently. _pick_poll's rule is a heuristic, and a
    # heuristic that resolves a genuine conflict without mentioning it is how a
    # wrong value reaches /props looking authoritative.
    decided = {bool(v) for v, _p in (_POLL_MATCHES or []) if v is not None}
    if len(decided) > 1:
        out.append(
            "WARNING: genie_config.json defines `poll` in %d places with "
            "conflicting values (%s). Using %s=%s; confirm that is the QnnHtp "
            "backend's copy, because the others are being ignored."
            % (len(_POLL_MATCHES),
               ", ".join("%s=%s" % (p, v) for v, p in _POLL_MATCHES),
               where, poll))
    # Absence first, then TRUTHINESS -- not `poll is True`. Identity-strict was
    # wrong for the thing being guarded: JSON `true` parses to Python True, but
    # `1` and `"true"` are valid config, both busy-wait, and both fell through
    # the old `is True` AND the `is None` below to produce no warning at all.
    # Silently accepting the expensive setting is the one outcome this check
    # exists to prevent, so it now fires on anything truthy and reports the
    # value as written rather than asserting "= true".
    if poll is None:
        out.append(
            "note: no `poll` key found in genie_config.json. The shipped "
            "default is true, which busy-waits on ~2.7 cores; if this bundle "
            "is slower than expected, add \"poll\": false to its QnnHtp block.")
    elif poll:
        out.append(
            "WARNING: this bundle has %s = %s. It busy-waits: ~2.7 host "
            "cores burned while IDLE, up to 36%% of decode lost, and about a "
            "quarter of the NPU+GPU concurrency win given away. Set it to "
            "false in genie_config.json and restart -- nothing measured got "
            "worse." % (where or "QnnHtp.poll", json.dumps(poll)))
    return out


def _sampler_warnings():
    """bundle_config_warnings' lines about the repetition penalty."""
    out = []
    # Only when a sampler block was actually read: {} is read_sampler's "no
    # answer", and a missing-penalty warning built on it would be a claim about
    # a block nobody saw. A config that failed to PARSE never gets this far --
    # bundle_config_warnings says that instead, once, rather than letting each
    # check restate it in different words as if it were a separate problem.
    sampler = read_sampler()
    if sampler:
        state = sampler_penalty_state(sampler)
        # Measured values, NOT the vendor's. Qualcomm's reference config for
        # this stack sets repetition-penalty 2.3 / presence 0.7 / frequency
        # 0.8, and at 2.3 this model stops repeating and starts MANGLING
        # instead: measured here, one 400-token answer rendered the same two
        # proper nouns as "MCPWeekly", "MPC Week", "MP Weekly", "MPWeekly" and
        # "YaLLABS" / "YaLLLab" / "YaLLab". For an agent workload that is worse
        # than the loop it fixes -- file paths and identifiers are exactly what
        # must survive verbatim. At 1.15 the same probe kept
        # src/genie_server.py, build_windowed, MCP Weekly and YawLabs all
        # byte-exact with no repeated sentences. Same precedent as `poll`
        # above: the vendor default is a starting point, not the answer.
        fix = ('add "token-penalty": {"version": 1, "penalize-last-n": 128, '
               '"repetition-penalty": 1.15, "presence-penalty": 0.0, '
               '"frequency-penalty": 0.3} to dialog.sampler in '
               "genie_config.json and restart. Sampling binds at "
               "GenieDialog_create, so it cannot be set per request. "
               "(Qualcomm's reference is 2.3/0.7/0.8, which measured here as "
               "aggressive enough to corrupt identifiers.)")
        if state == "absent":
            out.append(
                "WARNING: this bundle's sampler has no `token-penalty` block, "
                "so every repetition penalty defaults to 0 and NOTHING "
                "suppresses a loop. Long generations degenerate into the same "
                "paragraph repeated to max_tokens. To fix: " + fix)
        elif state == "no-window":
            out.append(
                "WARNING: `penalize-last-n` is 0, so this bundle's repetition "
                "penalties are applied to an empty window and do nothing. Set "
                "it (Qualcomm's reference is 64) or the penalties beside it "
                "are decorative.")
        elif state == "all-zero":
            out.append(
                "WARNING: this bundle's `token-penalty` window is set but every "
                "penalty in it is 0, so nothing is suppressed. To fix: " + fix)
    return out


def bundle_config_warnings():
    """Lines to print about a bundle configured to be slower than it needs to be.

    The settings below are worth more than anything else this server does, and
    each was previously left to whoever remembered to read the docs. This
    repo's habit everywhere else -- placement, port, Hexagon arch -- is to
    DERIVE the fact from the artifact and say so out loud rather than hope. This
    is that habit applied to the ones it had missed: two that decide throughput,
    and one that decides whether the output is usable at all.

    Warn, never refuse: a bundle is a large external artifact and a slow server
    is still a working one. Refusing to start would turn a performance note into
    an outage.
    """
    # Nothing to say about a bundle that is not there. load_engine is about to
    # exit naming the env vars, and a note in front of it reads as "your bundle
    # is misconfigured" when the answer is "you have not pointed me at one".
    if not config_present():
        return []
    out = []
    # PRESENT but not JSON: say THAT, and nothing about what is inside it. The
    # two checks below read keys out of the config, and on a file that failed
    # to parse their could-not-read values came out as findings -- "no `poll`
    # key found in genie_config.json", about a file nobody had managed to
    # read. Skipped, and said to be skipped: a check that did not run must not
    # read as a check that passed.
    broken = config_parse_error()
    if broken:
        out.append(
            "WARNING: could not parse genie_config.json (%s). Nothing can be "
            "read from it, so the `poll` and sampler checks were SKIPPED, not "
            "passed; fix the syntax error and restart." % broken)
    else:
        out.extend(_poll_warnings())
        out.extend(_sampler_warnings())
    lengths = read_context_lengths()
    if len(lengths) == 1:
        out.append(
            "WARNING: this is a SINGLE-length bundle (genie.context_lengths = "
            "%s). It runs every token against its whole compiled window, "
            "measured 2-3x slower on short prompts than a multi-length bundle "
            "of the SAME window. Re-export with several --context-lengths "
            "(+3.8%% size, zero extra HTP memory)." % lengths)
    return out


# "::1" is listed as loopback AND bindable: Server derives its address family
# from the host, so the IPv6 loopback is a working GENIE_HOST rather than one
# that passes every startup check and dies in the bind after the model load.
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def host_exposure_warning(host=None):
    """One WARNING line when the bind address is not loopback, else None.

    A sibling of bundle_config_warnings and the same contract: warn, never
    refuse. Exposing this deliberately behind a proxy is a legitimate thing to
    do, and the docs describe how. Doing it without knowing there is no auth in
    front of it is not, and the docs used to describe that too -- 0.0.0.0 was
    written up as a supported option with nothing attached about what it costs.
    """
    host = HOST if host is None else host
    if host in LOOPBACK_HOSTS:
        return None
    return ("WARNING: bound to %s, which is not loopback, and this server has "
            "NO authentication. Anyone who can reach this port can use the "
            "NPU, read what it generates, and wedge the device for every other "
            "client. Put something in front of it, or set "
            "GENIE_HOST=127.0.0.1." % host)


LIB_DIR = os.path.join(SDK_DIR, "lib", "aarch64-windows-msvc")


def hexagon_search_path():
    """Skel dirs for every Hexagon this box can ACTUALLY drive.

    A Hexagon is usable here only if the SDK ships BOTH halves:
      * lib/hexagon-vNN/unsigned/                    -- the DSP-side skel
      * lib/aarch64-windows-msvc/QnnHtpVNNStub.dll   -- the Windows-side stub

    This used to be hardcoded to hexagon-v73, which excluded X2 Elite (v81).
    Globbing every skel was the other extreme: QAIRT 2.45 ships skels for
    v66..v81 but Windows stubs for only a subset, because the rest are Android
    parts -- skel present, no way to reach it from Windows. Offering those
    would be a promise the box cannot keep.

    Intersecting the two halves is what makes the supported set
    self-maintaining: on QAIRT 2.45 v68, v73 and v81 fall out, a future
    Hexagon falls out the day QAIRT ships both halves for it, and nothing here
    has to be edited.

    Which archs those are is deliberately NOT restated as fact anywhere this
    function does not compute it. Three places in this repo claimed the answer
    was v73 and v81; the first real startup log printed three, including v68
    (8cx Gen 3 / Dev Kit 2023), because that stub ships too. The derivation was
    right the whole time and the prose beside it was stale.

    GENIE_HEXAGON_ARCH ("v81") pins one arch if you need to force it.
    Returns (path_string, usable_archs, skel_only_archs).
    """
    import glob
    import re
    stubs = set()
    for f in glob.glob(os.path.join(LIB_DIR, "QnnHtpV*Stub.dll")):
        m = re.match(r"QnnHtpV(\d+)Stub\.dll$", os.path.basename(f))
        if m:
            stubs.add("v" + m.group(1))

    pin = os.environ.get("GENIE_HEXAGON_ARCH", "").strip()
    usable, skel_only, dirs = [], [], []
    for d in sorted(glob.glob(os.path.join(SDK_DIR, "lib", "hexagon-v*", "unsigned"))):
        if not os.path.isdir(d):
            continue
        arch = os.path.basename(os.path.dirname(d)).replace("hexagon-", "")
        if arch not in stubs:
            skel_only.append(arch)
        elif not pin or arch == pin:
            usable.append(arch)
            dirs.append(d)
    return os.pathsep.join(dirs), usable, skel_only


# ---------------------------------------------------------------------------
# Genie C API (from include/Genie/GenieDialog.h, GenieCommon.h,
# GenieSampler.h and GenieTokenizer.h)
# ---------------------------------------------------------------------------
GENIE_STATUS_SUCCESS = 0
# The four non-fatal warnings, verbatim from GenieCommon.h (QAIRT 2.45).
# CONTEXT_EXCEEDED was declared here as 1 for a long time, which is the value
# of ABORTED -- so a context-full generation fell through to the raise below
# and surfaced as a 500, while an aborted one reported "length". Both are
# wrong and they are each other's symptom, which is why the whole set is
# declared now rather than the one constant that happened to be needed.
#
# Only two of the four are ways a generation ENDS: ABORTED (this server's own
# signal_abort landing) and CONTEXT_EXCEEDED (the window is full). The header
# defines BOUND_HANDLE and PAUSED beside them with no prose, and nothing this
# server does can produce either from a query -- it sends no PAUSE signal and
# frees no handle mid-query -- so _finish raises on them BY NAME rather than
# decoding them into a plausible finish_reason. They are declared so the
# raise can say which warning it was instead of a bare "status=2".
GENIE_STATUS_WARNING_ABORTED = 1
GENIE_STATUS_WARNING_BOUND_HANDLE = 2
GENIE_STATUS_WARNING_PAUSED = 3
GENIE_STATUS_WARNING_CONTEXT_EXCEEDED = 4
GENIE_STATUS_NAMES = {
    GENIE_STATUS_SUCCESS: "SUCCESS",
    GENIE_STATUS_WARNING_ABORTED: "WARNING_ABORTED",
    GENIE_STATUS_WARNING_BOUND_HANDLE: "WARNING_BOUND_HANDLE",
    GENIE_STATUS_WARNING_PAUSED: "WARNING_PAUSED",
    GENIE_STATUS_WARNING_CONTEXT_EXCEEDED: "WARNING_CONTEXT_EXCEEDED",
}
# The statuses a generation can END in, i.e. the ones _finish decodes. This
# is ALSO what EngineHealth counts as "ok": an abort is this server's own
# doing and a full window is a normal finish, and both used to be booked as
# failures because the health call compared against SUCCESS alone -- three
# stop-button presses in a row flipped /health to 503 "failing" on an engine
# that had done nothing wrong, and a router honouring the documented "503 =
# shed" contract would have kept it shed until a generation happened to
# succeed. One set, used by both, so the two cannot disagree again. (Health
# adds one condition the status cannot carry: an ABORTED that the watchdog
# sent, for a stall, is a failure -- see _run_query.)
GENIE_FINISHED_STATUSES = frozenset((GENIE_STATUS_SUCCESS,
                                     GENIE_STATUS_WARNING_ABORTED,
                                     GENIE_STATUS_WARNING_CONTEXT_EXCEEDED))

# GenieDialog_SentenceCode_t
SENTENCE_COMPLETE = 0
SENTENCE_BEGIN = 1
SENTENCE_CONTINUE = 2
SENTENCE_END = 3
SENTENCE_ABORT = 4

# GenieDialog_Action_t
GENIE_DIALOG_ACTION_ABORT = 0x01

Handle = C.c_void_p
# void callback(const char* response, int sentenceCode, const void* userData)
QUERY_CALLBACK = C.CFUNCTYPE(None, C.c_char_p, C.c_int, C.c_void_p)
# void alloc(const size_t size, const char** allocatedData)  (GenieCommon.h)
ALLOC_CALLBACK = C.CFUNCTYPE(None, C.c_size_t, C.POINTER(C.c_char_p))


# Qwen3's tool convention, lifted verbatim from the bundle's own
# tokenizer_config.json chat_template (the Jinja one). We render it by hand
# because this server is stdlib-only -- no Jinja -- but the strings and the
# ordering below are the template's, not invented. Qwen3 is (c) Alibaba
# Cloud and Apache-2.0 licensed; see NOTICE.
_TOOLS_PREAMBLE_HEAD = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>"""

_TOOLS_PREAMBLE_TAIL = """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

_NL = chr(10)
_SSE_GAP = (chr(10) * 2).encode("utf-8")   # blank line terminating an SSE frame

# Straight from the bundle's Jinja: enable_thinking=false prefills a CLOSED,
# empty think block so the model resumes after it instead of opening its own.
_NO_THINK = "<think>" + _NL + _NL + "</think>" + _NL + _NL

# <tool_call>{...}</tool_call> and the aliases this model actually produces.
# <tool_call> is the trained, in-vocab tag, but with the reasoning block
# suppressed Qwen3 improvises: <function_call> was observed on this box for the
# same prompt that wrapped correctly with thinking on. The backreference forces
# the closing tag to match the opening one, and the payload still has to parse
# as a call -- so widening the alternation cannot turn prose into a tool call.
# DOTALL so a pretty-printed argument object matches.
_TOOL_CALL_RE = re.compile(
    r"<(?P<tag>tool_call|function_call|tool_use)>\s*(?P<body>.*?)\s*</(?P=tag)>", re.DOTALL)


def _tool_arguments(raw):
    """(arguments, usable) for one call's `arguments` field.

    OpenAI sends `arguments` as a JSON STRING containing an object, Qwen3 often
    emits the object directly, and both shapes have to land on the same thing
    because a client cannot tell them apart. What this pins is the TYPE
    CONTRACT, which nothing previously enforced: the field came back as dict OR
    str OR int OR None depending on what the string happened to hold, and the
    downstream consumers cannot take that. The OpenAI emitter json.dumps it
    (_tool_args_json) and the Anthropic one puts it in tool_use.input, where
    Anthropic requires an object -- so a bare int here produced a block no
    Anthropic client accepts.

    Two cases that look alike and are not:

      * "" (or whitespace) is a WELL-FORMED zero-argument call, not junk. It
        parsed to nothing and fell through to the raw string, so `arguments`
        came back as "" where {} is what the call means -- and inconsistently
        with an OMITTED `arguments`, three lines away, which already defaulted
        to {}. Both now mean {}.

      * anything that does not resolve to an object -- unparseable, or valid
        JSON that is a scalar or a list -- is not a usable call. Coercing it to
        {} would invent an argument-free call the model never made, so it is
        reported unusable and the caller leaves the raw block VISIBLE, exactly
        as it already does for a block whose body does not parse. The model's
        output survives where a client can see it.
    """
    if isinstance(raw, dict):
        return raw, True
    if raw is None:
        return {}, True
    if not isinstance(raw, str):
        return None, False          # a number or a list is not an argument set
    if not raw.strip():
        return {}, True             # zero-argument call
    try:
        parsed = json.loads(raw)
    except Exception:
        return None, False
    return (parsed, True) if isinstance(parsed, dict) else (None, False)


def _bare_tool_calls(text):
    """Accept a whole-output JSON blob that is unambiguously a tool call.

    Suppressing the reasoning block makes Qwen3 sometimes emit the call JSON
    BARE -- correct name and arguments, no <tool_call> tags. Observed on this
    box: the same prompt wraps correctly with thinking on and skips the tags
    with it off. The caller asked for tools and the model produced a valid
    call, so recognising it is right; handing back a JSON blob as "content"
    would make every client re-implement this parse.

    Deliberately strict: whole output only (no prose around it), and BOTH
    "name" and "arguments" required. A bare {"name": ...} could be an ordinary
    JSON answer -- the pair together is the documented call shape and little
    else. Anything less certain stays text.
    """
    t = (text or "").strip()
    if not (t.startswith("{") or t.startswith("[")):
        return []
    try:
        obj = json.loads(t)
    except Exception:
        return []
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for o in items:
        if not (isinstance(o, dict) and "name" in o and "arguments" in o):
            return []
        args, usable = _tool_arguments(o["arguments"])
        if not usable:
            return []               # not certainly a call -> stays text
        calls.append({"name": o["name"], "arguments": args})
    return calls


def parse_tool_calls(text):
    """Split generated text into (visible_text, [{name, arguments}, ...]).

    Returns calls only when the JSON actually parses. A malformed block is
    LEFT IN the visible text rather than silently dropped -- the caller can
    then see what the model emitted instead of getting a mystery empty
    response, which is the same reason placement is asserted in bench.py.
    """
    calls, spans = [], []
    for m in _TOOL_CALL_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group("body"))
            name = obj["name"]
        except Exception:
            continue  # malformed -> leave the raw block visible
        args, usable = _tool_arguments(obj.get("arguments", {}))
        if not usable:
            continue  # same treatment as a malformed body: leave it visible
        calls.append({"name": name, "arguments": args})
        spans.append(m.span())
    out, prev = [], 0
    for a, b in spans:
        out.append(text[prev:a])
        prev = b
    out.append(text[prev:])
    visible = "".join(out).strip()
    if not calls:
        bare = _bare_tool_calls(visible)
        if bare:
            return "", bare
    return visible, calls


def _content_text(content):
    """Flatten a message `content` value to plain text.

    Both APIs allow content to be a LIST of blocks, not just a string --
    OpenAI SDKs emit [{"type":"text","text":...}] by default. ChatML.build
    string-concatenates, so an unflattened list raised TypeError and killed the
    handler thread, handing the client a dropped connection with no error body.
    The Anthropic path already flattened; this is the shared version so the two
    endpoints cannot drift apart again.

    Text and tool_result blocks contribute text; images and tool_use do not.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # A BARE block, one wrapper short of the list shape above:
    # {"type":"text","text":"what is 2+2"} sent as `content` itself. Clients
    # write it by hand, and a tool_result's own `content` arrives this way.
    # It used to fall through to "" -- the user's question, or a tool result's
    # file body, gone from the prompt while the request was answered 200. That
    # is the silent context loss this function exists to end, so it flattens
    # exactly as the one-element list it means; a block with no text in it
    # (an image, a tool_use) still contributes nothing, because that is what
    # the same block contributes inside a list.
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        # Nothing here can be rendered: a number, a bool. Refuse rather than
        # invent -- str(content) is the "stringify anything" rule the list
        # branch already refuses, and returning "" sends the model a turn with
        # no words in it. Raising is how every other unrenderable field in this
        # file answers (do_POST's catch-all turns it into a 400 naming the
        # type), and a 400 the client can read beats a 200 it cannot.
        raise TypeError("message content must be a string, a list of content "
                        "blocks, or one content block -- not %s"
                        % type(content).__name__)
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_result":
                parts.append(_content_text(b.get("content")))
    return "".join(parts)


# OpenAI renamed the system role to "developer": same instructions, current
# spelling, and its SDKs emit it. The renderer's unknown-role fall-through made
# one an ordinary USER turn, and that is wrong twice over -- the agent's
# instructions became evictable, so the model got dumber as the conversation
# grew, and with no message left of role "system" the template's
# default_system ("You are a helpful AI assistant.") was injected in FRONT of
# them, contradicting the instructions with a prompt nobody sent. Both halves
# are silent, which is the one thing this server refuses. So the two spellings
# are ONE role everywhere the system turn is decided: here, in the renderer,
# and at _fit's unit boundary -- "two spellings of it would disagree" is the
# argument for a single definition below, and it applies to this too.
#
# Any OTHER unrecognised role still renders as a user turn. There is nothing
# better to do with a role we do not know: dropping it would lose the words,
# which is the failure above, and it is at least a turn someone typed.
SYSTEM_ROLES = ("system", "developer")

# Roles that cannot START the unit being answered, and are never left at the
# head of what eviction keeps: a tool result belongs to the assistant turn
# that called it, and an instruction is not the turn it is an instruction for.
_NOT_UNIT_START = ("tool", *SYSTEM_ROLES)


def _leading_system(messages):
    """How many system messages OPEN the conversation: the run of them before
    the first turn of any other role.

    Those, and only those, are the system turn -- the one thing eviction never
    touches. A role=system message further in is a turn like any other (see
    _system_text for why the line is drawn here). One definition, shared by
    the renderer, the evictor and the note helpers, because "where does the
    system turn end" is what decides whether a retained note is found again --
    see _split_note -- and two spellings of it would disagree.
    """
    n = 0
    while n < len(messages) and messages[n].get("role") in SYSTEM_ROLES:
        n += 1
    return n


def _system_text(messages):
    """The text of the ONE system turn: every LEADING system message, in order.

    Joined with a blank line, empties skipped. The renderer used to take the
    first system message and `break`, while its body loop skipped EVERY system
    message as "already folded into the system turn" -- so a second one was
    dropped from the prompt with no log line and no error, in a server whose
    rule for lost context is that it is never silent. Real clients send more
    than one: a trailing system reminder appended per turn, a framework's
    per-turn injection, a narrator's event log.

    Leading ones only, though. Joining ALL of them in here, wherever they sat,
    is the obvious fix for that and the wrong one, because the system turn is
    unevictable: a client that keeps a per-turn reminder in its history then
    grows the one turn _fit can never shrink by a message per exchange.
    Measured at n_ctx=2048 with 400-char reminders: by 18 exchanges the stale
    reminders had pushed 34 of the 37 real turns out of the window, and by 20
    nothing fitted at all -- a 400 whose advice ("send a shorter message")
    cannot help, repeated on every later request because the client resends
    the same history. A system message AFTER the first turn of another role
    is therefore an ordinary turn: rendered inline at its position as its own
    system block, which is what the bundle's Jinja does with it, and evicted
    in order like the turns around it (ChatML.build, _fit).
    """
    texts = [_content_text(m.get("content"))
             for m in messages[:_leading_system(messages)]]
    return (_NL + _NL).join(t for t in texts if t)


class ChatML:
    """Prompt formatting from the bundle metadata's chat_template."""

    def __init__(self, tmpl):
        self.sys_pre = tmpl["system_prefix"]
        self.sys_suf = tmpl["system_suffix"]
        self.usr_pre = tmpl["user_prefix"]
        self.usr_suf = tmpl["user_suffix"]
        self.asst_pre = tmpl["assistant_prefix"]
        self.asst_suf = tmpl["assistant_suffix"]
        self.default_system = tmpl.get("default_system_prompt", "")

    def build(self, messages, tools=None, thinking=None):
        """Assemble a ChatML prompt ending with an open assistant turn.

        `thinking=None` means "whatever the server is configured to do", read
        from THINKING_DEFAULT. Spelling the default as a literal `True` here is
        what let this drift out of step with the policy when that flipped: the
        signature kept promising reasoning-enabled prompts the server would
        never itself produce. One source of truth, so it cannot happen twice.

        `tools` renders Qwen3's tool preamble into the system turn; assistant
        `tool_calls` and role="tool" results round-trip in the same shapes the
        bundle's own Jinja template uses, so a multi-turn tool conversation
        replays exactly as the model was trained to see it.

        `tools` must already be a list of objects -- _is_tool_list, checked at
        the door. This loop json.dumps whatever it iterates: a string came out
        as one quoted "function signature" per CHARACTER and a dict as its
        keys, a garbage schema served with a 200.

        One deliberate departure from the bundle's Jinja: the LEADING system
        messages -- those before the first turn of any other role -- are
        folded into the single system turn (_system_text), where the Jinja
        renders the second of two as a block of its own. That run is the
        system turn for every purpose here: it is what eviction anchors and
        what a retained note rides at the end of. A system message later in
        the conversation is rendered as the Jinja renders it, inline at its
        position as its own system block, and is an ordinary evictable turn
        (_system_text says what happened while those were folded in too). An
        EMPTY one renders nothing, as an empty leading one contributes nothing.

        Every json.dumps below is ensure_ascii=False, which is what the
        template's `tojson` does. The default escapes each non-ASCII character
        to a six-character backslash-u sequence, and for tool ARGUMENTS that is
        not cosmetic: the model emitted raw UTF-8, the engine committed exactly
        those bytes as the resident KV, and a history render that turns a path
        argument's one accented letter into an escape no longer starts with
        what the dialog holds -- the byte-prefix check refuses, correctly, and
        every turn after a tool call with a non-ASCII argument re-prefills the
        whole conversation. A string `arguments` is spliced VERBATIM, as the
        template does, so that half is only as exact as the bytes the response
        handed the client.
        """
        if thinking is None:
            thinking = THINKING_DEFAULT
        parts = []
        sys_text = _system_text(messages)
        if not sys_text and self.default_system:
            sys_text = self.default_system

        if tools:
            body = sys_text + (_NL + _NL if sys_text else "")
            body += _TOOLS_PREAMBLE_HEAD
            for t in tools:
                body += _NL + json.dumps(t, ensure_ascii=False)
            body += _TOOLS_PREAMBLE_TAIL
            parts.append(self.sys_pre + body + self.sys_suf)
        elif sys_text:
            parts.append(self.sys_pre + sys_text + self.sys_suf)

        # From the first turn that is not a leading system message: those are
        # in the system turn above.
        i, n = _leading_system(messages), len(messages)
        while i < n:
            m = messages[i]
            role = m.get("role", "user")
            content = _content_text(m.get("content"))
            if role in SYSTEM_ROLES:
                if content:
                    parts.append(self.sys_pre + content + self.sys_suf)
                i += 1
                continue
            if role == "tool":
                # Consecutive tool results share ONE user turn, per the template.
                chunk = []
                while i < n and messages[i].get("role") == "tool":
                    c = _content_text(messages[i].get("content"))
                    chunk.append("<tool_response>" + _NL + c + _NL + "</tool_response>")
                    i += 1
                parts.append(self.usr_pre + _NL.join(chunk) + self.usr_suf)
                continue
            if role == "assistant":
                # With thinking suppressed, the dialog's KV holds the prefilled
                # <think></think> in front of every assistant turn, because that
                # is what we sent. Re-rendering history WITHOUT it makes the
                # prompt diverge from resident state by exactly that string --
                # which silently defeats KV reuse (the prefix check refuses, as
                # it should) and re-prefills every turn. Render history the way
                # it was actually generated.
                body = ("" if thinking else _NO_THINK) + content
                # A newline goes in front of a call only when something of the
                # TURN'S OWN precedes it -- its content, or an earlier call --
                # which is the template's rule. This tested `if body:`, which
                # was the same thing until the prefill above joined `body`:
                # with thinking off it is never empty, so a turn that was
                # nothing but a call re-rendered with a newline the model never
                # emitted. One byte, and the prefix check refused reuse after
                # EVERY tool call -- the exact failure the prefill was put into
                # history to prevent.
                said = bool(content)
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    if said:
                        body += _NL
                    said = True
                    body += ("<tool_call>" + _NL + '{"name": "' + fn.get("name", "")
                             + '", "arguments": ' + args + "}" + _NL + "</tool_call>")
                parts.append(self.asst_pre + body + self.asst_suf)
                i += 1
                continue
            parts.append(self.usr_pre + content + self.usr_suf)
            i += 1

        parts.append(self.asst_pre)  # open assistant turn for generation
        if not thinking:
            parts.append(_NO_THINK)
        return "".join(parts)


# --- supervision ------------------------------------------------------------
# A wedged HTP is not a per-request failure and cannot be handled like one. The
# Genie query is a blocking call into native code: when the device stops making
# progress the calling thread is stuck inside the driver, holding the engine
# lock, and Python cannot reclaim it -- no timeout, no interrupt, no kill. Every
# later request then parks behind that lock until MAX_INFLIGHT is exhausted and
# the rest get a fast 429, which is why this presents from the outside as a
# server that 429s forever while sitting completely idle.
#
# Two consequences shape everything below. First, /health MUST stop saying "ok",
# because a health check that passes while nothing can be served is worse than
# no health check -- it is the signal a supervisor trusts to decide not to act.
# Second, the only real recovery is a fresh process: the stuck thread cannot be
# reclaimed in-process, so the honest move is to exit and let a supervisor
# restart, rather than linger in a state that answers nothing.
#
# Detection is by STALLED PROGRESS, not elapsed time. A long generation is not
# a wedge -- 2000 tokens at the slowest measured 3.3 t/s is ten minutes of
# perfectly healthy work -- but it emits tokens the whole way. A wedge emits
# nothing. So the clock that matters is time since the last token, which
# separates "slow" from "stopped" without capping how long a request may run.
FIRST_TOKEN_TIMEOUT_S = _float_env("GENIE_FIRST_TOKEN_TIMEOUT", 300.0)
STALL_TIMEOUT_S = _float_env("GENIE_STALL_TIMEOUT", 120.0)
WEDGE_GRACE_S = _float_env("GENIE_WEDGE_GRACE", 60.0)
FAIL_THRESHOLD = max(1, _int_env("GENIE_FAIL_THRESHOLD", 3))
# Exit rather than linger. 75 is EX_TEMPFAIL: "temporary failure, try again",
# which is exactly what a supervisor should read from it.
EXIT_WEDGED = 75
WEDGE_EXIT = os.environ.get("GENIE_WEDGE_EXIT", "1") not in ("0", "false", "no")
# How long shutdown waits for an aborted generation to release the engine lock
# before giving up on GenieDialog_free. An abort takes effect within one decode
# step (under a second at the slowest measured 3.3 t/s), so a lock still held
# after this is a stuck driver, and freeing under a stuck driver can hang too.
SHUTDOWN_FREE_TIMEOUT_S = 5.0


class EngineHealth:
    """Whether the resident engine can actually serve, as a state machine.

    Deliberately free of any Genie or socket dependency: it takes timestamps and
    returns a verdict, so the escalation logic can be tested without a device.
    The thing being guarded against is untestable by nature (a wedged driver),
    which is exactly why the DECISION about it has to be testable.
    """

    def __init__(self, first_token_timeout=None, stall_timeout=None,
                 grace=None, fail_threshold=None):
        self.first_token_timeout = (FIRST_TOKEN_TIMEOUT_S if first_token_timeout
                                    is None else first_token_timeout)
        self.stall_timeout = (STALL_TIMEOUT_S if stall_timeout is None
                              else stall_timeout)
        self.grace = WEDGE_GRACE_S if grace is None else grace
        self.fail_threshold = (FAIL_THRESHOLD if fail_threshold is None
                               else fail_threshold)
        self._lock = threading.Lock()
        self.started = None        # when the in-flight generation began
        self.last_progress = None  # when it last produced a token
        self.tokens = 0
        self.consecutive_failures = 0
        self.generations = 0
        self.stall_signalled_at = None   # when we first tried to abort a stall
        self.stall_native = True         # ...and whether an ABORT really went
        # A host-side native call that is NOT a generation -- see native_begin.
        self.native_since = None
        self.native_what = ""

    def begin(self, now):
        """A generation has the engine lock and is about to call into Genie.

        Called at lock ACQUISITION, ahead of the turn's first native call --
        the reset, the stop sequences, the sampler, the token cap -- and not
        just ahead of GenieDialog_query. Every one of those is a call into the
        same driver and can wedge exactly as the query can, and while this ran
        after them a hang in any of them left `started` None: /health said ok
        for as long as the outage lasted while every request 429d behind the
        lock, which is precisely the hole the supervision block above exists
        to close. Still AFTER the lock, though: a request waiting for the
        engine is not a stalled one, and counting it as such would let a busy
        server look wedged.
        """
        with self._lock:
            self.started = now
            self.last_progress = None
            self.tokens = 0
            self.stall_signalled_at = None

    def native_begin(self, now, what):
        """A native call that is NOT a generation has the engine lock.

        Today that is the tokenizer encode that sizes every request. It is a
        call into the same driver and can wedge the same way, so it gets the
        stall clock -- but it is not a generation, so it is not `generating`
        on /health, is not counted, and cannot touch the failure streak: a
        successful encode clearing three failed generations would report an
        engine that is failing as one that has recovered. Its own timestamp,
        rather than begin()'s, is what keeps those apart.
        """
        with self._lock:
            self.native_since = now
            self.native_what = what
            self.stall_signalled_at = None

    def native_end(self):
        with self._lock:
            self.native_since = None
            self.native_what = ""

    def progress(self, now):
        """A token came back. Called from Genie's callback thread, so it stays
        to a lock and two assignments -- this runs per token."""
        with self._lock:
            self.last_progress = now
            self.tokens += 1

    def end(self, ok, counted=True):
        """Close out a generation. `ok` is "it ended", not "it succeeded":
        GenieEngine passes `status in GENIE_FINISHED_STATUSES`, so an abort or
        a full window is not a failure but an error status or a throw is --
        and so is the one abort that is not a client's: the watchdog's, on a
        turn that had stalled (_Turn.stalled). A stall that an abort happened
        to clear is still the engine not serving.

        `counted=False` is for the server's OWN calls into the engine -- today
        just summarising evicted turns. Those still get full stall and failure
        supervision, because they run on the same device and can wedge it
        exactly as a client request can. What they are not is traffic anyone
        asked for, so counting them made `generations` on /health report more
        work served than any client ever requested -- a diagnostic that drifts
        from the thing it describes, which is what this endpoint exists to end.

        No timestamp: it used to take a `now` that every caller passed and
        nothing read. A parameter that does nothing is a claim the signature
        makes and the body does not keep.
        """
        with self._lock:
            self.started = None
            self.last_progress = None
            self.stall_signalled_at = None
            self.tokens = 0
            if counted:
                self.generations += 1
            self.consecutive_failures = 0 if ok else self.consecutive_failures + 1

    def note_stall_signalled(self, now, native=True):
        """The watchdog acted on a stall. The grace clock runs from the FIRST.

        `native` says whether a GenieDialog_signal actually went out. It does
        not when the stall is outside GenieDialog_query -- in the tokenizer, or
        in one of a turn's pre-query calls -- and `wedged` must not then report
        that an abort "did not take": none was sent, and that sentence is what
        an operator files against the driver.
        """
        with self._lock:
            if self.stall_signalled_at is None:
                self.stall_signalled_at = now
            self.stall_native = native

    def assess(self, now):
        """(state, detail). One of ok / failing / stalled / wedged.

        `stalled` means an abort is worth trying; `wedged` means it was tried
        and did not take, so the process is the only thing left to replace.
        """
        with self._lock:
            if self.started is not None:
                since_start = now - self.started
                if self.last_progress is None:
                    waited, limit, what = since_start, self.first_token_timeout, "first token"
                else:
                    waited, limit, what = (now - self.last_progress,
                                           self.stall_timeout, "further token")
                if waited <= limit:
                    return "ok", ""
                detail = ("no %s for %.0fs (limit %.0fs) after %d token(s); the "
                          "HTP has stopped making progress"
                          % (what, waited, limit, self.tokens))
            elif (self.native_since is not None
                    and now - self.native_since > self.first_token_timeout):
                # The first-token limit, because that is the generous one: an
                # encode answers in milliseconds, so anything past a limit
                # sized for prefill at depth is a driver that has stopped
                # answering host-side calls, not a slow encode.
                detail = ("no return from %s for %.0fs (limit %.0fs); the "
                          "driver has stopped answering host-side calls"
                          % (self.native_what, now - self.native_since,
                             self.first_token_timeout))
            else:
                if self.consecutive_failures >= self.fail_threshold:
                    return "failing", (
                        "%d consecutive generation failures; the engine is "
                        "returning errors rather than output"
                        % self.consecutive_failures)
                return "ok", ""
            if (self.stall_signalled_at is not None
                    and now - self.stall_signalled_at > self.grace):
                if self.stall_native:
                    return "wedged", detail + (
                        "; an abort was signalled %.0fs ago and did not take"
                        % (now - self.stall_signalled_at))
                return "wedged", detail + (
                    "; first seen %.0fs ago and still stuck. No ABORT was "
                    "sent: the stall is outside GenieDialog_query, where "
                    "none can be delivered"
                    % (now - self.stall_signalled_at))
            return "stalled", detail

    def snapshot(self, now):
        """What /health reports. Plain data, safe to call at any time."""
        state, detail = self.assess(now)
        with self._lock:
            return {
                "state": state,
                "detail": detail,
                "generating": self.started is not None,
                "tokens_in_flight": self.tokens,
                "generations": self.generations,
                "consecutive_failures": self.consecutive_failures,
            }


HEALTH = EngineHealth()


# What an abort DID, for the callers that report on it (the watchdog). False
# means there was no turn to abort. Both of these are truthy, so "was there
# one" still reads as a plain truth test.
ABORT_FLAGGED = "flagged"       # the turn is marked; no native signal was sent
ABORT_SIGNALLED = "signalled"   # GenieDialog_signal(ABORT) has gone to its query


class _Turn:
    """One generation's claim on the dialog: what an abort is aimed at.

    Every flag is read and written under GenieEngine._abort_lock only.
    `aborted` says someone asked for this turn to stop; `done` says its
    GenieDialog_query has returned (or was never started), after which an
    abort aimed at it is a no-op -- there is nothing of THIS turn's left to
    stop, and the dialog may already be another request's. `stalled` says WHO
    asked: the watchdog, because the turn had stopped making progress. That
    is the one abort that is the engine's fault rather than a client's
    choice, and _run_query books it as a failed generation -- see there.
    """
    __slots__ = ("aborted", "done", "stalled")

    def __init__(self):
        self.aborted = False
        self.done = False
        self.stalled = False


class EngineClosing(RuntimeError):
    """A turn refused at the door because the server is shutting down.

    A RuntimeError still, because that is what the refusal used to raise and
    what every caller here already handles (query_stream's worker turns it
    into result["error"], _summarize_turns falls back to plain eviction). The
    TYPE is what lets the one caller who can do better do better: Handler._run
    answers it 503 rather than 500, because shutting down is exactly the "503
    = shed, try elsewhere" contract this server documents -- the request was
    never started, nothing about it was wrong, and another leg can serve it
    now.
    """


class GenieEngine:
    """Resident Genie dialog on the HTP. All NPU access serialized by a lock."""

    def __init__(self, lib, dialog, tokenizer=None):
        self.lib = lib
        self.dialog = dialog
        self.tokenizer = tokenizer
        self.lock = threading.Lock()
        # Exact text the dialog's KV currently holds; None means "unknown,
        # re-prefill". Set by _commit, cleared on any failure, abort or reset.
        self._committed = None
        # Abort scoping. Every generation is a _Turn, and an abort is aimed at
        # ONE of them -- see signal_abort for who aims at which. `_live` is
        # the turn holding the engine lock, from acquisition until its
        # GenieDialog_query has returned, and `_querying` says it is inside
        # that call, which is the only time the native signal is sent. Both
        # live under `_abort_lock` together with the turn's own flags and the
        # native signal itself, so an abort cannot land on an idle dialog or
        # on ANOTHER request's generation. It could: the signal is
        # engine-global, and a handler whose post-loop write failed after its
        # worker had released the engine lock sent GenieDialog_signal(ABORT)
        # to whoever held the dialog next -- that client's answer came back
        # cut short as an ordinary "stop".
        #
        # A flag on the turn rather than a direct clear of _committed, because
        # the clear alone loses a race: the worker can finish and re-commit
        # AFTER the clear, re-arming reuse against a generation that was cut
        # short. The turn's flag is read once, under _abort_lock, as the turn
        # leaves its window, and nothing can set it after that.
        self._abort_lock = threading.Lock()
        self._live = None
        self._querying = False
        # The turn the CALLING thread is consuming -- what a bare
        # signal_abort() is aimed at.
        self._tls = threading.local()
        # The bundle's own sampler block, read at startup and used as the
        # restore baseline. The dialog is RESIDENT and shared across requests,
        # so a per-request override that is never undone leaks into the next
        # caller -- one request asking for temp 0 would silently make every
        # later request deterministic.
        self.default_sampler = {}
        self._sampler_dirty = False
        self._stop_dirty = False
        # Set by close(), once, and never cleared: GenieDialog_free has run and
        # `dialog` / `tokenizer` are None. Written under BOTH locks, so each
        # native call site reads it under the one it already holds.
        self._closed = False
        # Shutdown has STARTED: the dialog is still alive, but no new turn may
        # take it. Written under _abort_lock alone (begin_shutdown, and close()
        # before it goes for the engine lock), read by _run_query under the
        # engine lock -- it does not gate a native call, it gates a turn, and
        # the turn's gate has to be readable before the free has happened.
        # Never cleared: a close() that gave up is still a server that was
        # asked to stop, and there is no un-asking.
        self._closing = False

    def begin_shutdown(self):
        """Stop admitting turns, THEN abort whichever one holds the dialog.

        Returns what the abort did, like signal_abort.

        The order is the whole of it. An abort frees the engine lock, and the
        lock goes to whoever has waited longest -- which under load is a
        request parked in _run_query, not close(). GENIE_MAX_INFLIGHT is 2, so
        one running plus one queued is an ordinary loaded moment, and on Ctrl-C
        that queued turn used to win the race: it went on to GenieDialog_reset
        and a fresh GenieDialog_query, close() timed out after
        SHUTDOWN_FREE_TIMEOUT_S, printed "a generation is still inside the
        driver" about a driver that was working perfectly, and the process left
        with a generation running inside Genie and the dialog never freed --
        measured 5 of 5 against the real engine. The queued turn was never
        flagged, so nothing aborted it either.

        Marking `_closing` first closes that window: every turn still waiting
        for the lock raises EngineClosing when it gets it instead of starting,
        and close() has the lock one release later. It is set BEFORE the abort
        rather than inside close() because the gap between the two calls is
        itself the race -- the aborted turn can release the lock in it.
        """
        with self._abort_lock:
            self._closing = True
        return self.signal_abort(any_turn=True)

    def close(self, timeout=None):
        """Free the dialog and refuse every native call from then on.

        Returns whether the engine is closed. False means the engine lock
        could not be had within `timeout` seconds -- a generation is still
        inside the driver -- and then no HANDLE is touched: freeing one under
        a live query is a fault, and a free on a stuck driver can hang. The
        one thing that happens either way is `_closing`, set before the wait
        and never cleared: a caller asking to close has decided to stop
        serving, whether or not the driver lets go, and a turn refused
        meanwhile cannot be un-refused.

        The free alone was not enough. Handler and worker threads are
        daemons, so they outlive main()'s shutdown path for as long as the
        interpreter takes to leave, and one of them was always about to make
        another native call: the handler whose generation the shutdown abort
        had just ended goes on to count its usage (GenieTokenizer_encode, on
        the tokenizer that belonged to the freed dialog), and a request
        queued behind the lock goes on to GenieDialog_reset and
        GenieDialog_query. Each takes the engine lock the moment the free
        releases it, and each was a call into a freed handle: at best a
        0xC0000005 in the WER log of a process that was exiting cleanly,
        which is exactly the driver-fault evidence the docs lean on.

        So the flag and the nulled handles are set under the engine lock,
        ahead of the free, and every path that reaches Genie checks under that
        same lock: _run_query raises, _encode_count returns None (the callers'
        documented fallback to the estimate), and _abort -- under _abort_lock,
        which is why the write takes that one too -- sends nothing.

        `_closing` is set here as well as in begin_shutdown so that a bare
        close() -- a caller with no live turn to abort first -- does not hand
        the lock it is waiting for to a turn that would then start a query on
        a dialog about to be freed. begin_shutdown remains the one to call
        with a generation in flight: only setting the flag BEFORE the abort
        closes the window the abort itself opens.
        """
        with self._abort_lock:
            self._closing = True
        if not self.lock.acquire(timeout=-1 if timeout is None else timeout):
            return False
        try:
            with self._abort_lock:
                dialog, self.dialog, self.tokenizer = self.dialog, None, None
                self._closed = True
            if dialog is not None:
                try:
                    self.lib.GenieDialog_free(dialog)
                except Exception:
                    pass
            return True
        finally:
            self.lock.release()

    @staticmethod
    def _finish(status):
        """Genie status -> OpenAI finish_reason, for the statuses a query ENDS in.

        ABORTED is not a failure: it is this server's own signal_abort landing
        after the client hung up, so it reports like any other early stop
        rather than raising into a request nobody is reading any more.

        Anything else raises, and raises by NAME where the header has one.
        BOUND_HANDLE and PAUSED are warnings too, but neither is a way a query
        ends here -- nothing sends PAUSE and nothing frees a handle mid-query
        -- so decoding them into a plausible finish_reason would be a guess
        dressed as a result. A bare "status=2" sent the reader to the header;
        the name says what to look for.
        """
        if status == GENIE_STATUS_WARNING_CONTEXT_EXCEEDED:
            return "length"
        if status in (GENIE_STATUS_SUCCESS, GENIE_STATUS_WARNING_ABORTED):
            return "stop"
        name = GENIE_STATUS_NAMES.get(status)
        raise RuntimeError("GenieDialog_query failed, status=%s%s"
                           % (status, " (%s)" % name if name else ""))

    def set_stop_sequences(self, seqs):
        """Apply per-request stop sequences, clearing any previous ones.

        Must be called on EVERY request, not just those that specify `stop` --
        the dialog is resident, so a stop sequence set by one caller would
        otherwise silently truncate the next caller's output.

        Raises on a non-SUCCESS status rather than returning it. Both callers
        used to discard the return, so a rejected clear left the previous
        caller's list armed on the resident dialog with no log line -- the
        exact leak this method exists to prevent -- and a rejected set ran
        this caller's generation without the stops it asked for. Either way
        the request would generate against stop sequences nobody in it chose;
        failing it visibly is the honest outcome. `_stop_dirty` is left as it
        was, so the next request retries the clear.
        """
        if not seqs and not self._stop_dirty:
            return                      # nothing set, nothing to clear
        # Genie wants a keyed OBJECT, not a bare array: passing ["x"] returns
        # -8 "Top level config is not an object" and is silently ignored by the
        # generation. The idle value is [""], which is what the SDK's own
        # example dialog configs carry -- an empty string resets cleanly where
        # passing "" as the whole payload logs a JSON parse error.
        payload = json.dumps({"stop-sequence": list(seqs) if seqs else [""]})
        st = self.lib.GenieDialog_setStopSequence(self.dialog, payload.encode("utf-8"))
        if st != GENIE_STATUS_SUCCESS:
            raise RuntimeError(
                "GenieDialog_setStopSequence failed, status=%d; refusing to "
                "generate with %s" % (st, "another request's stop sequences "
                                      "still armed" if self._stop_dirty else
                                      "this request's stop sequences unset"))
        self._stop_dirty = bool(seqs)

    def apply_sampler(self, params):
        """Apply per-request sampler params, restoring bundle defaults when None.

        DOES NOT TAKE EFFECT on QAIRT 2.45 with this bundle. Measured directly:
        GenieDialog_getSampler returns a valid handle, GenieSamplerConfig_
        createFromJson({"sampler": {...}}) returns 0, GenieSampler_applyConfig
        returns 0 -- and generation is BYTE-IDENTICAL across seed 1 / 999 /
        12345 and temp 0.0 / 1.5 / 2.0. The dialog appears to bind its sampler
        at GenieDialog_create time, so a post-create apply is accepted and
        ignored.

        Kept, not deleted, because the call sequence is correct and costs one
        no-op per request -- if a later QAIRT honours it, this starts working
        with no changes. What is NOT done is pretending it works: the server
        logs the limitation once at startup and the docs say sampling is
        server-level (edit dialog.sampler in genie_config.json before load),
        not per-request.

        The restore-to-default path below is likewise correct-but-inert today.
        It stays because the resident-dialog hazard it guards against is real:
        if applyConfig ever starts working, an unrestored temp=0 from a tool
        turn would silently make every later request deterministic.
        """
        if not params and not self._sampler_dirty:
            return
        cfg = dict(self.default_sampler)
        cfg.update(params or {})
        sampler = Handle()
        if self.lib.GenieDialog_getSampler(self.dialog, C.byref(sampler)) != GENIE_STATUS_SUCCESS:
            return
        handle = Handle()
        # Keyed wrapper, not a bare object: a bare {...} returns -8
        # "Missing field: sampler or standalone-sampler".
        if self.lib.GenieSamplerConfig_createFromJson(
                json.dumps({"sampler": cfg}).encode("utf-8"),
                C.byref(handle)) != GENIE_STATUS_SUCCESS:
            return
        try:
            self.lib.GenieSampler_applyConfig(sampler, handle)
            self._sampler_dirty = bool(params)
        finally:
            self.lib.GenieSamplerConfig_free(handle)

    def _plan(self, prompt):
        """Decide whether this prompt CONTINUES the resident KV or replaces it.

        The dialog keeps its KV across queries; the unconditional reset was the
        only reason every turn re-prefilled the whole conversation. When the new
        prompt starts with exactly what the dialog already holds, we can send
        just the new suffix -- measured 0.31s vs 3.40s on the reset path for the
        same turn, and the gap widens with conversation length.

        Byte-exact prefix match is the whole safety argument: if the client
        edited history, the window evicted a turn, or the echoed assistant turn
        differs from what we generated by even a character, the match fails and
        we fall back to a full reset. A near-match is NOT good enough -- resuming
        on mismatched KV would silently answer from a history that never
        happened, which is far worse than paying for a re-prefill.

        Returns (text_to_send, reused).
        """
        c = self._committed
        if c and prompt.startswith(c) and len(prompt) > len(c):
            return prompt[len(c):], True
        # The record is dropped HERE, at the reset, not left for _commit to
        # overwrite: from this point the KV is empty, and a throw anywhere
        # between this reset and the commit (a driver error, a rejected
        # config) used to leave the PREVIOUS conversation's prefix recorded
        # against an empty KV -- its next turn then passed the byte-prefix
        # check and prefilled only the new suffix, with no system prompt and
        # no history behind it. _run_query's except path clears it too; this
        # is the truth stated at the point it becomes true.
        self._committed = None
        # The status is the only evidence the KV was actually cleared, and it
        # used to be discarded. A refused reset leaves the PREVIOUS
        # conversation resident while this turn prefills its prompt on top of
        # it and _commit then records prompt+generated as the resident text --
        # the next turn's byte-prefix check passes and the model answers from
        # a history that never happened, with no error and no log line, which
        # is exactly the silent wrong answer the docstring above refuses to
        # accept from a near-match. Raise instead, as set_stop_sequences does
        # for the same resident-dialog reason.
        st = self.lib.GenieDialog_reset(self.dialog)
        if st != GENIE_STATUS_SUCCESS:
            raise RuntimeError(
                "GenieDialog_reset failed, status=%d; refusing to generate "
                "against a KV that may still hold another conversation" % st)
        return prompt, False

    def _commit(self, prompt, generated, ok):
        """Record the exact text now resident in the dialog's KV.

        On ANY failure the resident state is unknown, so drop the record and
        force the next turn to re-prefill. Guessing here would poison every
        subsequent continuation. `ok` is the caller's whole verdict -- for
        _run_query that includes "and nobody asked for this turn to be
        aborted", read off the turn as it left its window.
        """
        if not ok:
            self._committed = None
            return
        # Exactly what was sent plus exactly what came back. Appending a turn
        # terminator we never sent would claim the KV holds a byte it may not,
        # and every later continuation would resume one token out of step.
        self._committed = prompt + generated

    def _encode_count(self, text):
        """Token count of `text` via the Genie tokenizer, or None on any failure.

        The CALLER holds the engine lock: this is count_tokens without the
        lock, for use from inside a turn that already has it -- the lock is
        not reentrant, so the public method would deadlock there.

        Supervised on the native-call clock (EngineHealth.native_begin), here
        rather than in count_tokens so that BOTH callers get it: this is a
        call into the same driver and can wedge the same way, and a hang in
        it used to be invisible -- /health said ok while every request 429d
        behind the lock. _run_query calls it only after its generation has
        been closed out, so the two clocks never overlap.

        None once the engine is closed -- checked HERE, under the lock, not
        only in count_tokens ahead of it: a handler that passed that check and
        then waited out the shutdown free on the lock arrives here with the
        tokenizer already gone.
        """
        if self._closed or not self.tokenizer or not text:
            return None
        HEALTH.native_begin(time.time(), "GenieTokenizer_encode")
        try:
            held = []  # keep the alloc'd buffer alive across the encode call

            def _alloc(size, out_pp):
                b = C.create_string_buffer(size if size > 0 else 1)
                held.append(b)
                out_pp[0] = C.cast(b, C.c_char_p)

            acb = ALLOC_CALLBACK(_alloc)
            tokptr = C.POINTER(C.c_int32)()
            ntok = C.c_uint32(0)
            st = self.lib.GenieTokenizer_encode(
                self.tokenizer, text.encode("utf-8"), acb,
                C.byref(tokptr), C.byref(ntok))
            return int(ntok.value) if st == GENIE_STATUS_SUCCESS else None
        except Exception:
            return None
        finally:
            HEALTH.native_end()

    def _new_turn(self):
        """A turn for the calling thread to run (query) or consume
        (query_stream), remembered on that thread so a bare signal_abort()
        from it knows which generation it means."""
        turn = _Turn()
        self._tls.turn = turn
        return turn

    def _run_query(self, prompt, emit, turn, max_tokens=None, stop=None,
                   sampler=None, commit=True, internal=False):
        """The ONE engine choreography. query() runs it on the caller's thread,
        query_stream() on a worker; nothing else calls into the dialog.

        It used to exist twice -- query() and query_stream's worker were a
        line-for-line copy of each other -- and the copies had drifted exactly
        as copies do: only query() knew `commit` and `internal`, so a streamed
        summarisation would have committed and been counted; only the worker
        cleared _committed on a throw, so the sync path could leave a stale
        prefix recorded against a reset KV. Every guard below is applied once
        because there is now one place to apply it.

        Under the engine lock, in order: HEALTH.begin, the per-turn dialog
        state (stop sequences, sampler, continue-or-reset, the token cap),
        GenieDialog_query with `emit(str)` fed per callback, then the health
        verdict, the KV record and the finish reason. Returns 'stop' |
        'length'. Raises when the query did not complete, and on ANY raise the
        KV record is dropped so the next turn re-prefills.

        `turn` is this generation's _Turn -- what signal_abort aims at. It is
        made by the caller, on the thread that will be the one to abort it.

        `internal=True` marks a call this server made for its own purposes
        rather than one a client asked for; it is supervised the same but is
        not counted as served traffic (EngineHealth.end). `commit=False` says
        the text left in the dialog is not the caller's conversation.
        """
        # Refuse a prompt that cannot be SENT before touching the engine.
        # json.loads accepts a lone surrogate escape ("\\ud83d", what JS emits
        # for a sliced emoji), count_tokens swallows the encode error while
        # sizing, and the encode inside the locked section used to raise after
        # _plan had already reset the dialog. A client error should cost the
        # client an error, not the engine its KV record.
        prompt.encode("utf-8")
        # Always a cap, and always applied: the cap lives on the resident
        # dialog, so a turn that skipped setMaxNumTokens ran under whatever the
        # previous turn had set. DEFAULT_MAX_TOKENS is floored at 1 and every
        # request cap is validated >= 1, so `cap` cannot wrap c_uint32.
        cap = max_tokens or DEFAULT_MAX_TOKENS
        seen = []
        callbacks = [0]
        # An incremental decoder, not bytes.decode(..., "replace") per callback.
        # A byte-level BPE token can be a PARTIAL UTF-8 sequence, and if Genie
        # hands those through as they come, the per-callback decode turned one
        # euro sign into two U+FFFD -- in the stream, and in the KV record. The
        # decoder holds an incomplete tail until the next callback completes
        # it, and the flush after the query replaces a tail that never was.
        # (The try/except that used to wrap the decode was unreachable: with
        # errors="replace" it cannot raise.)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def _cb(resp, code, _udata):
            if not resp:
                return
            callbacks[0] += 1
            HEALTH.progress(time.time())
            t = decoder.decode(resp)
            if not t:
                return              # a partial sequence, held for the next callback
            seen.append(t)
            try:
                emit(t)
            except Exception:
                pass                # a consumer's failure is not the generation's

        with self.lock:
            if self._closed or self._closing:
                # Shutdown reached the engine while this turn waited for the
                # lock -- either it has already freed the dialog (_closed) or
                # it has begun and is waiting for this very lock (_closing).
                # Both are the same answer: do not start. The second case is
                # the one that cost something, because the abort that ends the
                # live turn hands the lock to the LONGEST WAITER, and under
                # GENIE_MAX_INFLIGHT=2 that is a queued request, not close():
                # it reset the dialog and began a fresh query, close() timed
                # out on it, and the process left with a generation running
                # inside Genie -- see begin_shutdown.
                #
                # Ahead of HEALTH.begin: nothing was attempted, so there is no
                # generation to supervise or to book.
                with self._abort_lock:
                    turn.done = True
                if self._closed:
                    raise EngineClosing(
                        "the engine is closed: the server is shutting down "
                        "and the Genie dialog has been freed")
                raise EngineClosing(
                    "the engine is shutting down: no new generation will "
                    "start, the Genie dialog is about to be freed")
            # At lock ACQUISITION, ahead of every native call this turn makes
            # -- see EngineHealth.begin for why not just ahead of the query.
            HEALTH.begin(time.time())
            with self._abort_lock:
                self._live = turn
            aborted = False
            # None, not SUCCESS, so that "the query did not return" is what the
            # finally below sees. Pre-setting SUCCESS booked every throw inside
            # the try as a successful, counted generation -- resetting the
            # failure streak on the very call that had just failed.
            status = None
            try:
                self.set_stop_sequences(stop)
                self.apply_sampler(sampler)
                send, reused = self._plan(prompt)
                if reused:
                    print("[genie] kv reuse: prefilling %d new chars, not %d"
                          % (len(send), len(prompt)), flush=True)
                # "Always applied" is the claim the comment above `cap` makes,
                # and discarding this status was the one way it could be
                # false: GenieDialog.h documents ERROR_GENERAL here for a cap
                # that "could not be applied", a VALUE-dependent failure an
                # ordinary client max_tokens can trigger. The cap lives on the
                # resident dialog, so a rejected set runs this turn under
                # whatever the PREVIOUS request asked for -- and `capped`
                # below is computed against the REQUESTED cap, so a turn
                # silently held at 8 tokens when the client asked for 64 came
                # back short reporting finish_reason "stop", which an agentic
                # client reads as a complete answer.
                cap_st = self.lib.GenieDialog_setMaxNumTokens(
                    self.dialog, C.c_uint32(cap))
                if cap_st != GENIE_STATUS_SUCCESS:
                    raise RuntimeError(
                        "GenieDialog_setMaxNumTokens(%d) failed, status=%d; "
                        "refusing to generate under another request's token "
                        "cap" % (cap, cap_st))
                cb = QUERY_CALLBACK(_cb)  # keep ref alive for the blocking call
                with self._abort_lock:
                    # Checked and armed in one step, so an abort lands on one
                    # side or the other: before this it is the flag, from here
                    # on it is the native signal.
                    early = turn.aborted
                    self._querying = not early
                if early:
                    # Asked to stop before the query began -- the watchdog, on
                    # a stall in one of the calls above that then returned, or
                    # shutdown. Honoured by not starting: a generation nobody
                    # wants must not take the single-flight NPU, and the
                    # native signal cannot cover this case, because whether an
                    # ABORT delivered to a dialog that is not inside a query
                    # sticks is not knowable from here. A sticky one would cut
                    # the NEXT request short, which is why signal_abort sends
                    # none until `_querying` says the call is live.
                    status = GENIE_STATUS_WARNING_ABORTED
                else:
                    status = self.lib.GenieDialog_query(
                        self.dialog, send.encode("utf-8"), SENTENCE_COMPLETE, cb, None)
            except BaseException:
                self._committed = None      # dialog state unknown after a throw
                raise
            finally:
                with self._abort_lock:
                    # The turn leaves its window. After `done` nothing can set
                    # `aborted`, so what is read here is final -- an abort that
                    # arrived while Genie was producing its last token is seen
                    # even though the query then returned SUCCESS.
                    self._live = None
                    self._querying = False
                    turn.done = True
                    aborted = turn.aborted
                    stalled = turn.stalled
                # An abort is not a failure -- unless the WATCHDOG sent it.
                # ABORTED is in GENIE_FINISHED_STATUSES for the client's sake
                # (a stop button is nobody's fault), but the status cannot say
                # who asked, and a generation that stalled past its limit and
                # only ended because the watchdog cut it is the engine failing.
                # Booked as a finish it would RESET the failure streak: a
                # device that stalled on every turn and honoured each abort
                # would never report `failing`, and two real errors followed
                # by one stall would read as recovered.
                HEALTH.end(status in GENIE_FINISHED_STATUSES and not stalled,
                           counted=not internal)
            tail = decoder.decode(b"", final=True)
            if tail:
                seen.append(tail)
                try:
                    emit(tail)
                except Exception:
                    pass
            generated = "".join(seen)
            # Did the generation run into its cap? Genie reports SUCCESS at the
            # cap -- a normal sentence-end -- so the only way to know is to
            # count what came back. Until this counted, every capped generation
            # reported "stop" and a client could not tell a cut answer from a
            # complete one.
            #
            # The callback count is the CERTAIN one: at most one callback per
            # token (a token whose bytes are a partial character may not get
            # one of its own), so it can only run low, and the cap reached by
            # that count is the cap reached. When it falls short, a re-encode
            # of the text is the second opinion -- it is what makes "length"
            # real for output that is mostly multi-token characters. But it is
            # the TOKENIZER's split of the text, not the model's, and it can
            # run either way (a special token that detokenised to plain text
            # re-encodes as several). Good enough for a finish reason; not for
            # the KV record, which is why that uses `certain` alone.
            certain = capped = False
            if status == GENIE_STATUS_SUCCESS:
                certain = callbacks[0] >= cap
                capped = certain or (self._encode_count(generated) or 0) >= cap
            if commit:
                # The record is prompt + generated, and there are three cases
                # where that would be a lie, so the record is dropped instead.
                # Not-SUCCESS: an aborted, overflowed or failed generation
                # leaves an unrecorded tail. Asked to abort, whatever status
                # came back: the signal can land while Genie is producing its
                # last token, and SUCCESS then describes a generation that may
                # or may not have been cut. And a stop-sequence request that
                # did not run to its cap: Genie STRIPS the matched text from
                # what it hands back, but the tokens that began the match were
                # fed and sit in the KV, so after a hit the KV holds tokens
                # `generated` does not carry -- the next turn's byte-prefix
                # check passes anyway, and the continuation resumes with a few
                # model-generated tokens sitting before the turn terminator,
                # the one-token-out-of-step case _commit calls unacceptable.
                # Whether a sequence fired is not observable (see
                # _anthropic_stop_reason), so the record goes whenever one
                # could have; a generation CERTAIN to have reached its cap
                # could not have hit one.
                exact = (status == GENIE_STATUS_SUCCESS and not aborted
                         and not (stop and not certain))
                self._commit(prompt, generated, exact)
            else:
                # Internal calls (summarisation) leave the dialog holding text
                # that is NOT the caller's conversation, so recording it as
                # the resident prefix would be a false claim. None says
                # "unknown, re-prefill", which is the truth.
                self._committed = None
            finish = self._finish(status)
            return "length" if capped and finish == "stop" else finish

    def query(self, prompt, on_text, max_tokens=None, stop=None, sampler=None,
              commit=True, internal=False, result=None):
        """Run one query synchronously. on_text(str) is called per chunk.
        Returns 'stop' | 'length'. Serialized (NPU is single).

        `result`, when given, is a dict that gets result["aborted"]: whether
        anyone asked for this turn to stop. The return value cannot say --
        _finish reports ABORTED as an ordinary "stop", which is right for a
        client that has left and wrong for a caller that is about to KEEP
        what came back. It is set on a raise too.

        On the CALLING thread, not a worker, and deliberately: this is for a
        caller with nobody to abort FOR. Its one caller is _summarize_turns,
        the server's own internal generation, which has no client that can
        leave. A worker here would add a thread and a queue per call and buy no
        abortability. Everything else the two paths share through _run_query,
        and the watchdog's abort reaches this one exactly as it reaches a
        stream.

        No request handler uses it any more -- not even for a non-streaming
        response. Such a response writes nothing until the end, so on this
        method an abandoned one ran to its cap with the NPU held; Handler._run
        consumes query_stream for every response instead, which leaves the
        handler thread free to watch its socket between chunks and abort the
        turn when the client has gone."""
        turn = self._new_turn()
        try:
            return self._run_query(prompt, on_text, turn,
                                   max_tokens=max_tokens, stop=stop,
                                   sampler=sampler, commit=commit,
                                   internal=internal)
        finally:
            if result is not None:
                with self._abort_lock:
                    result["aborted"] = turn.aborted

    def query_stream(self, prompt, result, max_tokens=None, stop=None,
                     sampler=None, commit=True, internal=False):
        """Generator: yields text chunks, then sets result['finish'] (and
        result['error'] on failure, plus result['closing'] when that failure
        is shutdown refusing the turn) when done. The blocking Genie query runs
        on a WORKER thread so the consumer (the request/handler thread) can call
        signal_abort() on client disconnect -- a cross-thread signal, which is
        how Genie's abort is designed to be delivered. This is what actually
        frees the single-flight lock instead of running to max_tokens.

        Leaving early aborts. A consumer that stops iterating before the
        generation has ended -- a handler breaking out on a dead socket, or
        raising -- closes this generator, and the close aborts THIS stream's
        turn if it has not ended. Without it, a disconnect noticed BEFORE the
        query started (the first frame failing to write) was signalled once,
        to nothing, and the worker then ran to the cap with nobody reading and
        the single-flight NPU held the whole way. The abort is aimed at the
        turn (see signal_abort), so it can never be a stray one at whichever
        request holds the dialog next.

        The turn is made HERE, on the consumer's thread -- a generator's body
        runs on whoever iterates it -- which is what lets that thread's bare
        signal_abort() mean this stream and nothing else.
        """
        q = queue.Queue()
        turn = self._new_turn()

        def worker():
            try:
                finish = self._run_query(prompt, lambda t: q.put(("text", t)),
                                         turn, max_tokens=max_tokens, stop=stop,
                                         sampler=sampler, commit=commit,
                                         internal=internal)
                q.put(("done", finish))
            except EngineClosing as e:
                # Kept apart from every other failure all the way to the
                # consumer: the exception type is the only thing that survives
                # str(e), and Handler._run answers this one 503 rather than
                # 500. Nothing was attempted, so there is nothing to report
                # about the engine's health either.
                q.put(("closing", str(e)))
            except Exception as e:
                q.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        ended = False
        try:
            while True:
                kind, val = q.get()
                if kind == "text":
                    yield val
                elif kind == "done":
                    result["finish"] = val
                    ended = True
                    return
                else:
                    result["finish"] = "stop"
                    result["error"] = val
                    if kind == "closing":
                        result["closing"] = True
                    ended = True
                    return
        finally:
            if not ended:
                # By reference, not through the thread: a generator can be
                # finalised on a thread that never iterated it.
                self._abort(turn)

    def signal_abort(self, any_turn=False, stalled=False):
        """Abort a generation. Returns what that did: False when there was
        none to abort, else ABORT_SIGNALLED or ABORT_FLAGGED (see _abort).

        WHOSE is the whole point. The native signal is engine-global --
        GenieDialog_signal(ABORT) lands on the one resident dialog regardless
        of who asked -- and this used to send it unconditionally. A handler's
        post-loop write failing after its worker had released the engine lock
        aborted the NEXT request's generation, already inside its query, and
        that client got a truncated answer as an ordinary "stop"; a first
        frame failing to write while another request was generating did the
        same to that one.

        Bare, it means MINE: the turn the calling thread is running or
        consuming (query / query_stream register it), which is what a handler
        whose write just failed means by it. If that turn has ended, or the
        thread never had one, nothing happens -- there is no generation of
        this caller's left to stop. `any_turn=True` means whoever holds the
        dialog, and is for the two callers that are nobody's consumer: the
        watchdog aborting a stall, and shutdown. `stalled=True` is the
        watchdog's alone: it marks the turn as aborted FOR STALLING, which is
        what keeps that generation from being booked as a healthy finish.
        """
        return self._abort(None if any_turn else getattr(self._tls, "turn", None),
                           any_turn, stalled)

    def _abort(self, turn, any_turn=False, stalled=False):
        """signal_abort's body, for a turn held by reference.

        Everything happens under _abort_lock, so the turn cannot leave its
        window while the signal is being sent. The flag is what makes the
        abort stick for the KV record -- an aborted generation leaves a
        partial, unrecorded tail in the KV, and continuing from it would
        resume mid-sentence off a history never recorded -- and it is also
        all a turn that has not reached its query yet gets: _run_query
        honours it by not starting (and says there why no native signal is
        sent to a dialog that is not inside a query).

        ONE native signal per turn from its own consumer. A disconnect aborts
        the same turn twice, a few microseconds apart -- _Emitter.lost()
        signals, then Handler._run closes the generator it walks away from,
        and that close aborts too (it has to: it is the only abort a
        disconnect noticed BEFORE the query gets). The second is redundant,
        and it goes out just as the first is making GenieDialog_query return
        -- while `_querying` is still True, because Python's window is wider
        than the native one. A signal landing in that gap is a signal at an
        idle dialog: the one case _run_query refuses to create, because
        whether it sticks and cuts the NEXT request short is not knowable
        from here. `any_turn` is exempt -- the watchdog re-signals a stall
        every interval on purpose, and each of its STALL lines says so.

        Returns False (no turn to abort), ABORT_FLAGGED (marked, and no native
        signal has gone: the turn is outside GenieDialog_query, where none can
        be sent) or ABORT_SIGNALLED (its query has been sent one -- by this
        call or, for a consumer's repeat, by the one before it; a turn flagged
        before its query never starts it, so `_querying` with `aborted`
        already set means the earlier abort was a native one).
        """
        with self._abort_lock:
            if self._closed:
                return False        # the dialog is freed: nothing to signal
            if any_turn:
                turn = self._live
            if turn is None or turn.done:
                return False
            first = not turn.aborted
            turn.aborted = True
            if stalled:
                turn.stalled = True
            if turn is not self._live or not self._querying:
                return ABORT_FLAGGED
            if first or any_turn:
                try:
                    self.lib.GenieDialog_signal(self.dialog, GENIE_DIALOG_ACTION_ABORT)
                except Exception:
                    pass
            return ABORT_SIGNALLED

    def count_tokens(self, text):
        """Exact token count via the Genie tokenizer, or None on any failure
        (callers fall back to an estimate). Serialized with generation.

        That serialization has a cost worth naming, because it is invisible
        from the call site: this takes the SAME lock a generation holds, so a
        queued request cannot even be SIZED while another is decoding, and
        during a wedge it blocks with everything else. _fit's bisection and
        _tok_count's memo exist to keep the number of these calls near the
        floor (~log2(turns) per request, down from one per evicted turn)
        rather than to avoid the lock.

        Dropping the lock is NOT the obvious win it looks like: the tokenizer
        handle comes from the resident dialog, and whether it is safe to encode
        on one thread while another is inside GenieDialog_query is a property
        of the driver that cannot be established without the device. Concurrent
        HTP access is what wedges this part in the first place, so the lock
        stays until someone measures the alternative on hardware.

        Supervised on its own clock, inside _encode_count: a hang here used to
        be invisible -- /health said ok while every request 429d behind the
        lock this holds.

        None after close() as well, with no native call made: the tokenizer
        handle died with the dialog (see close, and _encode_count for where
        that is checked)."""
        if not self.tokenizer or not text:
            return None
        with self.lock:
            return self._encode_count(text)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_CLOSE = "</think>"
# The close must stand ALONE ON ITS LINE to count as one the model emitted
# structurally. That is the discriminator between the two ways a bare closing
# tag reaches the output, and without it the strip cannot tell them apart:
#
#   reopened block   "...used for NLP.\n</think>\n\nA GEMM is..."   <- strip
#   prose mention    "The </think> tag closes a reasoning block."   <- keep
#
# Gating on "did this request prefill a block" is necessary but NOT sufficient,
# because reasoning is suppressed by DEFAULT -- so nearly every real request
# prefills, and a bare-close test alone truncated a correct answer on the
# overwhelmingly common path. Measured: asking the server to repeat that
# sentence returned "tag closes a reasoning block." until this anchor landed.
#
# Residual risk, stated rather than hidden: an answer that puts the tag alone on
# a line -- inside a fenced code block showing the template, say -- still trips
# it. That is rarer than the inline mention by a wide margin, and the failure is
# now bounded to a shape the model has to go out of its way to produce.
_ORPHAN_CLOSE_RE = re.compile(r"(?:\A|\n)[ \t]*</think>[ \t]*(?=\n|\Z)")
# The same anchor for use MID-STREAM, where end-of-buffer is not end-of-output.
# The permissive form above accepts \Z, which while streaming only means "the
# tag is the last thing received SO FAR" -- and the next chunk can continue that
# same line, turning it into an inline mention after all. Measured: the chunks
# ["The\n", "</think>", " tag is how you close it."] made the stream emit "tag
# is how you close it." while the buffered path correctly kept the whole
# sentence, so the two paths disagreed about the same generation. Requiring a
# real newline defers the decision until the line is finished; flush() then
# applies the permissive form, because by then the generation really has ended.
_ORPHAN_CLOSE_MID_RE = re.compile(r"(?:\A|\n)[ \t]*</think>[ \t]*(?=\n)")


def _strip_orphan_think(text):
    """Drop a leading run that ends in a `</think>` we never opened.

    THE PREFILL CONTRACT, AND HOW THE MODEL BREAKS IT. With reasoning
    suppressed the prompt ends in a CLOSED, empty block (_NO_THINK), so the
    model should simply answer. Qwen3 does not always accept that: it writes
    its reasoning anyway, emits a bare closing tag, and then answers properly.
    The output is then the answer TWICE with a stray tag between -- measured
    here at 1 request in 6 on the shipped bundle, and it is the most visible
    "it repeats itself" complaint against this server.

    _THINK_RE cannot catch it. That pattern needs a MATCHED pair, and the
    opening tag is in the PROMPT rather than the output, so the orphan close
    never matches and the duplicate survives into the response.

    Only an UNMATCHED close is touched. A well-formed <think>...</think> is the
    model reasoning normally -- that belongs to STRIP_THINK, which is the
    caller's choice to make, not this function's. An unmatched close can only
    be the model closing the block the prefill opened, so what precedes it is
    reasoning the caller already asked not to receive. Removing it restores the
    contract; leaving it ships the answer twice.
    """
    m = _ORPHAN_CLOSE_RE.search(text)
    if m is None or "<think>" in text[:m.start()]:
        return text
    return text[m.end():].lstrip()


def _maybe_strip_think(text, prefilled=False):
    """Strip reasoning from `text`. `prefilled` says whether THIS request sent
    a closed think block, which is the only case an orphan close can be ours.

    Orphan-stripping is not gated on STRIP_THINK -- that flag chooses whether to
    show genuine reasoning, while this removes a DUPLICATED answer, which no
    caller wants in either setting. It IS gated on `prefilled`, and that gate is
    load-bearing rather than tidy: without it the strip fires on any answer
    whose first mention of the closing tag is bare, so asking this server about
    its own reasoning suppression returned "tag closes a reasoning block"
    instead of "The </think> tag closes a reasoning block". Silent truncation of
    a correct answer is a worse failure than the duplicate it was fixing, and
    a coding agent over this repo hits exactly that text.

    `prefilled` defaults to FALSE on purpose. A call site that forgets to pass
    it then ships a visible duplicate, which someone reports; the other default
    would silently eat content at a site nobody thought about.
    """
    if prefilled:
        text = _strip_orphan_think(text)
    return _THINK_RE.sub("", text) if STRIP_THINK else text


# How much text to withhold at the start of a stream while deciding whether an
# orphan close is coming. 0 disables the hold and streams every chunk as it
# arrives, at the cost of shipping the duplicate to streaming clients.
#
# WHAT THIS COSTS, MEASURED, because the number is a real trade and not a
# tuning knob. SSE cannot retract a frame, so a duplicate can only be kept out
# of a stream by not sending it yet -- which means holding the START of every
# response until the orphan is ruled in or out. There is no cheap value:
# sampled over 12 streamed replies on this bundle, the one orphan closed at
# offset 405, while 7 of the 12 replies were shorter than 600 characters
# end-to-end. So any hold big enough to catch an orphan also delivers a typical
# short reply as a SINGLE frame. Buffering is not a side effect of the setting;
# at these lengths it essentially IS the setting.
#
# A bounded hold was tried first and LEAKED, which is why the default is now
# unbounded. Set to 1024 -- already wide margin over that 405 -- the very next
# sample produced an orphan closing at offset 1080: the gate released at its
# limit and the duplicate went out anyway. Two observed offsets, 405 and 1080,
# is not a distribution you can fit a threshold to, and a threshold high enough
# to cover the larger one exceeds most whole replies, so the bound was buying
# nothing while still looking like protection.
#
# So: -1 (the default) holds until the question is actually ANSWERED -- an
# orphan close, or the end of the generation. 0 disables the gate and streams
# every chunk as it arrives. A positive value keeps the old bounded behaviour
# for anyone who wants it, with the leak above as the known cost.
#
# The honest statement of the default is that a streamed reply arrives as one
# frame at the end: correct, but not incremental. For the agent traffic this
# server exists to serve that is invisible, since the client consumes the
# finished message either way. For a human watching tokens appear it is not --
# that reader should set 0 and accept the occasional doubled answer. Memory is
# bounded by max_tokens, so holding the whole generation costs nothing else.
#
# The BUFFERED paths -- non-streaming, and both tool paths, which already hold
# the whole generation before answering -- strip the duplicate always and for
# free. This setting only governs the incremental paths.
ORPHAN_HOLD_CHARS = _int_env("GENIE_ORPHAN_HOLD_CHARS", -1)


class _OrphanGate:
    """Withholds the START of a stream until an orphan </think> is ruled in or out.

    The buffered paths can strip the duplicate after the fact because they hold
    the whole generation before answering. A stream cannot: once reasoning has
    gone out as content there is no frame that retracts it, and the client
    renders the answer twice -- so for streaming clients the fix has to happen
    BEFORE the first byte, not after the last.

    So the opening of a stream is held until one of two things is known:

      * an unmatched </think> arrives -- everything before it was reasoning, so
        drop it and emit from after the tag.
      * the hold limit passes without one -- this generation is not going to
        close a block it never opened, so release the buffer and stream on.

    The cost is bounded and paid once: after the gate opens, nothing is ever
    buffered again for the rest of the generation.
    """

    def __init__(self, limit=None, prefilled=True):
        self.buf = []
        self.limit = ORPHAN_HOLD_CHARS if limit is None else limit
        # Open from the start -- i.e. a pass-through -- when gating is disabled
        # (limit 0) OR when this request did not prefill a closed block, since
        # then a closing tag can only be the model's own prose and withholding
        # the text before it would truncate a correct answer. Otherwise a
        # negative limit holds indefinitely and the gate opens on the close tag
        # or on flush(), nowhere else.
        self.open = self.limit == 0 or not prefilled
        # Set when an orphan is actually removed, and cleared once real text has
        # gone out after it. The blank line separating the dropped block from
        # the answer usually arrives in the NEXT chunk, after the gate has
        # already opened -- so lstrip()ing only what was held leaves the client
        # a reply that starts with a newline. Narrow on purpose: with no drop
        # this is inert, so an ordinary reply's leading whitespace is untouched.
        self._dropped = False

    def _after_drop(self, text):
        """Swallow the gap left by a removed block, then get out of the way."""
        if not self._dropped:
            return text
        text = text.lstrip()
        if text:
            self._dropped = False
        return text

    def feed(self, chunk):
        """Text that may be emitted now, possibly ""."""
        if self.open:
            return self._after_drop(chunk)
        self.buf.append(chunk)
        # Searched over the ACCUMULATION, not the chunk: Genie hands back
        # whatever the tokenizer produced, so the tag routinely straddles two
        # callbacks and a per-chunk search would miss it.
        held = "".join(self.buf)
        m = _ORPHAN_CLOSE_MID_RE.search(held)
        if m is not None:
            self.open, self.buf = True, []
            if "<think>" in held[:m.start()]:
                return held      # matched pair: genuine reasoning, not ours
            self._dropped = True
            return self._after_drop(held[m.end():])
        if self.limit > 0 and len(held) >= self.limit:
            self.open, self.buf = True, []
            return held
        return ""

    def flush(self):
        """Whatever is still held when the generation ends.

        A short answer can finish inside the hold window, so without this the
        entire response would be buffered and then dropped -- the gate would
        turn a rare duplicate into a routine empty reply.
        """
        if self.open or not self.buf:
            return ""
        held = "".join(self.buf)
        self.open, self.buf = True, []
        # Permissive form here: the generation is over, so a close sitting at
        # the very end of the output has no newline after it and is structural
        # all the same. This is also what keeps a streamed result identical to
        # the buffered one -- feed() defers that case, flush() decides it.
        return _strip_orphan_think(held)


def _anthropic_text(content):
    """Anthropic content -> text. Delegates to the shared flattener so the two
    endpoints cannot diverge on block handling."""
    return _content_text(content)


def read_default_sampler(seed=None):
    """The bundle's own sampler block -- the baseline a per-request override
    is restored to. Read rather than hardcoded, same reasoning as n_ctx: the
    values belong to the bundle, and a literal here goes quietly wrong on the
    next bundle.

    `seed` overlays the per-process seed load_engine patched into the config
    TEXT. The on-disk block carries the bundle's shipped `"seed": 42`, and a
    baseline read straight from disk would restore exactly that -- so the day
    GenieSampler_applyConfig starts working, apply_sampler's restore path
    would re-arm the fixed-seed replay next_seed() exists to remove, and the
    dirty-set path would ship 42 merged under the caller's temperature. The
    baseline has to describe the sampler the dialog was CREATED with."""
    # Same read as read_sampler, kept behind it so the file is opened once and
    # the two cannot disagree. The {"version": 1} fallback stays: this value is
    # a RESTORE BASELINE, and an empty dict would restore nothing.
    base = dict(read_sampler()) or {"version": 1}
    if seed is not None:
        base["seed"] = seed
    return base


def _is_message_list(messages):
    """True only for the shape both prompt builders assume: a list of objects.

    `"messages": "hi"` and `["hi"]` are both truthy, so they cleared the
    required-check and then died inside the template render -- out of do_POST,
    with no response written at all. A client cannot tell that apart from the
    server having died, which is the worst answer available at the one moment
    it needs a real one.
    """
    return isinstance(messages, list) and all(isinstance(m, dict) for m in messages)


def _is_tool_list(tools):
    """True only for the shape the tool render assumes: a list of objects.

    The same trap as _is_message_list, one field over. `"tools": "ab"` and
    `{"k": 1}` are both truthy, so they cleared `req.get("tools") or None` and
    reached ChatML.build's `for t in tools: json.dumps(t)`, which iterates a
    string by CHARACTER and a dict by KEY: the model was handed `"a"` and `"b"`
    as its function signatures, the request answered 200, and nothing anywhere
    said the schema was garbage. On /v1/messages the same value dies inside
    _anthropic_tools (`"a".get`) with no response written at all.

    An empty list is a list: "no tools" is a fine thing to send.
    """
    return isinstance(tools, list) and all(isinstance(t, dict) for t in tools)


def _parse_token_cap(raw, field):
    """One spelling of the output cap -> int >= 1, or None for "not set".

    Absent, null, 0 and false all mean "not set". For 0 that is the contract
    clients relied on before the guard existed (`req.get(...) or DEFAULT`),
    and it now holds for a numeric 0 and the string "0" alike: they used to
    disagree -- 0 kept the default while "0" answered 400 -- because the zero
    check ran before the parse, for no reason a client could see. Anything
    else must parse as an integer >= 1; a float is truncated (2.5 -> 2), as
    int() does, since the accepted types are int, float and str.
    """
    if raw is None or raw is False:
        return None
    if raw is True or not isinstance(raw, (int, float, str)):
        raise ValueError("%s must be an integer, got %r" % (field, raw))
    try:
        n = int(raw)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is the one json.loads can actually produce: Infinity
        # and NaN are accepted by the parser, and int(float('inf')) raises
        # OverflowError rather than ValueError.
        raise ValueError("%s must be an integer, got %r" % (field, raw)) from None
    if n == 0:
        return None
    if n < 1:
        raise ValueError("%s must be >= 1, got %d" % (field, n))
    return n


def _max_tokens(req):
    """The output cap for this request, validated at the door.

    GenieDialog_setMaxNumTokens takes a c_uint32, so a negative value does not
    error -- it wraps, and -1 reaches the HTP as 4294967295. Neither downstream
    guard catches that: build_windowed budgets with max(0, max_tokens), so a
    negative reads as zero there and the overflow 400 never fires either. The
    request is admitted and then holds the single-flight NPU until it walks
    into the context wall. Two guards each seeing a different number is the
    reason this validates at the door instead of trusting either.

    NOT clamped to the window, and this docstring used to say it was. Anything
    that large fails the fits check in build_windowed a few lines later and
    gets the honest overflow 400, which names the number the CLIENT sent and
    tells them to lower it -- advice that only works if the figure quoted back
    is theirs. Clamping here would have silently rewritten it to the window
    size first. The fits check is also what bounds this below 2**32 before it
    reaches c_uint32.

    BOTH spellings are accepted. OpenAI deprecated `max_tokens` in favour of
    `max_completion_tokens`, and which one a client sends now depends on how
    old its SDK is -- so honouring one and ignoring the other means a caller
    gets an unbounded generation on a single-flight NPU because of the vintage
    of their library. Measured on the two official servers for these bundles:
    GenieAPIService honours NEITHER field (a 16-token cap returned 125 tokens
    either way) and geniex serve honours only the modern one. This server had
    the mirror of geniex's gap until it was tested for.

    A legacy `max_tokens` that is SET wins over the modern spelling -- the more
    explicit signal, from a client old enough to send it at all. One that is
    absent or 0 defers to it: {"max_tokens": 0, "max_completion_tokens": 16}
    used to return the default, the 0 having hidden the 16. Neither set means
    DEFAULT_MAX_TOKENS, which is floored at 1 where it is read.
    """
    n = _parse_token_cap(req.get("max_tokens"), "max_tokens")
    if n is None:
        n = _parse_token_cap(req.get("max_completion_tokens"),
                             "max_completion_tokens")
    return DEFAULT_MAX_TOKENS if n is None else n


def _stop_sequences(req):
    """OpenAI `stop` (string or list) and Anthropic `stop_sequences`."""
    v = req.get("stop")
    if v is None:
        v = req.get("stop_sequences")
    if v is None:
        return None
    if isinstance(v, str):
        v = [v]
    seqs = [x for x in v if isinstance(x, str) and x]
    return seqs or None


def _sampler_params(req, tools_active=False):
    """Map request sampling fields onto the bundle's sampler keys.

    Tool turns default to temp 0: Genie owns sampling and there is no grammar
    hook, so low temperature is the only lever we have on JSON validity. An
    explicit temperature in the request still wins -- the caller may know
    better than this default.

    WHAT THIS MAPPING CURRENTLY BUYS: nothing at runtime. apply_sampler is inert
    on QAIRT 2.45 (see its docstring), so a tool turn actually runs at whatever
    `dialog.sampler` in the bundle sets -- 0.8 on every bundle here, not the 0.0
    below. The "only lever we have on JSON validity" is therefore a lever that
    is not connected, and it is worth knowing that before attributing a
    malformed tool call to the model rather than to the temperature it was
    really sampled at.

    Kept mapping anyway, for the same reason apply_sampler is kept: the shape is
    right and it starts working the day a QAIRT honours a post-create apply. The
    tests here pin the MAPPING, which is all that can be pinned device-free --
    they are not evidence that the sampling happened.
    """
    out = {}
    if "temperature" in req and req["temperature"] is not None:
        out["temp"] = float(req["temperature"])
    elif tools_active:
        out["temp"] = 0.0
    if "top_p" in req and req["top_p"] is not None:
        out["top-p"] = float(req["top_p"])
    if "top_k" in req and req["top_k"] is not None:
        out["top-k"] = int(req["top_k"])
    if out.get("temp") == 0.0:
        out.setdefault("top-k", 1)      # temp 0 without top-k 1 is not greedy
    return out or None


def _wants_thinking(req):
    """Resolve the reasoning block for ONE request, newest convention first.

    Three spellings are accepted because three ecosystems disagree and a client
    should not have to know which one this server speaks:
      * chat_template_kwargs.enable_thinking  -- the de-facto Qwen3 convention
      * reasoning_effort: "none"              -- OpenAI's field
      * thinking: {"type": "disabled"}        -- Anthropic's field
    Absent all three, fall back to the server default (GENIE_THINKING).
    """
    kw = req.get("chat_template_kwargs")
    if isinstance(kw, dict) and "enable_thinking" in kw:
        return bool(kw["enable_thinking"])
    eff = req.get("reasoning_effort")
    if eff is not None:
        return str(eff).lower() not in ("none", "minimal", "off")
    th = req.get("thinking")
    if isinstance(th, dict) and th.get("type"):
        return th["type"] != "disabled"
    return THINKING_DEFAULT


def _anthropic_tools(tools):
    """Anthropic {name, description, input_schema} -> the OpenAI function shape.

    Qwen3 was trained with OpenAI-style function schemas inside <tools>, so we
    hand it the shape it knows rather than Anthropic's. Same information,
    familiar packaging -- the model's tool-call accuracy depends on it.
    """
    out = []
    for t in tools or []:
        out.append({"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }})
    return out or None


def _with_overhead(usage, overhead):
    """Attach summarisation cost to a usage block, only when there was any."""
    if overhead:
        usage["genie_context_overhead_tokens"] = overhead
    return usage


def _anthropic_stop_reason(finish, calls, stop):
    """Anthropic stop_reason, distinguishing a stop-sequence cut from a natural end.

    Reporting end_turn after a stop sequence fired tells the client the model
    finished on its own when it was actually cut, which is the difference
    between "done" and "resume from here".

    Imprecision worth stating: Genie STRIPS the matched text, so we cannot
    confirm which sequence fired, or distinguish a stop-sequence cut from a
    natural EOS on a request that also supplied stop sequences. We report
    stop_sequence whenever the caller asked for stop sequences and generation
    did not run to the token cap -- the caller opted into that boundary, so it
    is the likelier reading -- and leave `stop_sequence` null rather than guess
    which one.
    """
    if calls:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "stop_sequence" if stop else "end_turn"


def _anthropic_to_prompt(req, tools=None, max_tokens=0):
    """Build the ChatML prompt from an Anthropic Messages request.

    Anthropic carries tool traffic as content BLOCKS (tool_use on assistant
    turns, tool_result on user turns); ChatML wants them as assistant
    tool_calls and role="tool" messages. Translating here means a multi-turn
    tool conversation replays in exactly the shape the bundle's template
    expects, instead of being flattened to prose the model cannot act on.
    """
    msgs = []
    sysval = req.get("system")
    if sysval:
        msgs.append({"role": "system", "content": _anthropic_text(sysval)})
    for m in req.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content")
        blocks = content if isinstance(content, list) else None
        if blocks:
            def _of(kind, blocks=blocks):
                # blocks bound as a default: the closure is invoked in this
                # iteration, but binding it means a later edit cannot silently
                # make it read a rebound value.
                return [x for x in blocks
                        if isinstance(x, dict) and x.get("type") == kind]
            text = "".join(x.get("text", "") for x in _of("text"))
            results, uses = _of("tool_result"), _of("tool_use")
            if results:
                for r in results:
                    msgs.append({"role": "tool",
                                 "content": _anthropic_text(r.get("content"))})
                if text:
                    msgs.append({"role": role, "content": text})
                continue
            if uses:
                msgs.append({"role": "assistant", "content": text,
                             "tool_calls": [{"function": {
                                 "name": u.get("name", ""),
                                 "arguments": u.get("input", {})}} for u in uses]})
                continue
        msgs.append({"role": role, "content": _anthropic_text(content)})
    return build_windowed(msgs, tools=_anthropic_tools(tools),
                          thinking=_wants_thinking(req),
                          max_tokens=max_tokens)   # 4-tuple, passed through


# Whether count_tokens has an exact tokenizer behind it: None until load_engine
# has run, GENIE_STATUS_SUCCESS (0) when it does, otherwise the status
# GenieDialog_getTokenizer returned. Module-level rather than read off the
# engine so /health can report it without touching ENGINE -- the handler must
# not reach for the engine during a wedge (see the supervision block).
TOKENIZER_STATUS = None


def load_engine():
    """Load Genie.dll, create the dialog from the bundle config (resident).

    Returns the engine with its sampler restore baseline already set, because
    only this function knows which seed the dialog was created with (see
    read_default_sampler)."""
    global TOKENIZER_STATUS
    if not BUNDLE_DIR or not SDK_DIR:
        sys.exit("set GENIE_BUNDLE_DIR (the Genie bundle dir) and GENIE_SDK_DIR "
                 "(the QAIRT 2.45 root) -- see docs/GENIE_SERVER.md")
    if not os.path.isdir(BUNDLE_DIR):
        sys.exit("bundle dir not found: %s" % BUNDLE_DIR)
    if not os.path.isdir(LIB_DIR):
        sys.exit("SDK lib dir not found: %s (check GENIE_SDK_DIR)" % LIB_DIR)

    hex_path, hex_archs, hex_skel_only = hexagon_search_path()
    if not hex_archs:
        sys.exit("""no usable Hexagon under %s
A Hexagon needs BOTH lib/hexagon-vNN/unsigned and
lib/aarch64-windows-msvc/QnnHtpVNNStub.dll. Skels with no Windows stub
here: %s  (those are Android-only Hexagons; QAIRT ships their skel but
no Windows stub, so they cannot be driven from this OS.)
Check GENIE_SDK_DIR, or unset GENIE_HEXAGON_ARCH if you pinned an arch."""
                 % (SDK_DIR, ", ".join(hex_skel_only) or "(none)"))
    os.environ["ADSP_LIBRARY_PATH"] = hex_path
    note = ""
    if hex_skel_only:
        note = "  (skel-only, no Windows stub: %s)" % ", ".join(hex_skel_only)
    print("[genie] hexagon archs usable: %s%s" % (", ".join(hex_archs), note),
          flush=True)
    os.add_dll_directory(LIB_DIR)  # so Genie.dll's Qnn* deps resolve (py3.8+)
    os.environ["PATH"] = LIB_DIR + os.pathsep + os.environ.get("PATH", "")

    lib = C.WinDLL(os.path.join(LIB_DIR, "Genie.dll"))

    ConfigHandle = Handle
    lib.GenieDialogConfig_createFromJson.argtypes = [C.c_char_p, C.POINTER(ConfigHandle)]
    lib.GenieDialogConfig_createFromJson.restype = C.c_int
    lib.GenieDialog_create.argtypes = [ConfigHandle, C.POINTER(Handle)]
    lib.GenieDialog_create.restype = C.c_int
    lib.GenieDialog_query.argtypes = [Handle, C.c_char_p, C.c_int, QUERY_CALLBACK, C.c_void_p]
    lib.GenieDialog_query.restype = C.c_int
    lib.GenieDialog_reset.argtypes = [Handle]
    lib.GenieDialog_reset.restype = C.c_int
    lib.GenieDialog_setMaxNumTokens.argtypes = [Handle, C.c_uint32]
    lib.GenieDialog_setMaxNumTokens.restype = C.c_int
    lib.GenieDialog_free.argtypes = [Handle]
    lib.GenieDialog_free.restype = C.c_int
    lib.GenieDialog_signal.argtypes = [Handle, C.c_int]
    lib.GenieDialog_signal.restype = C.c_int
    lib.GenieDialog_getTokenizer.argtypes = [Handle, C.POINTER(Handle)]
    lib.GenieDialog_getTokenizer.restype = C.c_int
    # Stop sequences: a JSON array string, applied to the resident dialog.
    lib.GenieDialog_setStopSequence.argtypes = [Handle, C.c_char_p]
    lib.GenieDialog_setStopSequence.restype = C.c_int
    # Per-request sampling: get the dialog's sampler, build a config from JSON,
    # apply it. This is what lets a tool-call turn run at temp 0 while ordinary
    # chat keeps the bundle's creative defaults.
    lib.GenieDialog_getSampler.argtypes = [Handle, C.POINTER(Handle)]
    lib.GenieDialog_getSampler.restype = C.c_int
    lib.GenieSamplerConfig_createFromJson.argtypes = [C.c_char_p, C.POINTER(Handle)]
    lib.GenieSamplerConfig_createFromJson.restype = C.c_int
    lib.GenieSamplerConfig_free.argtypes = [Handle]
    lib.GenieSamplerConfig_free.restype = C.c_int
    lib.GenieSampler_applyConfig.argtypes = [Handle, Handle]
    lib.GenieSampler_applyConfig.restype = C.c_int
    lib.GenieTokenizer_encode.argtypes = [
        Handle, C.c_char_p, ALLOC_CALLBACK,
        C.POINTER(C.POINTER(C.c_int32)), C.POINTER(C.c_uint32)]
    lib.GenieTokenizer_encode.restype = C.c_int

    # Genie resolves the config's relative ctx-bin / tokenizer paths against CWD.
    os.chdir(BUNDLE_DIR)
    with open(os.path.join(BUNDLE_DIR, "genie_config.json"), "rb") as f:
        cfg_json = f.read()

    # Override the bundle's fixed `seed` in the config TEXT, not on disk. The
    # bundle is a large external artifact shared with other tools and other
    # sessions on this box; rewriting someone else's file to change our own
    # sampling would be a side effect nobody asked for. Genie only ever sees
    # this string, so patching it here is sufficient and leaves the artifact
    # untouched. See next_seed() for why the shipped 42 has to go.
    seed = next_seed()
    applied_seed = None     # what the dialog is actually created with, if we know
    try:
        _cfg = json.loads(cfg_json)
        _cfg["dialog"]["sampler"]["seed"] = seed
        cfg_json = json.dumps(_cfg).encode("utf-8")
        applied_seed = seed
        print("[genie] sampler seed: %d%s" % (seed, " (pinned by GENIE_SEED)"
                                              if FIXED_SEED is not None else
                                              " (fresh per process; the bundle "
                                              "ships a fixed 42)"), flush=True)
    except Exception as e:
        # A config we cannot parse is not a reason to refuse to start -- Genie
        # is about to parse it itself and will give a better error than we can.
        # Say what was lost, though, because the symptom of losing it silently
        # is "every answer is the same answer", which reads as a model problem.
        print("[genie] WARNING: could not set the sampler seed (%s); the "
              "bundle's fixed seed stands, so identical prompts will return "
              "identical answers." % e, flush=True)

    cfg = ConfigHandle()
    st = lib.GenieDialogConfig_createFromJson(cfg_json, C.byref(cfg))
    if st != GENIE_STATUS_SUCCESS:
        sys.exit("GenieDialogConfig_createFromJson failed, status=%d" % st)

    dialog = Handle()
    t0 = time.time()
    # Measured range, not an aspiration: ~11-15s warm on the 8192 bundle and
    # 34s after unrelated disk traffic has evicted it from the page cache. The
    # old "~8-12s" was under every reading taken since, which makes a normal
    # load look like a hang to anyone watching the line.
    print("[genie] loading model on the NPU (~11-15s warm, up to ~35s cold)...",
          flush=True)
    st = lib.GenieDialog_create(cfg, C.byref(dialog))
    if st != GENIE_STATUS_SUCCESS:
        # The overwhelmingly likely cause is an arch/version mismatch: a Genie
        # context binary is compiled for ONE dsp_arch AND one QAIRT version, so
        # a bundle built for another Hexagon cannot load here. A bare status
        # code sends people hunting through their config; name the real suspect
        # and show what this box can actually offer.
        sys.exit("""GenieDialog_create failed, status=%d
  bundle:      %s
  SDK:         %s
  archs here:  %s
A Genie bundle is locked to one Hexagon arch AND one QAIRT version.
If this bundle was built for an arch this box does not have (or for a
different QAIRT), it cannot load -- get a bundle matching one of the
archs above, or rebuild it for this device."""
                 % (st, BUNDLE_DIR, SDK_DIR, ", ".join(hex_archs)))
    print("[genie] model resident on HTP in %.1fs" % (time.time() - t0), flush=True)

    tok = Handle()
    TOKENIZER_STATUS = lib.GenieDialog_getTokenizer(dialog, C.byref(tok))
    tokenizer = tok if TOKENIZER_STATUS == GENIE_STATUS_SUCCESS else None
    if tokenizer is None:
        # Said out loud, because nothing else will: without the tokenizer every
        # window budget, every overflow 400, every usage figure and the
        # max_tokens finish are silently the len//4 estimate -- the silent
        # degradation this server refuses everywhere else -- and the only
        # symptom is numbers that are plausibly wrong.
        print("[genie] WARNING: GenieDialog_getTokenizer failed, status=%d. "
              "Token counts are now ESTIMATES (len/4): context budgets, "
              "overflow 400s, usage figures and the max_tokens finish are "
              "approximate until this is fixed." % TOKENIZER_STATUS, flush=True)
    engine = GenieEngine(lib, dialog, tokenizer)
    # The restore baseline carries the seed the dialog was CREATED with, not
    # the shipped 42 still sitting on disk -- see read_default_sampler.
    engine.default_sampler = read_default_sampler(seed=applied_seed)
    return engine


def load_chat_template():
    """The bundle's own `genie.chat_template`, or standard Qwen ChatML.

    Degrades like every other bundle reader rather than raising, and for the
    reason _load_bundle_json states: this runs at startup, from main(), before
    anything is listening, so an exception here turns a hand-edited or
    truncated metadata.json into a server that will not boot at all -- a raw
    traceback in place of the one prompt path every request goes through.
    Measured shapes that reached here uncaught: a truncated file
    (JSONDecodeError), a template missing `assistant_prefix` (KeyError), a
    template that is the Jinja STRING rather than the delimiter block
    (TypeError), and `"genie": null` or a non-object `genie` (AttributeError).

    Unlike its siblings it SAYS so, because the substitution is invisible
    otherwise: generic ChatML renders every bundle's turns plausibly, so a
    bundle whose real delimiters were dropped serves slightly-wrong prompts
    forever with nothing in the log to explain the quality. A MISSING file or
    a bundle with no chat_template at all is not a failure and stays quiet --
    that is the documented fallback, not a degradation.
    """
    # _bundle_file, not a bare join: with no bundle dir set the join is just
    # "metadata.json", and a file of that name in the working directory would
    # be adopted as this server's chat template.
    meta_path = _bundle_file("metadata.json")
    if meta_path and os.path.isfile(meta_path):
        try:
            meta = _load_bundle_json("metadata.json")
            tmpl = (meta.get("genie") or {}).get("chat_template")
            if tmpl:
                return ChatML(tmpl)
        except Exception as e:
            print("[genie] WARNING: %s is unusable as a chat template (%s: %s)"
                  " -- falling back to standard Qwen ChatML, so this bundle's"
                  " own delimiters are NOT being used"
                  % (meta_path, type(e).__name__, e), flush=True)
    # Fallback: standard Qwen ChatML.
    return ChatML({
        "system_prefix": "<|im_start|>system\n", "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n", "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n", "assistant_suffix": "<|im_end|>\n",
        "default_system_prompt": "You are a helpful AI assistant.",
    })


def probe_tool_support():
    """Does THIS bundle's tokenizer actually know the tool-call tokens?

    Derived from the artifact, never assumed. A bundle whose vocab lacks
    <tool_call> cannot emit a parseable call no matter what we put in the
    prompt, and answering normally while dropping the caller's tools is the
    exact silent degradation this server refuses elsewhere. Qwen3 bundles
    carry the tokens in added_tokens.json / tokenizer_config.json.

    Through _bundle_file like every other bundle reader, so an unset
    BUNDLE_DIR means "no files" rather than ./added_tokens.json from whatever
    directory the server was started in. main() cannot get here with it unset
    (load_engine exits first); a direct caller can.
    """
    for fn in ("added_tokens.json", "tokenizer_config.json"):
        path = _bundle_file(fn)
        if path is None:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                if "<tool_call>" in f.read():
                    return True
        except Exception:
            continue
    return False


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer that refuses to share a port on Windows.

    HTTPServer sets allow_reuse_address = 1 (SO_REUSEADDR). On POSIX that means
    "rebind a socket still in TIME_WAIT", which is what you want when
    restarting a server. On WINDOWS it means something else entirely: a second
    process can bind a port another process is actively serving. Both binds
    succeed, the ORIGINAL process keeps receiving the connections, and the new
    one sits there looking healthy while serving nobody.

    That is not theoretical -- it cost a real measurement here. A server
    started on the 8192 bundle logged a clean startup and the right HTP
    allocation, while every request was answered by an older process still
    holding the port with the 4096 bundle. The only reason it was caught is
    that /props disagreed with the bundle that had just been loaded. A
    benchmark that silently measures the wrong model is exactly the failure
    this repo keeps finding, so make the second bind fail instead.

    Leaving SO_REUSEADDR off is not enough on Windows, which is why
    server_bind below asks for the port EXCLUSIVELY: two sockets with
    different addresses on one port -- 0.0.0.0 and 127.0.0.1 -- bind
    happily without it, in either order. Measured here with this class: A
    binds 0.0.0.0:p as main() does during the load, port_in_use('127.0.0.1',
    p) says False because nothing is accepting yet, and a second instance on
    the default GENIE_HOST binds 127.0.0.1:p and loads a second bundle onto
    the HTP beside the first. Once both listen, a loopback connect goes to
    the more specific socket -- the "looks healthy while serving nobody"
    case above, with the wildcard instance on the losing end.

    Also picks its address family from the host. HTTPServer is AF_INET only,
    while LOOPBACK_HOSTS names "::1" as a recognised host and port_in_use
    happily probes it -- so GENIE_HOST=::1 passed every startup check, loaded
    the model for 11-35s, and then died in the bind with a bare gaierror. A
    literal with a colon is IPv6; a v4 address or a hostname cannot contain one.

    main() builds it with bind_and_activate=False and takes the two steps
    itself -- server_bind() before the model load, server_activate() after --
    so the port is claimed early and answers late; see there for why.
    """
    allow_reuse_address = (os.name != "nt")

    def __init__(self, addr, handler, bind_and_activate=True):
        if ":" in (addr[0] or ""):
            self.address_family = socket.AF_INET6
        super().__init__(addr, handler, bind_and_activate)

    def server_bind(self):
        # SO_EXCLUSIVEADDRUSE, Windows only and BEFORE the bind: it is the
        # only option that makes a port this process holds unavailable to
        # every other address on it, and it is the whole of what makes the
        # early bind the guard the comments in main() say it is. It is the
        # opposite of allow_reuse_address rather than a companion to it --
        # setting both fails the bind with WSAEINVAL -- and
        # allow_reuse_address is already False here on nt, so the two never
        # meet. Measured on Windows 11: 0.0.0.0 then 127.0.0.1 is refused
        # with WinError 10013 and the reverse order with 10048 (both named
        # in the "cannot bind" exit), while an ordinary restart right after
        # a clean exit still binds -- TIME_WAIT applies to the accepted
        # connections, not to the listening socket, so no SO_REUSEADDR is
        # needed to reclaim the port.
        excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if excl is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, excl, 1)
        super().server_bind()


def port_in_use(host, port, timeout=0.5):
    """Is something already accepting connections here?

    Checked BEFORE the model loads, and before the bind. When the bind came
    after the load it caught this on POSIX only after the 11-35s of loading a
    3 GB bundle onto the HTP, and with HTTPServer's defaults on Windows it
    would not catch it at all. main() binds ahead of the load now and Server
    does not share a port on Windows, so the bind is a second line behind this
    one -- but it fails with a bare errno, and only this check can say what
    the collision MEANS: that requests would go on hitting the other
    process's model. What this check cannot see is an instance that has
    bound and is still loading (nothing accepts yet); the bind is what stops
    a second server then, on any GENIE_HOST, because it asks for the port
    exclusively -- see Server.server_bind.

    A wildcard bind address is not a connectable one. GENIE_HOST=0.0.0.0 is the
    documented way to expose this server beyond loopback, and connecting to
    0.0.0.0 does not reach a listener on 127.0.0.1 -- so the check returned
    False against a live server in exactly the configuration where the server
    is shared. Probe loopback instead; a wildcard listener accepts there too.
    """
    probe = host
    if not host or host in ("0.0.0.0", "::", "*"):
        probe = "127.0.0.1"
    try:
        with socket.create_connection((probe, port), timeout=timeout):
            return True
    except OSError:
        return False


ENGINE = None
TEMPLATE = None
TOOLS_OK = False

# Bound the number of in-flight generation requests (1 running on the NPU + a
# small queue). Excess requests are rejected fast instead of piling up parked
# threads behind the single-flight lock. Floored at 1 -- see below.
# Concurrency vs KV reuse, a real tradeoff worth stating: the dialog holds ONE
# resident KV, so when two conversations interleave here each one resets the
# other's prefix and both pay a full re-prefill. MAX_INFLIGHT=2 keeps the
# default (one running, one queued) because a queued request still completes
# while a rejected one costs a client round-trip; set 1 to protect KV reuse for
# a single-conversation workload, higher only if callers prefer queueing to a
# fast 429. Correctness does not depend on the choice -- the engine lock
# serialises regardless -- only reuse hit-rate does.
#
# Floored at 1, never disabled. The NPU serves one request at a time, so an
# "unlimited" setting does not buy concurrency -- it just lets unbounded
# threads park on the engine lock until the box runs out of stack, and every
# one of those callers waits instead of getting a fast 429 it could act on.
MAX_INFLIGHT = max(1, _int_env("GENIE_MAX_INFLIGHT", 2))
_INFLIGHT = threading.BoundedSemaphore(MAX_INFLIGHT)


_TOK_CACHE = {}
# Handlers run on concurrent threads (ThreadingHTTPServer, MAX_INFLIGHT of
# them past the semaphore) and every one of them reaches _tok_count, so the
# cache is shared mutable state and is treated as such.
_TOK_CACHE_LOCK = threading.Lock()


def _tok_count(text):
    """Exact token count if the Genie tokenizer is available, else a ~4-char
    estimate. Never raises.

    Memoised on the exact text because the SAME large string is counted more
    than once per request -- _fit encodes the fitted prompt to check the budget
    and then usage encodes it again -- and each encode is a native call holding
    the engine lock over a string that can be thousands of tokens. Token counts
    are deterministic for identical text, so the cache cannot go stale; it is
    cleared wholesale past a handful of entries because only the current
    request's strings are ever reused.

    ONE lookup, under the lock -- not `if text in cache: return cache[text]`.
    Between that check and that subscript the other admitted request's insert
    can run the wholesale clear, and the subscript then raises KeyError out of
    a function documented never to raise. Every caller runs under do_POST's
    catch-all (_answer_failure), so the client is answered -- with the WRONG
    answer: while the window is being fitted it is a 400 "malformed request
    (KeyError ...)" for a request with nothing wrong with it, while a
    finished generation's usage is being counted it is a 500 for an answer
    that was complete, and on a stream, whose 200 has already gone out, it is
    the connection closed mid-response. (Before that catch-all existed it was
    a closed connection and no response on every path.) CPython's GIL happens
    not to switch threads between those two bytecodes today; that is an
    implementation detail of one interpreter build, not a guarantee, and a
    free-threaded build does not give it.

    The lock is NOT held across count_tokens. That call takes the ENGINE lock,
    which another request can hold for an entire generation; holding this one
    while waiting there would park every cache HIT behind an unrelated decode.
    Two threads that miss on the same text both count it and store the same
    number -- a wasted encode, never a wrong answer.
    """
    with _TOK_CACHE_LOCK:
        n = _TOK_CACHE.get(text)
    if n is not None:
        return n
    n = ENGINE.count_tokens(text) if ENGINE else None
    n = n if n is not None else max(0, len(text) // 4)
    with _TOK_CACHE_LOCK:
        if len(_TOK_CACHE) > 8:
            _TOK_CACHE.clear()
        _TOK_CACHE[text] = n
    return n


def _overflow_msg(prompt, max_tokens):
    """Say WHY it will not fit, with the numbers, not just that it did not."""
    return ("prompt does not fit the model's context window even after "
            "dropping older turns: %d prompt tokens + %d max_tokens exceeds "
            "n_ctx=%d (margin %d). This bundle is compiled at that window; "
            "send a shorter message, lower max_tokens, or use a bundle built "
            "with a larger --context-length."
            % (_tok_count(prompt), max_tokens, read_context_size(),
               WINDOW_MARGIN))


def _log_dropped(n):
    """Eviction is a real loss of information -- never let it be silent."""
    print("[genie] context window: dropped %d oldest message(s) to fit n_ctx=%d"
          % (n, read_context_size()), flush=True)


def _fit(messages, tools, thinking, budget):
    """Evict oldest turns until the render fits. Returns (prompt, kept, evicted, fits).

    Anchored: the system turn and tool schemas always survive -- dropping those
    is how an agent forgets it has tools, which reads as the model getting
    dumber rather than as context loss. A tool result is never separated from
    the assistant turn that called it, which a token-level evictor could not
    guarantee.

    The system turn is the LEADING system messages (_leading_system), not
    every message with that role. Anchoring all of them was harmless only
    while the renderer threw away all but the first; with their words in the
    prompt, a client that keeps per-turn system reminders in its history
    grows an unevictable turn by a message per exchange until nothing else
    fits (the numbers are in _system_text). A later one is evicted in order
    with the turns around it, and like them it is handed to the summariser
    rather than dropped on the floor.

    That holds at the TAIL too, which is where it used to break. A conversation
    that ends on a tool run -- every agent step does -- ends on a UNIT: the
    assistant turn that made the calls plus every result after it. Eviction may
    cut in front of that unit and no later. It used to floor at the last
    MESSAGE instead, so when only the final tool result fit, it was kept alone:
    `fits` came back True and the client got a 200 over a prompt opening on a
    bare <tool_response>, its <tool_call> and any sibling result silently gone.
    If the unit does not fit, nothing honest does -- `fits` is False and the
    caller answers 400 with the numbers.
    """
    lead = _leading_system(messages)
    sys_msgs, rest = messages[:lead], messages[lead:]
    # Where the current unit starts: the last message that is neither a tool
    # result nor an instruction (_NOT_UNIT_START -- a system message, or the
    # same thing spelled "developer"). 0 when there is none, which makes a
    # conversation of nothing but tool results one unevictable unit --
    # malformed, and the client's to fix. Not a system message either: a
    # client that appends a reminder after each turn ENDS its request on one,
    # and a unit that started there would leave the user's actual question
    # evictable while the reminder about it was kept.
    last = max((i for i, m in enumerate(rest)
                if m.get("role") not in _NOT_UNIT_START), default=0)

    def _drop(k):
        """Drop the k oldest turns, advancing past any now-orphaned tool
        results. Returns (kept, actual_dropped).

        Past an instruction at the cut as well. One left at the HEAD of what
        is kept would no longer be a later message: rendered after sys_msgs it
        is part of the leading run, folded into the system turn -- behind the
        retained note, which _split_note only finds as the LAST thing there
        -- and anchored on the second pass. It goes with the turns it sat
        among, to the summariser like them.

        Never cuts past `last`, and rest[last] is neither, so the advance
        always stops ON a real turn."""
        k = max(0, min(k, last))
        while k < last and rest[k].get("role") in _NOT_UNIT_START:
            k += 1
        return rest[k:], k

    def _render(kept):
        return TEMPLATE.build(sys_msgs + kept, tools=tools, thinking=thinking)

    prompt = _render(rest)                      # common case: nothing to evict
    if _tok_count(prompt) <= budget:
        return prompt, rest, [], True
    if last == 0:
        # Nothing left to evict but the current unit; the caller owes the
        # client a real error rather than a doomed query.
        return prompt, rest, [], False

    # Bisect for the FEWEST turns to drop. The previous linear scan re-rendered
    # and re-tokenized the entire prompt once per evicted message -- measured
    # at 90 full tokenizer calls on a 122-message conversation, each taking the
    # engine lock, and build_windowed runs this twice when summarising.
    # Dropping more turns can only shrink the prompt, so "fits" is monotonic in
    # k and a bisection reaches the same kept-set in ~log2(n) renders. `last`
    # is the ceiling, not len(rest) - 1: past it the kept set starts inside the
    # current unit, which is not a candidate at any size.
    lo, hi, best = 1, last, None
    while lo <= hi:
        mid = (lo + hi) // 2
        kept, n = _drop(mid)
        cand = _render(kept)
        if _tok_count(cand) <= budget:
            best = (cand, kept, n)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        kept, n = _drop(last)
        return _render(kept), kept, rest[:n], False
    cand, kept, n = best
    return cand, kept, rest[:n], True


def _transcript_cap():
    """How many characters of evicted turns to offer the summariser.

    Derived from the window, because the summarisation call has to fit the
    SAME window the request just overflowed. This was a literal 6000 -- about
    1500 tokens by this file's own 4-char estimate, which is comfortable at the
    4096 and 8192 the shipped bundles use and larger than the whole window of
    a bundle compiled at 1024. There the summarisation prompt itself overflowed
    on every eviction: a hard GenieDialog_query failure, an NPU call paid for
    nothing, and one more consecutive failure on HEALTH's way to "failing".

    Half the window's tokens at three characters each -- denser than the
    4-char estimate on purpose, since code and paths tokenise worse than prose
    -- and never more than the 6000 the recall measurements in
    docs/GENIE_SERVER.md were taken at, so nothing changes at 4096 and above.
    A character cap is only ever a first guess; _summarize_turns measures the
    prompt it actually built and halves this until it fits.
    """
    return min(6000, (read_context_size() // 2) * 3)


def _transcript(msgs, cap_chars=None):
    """Flatten turns to a compact transcript for summarisation.

    `cap_chars=None` means _transcript_cap(), the window-derived bound.
    """
    if cap_chars is None:
        cap_chars = _transcript_cap()
    lines = []
    for m in msgs:
        role = m.get("role", "user")
        # _content_text, not the raw value: content may be a LIST of blocks,
        # and .strip() on a list raises. The renderer was fixed for this; these
        # helpers are the other consumers and had to be fixed with it.
        c = _content_text(m.get("content")).strip()
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc)
            c = (c + " [called %s]" % fn.get("name", "")).strip()
        if c:
            lines.append("%s: %s" % (role, c))
    text = _NL.join(lines)
    # Bound the input: the summarisation call has to fit the SAME window we are
    # already over. Keep the TAIL -- the most recent evicted turns are the ones
    # most likely to still matter. The <= 0 case is spelled out because
    # text[-0:] is the WHOLE string, not none of it.
    if cap_chars <= 0:
        return ""
    return text[-cap_chars:] if len(text) > cap_chars else text


_SUMMARY_ASK = ("Condense this conversation excerpt into a few terse factual "
                "bullet points. Keep file paths, identifiers, decisions made, "
                "and results already obtained. Drop pleasantries and reasoning.")
# Below this much transcript the call is not worth making: the ask and a prior
# note already fill the window, and a summary of one clipped line is noise.
_MIN_TRANSCRIPT_CHARS = 200


def _summarize_turns(msgs, prior=""):
    """One cheap NPU call condensing evicted turns (plus any prior note).

    Returns (note, tokens); (None, 0) on any failure -- the caller then falls
    back to plain eviction. A summary is a nice-to-have; never let it break the
    request. The token count is returned so the cost can be surfaced instead of
    being spent invisibly on the caller's behalf. A generation that was
    ABORTED is a failure here, whatever text it had produced by then.

    The prompt is MEASURED before it is sent, against the window less the
    note's own cap and the margin, and the transcript is halved until it fits.
    Overflow is a hard query failure here exactly as it is for a client's
    prompt, and a failed internal call is still a failed call: it costs the
    NPU time and counts toward HEALTH's consecutive failures. A character cap
    cannot promise a fit -- dense text runs to a token per character -- so the
    count is what decides. If even a sliver will not fit (a small window, a
    long prior note) no call is made at all.
    """
    if ENGINE is None:
        return None, 0
    cap = _transcript_cap()
    budget = read_context_size() - summary_token_cap() - WINDOW_MARGIN
    while True:
        body = _transcript(msgs, cap)
        if prior:
            body = prior + _NL + body
        if not body.strip():
            return None, 0
        prompt = TEMPLATE.build(
            [{"role": "user", "content": _SUMMARY_ASK + _NL + _NL + body}],
            thinking=False)
        if _tok_count(prompt) <= budget:
            break
        if cap <= _MIN_TRANSCRIPT_CHARS:
            print("[genie] context window: evicted turns NOT summarised -- the "
                  "summarisation prompt does not fit n_ctx=%d either"
                  % read_context_size(), flush=True)
            return None, 0
        cap //= 2
    out, res = [], {}
    try:
        ENGINE.query(prompt, out.append, max_tokens=summary_token_cap(),
                     commit=False, internal=True, result=res)
    except Exception:
        return None, 0
    if res.get("aborted"):
        # The watchdog cut this generation for stalling (or shutdown did). The
        # engine reports that as an ordinary "stop" with whatever had been
        # produced, and a fragment -- "- user asked about pars" -- is not a
        # summary. It mattered less when a note lasted one request; the caller
        # now REMEMBERS the note as standing for these turns, so a fragment
        # kept here was never re-summarised and was fed forward as `prior`
        # into every later roll-up for the life of the conversation. A failed
        # summary instead: nothing is remembered, and the next request offers
        # the same turns to the summariser again.
        print("[genie] context window: the summary was cut short by an abort "
              "-- discarded, so the evicted turns are summarised again next "
              "time rather than remembered by a fragment", flush=True)
        return None, 0
    # prefilled=True unconditionally: this prompt is built with thinking=False
    # a few lines up, so a closing tag in the note can only be the model
    # reopening the block we prefilled. A retained note is the one place
    # reasoning must never land -- it rides in the system turn and is re-read on
    # every later request.
    text = _THINK_RE.sub("", _strip_orphan_think("".join(out))).strip()
    if not text:
        return None, 0
    return text, _tok_count(text)


def _split_note(sys_text):
    """(base, note) for a system turn's text; note is "" when it carries none.

    A note is recognised ONLY as the last thing in the system turn: the marker
    on a line of its own, and everything from there to the end. That is the one
    shape _apply_note writes. The old rule -- split on the marker wherever it
    appears -- treated any system prompt that QUOTED the marker as carrying a
    note: everything after the quote was cut out of the prompt and handed to
    the summariser as "prior context". rfind, so that a prompt which quotes the
    marker early and also carries a real note keeps its quote.
    """
    head = SUMMARY_MARKER + _NL
    at = sys_text.rfind(head)
    if at < 0 or (at > 0 and sys_text[at - 1] != _NL):
        return sys_text, ""
    return sys_text[:at].rstrip(), sys_text[at + len(head):].strip()


def _apply_note(messages, note):
    """Fold the note into the system turn, REPLACING any previous note.

    It rides in the system turn because that is the one thing eviction never
    touches -- a note stored anywhere else would itself be evicted, which is
    the problem it exists to solve.

    Returns ONE system message -- every LEADING system message's text
    (_system_text, exactly what the renderer would have joined) with the note
    last -- followed by everything after that leading run, in order, a later
    system message included: those are turns, not part of the system turn. One,
    so that the note is at the very end of the system turn however many system
    messages the client opened with; folded into the first of two it would
    land mid-turn, where _split_note would never find it again.

    With no system text at all the turn is seeded with the template's
    default_system. The renderer only falls back to that default when the
    system text is EMPTY, and a turn holding a note is not: a client that sends
    no system prompt -- entirely ordinary -- used to lose "You are a helpful AI
    assistant." on its first eviction and keep only the note.
    """
    base = _split_note(_system_text(messages))[0]
    if not base:
        base = getattr(TEMPLATE, "default_system", "") or ""
    joined = base + (_NL + _NL if base else "") + SUMMARY_MARKER + _NL + note
    return [{"role": "system", "content": joined},
            *messages[_leading_system(messages):]]


def _prior_note(messages):
    """The note a CLIENT sent back in its system turn, or "".

    Only a client that was handed the rendered system turn can do that -- a
    second genie_server in front of this one, say. An ordinary stateless client
    never can: no response carries the note. Their carry-forward is
    _NOTE_STATE, below.

    Through _system_text, so content that is a LIST of blocks is flattened
    first: `MARKER in [block, ...]` is a list-membership test that quietly
    returns False, the prior note goes unfound, and each eviction appends a
    fresh one until the notes themselves crowd out the window.
    """
    return _split_note(_system_text(messages))[1]


# The last note written and the evicted turns it stands for -- (key, count,
# note), or None. This is the carry-forward the marker was always described as
# providing and never did: the note goes into the PROMPT, no response returns
# it, and clients are told to resend their history verbatim, so the next
# request arrives with no trace of it. Every over-window request therefore
# re-summarised from scratch, from the last _transcript_cap() characters of
# what it evicted -- and a fact stated before that tail was gone for good
# after a few file reads, which is precisely the loss a note exists to prevent.
#
# Kept beside the engine's `_committed` in spirit, and ONE slot for the same
# reason that is one string: the dialog holds one resident conversation. Two
# over-window conversations interleaving here evict each other's note exactly
# as they evict each other's KV, and each then pays what every request paid
# before this existed. Keyed by content, so a match cannot hand one
# conversation another's note: the key digests the very turns the note
# summarises (and the client's own prior note, which it absorbed).
_NOTE_STATE = None
_NOTE_LOCK = threading.Lock()


def _note_key(client_note, msgs):
    """Digest of exactly what a note stands for: these turns, in this order."""
    h = hashlib.sha256()
    for part in [client_note] + [json.dumps(m, sort_keys=True, default=str)
                                 for m in msgs]:
        # json.dumps output is ASCII; the client's note is not, and may hold a
        # lone surrogate straight out of a JSON escape -- hence surrogatepass.
        # The NUL keeps ["ab", "c"] and ["a", "bc"] from digesting alike.
        h.update(part.encode("utf-8", "surrogatepass"))
        h.update(bytes(1))
    return h.hexdigest()


def _recall_note(client_note, evicted):
    """(note, covered): the stored note, if it stands for a PREFIX of `evicted`.

    ("", 0) otherwise. A stateless client resends the same history plus the new
    turns, so what this request evicts normally STARTS with what the last one
    evicted. `covered` is how many of them the note already accounts for; the
    caller summarises only the rest, with the note as `prior`.
    """
    with _NOTE_LOCK:
        state = _NOTE_STATE
    if state is None:
        return "", 0
    key, count, note = state
    if count > len(evicted) or _note_key(client_note, evicted[:count]) != key:
        return "", 0
    return note, count


def _remember_note(client_note, evicted, note):
    global _NOTE_STATE
    with _NOTE_LOCK:
        _NOTE_STATE = (_note_key(client_note, evicted), len(evicted), note)


def build_windowed(messages, tools=None, thinking=None, max_tokens=0,
                   summarize=None):
    """Render a prompt that FITS, summarising what it has to evict.

    Genie has no sliding-window mode -- QAIRT 2.45 exposes no such flag on
    genie-t2t-run and no equivalent config key -- and overflowing the compiled
    window is a hard GenieDialog_query failure, not a truncation. So eviction
    happens here, and (unless disabled) what leaves is condensed rather than
    discarded.

    Notes do not stack and do not restart. The previous note -- recalled from
    _NOTE_STATE when it stands for a prefix of what is being evicted now, else
    whatever the client's own system turn carried -- goes to the summariser as
    `prior` together with ONLY the newly evicted turns, and the result replaces
    it. When nothing new was evicted the stored note is reused as it is: no NPU
    call, no overhead, and no dialog reset, so the resident KV survives too.

    Returns (prompt, dropped, fits, overhead_tokens), where overhead_tokens is
    NPU work spent summarising rather than answering.
    """
    if summarize is None:
        summarize = SUMMARIZE_EVICTED
    if thinking is None:
        thinking = THINKING_DEFAULT   # see ChatML.build -- one source of truth
    budget = read_context_size() - max(0, max_tokens) - WINDOW_MARGIN

    prompt, kept, evicted, fits = _fit(messages, tools, thinking, budget)
    if not (fits and evicted and summarize):
        return prompt, len(evicted), fits, 0

    client_note = _prior_note(messages)
    note, covered = _recall_note(client_note, evicted)
    overhead = 0
    how = ("reused the retained %d-char note for %d evicted message(s) (no NPU "
           "call)" % (len(note), covered))
    if covered < len(evicted):
        fresh, overhead = _summarize_turns(evicted[covered:],
                                           prior=note or client_note)
        if fresh:
            note = fresh
            _remember_note(client_note, evicted, note)
            how = ("summarised %d evicted message(s) into a %d-char note (%d "
                   "tokens of NPU time)"
                   % (len(evicted) - covered, len(note), overhead))
        elif not note:
            return prompt, len(evicted), fits, overhead   # fall back to plain evict
        else:
            # The summariser failed, but the recalled note is still true of the
            # turns it covers. Keep it, leave _NOTE_STATE where it was, and let
            # the next request try the uncovered turns again.
            how = ("kept the previous %d-char note; %d newly evicted "
                   "message(s) could NOT be summarised into it"
                   % (len(note), len(evicted) - covered))

    # The system turn (the leading run) plus what survived. `kept` never opens
    # on a system message -- _fit's _drop sees to that -- so the note-bearing
    # message is the whole leading run of `merged` and the note stays last.
    merged = _apply_note(messages[:_leading_system(messages)], note) + kept
    # Second pass WITHOUT summarising: the note itself costs tokens and may push
    # the render back over budget. Re-fitting can only drop more turns, and
    # recursing here would summarise the summary on every request.
    p2, _, ev2, fits2 = _fit(merged, tools, thinking, budget)
    if fits2:
        print("[genie] context window: " + how, flush=True)
        return p2, len(evicted) + len(ev2), True, overhead
    return prompt, len(evicted), fits, overhead    # note did not fit; plain evict


# --- one generation driver, two wire formats --------------------------------
# The four response paths -- OpenAI and Anthropic, each streamed and not --
# used to be four hand-written functions, the Anthropic pair a ~120-line mirror
# of the OpenAI pair. They drifted exactly as copies do: one path counted usage
# AFTER the think-strip while the rest counted the raw generation, three
# reported an engine failure as assistant TEXT while the fourth sent a real
# error event, and every disconnect fix had to be found and applied twice.
#
# The choreography now exists once, in Handler._run: run the generation, notice
# a client that left, strip, parse tool calls, count. What genuinely differs
# between the two APIs is only how each moment is SPELLED on the wire, and that
# is all an emitter knows. Anything that is not spelling belongs in _run, where
# both APIs get it.

def _unsendable(prompt):
    """Why this rendered prompt cannot be sent to the engine, or None.

    json.loads accepts a lone surrogate escape ("\\ud83d" -- what JavaScript
    emits for a string cut through the middle of an emoji), and a str holding
    one cannot be encoded as UTF-8, which is the only form Genie takes. The
    engine refuses it before touching the dialog, but from there it surfaced as
    a 500, or as an error frame on a stream that had already answered 200. It
    is the CLIENT's text that is malformed, so it is answered 400 at the door.
    And it got likelier to arrive: ChatML.build renders tool arguments with
    ensure_ascii=False, so a surrogate inside one now reaches the prompt raw
    instead of as a harmless backslash-u escape.
    """
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError as e:
        return ("the request contains text that is not valid Unicode (%s at "
                "character %d of the rendered prompt); a lone surrogate such as "
                "\\ud83d usually means a string was cut through the middle of "
                "an emoji" % (e.reason, e.start))
    return None


def _tool_args_json(call):
    """A parsed call's arguments as the JSON STRING a response carries.

    ensure_ascii=False, and not for looks. ChatML.build splices a string
    `arguments` from the client's history VERBATIM, so on the OpenAI leg --
    where arguments travel as a string -- the next turn's prompt starts with
    what the dialog's KV holds only if this string is the bytes the model
    emitted. The default would hand back a backslash-u escape for the model's
    raw e-acute, the echo would no longer be a byte prefix of the resident
    conversation, and every turn after a tool call with a non-ASCII argument
    would re-prefill from scratch. The Anthropic leg carries an object, which
    the render dumps itself, but it goes through here too so the two legs
    cannot drift.
    """
    return json.dumps(call["arguments"], ensure_ascii=False)


class _Emitter:
    """The wire side of one response: the disconnect latch and the SSE writer.

    A subclass spells four moments in its API's shape -- start, text, finish,
    error -- streamed (SSE frames) or not (one JSON body). `streaming` is set by
    start(), which only a streamed response calls; finish() and error() read it
    to choose between a run of frames and a single body.
    """

    KEEPALIVE = b""

    def __init__(self, handler, prompt, overhead=0):
        self.h = handler
        self.prompt = prompt
        self.overhead = overhead
        self.streaming = False
        self.gone = False

    def lost(self):
        """The client left: latch it, and stop paying for the generation.

        Bare signal_abort() means THIS thread's turn (see GenieEngine
        .signal_abort). Before the query has started, or after it has ended,
        that is a no-op -- which is why _run also returns on `gone` before
        starting one, and closes the generator it walks away from.
        """
        self.gone = True
        ENGINE.signal_abort()

    def write(self, payload):
        """One SSE frame in ONE write; after the first failure, none at all.

        Once a write raises, every later one must be skipped: re-raising on a
        dead socket is what handle_one_request exists to suppress, and the
        closing frames are written from a different place than the token
        frames. One write per frame, too -- an Anthropic event used to go out
        as two (the `event:` line, then `data:`), so a client could be handed
        half an event, and with TCP_NODELAY each half was its own packet.

        socket.timeout is an OSError: a client that has not READ for
        SOCKET_TIMEOUT_S is treated as one that left, because on a
        single-flight NPU the two cost everyone else exactly the same.
        """
        if self.gone:
            return
        try:
            self.h.wfile.write(payload)
            self.h.wfile.flush()
        except (ConnectionError, OSError):
            self.lost()

    def probe(self):
        """Find out whether the client is still there while NOTHING is going out.

        A failed write is the only way a stream learns its reader is gone, and
        three situations write nothing for a whole generation: a tool turn
        (buffered until the call closes), a stream whose orphan gate is holding,
        and every non-streaming response. Without a probe each of those runs an
        abandoned turn to max_tokens with the single-flight NPU held against
        everyone else. A stream writes its API's keep-alive; a non-streaming
        response has no frame to write, so it asks the socket instead.
        """
        if self.streaming:
            self.write(self.KEEPALIVE)
        elif self.h._client_gone():
            self.lost()


class _OpenAIEmitter(_Emitter):
    """OpenAI Chat Completions: chat.completion, or chat.completion.chunk frames."""

    # An SSE comment: ignored by every client, there only so a write can fail.
    KEEPALIVE = b": keep-alive" + _SSE_GAP
    DONE = b"data: [DONE]" + _SSE_GAP

    def __init__(self, handler, prompt, cmpl_id, created, overhead=0,
                 include_usage=False):
        super().__init__(handler, prompt, overhead)
        self.cmpl_id = cmpl_id
        self.created = created
        self.include_usage = include_usage

    def _data(self, obj):
        self.write(b"data: " + json.dumps(obj).encode("utf-8") + _SSE_GAP)

    def _frame(self, delta, finish=None):
        return {"id": self.cmpl_id, "object": "chat.completion.chunk",
                "created": self.created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def _usage(self, raw):
        pt, ct = _tok_count(self.prompt), _tok_count(raw)
        # The overhead key is namespaced and only present when non-zero, so an
        # ordinary response is byte-identical to before and no standard field
        # is misreported. It is NPU time the request really spent -- on
        # summarising evicted history, not on the answer -- and spending it
        # invisibly is the same failure as the completion_tokens=0 bug.
        return _with_overhead({"prompt_tokens": pt, "completion_tokens": ct,
                               "total_tokens": pt + ct}, self.overhead)

    def _call(self, idx, call):
        return {"id": "call_%s_%d" % (self.cmpl_id, idx), "type": "function",
                "function": {"name": call["name"],
                             "arguments": _tool_args_json(call)}}

    def start(self, buffered):
        self.streaming = True
        self._data(self._frame({"role": "assistant"}))

    def text(self, out):
        self._data(self._frame({"content": out}))

    def finish(self, text, calls, finish, stop, raw):
        if calls:
            finish = "tool_calls"
        if not self.streaming:
            message = {"role": "assistant", "content": text or None}
            if calls:
                message["tool_calls"] = [self._call(i, c)
                                         for i, c in enumerate(calls)]
            self.h._json(200, {
                "id": self.cmpl_id, "object": "chat.completion",
                "created": self.created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": message,
                             "finish_reason": finish}],
                "usage": self._usage(raw)})
            return
        if text:
            self.text(text)
        for i, c in enumerate(calls):
            self._data(self._frame(
                {"tool_calls": [dict(index=i, **self._call(i, c))]}))
        self._data(self._frame({}, finish=finish))
        # Non-streaming responses always carried usage; streams carried none,
        # so a streaming client could not see token counts by any means.
        # OpenAI's shape for it is a final chunk with an EMPTY choices list,
        # sent only when the caller asked via stream_options.include_usage --
        # so a client that did not ask sees a byte-identical stream to before.
        # Skipped on a dead socket: the count is a native tokenizer call under
        # the engine lock, spent on a frame nobody will read.
        if self.include_usage and not self.gone:
            self._data({"id": self.cmpl_id, "object": "chat.completion.chunk",
                        "created": self.created, "model": MODEL_ID,
                        "choices": [], "usage": self._usage(raw)})
        self.write(self.DONE)

    def error(self, msg, code=500):
        # `code` is for the one failure that is not this server's fault and not
        # the client's: a turn refused because shutdown began (503, "shed and
        # try elsewhere"). It reaches only the non-streaming answer, because
        # once a stream's 200 has gone out there is no status left to choose --
        # the data frame below is the whole of what the client gets either way.
        if not self.streaming:
            self.h._json(code, {"error": {"message": msg, "type": "server_error"}})
            return
        # The 500's body as a data frame, which is how llama.cpp's server
        # reports a failure once the 200 has gone out, and what the OpenAI SDKs
        # raise on. It used to be a CONTENT delta reading "[error: ...]"
        # followed by finish_reason "stop", so an agent stored the error string
        # as the model's answer and carried on. No finish frame and no usage:
        # the turn did not finish, and saying how it "ended" is the lie. [DONE]
        # still closes the stream, for a client that reads until it sees one.
        self._data({"error": {"message": msg, "type": "server_error"}})
        self.write(self.DONE)


class _AnthropicEmitter(_Emitter):
    """Anthropic Messages: a message body, or the message_start..stop events."""

    # `ping` is a real Anthropic event, so this keep-alive needs no client-side
    # tolerance the way an SSE comment might.
    KEEPALIVE = (b"event: ping\n" + b"data: "
                 + json.dumps({"type": "ping"}).encode("utf-8") + _SSE_GAP)

    def __init__(self, handler, prompt, msg_id, model, overhead=0):
        super().__init__(handler, prompt, overhead)
        self.msg_id = msg_id
        self.model = model
        self.block_open = False     # content block 0, opened by start()

    def _event(self, etype, obj):
        self.write(("event: %s\n" % etype).encode("utf-8") + b"data: "
                   + json.dumps(obj).encode("utf-8") + _SSE_GAP)

    def _delta(self, idx, delta):
        self._event("content_block_delta", {"type": "content_block_delta",
                                            "index": idx, "delta": delta})

    def _block(self, idx, block, delta):
        """One whole content block: start, a single delta, stop."""
        self._event("content_block_start", {"type": "content_block_start",
                                            "index": idx, "content_block": block})
        self._delta(idx, delta)
        self._event("content_block_stop", {"type": "content_block_stop",
                                           "index": idx})

    def _tool_id(self, n):
        return "toolu_%s_%d" % (self.msg_id, n)

    def start(self, buffered):
        self.streaming = True
        self._event("message_start", {"type": "message_start", "message": {
            "id": self.msg_id, "type": "message", "role": "assistant",
            "model": self.model, "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": _tok_count(self.prompt),
                      "output_tokens": 0}}})
        if not buffered:
            # An incremental stream's text all lands in block 0, so it opens
            # now. A buffered (tool) turn emits whole blocks at the end instead,
            # once it knows how many there are and of which type.
            self.block_open = True
            self._event("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
            self.write(self.KEEPALIVE)

    def text(self, out):
        self._delta(0, {"type": "text_delta", "text": out})

    def finish(self, text, calls, finish, stop, raw):
        reason = _anthropic_stop_reason(finish, calls, stop)
        if not self.streaming:
            blocks = [{"type": "text", "text": text}] if text else []
            for n, c in enumerate(calls):
                blocks.append({"type": "tool_use", "id": self._tool_id(n),
                               "name": c["name"], "input": c["arguments"]})
            self.h._json(200, {
                "id": self.msg_id, "type": "message", "role": "assistant",
                "model": self.model, "content": blocks,
                "stop_reason": reason, "stop_sequence": None,
                "usage": _with_overhead(
                    {"input_tokens": _tok_count(self.prompt),
                     "output_tokens": _tok_count(raw)}, self.overhead)})
            return
        idx = 0
        if self.block_open:
            self._event("content_block_stop", {"type": "content_block_stop",
                                               "index": 0})
            self.block_open = False
            idx = 1
        if text:
            self._block(idx, {"type": "text", "text": ""},
                        {"type": "text_delta", "text": text})
            idx += 1
        for n, c in enumerate(calls):
            self._block(idx, {"type": "tool_use", "id": self._tool_id(n),
                              "name": c["name"], "input": {}},
                        {"type": "input_json_delta",
                         "partial_json": _tool_args_json(c)})
            idx += 1
        if not self.gone:       # as in the OpenAI twin: no count for nobody
            self._event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": None},
                "usage": _with_overhead({"output_tokens": _tok_count(raw)},
                                        self.overhead)})
        self._event("message_stop", {"type": "message_stop"})

    def error(self, msg, code=500):
        # As the OpenAI twin: `code` carries the 503 a shutdown refusal gets,
        # and only the non-streaming answer still has a status to carry it in.
        if not self.streaming:
            self.h._anthropic_error(code, "api_error", msg)
            return
        # Anthropic's own mid-stream failure shape, which its SDKs raise on. No
        # message_delta: a stop_reason says how the turn ENDED, and it did not
        # -- end_turn after an engine failure is what let a client file the
        # failure as a finished answer. message_stop still closes the stream.
        self._event("error", {"type": "error",
                              "error": {"type": "api_error", "message": msg}})
        self._event("message_stop", {"type": "message_stop"})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # StreamRequestHandler.setup() applies this to the connection with
    # settimeout(), so it bounds every blocking read and write on it. None is
    # the stdlib default -- wait forever -- and is what 0 or less asks for. See
    # SOCKET_TIMEOUT_S for what this does and does not limit. A timeout while
    # waiting for a request line (an idle keep-alive) is caught by the base
    # handle_one_request, which closes the connection; one while reading a
    # body is caught in do_POST; one while writing is the client having left.
    timeout = SOCKET_TIMEOUT_S if SOCKET_TIMEOUT_S > 0 else None

    def setup(self):
        super().setup()
        # Disable Nagle so per-token SSE frames flush immediately (no
        # delayed-ACK stalls stacking on top of decode latency).
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, *a):
        # No per-request access line: a streaming agent makes hundreds of
        # requests and the line for each says nothing the client does not
        # already know. What is NOT silent is a request that was not served --
        # see _log_line, which _json calls for every non-2xx it writes (bar a
        # /health 503, which is an answer rather than a refusal; see do_GET).
        pass

    def _log_line(self, what, msg):
        """One stdout line for a request that was not served as asked.

        log_message above is a no-op and nothing used to stand in for it, so an
        overflow 400, a 413, a load-shed 429/529 and an engine 500 all left no
        trace at all -- while eviction, a far smaller event, is "never silent"
        (_log_dropped). A client that was shed all afternoon was invisible in
        the log of the server that shed it.

        ASCII, one line, bounded: the message can quote the request (a bad
        max_tokens value, a parse error), a console in a legacy codepage
        raises on what it cannot encode, and a log line must never be the
        reason a response was not written -- hence the blanket except.
        """
        try:
            line = "[genie] %s %s %s: %s" % (
                what, getattr(self, "command", None) or "-",
                str(getattr(self, "path", None) or "-")[:200],
                " ".join(str(msg).split())[:300])
            print(line.encode("ascii", "backslashreplace").decode("ascii"),
                  flush=True)
        except Exception:
            pass

    def send_error(self, code, message=None, explain=None):
        # The stdlib's own refusals -- a malformed request line, an unsupported
        # method -- never pass through _json, so they are logged here.
        self._log_line(code, message or "")
        super().send_error(code, message, explain)

    def handle_one_request(self):
        # A client that disconnects (typed cancelling a turn, a closed keep-alive)
        # otherwise dumps a ConnectionReset/Aborted traceback from the base
        # handler. socket.timeout is an OSError too, so a write that timed out
        # (see `timeout`) ends the connection the same way.
        try:
            super().handle_one_request()
        except (ConnectionError, OSError):
            self.close_connection = True

    def _route(self):
        """The request path alone: no query string, no trailing slash.

        Routing used to compare self.path, which is the raw request target, so
        `/health?probe=1` -- a cache-buster, which is an ordinary thing for a
        health checker to send -- was a 404 from a healthy server.

        Never raises. urlsplit does, with ValueError, for an absolute-form
        target whose bracketed host is malformed (`GET http://[/v1/models`),
        and this runs first thing in do_GET and do_POST, outside every
        catch-all: the ValueError left through socketserver as a traceback and
        a connection closed with no response, where comparing the raw target
        had answered an ordinary 404. Garbage in, so the fallback only has to
        be something that routes nowhere -- the raw target less its query.
        """
        try:
            path = urlsplit(self.path).path
        except ValueError:
            path = self.path.split("?", 1)[0]
        return path.rstrip("/")

    def _json(self, code, obj, log=True):
        body = json.dumps(obj).encode("utf-8")
        if log and not 200 <= code < 300:
            err = obj.get("error") if isinstance(obj, dict) else None
            self._log_line(code, err.get("message", "") if isinstance(err, dict)
                           else "")
        # Read by do_POST's catch-all: once these are out, a failure can no
        # longer be answered with a status of its own.
        self._headers_sent = True
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _model_object():
        # Superset item: satisfies both OpenAI (id/object) and Anthropic
        # (type/id/display_name) model shapes.
        return {"id": MODEL_ID, "object": "model", "type": "model",
                "display_name": MODEL_ID, "owned_by": "qualcomm-genie-npu"}

    def do_GET(self):
        path = self._route()
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [self._model_object()]})
        elif path.startswith("/v1/models/"):
            # The retrieve-model route, which some SDKs call to validate a
            # model name before the first request. One model is served, so one
            # id resolves; anything else is told which.
            asked = unquote(path[len("/v1/models/"):])
            if asked == MODEL_ID:
                self._json(200, self._model_object())
            else:
                self._json(404, {"error": {
                    "message": "no model %r here; this server serves %r"
                               % (asked[:100], MODEL_ID),
                    "type": "invalid_request_error", "code": "model_not_found"}})
        elif path == "/props":
            # llama.cpp's metadata endpoint, which typed probes at startup to
            # size the context window and name the served model. Without it
            # typed falls back to DEFAULT_CONTEXT_WINDOW_TOKENS (200_000) and
            # plans every turn against a window ~49x larger than this bundle
            # has -- and that number is not decorative, it feeds the per-turn
            # token budget and the compaction threshold, so the client would
            # never suggest /compact and would overrun the model instead.
            #
            # The two fields typed actually reads:
            #   default_generation_settings.n_ctx -- the window
            #   model_alias / model_id            -- the served model's name
            #
            # Plus a namespaced `genie` block, because a router choosing among
            # an NPU, a GPU and a CPU endpoint cannot otherwise learn any of it
            # from HTTP. n_ctx alone is actively misleading here: it is the
            # SOFTWARE cap, while throughput is set by the compiled window and
            # by whether the bundle carries one graph or several -- two bundles
            # of the same n_ctx differ 2-3x on short prompts, and `poll` costs
            # about a quarter of the win from running this engine beside
            # another (both settings are a gain; see MULTI_ENGINE.md).
            # Namespaced so no llama.cpp-shaped field is misreported, and
            # additive so a client that ignores it sees what it saw before.
            #
            # `model_path` is deliberately OMITTED even though typed checks it
            # FIRST: it would take precedence and typed would then display the
            # bundle directory, disagreeing with the name /health and
            # /v1/models already report. One name everywhere beats a more
            # detailed name in one place.
            #
            # No modality field: absence reads as text-only, which is the
            # truth for this bundle. Claiming a modality it does not have
            # would be worse than saying nothing.
            poll, _where = read_poll_setting()
            lengths = read_context_lengths()
            self._json(200, {
                "default_generation_settings": {"n_ctx": read_context_size()},
                "model_alias": MODEL_ID,
                "model_id": MODEL_ID,
                "genie": {
                    "engine": "npu-hexagon-htp",
                    "single_flight": True,
                    "context_lengths": lengths,
                    "multi_length": len(lengths) > 1,
                    "poll": poll,
                },
            })
        elif path in ("/health", "/healthz"):
            # Reports the ENGINE's state, not the HTTP server's. Those come
            # apart precisely when it matters: a wedged HTP leaves this process
            # perfectly able to accept a connection and answer this endpoint
            # while being unable to serve a single token. The old unconditional
            # "ok" was therefore a check that could not fail -- it would have
            # told a supervisor everything was fine for as long as the wedge
            # lasted.
            #
            # This handler deliberately takes no engine lock, which is what
            # lets it answer AT ALL during a wedge: the lock is exactly what
            # the stuck thread is holding.
            #
            # `token_counts` comes from the module-level TOKENIZER_STATUS for
            # the same reason: it says whether every usage figure and every
            # window budget this server reports is the tokenizer's count or a
            # chars/4 estimate, and it must say so without asking the engine.
            # "estimated" before the engine has loaded, which is also true.
            snap = HEALTH.snapshot(time.time())
            code = 200 if snap["state"] == "ok" else 503
            # log=False: a 503 here is the ANSWER, not a request refused, and a
            # supervisor polls it for as long as the wedge lasts. The watchdog
            # announces the state change once; a line per poll would bury it.
            self._json(code, dict(
                snap, status=snap["state"], model=MODEL_ID,
                token_counts=("exact" if TOKENIZER_STATUS == GENIE_STATUS_SUCCESS
                              else "estimated")), log=False)
        else:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def _request_error(self, code, msg, path=None, etype=None):
        """One error, in whichever envelope the TARGETED api uses.

        do_POST used to fork on path for the tools refusal and the load shed but
        not for the two failures above them, so a malformed body sent to
        /v1/messages came back as {"error": {...}} with no top-level "type" --
        the OpenAI shape, which an Anthropic client cannot parse. A client that
        cannot read the error is told nothing at the one moment it needs to be
        told something, so the envelope has to follow the endpoint everywhere,
        not only where it was convenient.

        `etype` defaults by code and by API: a 4xx is the request's fault
        (invalid_request_error in both), a 5xx is this server's, which the two
        ecosystems spell differently -- api_error on Anthropic, server_error on
        OpenAI, the same pair the engine-failure 500s use.
        """
        anthropic = (path if path is not None else self._route()) == "/v1/messages"
        if etype is None:
            etype = ("invalid_request_error" if code < 500
                     else "api_error" if anthropic else "server_error")
        if anthropic:
            self._anthropic_error(code, etype, msg)
        else:
            self._json(code, {"error": {"message": msg, "type": etype}})

    def do_POST(self):
        path = self._route()
        # Per REQUEST, not per handler: one Handler serves every request on a
        # keep-alive connection. Read by _answer_failure below.
        self._headers_sent = False
        self._generating = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._request_error(400, "bad Content-Length header", path)
            self.close_connection = True
            return
        # This server reads exactly Content-Length bytes, so a chunked body
        # would leave its frames unread and desync the next request on a
        # keep-alive connection -- answered rather than half-consumed.
        if self.headers.get("Transfer-Encoding"):
            self._request_error(411, "chunked bodies are not supported; "
                                     "send Content-Length", path)
            self.close_connection = True
            return
        # Both bounds BEFORE the read and before the single-flight semaphore,
        # which is the whole point: MAX_INFLIGHT bounds generations, not bytes,
        # so neither was reachable only via a permit. The negative half is the
        # one that bites hardest -- rfile.read(-1) reads to EOF, which on a
        # keep-alive socket never comes, so a single request with
        # `Content-Length: -1` parks a handler thread for the life of the
        # process and costs nothing to send.
        if length < 0:
            self._request_error(400, "negative Content-Length", path)
            self.close_connection = True
            return
        if length > MAX_BODY_BYTES:
            self._request_error(413, "request body too large: %d bytes (limit %d)"
                                % (length, MAX_BODY_BYTES), path)
            self.close_connection = True
            return
        try:
            body = self.rfile.read(length)
        except OSError as e:
            # socket.timeout is an OSError: the client promised `length` bytes
            # and stopped sending them. Before Handler.timeout existed this
            # read parked the thread for good. Treated as the client having
            # left -- no response, since a peer that will not finish its own
            # request is not waiting to read one -- but never silently.
            self._log_line("dropped", "stopped sending its %d-byte body (%s)"
                           % (length, e))
            self.close_connection = True
            return
        try:
            req = json.loads(body or b"{}")
        except Exception as e:
            self._request_error(400, "bad JSON: %s" % e, path)
            return
        # Valid JSON that is not an object used to reach req.get() and raise,
        # which drops the connection with no response at all -- the one failure
        # a client cannot distinguish from the server being dead.
        if not isinstance(req, dict):
            self._request_error(400, "body must be a JSON object", path)
            return
        gen = {"/v1/chat/completions": self._openai_chat,   # OpenAI Chat Completions
               "/v1/messages": self._anthropic_messages      # Anthropic Messages (typed)
               }.get(path)
        if gen is None:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        # Tools are supported when the BUNDLE can do them, and refused
        # loudly when it cannot -- never accepted-and-dropped. The capability
        # is probed from the tokenizer's vocab at startup (probe_tool_support),
        # so a text-only bundle still gets the old honest 400 and typed's
        # probeLocalToolCalls still reads it as "disable tools for this
        # session" instead of shipping schemas the model would ignore.
        #
        # Checked BEFORE the single-flight lock: refusing costs no NPU time,
        # so it must not queue behind a live generation.
        if req.get("tools") and not TOOLS_OK:
            msg = ("tool calling is not supported: this bundle's tokenizer has "
                   "no <tool_call> token, so %s cannot emit a parseable call. "
                   "Retry without `tools`." % MODEL_ID)
            self._request_error(400, msg, path)
            return
        if not _INFLIGHT.acquire(blocking=False):
            # NPU is single-flight and the small queue is full -> shed load.
            # 429 on OpenAI, 529 on Anthropic -- the codes differ because the
            # two ecosystems spell backpressure differently, so this one cannot
            # use _request_error's shared code.
            if path == "/v1/messages":
                self._anthropic_error(529, "overloaded_error",
                                      "server busy; NPU is single-flight")
            else:
                self._json(429, {"error": {"message": "server busy; NPU is single-flight",
                                           "type": "overloaded_error"}})
            return
        try:
            gen(req)
        except (ConnectionError, OSError):
            # The client went away mid-response. Nothing to answer and nobody
            # to answer it to; handle_one_request closes the connection.
            raise
        except Exception as e:
            self._answer_failure(e, path)
        finally:
            _INFLIGHT.release()

    def _answer_failure(self, e, path):
        """The catch-all under gen(req): whatever went wrong, SAY so.

        There was none, so anything a handler did not anticipate raised out of
        do_POST and the client got a closed connection with no response -- the
        failure this file itself calls the one "a client cannot distinguish
        from the server being dead". Reachable with valid JSON and a wrong-typed
        field: `temperature: "hot"`, `stop: 5`, `stream_options: "yes"`, a
        tool_calls entry that is not an object. Each door check above exists
        because one such shape was found; this is for the ones not found yet.

        400 when the request never reached the engine and the exception is one
        a wrong-typed field produces -- it is the client's to fix, and a router
        must not shed it to the next engine as though this one were broken.
        Everything else is a 500. The message names the exception either way,
        because "bad request" alone sends the caller auditing the wrong thing.

        Only while the headers are unsent. After that a status line has gone
        out and a second one would corrupt the response, so the connection is
        closed -- which is at least unambiguous -- and the log says why.
        """
        what = "%s: %s" % (type(e).__name__, e)
        if self._headers_sent:
            self._log_line("failed mid-response", what)
            self.close_connection = True
            return
        if not self._generating and isinstance(
                e, (TypeError, ValueError, AttributeError, KeyError)):
            self._request_error(400, "malformed request (%s)" % what, path)
        else:
            self._request_error(500, what, path)

    def _openai_chat(self, req):
        messages = req.get("messages", [])
        if not messages:
            self._json(400, {"error": {"message": "messages required",
                                       "type": "invalid_request_error"}})
            return
        if not _is_message_list(messages):
            self._json(400, {"error": {"message": "messages must be a list of objects",
                                       "type": "invalid_request_error"}})
            return
        # Checked the way `messages` is, and for the same reason: ChatML.build
        # json.dumps whatever it iterates, so a string came out as one quoted
        # "function signature" per CHARACTER and a dict as its keys -- a garbage
        # schema served with a 200.
        tools = req.get("tools")
        if tools is not None and not _is_tool_list(tools):
            self._json(400, {"error": {"message": "tools must be a list of objects",
                                       "type": "invalid_request_error"}})
            return
        tools = tools or None
        stream = bool(req.get("stream", False))
        try:
            max_tokens = _max_tokens(req)
        except ValueError as e:
            self._request_error(400, str(e))
            return
        # Resolved ONCE and threaded through, rather than re-derived downstream.
        # It decides both what goes into the prompt and whether a closing think
        # tag in the OUTPUT can be ours to strip, and those two answers have to
        # come from the same call or the response path strips against a prompt
        # it did not send.
        thinking = _wants_thinking(req)
        # Every field that can REFUSE is read before the prompt is built:
        # building can evict, evicting can summarise, and summarising is an NPU
        # generation -- not something to spend on a request whose `temperature`
        # turns out to be "hot". A wrong-typed value raises here and do_POST's
        # catch-all answers it 400 (_answer_failure).
        stop = _stop_sequences(req)
        sampler = _sampler_params(req, tools_active=bool(tools))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        prompt, dropped, fits, overhead = build_windowed(
            messages, tools=tools, thinking=thinking, max_tokens=max_tokens)
        if not fits:
            self._json(400, {"error": {"message": _overflow_msg(prompt, max_tokens),
                                       "type": "invalid_request_error"}})
            return
        unsendable = _unsendable(prompt)
        if unsendable:
            self._request_error(400, unsendable)
            return
        if dropped:
            _log_dropped(dropped)
        created = int(time.time())
        cmpl_id = "chatcmpl-%d" % created
        gen_kw = {"stop": stop, "sampler": sampler, "overhead": overhead,
                  "prefilled": not thinking}
        if stream:
            self._stream(prompt, max_tokens, cmpl_id, created,
                         tools_active=bool(tools), include_usage=include_usage,
                         **gen_kw)
        else:
            self._complete(prompt, max_tokens, cmpl_id, created,
                           tools_active=bool(tools), **gen_kw)

    # The four response paths. Thin on purpose: each only picks an emitter and
    # says whether it streams -- everything they used to do four times over is
    # _run, below. `prefilled` defaults to False here as in _maybe_strip_think,
    # and for its reason: a caller that forgets it ships a visible duplicate
    # someone reports, where the other default would silently eat content.

    def _complete(self, prompt, max_tokens, cmpl_id, created, tools_active=False,
                  stop=None, sampler=None, overhead=0, prefilled=False):
        self._run(_OpenAIEmitter(self, prompt, cmpl_id, created, overhead),
                  max_tokens, tools_active, stop, sampler, prefilled, stream=False)

    def _stream(self, prompt, max_tokens, cmpl_id, created, tools_active=False,
                stop=None, sampler=None, overhead=0, include_usage=False,
                prefilled=False):
        self._run(_OpenAIEmitter(self, prompt, cmpl_id, created, overhead,
                                 include_usage=include_usage),
                  max_tokens, tools_active, stop, sampler, prefilled, stream=True)

    # ---- the one generation driver ----------------------------------------

    def _sse_headers(self):
        self._headers_sent = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length/chunked on an SSE body, so frame the response by
        # connection close -- otherwise read-to-EOF clients hang on keep-alive.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def _client_gone(self):
        """Has the peer closed the connection? Asked without reading a byte.

        A non-streaming response writes nothing until the answer is complete,
        so no write can fail and an abandoned request used to run to its cap
        with the single-flight NPU held the whole way. The socket still knows:
        a closed peer makes it READABLE with nothing to read. select() with a
        zero timeout never blocks, and MSG_PEEK leaves a pipelined next request
        in place for the read loop that owns it -- bytes waiting means a live
        client, only an empty read means a departed one. (This is the test
        llama.cpp's server applies between tokens, for the same reason.)

        A client that half-closes -- shuts down its sending side, then waits
        for the answer -- looks the same from here and is treated as gone. No
        HTTP/1.1 client library does that on a keep-alive connection.

        False with no socket at all, which is the device-free test harness.
        """
        conn = getattr(self, "connection", None)
        if conn is None:
            return False
        try:
            readable, _, _ = select.select([conn], [], [], 0)
            return bool(readable) and conn.recv(1, socket.MSG_PEEK) == b""
        except (BlockingIOError, InterruptedError, TimeoutError):
            return False            # nothing to read after all: still there
        except (OSError, ValueError):
            return True             # reset, or already closed under us

    def _run(self, em, max_tokens, tools_active, stop, sampler, prefilled, stream):
        """Generate once and answer through `em`. All four response paths.

        `stream` chooses SSE frames or one JSON body; `tools_active` chooses
        BUFFERED or incremental within a stream. A <tool_call> block means
        nothing until it CLOSES -- streaming it token by token would hand the
        client half a call to guess about -- so a tool turn is generated fully
        and then emitted as well-formed frames: still SSE (the client asked for
        SSE), just not incremental. Buffering is the honest trade; a partial
        tool call is not.

        What that means for GENIE_STRIP_THINK: only an INCREMENTAL stream is
        always faithful to the model, because a frame already sent cannot be
        retracted and a <think> pair cannot be recognised until it closes. Every
        buffered path -- non-streaming, and a tool stream on either API --
        holds the whole generation first and honours the flag. The orphan close
        is separate and is removed everywhere: by _maybe_strip_think on the
        buffered paths, by the gate on the incremental one.

        Always query_stream, never query, including for a non-streaming
        response: the generation then runs on a worker, this thread stays free
        to notice the client leaving (em.probe), and walking away from the
        generator aborts that turn and no other.

        Usage is counted from the RAW generation on every path. What survives
        the strip or the gate is the right thing to have DELIVERED and the
        wrong thing to have BILLED: the model produced the rest too, and spent
        the time doing it. Measured 2026-08-28, the same prompt and cap took
        ~5.5s under two seeds and reported 51 tokens in one and 103 in the
        other; anything dividing tokens by time then read half rate --
        bench_endpoint reported 9.25 t/s against a true 17.7. One path (the
        OpenAI tool stream) went on counting post-strip after the rest were
        fixed, under a comment claiming parity; one driver is the fix for that.
        """
        self._generating = True
        if stream:
            self._sse_headers()
            em.start(buffered=tools_active)
        else:
            # A non-streaming response has no first frame to fail, so it asks
            # the socket. Without this only a STREAM's early departure was
            # free: a non-streaming client that had already closed -- a
            # urllib/httpx timeout firing while build_windowed spent seconds
            # summarising, say -- went straight on to the query below.
            em.probe()
        if em.gone:
            # Left before the query started. signal_abort had no turn to aim
            # at, so without this the request still cost everyone the lock
            # wait, a full prefill and a token before the loop's first probe
            # or failed write noticed.
            self.close_connection = True
            return
        # The gate is the incremental path's strip. It holds the OPENING of the
        # stream, so while it holds nothing goes out -- the same silence as a
        # buffered turn, and probed the same way.
        gate = _OrphanGate(prefilled=prefilled) if stream and not tools_active else None
        res, chunks = {}, []
        gen = ENGINE.query_stream(em.prompt, res, max_tokens=max_tokens,
                                  stop=stop, sampler=sampler)
        try:
            for i, chunk in enumerate(gen):
                chunks.append(chunk)
                out = gate.feed(chunk) if gate is not None else ""
                if out:
                    em.text(out)
                elif (gate is None or not gate.open) and i % 8 == 0:
                    # Conditioned on the SILENCE, not on `out` being falsy: an
                    # empty chunk is no reason to probe, and once the gate is
                    # open em.text() detects a disconnect on its own.
                    em.probe()
                if em.gone:
                    break
        except Exception as e:
            # query_stream reports an engine failure through res["error"]; this
            # is for one that raises instead. Same answer either way, and with
            # a stream's 200 already sent it must not escape as a dropped
            # connection.
            res["error"] = str(e)
        finally:
            # Deterministically, not whenever the generator is collected:
            # closing it before the generation has ended is what aborts the
            # turn (see GenieEngine.query_stream).
            gen.close()
        if em.gone:
            self.close_connection = True    # nobody to answer, nothing to keep
            return
        if gate is not None:
            tail = gate.flush()
            if tail:
                em.text(tail)
        if res.get("error"):
            closing = res.get("closing")
            if stream:
                # The 200 has already gone out, so the line _json prints for a
                # non-2xx will never be written for this failure.
                self._log_line("refused mid-stream" if closing
                               else "engine failure mid-stream", res["error"])
            # 503, not 500, for a turn shutdown refused at the door: the
            # request was never started and nothing about it was wrong, which
            # is this server's documented "503 = shed" contract -- a router
            # can send it to another leg instead of counting it as a failure.
            em.error(res["error"], code=503 if closing else 500)
            return
        raw = "".join(chunks)
        text, calls = "", []
        if gate is None:
            text = _maybe_strip_think(raw, prefilled)
            if tools_active:
                text, calls = parse_tool_calls(text)
        em.finish(text, calls, res.get("finish", "stop"), stop, raw)

    # ---- Anthropic Messages API (POST /v1/messages) -----------------------

    def _anthropic_error(self, code, etype, msg):
        self._json(code, {"type": "error", "error": {"type": etype, "message": msg}})

    def _anthropic_messages(self, req):
        if not req.get("messages"):
            self._anthropic_error(400, "invalid_request_error", "messages required")
            return
        if not _is_message_list(req.get("messages")):
            self._anthropic_error(400, "invalid_request_error",
                                  "messages must be a list of objects")
            return
        # ECHOED, deliberately, where the OpenAI leg, /health, /v1/models and
        # /props all say MODEL_ID. The real Messages API answers with the model
        # the request named, and a router fanning one request out to several
        # engines matches the reply to the request by that field -- so a client
        # asking for "claude-x" is told "claude-x" by an NPU Qwen. That is a
        # routing label, not a claim about what ran; what is actually being
        # served is what the other four endpoints report.
        model = req.get("model") or MODEL_ID
        try:
            max_tokens = _max_tokens(req)
        except ValueError as e:
            self._anthropic_error(400, "invalid_request_error", str(e))
            return
        # Before _anthropic_to_prompt, which walks the list: a string here used
        # to die in _anthropic_tools on `"a".get` with no response written.
        tools = req.get("tools")
        if tools is not None and not _is_tool_list(tools):
            self._anthropic_error(400, "invalid_request_error",
                                  "tools must be a list of objects")
            return
        tools = tools or None
        # Read before the prompt is built, as on the OpenAI leg and for the
        # same reason: a wrong-typed field should refuse the request before it
        # has cost an NPU summarisation, not after.
        stop = _stop_sequences(req)
        sampler = _sampler_params(req, tools_active=bool(tools))
        prompt, dropped, fits, overhead = _anthropic_to_prompt(
            req, tools=tools, max_tokens=max_tokens)
        if not fits:
            self._anthropic_error(400, "invalid_request_error",
                                  _overflow_msg(prompt, max_tokens))
            return
        unsendable = _unsendable(prompt)
        if unsendable:
            self._anthropic_error(400, "invalid_request_error", unsendable)
            return
        if dropped:
            _log_dropped(dropped)
        msg_id = "msg_%d" % int(time.time())
        # Same resolution as the OpenAI leg. _anthropic_to_prompt derives this
        # internally to build the prompt; asking again here is cheap and pure,
        # and it keeps the response path from having to guess what was sent.
        gen_kw = {"stop": stop, "sampler": sampler, "overhead": overhead,
                  "prefilled": not _wants_thinking(req)}
        if bool(req.get("stream", False)):
            self._anthropic_stream(prompt, max_tokens, model, msg_id,
                                   tools_active=bool(tools), **gen_kw)
        else:
            self._anthropic_complete(prompt, max_tokens, model, msg_id,
                                     tools_active=bool(tools), **gen_kw)

    def _anthropic_complete(self, prompt, max_tokens, model, msg_id,
                            tools_active=False, stop=None, sampler=None,
                            overhead=0, prefilled=False):
        self._run(_AnthropicEmitter(self, prompt, msg_id, model, overhead),
                  max_tokens, tools_active, stop, sampler, prefilled, stream=False)

    def _anthropic_stream(self, prompt, max_tokens, model, msg_id,
                          tools_active=False, stop=None, sampler=None,
                          overhead=0, prefilled=False):
        self._run(_AnthropicEmitter(self, prompt, msg_id, model, overhead),
                  max_tokens, tools_active, stop, sampler, prefilled, stream=True)


def _exit_for_supervisor(detail):
    """Replace this process. The only recovery available for a wedged device.

    Split out and routed through watchdog's `on_wedge` so that a test can
    substitute it. That is not a stylistic preference: an os._exit reached by
    any path other than the injectable one takes the TEST RUNNER down with it,
    silently and with no output, which is a genuinely nasty thing to leave in
    the way of whoever writes the next test here.
    """
    if not WEDGE_EXIT:
        return None
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit, not sys.exit: sys.exit unwinds to main's `finally`, which calls
    # GenieDialog_free on the very driver that is already stuck -- that call can
    # hang too, and then the process never leaves at all. There is nothing worth
    # cleaning up in a process whose device is gone.
    os._exit(EXIT_WEDGED)


def _wedge_clearing_note():
    """What can still clear this stall -- the clause both STALL lines end on.

    Read at print time, not built once, because GENIE_WEDGE_EXIT decides
    whether there IS an exit to wait for and the stall lines used to promise
    one either way: "only the exit below can clear it", then a WEDGED stanza
    saying "Exiting 75 so a supervisor restarts a clean process", on a process
    that then stayed up forever. Whoever sets GENIE_WEDGE_EXIT=0 sets it
    BECAUSE nothing supervises this process, so those lines told exactly the
    operator who has to act by hand to sit and wait for a restart that was
    never coming. The startup banner already branches the same way.
    """
    if WEDGE_EXIT:
        return "only the exit below can clear it"
    return ("nothing in this process can clear it -- with GENIE_WEDGE_EXIT=0 "
            "the server stays up wedged, answering 503, until you restart it "
            "by hand")


def watchdog(engine, health, interval=5.0, on_wedge=None, iterations=None):
    """Escalate a stalled engine: signal an abort, then replace the process.

    Runs on its own thread precisely because the request threads are the ones
    that get stuck -- a watchdog that shared their fate could not report on it.

    The escalation has two steps because they have different costs. Signalling
    an abort is free and is the mechanism Genie provides for exactly this, so
    it is always worth one attempt. Exiting is not free -- it drops in-flight
    requests -- so it happens only after the abort has been given `grace`
    seconds to take and has not.

    `on_wedge` and `iterations` exist so the escalation can be tested without
    ending the test runner's own process. Every path to the exit goes through
    `on_wedge`, so overriding it is sufficient to make this safe in a test --
    there is no second route that could still terminate the runner.

    The steady states ANNOUNCE ONCE. `failing` persists until a generation
    succeeds and `wedged` persists forever under GENIE_WEDGE_EXIT=0, so
    printing them every `interval` reprinted a multi-line stanza every five
    seconds for as long as the outage lasted -- burying the first occurrence,
    which is the one carrying the original cause, under thousands of identical
    copies of itself. The latch clears when the engine returns to ok, so a
    second, genuinely new episode is announced again. `stalled` is deliberately
    NOT latched: each line marks a fresh abort signal, and it is bounded by the
    grace period rather than open-ended.
    """
    on_wedge = _exit_for_supervisor if on_wedge is None else on_wedge
    n = 0
    announced = None
    while iterations is None or n < iterations:
        n += 1
        time.sleep(interval)
        now = time.time()
        state, detail = health.assess(now)
        if state == "ok":
            announced = None
            continue
        if state == "failing":
            if announced != "failing":
                print("[genie] UNHEALTHY: %s" % detail, flush=True)
                announced = "failing"
            continue
        if state == "stalled":
            # The stall is announced BEFORE the attempt and what the attempt
            # DID after it, because signal_abort can block: it takes
            # _abort_lock, and a handler's own abort holds that across a
            # native GenieDialog_signal that a wedged driver need not return
            # from. A single line composed afterwards would say nothing at all
            # in the one case where this thread is the only one still talking.
            # So this line reports the stall and nothing else -- the claim
            # "signalling abort" belongs to the outcome lines below, which
            # know whether a signal went.
            print("[genie] STALL: %s" % detail, flush=True)
            sent = None
            try:
                # any_turn: this thread consumes no stream, so a bare
                # signal_abort() from it would be aimed at nothing. stalled:
                # so the turn is booked as the failure it is, not as the
                # ordinary finish a client's abort is.
                sent = engine.signal_abort(any_turn=True, stalled=True)
            except Exception as e:
                print("[genie] abort signal failed: %s" % e, flush=True)
            # signal_abort says what it did, and two of its answers mean NO
            # native ABORT went out. Say which, or the WEDGED line after it
            # claims Genie ignored a signal, which sends whoever reads it to
            # file "Genie ignores ABORT" against a driver that was never asked.
            if sent is False:
                # No turn held the dialog: the stall is inside a host-side
                # call (the tokenizer sizing a request).
                print("[genie] nothing in flight to abort: the stall is "
                      "in a host-side call, so %s" % _wedge_clearing_note(),
                      flush=True)
            elif sent == ABORT_FLAGGED:
                # A turn holds the dialog but is not inside GenieDialog_query:
                # it is stuck in one of the calls BEFORE it (the reset, the
                # stop sequences, the sampler, the token cap), where no signal
                # is sent -- see GenieEngine._run_query for why not.
                print("[genie] no native ABORT was sent: the turn is stalled "
                      "in a call BEFORE its query, which ABORT cannot reach. "
                      "It is flagged, so it will not start its query if that "
                      "call returns; otherwise %s" % _wedge_clearing_note(),
                      flush=True)
            elif sent:
                # The ordinary case, and the only one where the words are
                # simply true: the turn was inside its query and got the
                # signal. Said here rather than in the STALL line above, so
                # that "signalling abort" never appears over an abort that
                # was not sent.
                print("[genie] signalling abort to the generation in flight",
                      flush=True)
            # After the attempt, so the grace clock records what the attempt
            # was; an abort that RAISED still starts it, as it always did.
            health.note_stall_signalled(
                now, native=sent is not False and sent != ABORT_FLAGGED)
            continue
        # wedged
        if announced != "wedged":
            print("[genie] WEDGED: %s" % detail, flush=True)
            # Announced ONCE, so this stanza is the whole record of the event
            # for whoever reads the log afterwards -- which is why it must not
            # describe the other configuration's ending. Under
            # GENIE_WEDGE_EXIT=0 nothing exits: the process stays up with a
            # thread parked in the driver forever, /health answers 503, and
            # every generation request queues behind the stuck call until
            # GENIE_MAX_INFLIGHT sheds it. Only a restart by hand ends that,
            # and saying "Exiting 75 so a supervisor restarts a clean process"
            # told the one operator who has to do it that somebody else would.
            if WEDGE_EXIT:
                ending = ("Exiting %d so a supervisor restarts a clean "
                          "process. (GENIE_WEDGE_EXIT=0 to stay up and keep "
                          "reporting 503.)" % EXIT_WEDGED)
            else:
                ending = ("NOT exiting: GENIE_WEDGE_EXIT=0. This process "
                          "stays up wedged -- /health answers 503 and "
                          "generation requests queue behind the stuck call "
                          "or are shed -- until you restart it by hand. "
                          "(Unset GENIE_WEDGE_EXIT to exit %d instead, for a "
                          "supervisor to restart a clean process.)"
                          % EXIT_WEDGED)
            print("[genie] The engine cannot be recovered in this process: the "
                  "stuck call is inside the Genie driver, holding the engine "
                  "lock, and Python cannot reclaim a thread blocked in native "
                  "code. %s" % ending, flush=True)
            announced = "wedged"
        result = on_wedge(detail)
        # In production on_wedge is _exit_for_supervisor, which either replaces
        # the process or -- under GENIE_WEDGE_EXIT=0 -- returns None, and None
        # means keep watching and keep reporting 503 rather than spin silently
        # on a dead device. A non-None return is a TEST hook, nothing more: a
        # stand-in returns something to end this loop, and no production code
        # reads what watchdog returns.
        if result is not None:
            return result
    return None


def main():
    global ENGINE, TEMPLATE, TOOLS_OK
    # Before the model load, not after: loading is 11-35s of work (measured on
    # the 8192 multi bundle: 10.8-15.0s warm, 34.4s after heavy disk traffic --
    # the "30-50s" this said predated that measurement), and finding
    # out afterwards that the port is taken wastes all of it. On Windows the
    # bind would not report the collision at all -- see Server.
    if port_in_use(HOST, PORT):
        sys.exit(
            "something is already serving %s:%d.\n"
            "This server would appear to start normally while that other "
            "process kept answering, so requests would hit ITS model, not the "
            "bundle named here. Stop it first, or set GENIE_PORT to a free "
            "port." % (HOST, PORT))
    # Before the model load, for the same reason as the port check above:
    # these two settings are worth more than everything else this server does,
    # and finding out after 11-35s of loading that the bundle is configured to
    # run at half speed wastes all of it.
    for line in bundle_config_warnings():
        print("[genie] %s" % line, flush=True)
    TEMPLATE = load_chat_template()
    TOOLS_OK = probe_tool_support()
    # BIND before the model load, LISTEN after it. Two steps, because they
    # answer two different questions.
    #
    # The bind is early for the same reason as the port check: a bind that
    # cannot succeed -- an address this machine does not have (WinError
    # 10049), a port that became busy between the check and here -- should
    # cost nothing, not 11-35s of loading a bundle first. It also claims the
    # port for the length of the load: a second instance started meanwhile
    # passes port_in_use (nothing is accepting yet) and then fails ITS bind
    # by name, ahead of its own load, because Server asks for the port
    # exclusively -- whatever GENIE_HOST either instance uses, which is the
    # part SO_REUSEADDR-off alone did not cover (see Server.server_bind).
    # OverflowError is what bind() raises for a port outside 0-65535 --
    # GENIE_PORT=80800 parses as an integer, so _int_env has nothing to say
    # about it.
    #
    # The listen is late because "the port answers" has to go on meaning "the
    # model is resident". It did when the bind came after the load, and
    # clients were written to that: bench_servers.wait_port treats the first
    # successful connect as a server that started. Were the listen early too,
    # a load that failed seconds in -- GenieDialog_create refusing the bundle,
    # the HTP held by another session -- would exit AFTER the port had
    # answered, and that start failure would be reported as a server that
    # came up and then failed its first request. A bound socket that is not
    # listening refuses a connect exactly as a closed port does (measured
    # here on Windows 11, v4 and v6: ConnectionRefusedError 10061 after the
    # stack's ~2s of SYN retries, a timeout for any client that waits less),
    # so nothing connects until server_activate below.
    try:
        srv = Server((HOST, PORT), Handler, bind_and_activate=False)
        try:
            srv.server_bind()
        except BaseException:
            srv.server_close()
            raise
    except (OSError, OverflowError) as e:
        # Both Windows errnos for "that port is taken", because the exclusive
        # bind reports them by which side asked first: 10048 (address in use)
        # when this bind is the narrower one, 10013 (access forbidden) when it
        # is the wildcard trying to cover a port someone already holds on one
        # address. 10013 alone reads like a firewall or a privileged port, so
        # say what it means HERE -- and say that the other holder need not be
        # using the same GENIE_HOST, which is the collision port_in_use
        # cannot see because nothing is accepting during a load.
        busy = ""
        if getattr(e, "winerror", None) in (10013, 10048):
            busy = ("\nSomething already holds that port -- most likely "
                    "another instance of this server still loading its "
                    "bundle, on ANY GENIE_HOST: the bind is exclusive, so "
                    "0.0.0.0 and 127.0.0.1 cannot share a port.")
        sys.exit("cannot bind %s:%d: %s%s\nSet GENIE_HOST to an address this "
                 "machine has (127.0.0.1, ::1, or 0.0.0.0 to expose it) and "
                 "GENIE_PORT to a free port." % (HOST, PORT, e, busy))
    try:
        ENGINE = load_engine()
    except BaseException:
        # Every load_engine failure is a sys.exit. Give the port back on the
        # way out rather than leave it to interpreter teardown: an in-process
        # caller (a test, a wrapper) that catches the exit would otherwise
        # hold a bound, dead socket for as long as it lives.
        srv.server_close()
        raise
    try:
        srv.server_activate()
    except OSError as e:
        # Not reachable on Windows, where the bind above is exclusive. On
        # POSIX SO_REUSEADDR lets two sockets BIND one port while neither is
        # listening, and the loser finds out here.
        srv.server_close()
        sys.exit("cannot listen on %s:%d: %s\nSomething else started "
                 "serving that port while the model was loading."
                 % (HOST, PORT, e))
    print("[genie] endpoint on http://%s:%d  (model=%s)" % (HOST, PORT, MODEL_ID), flush=True)
    _exposure = host_exposure_warning()
    if _exposure:
        print("[genie] %s" % _exposure, flush=True)
    print("[genie]   POST /v1/chat/completions (OpenAI)   POST /v1/messages (Anthropic)",
          flush=True)
    print("[genie]   GET /v1/models   GET /v1/models/<id>   GET /props   "
          "GET /health (alias /healthz)", flush=True)
    _lengths = read_context_lengths()
    _poll, _ = read_poll_setting()
    # Printed even when nothing is wrong, so a log or a screenshot carries what
    # a measurement has to be filed under. Two bundles of the same n_ctx differ
    # 2-3x on short prompts, and this is the only place the difference shows.
    print("[genie]   bundle: n_ctx=%d  context_lengths=%s (%s)  poll=%s"
          % (read_context_size(),
             _lengths or "unknown",
             "multi-length" if len(_lengths) > 1 else
             "SINGLE-length -- 2-3x slower on short prompts"
             if len(_lengths) == 1 else "unreadable",
             "unset (ships true)" if _poll is None else _poll), flush=True)
    print("[genie]   sampling: server-level only (dialog.sampler in "
          "genie_config.json). Per-request temperature/top_p are accepted but "
          "NOT honoured -- QAIRT 2.45 ignores a post-create sampler apply.",
          flush=True)
    cap = summary_token_cap()
    if cap != SUMMARY_MAX_TOKENS:
        print("[genie]   summary note capped at %d tokens (n_ctx=%d), not the "
              "requested %d" % (cap, read_context_size(), SUMMARY_MAX_TOKENS),
              flush=True)
    # Both branches spelled out. Interpolating only the state into a fixed
    # sentence made the opt-in branch contradict itself -- it announced
    # reasoning ON and then told a user who had just set GENIE_THINKING=1 to
    # set GENIE_THINKING=1. A startup line that argues with itself is worse
    # than none, because it is read once, at the moment the operator is
    # deciding whether the server is configured the way they meant.
    if THINKING_DEFAULT:
        print("[genie]   reasoning: ON server-wide (GENIE_THINKING). Qwen3's "
              "<think> block costs 10-17x on an agent turn (41s vs 2.4s "
              "measured) and its length swings run to run, so agent clients "
              "should turn it off per request: "
              "chat_template_kwargs.enable_thinking=false, "
              "reasoning_effort=\"none\", or thinking={\"type\":\"disabled\"}.",
              flush=True)
    else:
        print("[genie]   reasoning: suppressed by default. Qwen3's <think> "
              "block costs 10-17x on an agent turn (41s vs 2.4s measured), so "
              "it is prefilled closed unless asked for. GENIE_THINKING=1 "
              "re-enables it server-wide; per request, "
              "chat_template_kwargs.enable_thinking=true, "
              "reasoning_effort=\"high\", or thinking={\"type\":\"enabled\"}.",
              flush=True)
    print("[genie]   tool calling: %s" %
          ("enabled (<tool_call> in bundle vocab)" if TOOLS_OK
           else "unsupported by this bundle -- requests with `tools` get a 400"),
          flush=True)
    print("[genie]   supervision: /health reports engine state and 503s when it "
          "cannot serve. A stall (no first token in %.0fs, or no further token "
          "in %.0fs) is aborted; if that does not take within %.0fs the process "
          "exits %d for a supervisor to restart%s."
          % (FIRST_TOKEN_TIMEOUT_S, STALL_TIMEOUT_S, WEDGE_GRACE_S, EXIT_WEDGED,
             "" if WEDGE_EXIT else " -- DISABLED by GENIE_WEDGE_EXIT=0"),
          flush=True)
    threading.Thread(target=watchdog, args=(ENGINE, HEALTH),
                     daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[genie] shutting down", flush=True)
    finally:
        # Abort whatever holds the dialog, then free it UNDER the engine lock.
        # Handler and worker threads are daemons, so a generation can still be
        # inside GenieDialog_query when Ctrl-C lands here, and freeing the
        # handle under it is a fault in a process that is already leaving: at
        # best a spurious 0xC0000005 in the WER log that muddies the
        # driver-fault evidence the docs rely on. The abort takes effect within
        # one decode step; if the lock is still held after that the driver is
        # stuck, and _exit_for_supervisor's reasoning applies -- a free on a
        # stuck driver can hang too, and there is nothing worth cleaning up.
        #
        # close(), not a bare GenieDialog_free: the daemons outlive this path,
        # and the free alone left each of them one lock acquisition away from
        # a native call on the freed handle -- see GenieEngine.close.
        #
        # begin_shutdown, not a bare signal_abort: the abort frees the engine
        # lock and the lock goes to the longest waiter, which under load is a
        # QUEUED request, not the close() below. It started a fresh query,
        # close() timed out on it, and this path printed the "still inside the
        # driver" line about a healthy driver while the process left with a
        # generation running. begin_shutdown refuses the queued turns first --
        # they get a 503 -- and then aborts.
        ENGINE.begin_shutdown()
        if not ENGINE.close(timeout=SHUTDOWN_FREE_TIMEOUT_S):
            print("[genie] a generation is still inside the driver; leaving "
                  "the dialog for the OS to reclaim", flush=True)


if __name__ == "__main__":
    main()
