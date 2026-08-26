#!/usr/bin/env python3
"""
OpenAI-compatible HTTP server for a Qualcomm Genie NPU LLM bundle
(Snapdragon, Hexagon HTP). Supported targets are the Windows-on-Snapdragon
Hexagons: v73 (X Elite / X Plus) and v81 (X2 Elite). That set is DERIVED at
startup, not hardcoded -- an arch counts only if the SDK ships both its skel
and its Windows stub -- so a future Hexagon works without editing this file,
and the Android-only archs (v75, v79) are excluded with a reason.

Loads the Genie context-binary bundle ONCE via the Genie C API (ctypes ->
Genie.dll) so the model stays resident on the HTP; every /v1/chat/completions
request reuses it (no ~8.5s per-request reload that genie-t2t-run.exe would pay).

Pure Python stdlib -- no pip dependencies. Must run on a native ARM64 (aarch64)
Python, because Genie.dll and its Qnn* deps are aarch64-windows-msvc.

Config via environment (all have sensible defaults for this repo's scratchpad):
  GENIE_BUNDLE_DIR   dir with genie_config.json + part*_of_*.bin + tokenizer.json
  GENIE_SDK_DIR      QAIRT 2.45 SDK root (contains lib/aarch64-windows-msvc + lib/hexagon-v*)
  GENIE_HEXAGON_ARCH pin one skel arch (e.g. "v81"); default = offer them all
  GENIE_HOST         bind host   (default 127.0.0.1)
  GENIE_PORT         bind port   (default 8080)
  GENIE_MODEL_ID     model id reported to clients (default qwen3-4b-npu)
  GENIE_MAX_TOKENS   default max generated tokens if request omits it (default 512)
  GENIE_STRIP_THINK  "1" strips <think>...</think> from content (default 0 = faithful)
  GENIE_THINKING     "1" re-enables Qwen3's reasoning block (default 0 = suppressed).
                     Per request: chat_template_kwargs.enable_thinking,
                     reasoning_effort:"none", or thinking:{"type":"disabled"}
"""

import ctypes as C
import json
import os
import queue
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
# The bundle (context binaries + config + tokenizer) and the QAIRT 2.45 runtime
# are large external artifacts that do NOT live in this repo. Point these at
# wherever you extracted them (see docs/GENIE_SERVER.md). run-genie-server.ps1
# sets them for you if you edit the paths there.
BUNDLE_DIR = os.environ.get("GENIE_BUNDLE_DIR", "")
SDK_DIR = os.environ.get("GENIE_SDK_DIR", "")
HOST = os.environ.get("GENIE_HOST", "127.0.0.1")
PORT = int(os.environ.get("GENIE_PORT", "8080"))
MODEL_ID = os.environ.get("GENIE_MODEL_ID", "qwen3-4b-npu")
DEFAULT_MAX_TOKENS = int(os.environ.get("GENIE_MAX_TOKENS", "512"))
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
# and a 500.
WINDOW_MARGIN = int(os.environ.get("GENIE_WINDOW_MARGIN", "64"))
# Plain eviction drops the oldest turns outright, so the agent forgets it
# already read a file and reads it again -- burning the window a second time on
# information it had. Summarising the turns on their way out keeps the facts and
# discards only the tokens. Costs one extra NPU call, and ONLY when eviction was
# going to happen anyway (i.e. the alternative was losing the content).
SUMMARIZE_EVICTED = os.environ.get("GENIE_SUMMARIZE_EVICTED", "1") != "0"
SUMMARY_MAX_TOKENS = int(os.environ.get("GENIE_SUMMARY_MAX_TOKENS", "192"))


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
# Marker delimiting the retained note inside the system turn. Load-bearing:
# it is how a LATER eviction finds the previous note and re-summarises it
# together with the newly evicted turns, instead of stacking note after note
# until the notes themselves fill the window.
SUMMARY_MARKER = "[earlier context]"

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
        with open(os.path.join(BUNDLE_DIR, "genie_config.json"),
                  encoding="utf-8") as f:
            cfg = json.load(f)
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
    costs up to 55% of decode on top. It also decides whether running this
    engine beside a GPU one is a 1.45x gain or a 0.78x LOSS, because the OpenCL
    backend needs those same host cores to dispatch a kernel per token.

    Nearly every retracted number in docs/ traces back to this flag being true
    and nobody noticing. Noticing is cheap; the docs are the record of what not
    noticing costs.
    """
    global _POLL_MATCHES
    if _POLL_MATCHES is None:
        try:
            with open(os.path.join(BUNDLE_DIR, "genie_config.json"),
                      encoding="utf-8") as f:
                _POLL_MATCHES = _find_all(json.load(f), "poll")
        except Exception:
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
            with open(os.path.join(BUNDLE_DIR, "metadata.json"),
                      encoding="utf-8") as f:
                v = (json.load(f).get("genie") or {}).get("context_lengths")
            # isinstance, not truthiness: a bare string is iterable, so
            # "8192" yielded [8, 1, 9, 2] -- a bundle reported as multi-length
            # with four invented graph lengths, which is exactly the misreport
            # this field was added to prevent. Anything that is not a list
            # claims nothing.
            _CONTEXT_LENGTHS = [int(x) for x in v] if isinstance(v, list) else []
        except Exception:
            _CONTEXT_LENGTHS = []
    return _CONTEXT_LENGTHS


def bundle_config_warnings():
    """Lines to print about a bundle configured to be slower than it needs to be.

    Both settings below are worth more than anything else this server does, and
    both were previously left to whoever remembered to read the docs. This
    repo's habit everywhere else -- placement, port, Hexagon arch -- is to
    DERIVE the fact from the artifact and say so out loud rather than hope. This
    is that habit applied to the two it had missed.

    Warn, never refuse: a bundle is a large external artifact and a slow server
    is still a working one. Refusing to start would turn a performance note into
    an outage.
    """
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
            "cores burned while IDLE, up to 55%% of decode lost, and NPU+GPU "
            "concurrency turned from a 1.45x gain into a 0.78x loss. Set it to "
            "false in genie_config.json and restart -- nothing measured got "
            "worse." % (where or "QnnHtp.poll", json.dumps(poll)))
    lengths = read_context_lengths()
    if len(lengths) == 1:
        out.append(
            "WARNING: this is a SINGLE-length bundle (genie.context_lengths = "
            "%s). It runs every token against its whole compiled window, "
            "measured 2-3x slower on short prompts than a multi-length bundle "
            "of the SAME window. Re-export with several --context-lengths "
            "(+3.8%% size, zero extra HTP memory)." % lengths)
    return out


LIB_DIR = os.path.join(SDK_DIR, "lib", "aarch64-windows-msvc")


def hexagon_search_path():
    """Skel dirs for every Hexagon this box can ACTUALLY drive.

    A Hexagon is usable here only if the SDK ships BOTH halves:
      * lib/hexagon-vNN/unsigned/                    -- the DSP-side skel
      * lib/aarch64-windows-msvc/QnnHtpVNNStub.dll   -- the Windows-side stub

    This used to be hardcoded to hexagon-v73, which excluded X2 Elite (v81).
    Globbing every skel was the other extreme: QAIRT 2.45 ships skels for
    v66..v81, but Windows stubs for only a subset, because v75 (8 Gen 3) and
    v79 (8 Elite) are Android parts -- skel present, no way to reach it from
    Windows. Offering those would be a promise the box cannot keep.

    Intersecting the two halves is what makes the supported set
    self-maintaining: v73 (X Elite / X Plus) and v81 (X2 Elite) fall out
    today, a future Hexagon falls out the day QAIRT ships both halves for it,
    and nothing here has to be edited.

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
# Genie C API (from include/Genie/GenieDialog.h + GenieCommon.h)
# ---------------------------------------------------------------------------
GENIE_STATUS_SUCCESS = 0
GENIE_STATUS_WARNING_CONTEXT_EXCEEDED = 1  # non-fatal (context full)

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
# ordering below are the template's, not invented.
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
    downstream consumers cannot take that. `_complete` json.dumps it and
    `_anthropic_complete` puts it in tool_use.input, where Anthropic requires an
    object -- so a bare int here produced a block no Anthropic client accepts.

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
    parts = []
    if isinstance(content, list):
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
        """
        if thinking is None:
            thinking = THINKING_DEFAULT
        parts = []
        sys_text = ""
        for m in messages:
            if m.get("role") == "system":
                sys_text = _content_text(m.get("content"))
                break
        if not sys_text and self.default_system:
            sys_text = self.default_system

        if tools:
            body = sys_text + (_NL + _NL if sys_text else "")
            body += _TOOLS_PREAMBLE_HEAD
            for t in tools:
                body += _NL + json.dumps(t)
            body += _TOOLS_PREAMBLE_TAIL
            parts.append(self.sys_pre + body + self.sys_suf)
        elif sys_text:
            parts.append(self.sys_pre + sys_text + self.sys_suf)

        i, n = 0, len(messages)
        while i < n:
            m = messages[i]
            role = m.get("role", "user")
            content = _content_text(m.get("content"))
            if role == "system":
                i += 1                       # already folded into the system turn
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
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    if body:
                        body += _NL
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
FIRST_TOKEN_TIMEOUT_S = float(os.environ.get("GENIE_FIRST_TOKEN_TIMEOUT", "300"))
STALL_TIMEOUT_S = float(os.environ.get("GENIE_STALL_TIMEOUT", "120"))
WEDGE_GRACE_S = float(os.environ.get("GENIE_WEDGE_GRACE", "60"))
FAIL_THRESHOLD = max(1, int(os.environ.get("GENIE_FAIL_THRESHOLD", "3")))
# Exit rather than linger. 75 is EX_TEMPFAIL: "temporary failure, try again",
# which is exactly what a supervisor should read from it.
EXIT_WEDGED = 75
WEDGE_EXIT = os.environ.get("GENIE_WEDGE_EXIT", "1") not in ("0", "false", "no")


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

    def begin(self, now):
        """A generation has the engine lock and is about to call into Genie."""
        with self._lock:
            self.started = now
            self.last_progress = None
            self.tokens = 0
            self.stall_signalled_at = None

    def progress(self, now):
        """A token came back. Called from Genie's callback thread, so it stays
        to a lock and two assignments -- this runs per token."""
        with self._lock:
            self.last_progress = now
            self.tokens += 1

    def end(self, ok, now=None, counted=True):
        """Close out a generation.

        `counted=False` is for the server's OWN calls into the engine -- today
        just summarising evicted turns. Those still get full stall and failure
        supervision, because they run on the same device and can wedge it
        exactly as a client request can. What they are not is traffic anyone
        asked for, so counting them made `generations` on /health report more
        work served than any client ever requested -- a diagnostic that drifts
        from the thing it describes, which is what this endpoint exists to end.
        """
        with self._lock:
            self.started = None
            self.last_progress = None
            self.stall_signalled_at = None
            self.tokens = 0
            if counted:
                self.generations += 1
            self.consecutive_failures = 0 if ok else self.consecutive_failures + 1

    def note_stall_signalled(self, now):
        with self._lock:
            if self.stall_signalled_at is None:
                self.stall_signalled_at = now

    def assess(self, now):
        """(state, detail). One of ok / failing / stalled / wedged.

        `stalled` means an abort is worth trying; `wedged` means it was tried
        and did not take, so the process is the only thing left to replace.
        """
        with self._lock:
            if self.started is None:
                if self.consecutive_failures >= self.fail_threshold:
                    return "failing", (
                        "%d consecutive generation failures; the engine is "
                        "returning errors rather than output"
                        % self.consecutive_failures)
                return "ok", ""
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
            if (self.stall_signalled_at is not None
                    and now - self.stall_signalled_at > self.grace):
                return "wedged", detail + (
                    "; an abort was signalled %.0fs ago and did not take"
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


class GenieEngine:
    """Resident Genie dialog on the HTP. All NPU access serialized by a lock."""

    def __init__(self, lib, dialog, tokenizer=None):
        self.lib = lib
        self.dialog = dialog
        self.tokenizer = tokenizer
        self.lock = threading.Lock()
        # Exact text the dialog's KV currently holds; None means "unknown,
        # re-prefill". Set by _commit, cleared on any failure or abort.
        self._committed = None
        # Set by signal_abort (handler thread) and consumed by _commit (worker
        # thread, under the lock). signal_abort clearing _committed directly is
        # not enough: the worker can finish and re-commit AFTER the clear,
        # re-arming reuse against a generation that was cut short.
        self._aborted = False
        # Assistant turn terminator, needed to reconstruct what the dialog
        # holds after a generation. Filled in from the template at startup.
        self.asst_suffix = ""
        # The bundle's own sampler block, read at startup and used as the
        # restore baseline. The dialog is RESIDENT and shared across requests,
        # so a per-request override that is never undone leaks into the next
        # caller -- one request asking for temp 0 would silently make every
        # later request deterministic.
        self.default_sampler = {}
        self._sampler_dirty = False
        self._stop_dirty = False

    @staticmethod
    def _finish(status):
        if status == GENIE_STATUS_WARNING_CONTEXT_EXCEEDED:
            return "length"
        if status != GENIE_STATUS_SUCCESS:
            raise RuntimeError("GenieDialog_query failed, status=%d" % status)
        return "stop"

    def set_stop_sequences(self, seqs):
        """Apply per-request stop sequences, clearing any previous ones.

        Must be called on EVERY request, not just those that specify `stop` --
        the dialog is resident, so a stop sequence set by one caller would
        otherwise silently truncate the next caller's output.
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
        if st == GENIE_STATUS_SUCCESS:
            self._stop_dirty = bool(seqs)
        return st

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
        self.lib.GenieDialog_reset(self.dialog)
        return prompt, False

    def _commit(self, prompt, generated, ok):
        """Record the exact text now resident in the dialog's KV.

        On ANY failure the resident state is unknown, so drop the record and
        force the next turn to re-prefill. Guessing here would poison every
        subsequent continuation.
        """
        if not ok or self._aborted:
            self._committed = None
            self._aborted = False
            return
        # Exactly what was sent plus exactly what came back. Appending a turn
        # terminator we never sent would claim the KV holds a byte it may not,
        # and every later continuation would resume one token out of step.
        self._committed = prompt + generated

    def query(self, prompt, on_text, max_tokens=None, stop=None, sampler=None,
              commit=True, internal=False):
        """Run one query synchronously (for non-streaming). on_text(str) is
        called per chunk. Returns 'stop' | 'length'. Serialized (NPU is single).

        `internal=True` marks a call this server made for its own purposes
        rather than one a client asked for; it is supervised the same but is
        not counted as served traffic. See EngineHealth.end."""
        with self.lock:
            self._aborted = False       # stale abort must not poison this turn
            self.set_stop_sequences(stop)
            self.apply_sampler(sampler)
            send, reused = self._plan(prompt)
            if reused:
                print("[genie] kv reuse: prefilling %d new chars, not %d"
                      % (len(send), len(prompt)), flush=True)
            if max_tokens:
                self.lib.GenieDialog_setMaxNumTokens(self.dialog, C.c_uint32(max_tokens))

            seen = []

            def _cb(resp, code, _udata):
                if resp:
                    try:
                        t = resp.decode("utf-8", "replace")
                    except Exception:
                        return
                    seen.append(t)
                    HEALTH.progress(time.time())
                    try:
                        on_text(t)
                    except Exception:
                        pass

            cb = QUERY_CALLBACK(_cb)  # keep ref alive for the blocking call
            # begin() only after the lock is held: a request WAITING for the
            # engine is not a stalled one, and counting it as such would let a
            # busy server look wedged.
            HEALTH.begin(time.time())
            status = GENIE_STATUS_SUCCESS
            try:
                status = self.lib.GenieDialog_query(
                    self.dialog, send.encode("utf-8"), SENTENCE_COMPLETE, cb, None)
            finally:
                HEALTH.end(status == GENIE_STATUS_SUCCESS, time.time(),
                           counted=not internal)
            # commit=False for internal calls (summarisation): they leave the
            # dialog holding text that is NOT the caller's conversation, so
            # recording it as the resident prefix would be a false claim. None
            # says "unknown, re-prefill", which is the truth.
            if commit:
                self._commit(prompt, "".join(seen), status == GENIE_STATUS_SUCCESS)
            else:
                self._committed = None
            return self._finish(status)

    def query_stream(self, prompt, result, max_tokens=None, stop=None,
                     sampler=None):
        """Generator: yields text chunks, then sets result['finish'] (and
        result['error'] on failure) when done. The blocking Genie query runs on
        a WORKER thread so the consumer (the request/handler thread) can call
        signal_abort() on client disconnect -- a cross-thread signal, which is
        how Genie's abort is designed to be delivered. This is what actually
        frees the single-flight lock instead of running to max_tokens."""
        q = queue.Queue()

        def worker():
            try:
                with self.lock:
                    self._aborted = False
                    self.set_stop_sequences(stop)
                    self.apply_sampler(sampler)
                    send, reused = self._plan(prompt)
                    if reused:
                        print("[genie] kv reuse: prefilling %d new chars, not %d"
                              % (len(send), len(prompt)), flush=True)
                    if max_tokens:
                        self.lib.GenieDialog_setMaxNumTokens(self.dialog, C.c_uint32(max_tokens))
                    seen = []

                    def _cb(resp, code, _udata):
                        if resp:
                            try:
                                t = resp.decode("utf-8", "replace")
                            except Exception:
                                return
                            seen.append(t)
                            HEALTH.progress(time.time())
                            q.put(("text", t))

                    cb = QUERY_CALLBACK(_cb)
                    HEALTH.begin(time.time())
                    status = GENIE_STATUS_SUCCESS
                    try:
                        status = self.lib.GenieDialog_query(
                            self.dialog, send.encode("utf-8"), SENTENCE_COMPLETE, cb, None)
                    finally:
                        HEALTH.end(status == GENIE_STATUS_SUCCESS, time.time())
                    self._commit(prompt, "".join(seen),
                                 status == GENIE_STATUS_SUCCESS)
                q.put(("done", self._finish(status)))
            except Exception as e:
                self._committed = None      # dialog state unknown after a throw
                q.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        while True:
            kind, val = q.get()
            if kind == "text":
                yield val
            elif kind == "done":
                result["finish"] = val
                return
            else:
                result["finish"] = "stop"
                result["error"] = val
                return

    def signal_abort(self):
        """Ask Genie to abort the in-progress query (client disconnected) so the
        single-flight lock frees without generating to max_tokens."""
        # An aborted generation leaves a partial, unrecorded tail in the KV --
        # continuing from it would resume mid-sentence off a history we never
        # recorded. Force the next turn to re-prefill. The flag is what makes
        # this stick: clearing _committed alone loses the race against a worker
        # that commits after the abort lands.
        self._aborted = True
        self._committed = None
        try:
            self.lib.GenieDialog_signal(self.dialog, GENIE_DIALOG_ACTION_ABORT)
        except Exception:
            pass

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
        stays until someone measures the alternative on hardware."""
        if not self.tokenizer or not text:
            return None
        try:
            with self.lock:
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


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _maybe_strip_think(text):
    return _THINK_RE.sub("", text) if STRIP_THINK else text


def _anthropic_text(content):
    """Anthropic content -> text. Delegates to the shared flattener so the two
    endpoints cannot diverge on block handling."""
    return _content_text(content)


def read_default_sampler():
    """The bundle's own sampler block -- the baseline a per-request override
    is restored to. Read rather than hardcoded, same reasoning as n_ctx: the
    values belong to the bundle, and a literal here goes quietly wrong on the
    next bundle."""
    try:
        with open(os.path.join(BUNDLE_DIR, "genie_config.json"), encoding="utf-8") as f:
            return dict(json.load(f)["dialog"]["sampler"])
    except Exception:
        return {"version": 1}


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


def load_engine():
    """Load Genie.dll, create the dialog from the bundle config (resident)."""
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
here: %s  (v75 / v79 are Android parts and never have one.)
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

    cfg = ConfigHandle()
    st = lib.GenieDialogConfig_createFromJson(cfg_json, C.byref(cfg))
    if st != GENIE_STATUS_SUCCESS:
        sys.exit("GenieDialogConfig_createFromJson failed, status=%d" % st)

    dialog = Handle()
    t0 = time.time()
    print("[genie] loading model on the NPU (this takes ~8-12s)...", flush=True)
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
    tokenizer = tok if lib.GenieDialog_getTokenizer(
        dialog, C.byref(tok)) == GENIE_STATUS_SUCCESS else None
    return GenieEngine(lib, dialog, tokenizer)


def load_chat_template():
    meta_path = os.path.join(BUNDLE_DIR, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tmpl = meta.get("genie", {}).get("chat_template")
        if tmpl:
            return ChatML(tmpl)
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
    """
    for fn in ("added_tokens.json", "tokenizer_config.json"):
        try:
            with open(os.path.join(BUNDLE_DIR, fn), encoding="utf-8") as f:
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
    """
    allow_reuse_address = (os.name != "nt")


def port_in_use(host, port, timeout=0.5):
    """Is something already accepting connections here?

    Checked BEFORE the model loads. The bind itself would catch this on POSIX,
    but only after 30-50s of loading a 3 GB bundle onto the HTP -- and on
    Windows it would not catch it at all.

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
MAX_INFLIGHT = max(1, int(os.environ.get("GENIE_MAX_INFLIGHT", "2")))
_INFLIGHT = threading.BoundedSemaphore(MAX_INFLIGHT)


_TOK_CACHE = {}


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
    """
    if text in _TOK_CACHE:
        return _TOK_CACHE[text]
    n = ENGINE.count_tokens(text) if ENGINE else None
    n = n if n is not None else max(0, len(text) // 4)
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
    """
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    def _drop(k):
        """Drop the k oldest turns, advancing past any now-orphaned tool
        results. Returns (kept, actual_dropped)."""
        k = max(0, min(k, len(rest)))
        while k < len(rest) - 1 and rest[k].get("role") == "tool":
            k += 1
        return rest[k:], k

    def _render(kept):
        return TEMPLATE.build(sys_msgs + kept, tools=tools, thinking=thinking)

    prompt = _render(rest)                      # common case: nothing to evict
    if _tok_count(prompt) <= budget:
        return prompt, rest, [], True
    if len(rest) <= 1:
        # Nothing left to evict but the current turn; the caller owes the
        # client a real error rather than a doomed query.
        return prompt, rest, [], False

    # Bisect for the FEWEST turns to drop. The previous linear scan re-rendered
    # and re-tokenized the entire prompt once per evicted message -- measured
    # at 90 full tokenizer calls on a 122-message conversation, each taking the
    # engine lock, and build_windowed runs this twice when summarising.
    # Dropping more turns can only shrink the prompt, so "fits" is monotonic in
    # k and a bisection reaches the same kept-set in ~log2(n) renders.
    lo, hi, best = 1, len(rest) - 1, None
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
        kept, n = _drop(len(rest) - 1)
        return _render(kept), kept, rest[:n], False
    cand, kept, n = best
    return cand, kept, rest[:n], True


def _transcript(msgs, cap_chars=6000):
    """Flatten turns to a compact transcript for summarisation."""
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
    # most likely to still matter.
    return text[-cap_chars:] if len(text) > cap_chars else text


def _summarize_turns(msgs, prior=""):
    """One cheap NPU call condensing evicted turns (plus any prior note).

    Returns (note, tokens); (None, 0) on any failure -- the caller then falls
    back to plain eviction. A summary is a nice-to-have; never let it break the
    request. The token count is returned so the cost can be surfaced instead of
    being spent invisibly on the caller's behalf.
    """
    if ENGINE is None:
        return None, 0
    body = _transcript(msgs)
    if prior:
        body = prior + _NL + body
    if not body.strip():
        return None, 0
    ask = ("Condense this conversation excerpt into a few terse factual bullet "
           "points. Keep file paths, identifiers, decisions made, and results "
           "already obtained. Drop pleasantries and reasoning."
           + _NL + _NL + body)
    prompt = TEMPLATE.build([{"role": "user", "content": ask}], thinking=False)
    out = []
    try:
        ENGINE.query(prompt, out.append, max_tokens=summary_token_cap(),
                     commit=False, internal=True)
    except Exception:
        return None, 0
    text = _THINK_RE.sub("", "".join(out)).strip()
    if not text:
        return None, 0
    return text, _tok_count(text)


def _apply_note(messages, note):
    """Fold the note into the system turn, REPLACING any previous note.

    It rides in the system turn because that is the one thing eviction never
    touches -- a note stored anywhere else would itself be evicted, which is
    the problem it exists to solve.
    """
    out, placed = [], False
    for m in messages:
        if m.get("role") == "system" and not placed:
            base = _content_text(m.get("content")).split(SUMMARY_MARKER)[0].rstrip()
            joined = base + (_NL + _NL if base else "") + SUMMARY_MARKER + _NL + note
            out.append(dict(m, content=joined))
            placed = True
        else:
            out.append(m)
    if not placed:
        out.insert(0, {"role": "system", "content": SUMMARY_MARKER + _NL + note})
    return out


def _prior_note(messages):
    # Flatten before the membership test: `MARKER in [block, ...]` is a LIST
    # membership check, which quietly returns False instead of raising. The
    # prior note then goes unfound and each eviction appends a fresh one until
    # the notes themselves crowd out the window -- exactly what the marker
    # exists to prevent.
    for m in messages:
        if m.get("role") != "system":
            continue
        c = _content_text(m.get("content"))
        if SUMMARY_MARKER in c:
            return c.split(SUMMARY_MARKER, 1)[1].strip()
    return ""


def build_windowed(messages, tools=None, thinking=None, max_tokens=0,
                   summarize=None):
    """Render a prompt that FITS, summarising what it has to evict.

    Genie has no sliding-window mode -- QAIRT 2.45 exposes no such flag on
    genie-t2t-run and no equivalent config key -- and overflowing the compiled
    window is a hard GenieDialog_query failure, not a truncation. So eviction
    happens here, and (unless disabled) what leaves is condensed rather than
    discarded.

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

    note, overhead = _summarize_turns(evicted, prior=_prior_note(messages))
    if not note:
        return prompt, len(evicted), fits, overhead   # fall back to plain evict

    merged = _apply_note([m for m in messages if m.get("role") == "system"], note) + kept
    # Second pass WITHOUT summarising: the note itself costs tokens and may push
    # the render back over budget. Re-fitting can only drop more turns, and
    # recursing here would summarise the summary on every request.
    p2, _, ev2, fits2 = _fit(merged, tools, thinking, budget)
    if fits2:
        print("[genie] context window: summarised %d evicted message(s) into a "
              "%d-char note (%d tokens of NPU time)"
              % (len(evicted), len(note), overhead), flush=True)
        return p2, len(evicted) + len(ev2), True, overhead
    return prompt, len(evicted), fits, overhead    # note did not fit; plain evict


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        # Disable Nagle so per-token SSE frames flush immediately (no
        # delayed-ACK stalls stacking on top of decode latency).
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, *a):  # quieter
        pass

    def handle_one_request(self):
        # A client that disconnects (typed cancelling a turn, a closed keep-alive)
        # otherwise dumps a ConnectionReset/Aborted traceback from the base handler.
        try:
            super().handle_one_request()
        except (ConnectionError, OSError):
            self.close_connection = True

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            # Superset item: satisfies both OpenAI (id/object) and Anthropic
            # (type/id/display_name) model-list shapes.
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "type": "model",
                 "display_name": MODEL_ID, "owned_by": "qualcomm-genie-npu"}
            ]})
        elif self.path.rstrip("/") == "/props":
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
            # of the same n_ctx differ 2-3x on short prompts, and `poll` decides
            # whether running this engine beside another is a gain or a loss.
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
        elif self.path.rstrip("/") in ("/health", "/healthz"):
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
            snap = HEALTH.snapshot(time.time())
            code = 200 if snap["state"] == "ok" else 503
            self._json(code, dict(snap, status=snap["state"], model=MODEL_ID))
        else:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def _request_error(self, code, msg, path=None,
                       etype="invalid_request_error"):
        """One error, in whichever envelope the TARGETED api uses.

        do_POST used to fork on path for the tools refusal and the load shed but
        not for the two failures above them, so a malformed body sent to
        /v1/messages came back as {"error": {...}} with no top-level "type" --
        the OpenAI shape, which an Anthropic client cannot parse. A client that
        cannot read the error is told nothing at the one moment it needs to be
        told something, so the envelope has to follow the endpoint everywhere,
        not only where it was convenient.
        """
        if (path if path is not None else self.path.rstrip("/")) == "/v1/messages":
            self._anthropic_error(code, etype, msg)
        else:
            self._json(code, {"error": {"message": msg, "type": etype}})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._request_error(400, "bad JSON: %s" % e, path)
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
        finally:
            _INFLIGHT.release()

    def _openai_chat(self, req):
        messages = req.get("messages", [])
        if not messages:
            self._json(400, {"error": {"message": "messages required",
                                       "type": "invalid_request_error"}})
            return
        tools = req.get("tools") or None
        stream = bool(req.get("stream", False))
        max_tokens = int(req.get("max_tokens") or DEFAULT_MAX_TOKENS)
        prompt, dropped, fits, overhead = build_windowed(
            messages, tools=tools, thinking=_wants_thinking(req),
            max_tokens=max_tokens)
        if not fits:
            self._json(400, {"error": {"message": _overflow_msg(prompt, max_tokens),
                                       "type": "invalid_request_error"}})
            return
        if dropped:
            _log_dropped(dropped)
        created = int(time.time())
        cmpl_id = "chatcmpl-%d" % created
        gen_kw = {"stop": _stop_sequences(req),
                  "sampler": _sampler_params(req, tools_active=bool(tools)),
                  "overhead": overhead}
        if stream:
            self._stream(prompt, max_tokens, cmpl_id, created,
                         tools_active=bool(tools),
                         include_usage=bool((req.get("stream_options") or {})
                                            .get("include_usage")),
                         **gen_kw)
        else:
            self._complete(prompt, max_tokens, cmpl_id, created,
                           tools_active=bool(tools), **gen_kw)

    def _complete(self, prompt, max_tokens, cmpl_id, created, tools_active=False,
                  stop=None, sampler=None, overhead=0):
        chunks = []
        try:
            finish = ENGINE.query(prompt, chunks.append, max_tokens=max_tokens,
                                   stop=stop, sampler=sampler)
        except Exception as e:
            self._json(500, {"error": {"message": str(e), "type": "server_error"}})
            return
        raw = _maybe_strip_think("".join(chunks))
        content = raw
        tool_calls = []
        if tools_active:
            content, calls = parse_tool_calls(content)
            for idx, c in enumerate(calls):
                tool_calls.append({
                    "id": "call_%s_%d" % (cmpl_id, idx),
                    "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c["arguments"])},
                })
        message = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        # Count what the MODEL produced, not what survives parsing: the
        # <tool_call> block is real generated output, and billing/budgeting a
        # tool turn as 0 tokens is a lie the client cannot detect.
        pt, ct = _tok_count(prompt), _tok_count(raw)
        usage = {"prompt_tokens": pt, "completion_tokens": ct,
                 "total_tokens": pt + ct}
        if overhead:
            # Namespaced and only present when non-zero, so an ordinary
            # response is byte-identical to before and no standard field is
            # misreported. This is NPU time the request really spent -- on
            # summarising evicted history, not on the answer -- and spending it
            # invisibly is the same failure as the completion_tokens=0 bug.
            usage["genie_context_overhead_tokens"] = overhead
        self._json(200, {
            "id": cmpl_id, "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        })

    def _stream(self, prompt, max_tokens, cmpl_id, created, tools_active=False,
                stop=None, sampler=None, overhead=0, include_usage=False):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length/chunked on an SSE body, so frame the response by
        # connection close -- otherwise read-to-EOF clients hang on keep-alive.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        gone = {"v": False}

        def sse(obj):
            if gone["v"]:
                return
            try:
                self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except (ConnectionError, OSError):
                gone["v"] = True
                ENGINE.signal_abort()

        def frame(delta, finish=None):
            return {"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        # NOTE: with STRIP_THINK we cannot cleanly strip mid-stream, so streamed
        # output is always faithful (includes <think>); non-stream honors the flag.
        sse(frame({"role": "assistant"}))

        def done(text=""):
            """Optional usage frame, then [DONE].

            Non-streaming responses carry usage (including the summarisation
            overhead); streams carried none at all, so a streaming client could
            not see token counts by any means. OpenAI's shape for this is a
            final chunk with an EMPTY choices list, emitted only when the
            caller asked via stream_options.include_usage -- so clients that
            did not ask see a byte-identical stream to before.
            """
            if include_usage and not gone["v"]:
                pt, ct = _tok_count(prompt), _tok_count(text)
                usage = {"prompt_tokens": pt, "completion_tokens": ct,
                         "total_tokens": pt + ct}
                if overhead:
                    usage["genie_context_overhead_tokens"] = overhead
                sse({"id": cmpl_id, "object": "chat.completion.chunk",
                     "created": created, "model": MODEL_ID,
                     "choices": [], "usage": usage})
            if not gone["v"]:
                try:
                    self.wfile.write(("data: [DONE]" + chr(10) * 2).encode("utf-8"))
                    self.wfile.flush()
                except (ConnectionError, OSError):
                    pass

        if tools_active:
            # A <tool_call> block means nothing until it CLOSES -- streaming it
            # token-by-token would hand the client half a call to guess about.
            # So generate fully, then emit well-formed frames: still SSE (the
            # client asked for SSE), just not incremental. Buffering is the
            # honest trade; a partial tool call is not.
            def probe():
                """An SSE comment: ignored by every client, but a FAILED write
                is the only way to learn the caller is gone while we are
                buffering and therefore emitting nothing. Without it a client
                that walks away from a tool turn leaves the generation running
                to max_tokens, holding the single-flight NPU against everyone
                else -- the exact hazard signal_abort exists to prevent on the
                plain path."""
                if gone["v"]:
                    return
                try:
                    self.wfile.write(b": keep-alive" + _SSE_GAP)
                    self.wfile.flush()
                except (ConnectionError, OSError):
                    gone["v"] = True
                    ENGINE.signal_abort()

            # query_stream (not query) so the generation runs on a worker
            # thread and signal_abort can actually reach it.
            buf, res = [], {}
            for i, chunk in enumerate(ENGINE.query_stream(
                    prompt, res, max_tokens=max_tokens, stop=stop,
                    sampler=sampler)):
                buf.append(chunk)
                if i % 8 == 0:
                    probe()
                if gone["v"]:
                    break
            if res.get("error"):
                sse(frame({"content": "[error: %s]" % res["error"]}, finish="stop"))
                done(_maybe_strip_think("".join(buf)))
                return
            finish = res.get("finish", "stop")
            text, calls = parse_tool_calls(_maybe_strip_think("".join(buf)))
            if text:
                sse(frame({"content": text}))
            for idx, c in enumerate(calls):
                sse(frame({"tool_calls": [{
                    "index": idx,
                    "id": "call_%s_%d" % (cmpl_id, idx),
                    "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c["arguments"])}}]}))
            sse(frame({}, finish="tool_calls" if calls else finish))
            # Strip <think> before counting, exactly as _complete does: the same
            # turn must not report different completion_tokens purely because
            # the client chose to stream.
            done(_maybe_strip_think("".join(buf)))
            return

        res = {}
        seen = []
        for chunk in ENGINE.query_stream(prompt, res, max_tokens=max_tokens,
                                        stop=stop, sampler=sampler):
            seen.append(chunk)
            sse(frame({"content": chunk}))
            if gone["v"]:
                break
        if res.get("error") and not gone["v"]:
            sse(frame({"content": "\n[error: %s]" % res["error"]}))
        sse(frame({}, finish=res.get("finish", "stop")))
        done("".join(seen))

    # ---- Anthropic Messages API (POST /v1/messages) -----------------------

    def _anthropic_error(self, code, etype, msg):
        self._json(code, {"type": "error", "error": {"type": etype, "message": msg}})

    def _anthropic_messages(self, req):
        if not req.get("messages"):
            self._anthropic_error(400, "invalid_request_error", "messages required")
            return
        model = req.get("model") or MODEL_ID
        max_tokens = int(req.get("max_tokens") or DEFAULT_MAX_TOKENS)
        tools = req.get("tools") or None
        prompt, dropped, fits, overhead = _anthropic_to_prompt(
            req, tools=tools, max_tokens=max_tokens)
        if not fits:
            self._anthropic_error(400, "invalid_request_error",
                                  _overflow_msg(prompt, max_tokens))
            return
        if dropped:
            _log_dropped(dropped)
        msg_id = "msg_%d" % int(time.time())
        gen_kw = {"stop": _stop_sequences(req),
                  "sampler": _sampler_params(req, tools_active=bool(tools)),
                  "overhead": overhead}
        if bool(req.get("stream", False)):
            self._anthropic_stream(prompt, max_tokens, model, msg_id,
                                   tools_active=bool(tools), **gen_kw)
        else:
            self._anthropic_complete(prompt, max_tokens, model, msg_id,
                                     tools_active=bool(tools), **gen_kw)

    def _anthropic_complete(self, prompt, max_tokens, model, msg_id,
                            tools_active=False, stop=None, sampler=None,
                            overhead=0):
        chunks = []
        try:
            finish = ENGINE.query(prompt, chunks.append, max_tokens=max_tokens,
                                   stop=stop, sampler=sampler)
        except Exception as e:
            self._anthropic_error(500, "api_error", str(e))
            return
        raw = _maybe_strip_think("".join(chunks))
        content = raw
        calls = []
        if tools_active:
            content, calls = parse_tool_calls(content)
        blocks = []
        if content:
            blocks.append({"type": "text", "text": content})
        for idx, c in enumerate(calls):
            blocks.append({"type": "tool_use",
                           "id": "toolu_%s_%d" % (msg_id, idx),
                           "name": c["name"], "input": c["arguments"]})
        reason = _anthropic_stop_reason(finish, calls, stop)
        self._json(200, {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": blocks,
            "stop_reason": reason,
            "stop_sequence": None,
            "usage": _with_overhead({"input_tokens": _tok_count(prompt),
                                     "output_tokens": _tok_count(raw)},
                                    overhead),
        })

    def _anthropic_stream(self, prompt, max_tokens, model, msg_id,
                          tools_active=False, stop=None, sampler=None,
                          overhead=0):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Frame the SSE body by connection close (no Content-Length/chunked).
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        gone = {"v": False}

        def ev(etype, obj):
            if gone["v"]:
                return
            try:
                self.wfile.write(("event: %s\n" % etype).encode("utf-8"))
                self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except (ConnectionError, OSError):
                gone["v"] = True          # client disconnected mid-stream
                ENGINE.signal_abort()     # cross-thread abort -> free the NPU lock

        ev("message_start", {"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": _tok_count(prompt), "output_tokens": 0}}})
        if tools_active:
            # Same reasoning as the OpenAI stream: a tool_use block only means
            # something once complete, so buffer the generation and emit whole
            # blocks rather than a half-formed call the client must guess at.
            # query_stream (not query) so signal_abort can reach the
            # generation, and a periodic ping so a client that walks away is
            # actually noticed -- while buffering we emit nothing, so without a
            # probe the write callback never fires and the abandoned turn holds
            # the single-flight NPU to max_tokens. `ping` is a real Anthropic
            # event, so this needs no client-side tolerance.
            buf, res = [], {}
            for i, chunk in enumerate(ENGINE.query_stream(
                    prompt, res, max_tokens=max_tokens, stop=stop,
                    sampler=sampler)):
                buf.append(chunk)
                if i % 8 == 0:
                    ev("ping", {"type": "ping"})
                if gone["v"]:
                    break
            if res.get("error"):
                ev("error", {"type": "error",
                             "error": {"type": "api_error",
                                       "message": res["error"]}})
                ev("message_stop", {"type": "message_stop"})
                return
            finish = res.get("finish", "stop")
            text, calls = parse_tool_calls(_maybe_strip_think("".join(buf)))
            idx = 0
            if text:
                ev("content_block_start", {"type": "content_block_start", "index": idx,
                    "content_block": {"type": "text", "text": ""}})
                ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                    "delta": {"type": "text_delta", "text": text}})
                ev("content_block_stop", {"type": "content_block_stop", "index": idx})
                idx += 1
            for n, c in enumerate(calls):
                ev("content_block_start", {"type": "content_block_start", "index": idx,
                    "content_block": {"type": "tool_use",
                                      "id": "toolu_%s_%d" % (msg_id, n),
                                      "name": c["name"], "input": {}}})
                ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                    "delta": {"type": "input_json_delta",
                              "partial_json": json.dumps(c["arguments"])}})
                ev("content_block_stop", {"type": "content_block_stop", "index": idx})
                idx += 1
            reason = _anthropic_stop_reason(finish, calls, stop)
            ev("message_delta", {"type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": None},
                "usage": _with_overhead({"output_tokens": _tok_count(
                    _maybe_strip_think("".join(buf)))}, overhead)})
            ev("message_stop", {"type": "message_stop"})
            return

        ev("content_block_start", {"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}})
        ev("ping", {"type": "ping"})

        out = []
        res = {}
        for chunk in ENGINE.query_stream(prompt, res, max_tokens=max_tokens,
                                        stop=stop, sampler=sampler):
            out.append(chunk)
            ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": chunk}})
            if gone["v"]:
                break
        if res.get("error") and not gone["v"]:
            ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "\n[error: %s]" % res["error"]}})
        finish = res.get("finish", "stop")
        ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        ev("message_delta", {"type": "message_delta",
            "delta": {"stop_reason": _anthropic_stop_reason(finish, None, stop),
                      "stop_sequence": None},
            "usage": _with_overhead({"output_tokens": _tok_count("".join(out))},
                                    overhead)})
        ev("message_stop", {"type": "message_stop"})


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
            print("[genie] STALL: %s -- signalling abort" % detail, flush=True)
            health.note_stall_signalled(now)
            try:
                engine.signal_abort()
            except Exception as e:
                print("[genie] abort signal failed: %s" % e, flush=True)
            continue
        # wedged
        if announced != "wedged":
            print("[genie] WEDGED: %s" % detail, flush=True)
            print("[genie] The engine cannot be recovered in this process: the "
                  "stuck call is inside the Genie driver, holding the engine "
                  "lock, and Python cannot reclaim a thread blocked in native "
                  "code. Exiting %d so a supervisor restarts a clean process. "
                  "(GENIE_WEDGE_EXIT=0 to stay up and keep reporting 503.)"
                  % EXIT_WEDGED, flush=True)
            announced = "wedged"
        result = on_wedge(detail)
        # Only reached when the exit was declined (GENIE_WEDGE_EXIT=0, or a
        # test's stand-in). Keep watching and keep reporting 503 rather than
        # spinning silently on a dead device.
        if result is not None:
            return result
    return None


def main():
    global ENGINE, TEMPLATE, TOOLS_OK
    # Before the model load, not after: loading is 30-50s of work, and finding
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
    # and finding out after 30-50s of loading that the bundle is configured to
    # run at half speed wastes all of it.
    for line in bundle_config_warnings():
        print("[genie] %s" % line, flush=True)
    TEMPLATE = load_chat_template()
    TOOLS_OK = probe_tool_support()
    ENGINE = load_engine()
    ENGINE.asst_suffix = TEMPLATE.asst_suf
    ENGINE.default_sampler = read_default_sampler()
    srv = Server((HOST, PORT), Handler)
    print("[genie] endpoint on http://%s:%d  (model=%s)" % (HOST, PORT, MODEL_ID), flush=True)
    print("[genie]   POST /v1/chat/completions (OpenAI)   POST /v1/messages (Anthropic)",
          flush=True)
    print("[genie]   GET /v1/models   GET /health", flush=True)
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
        try:
            ENGINE.lib.GenieDialog_free(ENGINE.dialog)
        except Exception:
            pass


if __name__ == "__main__":
    main()
