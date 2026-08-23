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
  GENIE_THINKING     "0" suppresses Qwen3's reasoning block entirely (default 1 = on).
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
# 300 output tokens thinking -- 37s for one agent step at 13 t/s. The bundle's
# own template supports suppressing it by PREFILLING an empty think block, so
# expose that as a knob. Default stays ON (faithful to the model); agent
# clients that want the latency back turn it off per request or per server.
THINKING_DEFAULT = os.environ.get("GENIE_THINKING", "1") != "0"
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
# Marker delimiting the retained note inside the system turn. Load-bearing:
# it is how a LATER eviction finds the previous note and re-summarises it
# together with the newly evicted turns, instead of stacking note after note
# until the notes themselves fill the window.
SUMMARY_MARKER = "[earlier context]"

_CONTEXT_SIZE = None


def read_context_size(default=4096):
    """The context length this bundle was COMPILED with, from genie_config.json.

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
        with open(os.path.join(BUNDLE_DIR, "genie_config.json"), "r",
                  encoding="utf-8") as f:
            cfg = json.load(f)
        size = int(cfg["dialog"]["context"]["size"])
        _CONTEXT_SIZE = size if size > 0 else default
    except Exception:
        _CONTEXT_SIZE = default
    return _CONTEXT_SIZE


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
        args = o["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
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
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
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

    def build(self, messages, tools=None, thinking=True):
        """Assemble a ChatML prompt ending with an open assistant turn.

        `tools` renders Qwen3's tool preamble into the system turn; assistant
        `tool_calls` and role="tool" results round-trip in the same shapes the
        bundle's own Jinja template uses, so a multi-turn tool conversation
        replays exactly as the model was trained to see it.
        """
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

    def query(self, prompt, on_text, max_tokens=None, stop=None, sampler=None):
        """Run one query synchronously (for non-streaming). on_text(str) is
        called per chunk. Returns 'stop' | 'length'. Serialized (NPU is single)."""
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
                    try:
                        on_text(t)
                    except Exception:
                        pass

            cb = QUERY_CALLBACK(_cb)  # keep ref alive for the blocking call
            status = self.lib.GenieDialog_query(
                self.dialog, send.encode("utf-8"), SENTENCE_COMPLETE, cb, None)
            self._commit(prompt, "".join(seen), status == GENIE_STATUS_SUCCESS)
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
                            q.put(("text", t))

                    cb = QUERY_CALLBACK(_cb)
                    status = self.lib.GenieDialog_query(
                        self.dialog, send.encode("utf-8"), SENTENCE_COMPLETE, cb, None)
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
        (callers fall back to an estimate). Serialized with generation."""
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
            def _of(kind):
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
                          max_tokens=max_tokens)


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
        with open(meta_path, "r", encoding="utf-8") as f:
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
            with open(os.path.join(BUNDLE_DIR, fn), "r", encoding="utf-8") as f:
                if "<tool_call>" in f.read():
                    return True
        except Exception:
            continue
    return False


ENGINE = None
TEMPLATE = None
TOOLS_OK = False

# Bound the number of in-flight generation requests (1 running on the NPU + a
# small queue). Excess requests are rejected fast instead of piling up parked
# threads behind the single-flight lock. 0 disables the cap.
MAX_INFLIGHT = int(os.environ.get("GENIE_MAX_INFLIGHT", "2"))
_INFLIGHT = threading.BoundedSemaphore(MAX_INFLIGHT) if MAX_INFLIGHT > 0 else None


def _tok_count(text):
    """Exact token count if the Genie tokenizer is available, else a ~4-char
    estimate. Never raises."""
    n = ENGINE.count_tokens(text) if ENGINE else None
    return n if n is not None else max(0, len(text) // 4)


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

    Returns None on any failure -- the caller then falls back to plain
    eviction. A summary is a nice-to-have; never let it break the request.
    """
    if ENGINE is None:
        return None
    body = _transcript(msgs)
    if prior:
        body = prior + _NL + body
    if not body.strip():
        return None
    ask = ("Condense this conversation excerpt into a few terse factual bullet "
           "points. Keep file paths, identifiers, decisions made, and results "
           "already obtained. Drop pleasantries and reasoning."
           + _NL + _NL + body)
    prompt = TEMPLATE.build([{"role": "user", "content": ask}], thinking=False)
    out = []
    try:
        ENGINE.query(prompt, out.append, max_tokens=SUMMARY_MAX_TOKENS)
    except Exception:
        return None
    text = _THINK_RE.sub("", "".join(out)).strip()
    return text or None


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


def build_windowed(messages, tools=None, thinking=True, max_tokens=0,
                   summarize=None):
    """Render a prompt that FITS, summarising what it has to evict.

    Genie has no sliding-window mode -- QAIRT 2.45 exposes no such flag on
    genie-t2t-run and no equivalent config key -- and overflowing the compiled
    window is a hard GenieDialog_query failure, not a truncation. So eviction
    happens here, and (unless disabled) what leaves is condensed rather than
    discarded.

    Returns (prompt, dropped, fits).
    """
    if summarize is None:
        summarize = SUMMARIZE_EVICTED
    budget = read_context_size() - max(0, max_tokens) - WINDOW_MARGIN

    prompt, kept, evicted, fits = _fit(messages, tools, thinking, budget)
    if not (fits and evicted and summarize):
        return prompt, len(evicted), fits

    note = _summarize_turns(evicted, prior=_prior_note(messages))
    if not note:
        return prompt, len(evicted), fits          # fall back to plain eviction

    merged = _apply_note([m for m in messages if m.get("role") == "system"], note) + kept
    # Second pass WITHOUT summarising: the note itself costs tokens and may push
    # the render back over budget. Re-fitting can only drop more turns, and
    # recursing here would summarise the summary on every request.
    p2, _, ev2, fits2 = _fit(merged, tools, thinking, budget)
    if fits2:
        print("[genie] context window: summarised %d evicted message(s) into a "
              "%d-char note" % (len(evicted), len(note)), flush=True)
        return p2, len(evicted) + len(ev2), True
    return prompt, len(evicted), fits              # note did not fit; plain evict


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
            # Only the two fields typed actually reads are emitted:
            #   default_generation_settings.n_ctx -- the window
            #   model_alias / model_id            -- the served model's name
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
            self._json(200, {
                "default_generation_settings": {"n_ctx": read_context_size()},
                "model_alias": MODEL_ID,
                "model_id": MODEL_ID,
            })
        elif self.path.rstrip("/") in ("/health", "/healthz"):
            self._json(200, {"status": "ok", "model": MODEL_ID})
        else:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._json(400, {"error": {"message": "bad JSON: %s" % e,
                                       "type": "invalid_request_error"}})
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
            if path == "/v1/messages":
                self._anthropic_error(400, "invalid_request_error", msg)
            else:
                self._json(400, {"error": {"message": msg,
                                           "type": "invalid_request_error"}})
            return
        if _INFLIGHT is not None and not _INFLIGHT.acquire(blocking=False):
            # NPU is single-flight and the small queue is full -> shed load.
            if path == "/v1/messages":
                self._anthropic_error(529, "overloaded_error", "server busy; NPU is single-flight")
            else:
                self._json(429, {"error": {"message": "server busy; NPU is single-flight",
                                           "type": "overloaded_error"}})
            return
        try:
            gen(req)
        finally:
            if _INFLIGHT is not None:
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
        prompt, dropped, fits = build_windowed(
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
                  "sampler": _sampler_params(req, tools_active=bool(tools))}
        if stream:
            self._stream(prompt, max_tokens, cmpl_id, created,
                         tools_active=bool(tools), **gen_kw)
        else:
            self._complete(prompt, max_tokens, cmpl_id, created,
                           tools_active=bool(tools), **gen_kw)

    def _complete(self, prompt, max_tokens, cmpl_id, created, tools_active=False,
                  stop=None, sampler=None):
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
        self._json(200, {
            "id": cmpl_id, "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct},
        })

    def _stream(self, prompt, max_tokens, cmpl_id, created, tools_active=False,
                stop=None, sampler=None):
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

        def done():
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
            buf = []
            try:
                finish = ENGINE.query(prompt, buf.append, max_tokens=max_tokens,
                                   stop=stop, sampler=sampler)
            except Exception as e:
                sse(frame({"content": "[error: %s]" % e}, finish="stop"))
                done()
                return
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
            done()
            return

        res = {}
        for chunk in ENGINE.query_stream(prompt, res, max_tokens=max_tokens,
                                        stop=stop, sampler=sampler):
            sse(frame({"content": chunk}))
            if gone["v"]:
                break
        if res.get("error") and not gone["v"]:
            sse(frame({"content": "\n[error: %s]" % res["error"]}))
        sse(frame({}, finish=res.get("finish", "stop")))
        done()

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
        prompt, dropped, fits = _anthropic_to_prompt(req, tools=tools,
                                                     max_tokens=max_tokens)
        if not fits:
            self._anthropic_error(400, "invalid_request_error",
                                  _overflow_msg(prompt, max_tokens))
            return
        if dropped:
            _log_dropped(dropped)
        msg_id = "msg_%d" % int(time.time())
        gen_kw = {"stop": _stop_sequences(req),
                  "sampler": _sampler_params(req, tools_active=bool(tools))}
        if bool(req.get("stream", False)):
            self._anthropic_stream(prompt, max_tokens, model, msg_id,
                                   tools_active=bool(tools), **gen_kw)
        else:
            self._anthropic_complete(prompt, max_tokens, model, msg_id,
                                     tools_active=bool(tools), **gen_kw)

    def _anthropic_complete(self, prompt, max_tokens, model, msg_id,
                            tools_active=False, stop=None, sampler=None):
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
            "usage": {"input_tokens": _tok_count(prompt),
                      "output_tokens": _tok_count(raw)},
        })

    def _anthropic_stream(self, prompt, max_tokens, model, msg_id,
                          tools_active=False, stop=None, sampler=None):
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
            buf = []
            try:
                finish = ENGINE.query(prompt, buf.append, max_tokens=max_tokens,
                                   stop=stop, sampler=sampler)
            except Exception as e:
                ev("error", {"type": "error",
                             "error": {"type": "api_error", "message": str(e)}})
                ev("message_stop", {"type": "message_stop"})
                return
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
                "usage": {"output_tokens": _tok_count(
                    _maybe_strip_think("".join(buf)))}})
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
            "usage": {"output_tokens": _tok_count("".join(out))}})
        ev("message_stop", {"type": "message_stop"})


def main():
    global ENGINE, TEMPLATE, TOOLS_OK
    TEMPLATE = load_chat_template()
    TOOLS_OK = probe_tool_support()
    ENGINE = load_engine()
    ENGINE.asst_suffix = TEMPLATE.asst_suf
    ENGINE.default_sampler = read_default_sampler()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print("[genie] endpoint on http://%s:%d  (model=%s)" % (HOST, PORT, MODEL_ID), flush=True)
    print("[genie]   POST /v1/chat/completions (OpenAI)   POST /v1/messages (Anthropic)",
          flush=True)
    print("[genie]   GET /v1/models   GET /health", flush=True)
    print("[genie]   sampling: server-level only (dialog.sampler in "
          "genie_config.json). Per-request temperature/top_p are accepted but "
          "NOT honoured -- QAIRT 2.45 ignores a post-create sampler apply.",
          flush=True)
    print("[genie]   tool calling: %s" %
          ("enabled (<tool_call> in bundle vocab)" if TOOLS_OK
           else "unsupported by this bundle -- requests with `tools` get a 400"),
          flush=True)
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
