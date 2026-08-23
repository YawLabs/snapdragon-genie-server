#!/usr/bin/env python3
"""
OpenAI-compatible HTTP server for a Qualcomm Genie NPU LLM bundle
(Snapdragon X Elite / Hexagon v73).

Loads the Genie context-binary bundle ONCE via the Genie C API (ctypes ->
Genie.dll) so the model stays resident on the HTP; every /v1/chat/completions
request reuses it (no ~8.5s per-request reload that genie-t2t-run.exe would pay).

Pure Python stdlib -- no pip dependencies. Must run on a native ARM64 (aarch64)
Python, because Genie.dll and its Qnn* deps are aarch64-windows-msvc.

Config via environment (all have sensible defaults for this repo's scratchpad):
  GENIE_BUNDLE_DIR   dir with genie_config.json + part*_of_*.bin + tokenizer.json
  GENIE_SDK_DIR      QAIRT 2.45 SDK root (contains lib/aarch64-windows-msvc + lib/hexagon-v73)
  GENIE_HOST         bind host   (default 127.0.0.1)
  GENIE_PORT         bind port   (default 8080)
  GENIE_MODEL_ID     model id reported to clients (default qwen3-4b-npu)
  GENIE_MAX_TOKENS   default max generated tokens if request omits it (default 512)
  GENIE_STRIP_THINK  "1" strips <think>...</think> from content (default 0 = faithful)
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
    try:
        with open(os.path.join(BUNDLE_DIR, "genie_config.json"), "r",
                  encoding="utf-8") as f:
            cfg = json.load(f)
        size = cfg["dialog"]["context"]["size"]
        return int(size) if int(size) > 0 else default
    except Exception:
        return default


LIB_DIR = os.path.join(SDK_DIR, "lib", "aarch64-windows-msvc")
HEXAGON_DIR = os.path.join(SDK_DIR, "lib", "hexagon-v73", "unsigned")

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

    def build(self, messages):
        """Assemble a ChatML prompt ending with an open assistant turn."""
        parts = []
        have_system = any(m.get("role") == "system" for m in messages)
        if not have_system and self.default_system:
            parts.append(self.sys_pre + self.default_system + self.sys_suf)
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            if role == "system":
                parts.append(self.sys_pre + content + self.sys_suf)
            elif role == "assistant":
                parts.append(self.asst_pre + content + self.asst_suf)
            else:  # user (and any tool/other role folded to user)
                parts.append(self.usr_pre + content + self.usr_suf)
        parts.append(self.asst_pre)  # open assistant turn for generation
        return "".join(parts)


class GenieEngine:
    """Resident Genie dialog on the HTP. All NPU access serialized by a lock."""

    def __init__(self, lib, dialog, tokenizer=None):
        self.lib = lib
        self.dialog = dialog
        self.tokenizer = tokenizer
        self.lock = threading.Lock()

    @staticmethod
    def _finish(status):
        if status == GENIE_STATUS_WARNING_CONTEXT_EXCEEDED:
            return "length"
        if status != GENIE_STATUS_SUCCESS:
            raise RuntimeError("GenieDialog_query failed, status=%d" % status)
        return "stop"

    def query(self, prompt, on_text, max_tokens=None):
        """Run one query synchronously (for non-streaming). on_text(str) is
        called per chunk. Returns 'stop' | 'length'. Serialized (NPU is single)."""
        with self.lock:
            self.lib.GenieDialog_reset(self.dialog)
            if max_tokens:
                self.lib.GenieDialog_setMaxNumTokens(self.dialog, C.c_uint32(max_tokens))

            def _cb(resp, code, _udata):
                if resp:
                    try:
                        on_text(resp.decode("utf-8", "replace"))
                    except Exception:
                        pass

            cb = QUERY_CALLBACK(_cb)  # keep ref alive for the blocking call
            status = self.lib.GenieDialog_query(
                self.dialog, prompt.encode("utf-8"), SENTENCE_COMPLETE, cb, None)
            return self._finish(status)

    def query_stream(self, prompt, result, max_tokens=None):
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
                    self.lib.GenieDialog_reset(self.dialog)
                    if max_tokens:
                        self.lib.GenieDialog_setMaxNumTokens(self.dialog, C.c_uint32(max_tokens))

                    def _cb(resp, code, _udata):
                        if resp:
                            try:
                                q.put(("text", resp.decode("utf-8", "replace")))
                            except Exception:
                                pass

                    cb = QUERY_CALLBACK(_cb)
                    status = self.lib.GenieDialog_query(
                        self.dialog, prompt.encode("utf-8"), SENTENCE_COMPLETE, cb, None)
                q.put(("done", self._finish(status)))
            except Exception as e:
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
    """Flatten an Anthropic content value (str, or a list of content blocks)
    down to plain text for this text-only model. Text and tool_result blocks
    contribute text; images / tool_use are ignored."""
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
                    parts.append(_anthropic_text(b.get("content")))
    return "".join(parts)


def _anthropic_to_prompt(req):
    """Build the ChatML prompt from an Anthropic Messages request. Anthropic
    keeps `system` as a top-level field, so fold it in as a system message."""
    msgs = []
    sysval = req.get("system")
    if sysval:
        msgs.append({"role": "system", "content": _anthropic_text(sysval)})
    for m in req.get("messages", []):
        msgs.append({"role": m.get("role", "user"),
                     "content": _anthropic_text(m.get("content"))})
    return TEMPLATE.build(msgs)


def load_engine():
    """Load Genie.dll, create the dialog from the bundle config (resident)."""
    if not BUNDLE_DIR or not SDK_DIR:
        sys.exit("set GENIE_BUNDLE_DIR (the Genie bundle dir) and GENIE_SDK_DIR "
                 "(the QAIRT 2.45 root) -- see docs/GENIE_SERVER.md")
    if not os.path.isdir(BUNDLE_DIR):
        sys.exit("bundle dir not found: %s" % BUNDLE_DIR)
    if not os.path.isdir(LIB_DIR):
        sys.exit("SDK lib dir not found: %s (check GENIE_SDK_DIR)" % LIB_DIR)

    os.environ["ADSP_LIBRARY_PATH"] = HEXAGON_DIR
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
        sys.exit("GenieDialog_create failed, status=%d" % st)
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


ENGINE = None
TEMPLATE = None

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
        # REFUSE a tools payload rather than dropping it.
        #
        # This bundle is a text-only 4B build: it cannot emit tool_use /
        # tool_calls blocks. Accepting `tools` and answering normally -- which
        # is what this server did before -- looks like success to every client,
        # so the caller registers its tool set and then watches every tool call
        # silently not happen. There is no error to find and nothing on screen
        # says why.
        #
        # A 400 naming the limitation converts that into something a client can
        # act on. typed probes exactly this at startup (probeLocalToolCalls),
        # and reads a 4xx as "tools unsupported" -> it disables them for the
        # session and says so, instead of shipping schemas the model ignores.
        #
        # Checked BEFORE the single-flight lock: refusing costs no NPU time, so
        # it must not queue behind a live generation.
        if req.get("tools"):
            msg = ("tool calling is not supported: %s is a text-only build and "
                   "cannot emit tool_use blocks. Retry without `tools`." % MODEL_ID)
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
        stream = bool(req.get("stream", False))
        max_tokens = int(req.get("max_tokens") or DEFAULT_MAX_TOKENS)
        prompt = TEMPLATE.build(messages)
        created = int(time.time())
        cmpl_id = "chatcmpl-%d" % created
        if stream:
            self._stream(prompt, max_tokens, cmpl_id, created)
        else:
            self._complete(prompt, max_tokens, cmpl_id, created)

    def _complete(self, prompt, max_tokens, cmpl_id, created):
        chunks = []
        try:
            finish = ENGINE.query(prompt, chunks.append, max_tokens=max_tokens)
        except Exception as e:
            self._json(500, {"error": {"message": str(e), "type": "server_error"}})
            return
        content = _maybe_strip_think("".join(chunks))
        pt, ct = _tok_count(prompt), _tok_count(content)
        self._json(200, {
            "id": cmpl_id, "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct},
        })

    def _stream(self, prompt, max_tokens, cmpl_id, created):
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
        res = {}
        for chunk in ENGINE.query_stream(prompt, res, max_tokens=max_tokens):
            sse(frame({"content": chunk}))
            if gone["v"]:
                break
        if res.get("error") and not gone["v"]:
            sse(frame({"content": "\n[error: %s]" % res["error"]}))
        sse(frame({}, finish=res.get("finish", "stop")))
        if not gone["v"]:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (ConnectionError, OSError):
                pass

    # ---- Anthropic Messages API (POST /v1/messages) -----------------------

    def _anthropic_error(self, code, etype, msg):
        self._json(code, {"type": "error", "error": {"type": etype, "message": msg}})

    def _anthropic_messages(self, req):
        if not req.get("messages"):
            self._anthropic_error(400, "invalid_request_error", "messages required")
            return
        model = req.get("model") or MODEL_ID
        max_tokens = int(req.get("max_tokens") or DEFAULT_MAX_TOKENS)
        prompt = _anthropic_to_prompt(req)
        msg_id = "msg_%d" % int(time.time())
        if bool(req.get("stream", False)):
            self._anthropic_stream(prompt, max_tokens, model, msg_id)
        else:
            self._anthropic_complete(prompt, max_tokens, model, msg_id)

    def _anthropic_complete(self, prompt, max_tokens, model, msg_id):
        chunks = []
        try:
            finish = ENGINE.query(prompt, chunks.append, max_tokens=max_tokens)
        except Exception as e:
            self._anthropic_error(500, "api_error", str(e))
            return
        content = _maybe_strip_think("".join(chunks))
        self._json(200, {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": "max_tokens" if finish == "length" else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": _tok_count(prompt),
                      "output_tokens": _tok_count(content)},
        })

    def _anthropic_stream(self, prompt, max_tokens, model, msg_id):
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
        ev("content_block_start", {"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}})
        ev("ping", {"type": "ping"})

        out = []
        res = {}
        for chunk in ENGINE.query_stream(prompt, res, max_tokens=max_tokens):
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
            "delta": {"stop_reason": "max_tokens" if finish == "length" else "end_turn",
                      "stop_sequence": None},
            "usage": {"output_tokens": _tok_count("".join(out))}})
        ev("message_stop", {"type": "message_stop"})


def main():
    global ENGINE, TEMPLATE
    TEMPLATE = load_chat_template()
    ENGINE = load_engine()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print("[genie] endpoint on http://%s:%d  (model=%s)" % (HOST, PORT, MODEL_ID), flush=True)
    print("[genie]   POST /v1/chat/completions (OpenAI)   POST /v1/messages (Anthropic)",
          flush=True)
    print("[genie]   GET /v1/models   GET /health", flush=True)
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
