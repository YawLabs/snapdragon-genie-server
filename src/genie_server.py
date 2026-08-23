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

Handle = C.c_void_p
# void callback(const char* response, int sentenceCode, const void* userData)
QUERY_CALLBACK = C.CFUNCTYPE(None, C.c_char_p, C.c_int, C.c_void_p)


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
    """Resident Genie dialog on the HTP. All queries serialized by a lock."""

    def __init__(self, lib, dialog):
        self.lib = lib
        self.dialog = dialog
        self.lock = threading.Lock()

    def query(self, prompt, on_text, max_tokens=None):
        """Run one query. on_text(str) is called for each streamed chunk.
        Returns finish_reason ('stop' | 'length'). Serialized (NPU is single)."""
        with self.lock:
            self.lib.GenieDialog_reset(self.dialog)
            if max_tokens:
                self.lib.GenieDialog_setMaxNumTokens(self.dialog, C.c_uint32(max_tokens))

            state = {"finish": "stop"}

            def _cb(resp, code, _udata):
                if resp:
                    try:
                        on_text(resp.decode("utf-8", "replace"))
                    except Exception:
                        pass
                if code == SENTENCE_ABORT:
                    state["finish"] = "stop"

            cb = QUERY_CALLBACK(_cb)  # keep ref alive for the blocking call
            status = self.lib.GenieDialog_query(
                self.dialog, prompt.encode("utf-8"), SENTENCE_COMPLETE, cb, None
            )
            if status == GENIE_STATUS_WARNING_CONTEXT_EXCEEDED:
                state["finish"] = "length"
            elif status != GENIE_STATUS_SUCCESS:
                raise RuntimeError("GenieDialog_query failed, status=%d" % status)
            return state["finish"]


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _maybe_strip_think(text):
    return _THINK_RE.sub("", text) if STRIP_THINK else text


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

    return GenieEngine(lib, dialog)


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

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "qualcomm-genie-npu"}
            ]})
        elif self.path.rstrip("/") in ("/health", "/healthz"):
            self._json(200, {"status": "ok", "model": MODEL_ID})
        else:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._json(400, {"error": {"message": "bad JSON: %s" % e,
                                       "type": "invalid_request_error"}})
            return

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
        self._json(200, {
            "id": cmpl_id, "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }],
            # Best-effort usage (NPU tokenizer count not surfaced cheaply).
            "usage": {"prompt_tokens": len(prompt) // 4,
                      "completion_tokens": len(content) // 4,
                      "total_tokens": (len(prompt) + len(content)) // 4},
        })

    def _stream(self, prompt, max_tokens, cmpl_id, created):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        def frame(delta, finish=None):
            return {"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        sse(frame({"role": "assistant"}))
        # NOTE: with STRIP_THINK we cannot cleanly strip mid-stream, so streamed
        # output is always faithful (includes <think>); non-stream honors the flag.
        try:
            finish = ENGINE.query(
                prompt, lambda t: sse(frame({"content": t})), max_tokens=max_tokens)
        except Exception as e:
            sse(frame({"content": "\n[error: %s]" % e}, finish="stop"))
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            return
        sse(frame({}, finish=finish))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global ENGINE, TEMPLATE
    TEMPLATE = load_chat_template()
    ENGINE = load_engine()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print("[genie] OpenAI-compatible endpoint on http://%s:%d  (model=%s)"
          % (HOST, PORT, MODEL_ID), flush=True)
    print("[genie]   POST /v1/chat/completions   GET /v1/models   GET /health", flush=True)
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
