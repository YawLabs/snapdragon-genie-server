# Genie NPU Server (OpenAI-compatible)

A local OpenAI-compatible HTTP endpoint backed by a Qualcomm Genie context-binary
bundle running on the Snapdragon X Elite NPU (Hexagon v73). The model is loaded
**once** (resident on the HTP via the Genie C API) so requests don't pay the
~35-50s reload that `genie-t2t-run.exe` would incur per invocation.

## Requirements

- **Native ARM64 Python** (aarch64). Genie.dll and its Qnn* deps are
  `aarch64-windows-msvc`; an x64/emulated Python cannot load them.
- The QAIRT 2.45 runtime (extracted) and a precompiled Genie bundle for X Elite.
  Defaults point at this repo's scratchpad layout.
- No pip packages. Pure Python stdlib.

## Run

The Genie bundle and the QAIRT 2.45 runtime are large external artifacts and are
**not** in this repo -- point `GENIE_BUNDLE_DIR` / `GENIE_SDK_DIR` at wherever you
extracted them (edit the defaults in `run-genie-server.ps1`, or set the env vars).

```powershell
$env:GENIE_BUNDLE_DIR = "...\qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite"
$env:GENIE_SDK_DIR    = "...\qairt\2.45.0.260326"
$env:GENIE_PORT       = "8123"    # 8080 often collides with a llama-server
python src\genie_server.py
# or: powershell -File src\run-genie-server.ps1
```

Startup prints `model resident on HTP in <N>s` then the endpoint URL. First load
is ~35-50s; after that every request reuses the resident model.

## Endpoints

- `POST /v1/chat/completions` -- OpenAI chat API. Supports `messages`, `stream`
  (SSE), `max_tokens`, `stop`. ChatML template is taken from the bundle's own
  `metadata.json` chat_template.
- `GET /v1/models` -- lists the served model id (`GENIE_MODEL_ID`).
- `GET /health` -- liveness.

```bash
curl http://127.0.0.1:8123/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b-npu","messages":[{"role":"user","content":"Hi"}],"max_tokens":128}'
```

## Environment

| var | default | meaning |
|---|---|---|
| `GENIE_BUNDLE_DIR` | scratchpad 4B bundle | dir with genie_config.json + part*_of_*.bin + tokenizer.json |
| `GENIE_SDK_DIR` | scratchpad 2.45 SDK | QAIRT 2.45 root (lib/aarch64-windows-msvc, lib/hexagon-v73) |
| `GENIE_HOST` / `GENIE_PORT` | 127.0.0.1 / 8080 | bind address |
| `GENIE_MODEL_ID` | qwen3-4b-npu | id reported to clients |
| `GENIE_MAX_TOKENS` | 512 | default cap when a request omits max_tokens |
| `GENIE_STRIP_THINK` | 0 | 1 strips `<think>...</think>` from non-streamed content |

## Notes / limitations

- **Single-flight.** The NPU serves one query at a time (concurrent HTP access
  wedges the device), so requests are serialized by a lock. Fine for one agent.
- **Reasoning model.** Qwen3 emits a `<think>...</think>` block before the answer.
  Non-streaming honors `GENIE_STRIP_THINK=1`; streaming is always faithful
  (can't cleanly strip mid-stream).
- **Throughput is bandwidth-bound.** Decode is ~13 t/s on a quiet box; it drops
  sharply under memory pressure (the X Elite's 32 GB LPDDR5x is shared by CPU/GPU/NPU),
  so a large resident model elsewhere (e.g. a 26 GB llama-server) will slow it.
- **`finish_reason`** reports `length` only on context-limit; a `max_tokens` cap
  currently reports `stop` (Genie signals a normal sentence-end at the cap).
- Model swaps: point `GENIE_BUNDLE_DIR` at the 1.7B bundle for lower latency, or
  any other X-Elite Genie bundle.
