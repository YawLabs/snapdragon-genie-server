# Genie NPU Server (OpenAI-compatible)

A local OpenAI-compatible HTTP endpoint backed by a Qualcomm Genie context-binary
bundle running on the Snapdragon X Elite NPU (Hexagon v73). The model is loaded
**once** (resident on the HTP via the Genie C API) so requests don't pay the
~35-50s reload that `genie-t2t-run.exe` would incur per invocation.

## Requirements

- **Native ARM64 Python** (aarch64). Genie.dll and its Qnn* deps are
  `aarch64-windows-msvc`; an x64/emulated Python cannot load them.
- The QAIRT 2.45 runtime (extracted) and a Genie bundle matching this box's
  Hexagon. Supported: **v73 (X Elite / X Plus)** and **v81 (X2 Elite)** -- the
  two Windows-on-Snapdragon parts. The server derives that set at startup by
  intersecting `lib/hexagon-v*/unsigned` (DSP skel) with
  `lib/aarch64-windows-msvc/QnnHtpV*Stub.dll` (Windows stub); an arch needs
  both. v75 (8 Gen 3) and v79 (8 Elite) ship a skel but no Windows stub -- they
  are Android parts -- and are reported as skipped rather than silently
  offered. `GENIE_HEXAGON_ARCH=v81` pins one arch.
- A bundle is locked to one arch AND one QAIRT version; a mismatch fails at
  `GenieDialog_create` with a message naming the archs this box can offer.

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
  (SSE), `max_tokens`, and `tools`. (`stop` is NOT implemented -- the Genie C
  API exposes `GenieDialog_setStopSequence`, but this server does not wire it
  yet.) ChatML template is taken from the bundle's own
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
| `GENIE_SDK_DIR` | scratchpad 2.45 SDK | QAIRT 2.45 root (lib/aarch64-windows-msvc, lib/hexagon-v*) |
| `GENIE_HEXAGON_ARCH` | unset | pin one skel arch (`v81`); default offers all |
| `GENIE_SUMMARIZE_EVICTED` | 1 | 0 disables summarising evicted turns (plain drop) |
| `GENIE_SUMMARY_MAX_TOKENS` | 192 | cap on the retained note |
| `GENIE_WINDOW_MARGIN` | 64 | headroom left between prompt and n_ctx |
| `GENIE_HOST` / `GENIE_PORT` | 127.0.0.1 / 8080 | bind address |
| `GENIE_MODEL_ID` | qwen3-4b-npu | id reported to clients |
| `GENIE_MAX_TOKENS` | 512 | default cap when a request omits max_tokens |
| `GENIE_STRIP_THINK` | 0 | 1 strips `<think>...</think>` from non-streamed content |
| `GENIE_THINKING` | 1 | 0 suppresses Qwen3's reasoning block server-wide. Per request: `chat_template_kwargs.enable_thinking`, `reasoning_effort:"none"`, or `thinking:{"type":"disabled"}` |

## Notes / limitations

- **Context window: evict, don't crash.** The compiled window is fixed (this
  bundle: 4096) and Genie has NO sliding-window mode -- QAIRT 2.45 exposes no
  such flag on `genie-t2t-run` and no equivalent config key, and overflowing is
  a hard `GenieDialog_query` failure, not a truncation. So the server evicts:
  oldest turns are dropped until the prompt fits, with the system turn and tool
  schemas anchored and tool results never separated from the call that produced
  them. Eviction is logged (never silent). A single message too big to fit even
  alone gets a 400 naming the token counts, not a doomed query.
  `GENIE_WINDOW_MARGIN` (default 64) is the headroom left for generation.

- **Eviction summarises instead of discarding.** Dropping the oldest turns
  outright makes the agent forget it already read a file and read it again --
  burning the window a second time on information it had. So when eviction
  fires, the outgoing turns are condensed by one NPU call into a short note
  folded into the SYSTEM turn (the one thing eviction never touches). A later
  eviction re-summarises the previous note together with the newly evicted
  turns, so notes never stack.

  Measured, same 30-turn conversation with a fact stated at the start:

  | | wall | answer |
  |---|---|---|
  | `GENIE_SUMMARIZE_EVICTED=1` (default) | 12.6s | recalled the key |
  | `=0` | 8.5s | lost it |

  The extra ~4s is paid only when eviction was going to happen anyway. If the
  summarisation call fails, or the note itself will not fit, the server falls
  back to plain eviction -- a summary is never allowed to break a request.
  `GENIE_SUMMARY_MAX_TOKENS` (default 192) bounds the note.

- **KV reuse across turns.** The dialog keeps its KV between queries, so when a
  request's prompt is a byte-exact extension of what the dialog already holds,
  only the new suffix is prefilled. Measured: 1.23s cold, then 0.66s / 0.68s on
  the two following turns of the same conversation, versus 5.48s for an
  unrelated one. The match must be exact -- edited history, an evicted turn, or
  an aborted generation all fall back to a full re-prefill, because resuming on
  mismatched KV would answer from a history that never happened.
  (`GenieDialog_save`/`restore` also exist and work -- measured ~75 KB/token on
  disk, ~128 MB at 1711 tokens -- but they are not used: in-memory continuation
  is free and this server serves one conversation at a time.)

- **Single-flight.** The NPU serves one query at a time (concurrent HTP access
  wedges the device), so requests are serialized by a lock. Fine for one agent.
- **Tool calling works, and thinking dominates its latency.** Enabled when the
  bundle's tokenizer carries `<tool_call>` (probed at startup; a bundle without
  it still gets an honest 400). Measured on this box, same prompt and same
  correct call:

  | | wall | completion tokens |
  |---|---|---|
  | thinking on (default) | 10.7 - 41 s | 113 - 300 |
  | thinking off | 1.8 - 2.4 s | 17 - 25 |

  Nearly all of the default-path cost is the `<think>` block, and its length
  varies a lot run to run -- so agent step latency is not just slow but
  unpredictable. For agentic use, turn thinking off.

- **Tool-call wrappers vary.** `<tool_call>` is the trained, in-vocab tag, but
  with thinking suppressed the model also emits `<function_call>` and
  occasionally bare JSON with no wrapper. The parser accepts all of these; a
  block whose payload does not parse is left VISIBLE in the content rather than
  silently dropped, so a malformed call is debuggable instead of invisible.

- **Throughput is bandwidth-bound.** Decode is ~13 t/s on a quiet box; it drops
  sharply under memory pressure (the X Elite's 32 GB LPDDR5x is shared by CPU/GPU/NPU),
  so a large resident model elsewhere (e.g. a 26 GB llama-server) will slow it.
- **`finish_reason`** reports `length` only on context-limit; a `max_tokens` cap
  currently reports `stop` (Genie signals a normal sentence-end at the cap).
- Model swaps: point `GENIE_BUNDLE_DIR` at the 1.7B bundle for lower latency, or
  any other X-Elite Genie bundle.
