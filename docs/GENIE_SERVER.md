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

Startup prints `model resident on HTP in <N>s` then the endpoint URL. Load is
~30-50s cold and ~7-8s once the 3 GB of context binaries are in the OS page
cache (measured both ways on the same bundle); after that every request reuses
the resident model.

## Endpoints

- `POST /v1/chat/completions` -- OpenAI chat API. Supports `messages`, `stream`
  (SSE), `max_tokens`, `stop` (also Anthropic `stop_sequences`), and `tools`.
  ChatML template is taken from the bundle's own
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
| `GENIE_SUMMARY_MAX_TOKENS` | 192 | cap on the retained note. Clamped at runtime to `n_ctx / 8` (floor 32) so the note cannot crowd out the window on a small-context bundle; the server logs the clamp when it bites. |
| `GENIE_WINDOW_MARGIN` | 64 | headroom left between prompt and n_ctx |
| `GENIE_MAX_INFLIGHT` | 2 | requests admitted at once (1 running + queue). Floored at 1 -- it cannot be disabled, since the NPU is single-flight and an unbounded setting only parks threads on the engine lock. Set 1 to protect KV reuse: two interleaved conversations share one resident KV and reset each other's prefix. |
| `GENIE_HOST` / `GENIE_PORT` | 127.0.0.1 / 8080 | bind address |
| `GENIE_MODEL_ID` | qwen3-4b-npu | id reported to clients |
| `GENIE_MAX_TOKENS` | 512 | default cap when a request omits max_tokens |
| `GENIE_STRIP_THINK` | 0 | 1 strips `<think>...</think>` from non-streamed content |
| `GENIE_THINKING` | 1 | 0 suppresses Qwen3's reasoning block server-wide. Per request: `chat_template_kwargs.enable_thinking`, `reasoning_effort:"none"`, or `thinking:{"type":"disabled"}` |

## Notes / limitations

- **Context window: evict, don't crash.** The compiled window is fixed (read
  from the bundle, reported at `/props`; the two bundles here are 4096 and
  16384) and Genie has NO sliding-window mode -- QAIRT 2.45 exposes no
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

- **A buffered tool stream is still abortable.** Tool responses are buffered
  (a half-emitted `<tool_call>` is worse than a slower one), which means
  nothing is written while the model generates -- so the usual
  disconnect-detection-by-failed-write never fires. Both stream paths emit a
  lightweight probe every 8 chunks (an SSE comment on the OpenAI side, a real
  `ping` event on the Anthropic side) purely so a departed client is noticed.
  Without it an abandoned tool turn runs to `max_tokens` holding the
  single-flight NPU against every other caller.

- **Streaming reports usage too.** OpenAI streams emit a final chunk with an
  empty `choices` list carrying `usage`, but only when the caller sets
  `stream_options.include_usage` -- clients that do not ask see a
  byte-identical stream to before. Anthropic streams carry `output_tokens` in
  `message_delta` as usual. Both include the summarisation overhead below when
  there was any.

- **Summarisation cost is reported, not hidden.** A request that evicts spends
  extra NPU time condensing the outgoing turns. That shows up as
  `usage.genie_context_overhead_tokens` on non-streaming responses (present
  only when non-zero, so an ordinary response is unchanged) and in the server
  log line. The summarisation call deliberately does NOT claim the resident KV
  -- it leaves text in the dialog that is not the caller's conversation, so it
  records "unknown" and the next turn re-prefills rather than resuming from a
  false prefix.

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

- **Sampling is server-level, not per-request.** `temperature` / `top_p` /
  `top_k` are accepted and **not honoured**. Measured directly against QAIRT
  2.45: `GenieDialog_getSampler` returns a valid handle,
  `GenieSamplerConfig_createFromJson({"sampler": {...}})` returns 0, and
  `GenieSampler_applyConfig` returns 0 -- yet generation is byte-identical
  across seeds 1 / 999 / 12345 and temperatures 0.0 / 1.5 / 2.0. The dialog
  binds its sampler at `GenieDialog_create` time. To change sampling, edit
  `dialog.sampler` in `genie_config.json` before the server loads it. The
  server prints this limitation at startup rather than letting it be silent.

  Two JSON shapes worth knowing, both found by probing: the sampler config
  must be wrapped as `{"sampler": {...}}` (a bare object returns -8 "Missing
  field"), and stop sequences must be `{"stop-sequence": [...]}` (a bare array
  returns -8 "Top level config is not an object" and is silently ignored).

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

- **The compiled context window is a per-token tax, paid whether or not you
  use it.** A Genie bundle's KV tensors are graph INPUTS statically shaped to
  the compiled window -- `past_key_0_in: [8, 1, 128, n_ctx-1]`, `uint8` -- so
  every decode step feeds the whole buffer through the HTP no matter how few
  positions are filled. Cost therefore tracks the window the bundle was BUILT
  at, not the context actually in play.

  Measured on this box, same server, same prompts, minutes apart, quiet box
  (11.9 GB free). The two bundles are the same model: `precision`,
  `tool_versions`, `chipset_attributes`, `config.json`, the chat template and
  every tokenizer file are byte-identical, and the ONLY difference in
  `metadata.json` is the KV shapes.

  | compiled n_ctx | HTP alloc | prefill (median) | decode (median) |
  |---|---|---|---|
  | 4096 | 328 MB | **914 t/s** (845-960) | **13.0 t/s** (11.2-13.2) |
  | 16384 | 1195 MB | **167 t/s** (160-169) | **3.1 t/s** (3.0-3.2) |

  Reproduce with `python src/bench_endpoint.py` against each bundle; decode is
  measured as the delta between an N-token and a 1-token run at the same depth,
  so prefill is subtracted out rather than folded into the rate.

  Both curves are FLAT with depth, which is the tell. The 16k bundle prefilled
  at 161 / 164 / 169 / 167 / 168 t/s across 469 / 1344 / 2657 / 6157 / 10532
  prompt tokens, and decoded at 3.13 t/s with 469 tokens of context versus 3.02
  t/s with 10532 -- a spread of 0.15 t/s over a 22x change in context. So the 4x
  window costs ~4x on decode and ~5.5x on prefill *at an almost empty context*.
  A 10532-token prefill takes **63 seconds**.

  (The 4096 bundle does show a mild real depth effect on top of the fixed tax --
  13.2 t/s shallow falling to 11.2 t/s at 2657 tokens. It is small next to the
  4x difference between bundles, and the 16k bundle's near-total flatness is
  what says the dominant term is the compiled window, not the fill.)

  So a bigger bundle is a **capability tier, not an upgrade**: it buys window
  that 4096 cannot hold at all, and charges for it on every request including
  the short ones. Pick the smallest window that fits the workload, and prefer
  eviction + summarisation over a bigger bundle when the history is
  compressible -- which is the case this server is built around.

- **`dialog.context.size` cannot buy that tax back.** It is a software-side
  limit, not a graph selector. Setting it to 1024 against the 4096-compiled
  bundle left the HTP allocation byte-identical (343,933,440 both ways) and
  decode unchanged at ~11.0 t/s. The window is fixed at export time
  (`--context-lengths`); lowering the config only lowers the ceiling the
  evictor works against. Related trap: the 4k bundle's `metadata.json`
  advertises `genie.context_lengths = [512, 1024, 2048, 3072, 4096]`, which
  lists what the model can be EXPORTED at -- not multiple graphs inside the
  `.bin`. Its actual KV shape is `4095`.

- **KV is `uint8`, ~72 KB/token** -- read off the bundle's own `metadata.json`
  (`past_key_0_in` / `past_value_0_in` are `dtype: uint8` with a fixed quant
  scale), not assumed. For this model that is 36 layers x 2 x 8 KV heads x 128
  head_dim x 1 byte = 73,728 B/token; the measured HTP allocation delta between
  the two bundles works out to 73,983 B/token, 0.3% off. Any fp16 estimate of
  KV size for this bundle is 2x too high.

- **Throughput is bandwidth-bound.** Decode is ~13 t/s on a quiet box for
  the 4096 bundle (the figure belongs to the bundle's window, not to the
  server -- see the table above); it drops
  sharply under memory pressure (the X Elite's 32 GB LPDDR5x is shared by CPU/GPU/NPU),
  so a large resident model elsewhere (e.g. a 26 GB llama-server) will slow it.
- **`finish_reason`** reports `length` only on context-limit; a `max_tokens` cap
  currently reports `stop` (Genie signals a normal sentence-end at the cap).
- Model swaps: point `GENIE_BUNDLE_DIR` at the 1.7B bundle for lower latency, or
  any other X-Elite Genie bundle. Note that the bundle's **compiled window** is
  as big a latency lever as its parameter count -- see the window-tax note
  above before assuming a larger-context bundle is strictly better.
