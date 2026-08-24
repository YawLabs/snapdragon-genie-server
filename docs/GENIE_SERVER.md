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
| (not an env var) | -- | **`poll: false` in the bundle's `genie_config.json`** -- see the poll note below. Worth up to +55% decode and frees 2.7 idle cores. |
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

- **It will not start on a port something else is already serving.** Checked
  before the model loads, so a collision costs 0.3s rather than 30-50s of
  loading followed by a failure. The check exists because on Windows the bind
  does NOT fail: `HTTPServer` sets `allow_reuse_address`, which on POSIX means
  "rebind a TIME_WAIT socket" but on Windows lets a second process bind a port
  another process is actively serving. Both binds succeed and the OLD process
  keeps answering -- so the new server logs a clean startup, reports the right
  HTP allocation for its bundle, and serves nobody, while requests are answered
  by whatever was already there. That happened during the window benchmarking
  and was caught only because `/props` disagreed with the bundle just loaded.
  A server that silently answers from the wrong model is the same failure mode
  as a silent CPU fallback, so it now refuses instead.

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

- **Set `poll: false` in the bundle's `genie_config.json`. It is the single
  biggest free win here.** The QnnHtp backend block ships `"poll": true`, which
  busy-waits. Measured on this box: a resident server with `poll: true` burns
  **270% CPU (2.7 cores) while completely idle**, no requests in flight -- and
  the spinning threads compete with the work, so it is slower as well as
  wasteful:

  | compiled n_ctx | decode, `poll: true` | decode, `poll: false` | idle CPU |
  |---|---|---|---|
  | 4096 | 11.6 t/s | **18.0 t/s** (+55%) | 267% -> 0% |
  | 8192 | 7.9 t/s | **8.8 t/s** (+11%) | 267% -> 0% |
  | 16384 | 3.2 t/s | **3.3 t/s** (+2%) | 267% -> 0% |

  The idle figure is controlled, because the first attempt at it was not. It
  was originally sampled just after a benchmark, which cannot distinguish an
  idle spin from a generation still draining -- a fair objection raised against
  it. Re-run on a server that has answered nothing but `/health`, same bundle
  and same 29 threads, 30-second samples: **267.1%** with `poll: true`, **0.0%**
  with `poll: false`, and **0.1%** with `poll: false` thirty seconds after a
  generation. It is a spin, not drain.

  Prefill improves too (4096: 629 -> 1157 t/s median) and run-to-run noise
  drops sharply (1.85 -> 0.23 t/s at 4096). Nothing measured got worse. The
  penalty shrinks as the window grows because the NPU work per token grows
  while the CPU spin stays constant, so a small-window bundle -- the fast,
  latency-sensitive case -- is hurt most.

  Two consequences beyond throughput. Idle CPU is not free on a laptop, and
  more importantly **an idle NPU server was stealing 2.7 cores from anything
  else on the box**, which contaminates any concurrent benchmark of another
  engine and quietly undermines the multi-engine plan in `MULTI_ENGINE.md`.
  Every number in this file predating this finding was measured with
  `poll: true` and is therefore pessimistic.

- **The compiled context window is a per-token tax, paid whether or not you
  use it.** A Genie bundle's KV tensors are graph INPUTS statically shaped to
  the compiled window -- `past_key_0_in: [8, 1, 128, n_ctx-1]`, `uint8` -- so
  every decode step feeds the whole buffer through the HTP no matter how few
  positions are filled. Cost therefore tracks the window the bundle was BUILT
  at, not the context actually in play.

  Three windows, same model, `poll: false`, quiet box, one harness
  (`python src/bench_endpoint.py`). The 8192 and 16384 bundles came from
  `qai-hub-models export` and are the same model as the 4096 prebuilt:
  `precision`, `tool_versions`, `chipset_attributes`, `config.json`, the chat
  template and every tokenizer file are byte-identical, and the only
  `metadata.json` difference is the KV shapes.

  | compiled n_ctx | HTP alloc | prefill (median) | decode (median) | cost per doubling |
  |---|---|---|---|---|
  | 4096 | 328 MB | **1157 t/s** | **18.0 t/s** | -- |
  | 8192 | 647 MB | **458 t/s** | **8.8 t/s** | 2.05x decode, 2.54x prefill |
  | 16384 | 1195 MB | **176 t/s** | **3.3 t/s** | 2.69x decode, 2.59x prefill |

  So **decode is roughly inverse-linear in the window up to 8192 and worse
  beyond it**: the first doubling costs 2.05x (almost exactly the 2x a fixed
  per-token tax predicts), the second 2.69x. Prefill is consistently worse than
  inverse-linear, ~2.55x per doubling. HTP allocation is exactly linear at
  73,983 bytes per token of window, which doubles as a check that a bundle is
  the window it claims -- the 8192 bundle allocated 646,971,904 bytes against a
  646,971,904 prediction.

  Decode is FLAT with depth on both self-exported bundles, which is the tell:
  the 16384 bundle decoded 3.26 t/s holding 469 tokens and 3.27 t/s holding
  10532, and the 8192 bundle 8.77 versus 8.81. Cost is set by the compiled
  window, not by how much of it is live. A 10532-token prefill on the 16k
  bundle takes **60 seconds** of wall time.

  **The 4096 prebuilt is the exception, and the reason is still open.** It is
  NOT flat: measured with shallow and deep runs INTERLEAVED (250, 3300, 250,
  3300, 250, 3300) so that a drift over the run could not be mistaken for a
  depth effect, it gives **18.9 / 18.5 / 18.7 t/s at 250 tokens against
  12.8 / 13.0 / 13.3 at 3300** -- a real ~30% decline that tracks depth rather
  than elapsed time. Both self-exported bundles are flat over far wider spans.

  So a prebuilt appears to pay only for the context actually in use, while a
  single-length export pays for its whole compiled window on every token. That
  is a meaningful artifact-quality difference and worth chasing, because it is
  the difference between 18.9 and 8.8 t/s on a short prompt.

  **A tempting explanation was tested and REFUTED.** The prebuilt advertises
  `genie.context_lengths = [512, 1024, 2048, 3072, 4096]` where the exports
  advertise one value, which suggested it carries several graphs and picks the
  smallest that fits -- and a coarse sweep did look like plateaus stepping down
  at those boundaries. It was an artifact of the bin edges. A targeted sweep
  either side of the 512-graph boundary (requested depths 440 / 470 / 490 /
  510 / 540, where the switch would have to fall between 490 and 510) shows a
  smooth -2.2% / -1.7% / -2.1% / -4.1% slide with no step at all. Discrete
  graph selection would have produced roughly flat readings then one sharp
  drop. Whatever the prebuilt does differently, it is not that, and exporting
  with several `--context-lengths` values is NOT known to buy it back.

  Practical upshot: **8192 is the sweet spot for the agent workload.** It buys
  2x the context of 4096 for about half the decode rate, which is the fair
  exchange rate; 16384 buys 4x the context for less than a fifth. Prefer the
  smallest window the workload needs, and prefer eviction + summarisation over
  a bigger bundle when the history compresses -- which is what this server is
  built around.

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

- **Throughput is bandwidth-bound.** Decode is ~18 t/s on a quiet box for the
  4096 bundle with `poll: false` (~12 t/s as the bundle ships). The figure
  belongs to the bundle's window and its poll setting, not to the server -- see
  the tables above. It drops
  sharply under memory pressure (the X Elite's 32 GB LPDDR5x is shared by CPU/GPU/NPU),
  so a large resident model elsewhere (e.g. a 26 GB llama-server) will slow it.
- **`finish_reason`** reports `length` only on context-limit; a `max_tokens` cap
  currently reports `stop` (Genie signals a normal sentence-end at the cap).
- Model swaps: point `GENIE_BUNDLE_DIR` at the 1.7B bundle for lower latency, or
  any other X-Elite Genie bundle. Note that the bundle's **compiled window** is
  as big a latency lever as its parameter count -- see the window-tax note
  above before assuming a larger-context bundle is strictly better.
