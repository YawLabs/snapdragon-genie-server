# Brief: local multi-engine routing in typed

Self-contained handoff for an agent working in the **typed** repo. You do not
need the `snapdragon-npu-llm` repo to act on this. Every number below is
measured on a Snapdragon X Elite (X1E80100, Hexagon v73), Windows on ARM64.

## What exists today

typed's free local tier runs `llama-server`. Separately there is now a
**Genie NPU server** that puts a Qwen3-4B w4a16 bundle resident on the Hexagon
NPU and serves it over HTTP. It speaks both APIs typed already knows:

| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | OpenAI, SSE streaming, `tools`, `stop` |
| `POST /v1/messages` | Anthropic, SSE streaming, `tools`, `stop_sequences` |
| `GET /props` | llama.cpp-shaped: `default_generation_settings.n_ctx`, `model_alias` |
| `GET /v1/models` | superset item satisfying both OpenAI and Anthropic shapes |
| `GET /health` | liveness only -- see the caveat below |

Default bind is `127.0.0.1:8123`.

## The one architectural decision already made

**The router goes in typed. Do not build it into the NPU server.**

That server is deliberately one model on one engine. typed already selects
backends and already probes these endpoints (`/props` for the window,
`probeLocalToolCalls` for tool support), so the capability-negotiation
machinery lives there. Putting dispatch in the engine would duplicate it and
conflate serving with routing.

## The engines to route across

Same model, three backends, one shared 31.6 GB memory pool (no dedicated VRAM
-- the Adreno reports `AdapterRAM: not reported / shared`).

| engine | server | prefill t/s | decode t/s | status |
|---|---|---|---|---|
| NPU (Hexagon) | Genie server | 277* | 13.2 | best; **single-flight** |
| GPU (Adreno) | `llama-server` | 117 | 6.0 | works |
| CPU (KleidiAI) | `llama-server` | fine | **~0.2** | **broken, unexplained** |

Resident cost is ~3.3 GB per instance at 4k context, ~4.2 GB at 16k (KV is
`uint8`, ~72 KB/token, allocated for the whole compiled window at load) -- so
three instances fit in ~10-13 GB. **Capacity is not the constraint; memory
bandwidth is.** Decode streams the whole model per token and all engines share one bus,
so concurrent instances divide throughput rather than multiplying it. Budget
for **1.5-2x aggregate, not 3x**.

\* The NPU prefill figure is understated -- a re-measurement on a quiet box
gave a median **971 t/s** on the same 4096 bundle (decode agreed, 13.0). The
original sweep looks to have been taken while the box was loaded. Do not plan
capacity against 277.

Do not plan around the CPU leg until its 0.2 t/s decode is explained. Treat
this as a two-engine design today.

## Why route at all

**The NPU serves exactly one request at a time.** Concurrent Hexagon access
wedges the device, so the server serializes behind a lock. A second engine
gives you concurrency, plus somewhere to go when the HTP throws its transient
`Code 1003` device fault.

## What to build

1. **Endpoint registry** -- per local endpoint: base URL, `n_ctx` (from
   `/props`), tools supported (from a startup probe), measured decode rate.
2. **Route by request shape** -- does it need tools? a window larger than the
   endpoint's `n_ctx`? is it latency-sensitive?
3. **Failover on backpressure.** This signal already exists: when its small
   queue is full the NPU server returns **`429`** (OpenAI) / **`529`**
   (Anthropic) with `"server busy; NPU is single-flight"`. It was not built for
   routing, but it is precisely the shed-to-next-engine signal a dispatcher
   needs. Treat it as "try another engine", not as an error to surface.
4. **Health checks that survive the wedge.** `/health` answering is **not**
   proof the device will execute -- the `1003` fault happens at execute time,
   not at load. A real check needs a tiny generation, not a liveness ping.

## Capability limits worth encoding

These are properties of the NPU endpoint that a router must not assume away:

- **Context is 4096 tokens** on the default bundle. Read it from `/props`; do
  not hardcode. For scale: a realistic agent preamble (system prompt + 6 tool
  schemas + one user turn) measured **626 tokens**, leaving ~3470, and real
  source code runs ~10-13 tokens/line. That is roughly one medium file in
  context.

- **A 16384-token bundle now exists, and it is a TIER, not an upgrade.** The
  16k rebuild finished and runs. But a Genie bundle's KV tensors are graph
  inputs statically shaped to the compiled window, so the whole buffer moves on
  every decode step regardless of how full it is. Measured on the same box,
  same server, same prompts, both bundles otherwise byte-identical:

  | compiled n_ctx | prefill t/s (median) | decode t/s (median) |
  |---|---|---|
  | 4096 | **971** (938-1016) | **13.0** (11.2-13.2) |
  | 16384 | **171** (168-181) | **3.1** (3.0-3.2) |

  Both are FLAT with depth -- the 16k bundle decodes at 3.13 t/s with 469
  tokens of context and 3.02 t/s with 10532, so the ~4x penalty applies to
  short requests too. A 10532-token prefill takes **63 seconds**.

  **Routing rule that falls out of this: send a request to the smallest window
  that fits it.** Do not treat a larger `n_ctx` as strictly better when
  ranking endpoints -- on this engine it is a latency class. The 16k endpoint
  earns its cost only for requests that genuinely cannot fit in 4096, and even
  then the 4k endpoint plus server-side eviction/summarisation is usually the
  faster answer.
- **Tool calling works.** Verified end-to-end on both APIs. If a bundle cannot
  do tools, the server returns a `400` naming the limitation rather than
  accepting `tools` and ignoring them -- so a 4xx on a tools probe means
  "disable tools for this session", exactly as `probeLocalToolCalls` expects.
- **Reasoning is expensive and should usually be off.** Qwen3 emits a `<think>`
  block by default: measured **41s vs 2.4s** for the same tool-calling turn.
  Suppress it per request with any of `chat_template_kwargs.enable_thinking:
  false`, `reasoning_effort: "none"`, or `thinking: {"type": "disabled"}`.
  **For agentic use, send one of these.**
- **`stop` / `stop_sequences` work.** `stop_reason` distinguishes
  `stop_sequence` from `end_turn`, though the matched sequence is reported as
  `null` (the runtime strips it before we see it).
- **Sampling is server-level, not per-request.** `temperature` / `top_p` /
  `top_k` are accepted and **not honoured** -- the runtime binds its sampler at
  load time and ignores a later change. Do not build routing logic that depends
  on varying temperature per request.
- **Context overflow is handled, not fatal.** Oversized history is evicted
  oldest-first with the system turn and tool schemas anchored, and the evicted
  turns are summarised into a retained note rather than dropped. A single
  message too large to fit even alone returns a `400` naming the token counts.
- **Multi-turn is cheap if you resend history verbatim.** The server reuses the
  resident KV when a prompt is a byte-exact extension of the previous one:
  measured 1.23s cold, then 0.66s / 0.68s on following turns. **Editing earlier
  turns forfeits this** and forces a full re-prefill -- worth knowing before
  the client rewrites history between turns.

## Measure before designing around it

- Concurrent GPU + NPU contention has **not** been measured. Both hit the same
  memory controller. Get a number before committing to a routing policy.
- Whether a second resident model raises the NPU's `1003` rate is unknown.
- Benchmark on a quiet box. During an unrelated build here free memory hit
  **1.1 GB**, and any numbers taken then would have been meaningless.
