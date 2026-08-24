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
| NPU (Hexagon) | Genie server | 277* | 18.0* | **single-flight**; best prefill, near-immune to host load |
| GPU (Adreno) | `llama-bench` -- see note | **226.8** | **18.05** | fastest decode on an idle box, **-64% on a busy one**; not reachable over HTTP today |
| CPU (KleidiAI) | `llama-server` | fine | **~0.2** | **broken, unexplained** |

Resident cost is ~3.3 GB per instance at 4k context, ~4.2 GB at 16k (KV is
`uint8`, ~72 KB/token, allocated for the whole compiled window at load) -- so
three instances fit in ~10-13 GB. **Capacity is not the constraint. Neither is
memory bandwidth** -- this brief said it was until 2026-08-23, and the
measurement below falsifies it.

**`llama-server` cannot drive the Adreno.** The build in
`llama-qnn-fork/build-3way` silently loads on CPU (`kleidiai`, `n_threads=12`,
zero OpenCL init) even with `-ngl 99 --device GPUOpenCL --fit off`, and
`--list-devices` prints nothing. Nothing errors -- requests are answered, at CPU
speed, by the wrong engine. `llama-bench` from the same directory and the same
DLLs drives the GPU correctly. So the GPU numbers here were taken through
`llama-bench`, and **"GPU -> `llama-server` over HTTP" is not a working path
until a build ships whose server initialises OpenCL.** Check the backend line in
the server's own startup log before believing any GPU figure taken over HTTP.

**Where the GPU figures come from.** Measured on a cooled, quiet box 2026-08-23
with `src/bench_contention.py` in `snapdragon-npu-llm`; Qwen3-4B Q4_K_M
(2.32 GiB GGUF), decode at context depth 469, n=3. **Supersedes 117 / 6.0** --
decode was understated 3.0x. Those retired figures came from the same loaded
window as the NPU's 277, and paid whatever a resident NPU server costs on top
(see below). The CPU row came from that same window, has **not** been
re-measured, and should still be treated as a loaded-box number.

**On the GPU leg, Q4_K_M is the fast path -- not merely the smaller file.**
Measured at d0 on the same box: Q4_K_M (2.32 GiB) decodes **19.39** t/s, Q8_0
(3.98 GiB) **9.14**. Weights grew 1.71x but decode fell 2.12x, and achieved
bandwidth *fell* from 45.0 to 36.4 GB/s -- the signature of leaving an optimised
kernel, not of moving more bytes. `GGML_OPENCL_SOA_Q` and
`GGML_OPENCL_USE_ADRENO_KERNELS` target Q4, so Q8_0 pays twice. **Do not offer
Q8_0 as a "higher quality" GPU tier on this backend.**

\* Both NPU figures are understated. Re-measured on a quiet box with
`poll: false`, the same 4096 bundle gives a median **1157 t/s** prefill and
**18.0 t/s** decode. The original sweep looks to have been taken on a loaded
box, and everything before 2026-08-24 additionally paid the `poll: true`
busy-wait penalty. Do not plan capacity against 277 / 13.2. The 2026-08-23
contention run independently put the NPU at 855-938 t/s prefill and **12.82 t/s
decode at depth 469** -- confirming the old 13.2, and saying the 18.0 is a
near-empty-context number, since it was taken at d~0. That shape resembles the
prebuilt bundle's depth curve rather than a self-export's (see the window
section below), so record which bundle a rate came from as well as at what
depth. **Quote NPU decode with its depth.**

Do not plan around the CPU leg until its 0.2 t/s decode is explained. Treat
this as a two-engine design today.

## Concurrency measured: it is a net loss (2026-08-23)

Both engines hot, same model on each leg, decode at d469, every sample gated to
>=92% of base clock:

| | solo | contended | retains |
|---|---|---|---|
| GPU | **18.05** t/s | **7.27** +-0.72 | 40% |
| NPU | **12.82** t/s | **6.88** | 54% |

Aggregate while both are hot is **14.15 t/s**, against **18.05** for the best
single engine. That is **0.78x -- routing to a second hot engine makes the box
slower**, at 46% of the additive ideal (30.87). The old "budget for 1.5-2x
aggregate" line in this brief was derived from the bandwidth model, not from a
measurement, and it does not survive one.

**And it is not the bus.** Both engines together drew **32.7 GB/s, 24% of the
135.2 GB/s peak**, while each lost more than half its throughput; the additive
solo case would have been 71.4 GB/s (53%), and this silicon streams ~110-115
GB/s in practice. A bus at a quarter of capacity is not the constraint.

**A resident NPU server costs the GPU 15% while doing nothing at all.** A/B from
a cooled start: GPU decodes **17.94** t/s with the NPU server stopped, **15.26**
with it loaded and idle, **7.27** with it serving. The idle penalty is paid by a
process running zero inference, moving zero inference bytes, and measurably
**0.00 CPU cores** -- so neither bandwidth nor CPU contention explains it. The
surviving hypothesis is a **shared package power budget** (the bundle pins
`perf_profile: "burst"` with `rpc_control_latency: 100`), which fits the
zero-core penalty and the variance jump, but **has not been measured**. Treat
"it is power" as untested; treat "it is not bandwidth and not CPU" as
established.

**Host load hits the two engines completely differently.** Measured under an
unrelated ~20% CPU job versus a cooled box: NPU 12.67 vs 12.82 (**-1.2%**), GPU
11.04 vs 18.05 (**-64%**). Hexagon has its own clock domain; the OpenCL path is
host-dispatch-bound per token. typed's local tier runs on a developer machine
that is usually compiling or running tests, so **a router should not rank these
engines off idle-box numbers** -- the GPU's 18.05 is a best case that a build
running in another window erases, while the NPU's rate barely moves.

Three rules fall out for the router:

1. **Never fan a single workload across both engines for speed.** It is 0.78x.
2. **Route to a second engine for concurrency and failover only** -- a queued
   request served slowly still beats one waiting behind the single-flight lock.
3. **Stop an idle engine rather than parking it hot.** A quarter of the GPU's
   loss is the mere residency of an unused NPU server.

Reproduce any of this with `src/bench_contention.py` in `snapdragon-npu-llm`
(`--npu` / `--gpu` base URLs, `--depth`, `--repeat`; it refuses to run on a
loaded box unless you pass `--allow-loaded`, which stamps every result LOADED).
Note that it expects both legs over HTTP, so the GPU leg needs the
`llama-server` problem above solved first, or driving by hand.

## Why route at all

**The NPU serves exactly one request at a time.** Concurrent Hexagon access
wedges the device, so the server serializes behind a lock. A second engine
gives you concurrency, plus somewhere to go when the HTP throws its transient
`Code 1003` device fault. What it does **not** give you is throughput -- see the
0.78x measurement above.

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
5. **Engine lifecycle, not just engine selection.** A loaded-but-idle NPU
   server costs the GPU 15% of its decode rate, so "keep every engine warm so
   dispatch is instant" is the wrong default here: start the engine the route
   picks and stop the one it does not, and weigh the load time against the
   residency tax rather than assuming a hot spare is free.

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
  | 4096 | **1157** | **18.0** |
  | 8192 | **458** | **8.8** |
  | 16384 | **176** | **3.3** |

  Decode is FLAT with depth on both self-exported bundles -- the 16k decodes at
  3.26 t/s with 469 tokens of context and 3.27 t/s with 10532 -- so the penalty
  applies to short requests too. A 10532-token prefill takes **60 seconds**.
  (All measured with `poll: false` in the bundle config; as shipped,
  `poll: true` busy-waits and costs up to 55% of decode plus 2.7 idle cores.)

  One caveat for a router that measures its own endpoints: the 4096 PREBUILT
  behaves differently from the self-exported bundles -- its decode falls ~30%
  from 18.9 t/s at 250 tokens of context to 13.0 at 3300, where the exports are
  flat. So a single decode-rate number per endpoint is only safe for a
  single-length export. Sample at a depth representative of the traffic, or
  record a rate per depth band.

  **Routing rule that falls out of this: send a request to the smallest window
  that fits it.** Do not treat a larger `n_ctx` as strictly better when ranking
  endpoints -- on this engine it is a latency class. Decode is roughly
  inverse-linear in the window to 8192 and worse beyond, so 8192 is the
  sensible default tier and 16384 earns its cost only for requests that
  genuinely cannot fit in 8192 -- and even then, a smaller endpoint plus
  server-side eviction/summarisation is often the faster answer.
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

- ~~Concurrent GPU + NPU contention has **not** been measured.~~ **Answered
  2026-08-23: 0.78x, and the cause is not the memory bus.** Numbers and caveats
  in the contention section above -- one pair of engines, one model, one depth,
  n=3, the GPU leg driven by `llama-bench` because `llama-server` cannot reach
  the Adreno.
- **Does `perf_profile` explain the 15% idle-residency penalty?** It is the one
  question the contention run leaves open, and it is one config edit away:
  lower `perf_profile` from `"burst"` in `htp_backend_ext_config.json` and
  re-run `bench_contention.py`. If burst power is the cause, a lower profile may
  return most of that 15% for some latency. Nobody has run it, so do not assume
  the penalty is fixed cost -- or that it is avoidable.
- Whether a second resident model raises the NPU's `1003` rate is unknown.
- Benchmark on a quiet box, and on a **cool** one. Thermals alone move the GPU
  leg **1.64x**: the same d469 measurement gave 11.04 with an unrelated export
  running, 16.86 with the box merely warm, and 18.05 cooled and clean. Sustained
  GPU load drove the clock to **48.9% of base**, and a sequential depth sweep
  taken across that decay produced a clean monotonic 19.65 -> 11.03 t/s that
  looked exactly like a depth effect and was not. Recovery takes ~2 minutes;
  `bench_contention.py` blocks on clock recovery before every sample for this
  reason. During an unrelated build here free memory hit **1.1 GB**, and any
  numbers taken then would have been meaningless.
