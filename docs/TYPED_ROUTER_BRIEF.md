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
| NPU (Hexagon) | Genie server | **855-938** | **18.55** | **single-flight**; ~4x the GPU's prefill, near-immune to host load |
| GPU (Adreno) | `llama-bench` -- see note | **226.8** | **18.05** | ties the NPU on decode on a quiet box, **-64% on a busy one**; not reachable over HTTP today |
| CPU (KleidiAI) | `llama-server` | fine | **22.6 @ d0, 13.2 @ d469** | NOT broken -- the 0.2 is retracted. Fastest at empty context, **slowest of the three at agent depth**; ~30% relative variance even on a quiet box |

Both NPU figures are corrected upward from the 277 / ~13 this brief carried
until 2026-08-24 -- see the correction note below. **Decode is a tie**: 18.55
against 18.05 is not a gap worth routing on. The two engines separate on prefill
and on host-load sensitivity, not on decode speed, and any ranking that calls
one of them "the faster decoder" is wrong in whichever direction it points.

Resident cost is ~3.3 GB per instance at 4k context, ~4.2 GB at 16k (KV is
`uint8`, ~72 KB/token, allocated for the whole compiled window at load) -- so
three instances fit in ~10-13 GB. **Capacity is not the constraint. Memory
bandwidth is.** This brief asserted that, retracted it on 2026-08-23 on the
strength of a measurement that turned out to have been taken on a misconfigured
bundle, and reinstates it here with numbers behind it -- see the concurrency
section.

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

**Where the NPU figures come from, and why both of them moved.** Corrected
2026-08-24; the old 277 / ~13 pair was wrong for two unrelated reasons.
**Prefill** was measured on a loaded box: on a quiet one the same 4096 bundle
gives **855-938 t/s** at d469, and a near-empty-context sweep gives a median
1157. **Decode** was measured against a bundle shipping `"poll": true`, whose
idle busy-wait burns 2.7 host cores; with `poll: false` the same bundle decodes
**18.55 t/s at d469**, against every ~13 t/s figure this brief has carried (13.2
in the table, 12.82 from the first contention run). The `poll` flag does not
touch prefill and the loaded box did not touch decode -- two corrections, two
causes. **Do not plan capacity against 277 / 13.2.**

That also dissolves part of a question this brief raised -- but only part, and
the remainder matters for routing. It read 18.0 t/s at near-empty context
against 12.82 at d469, and concluded NPU decode falls ~30% with depth. At d469
that conclusion is indeed dead: 18.0 at d~0 against 18.55 at d469 is flat within
noise, and the busy-wait explains the gap.

**But d0 and d469 are both shallow, and the effect is not there -- it is
deeper.** Swept across the whole window on the 4096 bundle with `poll: false`,
and INTERLEAVED (250, 3300, 250, 3300, 250, 3300) so that drift over the run
could not masquerade as depth:

| depth | 250 | 600 | 1200 | 2200 | 2900 | 3700 |
|---|---|---|---|---|---|---|
| decode t/s | 18.5 | 17.2 | 15.0 | 14.9 | 12.7 | 11.5 |

Interleaved, 250 gives 18.92 / 18.45 / 18.67 against 12.77 / 12.99 / 13.32 at
3300 -- tracking depth, not elapsed time, with the deep samples if anything
rising. The d469 readings (18.02 / 18.04 / 18.19) agree with the flat finding
exactly; the decline simply starts past ~600 tokens.

The distinction that actually predicts this is **prebuilt versus self-exported**,
not which window. The 4096 bundle is Qualcomm's prebuilt and pays for the
context in use; both self-exported bundles are genuinely flat across far wider
spans (16384: 3.26 t/s at 469 against 3.27 at 10532), paying for their whole
compiled window on every token. Why is unexplained -- see the window section.

For a router: **a rate measured at d469 will overstate the 4096 endpoint by
~40% on a long prompt.** Record the depth a rate was taken at, and do not
extrapolate a shallow sample across the window.

**The CPU leg is not broken -- but this is still a two-engine design, for a
different reason.** The ~0.2 t/s this brief carried was retracted on 2026-08-24.
It was never a property of the CPU backend, but the compound of a `poll: true`
Genie server busy-waiting on 2.7 host cores, concurrent benchmarks from other
sessions on this shared box, and a thread-count effect (all-cores default costs
2-5x on this hardware; use about half the cores).

What replaced it is not an endorsement. CPU is ~27% slower than the
accelerators at agent depth, is much the noisiest leg, and is *predicted* --
unmeasured -- to starve the GPU through the same host-core mechanism the
busy-wait demonstrated at 60%. So CPU stays out of the default pair on
evidence rather than on breakage. **CPU+NPU may well be fine**, since the NPU
proved insensitive to host load; that is a live open question, not a closed
exclusion.

Re-measured on a verified-quiet box, `--device none -t 6`, r=5, Qwen3-4B-Q4_K_M:
**22.57 +-6.74 t/s at d0** and **13.15 +-6.31 at d469**.

**Route on the depth-qualified ranking, never on a single number.** The order
inverts between an empty prompt and a realistic agent turn:

| | CPU | GPU | NPU |
|---|---|---|---|
| d0 | **~22.6** | 19.7 | ~18.5 |
| d469 | 13.2 | 18.05 | **18.55** |

The load-bearing number is the fall-off, not either endpoint: **CPU loses 42%
between d0 and d469 against the GPU's 8%.** That single comparison is why
"fastest decoder" and "broken at 0.2" are both wrong. CPU is
competitive-to-fastest on short prompts and the slowest of the three at the
depth an agent actually runs at, so a router that picks CPU off a d0 benchmark
picks wrong for real traffic.

Two caveats a dispatcher must encode. CPU variance is ~30% relative even on a
quiet box, far noisier than NPU or GPU, so a single sample is not a rate --
take a median or treat CPU as a range. And CPU decode falls off with depth
considerably harder than either accelerator, so any measured rate must carry
the depth it was taken at.

Provenance: measured independently by two sessions on this box on 2026-08-24.
Canonical write-up is **ADR 019, `17da11da` on YawLabs/typed master**
(`docs/adr/019-local-multi-engine-routing.md`; supersedes `3c99a1af`). A second
source recorded `t6 26.2 +-1.8, t12 11.9 +-5.2` at tg16 (d0), `30.2 / 6.2` at
tg8, and `pp512 t12 115`, corroborating the shallow end.

**The deep ranking is not measured, and should not be inferred from the table
above.** Past roughly 600 tokens BOTH accelerators fall, not just the CPU -- the
NPU from 18.5 to 12.8 by d2657-3300, and the GPU by an unknown amount, because
the only GPU sweep reaching d1024/d2048 was thermally confounded. So the d469
row is the deepest point with a clean measurement behind every engine. Routing
policy for long prompts is currently an extrapolation; treat it as one until
somebody sweeps all three deep on a clock-gated run.

One caveat on the poll finding, stated at the right size. The NPU-solo half is a
deliberate, controlled experiment and is multi-sourced: a flip across all three
windows gives 11.6 vs 18.0 t/s decode and 267.1% vs 0.0% idle CPU on a server
that had answered nothing but `/health`, independently reproducing another
session's 12.82 -> 18.55 and 2.8-core spin. What rests on inference is only the
attribution of the CONCURRENCY flip (0.78x -> 1.45x) to the same cause -- the
GPU's solo rate being identical across both configurations is what makes that
inference a strong one, but it is not an A/B. Closing it means re-running the
contention benchmark against a `poll: true` bundle at matched conditions (d469,
n=3, Q4_K_M on the GPU leg via `llama-bench`, poll value recorded in the
output, and clock-gated -- the harness gate was dead code until `f3cd053`, so
that run should use the fixed version and additionally sample the clock DURING
each measurement, not only at entry).

## Concurrency measured: 1.45x (corrected 2026-08-24)

**This section reported 0.78x and "it is a net loss" until 2026-08-24.** That
measurement was taken against bundles shipping `"poll": true`, where an idle
Genie server busy-waits on 2.7 host cores -- cores the OpenCL backend needs to
dispatch a kernel every token. It was measuring CPU starvation, not contention.
With `poll: false` the pair is a **1.45x gain**. Both configurations are below,
because the losing one is what a bundle does out of the box.

Both engines hot, same model on each leg, decode at d469, n=3.

**Provenance correction 2026-08-24, stated at the size the evidence supports.**
This paragraph credited `bench_contention.py` with gating every sample to >=92%
of base clock. Two things are now clear and they point in different directions.

*Verified:* the harness cannot gate anything. It defines `wait_for_cool()` and a
`cool_floor` parameter, but neither `measure()` call site passes it and no CLI
flag can set it, so the gate has never executed. Confirmed by reading both call
sites. That is a real defect for whoever runs it next -- fixed in `f3cd053`.

*Verified, for the figures carrying a stddev:* `bench_contention.py` contains
zero `+-` format specifiers, so it cannot have emitted `18.05 +-0.13`,
`13.47 +-0.21`, `7.27 +-0.72` or `22.57 +-6.74`. Those are `llama-bench`'s
stddev column. Checkable from the source in this repo without trusting anyone's
account.

*Still attested, and it is a narrower set than the argument above covers:* the
bare figures -- NPU 18.55 / 13.35 and the retention percentages -- carry no
stddev, and the harness's own vocabulary includes
`pair: %.2f -> %.2f t/s (keeps %.1f%%)`. So the format fingerprint does NOT
exclude it for those, and their provenance rests on the primary source's
account rather than on artifacts. Nor is the gating itself checkable at all:
the harness writes JSON only when asked and under a caller-chosen name, so no
output files existing is not evidence either way, and the Bash tool runs
non-interactive shells that never write `~/.bash_history`, so its silence is
another absence that proves nothing.

**The caveat that survives either account, and the one worth encoding:** a gate
tests the clock *before* a sample and says nothing during it. A `llama-bench -r
3` run takes one to two minutes, and sustained load takes this box to 48.9% of
base, so a figure can be gated at entry and still decay through its own
measurement. That applies equally to gated and ungated runs, and equally to the
numbers elsewhere in this brief -- the window-tax measurements were taken on a
verified-quiet box but were never clock-gated at all.

So: treat the 1.45x-vs-0.78x ratio as sound (a between-configuration difference
measured the same way on both sides, with the GPU's solo rate identical across
them), and treat every absolute rate here -- retention percentages included --
as gated at entry at best and unmonitored throughout. Not wrong; bounded.

**`poll: false` -- correct configuration:**

| | solo | contended | retains |
|---|---|---|---|
| NPU | **18.55** t/s | **13.35** | 72% |
| GPU | **18.05** t/s | **13.47** +-0.21 | 75% |

Aggregate **26.82 t/s** against **18.55** for the best single engine: **1.45x**,
and **73% of the additive ideal** (36.60).

**`poll: true` -- the shipped default:**

| | solo | contended | retains |
|---|---|---|---|
| NPU | 12.82 t/s | 6.88 | 54% |
| GPU | 18.05 t/s | 7.27 +-0.72 | 40% |

Aggregate 14.15 t/s against 18.05: **0.78x, a net loss**, at 46% of that run's
additive ideal (30.87). The GPU's solo rate is identical in both tables --
nothing about the GPU changed. The busy-wait costs about half the pair's
throughput and inverts the verdict.

**And it is the bus.** This brief retired the bandwidth premise on 2026-08-23;
restore it. With `poll: false` the additive demand is **84.5 GB/s** and the pair
actually draws **62.0 GB/s** -- 63% and 46% of the 135.2 GB/s theoretical peak.
But peak is the wrong yardstick: this silicon streams ~105-115 GB/s in practice,
so 84.5 is roughly **75-80% of the achievable ceiling**, and giving up 27%
against additive at that loading is ordinary bus contention rather than
something needing a new mechanism. (The `poll: true` run drew 71.4 additive /
32.7 actual, and *that* is what made bandwidth look refuted: losing more than
half your throughput at a quarter of peak is not a bandwidth story, and it was
not one.) Worth noting because it is the only prediction here that preceded its
measurement: a bandwidth model gave **1.48x** for this pair before these numbers
existed, and the measurement came back **1.45x**.

**There is no idle-residency penalty under `poll: false`, and the power
hypothesis is withdrawn.** This brief reported that an NPU server merely loaded
and idle cost the GPU 15% (17.94 -> 15.26 t/s), said neither bandwidth nor CPU
could explain a penalty from a zero-core process, and floated a shared package
power budget (`perf_profile: "burst"`, `rpc_control_latency: 100`). Re-measured
2026-08-24 against a server verified clean -- it bound the port itself, served
real inference, `poll: false` in its config, 0.00 idle cores over 15 s -- the
GPU at d469 decodes **18.03 +-0.08** t/s with the NPU server resident and idle,
against 17.94 / 18.05 / 18.10 across three cooled runs with it stopped. **That
is inside the noise: a hot spare costs nothing.** The old 15%, and an earlier
31% (18.05 -> 11.57), were both `poll: true`-era, where "idle" meant 2.7
spinning cores against a GPU that gives up 64% of its rate to host load. **Do
not carry the power story forward.**

**Host load hits the two engines completely differently.** Under an unrelated
~20% CPU job versus a cooled box: NPU **-1.2%**, GPU **-64%**. Hexagon has its
own clock domain; the OpenCL path is host-dispatch-bound per token. typed's
local tier runs on a developer machine that is usually compiling or running
tests, so **a router should not rank these engines off idle-box numbers** -- the
GPU's 18.05 is a best case that a build running in another window erases, while
the NPU's rate barely moves. (The absolute pair behind that -1.2%, 12.67 against
12.82, is `poll: true`-era; the ratio is the finding and its mechanism does not
depend on the flag.)

**Caveat: the poll comparison was not a controlled experiment.** The value was
changed on disk, by a third party, at 2026-08-24 02:23 -- between the two halves
of this measurement. Which half a given sample belongs to is inferred from that
file mtime and from server start times, not from a variable held under control.
Confirming it deliberately is about fifteen minutes (the `.orig` bundle configs
still carry the shipped `true`: flip, run, flip back, run) and **has not been
done**. Weight the result accordingly: the direction is not in doubt, the
attribution to `poll` is well-supported but inferred.

Three rules fall out for the router:

1. **Check `poll` before trusting any local concurrency number, including
   these.** `"poll": true` is the shipped default and it turns 1.45x into 0.78x.
2. **Fanning across both engines is worth ~1.45x, not 2x.** Bandwidth is the
   ceiling and the pair already draws three-quarters of what this memory system
   delivers, so plan for diminishing returns and do not assume a third engine
   adds a third.
3. **Route to a second engine for concurrency and failover as well as for
   speed** -- a queued request served at 72-75% of solo rate beats one waiting
   behind the single-flight lock.

Reproduce any of this with `src/bench_contention.py` in `snapdragon-npu-llm`
(`--npu` / `--gpu` base URLs, `--depth`, `--repeat`; it refuses to run on a
loaded box unless you pass `--allow-loaded`, which stamps every result LOADED).
Note that it expects both legs over HTTP, so the GPU leg needs the
`llama-server` problem above solved first, or driving by hand.

## Why route at all

**The NPU serves exactly one request at a time.** Concurrent Hexagon access
wedges the device, so the server serializes behind a lock. A second engine gives
you concurrency, somewhere to go when the HTP throws its transient `Code 1003`
device fault, and -- corrected 2026-08-24 -- **throughput as well**: 1.45x for
the pair, bounded by memory bandwidth rather than by either engine. The "what it
does not give you is throughput" line this section carried was measured on a
misconfigured bundle and is withdrawn.

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
5. **Engine configuration, not just engine selection.** One setting dominates
   everything else on this hardware: `"poll": false` in the bundle's
   `genie_config.json`. Shipped as `true` it busy-waits on 2.7 cores while idle,
   costs up to 55% of NPU decode, and turns concurrent GPU + NPU serving from a
   1.45x gain into a 0.78x loss. If typed ever manages these bundles, assert the
   flag rather than trusting the vendor default.
6. **Engine lifecycle -- and a hot spare is free.** This brief recommended
   stopping an idle engine rather than parking it hot, on the strength of a 15%
   penalty a merely-resident NPU server imposed on the GPU. That penalty was
   the busy-wait. Measured under `poll: false`, an idle resident NPU server
   costs the GPU nothing -- 18.03 t/s against ~18.0 stopped -- so **parking an
   idle engine hot is fine.** Keep it warm and spend the load time only when
   you have another reason to. The old advice ("start the engine the route
   picks and stop the one it does not") described a misconfigured bundle and is
   withdrawn.

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
  bundle appears to behave differently from the self-exported ones -- its decode
  fell ~30%, from 18.9 t/s at 250 tokens of context to 13.0 at 3300, where the
  exports are flat. **Treat that as unconfirmed.** The same shape turned up on a
  self-export (18.0 at d~0 against 12.82 at d469) and proved to be the
  `poll: true` busy-wait rather than depth; the prebuilt sweep was taken in the
  same era and has not been repeated. Until it is, sample at a depth
  representative of the traffic or record a rate per depth band -- sound advice
  whether or not the falloff is real.

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

- ~~Concurrent GPU + NPU contention has **not** been measured.~~ ~~Answered
  2026-08-23: 0.78x, and the cause is not the memory bus.~~ **Answered
  2026-08-24: 1.45x, and the cause is the memory bus.** Numbers and caveats in
  the concurrency section above -- one pair of engines, one model, one depth,
  n=3, the GPU leg driven by `llama-bench` because `llama-server` cannot reach
  the Adreno.
- **Confirm the `poll` A/B deliberately.** The whole reversal above rests on a
  flag a third party changed on disk between the two halves of the measurement,
  not on a controlled experiment. It is ~15 minutes -- flip it back, re-run
  `bench_contention.py`, flip it forward, re-run -- and nobody has done it.
  Until then the 1.45x is well-supported but its attribution is inferred.
  **When you do it, verify the server you launched is the one answering.** The
  launcher refuses a port something else already holds, so a readiness check
  that merely curls the port can pass against the PREVIOUS process and hand you
  samples labelled with a config they were never served under. That is exactly
  how this brief came to believe in a 15% idle penalty and a package power
  budget. Check the launcher exit status and the PID owning the port, not just
  that something answers.
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
