# Running NPU + GPU + CPU instances concurrently

Design note for serving the same model on several engines at once, and where
that work belongs. Nothing here is implemented; the measurements are.

## The memory model, corrected

The obvious framing -- "16 GB for the NPU, 16 GB for the GPU, 16 GB for the
CPU" -- does not describe this hardware. Measured on this box:

| | |
|---|---|
| Total physical | **31.6 GB, one pool** |
| Adreno X1-85 `AdapterRAM` | **not reported / shared** -- no dedicated VRAM |

CPU, GPU and NPU all allocate from the same LPDDR5x. There is no per-engine
budget to divide, and 16+16+16 asks for 48 GB on a 31.6 GB machine.

The good news is that the real per-instance cost is nowhere near 16 GB:

| | weights | KV cache | resident |
|---|---|---|---|
| one instance @ 4k ctx | ~3.0 GB | ~0.29 GB | **~3.3 GB** |
| one instance @ 16k ctx | ~3.0 GB | ~1.18 GB | **~4.2 GB** |

Three instances land around **10 GB at 4k** or **13 GB at 16k**, which fits
comfortably. **Capacity was never the constraint.**

KV is **~72 KB/token, not 144** -- corrected 2026-08-23 by reading the
bundle's own `metadata.json` rather than assuming. `past_key_0_in` /
`past_value_0_in` are `dtype: uint8` with a fixed quant scale, so the arithmetic
is 36 layers x 2 x 8 KV heads x 128 head_dim x **1 byte** = 73,728 B/token. The
earlier figure assumed fp16 and was exactly 2x high. Confirmed against the
HTP allocator: the 4096 bundle reports 343,933,440 bytes across 8 buffers and
the 16384 bundle 1,253,048,832 -- a delta of 73,983 B per extra token of
window, 0.3% off the uint8 prediction.

**And that KV is allocated for the whole COMPILED window up front, not as the
context fills**, because the KV tensors are statically-shaped graph inputs.
That makes window size a throughput knob, not just a memory one. Three windows
of the same model, measured with `poll: false` on a quiet box: **18.0 t/s at
4096, 8.8 at 8192, 3.3 at 16384** -- at identical, nearly empty context. Decode
is about inverse-linear in the window to 8192 (2.05x cost per doubling) and
worse past it (2.69x), so 8192 is the sweet spot and 16384 is a specialist
tier. See the window-tax note in `GENIE_SERVER.md`; it is the single
most important number for sizing a multi-engine deployment, because a bigger
window costs every request rather than only the long ones.

## An idle NPU server was stealing 2.7 cores

Measured 2026-08-24, and it lands squarely on this document's premise. The
Genie bundle's QnnHtp block ships `"poll": true`, which busy-waits: a resident
`genie_server` with no requests in flight burned **270% CPU -- 2.7 cores --
doing nothing**. Setting `poll: false` in `genie_config.json` takes that to
**0%** and makes the NPU *faster* (decode +55% at a 4096 window, +11% at 8192,
+2% at 16384; prefill up; run-to-run noise down 8x). Nothing measured got worse.

Three consequences for the multi-engine design here:

- **The CPU and GPU legs were competing with a spinning NPU server.** Any
  measurement of another engine taken while the NPU server was resident is
  pessimistic by up to 2.7 cores of stolen CPU. That includes the baselines
  below.
- **The 0.2 t/s CPU anomaly was exactly this, and is RETRACTED (2026-08-24).**
  It was never a property of the CPU backend: it was the compound of a
  `poll: true` Genie server busy-waiting on 2.7 host cores, concurrent
  benchmarks from other sessions on this shared box, and a thread-count effect
  (the all-cores default costs 2-5x on this hardware -- use about half). See the
  baselines table below for the re-measured figures.
- **It invalidates the first concurrency run outright.** That measurement was
  taken across the flag change, and its `poll: true` half had a spinning NPU
  server stealing the host CPU the OpenCL backend needs to dispatch a kernel
  every token. That half produced a 0.78x "net loss" and a conclusion that the
  memory bus was not the constraint. Neither survived repeating the same
  measurement with `poll: false`: the corrected result is a 1.45x **gain**, and
  the bus **is** the constraint. See the section below.

**RESOLVED 2026-08-24, and the reconciliation guessed at below is the right
one.** The contention run sampled a resident `genie_server` for 30 s with
nothing in flight and measured **0.1% of one core**, against 270% here, and
proposed that the two accounts could only agree if the low sample was taken
with `poll: false` already in force. That is exactly what happened: the
bundles on disk were switched to `poll: false` on 2026-08-24, so any sample
after that reads ~0% whatever the history.

The objection to the original 270% was also fair -- it was sampled just after a
benchmark, so "still draining" could not be ruled out from that measurement
alone. It survives a proper control. Same bundle, same 29 threads, same box,
30 s samples, the only variable being `poll`:

| condition | idle CPU |
|---|---|
| `poll: true`, has NEVER served a generation | **267.1%** |
| `poll: false`, has NEVER served a generation | **0.0%** |
| `poll: false`, 30 s after one generation | **0.1%** |

A server that has answered nothing but `/health` still burns 2.7 cores with
`poll: true`, so it is an idle spin and not drain. The 0.1% reading reproduces
exactly on the `poll: false` side.

What this *does* change is every number taken against a `poll: true` bundle --
which is how they ship. Each was paying 2.7 cores that had nothing to do with
the engine under test, and on the GPU leg that is not a rounding error: the
OpenCL path is host-dispatch-bound per token and gives up 64% of its decode rate
to an unrelated ~20% CPU job. The 15% penalty this file recorded for a *merely
resident* NPU server was the busy-wait as well -- re-measured with
`poll: false` it is **zero**. So was the NPU's own ~13 t/s decode baseline,
which is really 18.55. The section below has been re-measured too and now reads
**1.45x**, not the 0.78x it reported before this finding landed.

## Running both engines is a 1.45x gain, and bandwidth is why

**Corrected 2026-08-24. Read this if you saw the earlier version.** This section
was headed *"Running both engines is a net loss, and bandwidth is not why"*,
reported **0.78x**, and explicitly retired the memory bus as the mechanism. The
numbers were real, but they were taken against bundles carrying `"poll": true`
while a busy-waiting NPU server burned 2.7 host cores throughout -- cores the
OpenCL backend needs, per token, to dispatch its kernels. That run measured CPU
starvation wearing contention's clothes. Re-measured with `poll: false` the pair
is a **1.45x gain**, and the bandwidth model the previous revision threw out is
the thing that predicts it. Both configurations are kept below, because the
wrong one is what you get if you run a bundle as it ships.

Same model on both legs (Qwen3-4B: Genie w4a16 4096 bundle on the HTP, Q4_K_M
2.32 GiB GGUF on the Adreno), decode at context depth 469, n=3, every sample
gated to >=92% of base clock before it is taken. `src/bench_contention.py`
drives both engines at once and confines sampling to the overlapped stretch.

**`poll: false` -- the current, correct configuration:**

| | solo | contended | retains |
|---|---|---|---|
| NPU @ d469 | **18.55** t/s | **13.35** | 72% |
| GPU @ d469 | **18.05** t/s | **13.47** +-0.21 | 75% |

Aggregate with both hot is **26.82 t/s** against **18.55** for the best single
engine -- **1.45x** -- and **73% of the additive ideal** (36.60). A quarter of
the theoretical gain goes to running them together; the rest is real.

**`poll: true` -- what you get if you leave the bundle as shipped:**

| | solo | contended | retains |
|---|---|---|---|
| NPU @ d469 | 12.82 t/s | 6.88 | 54% |
| GPU @ d469 | 18.05 t/s | 7.27 +-0.72 | 40% |

Aggregate **14.15 t/s** against 18.05 -- **0.78x, a net loss** -- at 46% of that
run's additive ideal (30.87). Note that the GPU's *solo* rate is identical
across the two tables: nothing about the GPU changed. What changed is that the
NPU's solo rate rose 45% and the two stopped fighting over the host CPU while
contended. **The busy-wait costs roughly half the pair's throughput and turns a
1.45x win into a 0.78x loss.** It is the highest-leverage line of configuration
on this page.

**The bus is the constraint after all.** The previous revision retired the
bandwidth premise on the strength of the second table. Restore it.

| | bandwidth | % of 135.2 GB/s peak |
|---|---|---|
| `poll: false`, additive ideal | 84.5 GB/s | 63% |
| `poll: false`, actual both hot | **62.0 GB/s** | **46%** |
| `poll: true`, additive ideal | 71.4 GB/s | 53% |
| `poll: true`, actual both hot | 32.7 GB/s | 24% |

The theoretical peak is the wrong yardstick. This silicon streams roughly
**105-115 GB/s** in practice, so the additive demand of 84.5 GB/s is about
**75-80% of the achievable ceiling** -- and a 27% shortfall against additive at
that loading is ordinary bus contention, not an anomaly wanting a novel
mechanism. It was the `poll: true` row that made bandwidth look refuted: two
engines each losing more than half their throughput while together drawing a
quarter of peak genuinely is inexplicable as a bandwidth story. It was never a
bandwidth story. It was CPU starvation.

Worth recording plainly, because it is the only prediction on this page made
before its measurement existed: **the bandwidth model gave 1.48x for this pair,
and the measurement came back 1.45x.** That model was written down, discarded on
the `poll: true` data, and is now the closest thing here to something validated.

**The idle-residency penalty was the busy-poll, and under `poll: false` there
is none.** This section previously reported that a merely-resident,
zero-inference NPU server cost the GPU 15% (17.94 -> 15.26 t/s), argued that
neither bandwidth nor CPU could explain a penalty imposed by an idle process,
and floated a shared package power budget -- `perf_profile: "burst"`,
`rpc_control_latency: 100` -- as the surviving hypothesis. The premise and the
hypothesis both go. Re-measured 2026-08-24 against a server verified clean --
it bound the port itself, served real inference, `poll: false` in its config,
0.00 idle cores over 15 s -- with the GPU at d469, n=3, cooled to >=92% of base:

| GPU @ d469 | rate |
|---|---|
| NPU server **resident but idle** | **18.03** +-0.08 t/s |
| NPU server stopped | 17.94 +-0.07 / 18.05 / 18.10 (three cooled runs) |

**There is no idle-residency penalty.** 18.03 against ~18.0 is inside the
noise, so an idle NPU server is free to leave resident once the busy-poll is
off -- park it, do not stop it. The 15% (17.94 -> 15.26) and an earlier 31%
(18.05 -> 11.57) were both `poll: true`-era, where "idle" meant 2.7 spinning
cores against a GPU leg that gives up 64% of its rate to host load. **The power
hypothesis is withdrawn**: there is no longer an anomaly for it to explain, and
a speculative mechanism should not outlive the thing it was invented for.

**Caveat: the poll comparison was not a controlled experiment.** The value was
changed on disk by someone else at 2026-08-24 02:23, between the two halves of
this measurement. Which half a given sample belongs to is *inferred* -- from
that file mtime and from server start times -- not from a variable held under
control. The `.orig` backups still carry the shipped `true`, so confirming the
whole result deliberately is about fifteen minutes: flip the flag back, re-run
`bench_contention.py`, flip it forward, re-run. **That was done on 2026-09-03 --
see the controlled A/B below**, which is where the net-loss half of this
section's conclusion was refuted. Read this inferred split as the weaker
evidence it is; the A/B supersedes it.

Two things follow for the design, and they cut back the other way from the
previous revision:

- **A second hot engine is worth ~1.45x -- if you configure for it.** The gain
  is real but it is not additive, and it does not merely shrink when the bundle
  ships `"poll": true`: it inverts. Check that flag before quoting any
  concurrency number, yours or anyone else's.
- **Bandwidth is the ceiling to plan against.** At 84.5 GB/s of additive demand
  the pair already sits at three-quarters of what this memory system delivers,
  so a *third* engine has very little headroom left to divide, whatever its
  compute. That is a better-founded reason to be sceptical of a three-engine
  design than the one this file gave a revision ago.

Measured single-engine baselines (prefill / decode, tokens/sec):

| engine | prefill | decode | notes |
|---|---|---|---|
| NPU (Genie / QnnHtp) | **855-938** @ d469 | **18.55** @ d469 | `poll: false`, quiet box 2026-08-24; supersedes 277 / 13.2 |
| GPU (Adreno) | **226.8** @256 | **18.05** @ d469 | quiet box 2026-08-23; supersedes 117 / 6.0 |
| CPU (KleidiAI) | ~115 @ pp512 | **22.57** @ d0, **13.15** @ d469 | NOT broken -- the 0.2 is RETRACTED. Quiet box 2026-08-24, `--device none -t 6`, r=5, Qwen3-4B-Q4_K_M; +-6.7 / +-6.3 |

The **GPU** row is a 3.0x correction on decode and roughly 1.9x on prefill. The
retired 117 / 6.0 pair was taken in the same loaded window as the NPU's 277, and
paid a resident NPU server on top.

The **NPU** row is a **~43% correction on decode** and a ~3.3x correction on
prefill, and the two have different causes. Every ~13 t/s decode figure this
file has carried -- 13.2 in the old table, 13.0 from the prebuilt sweep, 12.82
from the first contention run -- was measured against a `poll: true` bundle and
carries the busy-wait. Prefill moved for the unrelated reason that the original
277 sweep was taken on a loaded box; the `poll` flag does not touch prefill, and
855-938 stands as measured.

The **CPU** row has now been re-measured on a verified-quiet box and the ~0.2
is withdrawn. Three things were folded into that figure and only one of them
was the CPU: the busy-waiting NPU server above, other sessions benchmarking the
same box, and a thread-count effect -- the all-cores default costs 2-5x here, so
the re-run used `-t 6` on a 12-core part. What replaced it is not an
endorsement, though. CPU loses **42%** between d0 and d469 against the GPU's 8%,
so it is competitive-to-fastest on an empty prompt and the slowest of the three
at the depth an agent actually runs at. It is also much the noisiest leg
(~30% relative variance on a quiet box), so a single sample is not a rate.
**A router that ranks CPU off a d0 benchmark picks wrong for real traffic.**

**The two engines are near-identical decoders.** 18.55 against 18.05 at the same
depth is not a gap worth routing on, which means neither "the NPU is the fastest
engine" nor "on a cooled box the Adreno decodes faster than the NPU" is true --
and this file has asserted both, in that order, inside two days. What actually
separates them is two other axes:

- **Prefill**, where the NPU is about **4x** faster: 855-938 against 226.8.
- **Host-load sensitivity**, where the NPU is nearly immune and the GPU is not.
  Under an unrelated ~20% CPU job the NPU gave up **1.2%** and the GPU **64%**.
  Hexagon has its own clock domain; the OpenCL path is host-dispatch-bound per
  token. (The absolute pair behind that -1.2% -- 12.67 against 12.82 -- is
  `poll: true`-era and superseded. The ratio is the finding, and its mechanism
  does not depend on the flag.)

So the split is prefill-heavy versus decode-heavy, and busy box versus quiet
box. It is not fast versus slow.

**And NPU decode is flat with depth on the 4096 export after all.** This file
flagged a ~29% falloff -- 18.0 at near-empty context against 12.82 at d469 --
and called it worth resolving before either number was quoted. It is resolved:
the 12.82 was a busy-wait artefact, the corrected d469 rate is 18.55, and 18.0
at d~0 against 18.55 at d469 is flat within noise. That matches the 16384
bundle, which was flat all along (3.26 at d469, 3.27 at d10532). The GPU by
comparison loses about 8% over the same span (19.7 -> 18.05).

**This is still a two-engine design, but on evidence rather than on breakage.**
The "CPU is broken at 0.2 t/s" reason is gone. What replaces it is narrower and
better founded: CPU is ~27% slower than either accelerator at agent depth, is
the noisiest of the three, and the two working engines already draw about
three-quarters of this memory system's achievable bandwidth -- so a third
instance *of the same 4B model* divides a nearly-full bus.

That last clause is the one worth reading carefully, because bandwidth demand
scales with weight bytes per token rather than with engine count. A third leg
running a SMALLER model is a different proposition and has not been measured.
**CPU+NPU specifically may well be fine**, since the NPU proved insensitive to
host load; what is predicted -- and unmeasured -- is that CPU would starve the
GPU through the same host-core mechanism the busy-wait demonstrated at 60%.
Open question, not a closed exclusion. CPU also still matters far beyond this
box: it is the fallback every non-Snapdragon user lands on.

## The controlled poll A/B (2026-09-03) -- and a measurement bug it exposed

The reversal above was never a controlled experiment: a third party flipped
`poll` on disk between the two halves, and which half a sample belonged to was
inferred from file mtimes. Run deliberately now -- same box, same two engines,
same command, flag flipped between arms, nothing else touched:

| | `poll: false` | `poll: true` | ratio |
|---|---|---|---|
| NPU solo | **18.46** t/s | 13.52 | 1.37x |
| NPU contended | 17.00 (keeps 90.4%) | 12.23 (keeps 94.7%) | |
| GPU solo | 18.20 | 18.47 | **1.00x** |
| GPU contended | 14.37 (keeps 79.2%) | 11.04 (keeps 64.0%) | |
| **aggregate, both hot** | **31.37 t/s** | **23.27 t/s** | **1.35x** |
| vs best single engine | 1.70x | 1.26x | |

**What is confirmed.** `poll: false` is worth **1.35x on aggregate
throughput**, and the NPU's own solo rate is 1.37x -- squarely inside the
1.45x/1.55x this repo has claimed. The arm assignment is no longer inferred:
`poll: true` was verified live by its own signature, **291.6% CPU (2.9 cores)
burned while completely idle**, against 0% on the other arm. That is the
control the 08-24 measurement lacked.

**What is REFUTED: the net loss.** This file and the router brief both say
`poll: true` turns concurrency into a **0.78x loss**. It does not reproduce.
Both arms are a GAIN over the best single engine -- 1.70x and **1.26x**. Two
hot engines are worth running under either setting; `poll: true` just wastes
about a quarter of the win. Prefer the aggregate row above to the
"speedup vs best single engine" column, for exactly the reason this file
already gives: that denominator moves with the variable under test.

**The GPU solo row is the internal control, and it also corrects a claim.**
18.20 against 18.47 is unchanged, confirming "nothing about the GPU changed".
But that same row is measured with an IDLE `poll: true` NPU server resident,
so the reported 25-32% penalty an idle busy-wait imposes on the GPU **does not
appear here either** -- 2.9 spinning cores cost the GPU leg nothing measurable
on this 12-core part. The busy-wait's cost shows up when the NPU is
GENERATING (GPU keeps 79.2% vs 64.0%), not when it merely sits there.

**The measurement bug, which is the reason this took three attempts.** The
first two runs came back flagged SUSPECT in OPPOSITE directions (box "got
faster", then "decayed monotonically"), with NPU solo spanning 15.99-29.32
t/s. It was not thermal. `measure_decode` runs the same prompt at 1 token and
at 1+N and subtracts, so prefill cancels -- but **only if both calls land on
the same compiled graph.** At the default depth 500 with 120 tokens, the
1-token call sits in `cl512` and the 120-token call in `cl1024`: two different
graphs, so the prefill does not cancel and the difference is garbage. Inside a
single graph the same engine is rock steady -- 17.91 / 18.13 / 17.70 at d250,
15.13 / 15.07 / 14.88 at d1082, same-depth noise 0.43 t/s. Moving the A/B to
d250 (250 + 120 = 370, entirely inside `cl512`) produced a clean run on the
first try.

**So: on a multi-length bundle, choose a depth where prompt + generated
tokens stay inside ONE compiled length.** This is the same trap this repo
already documented once, from the other side -- a boundary sweep that forgot
the generation and measured a plateau. It applies to the historical numbers
too: d469 + 120 = 589 crosses 512 on the 4096 prebuilt AND on the 8192-multi,
so the 1.45x and every d469 figure taken with this harness carries it. That
does not overturn them -- the arms shared the confound -- but it explains
their run-to-run spread, and new work should not repeat it.

Caveat carried: the `poll: true` arm still tripped the drift check (+12.6%,
box got faster), so its retention percentages are flattered; the aggregate
gap it sits inside is 35%, far larger than that drift, so the direction is
safe. The clock dipped into the 47-64% band during every run on this box,
as it always does under sustained load.

## Why it is still worth doing

Not for memory -- capacity was never the constraint -- but, now measured, for
throughput *and* concurrency. Two reasons stack.

The first is **throughput: 1.45x** for the pair, at 73% of the additive ideal.
That is not the 2x a naive reading hopes for, and it needs `poll: false` to
exist at all, but it is a gain. The "not for throughput at all" verdict this
section carried on 2026-08-23 is withdrawn -- it was measured against a spinning
NPU server.

The second is that **the NPU is single-flight**: concurrent HTP access wedges the
device, so `genie_server` serializes every request behind a lock. One instance
serves exactly one request at a time.

A second engine buys:

- **Throughput** -- 26.82 t/s aggregate against 18.55 for the best single
  engine. Bounded by the memory bus rather than by either engine, which is also
  why a third is unlikely to buy much.
- **Concurrency** -- a second request stops queueing behind the first. It runs
  slower than it would alone (each engine retains 72-75% while the other is
  hot), but it runs instead of waiting.
- **Failover** -- somewhere to route when the HTP throws its transient
  `Code 1003`, which is a device-state fault no amount of retrying fixes.
- **Tiering** -- but not on decode rate, in either direction. The two decode
  within 3% of each other (18.55 NPU, 18.05 GPU at d469). They separate on
  **prefill**, where the NPU is ~4x faster, and on **host load**, where the NPU
  gives up 1.2% to an unrelated ~20% CPU job and the GPU gives up 64%. On a
  developer machine that is usually compiling or running tests -- typed's actual
  deployment -- the NPU's flatness is worth more than the GPU's idle-box peak.

## Shape: one Genie server, N llama-servers, router in typed

The three "instances" are not three copies of this server:

- **NPU** -> `genie_server.py` (this repo), Genie context binary on the HTP.
- **GPU / CPU** -> `llama-server`, which typed's free tier already runs.

**Read this before you try to serve the GPU leg over HTTP.** `llama-server`
from `llama-qnn-fork/build-3way` **cannot drive the Adreno**. It silently loads
on CPU -- `kleidiai`, `n_threads=12`, zero OpenCL init -- even with
`-ngl 99 --device GPUOpenCL --fit off`, and `--list-devices` prints nothing at
all in that build. There is no error to notice; the request is answered, just by
the wrong engine at CPU speed. `llama-bench` from the *same* directory against
the *same* DLLs drives the Adreno correctly, which is why the contention
measurement above had to drive the GPU leg through `llama-bench` and only the
NPU leg over HTTP.

**RESOLVED 2026-09-03: a build now ships whose `llama-server` initialises
OpenCL.** `llama-qnn-fork/build-arm64-windows-llvm-release` (build 10672)
serves the Adreno over HTTP -- `using device GPUOpenCL`, `offloaded 33/33
layers to GPU`, GPU engine counter 59% during decode, smoke-verified through
`src/run-llama-server.ps1 -Leg gpu` (see `MODEL_OPTIONS.md`). The build-3way
failure above is a property of THAT build, not of llama-server -- and the
advice survives the fix: verify the backend line in the server's own startup
log before trusting any GPU number taken over HTTP, because the failure mode
is silent.

So most of it exists. The missing piece is dispatch, and **that belongs in
typed, not here**:

- typed already selects backends and already probes these endpoints -- `/props`
  for the context window, `probeLocalToolCalls` for tool support. The
  capability-negotiation machinery is there.
- `genie_server` is deliberately one model on one engine. Turning it into a
  multi-engine router would conflate serving with dispatch and duplicate what
  typed does.
- The engines differ in ways only the client can weigh: context window, tool
  support, grammar support, latency, quality.

### What typed would need

1. A registry of local endpoints with capabilities (n_ctx, tools yes/no,
   measured decode rate).
2. Routing by request shape -- needs tools? needs a long window? latency
   sensitive?
3. Failover on backpressure. **This signal already exists:** `genie_server`
   returns `429` (OpenAI) / `529` (Anthropic) with
   `"server busy; NPU is single-flight"` once its small queue is full. It was
   not built for routing, but it is exactly what a dispatcher needs to shed to
   the next engine.
4. ~~Health checks that survive the HTP wedge~~ -- **done 2026-08-24.**
   `/health` used to be a liveness ping, and the objection here was right: it
   answering was not proof the device would execute, since `1003` fails at
   execute time rather than at load. It now reports engine state and returns
   503 when the engine cannot serve, deliberately touching nothing on the
   engine so it still answers while a wedged thread holds the lock. A stall is
   aborted; if that does not take, the process exits 75 and the launcher
   restarts it. A dispatcher can treat 503 as "shed to another engine" and 200
   as a real capability claim.

## Answered: does concurrent GPU + NPU inference hold up?

**Yes -- 1.45x, on a correctly configured bundle.** Measured 2026-08-24 with
`src/bench_contention.py`: **26.82 t/s** aggregate against **18.55** for the best
single engine, 73% of the additive ideal, drawing 46% of peak bandwidth against
an additive demand of 63%. The mechanism is the memory bus, and the full numbers
are in the contention section above.

**This page answered "no" on 2026-08-23** -- 0.78x, a net loss, bandwidth ruled
out. That answer was taken against `"poll": true` bundles whose idle busy-wait
stole 2.7 host cores from the GPU's per-token dispatch path. It is superseded outright: the
controlled A/B measured that same shipped configuration at 1.26x over the best
single engine -- still a gain. The net loss does not reproduce under any
setting tested.

Caveats that travel with the corrected answer: one pair of engines, one model,
one depth (d469), n=3; the GPU leg was driven by `llama-bench` rather than over
HTTP, for the reason in the section above; the aggregate combines two
experiments, each measuring one engine precisely while the other was driven,
because the two legs need different harnesses; and the `poll` comparison itself
was not a controlled A/B -- see the caveat in the contention section. Both
directions were measured and retention is roughly symmetric (72% NPU, 75% GPU).

## Open questions

- ~~**Confirm the `poll` comparison deliberately.**~~ **DONE 2026-09-03, and
  the answer splits in two.** `poll: false` is confirmed better -- but the
  *inversion* is not. See "The controlled poll A/B" below.
- ~~**Why is CPU decode 0.2 t/s?**~~ **Answered 2026-08-24: it was not.** The
  figure was an artifact of a busy-waiting NPU server, co-tenant benchmarks and
  an all-cores thread count. Re-measured quiet at `-t 6`: 22.57 t/s at d0,
  13.15 at d469. What was still open is narrower -- **does a CPU leg starve the
  other engine the way the busy-wait did?** **Half-answered 2026-09-03 for
  CPU+NPU, and the prediction was right in one direction and backwards in the
  other.** Measured on the deployed pairing (Genie 4B on the HTP, Qwen3.5-9B
  Q4_0 on the CPU at `-t 6`), d250, both hot:

  | leg | solo | contended | keeps |
  |---|---|---|---|
  | NPU 4B | 18.72 t/s | 17.63 | **92.6%** |
  | CPU 9B | 11.62 t/s | 6.67 (5.32-10.24) | **~57%** |

  So the NPU shrugs the CPU leg off, exactly as "host-load-insensitive"
  predicts -- but the CPU leg pays heavily, which the host-core story does not
  explain and bandwidth does: the 9B streams ~5.4 GB of weights per token
  (~63 GB/s at 11.6 t/s) against the NPU's ~2.1 GB (~39 GB/s), so together
  they ask ~100 GB/s of a bus that delivers 105-115. **The bigger model is
  the one that starves.** Aggregate is still a gain -- 24.30 t/s against
  18.72 for the best single engine, 1.30x. Caveats: the contended CPU samples
  are noisy (5.32-10.24) and a closing solo re-check came back 15% low, so
  treat ~57% as a floor on retention. CPU+GPU remains unmeasured.
- **Does a second resident model change the NPU's `1003` rate?** Memory pressure
  is a plausible aggravator; unproven.
- **Is there headroom for a third engine at all?** The pair already draws
  ~75-80% of achievable bandwidth. This is now a bandwidth question with a
  discouraging prior rather than an open one, but nobody has measured a third
  leg.

## Testing caveat

Do not benchmark this while anything large is running elsewhere. During the
16k export, free physical memory on this box was **1.1 GB** -- any
multi-instance numbers taken then would be meaningless. The same warning
already applies to the GEMM benchmark: a loaded machine inflates the NPU's
apparent win by slowing its baseline.

**Independently corroborated on a different stack, 2026-08-26.** A session
benchmarking the llama.cpp/ggml QNN path on this same box -- no Genie runtime
in the loop -- measured the same effect from the other side: an IDENTICAL first
leg repeated at the end of a four-leg sweep came back **26% low on AC and 43%
low on battery**, with the clock sliding **87% to 69% of base** across the four.
Counterbalancing the order (A,B,C,C,B,A) with 120 s cooldowns and averaging
each pair reproduced every backend to within 0.1-3.5%.

Two things worth taking from that. The effect is a property of the BOX, not of
either stack, since it reproduces through a completely different runtime. And
the cheap control is one this file's harness now implements: **repeat the
first leg last.** `bench_contention.py` interleaves solo against contended and
flags a monotonic decline afterwards (`drift_note`), and `bench_endpoint.py`
brackets its decode probe before and after a `--prefill-only` sweep -- but
nothing re-ran leg one at the END, which is the single measurement that turns
"the later legs look slower" from a suspicion into a number. Added:
`bench_contention.py` now closes every sweep by re-running the leg it opened
with, under the same cool gate, and reports the drift both ways -- slower means
decay landed in the numerator and contention is overstated, faster means the
OPENING sample was the degraded one and every retention percentage above it is
flattered. `--no-closing-recheck` skips it. Written up on that session's side in
its `docs/backend/QNN.md`.

Thermals are the half of this that is easy to miss, and on the GPU leg they are
worth **1.64x**. The same measurement at d469, varying only the state of the
box: 11.04 with an unrelated export running, 11.57 with a `poll: true` NPU
server merely resident, 16.86 with the NPU stopped but the box warm, **18.05**
cooled and clean. (That 11.57 is a busy-poll number. A `poll: false` server
sitting resident and idle costs nothing -- 18.03. See the concurrency section.)
Sustained GPU load drove the clock from ~94% to **48.9% of base**, and a
hand-run sequential depth sweep taken across that decay produced a clean
monotonic 19.65 -> 11.03 t/s that looked exactly like a depth effect and was
nothing of the kind. Recovery takes about two minutes. Any figure quoted without
its thermal and load state is meaningless, which is why `bench_contention.py`
blocks until the clock is back to >=92% of base before each sample rather than
detecting the drift afterwards and warning about it.

**Verify that the server you launched is the one answering.** Not hypothetical:
two restarts during the concurrency work never bound the port at all --
`genie_server` correctly refused, with *"something is already serving
127.0.0.1:8123"* -- while a readiness check that merely curled the port passed,
because the OLD `poll: true` process was still there answering it. Samples
labelled `poll: false` were served by a `poll: true` server, and that alone is
why this page spent a revision believing an idle process cost the GPU 15% and
inventing a package power budget to explain it. **A liveness probe proves
something is listening; it does not prove it is yours.** Anything that starts an
engine before measuring it must check the launcher's exit status and the PID
that owns the port -- and, when the measurement is about a config change, the
config the live process actually loaded.
