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
| one instance @ 4k ctx | ~3.0 GB | ~0.6 GB | **~3.6 GB** |
| one instance @ 16k ctx | ~3.0 GB | ~2.3 GB | **~5.3 GB** |

Three instances land around **11 GB at 4k** or **16 GB at 16k**, which fits
comfortably. KV is ~144 KB/token (36 layers, 8 KV heads GQA, head_dim 128,
fp16). **Capacity was never the constraint.**

## The actual constraint is bandwidth

Decode streams the whole model per token, and every engine shares one memory
bus. Three concurrent decoders do not triple throughput -- they divide the same
bandwidth. This is already documented for the single-engine case (a large
resident model elsewhere slows the NPU sharply); adding engines makes the
contention deliberate rather than accidental.

Realistic expectation is **1.5-2x aggregate**, not 3x, and every individual
engine gets slower while the others are busy.

Measured single-engine baselines (prefill / decode, tokens/sec):

| engine | prefill | decode | notes |
|---|---|---|---|
| NPU (Genie / QnnHtp) | 277 | 13.2 | best of the three |
| GPU (Adreno) | 117 | 6.0 | ~half the NPU's decode |
| CPU (KleidiAI) | fine | **~0.2** | **broken -- unresolved anomaly** |

**The CPU leg is not worth building yet.** At 0.2 t/s it contributes nothing
while consuming bandwidth the other two need. Until that anomaly is understood
this is a **two**-engine design. Fixing CPU decode is worth more than adding a
third instance, and it matters far beyond this box: CPU is the fallback every
non-Snapdragon user lands on.

## Why it is still worth doing

Not for memory, and not really for throughput. The reason is that **the NPU is
single-flight**: concurrent HTP access wedges the device, so `genie_server`
serializes every request behind a lock. One instance serves exactly one request
at a time.

A second engine buys:

- **Concurrency** -- a second request stops queueing behind the first.
- **Failover** -- somewhere to route when the HTP throws its transient
  `Code 1003`, which is a device-state fault no amount of retrying fixes.
- **Tiering** -- the NPU is fastest, the GPU can hold a larger or smarter model.

## Shape: one Genie server, N llama-servers, router in typed

The three "instances" are not three copies of this server:

- **NPU** -> `genie_server.py` (this repo), Genie context binary on the HTP.
- **GPU / CPU** -> `llama-server`, which typed's free tier already runs.

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
4. Health checks that survive the HTP wedge -- `/health` answering is not
   proof the device will execute, since `1003` fails at execute time, not at
   load.

## Open questions

- **Does concurrent GPU + NPU inference actually hold up?** Both reach the same
  memory controller; contention is measurable but has not been measured here.
  Benchmark before designing around a number.
- **Why is CPU decode 0.2 t/s?** Blocks the third engine and is the highest
  leverage unknown on this list.
- **Does a second resident model change the NPU's `1003` rate?** Memory
  pressure is a plausible aggravator; unproven.

## Testing caveat

Do not benchmark this while anything large is running elsewhere. During the
16k export, free physical memory on this box was **1.1 GB** -- any
multi-instance numbers taken then would be meaningless. The same warning
already applies to the GEMM benchmark: a loaded machine inflates the NPU's
apparent win by slowing its baseline.
