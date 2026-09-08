# snapdragon-genie-server

**What this repo is: a measurement log for running an LLM on the Snapdragon X
Elite Hexagon NPU, and a server that applies what it found.**

The findings are the valuable part and they are portable -- most are properties
of Genie and of the AI Hub bundles, not of this code, so they hold whichever
server you run. Several of them are worth 2-3x and are invisible in every
published descriptor of the bundle they apply to.

The server is the reference implementation. `src/genie_server.py` keeps a Qwen3
w4a16 Genie bundle resident on the Hexagon and serves it over the OpenAI *and*
Anthropic APIs, with streaming, tool calls, stop sequences and context eviction
-- see [docs/GENIE_SERVER.md](docs/GENIE_SERVER.md). It is **not** faster than
Qualcomm's own server; that was measured, and decode is a tie. Read
[what this found](#what-this-found) first and the server second.

The repo also carries the other route to the same silicon: **ONNX Runtime + the
QNN Execution Provider**, where `src/bench.py` is a single-GEMM micro-benchmark
with HTP placement verified on every run. That path has its own findings, kept
separable below.

## What this found

Ranked by what would change if you did not know it. Every number here was
measured on one X1E80100; the conditions are in the linked docs, and
[Proven vs untested](#proven-vs-untested-here-on-this-machine) says which
claims are neither.

### The five that change what you do today

1. **The shipped bundles set `"poll": true`, and it costs 2.7 idle cores and up
   to 36% of decode.** A resident server that had answered nothing but `/health`
   burned 267% CPU against a 0.0% control. Setting it false took decode at 4096
   from 11.6 to **18.0 t/s**. It is one line in the bundle's
   `genie_config.json`, it is a vendor default, and nothing measured got worse.

2. **A single-length export pays for its whole compiled window on every token.**
   Same model, same 8192 window, same HTP allocation *to the byte*
   (646,971,904): a multi-length bundle serves **1382 t/s prefill against 463**
   (2.98x) and **18.2 t/s decode against 8.8** (2.07x), for +3.8% on disk. So
   always export with several `--context-lengths`. `metadata.json` cannot tell
   the two apart -- same 28 inputs, 25 outputs, same KV shape -- the difference
   is 2 compiled graphs against 10, visible only in the context binary itself.

3. **Per-request sampling is inert on QAIRT 2.45, and it returns success.**
   `GenieDialog_getSampler`, `GenieSamplerConfig_createFromJson` and
   `GenieSampler_applyConfig` all return 0, and generation is byte-identical
   across seeds 1/999/12345 and temps 0.0/1.5/2.0. The sampler binds at
   `GenieDialog_create`. Any OpenAI-compatible layer over Genie that accepts
   `temperature` is lying unless it tested that the output moved.

4. **The bundles ship `"seed": 42`, and Genie re-seeds on every dialog reset --
   so an identical prompt returns a byte-identical answer, forever.** At temp
   0.8 the model is nominally sampling with the dice reset before every roll,
   which presents as a stubborn model rather than a config bug. `"seed": -1`
   does not rescue it: `reset()` re-seeds with `_seed` unconditionally, so -1
   casts to a fixed uint32. Rewriting the seed in the config text fixes it per
   *process*; per *request* needs a QAIRT that honours a post-create apply.

5. **Genie has no sliding window. Overflowing the compiled window is a hard
   query failure, not a truncation** -- so anything serving on top of it has to
   evict, and eviction has to keep the system turn and never orphan a tool
   result from its call.

### If you use ONNX Runtime + the QNN EP instead

| finding | consequence |
|---|---|
| The legacy `providers=[("QNNExecutionProvider", ...)]` argument is **silently ignored** under the dynamic-EP model | your op runs on CPU at ~1/10th speed, and `get_providers()` still lists QNN |
| `get_providers()` is not proof of placement | only the QNN HTP compile stages are, and reading them needs fd-level capture of the native log |
| The HTP can wedge into a transient `Code 1003` -- graph compiles, first execute fails | ORT then **silently rebuilds on CPU and retries**, reporting a ~1.0x "NPU" number |

Detail and the working attach are in
[the plugin-EP section](#the-one-gotcha-that-will-eat-your-afternoon-the-plugin-ep-attach).

### If you are choosing or building a server

| finding | detail |
|---|---|
| **Neither Qualcomm server honours an output cap correctly** | GenieAPIService ignores `max_tokens` *and* `max_completion_tokens` (16 requested, 125 returned); `geniex serve` honours only the modern spelling |
| **`geniex serve` ignores stop sequences** | `stop: ["four"]` returned output byte-identical to the unstopped run |
| **Decode is a tie** | 1.01x / 1.00x / 1.09x at depths 250/1500/3000, ranges overlapping. No serving layer makes the Hexagon emit tokens faster |
| Genie wants `{"stop-sequence": [...]}` | a bare array returns -8 "Top level config is not an object" and is then silently ignored by the generation |
| GenieAPIService reports `usage` as all zeros | so a client cannot bound a generation *or* detect that it failed to |

Measured with `src/bench_servers.py` and `src/probe_server_semantics.py` --
standing the other servers up on the same bundle is
[written down](docs/GENIE_SERVER.md#reproducing-the-cross-server-comparison),
because one of them takes four undocumented steps. The comparison is in
[what else serves these bundles](#what-else-serves-these-bundles-and-what-this-does-differently).

### If you are measuring anything on this hardware

| trap | what it does to your numbers |
|---|---|
| **The precondition is a settled pack, not "on AC"** | below ~20-25% charge the box halves prefill while `PowerOnline` reads True |
| **A depth that crosses a compiled-graph boundary** | the 1-token and N-token calls run different graphs, the subtraction stops cancelling, and the noise reads as a convincing thermal curve |
| **A decode window under ~16 steps** | measures per-request overhead, not decode: a 4-step window reported 0.60 t/s against a true 17.6 |
| **Sequential A/B on a drifting box** | hands all the drift to whichever arm ran second; interleave instead |
| **On Windows, a second process can bind a port another is serving** | both binds succeed, the OLD process keeps answering, and your new server logs a clean start while serving nobody |
| **A 200 from `/health` does not mean the model generates** | Qwen3.5-9B Q8_0 on the llama-qnn fork build answers every request with an empty completion and `finish_reason: stop`, at an impossible 66 t/s. Smoke-test the tokens, not the status code |

## Target

| | |
|---|---|
| Hardware | Snapdragon X Elite / X Plus (Hexagon v73, dev box is X1E80100) or X2 Elite (v81) -- the Windows-on-Snapdragon Hexagons. Verified on v73 only. |
| OS | Windows on ARM64 (tested on Windows 11, build 26200) |
| Python | 3.14 (win_arm64 wheels also exist for 3.11 / 3.12 / 3.13) |
| Runtime | onnxruntime 1.29.0 + onnxruntime-qnn 2.5.0 |

## The one gotcha that will eat your afternoon: the plugin-EP attach

`onnxruntime-qnn` (2.5.0) is **not** a normal EP baked into an onnxruntime
build. It is a **plugin** for onnxruntime 1.29's *dynamic-EP* model. You
register the plugin library, then attach it to a session **by device**.

The trap: the old-style

```python
# WRONG under the dynamic-EP model -- SILENTLY IGNORED. Runs on CPU.
ort.InferenceSession(model, providers=[("QNNExecutionProvider", {...})])
```

builds a session that *looks* fine -- no error, and `get_providers()` can even
list `QNNExecutionProvider` -- while the op quietly runs on CPU at ~1/10th the
speed. The correct attach is:

```python
import onnxruntime as ort, onnxruntime_qnn as q
ort.register_execution_provider_library(q.get_ep_name(), q.get_library_path())

# get_ep_devices() returns BOTH a QNN NPU and a QNN GPU device -- pick the NPU
npu = [d for d in ort.get_ep_devices()
       if d.ep_name == "QNNExecutionProvider"
       and d.device.type == ort.OrtHardwareDeviceType.NPU][0]

so = ort.SessionOptions()
so.add_provider_for_devices([npu], {"htp_performance_mode": "burst"})
sess = ort.InferenceSession(model, sess_options=so)
```

`src/qnn_ep.py` wraps exactly this.

### Verifying the NPU actually ran (never trust it)

`get_providers()` listing `QNNExecutionProvider` is **necessary but not
sufficient** -- individual nodes can still fall back to CPU. The only
trustworthy signal is that the **QNN HTP graph compiler emits its compile
stages** ("Graph Sequencing for Target", "Finalizing Graph Sequence", "VTCM
Allocation") at ORT `log_severity_level <= 3`. A silent CPU fallback prints
none of them and runs ~10x slower.

`qnn_ep.build_session(...)` captures that native log at the file-descriptor
level during session construction and **raises `PlacementError`** if the HTP
compile stages are absent. Placement is asserted, not hoped for.

Note also: the QNN EP HTP only offloads **quantized (QDQ INT8/INT4)** graphs or
supported float graphs. A plain FP32 op with a bad attach falls back to CPU
silently -- which is exactly why the verification above matters.

## Install

The wheels for `onnxruntime` and `onnxruntime-qnn` **ship the same
`onnxruntime` module** and collide. Install into a clean venv that has **no
plain `onnxruntime`**:

```powershell
# Windows ARM64, Python 3.11-3.14
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # onnxruntime-qnn pulls onnxruntime 1.29 as a dep
```

If a plain `onnxruntime` is already present, do **not** try to patch it in
place (`uninstall onnxruntime` then `--force-reinstall onnxruntime-qnn
--no-deps` is not enough). Start from a fresh venv.

## Run the benchmark

```powershell
python src\bench.py --all
# or individual cases:
python src\bench.py --fp16 --shape 512,4096,4096 --iters 30
python src\bench.py --int8
python src\bench.py --sweep
```

It generates the ONNX GEMM models on the fly (into a temp dir), builds an HTP
session (placement-verified) and a CPU-EP session for each, times both, and
prints ms/run, GOP/s, and the NPU-vs-CPU speedup. `--no-verify` downgrades the
placement check from hard-fail to a flag in the output.

### End-to-end LLM throughput

`src/bench.py` measures one matmul. To measure a whole served model, point
`src/bench_endpoint.py` at a running `genie_server.py`:

```powershell
python src\bench_endpoint.py                          # prefill + decode sweep
python src\bench_endpoint.py --base http://127.0.0.1:8123 --decode-only
```

It reports prefill and decode in tokens/sec at several context depths. Decode
is measured as the delta between an N-token and a 1-token run at the same depth,
so prefill cancels instead of being folded into the rate; prefill in turn has
one decode step removed, since a 1-token cap still generates a token and leaving
it in understates prefill by ~16% at shallow depths. Failed requests (a 429, a
400) skip that point rather than killing a twenty-minute sweep. Because it speaks plain OpenAI HTTP, the same command benchmarks
a `llama-server` GPU or CPU leg -- which is the only way to get a cross-engine
comparison on identical prompts.

**Read the result next to the bundle's `n_ctx`.** On Genie the compiled context
window sets throughput for every request, so a run is only comparable to
another run at the same `/props` `n_ctx` -- see the window-tax note in
`docs/GENIE_SERVER.md`.

## Results

These are **single-GEMM micro-benchmarks** -- the prefill matmul primitive
`[M tokens, K] x [K, N]`, **not** full-model tokens/sec. FP32-IO GEMMs run on
the HTP in FP16.

### Reference measurement (originally captured on this box -- cite, do not inflate)

| Case | NPU/HTP | ORT CPU EP | NPU win |
|---|---|---|---|
| FP16 GEMM 512x4096x4096 | 3477 GFLOP/s | 415 GFLOP/s | 8.4x |
| INT8 QDQ GEMM 512x4096x4096 | 2620 GOP/s | 1285 GOP/s | 2.0x |

This table used to carry three more rows -- an FP16 sweep reporting 19.2x /
11.8x / 10.3x at 128 / 512 / 2048 tokens, with both ms columns blank. They are
gone. Nothing in this repo, and nothing anywhere in its history, records the
measurements behind them: no ms, no GFLOP/s, no log, no script. A speedup ratio
with no timings under it is not a result, and nothing else on this page is
allowed to lean on one.

### Verified HTP reproduction (captured this session, real HTP)

Real HTP execution, placement-verified by the QNN compile stages *and* by the
order-of-magnitude speedup. `bench.py` is the clean consolidation of exactly the
two scripts that produced these:

```
=== FP16 GEMM 512x4096x4096 (FP32 IO, HTP runs FP16) ===
  NPU/HTP     5.81 ms/run   2956.8 GFLOP/s   [HTP verified]
  CPU EP    127.72 ms/run    134.5 GFLOP/s
  NPU speedup: 22.0x

=== INT8 QDQ GEMM 512x4096x4096 (HTP fuses to int8) ===
  NPU/HTP    12.98 ms/run   1323.3 GOP/s     [HTP verified]
  CPU EP     67.36 ms/run    255.0 GOP/s
  NPU speedup: 5.2x

=== FP16 prompt-length sweep (K=N=4096) ===
 tokens   NPU ms   CPU ms   NPU win
    128     1.23    37.56    30.4x
    512     5.21    53.68    10.3x
   2048    20.89   215.99    10.3x
```

**Read the 22.0x and the 10.3x together -- they are the same GEMM.** The
standalone FP16 case and the sweep's 512-token row are both `512x4096x4096`,
in the run pasted above, minutes apart. Working the printed ms back through
`ops = 2*M*K*N` (what `bench.py` computes) shows where the gap lives:

| row | M | NPU ms | NPU GFLOP/s | CPU ms | CPU GFLOP/s | win |
|---|---|---|---|---|---|---|
| standalone | 512 | 5.81 | 2957 | 127.72 | 134.5 | 22.0x |
| sweep | 128 | 1.23 | 3492 | 37.56 | 114.3 | 30.4x |
| sweep | 512 | 5.21 | 3298 | 53.68 | 320.0 | 10.3x |
| sweep | 2048 | 20.89 | 3290 | 215.99 | 318.2 | 10.3x |

The NPU leg is steady -- 2957 to 3492 GFLOP/s across all four rows, a 1.18x
spread. The CPU EP is not: the same 512-token GEMM reads 134.5 GFLOP/s
standalone and 320.0 in the sweep, a **2.38x** move. On that same row the NPU
shifted only 1.12x (2957 -> 3298), and 2.38 / 1.12 = 2.13 -- which is the
22.0x-to-10.3x collapse. **It is the baseline that moved, not the NPU.** So
22.0x is not a headline: it is one anomalously slow CPU measurement, and any
speedup quoted here inherits whatever the ORT CPU EP is doing that minute.

Thermal state and machine load do move these numbers, and a busy box gives a
slower CPU EP and thus a *larger* apparent NPU win -- but that is not what
happened here, and the direction is worth stating. `--all` runs the sweep
**last** (`src/bench.py:250-256`), so a warming box predicts the sweep's CPU leg
to be the slower of the two. It is the faster one, by 2.4x. That is unexplained.

Two harness limits to know before quoting any of this: `bench.py` reports a bare
mean over 30 iterations with no dispersion, and it always times the NPU leg
before the CPU leg with no interleaving, so a drifting box shows up as a shifted
ratio rather than as visible spread. Nothing here checks that the NPU's output is
numerically *correct*, and the FP16 rows compare fp16-on-HTP against fp32-on-CPU
-- a speed comparison, not an identical computation.

The load-bearing, stable result is the **order-of-magnitude FP16 NPU advantage**
with **HTP placement verified** -- not the GFLOP/s to three digits, and not any
single speedup ratio.

### A real gotcha you will hit: transient HTP `Code 1003`

After heavy back-to-back QNN usage this NPU can wedge: the graph still *compiles*
onto the HTP (compile stages appear, so a compile-only check passes), but the
first HTP *execute* fails with `QNN_COMMON_ERROR_SYSTEM ... Code: 1003`. By
default ONNX Runtime then **silently rebuilds the session on CPU and retries**,
so a naive benchmark reports a ~1.0x "NPU" result that is really CPU. `bench.py`
disables that fallback (`session.disable_fallback()`) so the run surfaces
honestly instead:

```
=== FP16 GEMM 512x4096x4096 (FP32 IO, HTP runs FP16) ===
  NPU/HTP  RUN FAILED: EPFail: [ONNXRuntimeError] : 11 : EP_FAIL : ... QNN graph
           execute error. Error: QNN_COMMON_ERROR_SYSTEM ... Code: 1003
           (HTP graph compiled=True, but execute failed -- transient device
            error, NOT reported as a CPU number)
  CPU EP     34.67 ms/run    495.5 GFLOP/s
```

It is a device-state issue, not a code bug -- when it is active it breaks the
original proven scripts identically. Recovery: let the NPU idle (or reboot); do
not trust a 1.0x "NPU" number, which is the whole reason placement is asserted
at both compile time and run time.

One more honesty note: the ORT **CPU EP** is a weaker baseline than
llama.cpp's KleidiAI-tuned ARM64 kernels, so the *real* NPU-vs-llama.cpp edge on
a full model is **smaller** than the NPU-vs-ORT-CPU-EP ratios above.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

Lint with the same config CI would have used, if there were CI:

```powershell
python -m ruff check src tests
```

417 tests, and **none of them need the NPU, a Genie bundle, or the QAIRT
SDK** -- they drive the handlers with a fake socket and a stub engine, so they
run anywhere.

That device-free property is load-bearing rather than incidental, and it has a
cost worth stating: the ctypes bindings and every Genie call are NOT covered.
A regression there is invisible until the server is actually started, so
starting it remains part of checking a change that touches the engine.

The Genie C API is deliberately NOT mocked. Two payload shapes it requires
(`{"stop-sequence": [...]}` and `{"sampler": {...}}`) were discovered only by
calling the real library: the obvious shapes were rejected or, worse, accepted
and silently ignored. A mock would have encoded the wrong assumption and made
the suite agree with a bug. Anything crossing that boundary belongs in a
hardware-gated integration test instead.

## What else serves these bundles, and what this does differently

**QAIRT ships no server.** Its whole Genie surface is one-shot CLIs
(`genie-t2t-run` builds a dialog, runs one query, exits), the C API, and Python
bindings; searching the 2.45 SDK on this box by filename, shipped source,
binary strings and `Genie.dll`'s import table (which pulls in no sockets
library) finds no HTTP anywhere. A serving layer is something you build -- and
several people have, so "nobody else did this" is not the reason to use this
one.

**Qualcomm ships two, beside the SDK rather than in it.** `geniex serve` in the
GenieX CLI (confirmed on this box: `serve -- Run the GenieX Server`) serves
QAIRT bundles *and* llama.cpp GGUF from one endpoint.
[qualcomm/qai-appbuilder](https://github.com/qualcomm/qai-appbuilder)'s
`GenieAPIService` is OpenAI-compatible, runs QnnHtp on Windows on ARM64, eats
the same AI Hub context binaries this server does, and adds LoRA, VLM and
embeddings that this has none of. Third-party, `npurun` reaches the same Genie
API through Rust FFI, and llama.cpp's Hexagon backend is now upstream.

**Both Qualcomm servers have now been run here, on the byte-identical bundle**
-- same directory, `poll: false`, `seed: 42`, `context_lengths
[512,1024,2048,4096,8192]`, ctx 8192 -- and both serve it correctly. What
follows is what that turned up. **No throughput comparison is published**, for
a reason given at the end.

| | this server | GenieAPIService v2.3.7 | `geniex serve` v0.5.0 |
|---|---|---|---|
| `max_tokens` (legacy OpenAI) | honoured | **ignored** | ignored |
| `max_completion_tokens` (current) | honoured | **ignored** | honoured |
| `usage` token accounting | reported | **all zeros** | reported |
| reasoning suppressed by default | yes | no | no |

**An output cap is the finding that matters.** Asked for 16 tokens,
GenieAPIService returned 125 under *both* spellings, `finish_reason: "stop"`,
and a `usage` block of zeros -- so a client cannot bound a generation, and
cannot tell from the response that it failed to. On a single-flight NPU that is
not a cosmetic gap: one caller's unbounded generation is every other caller's
queue. `geniex serve` honours the cap, but only under the modern spelling, so
an older SDK sending `max_tokens` gets an unbounded answer instead of an error.

**This server had the mirror of geniex's gap until the same test was pointed at
it** -- it honoured `max_tokens` and ignored `max_completion_tokens`. That is
fixed, and the fix exists because the comparison was run rather than assumed.
Reciprocally, the "OpenAI-compatible" claim in the row above is worth
distrusting everywhere it appears, this repo included, until a cap is actually
sent and the returned token count is counted.

**Getting there is not a fair fight either.** The `GenieAPIService_Stable`
package cannot load an AI Hub bundle at all: its detector tries QNN, MNN and
GGUF and rejects a valid Genie config with no reason given. The v2.3.7 build
loads it only with the `.bin` files sitting beside `config.json`, and only once
the model is declared `backend: "qnn"` in `service_config.json`. Both ship a
QAIRT 2.44 runtime against 2.45-compiled binaries, which is the unsupported
direction, so the local 2.45 DLLs had to be grafted in. `geniex pull
--model-hub localfs` took the same bundle in one command.

**So why not just proxy `geniex serve`?** It is the obvious design once decode
turns out to be a tie -- translate Anthropic to OpenAI, rename the token cap,
strip the orphan think tag, and inherit Qualcomm's maintenance plus GGUF and
CPU/GPU fallback. Three probes decide it:

| probe | this server | `geniex serve` |
|---|---|---|
| identical prompt, 3 runs | replays (per-process seed) | replays |
| prompt past the 8192 window | drops older turns first, then a 400 naming the counts | flat 400, no eviction |
| `stop: ["four"]` | honoured | **ignored** -- byte-identical to the unstopped run |

Eviction and think-tag stripping a proxy could do itself. Stop sequences it
cannot, and that is the one that matters: a proxy can cut the stream it returns,
but the engine behind it keeps generating to its own cap, so on a single-flight
NPU the device stays busy producing tokens nobody is reading. Determinism is
worse -- neither server fixes it, because the seed binds at dialog creation and
a post-create sampler apply is inert on QAIRT 2.45, so no HTTP layer can reach
it from outside the process.

That is the honest case for the direct path: not speed, but the two places
where being inside the process is the only way to reach the knob.

**Decode is a tie, and that is the useful result.** `src/bench_servers.py`
runs the two servers A/B/B/A/A/B, three passes each, restarting between passes
because the Hexagon is single-flight and they cannot both hold it. Decode is a
two-request delta at one prompt (`max_tokens` 1 vs 121), so prefill and
per-request HTTP overhead cancel -- which is what makes two different HTTP
stacks comparable at all. Depths keep prompt + 120 generated tokens inside one
compiled graph. Tokens are counted locally from the returned text with the
bundle's tokenizer, never from `usage`, because one server reports zeros there
and two instruments would not be one measurement.

| depth | this server | `geniex serve` | ratio |
|---|---|---|---|
| 250  | **17.69** t/s (16.33-18.35) | 17.43 (16.11-17.58) | 1.01x |
| 1500 | **14.06** t/s (13.78-14.37) | 14.00 (10.57-14.07) | 1.00x |
| 3000 | **10.77** t/s (10.54-10.86) | 9.86 (8.70-10.51) | 1.09x |

Medians of n=3, full range in brackets. At 250 and 1500 the ranges overlap and
the two are indistinguishable. At 3000 they technically do not -- ours' slowest
sample beat geniex's fastest by 0.03 t/s -- which with n=3 is a coin flip, not
a finding. Both of geniex's low outliers came from its FIRST pass; drop that as
warm-up and it matches everywhere.

**So do not choose this for speed.** Both servers drive the same bundle through
the same Genie C API onto the same Hexagon, and decode is bandwidth-bound on
the NPU rather than anything the serving layer does. A server cannot make the
HTP emit tokens faster, and this one does not. What differs is the table above
this one -- the API surface, the output cap, the token accounting -- and the
window-tax and `poll` findings, which are properties of the bundle and transfer
to whichever server you run.

*Conditions: X1E80100, Windows 11 26200, pack 73% on AC, an ordinary desktop
session running. Base clock varied 42-79% across passes, which is why the runs
are interleaved rather than sequential -- drift then lands on both arms instead
of on whichever ran second. The 5x wider spread on geniex is itself a
measurement, and a reason to read a single sample from either as a range.*

What is actually different here, each verified in this tree:

- **It speaks Anthropic, not only OpenAI.** `POST /v1/messages` with real
  Anthropic SSE, `input_schema` tools converted to the function shape Qwen3 was
  trained on, `stop_reason` mapping, and 529 rather than 429 for backpressure --
  so a Claude-shaped client points at it unproxied. Every option above is
  OpenAI-only.
- **It checks the two facts that decide your throughput BEFORE the 11-35s model
  load.** A bundle shipping `poll: true` (267% idle CPU; +55% decode when
  flipped) and a single-length export (2.98x slower prefill than a multi-length
  bundle at the same window). Neither is visible in `metadata.json` -- the two
  8192 bundles differ only in `genie.context_lengths` -- so no latency probe at
  a single depth distinguishes them.
- **It re-seeds per PROCESS, which is a smaller claim than it sounds and is
  stated that way on purpose.** The bundles ship `"seed": 42`, and Genie
  re-seeds its RNG from the config on every `GenieDialog_reset` -- so a bundle
  loaded verbatim replays byte-identical output for a repeated prompt, forever,
  and `"seed": -1` does not fix it (the constructor reads -1 as "seed from the
  clock", but `reset()` re-seeds with `_seed` unconditionally). This rewrites
  the seed in the config text at load, so two server runs differ. **Within one
  run an identical prompt still replays its identical answer** -- measured, and
  `geniex serve` does the same. Fixing it properly needs a QAIRT that honours a
  post-create sampler apply; on 2.45 that call is accepted and ignored.
- **It expects the NPU to wedge.** Wedges are detected by stalled token progress
  rather than elapsed time, `/health` answers without taking the engine lock,
  and the process exits for its supervisor instead of unwinding through a
  `GenieDialog_free` that can itself hang on a stuck driver. 34 device-free
  tests cover that decision layer.
- **It handles Qwen3 reopening its own think block** -- measured at 1 request in
  6, and 1 in 12 on a second sample, each producing the answer twice. A
  matched-pair regex provably cannot catch it, because the opening tag is in the
  prompt rather than in the output.

Reasons to use something else, none of them hypothetical:

- **You need concurrency.** This is single-flight, and worse than it sounds: the
  dialog holds one resident KV, so two interleaved conversations reset each
  other's prefix and both pay a full re-prefill.
- **You need per-request sampling.** `temperature` and `top_p` are accepted and
  do nothing -- Genie binds the sampler at dialog creation on 2.45, which this
  says at startup rather than hiding. `GenieAPIService` drives the sampler per
  request; this cannot.
- **You need GGUF, CPU/GPU fallback, LoRA, VLM or embeddings.** `geniex serve`
  and `GenieAPIService` cover those. This serves one Genie bundle, one model per
  process.
- **You are not on Windows on ARM64,** or you want a stack proven on more than
  one machine. Everything here is a single X1E80100.
- **You only wanted the findings.** `poll: false`, multi-length exports and the
  window tax are properties of Genie and the bundle, not of this server. Read
  [docs/GENIE_SERVER.md](docs/GENIE_SERVER.md), make two config edits, and keep
  whatever you are already running.

## Proven vs untested (here, on this machine)

**Proven:**
- The plugin-EP attach (`register_execution_provider_library` +
  `add_provider_for_devices`) drives the Hexagon HTP from Python 3.14.
- HTP placement verification via the QNN compile-stage log works and
  distinguishes a real HTP run from a silent CPU fallback.
- The single-GEMM benchmark (FP16, INT8 QDQ, prompt-length sweep) runs and
  shows an order-of-magnitude FP16 NPU advantage over the ORT CPU EP.
- **A full LLM runs on the NPU and serves HTTP.** A Qwen3-4B w4a16 Genie bundle
  sits resident on the HTP behind `src/genie_server.py`, answering both the
  OpenAI and Anthropic APIs with streaming, tool calls, stop sequences and
  context eviction. See [docs/GENIE_SERVER.md](docs/GENIE_SERVER.md). A
  Qwen3-8B tier serves the same way (`-Model qwen3-8b`); Qwen3.5-9B cannot be
  a Genie bundle today and serves via `src/run-llama-server.ps1` on the CPU
  or Adreno instead -- the model matrix and the reasons are in
  [docs/MODEL_OPTIONS.md](docs/MODEL_OPTIONS.md).
- **Full-model prefill and decode, measured** via `src/bench_endpoint.py`. The
  variable that matters is the bundle's LENGTH CLASS, not its window: a
  single-length export pays for its whole compiled window on every token, while
  a multi-length export pays only for the context actually in use. Same model,
  same 8192 window, same HTP allocation to the byte:

  | 8192 bundle | prefill t/s @ d469 / d2657 / d6157 | decode t/s @ d250 / d3300 / d6000 |
  |---|---|---|
  | single-length `[8192]` | 463 / 461 / 456 | 8.8 / 8.8 / 8.8 |
  | multi-length `[512..8192]` | **1382 / 997 / 636** | **18.2 / 11.5 / 8.1** |

  Those are per-depth figures, not medians. The mechanism was read off the
  artifact rather than inferred from timings: `qnn-context-binary-utility` shows
  **2** compiled graphs in the single-length bundle's `part2_of_4.bin` against
  **10** in the multi-length one's (`prompt_ar128_cl<N>` and `token_ar1_cl<N>`,
  one pair per compiled length), so a single-length bundle runs every token
  against its full window. It costs +3.8% bundle size and **zero** extra HTP
  memory -- both 8192 bundles allocate exactly 646,971,904 bytes. The
  launcher's default is the multi-length 8192 bundle. Full detail, including a
  refutation of this finding that was itself wrong and had to be retracted, in
  [docs/GENIE_SERVER.md](docs/GENIE_SERVER.md).
- **`poll: false` belongs in every bundle config.** As shipped, `"poll": true`
  busy-waits: a resident server burned 270% CPU (2.7 cores) while completely
  idle, against a 0.0% control, and the spinning threads compete with the work.
  Disabling it is worth **up to +55% decode** at 4096 (11.6 -> 18.0 t/s) --
  which is a **36%** slowdown while it is on, not a 55% one. The two framings
  divide the same pair of numbers in opposite directions and are easy to mix up.
  Disabling costs nothing measured.
- **Model conversion via Qualcomm AI Hub**, run end-to-end here: the 8192 and
  16384 bundles were built with `qai-hub-models export` (from WSL -- the Windows
  path dies on `fcntl`). The 4096 bundles are Qualcomm prebuilts, fetched rather
  than exported, so do not read the 4096-vs-8192 gap as an export artifact.

**Untested here (documented, not measured):**
- **The `onnxruntime-genai` path to full-model decode.** genai 0.15.2 has no
  cp314 win_arm64 wheel (max cp313), so it cannot share the proven 3.14 venv,
  and it is now the *alternative* rather than the plan -- Genie got there
  first. See [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md).
- **The other conversion pipelines** (Olive INT4, Foundry Local) -- documented
  from vendor tooling, not run end-to-end here.
- **Speculative decoding** (SSD / Eaglet). Not config-only on this bundle:
  both need a re-export. LADE *is* config-only and was measured -- it breaks
  tool calling, so it is disqualified rather than merely unproven.
- **Power draw**, which is the NPU's real claimed edge over CPU and GPU.

## Layout

```
src/qnn_ep.py             register + pick-NPU + build-session + HTP placement assertion
src/bench.py              CLI GEMM benchmark (FP16 / INT8 QDQ / prompt-length sweep)
src/genie_server.py       OpenAI + Anthropic HTTP server over a resident Genie bundle
src/bench_endpoint.py     prefill/decode benchmark against any OpenAI-compatible server
src/genie_smoke.py        minimal one-shot Genie generation, for isolating server bugs
src/bench_contention.py   two engines at once: solo vs contended, cool-gated sampling
src/bench_servers.py      interleaved A/B against another server on the SAME bundle
src/probe_server_semantics.py  seed replay / overflow / stop-sequence probes
src/run-genie-server.ps1  launcher + supervisor; finds the bundle/SDK itself (-Model picks 4B/8B)
src/run-llama-server.ps1  Qwen3.5-9B llama-server legs: CPU (Q4_0) / Adreno (Q4_K_M)
tests/                    417 device-free tests (no NPU, no bundle, no SDK needed)

docs/GENIE_SERVER.md      the server: endpoints, env vars, and its measured limits
docs/IMPLEMENTATION_PLAN.md  living plan + decision log; start here for the why
docs/MODEL_CONVERSION.md  full-LLM path: Olive / AI Hub / Foundry Local + genai caveat
docs/MODEL_OPTIONS.md     the model matrix: what serves where (4B/8B NPU, 9B llama.cpp) and why
docs/MULTI_ENGINE.md      running NPU + GPU + CPU at once -- 1.45x measured, and why
docs/TYPED_ROUTER_BRIEF.md  self-contained handoff for the routing work in typed
requirements.txt          onnxruntime-qnn, onnx, numpy (genai is separate/optional)
requirements-dev.txt      pytest only; the server itself has NO pip dependencies
```

## License

Apache-2.0 -- see [LICENSE](LICENSE). The patent grant is the reason for
Apache over MIT here: this is accelerator code, and the surrounding silicon is
patented territory.

Third-party attributions are in [NOTICE](NOTICE) -- the Qwen3 chat template
rendered by hand in `src/genie_server.py` (Qwen3 is (c) Alibaba Cloud,
Apache-2.0), and the Genie C API constants and ctypes signatures derived from
Qualcomm's published QAIRT headers, which are interoperability declarations
rather than redistributed SDK material.

**No vendor binaries or weights ship in this repo.** No model weights, no
bundle artifacts, no QAIRT binaries -- every tracked file is text. The one
piece of third-party text that does ship is the Qwen3 chat template named
above. The SDK and the bundles are obtained separately, by you, from Qualcomm.

Snapdragon, Hexagon, Adreno and Qualcomm are trademarks of Qualcomm
Incorporated or its subsidiaries; Qualcomm AI Runtime (QAIRT) and Genie are
Qualcomm Technologies product names. All are used here only to identify the
hardware and software this runs on. This project is not affiliated with,
sponsored by, or endorsed by Qualcomm, Microsoft or Alibaba, and no endorsement
is implied by any measurement published here.
