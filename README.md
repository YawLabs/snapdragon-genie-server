# snapdragon-npu-llm

Running LLM compute on the **Snapdragon X Elite Hexagon NPU (HTP)** via
**ONNX Runtime + the QNN Execution Provider**. This is the *productized* NPU
path -- the same silicon a llama.cpp QNN backend targets, but reached through
Microsoft/Qualcomm's shipping runtime stack instead of a custom ggml backend.

The concrete, reproducible thing in this repo is a **single-GEMM
micro-benchmark** that places the LLM prefill matmul primitive on the HTP and
compares it to the ONNX Runtime CPU EP, with **HTP placement verified on every
run** (not assumed).

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

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

Lint with the same config CI would have used, if there were CI:

```powershell
python -m ruff check src tests
```

416 tests, and **none of them need the NPU, a Genie bundle, or the QAIRT
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
src/run-genie-server.ps1  launcher + supervisor; finds the bundle/SDK itself (-Model picks 4B/8B)
src/run-llama-server.ps1  Qwen3.5-9B llama-server legs: CPU (Q8_0) / Adreno (Q4_K_M)
tests/                    416 device-free tests (no NPU, no bundle, no SDK needed)

docs/GENIE_SERVER.md      the server: endpoints, env vars, and its measured limits
docs/IMPLEMENTATION_PLAN.md  living plan + decision log; start here for the why
docs/MODEL_CONVERSION.md  full-LLM path: Olive / AI Hub / Foundry Local + genai caveat
docs/MODEL_OPTIONS.md     the model matrix: what serves where (4B/8B NPU, 9B llama.cpp) and why
docs/MULTI_ENGINE.md      running NPU + GPU + CPU at once -- 1.45x measured, and why
docs/TYPED_ROUTER_BRIEF.md  self-contained handoff for the routing work in typed
requirements.txt          onnxruntime-qnn, onnx, numpy (genai is separate/optional)
requirements-dev.txt      pytest only; the server itself has NO pip dependencies
```

## Relationship to the llama.cpp QNN backend

This is the **productized** NPU path (ONNX Runtime + QNN EP: shipping runtime,
Olive/genai tooling, pre-converted assets). It is intentionally **separate**
from a custom llama.cpp QNN/ggml backend, which reaches the same HTP through a
hand-written ggml backend. Different stacks, same silicon; this repo does not
depend on or touch that one.

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
