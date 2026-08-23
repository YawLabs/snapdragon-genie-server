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

## Results

These are **single-GEMM micro-benchmarks** -- the prefill matmul primitive
`[M tokens, K] x [K, N]`, **not** full-model tokens/sec. FP32-IO GEMMs run on
the HTP in FP16.

### Reference measurement (originally captured on this box -- cite, do not inflate)

| Case | NPU/HTP | ORT CPU EP | NPU win |
|---|---|---|---|
| FP16 GEMM 512x4096x4096 | 3477 GFLOP/s | 415 GFLOP/s | 8.4x |
| INT8 QDQ GEMM 512x4096x4096 | 2620 GOP/s | 1285 GOP/s | 2.0x |
| FP16 sweep @ 128 tokens | -- | -- | 19.2x |
| FP16 sweep @ 512 tokens | -- | -- | 11.8x |
| FP16 sweep @ 2048 tokens | -- | -- | 10.3x |

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

The FP16 GEMM NPU advantage (~10x and up) reproduces cleanly. Absolute GOP/s and
the exact speedup move run-to-run with thermal state and machine load (a busy
machine gives a slower CPU EP and thus a *larger* apparent NPU win) -- so the
numbers here differ from the reference table above, and both differ from a
lightly-loaded box. The load-bearing, stable result is the **order-of-magnitude
FP16 NPU advantage** with **HTP placement verified**, not the GFLOP/s to three
digits.

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

**Untested here (documented, not measured):**
- **Full-model decode tokens/sec.** The LLM wrapper `onnxruntime-genai` (0.15.2)
  has no cp314 win_arm64 wheel (max cp313), so it cannot share the proven 3.14
  venv. A genai decode harness needs a separate cp311/312/313 venv. See
  [docs/MODEL_CONVERSION.md](docs/MODEL_CONVERSION.md).
- **Model conversion pipelines** (Olive INT4, Qualcomm AI Hub, Foundry Local) --
  documented from vendor tooling, not run end-to-end here.

## Layout

```
src/qnn_ep.py            register + pick-NPU + build-session + HTP placement assertion
src/bench.py             CLI GEMM benchmark (FP16 / INT8 QDQ / prompt-length sweep)
docs/MODEL_CONVERSION.md full-LLM path: Olive / AI Hub / Foundry Local + genai caveat
requirements.txt         onnxruntime-qnn, onnx, numpy (genai is separate/optional)
```

## Relationship to the llama.cpp QNN backend

This is the **productized** NPU path (ONNX Runtime + QNN EP: shipping runtime,
Olive/genai tooling, pre-converted assets). It is intentionally **separate**
from a custom llama.cpp QNN/ggml backend, which reaches the same HTP through a
hand-written ggml backend. Different stacks, same silicon; this repo does not
depend on or touch that one.
