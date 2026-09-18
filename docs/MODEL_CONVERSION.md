# Model conversion: getting a full LLM onto the Hexagon NPU

The GEMM benchmark in this repo isolates the prefill matmul primitive. Running
a *whole* model (tokenizer, KV cache, generation loop) on the HTP is a
different job, and as of 2026-08-22 it is a done one: a Qwen3-4B w4a16 **Genie
bundle** sits resident on the HTP behind `src/genie_server.py` and serves both
the OpenAI and Anthropic APIs. That route is the first section below. The
ONNX Runtime GenAI routes this document used to lead with are kept after it
as the *untested* alternatives they are.

Rewritten 2026-09-16. Until then this file was its 2026-08-22 options text,
never updated after the server landed: it said the only proven result here
was the GEMM benchmark, that full-model decode was untested and "blocked by
the genai cp314 gap", and it did not mention Genie or `qai-hub-models` at
all. Every one of those claims was false for three weeks while
`docs/IMPLEMENTATION_PLAN.md` listed "MODEL_CONVERSION.md upgraded to the
runnable recipe" as a delivered Phase 2 item.

## The proven path: a Genie bundle from Qualcomm AI Hub

Qualcomm's `qai-hub-models` builds (or ships) a **Genie bundle** -- HTP context
binaries + `genie_config.json` + tokenizer + `metadata.json` carrying the chat
template -- compiled for the local chipset, which dissolves the arch/version
lock that killed every third-party prebuilt surveyed in
`docs/IMPLEMENTATION_PLAN.md` Phase 1. Two ways to get one:

```bash
# Fetch Qualcomm's own prebuilt for this part (what the 4096 bundles here are):
qai-hub-models fetch qwen3_8b -r genie -p w4a16 -c qualcomm-snapdragon-x-elite --extract -o <dir>

# Export your own at a chosen window (what the 8192 / 16384 bundles here are):
qai-hub-models export qwen3_4b --chipset qualcomm-snapdragon-x-elite \
  --context-lengths 512,1024,2048,4096,8192
```

Things this repo learned the hard way, each with its measurement elsewhere:

- **Export runs on Linux/WSL.** The Windows path dies on `fcntl`. Fetch works
  anywhere. Exports take hours and the AI Hub jobs survive a dead client --
  check for job ids before concluding anything is lost
  (`docs/MODEL_OPTIONS.md`, "Export provenance").
- **Always pass several `--context-lengths`.** A single-length export pays for
  its whole compiled window on every token; a multi-length one runs each token
  against the smallest compiled graph that fits. Same model, same 8192
  window, same HTP allocation to the byte: 1382 against 463 t/s prefill and
  18.2 against 8.8 t/s decode on short prompts, for +3.8% on disk
  (`docs/IMPLEMENTATION_PLAN.md`, "The window tax", ANSWERED 2026-08-24).
- **Two per-machine fixes go into `genie_config.json` before first serve**,
  with the shipped copy kept as `genie_config.json.orig`: `"poll": false` in
  the QnnHtp block (as shipped, `true` busy-waits on 2.7 idle cores and costs
  up to 36% of decode) and a `token-penalty` block (1.15 / last-n 128 /
  freq 0.3; exports ship none, so long generations can loop). Both are in
  `docs/MODEL_OPTIONS.md`.
- **The runtime must match the bundle.** A bundle is locked to one Hexagon
  arch and one QAIRT version; everything served here is QAIRT 2.45 against
  v73, and `src/genie_server.py` derives the archs this box can drive at
  startup rather than trusting a config.
- **Not every model has a target.** `qai-hub-models` carries `qwen3_4b` and
  `qwen3_8b` but no `qwen3_5_9b`, and the Qwen3.5 entries that do exist are
  GGUFs for GenieX, not Genie bundles -- so Qwen3.5-9B serves through
  llama.cpp here instead (`docs/MODEL_OPTIONS.md`).

Bundles on this box, all Qwen3 w4a16 for X Elite: the 4B and 8B 4096
prebuilts (multi-length out of the box), the self-exported 4B `ctx8192`
(single-length), `ctx16384` (single-length) and `ctx8192-multi` -- the
launcher's default -- and the 8B `ctx8192-multi`. Serve any of them with
`src/run-genie-server.ps1 -Model ...`; the server, its endpoints and its
measured limits are in `docs/GENIE_SERVER.md`. Measured on the default:
~1382 t/s prefill at d469 and 18.2 t/s decode at d250; the 4096 prebuilt
gives 855-938 t/s prefill and 18.55 t/s decode at d469.

## The fallback: the hand-rolled QAIRT chain

For a model AI Hub does not carry, the SDK's own tools convert ONNX -> DLC ->
quantized DLC -> v73 context binary. That chain was run end-to-end on this box
on a small graph (the output parses with `"dspArch": 73`), and its LLM front
end -- HF to ONNX with KV-cache I/O -- was never run, because AI Hub export
covers it. The recipe, tool-by-tool host-arch notes, and the composer path
that turned out to be a CPU backend and dead on ARM64 anyway, are all in
`docs/IMPLEMENTATION_PLAN.md`, Phase 2. Note it was verified on QAIRT 2.34 and
has not been re-run on the 2.45 the bundles now target. Reach for it last.

## Untested alternatives: ONNX Runtime + `onnxruntime-genai` + QNN EP

Microsoft's path: a model converted into a QNN-friendly ONNX form (QDQ INT8 or
INT4, HTP-partitionable), driven by `onnxruntime-genai`, which owns the
tokenizer, the KV cache and the decode loop. More ecosystem tooling (Olive),
more moving parts than Genie, no native speculative decoding. Documented from
vendor tooling; not run end-to-end here.

### The Python-version caveat, corrected

This section used to say `onnxruntime-genai` 0.15.2 has **no** cp314
win_arm64 wheel and that a genai harness therefore could not share the
benchmark's Python 3.14 venv. That was wrong: PyPI lists
`onnxruntime_genai-0.15.2-cp314-cp314-win_arm64.whl` alongside cp311, cp312
and cp313 (re-checked 2026-09-16), and `docs/IMPLEMENTATION_PLAN.md` Phase 1
records genai 0.15.2 loaded in this box's own stack. A separate venv is still
sensible, but as hygiene, not because of a version gap, and nothing here was
ever blocked by it. (The reason this paragraph gave -- that genai and
`onnxruntime-qnn` "can collide on the `onnxruntime` module" -- holds for the
old onnxruntime-qnn 1.x wheels, which carry a whole `onnxruntime/` package,
and not for the 2.5.0 pinned here: opened 2026-09-17, that wheel ships only
`onnxruntime_qnn/` and declares `onnxruntime>=1.24.2`, the same plain wheel
genai's `onnxruntime>=1.20.1` asks for. Sharing a venv is therefore plausible
on paper and untested; see the README's Install section.) Keep genai out of
`requirements.txt` for the reason that does hold: the GEMM benchmark does not
need it.

### Path A -- Microsoft Olive (auto-optimize + quantize)

Olive can pull an HF model, quantize it, and emit an ONNX model partitioned
for the QNN EP:

```bash
pip install olive-ai onnxruntime-genai   # in a venv separate from the benchmark's
olive auto-opt \
  --model_name_or_path <hf-org/model> \
  --provider QNNExecutionProvider \
  --precision int4 \
  --use_model_builder True \
  --output_path ./models/<model>-qnn-int4
```

`--use_model_builder True` routes through the ONNX Runtime GenAI model builder
so the result is directly loadable by `onnxruntime-genai`. For some models an
extra Qualcomm AI Hub compile step produces the HTP context binary; Olive can
emit the pre-compile ONNX and AI Hub finishes the HTP lowering.

### Path B -- pre-converted ONNX-QNN assets (skip conversion)

Search Hugging Face for tags like `qnn`, `hexagon`, or repos named
`*-hexagon-npu-assets`; pull the directory (model.onnx + genai_config.json +
tokenizer files) and hand it to `onnxruntime-genai`. **This is the one
alternative that was actually tried here, and it dead-ended**: the llmware
Qwen2.5-1.5B ONNX-QNN assets were built for qairt 2.36 / ort 1.22 / ortg 0.9,
and no public genai wheel both reads that format and carries a QNN EP.
Prebuilt ONNX-QNN models are locked to the vendor's runtime *build*, not just
its version -- the full account is `docs/IMPLEMENTATION_PLAN.md` Phase 1,
"PATH A EXHAUSTED".

### Path C -- Microsoft Foundry Local (turnkey)

Foundry Local ships NPU-targeted model packages and a runner:

```bash
foundry model list --filter device=NPU
foundry model run <npu-model>
```

Least control, least setup. Not run here; Genie answered the "does a real
model decode on the NPU at all" question it would have been a sanity check
for.

### Sketch of a genai decode harness (if a model ever exists)

```python
import onnxruntime_genai as og
model = og.Model("./models/<model>-qnn-int4")   # genai_config.json selects QNN EP
tok = og.Tokenizer(model)
params = og.GeneratorParams(model)
params.set_search_options(max_length=256)
gen = og.Generator(model, params)
gen.append_tokens(tok.encode("Explain the Hexagon NPU in one sentence."))
while not gen.is_done():
    gen.generate_next_token()
    print(tok.decode([gen.get_next_tokens()[0]]), end="", flush=True)
```

The QNN EP is selected via the model's `genai_config.json` (`provider_options`
with `QNNExecutionProvider`), not by the Python attach API this repo's benchmark
uses. Verify HTP placement the same way: run with ORT log severity <= 3 and
confirm the "Finalizing Graph Sequence" compile stages appear.

## Status

- **Proven here and serving: Genie via `qai-hub-models` (fetch or export).**
  Full-model prefill and decode are measured per bundle in
  `docs/GENIE_SERVER.md` and `docs/IMPLEMENTATION_PLAN.md`; the model matrix
  is `docs/MODEL_OPTIONS.md`.
- **Verified on a toy graph, LLM front never run:** the hand-rolled QAIRT
  chain (Phase 2 fallback), on QAIRT 2.34.
- **Tried and dead-ended:** public prebuilt ONNX-QNN assets via pip genai
  (runtime-build lock).
- **Untested on this machine:** Olive auto-opt, Foundry Local, and any genai
  decode harness. Nothing blocks them -- the cp314 gap this file cited did not
  exist -- they simply lost to a route that worked first.
