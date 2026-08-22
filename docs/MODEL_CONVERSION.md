# Model conversion: getting a full LLM onto the Hexagon NPU

The GEMM benchmark in this repo isolates the prefill matmul primitive. To run a
*whole* model (tokenizer, KV-cache, generation loop) on the HTP you need two
more things this benchmark deliberately skips:

1. A model converted/quantized into a QNN-friendly ONNX form (QDQ INT8 or INT4,
   HTP-partitionable).
2. `onnxruntime-genai` -- the generation wrapper that owns the tokenizer, the
   KV-cache, and the decode loop.

## The Python-version caveat (read first)

`onnxruntime-genai` (latest 0.15.2) has **no cp314 win_arm64 wheel** -- the
newest it publishes is **cp313**. This benchmark's proven venv is Python
**3.14**, so a genai decode harness cannot share it. Options:

- Create a **separate cp311 / cp312 / cp313 venv** just for the genai LLM path.
- Or wait for a cp314 genai wheel and reuse the 3.14 venv.

`onnxruntime-qnn` itself does ship a cp314 win_arm64 wheel, which is why the
GEMM benchmark runs on 3.14 -- the gap is genai-only.

Keep the two venvs separate; do not mix genai into `requirements.txt`.

## Path A -- Microsoft Olive (auto-optimize + quantize)

Olive is Microsoft's model-optimization toolkit. It can pull an HF model,
quantize it, and emit an ONNX model partitioned for the QNN EP:

```bash
pip install olive-ai onnxruntime-genai   # cp311/312/313 venv
olive auto-opt \
  --model_name_or_path <hf-org/model> \
  --provider QNNExecutionProvider \
  --precision int4 \
  --use_model_builder True \
  --output_path ./models/<model>-qnn-int4
```

`--use_model_builder True` routes through the ONNX Runtime GenAI model builder
so the result is directly loadable by `onnxruntime-genai`. For some models an
extra **Qualcomm AI Hub** compile step is needed to produce the HTP context
binary (`qai-hub` + a target-device compile job); Olive can emit the pre-compile
ONNX and AI Hub finishes the HTP lowering.

## Path B -- pre-converted assets (skip conversion)

Many models already have QNN/HTP-ready ONNX published. Search Hugging Face for
tags like `qnn`, `hexagon`, or repos named `*-hexagon-npu-assets`. Pull the
directory (model.onnx + genai_config.json + tokenizer files) and hand it
straight to `onnxruntime-genai`. Fastest way to a working decode loop; you are
trusting someone else's quantization recipe.

## Path C -- Microsoft Foundry Local (turnkey)

Foundry Local ships NPU-targeted model packages and a runner:

```bash
foundry model list --filter device=NPU
foundry model run <npu-model>
```

Least control, least setup -- good for a quick "does a real model decode on the
NPU at all" sanity check before investing in an Olive pipeline.

## Sketch of a genai decode harness (once a model exists)

```python
# cp311/312/313 venv with onnxruntime-genai + onnxruntime-qnn
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

- Paths A/B/C are documented from Qualcomm/Microsoft tooling references.
- **Untested on this machine** (blocked by the genai cp314 gap above). The
  proven-here result is the GEMM benchmark, not full-model decode t/s.
