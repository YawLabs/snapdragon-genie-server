# NPU LLM Inference: Robustness Implementation Plan

Living document. Status markers: [ ] todo, [~] in progress, [x] done, [!] blocked.
Last updated: 2026-08-22 (PATH A composer proven DEAD on this box -- no aarch64 GenAiTransformer backend + it is
CPU anyway; PATH B / QnnHtp v73 is the only local NPU route, still gated on the HF->ONNX+KV front end).

## The problem this solves

The eager llama.cpp QNN backend (in `llama-qnn-fork`) is proven in micro-benchmarks
(test-backend-ops 43/43; single-matmul 6-11 TFLOP/s with burst + static weights, beating
ONNX QNN EP) but **hangs on real end-to-end model inference**. Root cause is architectural:
it JIT-compiles a QNN graph **per op, during inference**. Some real-model shape wedges the
HTP inside `graphFinalize`/`graphExecute`, the call never returns, and:

1. A call-level watchdog (detached thread + timeout) converts the hang into a *failure*, but
   llama.cpp then aborts the whole decode (`llama_decode` returns -3) -- there is no graceful
   mid-batch CPU fallback; `supports_op` degrade only affects *future* ops.
2. The abandoned watchdog thread stays stuck in the QNN driver and **blocks process exit**.

Conclusion: a per-op-JIT design is fundamentally at odds with the HTP. The fix is not a patch
to the eager backend -- it is a different execution model.

## The core insight (why the ONNX/Genie path is robust by design)

Both viable NPU LLM runtimes are **ahead-of-time (AOT)**: the whole model is compiled to a QNN
**context binary once** (offline or at load), then executed. There is **no per-op runtime
finalize during inference**, so the entire class of mid-decode hangs is designed out. Shapes
the HTP cannot run are resolved at **compile/partition time** (fall back to CPU EP there), not
mid-batch. And they run in their **own process** -- no leaked-thread-blocks-exit problem.

I already proved the QNN EP runtime works on this box this session: INT8 QDQ matmul at
2620 GOP/s on the HTP, FP16 at ~3477 GFLOP/s, plugin-EP attach + HTP placement verified.

## Answers to the two questions

1. **Will it work with `snapdragon-npu-onnx` (ONNX-RT + QNN EP)?** Yes. It is AOT, robust by
   design, and the runtime is already proven working here. This is the right home for it.
2. **Other options, ranked:**
   - **Genie (Qualcomm's LLM runtime)** -- PRIMARY. `genie-t2t-run.exe` ships in the QAIRT SDK
     for `aarch64-windows-msvc`, with a C API (`GenieCommon.h` etc.). Purpose-built for
     LLM-on-HTP: AOT context binary + KV cache + tokenizer + sampling, all handled. Fastest
     path to a working, robust NPU LLM.
   - **ONNX-RT + onnxruntime-genai + QNN EP** -- ALT. Microsoft's path; `onnxruntime-genai`
     0.15.2 has a cp314 win_arm64 wheel. More ecosystem tooling (Olive), slightly more moving
     parts than Genie.
   - **llama.cpp load-time pre-validation OR out-of-process QNN worker** -- NOT worth it. Both
     essentially reimplement what Genie already does. Keep the llama.cpp QNN backend only as
     the proven micro-benchmark artifact; use CPU + Adreno-GPU for GGUF models there.

## Recommended approach

Home project: **`snapdragon-npu-onnx`**. Engine: **Genie on the QnnHtp backend** (Path B). The Genie
**GenAiTransformer** path (Path A composer) is DEAD on this box -- its backend DLL is x86_64-only in QAIRT 2.34
AND it is a CPU backend, not the HTP -- so the only local NPU route is Path B: convert to v73 QnnHtp context
binaries and run them via `genie-t2t-run.exe` (whose QnnHtp backend IS native aarch64). The remaining gate is
the HF->ONNX-with-KV-cache front end (Phase 2 Path B step 1). ONNX-RT + genai stays a documented alternative.

---

## Phases

### Phase 0 -- environment [x] DONE
- [x] QAIRT 2.34 SDK on disk; QnnHtp + Genie binaries for aarch64-windows-msvc identified.
- [x] `onnxruntime-qnn` 2.5.0 working (plugin-EP attach proven, HTP placement verified).
- [x] Extracted Genie tools + `Genie.dll`/`Genie.lib` + `include/Genie` headers. Deps that must sit
      next to `genie-t2t-run.exe` on PATH: `Genie.dll`, `QnnHtp.dll`, `QnnHtpV73Stub.dll` (v73 = X Elite),
      `QnnSystem.dll`.
- [x] Confirmed `genie-t2t-run.exe` runs (usage prints, exit 0). CLI shape:
      `genie-t2t-run.exe -c <dialog.json> -p "<prompt>"`; supports `--save/--restore` (KV session
      state -> agent multi-turn), `--profile`, `--log`, LoRA, and token/embedding inputs.
      Genie handles the LLM loop (AOT context binary + tokenizer + KV + sampling) -- no per-op JIT.

### Phase 1 -- prove Genie generates text on the HTP [!] prebuilt path BLOCKED
Goal: one real end-to-end NPU generation with measured t/s, no hang.
- [!] FINDING: no prebuilt Genie bundle matches this box. A Genie context binary is tied to BOTH
      the HTP arch AND the QAIRT version, and this box is **v73 + QAIRT 2.34 + Windows**. Surveyed
      HF prebuilts all miss: imi2/QNN-HTP-LLM-Genie (Android, v79), piffie/...X2-Elite (v81),
      Opt-AI/...8gen3 (labelled 8gen3 but htp config says **dsp_arch v75**, and **QAIRT 2.42**).
      Wrong arch OR wrong version -> won't load. (This box is definitively v73: the HMX_V2
      probe errored "not supported in current architecture: 73".)
- [ ] PIVOT -- two ways to get a v73/2.34-local artifact instead of a prebuilt:
      (A) **ONNX-RT + onnxruntime-genai + QNN EP** -- provide an arch-AGNOSTIC ONNX LLM; the QNN EP
          AOT-compiles it for the LOCAL v73 device at session init. Sidesteps the prebuilt-arch
          mismatch entirely. Risk: genai+QNN-EP LLM support on Win-ARM is newer/finicky; genai and
          onnxruntime-qnn may collide on the `onnxruntime` module (isolate in a clean venv).
      (B) **Convert for v73/2.34 (Phase 2)** -- qairt-converter + qnn-context-binary-generator with
          `--dsp_arch v73` using the local QAIRT 2.34 tools. Reliable but multi-step.
      Recommended: try (A) first (less to build); fall to (B) if genai+QNN-EP LLM path stalls.
- [!] PATH A EXHAUSTED (prebuilt ONNX-QNN via pip): DEAD END without the model's exact runtime build.
      Tried `llmware/qwen2.5-1.5b-instruct-onnx-qnn` (int4, EPContext .bin context binaries, built
      for qairt 2.36 / ort 1.22 / ortg 0.9):
      * genai 0.15.2 (my stack, has the QNN-EP plugin `register_execution_provider_library`) -> fails
        to LOAD the model: "system error number 13" -- the decoder-pipeline/EPContext format changed
        between ortg 0.9 and 0.15.
      * genai 0.9.2 (arm64 py3.12 venv via uv; reads the 0.9 format) -> "QNN execution provider is
        not supported in this build" -- genai 0.9's bundled onnxruntime has NO QNN EP baked in, and
        there is no clean `onnxruntime-genai-qnn` pip package (404). QNN-in-genai-0.9 requires the
        vendor's specific build (llmware's), not a public wheel.
      * genai 0.15 model builder (`python -m onnxruntime_genai.models.builder`) -> import fails
        (needs torch+transformers; torch on win-arm64 is its own problem) and QNN-target support is
        unconfirmed.
      CONCLUSION: prebuilt ONNX-QNN models are runtime-BUILD-locked, not just version-locked. There is
      no pip path to genai+QNN that matches a public prebuilt on this box. => Path B (local conversion
      with the SDK's own CLI tools, version-consistent by construction) is the reliable route, and
      Genie (not ONNX-genai) is the runtime to target -- matching the independent recommendation that
      Genie is the deployment path and ONNX-RT genai is only for portability.
- [ ] Write the `genie_config.json` (backend = QnnHtp, context binary path, tokenizer, sampler).
- [ ] Run `genie-t2t-run.exe --config genie_config.json --prompt "..."` -> generated text.
- [ ] Measure prefill (TTFT) and decode t/s; compare to llama.cpp CPU + Adreno-GPU numbers
      (14B Q4_0 GPU baseline: 73.79 pp / 4.46 tg).
Deliverable: `src/genie_run.py` (or a documented CLI invocation) + a results row.
Exit criterion: NPU generates coherent text end-to-end without hanging.

### Phase 2 -- local v73/2.34 model conversion pipeline [~] FEASIBLE, VERIFIED ON THIS BOX
Goal: turn an arbitrary HF model into a Genie artifact for v73 + QAIRT 2.34 + Win-ARM64 -- the AOT
compile that replaces llama.cpp's per-op JIT. This is "path B" from Phase 1 (build a local artifact
instead of a version-locked prebuilt). **VERDICT: local conversion IS feasible on this Win-ARM box.**
The whole ONNX -> DLC -> INT8 DLC -> **v73 context binary** chain was run end-to-end here and the
output `.bin` parses with `"dspArch": 73`. Only the LLM-specific front (HF -> ONNX-with-KV-cache) and a
full-size LLM run remain unproven.

#### Tooling and host-arch availability (from `qairt.zip`, verified by running `--help` + real jobs)
The QAIRT 2.34 zip ships these host trees: `aarch64-windows-msvc` (native ARM64 runtime exes),
`arm64x-windows-msvc` (ARM64EC converter *scripts* + arm64ec native `.pyd`), `x86_64-windows-msvc`,
`x86_64-linux-clang`, plus Android/oe-linux. What matters for THIS machine:

| Tool | Purpose | Host build(s) present | Runs here? |
|---|---|---|---|
| `qairt-converter` | ONNX/PT/TF/TFLite/**GGUF** -> `.dlc` | arm64x-win, x86_64-win/linux (Python) | **YES, native** (arm64ec `.pyd` via emulated x64 Py3.10). VERIFIED: Relu ONNX -> `.dlc`. |
| `qairt-quantizer` | `.dlc` -> INT4/INT8 `.dlc` | same | **YES, native**. VERIFIED: float `.dlc` -> INT8 `.dlc` (ran CPU calibration). |
| `qnn-context-binary-generator.exe` | `.dlc`/`.so` + HTP -> context `.bin` | **aarch64-windows-msvc** (native ARM64 exe) + x86_64 | **YES, native**. VERIFIED: INT8 `.dlc` -> `relu_v73.bin`, parsed `"dspArch":73`. |
| `qnn-context-binary-utility.exe` | inspect/validate context `.bin` | aarch64-windows-msvc | YES, native. Used to confirm `dspArch=73`. |
| `qnn-model-lib-generator` | converter `.cpp/.bin` -> model `.dll` | arm64x-win, x86_64 (Python) | Launches, but `LIB_TARGETS` is **empty** on arm64x (no MSVC target wired). **SKIP it** -- the DLC flow feeds `--dlc_path` straight to the context-bin generator. |
| `qnn-genai-transformer-composer` | HF dir -> single Genie `.bin` (native INT4) | **x86_64-win / x86_64-linux ONLY** -- no ARM build; needs `QnnGenAiTransformerComposerQuantizer.dll` (x64-only) | **YES, under x64 emulation** (VERIFIED `--help`). Needs a DLL-dir workaround (see below). |
| `genie-t2t-run.exe` | the LLM runtime loop | aarch64-windows-msvc (native) | Native exe runs, but **only QnnHtp** backend `.dll`s ship for aarch64-win. The **GenAiTransformer** backend (`QnnGenAiTransformer.dll` + CpuOpPkg/Model, `QnnSystem.dll`, `QnnCpu.dll`) is **x86_64-ONLY**, and there is **no x86_64 `genie-t2t-run.exe`** -- so PATH A cannot run here at all (see PATH A blocker). |

#### Environment setup (do this once) [x] VERIFIED
- **Python: use x64 CPython 3.10, NOT the box's native 3.14.** The SDK's native `.pyd` are ABI-locked to
  CPython 3.6/3.8/3.10; the unsuffixed default is **3.10** (imports `python310.dll`). Local `python`=3.14
  ARM64 will `ImportError`. Use the uv x64 build: `cpython-3.10.20-windows-x86_64` (Py3.8 also works via
  the `*38.pyd`). Under x64 emulation `platform.processor()` returns `"ARMv8..."`, so the converter's
  loader (`qti/aisw/converters/common/__init__.py`) selects the **windows-arm64ec** native modules -- which
  execute natively via ARM64EC. Verified: `libPyIrGraph.pyd` (arm64ec) loads in emulated x64 Py3.10.
- **Create a venv and pin deps:** `numpy==1.26.4` (converter needs numpy<2), `pyyaml`, `packaging`,
  **`onnx==1.16.1`** (do NOT use onnx>=1.18 -- it removed `onnx.mapping`, which the converter imports; onnx
  1.22 silently makes the onnx frontend a no-op that dies with `'NoneType' has no attribute 'AttributeProto'`).
  For the composer path add `sentencepiece`, `tqdm`.
- **Env vars per invocation:** `QNN_SDK_ROOT=<sdk>`, `PYTHONPATH=<sdk>/lib/python`. (`bin/envsetup.ps1`
  only wires `aarch64-windows-msvc`, not the arm64x converter tools -- set these by hand.)

#### PATH A -- GenAiTransformer via composer  [!] DEAD ON THIS BOX (2026-08-22)
BLOCKER (verified end-to-end): the composer step SUCCEEDS -- Qwen2.5-1.5B `--quantize Q4` produced a valid
1.9 GB GGUF `model.bin` in 114 s. But `genie-t2t-run.exe` (native aarch64) then fails to load the backend:
`[ERROR] Unable to load backend. dlopen error #126 -> QnnGenAiTransformer.dll -> QNN initialization failed`.
Root cause: **no aarch64 build of `QnnGenAiTransformer.dll` exists in QAIRT 2.34** (aarch64 ships only the
GenAiTransformer *headers* + QnnHtp backend); the backend DLLs are x86_64-only, and there is no x86_64
`genie-t2t-run.exe` to drive them under emulation. Copying the x64 DLLs next to the ARM64 exe fails on arch
mismatch (same error 126). NOT fixable in place -- needs a native ARM64 GenAiTransformer backend that is not
on disk. **Second, deeper reason to abandon PATH A even if it loaded:** GenAiTransformer uses a **CpuOpPkg
(CPU)** -- it is a CPU backend, NOT the HTP/NPU. So PATH A was never the NPU path. => the real NPU route is
**PATH B (QnnHtp v73 context binaries)** below. The compose artifacts (`model.bin`, `genie_config.json`,
`qwen2.5-1.5b-composer.json`) sit in `scratchpad\genie-qwen15\` if a native ARM64 backend ever appears.

Extra gotchas found (kept for reference, not that PATH A is viable here):
- Composer auto-detect only knows **Qwen1** (`QWenLMHeadModel`); Qwen2.5 (`Qwen2ForCausalLM`) falls through and
  needs a hand-built `--config_file` (generic llama3.2-1b template + QKV biases `attention_{q,k,v}_bias` + Qwen
  dims: n_embd 1536, n_ff 8960, 28 layers, 12 heads, 2 KV heads, n_rot 128, rope 1e6, eps 1e-6, tied embeddings).
- The `PROCESSOR_IDENTIFIER=AMD64` add_dll_directory workaround is necessary but NOT sufficient: the Q4 quantizer
  loads via `ctypes.util.find_library(...)` which searches **PATH**, not add_dll_directory -- must prepend
  `<sdk>\lib\x86_64-windows-msvc` to PATH, and run from **PowerShell** (Git Bash MSYS mangles PATH -> find_library
  returns None -> `TypeError: NoneType is not iterable`).

Superseded historical note (was: "recommended for LLMs"):
One tool does HF -> quantized Genie binary; this is the documented Windows tutorial path
(`docs/Genie/general/tutorials/dialog/llama-2-7b/genai/windows/windows.html`). Composer supports
`llama`/`qwen`/`gpt2` architectures -- covers Llama-3.2-1B and Qwen2.5-1.5B.
1. `git lfs` clone the HF model dir (e.g. `meta-llama/Llama-3.2-1B`, `Qwen/Qwen2.5-1.5B`).
2. Compose (x64-emulated Py3.10). **DLL-dir workaround:** `qti/aisw/genai/__init__.py` forgets
   `os.add_dll_directory` on the `"ARMv8"` branch, so set `PROCESSOR_IDENTIFIER` to a string containing
   `AMD64` to take the branch that adds `<sdk>/lib/x86_64-windows-msvc` (which holds the composer's x64
   `QnnGenAiTransformerComposerQuantizer.dll`):
   ```
   QNN_SDK_ROOT=<sdk> PYTHONPATH=<sdk>/lib/python PROCESSOR_IDENTIFIER="AMD64 (emulated)" \
   python <sdk>/bin/x86_64-windows-msvc/qnn-genai-transformer-composer \
     --model <hf_dir> --quantize Q4 --export_tokenizer_json --outfile model.bin
   ```
   `--quantize`: **`Q4`** = block-32 INT4, highest accuracy (task target); `Z4`/`Z8` = block-128 INT4/INT8,
   faster. `--export_tokenizer_json` writes a HF `tokenizer.json` next to `--outfile`.
3. Genie config: copy `examples/Genie/configs/llama2-7b/llama2-7b-genaitransformer.json`; set backend
   `type:"QnnGenAiTransformer"` with `n-layer`/`n-embd`/`n-heads` matching the model, `model.library.model-bin`
   = `model.bin`, `tokenizer.path` = exported `tokenizer.json`, `context.n-vocab`/`bos`/`eos` per the model.
4. Run natively: `genie-t2t-run.exe -c genie_config.json -p "..."` (deps on PATH from
   `lib/aarch64-windows-msvc`: `Genie.dll`, `QnnGenAiTransformer.dll`, `QnnGenAiTransformerCpuOpPkg.dll`,
   `QnnGenAiTransformerModel.dll`, `QnnSystem.dll`).
   Risk: composer is x64-emulated + arch-limited; needs the DLL-dir workaround; unproven on a full model.

#### PATH B (general fallback) -- QnnHtp context binaries  [~] pipeline VERIFIED end-to-end (Relu), LLM front pending
Use when a model isn't a composer-supported arch, or you want the QnnHtp backend. Steps 2-4 were RUN on
this box successfully; step 1 (the hard part) is not yet.
1. **HF -> ONNX with KV cache** [ ] NOT YET RUN: `python -m onnxruntime_genai.models.builder -m <hf_id>
   -o <onnx_dir> -p int4 -e cpu` (Microsoft's builder; emits an LLM ONNX with KV-cache I/O, optionally
   split prompt/token graphs). This is the biggest unknown -- see risks. (`qairt-converter` can also take a
   `.gguf` directly via `--gguf_config`, an alternate front.)
2. **ONNX -> float DLC** [x] VERIFIED:
   `python <sdk>/bin/arm64x-windows-msvc/qairt-converter -i model.onnx -o model.dlc --target_backend HTP`
   (`--target_soc_model` optional; omit -> backend-generic HTP graph.)
3. **DLC -> INT4/INT8 DLC** [x] VERIFIED (INT8):
   `python <sdk>/bin/arm64x-windows-msvc/qairt-quantizer -i model.dlc -o model_quant.dlc
   --input_list calib.txt --weights_bitwidth 4 --act_bitwidth 16 --use_per_row_quantization`
   (`--weights_bitwidth 4` = native INT4 weights; `--use_per_row_quantization` = rowwise for MatMul/FC, the
   LLM-relevant knob; `--act_bitwidth 16` keeps activations at a16 for LLM accuracy. `calib.txt` lists raw
   input files, one `name:=path` per line.)
4. **Quantized DLC -> v73 context binary** [x] VERIFIED (`"dspArch":73`):
   ```
   qnn-context-binary-generator.exe --backend <sdk>/lib/aarch64-windows-msvc/QnnHtp.dll \
     --dlc_path model_quant.dlc --config_file htp_config_v73.json \
     --binary_file model_v73 --output_dir out
   ```
   `--dlc_path` needs `QnnModelDlc.dll`; graph-prepare needs `QnnHtpPrepare.dll` + `QnnHtpV73Stub.dll`
   (all in `lib/aarch64-windows-msvc`, put that dir on PATH). **v73 targeting** goes in the config JSON:
   ```json
   { "devices": [ { "dsp_arch": "v73", "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ] }
   ```
   `dsp_arch:"v73"` maps to `QNN_HTP_DEVICE_ARCH_V73=73` (`include/QNN/HTP/QnnHtpDevice.h`). Note: the X Elite
   SoC (SC8380) is **not** in the public `QNN_SOC_MODEL_*` enum (`QnnTypes.h` stops at 59), so use
   `dsp_arch` -- do NOT try to guess a `soc_model` int. On-device you may also omit it (auto-detect the local
   v73). LLMs emit multiple `.bin`s (prompt + token graphs); list them all.
5. Genie config: copy `examples/Genie/configs/llama2-7b/llama2-7b-htp-windows.json`; set
   `model.binary.ctx-bins` = [the `.bin`(s)], `tokenizer.path`, `context.size/n-vocab/bos/eos`, and
   `engine.backend.extensions` = `examples/Genie/configs/htp_backend_ext_config.json` (burst profile).
6. Run: `genie-t2t-run.exe -c genie_config.json -p "..."` (deps: `Genie.dll`, `QnnHtp.dll`,
   `QnnHtpV73Stub.dll`, `QnnSystem.dll`).

#### Biggest risks / open blockers
- **HF -> LLM-ONNX (Path B step 1) is the only unproven stage** and the hardest: it must emit correct
  KV-cache I/O and (for larger models) a prompt/token graph split. `onnxruntime-genai.models.builder` or
  Qualcomm AI Hub automate it; hand-rolling `torch.onnx` will not produce a Genie-shaped graph.
- **Converter MatMul quirk (minor):** hand-built 2D/3D `MatMul`-with-initializer ONNX tripped the converter's
  rank-alignment pass (`ReshapeOp::calculateShape ... 512 != 568`) on BOTH arm64ec and x86_64 native modules
  -- so it's a converter edge case with toy graphs, **not** an arch bug. Real builder-exported ONNX takes the
  validated path; Relu converted cleanly. Watch for it on unusual custom ops.
- **Path A composer is x64-emulated + arch-limited + needs the `PROCESSOR_IDENTIFIER=AMD64` DLL-dir
  workaround.** No native ARM composer exists; the GenAiTransformer *runtime* is native, the *builder* is not.
- Quantization quality of native INT4 (a16w4) vs the GGUF Q4/Q6 baselines is still to be measured once a real
  model is through.

Deliverable: `docs/MODEL_CONVERSION.md` upgraded from options-doc to this runnable recipe.
Exit criterion: a self-converted small dense model (Llama-3.2-1B / Qwen2.5-1.5B) runs under Phase 1.

### Phase 3 -- serving for the agent workload [ ]
Goal: a drop-in local endpoint the agent config can point at.
- [ ] Wrap Genie (C API via ctypes, or the onnxruntime-genai server) in an OpenAI-compatible
      HTTP server (`/v1/chat/completions`, streaming).
- [ ] Prompt-cache / KV-reuse for agent multi-turn (Genie supports session save/restore).
- [ ] Benchmark against the llama.cpp CPU+GPU numbers on the same prompts; report prefill/decode
      and power (NPU's real edge: sustained + efficient).
Deliverable: `src/server.py` + a benchmark table.
Exit criterion: the agent config runs against the NPU endpoint end-to-end.

### Phase 4 -- optional llama.cpp bridge [ ]
Only if you want llama.cpp's ecosystem (GGUF, samplers, grammar) on the NPU:
- [ ] An out-of-process RPC backend: llama.cpp's `ggml-rpc` client talks to a Genie/ONNX NPU
      worker process that is killable on hang. This isolates the driver-hang risk out of the
      main process. Likely unnecessary if Phase 3 serving is enough.

---

## Why Genie also wins the DECODE bottleneck (speculative decoding)

Every measurement this session showed decode is **memory-bandwidth-bound** on the shared LPDDR5x --
the whole 8-12GB model streams per token, so NPU/GPU/CPU all plateau around the same low tg (the
14B GPU decode was 4.46 tg). Speculative decoding attacks exactly this: draft K tokens, verify them
in ONE weight-streaming pass, so throughput scales with the acceptance rate instead of one token per
memory sweep. **Genie ships optimized speculative methods natively, configured purely in the Dialog
JSON (no external code):**
- **SSD (Self-Speculative Decoding)** -- model drafts its own future tokens via early-exit/simplified
  heads, **no external draft model needed**. Easiest to enable (`"dialog_type": "ssd"`). Try FIRST.
- **Eaglet (EAGLE-based)** -- a trained draft head predicting the target's features; highest acceptance
  rates. Needs the draft-head artifact. Best steady-state decode.
- **LADE (Lookahead Decoding)** -- parallel n-gram speculation, no draft model.

Implication: the decode wash we measured is a *baseline*, not a ceiling. Once a Genie bundle runs
(Phase 1/2), enabling SSD/Eaglet is a config-only change that can multiply decode tg -- the single
biggest lever for the agent workload, and something the llama.cpp eager backend never offered.
=> This makes Genie the clear runtime target: it fixes both the robustness wall (AOT) AND the decode
bottleneck (native speculative), with zero extra code.

Added to Phase 3 (serving): after a bundle runs, sweep `dialog_type` = basic -> ssd -> lade -> eaglet
and record tg uplift + acceptance rate per model.

## Decision log
- 2026-08-22: chose AOT (Genie/ONNX) over patching the eager llama.cpp backend -- the per-op-JIT
  hang is architectural, and a load-time-prevalidation / out-of-process-worker fix for llama.cpp
  would reimplement Genie. Genie is in the SDK already.
- 2026-08-22: Genie confirmed as the runtime target (not ONNX-genai): it natively solves the decode
  bottleneck via speculative decoding (SSD/Eaglet/LADE, JSON-configured) on top of the AOT robustness
  win. ONNX-genai stays the portability-only fallback.
- 2026-08-22: Genie-vs-ONNX-RT-genai SETTLED for decode. Reported (vendor/single-sourced, TO VALIDATE):
  ONNX-RT genai lists speculative decoding as "on the roadmap" (not in stable), while Genie ships it;
  Genie claims ~4x decode speedup combining speculative decoding + MXFP6 quant, and the efficiency
  comes from FUSED AOT VERIFICATION -- draft + verify compiled into ONE context binary, fusing the
  draft-KV and target-verify memory ops to minimize NPU<->RAM movement (which IS the decode
  bottleneck). Also a claim: ~40% gain on heterogeneous hardware. THESE ARE VENDOR CLAIMS -- Phase 3's
  `dialog_type` sweep (basic/ssd/lade/eaglet) + a MXFP6-vs-int4 pass will measure the ACTUAL uplift on
  our v73/32GB box and models; do not quote the 4x/40% as measured until then. Net: another reason
  Genie is the target -- ONNX-RT can't match decode without spec decoding it doesn't yet ship.
  New lever noted: MXFP6 (microscaling FP6) quantization as a Genie option alongside int4/int8.

## Open questions / risks
- Does a prebuilt Genie model exist for a Qwen3 / Llama size that fits 32GB, or is Phase 2
  conversion required first? (Resolve in Phase 1 step 1.)
- Genie's Windows-ARM64 model artifacts must match this HTP arch (v73). Verify context-binary
  target arch on load.
- Quantization quality: native INT4 on HTP vs the GGUF Q4/Q6 baselines -- compare perplexity
  once a converted model exists (ties into the model-selection quality pass).
