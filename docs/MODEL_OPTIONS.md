# Model options: what serves where, and why

Three models, three launch paths, one box. Added 2026-09-03 when "let's also
serve Qwen3.5-9B beside the Qwen3-4B" turned out to fork on a fact worth
recording: the newest model here is the one the NPU cannot run.

| model | engine | launch | port | id at /props |
|---|---|---|---|---|
| Qwen3-4B (default) | NPU (Hexagon HTP) | `src\run-genie-server.ps1` | 8123 | `qwen3-4b-npu` |
| Qwen3-8B | NPU (Hexagon HTP) | `src\run-genie-server.ps1 -Model qwen3-8b` | 8123 | `qwen3-8b-npu` |
| Qwen3.5-9B Q8_0 | CPU (KleidiAI) | `src\run-llama-server.ps1` | 8080 | `unsloth/Qwen3.5-9B-GGUF:Q8_0` |
| Qwen3.5-9B Q4_K_M | GPU (Adreno OpenCL) | `src\run-llama-server.ps1 -Leg gpu` | 8124 | `qwen3.5-9b-gpu` |

The two llama-server rows are ONE model with two leg-specific quants -- see
"the quant is per-leg" below. All four speak OpenAI chat completions; the
Genie rows also speak Anthropic `/v1/messages`.

## Why Qwen3.5-9B is not a Genie bundle, and cannot currently become one

Checked 2026-09-03, three independent ways:

- **No export target.** `qai-hub-models` has no `qwen3_5_9b` -- confirmed in
  the WSL export venv (0.60.0) and on GitHub main. It is an open feature
  request (qualcomm/ai-hub-models#287, closed as duplicate of the #221
  aggregator). The 9B slot upstream is empty, not merely unbuilt here.
- **The Qwen3.5 entries that DO exist are not Genie bundles.** `qwen3_5_0_8b`
  and `qwen3_5_2b` ship no `export.py` -- they are fetch-only release assets,
  and the asset is a plain GGUF pointed at the `geniex_llamacpp` runtime
  (GenieX is Qualcomm's llama.cpp-based runtime). Nothing in that chain
  produces the HTP context binaries `genie_server.py` loads.
- **The architecture is different in kind, not just size.** Qwen3.5 is
  `qwen3_5` -- hybrid linear attention, three of every four layers linear,
  `Qwen3_5ForConditionalGeneration` -- where everything the classic Genie
  chain compiles here is standard-attention `qwen3`. There is also an open
  GenieX crash report for Qwen3.5 on its NPU backend (qualcomm/GenieX#1178),
  so even the llama.cpp-on-Hexagon route is not there yet for this family.

When qai-hub-models grows a `qwen3_5_9b` export target, the normal chain
(`docs/IMPLEMENTATION_PLAN.md`, `export-8192-multi.sh` in the artifacts dir)
should apply unchanged. Until then the 9B serves through llama.cpp.

## The Qwen3.5-9B llama-server legs

`src\run-llama-server.ps1` encodes the operator's known-good serving command
(fork build `llama-qnn-fork\build-arm64-windows-llvm-release` -- it carries
the agent-mode flags and its llama-server initialises OpenCL; an earlier
build's server could not reach the Adreno at all). Env overrides: `LLAMA_HF`
/ `LLAMA_GGUF`, `LLAMA_HOST`, `LLAMA_PORT`, `LLAMA_CTX`, `LLAMA_THREADS`,
`LLAMA_ALIAS`, `LLAMA_SLOT_DIR`, `LLAMA_BIN_DIR`, `LLAMA_EXTRA_ARGS`,
`LLAMA_HEALTH_TIMEOUT`.

**The quant is per-leg, and the ranking inverts between legs.**

- **CPU leg (default): Q8_0.** KleidiAI's int8 kernels accelerate Q4_0 and
  Q8_0 -- and nothing else. The build says so itself when handed the wrong
  one: `kleidiai: no kernel for tensor type q4_K, not accelerated by KleidiAI
  (kernels available for Q4_0 and Q8_0)`. Fetched by the server via
  `-hf unsloth/Qwen3.5-9B-GGUF:Q8_0` (~9.5 GB, HF cache, once).
- **GPU leg (`-Leg gpu`): Q4_K_M.** The OpenCL SOA_Q / Adreno kernels target
  Q4: measured on this box (Qwen3-4B, d0), Q4_K_M decodes 19.39 t/s against
  Q8_0's 9.14 -- the bigger file is also the slower one there. Local file at
  `<root>\gguf\Qwen3.5-9B-Q4_K_M.gguf`
  (`hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir <root>\gguf`).

Do not swap either quant onto the other leg for "quality"; each pays roughly
2x throughput on the wrong engine.

**Placement is verified, not assumed.** The launcher greps the startup log
for the `using device GPUOpenCL` line (passing `-lv 5` on the gpu leg,
because at default verbosity this build prints no device line at all) and
warns loudly when it is absent -- a llama-server has been observed on this
box silently serving from the CPU while asked for the GPU. Smoke-verified
2026-09-03: `offloaded 33/33 layers to GPU`, GPU engine counter 59% during
decode, server-reported **prefill 36.2 t/s, decode 5.6 t/s** -- taken on a
warm, loaded box (the CPU 9B resident, ~4-6 GB free RAM), so treat as a
floor, not a rate.

The CPU leg's alias is deliberately the bare `-hf` spec
(`unsloth/Qwen3.5-9B-GGUF:Q8_0`): that is what the hand-run instances have
always advertised, and a client keyed on it would break if the launcher
renamed it. The GPU leg is new, so it gets a clean `qwen3.5-9b-gpu`.

## The Qwen3-8B NPU tier

One generation older than the 9B, similar size, and it DOES have the full
Genie path. Installed from the AI Hub prebuilt:

```bash
qai-hub-models fetch qwen3_8b -r genie -p w4a16 -c qualcomm-snapdragon-x-elite --extract -o <dir>
```

(4.57 GB download, 5.2 GB extracted, QAIRT 2.45 -- matches the installed
runtime.) The bundle landed in `<root>\bundles\
qwen3_8b-genie-w4a16-qualcomm_snapdragon_x_elite` and is **multi-length
[512, 1024, 2048, 3072, 4096]** out of the box, same as the 4B prebuilt --
so it pays for context in use, not for its whole window.

Per-machine fixes applied on install, per this repo's own findings
(`genie_config.json.orig` keeps the shipped copy):

- `poll: false` -- shipped `true`, the 2.7-idle-core busy-wait.
- `token-penalty` block (1.15 / last-n 128 / freq 0.3) -- shipped absent, so
  nothing suppressed a repetition loop.

`run-genie-server.ps1 -Model qwen3-8b` selects it and reports
`qwen3-8b-npu`; an explicit `-Model` also overrides a lingering
`GENIE_BUNDLE_DIR` / `GENIE_MODEL_ID` from the environment, so a stale shell
export cannot hand the 8B bundle the 4B's id. Smoke-verified 2026-09-03:
loads in 18.7s, `/props` reports the multi-length list, and at the
steady-state config (`poll: false`, interrupts healthy -- see the next
section) 110 tokens in 9.5s end-to-end, ~12 t/s effective.

Expect roughly half the 4B's decode at the same window -- decode here is
bandwidth-bound and the 8B moves ~2x the weight bytes per token. Its window
is 4096 (the prebuilt's); an 8192 multi-length export via the normal chain
is the upgrade path if the window matters more than the download.

## 2026-09-03: interrupt delivery can degrade, and then `poll: false` is the slow setting

Found during the 8B smoke test, isolated by A/B on the 4B, same box state,
minutes apart, identical request:

| `poll` | same 100-token request |
|---|---|
| `false` | **>120s, ~0.3 t/s** (both 4B and 8B) |
| `true` | **5.8s** (4B), ~8s (8B) |

With `poll: false` the driver waits on an HTP interrupt per token; in this
degraded state every wait ate ~3s, a ~36x collapse that looks exactly like a
broken model or a dying bundle and is neither. Busy-polling bypasses the
interrupt path entirely, which is what makes the A/B diagnostic: **crawling
at `poll: false` but normal at `poll: true` means interrupt delivery is
degraded, not that the bundle or the model is bad.** Not memory pressure
(weights resident, ~4 page reads/sec), not thermals (the poll:true run was
fast on the same warm box), not load (CPU mostly idle).

**The fix is a driver restart, no reboot needed -- verified same day.** From
an elevated **PowerShell** (double-quoted so it also survives cmd.exe, where
single quotes are literal and the bare `&`s split the command):

```
pnputil /restart-device "ACPI\QCOM0D0A\2&DABA3FF&0"
```

(the "Snapdragon X Elite - Hexagon NPU" ComputeAccelerator node; enumerate
with `Get-PnpDevice` if the instance id differs on another box). Measured:
the identical request went from >120s before the restart to **4.6s** after,
at `poll: false`. The `aihost.exe` / `AIXHost.exe` pair kept the same PIDs
through the restart, so this state lives in the driver itself -- it is a
DIFFERENT failure mode from the unreapable-aihost AIX wedge this box has
also exhibited (that one does need a reboot).

Both bundles are LEFT at `poll: false` -- the steady-state setting this repo
measured 1.45x-vs-0.78x arguments for -- because its failure mode here is
loud (a crawl you cannot miss), while `poll: true` left in place after the
driver heals fails silently (2.7 idle cores, up to -55% decode, NPU+GPU
concurrency inverted). If a Genie server crawls at ~0.3 t/s: restart the
device as above and re-test.

One more consequence worth writing down: the 4B numbers throughout these
docs were measured with interrupts healthy. A number taken in the degraded
state is not comparable to any of them, whatever the poll setting.
