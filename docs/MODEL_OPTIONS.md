# Model options: what serves where, and why

Three models, three launch paths, one box. Added 2026-09-03 when "let's also
serve Qwen3.5-9B beside the Qwen3-4B" turned out to fork on a fact worth
recording: the newest model here is the one the NPU cannot run.

| model | engine | launch | port | id at /props |
|---|---|---|---|---|
| Qwen3-4B (default) | NPU (Hexagon HTP) | `src\run-genie-server.ps1` | 8123 | `qwen3-4b-npu` |
| Qwen3-8B | NPU (Hexagon HTP) | `src\run-genie-server.ps1 -Model qwen3-8b` | 8123 | `qwen3-8b-npu` |
| Qwen3.5-9B Q4_0 | CPU (KleidiAI) | `src\run-llama-server.ps1` | 8080 | `unsloth/Qwen3.5-9B-GGUF:Q4_0` |
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

When qai-hub-models grows a `qwen3_5_9b` export target, the normal chain in
`docs/IMPLEMENTATION_PLAN.md` should apply unchanged. Until then the 9B serves
through llama.cpp.

## GenieX: Qwen3.5 DOES reach the NPU now -- it is just not worth it yet (2026-09-03)

The "GenieX crashes on Qwen3.5 NPU" caveat above is STALE: qualcomm/GenieX
issue #1178 was fixed in PR #1248 (merged 2026-08-03 -- the VLM projector was
initialising the OpenCL backend on NPU runs), and GenieX CLI v0.5.0 carries
the fix. Installed and verified on this box (installer:
`qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-cli.exe`,
lands in `%LOCALAPPDATA%\GenieX CLI`, bundles QAIRT 2.45 + a llama.cpp
runtime with the experimental ggml-hex Hexagon backend):

```
geniex pull unsloth/Qwen3.5-9B-GGUF:Q4_0
geniex infer unsloth/Qwen3.5-9B-GGUF:Q4_0 --compute npu --think=false -p "..."
```

Placement is real -- the debug log shows `ggml-hex: Hexagon Arch version
v73`, an HTP0 session allocated over FastRPC, and layers assigned to HTP0.
(No NPU perf-counter set exists on this box, so the log is the only placement
probe.) But single-sample rates, warm loaded box, no cool gate -- indicative
only:

| Qwen3.5 Q4_0 | `--compute npu` | `--compute cpu` | `hybrid` |
|---|---|---|---|
| 0.8B | 28.3 t/s | **56.8** | -- |
| 9B | 6.7 t/s | **16.0** | 9.6 |

CPU beats the ggml-hex NPU path ~2.4x at both sizes, so GenieX buys no
throughput today -- the fork llama-server legs above remain the way to serve
this model. What the probe DID establish: Qwen3.5 executes on the Hexagon
without crashing (fix confirmed), `hybrid` does not crash on X Elite (the
still-open #1250 hybrid crash is a QCS9075/OpenCL issue), and the ggml-hex
path is a second, GGUF-native road to the NPU that needs no AI Hub export --
worth re-probing each GenieX release, since the backend is marked
experimental and 6.7 t/s is already within 2x of what a native w4a16 Genie
bundle of this size should do (~8 t/s by bandwidth scaling from the 8B's
~12). Note `geniex infer` takes catalogue/HF names only -- it cannot serve a
local GGUF path, so its cache duplicates any GGUF the llama legs already
have.

## The Qwen3.5-9B llama-server legs

`src\run-llama-server.ps1` encodes the operator's known-good serving command
(fork build `llama-qnn-fork\build-arm64-windows-llvm-release` -- it carries
the agent-mode flags and its llama-server initialises OpenCL; an earlier
build's server could not reach the Adreno at all). Env overrides: `LLAMA_HF`
/ `LLAMA_GGUF`, `LLAMA_HOST`, `LLAMA_PORT`, `LLAMA_CTX`, `LLAMA_THREADS`,
`LLAMA_ALIAS`, `LLAMA_SLOT_DIR`, `LLAMA_BIN_DIR`, `LLAMA_EXTRA_ARGS`,
`LLAMA_HEALTH_TIMEOUT`.

**The quant is per-leg, and the ranking inverts between legs.**

- **CPU leg (default): Q4_0.** KleidiAI's int8 kernels accelerate Q4_0 and
  Q8_0 -- and nothing else. The build says so itself when handed the wrong
  one: `kleidiai: no kernel for tensor type q4_K, not accelerated by KleidiAI
  (kernels available for Q4_0 and Q8_0)`. Fetched by the server via
  `-hf unsloth/Qwen3.5-9B-GGUF:Q4_0` (~5.4 GB, HF cache, once). **It was Q8_0
  until 2026-09-03 -- see the correctness note below; Q8_0 does not generate
  on this build, and Q4_0 is both the working quant and a KleidiAI one, so
  the switch gives up nothing.**
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

## Qwen3.5-9B Q8_0 does not generate on this build (2026-09-03)

The CPU leg served **empty completions** -- one token, `finish_reason: stop`,
`content: ""` -- for every request. Not the chat template: the raw
`/completion` endpoint, which bypasses the template entirely, returned
`stop_type: eos` after one token too. Not the flags, not the backend. Same
build, same CPU backend, same minimal args, varying only the quant:

| quant | output | decode |
|---|---|---|
| **Q8_0** | empty, or a run of bare newlines | 66 t/s -- physically impossible for a 9.5 GB model on 6 cores, the tell that it was not computing the model |
| Q4_K_M | coherent | 10.40 t/s |
| **Q4_0** | coherent | **12.00 t/s** |

So the CPU leg default is now Q4_0, verified end-to-end through the launcher
with the full production flag set (11.73 t/s, coherent answer). The GPU leg's
Q4_K_M was never affected. The launcher warns on any Q8_0 selection rather
than silently serving nothing.

Worth stating plainly because it is the kind of failure that hides: the server
starts, reports healthy, answers every request with HTTP 200, and returns an
empty string. `/health` and `/props` cannot see it. **A launcher smoke test
that only checks health would pass on a server that generates nothing** --
check for non-empty content.

The CPU leg's alias is deliberately the bare `-hf` spec
(`unsloth/Qwen3.5-9B-GGUF:Q4_0` since the quant correction; it read `:Q8_0`
while that was the default): passing no `-a` means the served id tracks the
quant, and a client keyed on the old string needs updating once, here. The GPU leg is new, so it gets a clean `qwen3.5-9b-gpu`.

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
bandwidth-bound and the 8B moves ~2x the weight bytes per token.

## The 8B 8192 multi-length tier (2026-09-04)

`run-genie-server.ps1 -Model qwen3-8b-8192` serves a self-exported 8192
multi-length build (`qwen3_8b-genie-w4a16-x-elite-ctx8192-multi`, id
`qwen3-8b-8192-npu`) -- twice the prebuilt's window, `context_lengths
[512, 1024, 2048, 4096, 8192]` so a short prompt still runs against the
smallest graph that fits. The 4096 prebuilt stays the default 8B: on this
engine a bigger compiled window is a per-token tax, not a free upgrade.

Measured decode, boundary-safe depths (prompt + generated tokens inside one
compiled length):

| depth | decode | note |
|---|---|---|
| 250 | **10.92 t/s** | same-depth noise 0.13 t/s over n=3 |
| 978 | 10.08 t/s | flat with depth, as multi-length predicts |

**Condition, because it matters:** taken with the box ON BATTERY at 33%
pack, clock sampled 22-49% of base. Decode is largely immune to pack state
on this hardware (17.70 charging at 33% against 17.91 settled elsewhere in
these docs) and the samples are tight, so treat these as sound for decode
and do NOT quote any prefill figure from that run.

**Install notes.** The export's own directory name is byte-identical to the
prebuilt's (`qwen3_8b-genie-w4a16-qualcomm_snapdragon_x_elite`), so it MUST
be renamed on install or it silently overwrites the working 4096 bundle.
Both per-machine fixes were applied (`poll: false`, token-penalty
1.15/128/0.3) with `genie_config.json.orig` kept beside them.

**Export provenance.** Three attempts, ~9 hours. The first died after
uploading all five parts without creating a single AI Hub job (nothing
recoverable). The second was killed mid-upload when WSL ITSELF restarted --
`uptime: up 0 minutes`, no EXIT line, no traceback, dmesg gone with the VM.
The third succeeded (`EXIT=0`, 4h40m) after capping WSL memory in
`.wslconfig` (10 GB + 16 GB swap, against an unbounded ~15.8 GB default)
and freeing the ~6 GB the 9B leg held. Worth knowing that `EXIT=1` alone
does not mean failure here: the 4B's own successful export also exited 1,
throwing at `LINKING_MODELS` after its jobs existed, and was recovered by
job id. What separates the cases is whether AI Hub jobs were created --
check for job ids before concluding anything is lost.

**A WSL restart appears to degrade HTP interrupt delivery.** The first smoke
test of this bundle ran at ~0.5 t/s (87 tokens in flight after three
minutes) with `/health` reporting a healthy, generating engine. That is the
documented interrupt-degradation signature, and
`pnputil /restart-device "ACPI\QCOM0D0A\2&DABA3FF&0"` from an elevated
PowerShell restored it immediately -- 107 tokens in 10.7s. Suspect the
device, not the bundle, when a fresh export seems catastrophically slow.

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
measured concurrency arguments for -- because its failure mode here is
loud (a crawl you cannot miss), while `poll: true` left in place after the
driver heals fails silently (2.7 idle cores, up to -36% decode, NPU+GPU
concurrency inverted). If a Genie server crawls at ~0.3 t/s: restart the
device as above and re-test.

One more consequence worth writing down: the 4B numbers throughout these
docs were measured with interrupts healthy. A number taken in the degraded
state is not comparable to any of them, whatever the poll setting.
