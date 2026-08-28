# Genie NPU Server (OpenAI-compatible)

A local OpenAI-compatible HTTP endpoint backed by a Qualcomm Genie context-binary
bundle running on the Snapdragon X Elite NPU (Hexagon v73). The model is loaded
**once** (resident on the HTP via the Genie C API) so requests don't pay the
~35-50s reload that `genie-t2t-run.exe` would incur per invocation.

## Requirements

- **Native ARM64 Python** (aarch64). Genie.dll and its Qnn* deps are
  `aarch64-windows-msvc`; an x64/emulated Python cannot load them.
- The QAIRT 2.45 runtime (extracted) and a Genie bundle matching this box's
  Hexagon. The server DERIVES the supported set at startup by intersecting
  `lib/hexagon-v*/unsigned` (DSP skel) with
  `lib/aarch64-windows-msvc/QnnHtpV*Stub.dll` (Windows stub); an arch needs
  both. On QAIRT 2.45 that yields **v68, v73 and v81** -- the Hexagons with a
  Windows stub, covering 8cx Gen 3 through X2 Elite. The skels with no stub
  here (v66, v69, v75, v79) are Android parts and are reported as skipped
  rather than silently offered. `GENIE_HEXAGON_ARCH=v81` pins one arch.

  Do not read that list as a constant: it is whatever the installed SDK
  supports, which is the point of deriving it. This doc claimed "v73 and v81,
  the two Windows-on-Snapdragon parts" until a real startup log showed three --
  the mechanism was right and the hand-written example beside it was not.
- A bundle is locked to one arch AND one QAIRT version; a mismatch fails at
  `GenieDialog_create` with a message naming the archs this box can offer.

- No pip packages. Pure Python stdlib.

## Run

The Genie bundle and the QAIRT 2.45 runtime are large external artifacts and are
**not** in this repo. The normal way to run is the launcher, which finds them
itself -- it looks for a `genie-npu` directory beside this repo holding
`bundles/` and `qairt/`, and picks the newest QAIRT under it:

```powershell
cd <your clone of this repo>   # the path below is relative to the repo root
powershell -File src\run-genie-server.ps1
```

That serves on `127.0.0.1:8123`, and supervises: see Supervision below.

If the artifacts live elsewhere, point `GENIE_NPU_ROOT` at the directory
holding them, or set the two paths directly. The launcher checks both exist
before the ~10s model load and exits naming what it tried, rather than failing
deep inside the server:

```powershell
$env:GENIE_NPU_ROOT = "D:\genie-npu"
# or individually:
$env:GENIE_BUNDLE_DIR = "...\qwen3_4b-genie-w4a16-x-elite-ctx8192-multi"
$env:GENIE_SDK_DIR    = "...\qairt\2.45.0.260326"
```

Running `python src\genie_server.py` directly works too, but then nothing
supervises it and the port defaults to 8080 rather than 8123.

Startup prints `model resident on HTP in <N>s` then the endpoint URL, and
after that every request reuses the resident model.

Re-measured 2026-08-26 on the 8192 multi-length bundle (the launcher default),
because the figures here were an older bundle's and the startup string still
advertises a range no run has produced:

| | seconds | n |
|---|---|---|
| cold | 34.4 | 1 |
| warm | 10.8, 12.7, 15.0 | 3 |

State the spread, not the best sample: warm is a **10.8-15.0** band, not the
10.8 it is tempting to quote, and 39% separates its ends. The cold reading is
n=1 and its cold-ness was self-inflicted -- it followed a recursive grep over
the whole QAIRT SDK, which is exactly the kind of thing that evicts a 3 GB
bundle from the page cache. Treat it as "after heavy unrelated disk traffic"
rather than as a reboot-cold number.

Worth recording how that was checked, since a neighbouring session raised it:
the three readings above ran back-to-back, and on this box a sequential series
is normally suspect -- an identical leg repeated at the end of a four-leg sweep
came back 26% low on AC and 43% low on battery, with the clock sliding 87% to
69% of base (measured by that session on the llama.cpp path; see the thermal
note in `MULTI_ENGINE.md`, which this independently corroborates). That
mechanism is ruled out HERE, but by the data rather than by assertion: decay
predicts monotonically slower, and these went 34.4 -> 10.8 -> 15.0. The largest
is first and the series is not monotonic, which is a page-cache signature and
the opposite of a thermal one. The n=1 and the 39% warm spread stand
regardless.

## Endpoints

- `POST /v1/chat/completions` -- OpenAI chat API. Supports `messages`, `stream`
  (SSE), `max_tokens`, `stop` (also Anthropic `stop_sequences`), and `tools`.
  ChatML template is taken from the bundle's own
  `metadata.json` chat_template.
- `GET /v1/models` -- lists the served model id (`GENIE_MODEL_ID`).
- `GET /props` -- llama.cpp-shaped metadata: `default_generation_settings.n_ctx`
  and `model_alias` / `model_id`, which is what typed reads. Plus a namespaced
  `genie` block carrying what a router cannot otherwise learn over HTTP:

  | field | why a dispatcher needs it |
  |---|---|
  | `engine` | `npu-hexagon-htp` -- which silicon is answering |
  | `single_flight` | `true`; the constraint behind the 429/529, stated rather than discovered from one |
  | `context_lengths` | the graphs compiled into the bundle |
  | `multi_length` | `false` means 2-3x slower on short prompts at the SAME `n_ctx` |
  | `poll` | `true` means an idle 2.7-core busy-wait, and NPU+GPU concurrency is a 0.78x LOSS rather than a 1.45x gain |

  `n_ctx` alone is not enough to rank this endpoint against a GPU or CPU one,
  and on this engine it is actively misleading: it is the SOFTWARE cap
  (`dialog.context.size`), while throughput is set by the compiled window and by
  how many graphs the bundle carries. Two bundles reporting the same `n_ctx`
  differ 2-3x on a short prompt. The block is additive and namespaced, so a
  client that ignores it sees exactly the response it saw before.
- `GET /health` -- **engine** state, not process liveness. `200` when the
  server can actually generate; `503` with a `state` of `failing`, `stalled` or
  `wedged` when it cannot. The body carries `detail` (why), `generating`,
  `tokens_in_flight`, `generations` and `consecutive_failures`.

  The distinction is the entire point: a wedged HTP leaves this process
  perfectly able to accept a connection and answer this endpoint while unable
  to serve a single token, so a plain liveness ping reports healthy for as long
  as the outage lasts. The handler touches nothing on the engine, which is what
  lets it answer *during* a wedge -- the engine lock is exactly what the stuck
  thread is holding.

```bash
curl http://127.0.0.1:8123/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b-npu","messages":[{"role":"user","content":"Hi"}],"max_tokens":128}'
```

## Environment

| var | default | meaning |
|---|---|---|
| `GENIE_BUNDLE_DIR` | the 8192 multi-length bundle | dir with genie_config.json + part*_of_*.bin + tokenizer.json. **Prefer a MULTI-length bundle** -- check `genie.context_lengths` in its metadata.json; a single-length one is 2-3x slower on short prompts. |
| `GENIE_SDK_DIR` | scratchpad 2.45 SDK | QAIRT 2.45 root (lib/aarch64-windows-msvc, lib/hexagon-v*) |
| `GENIE_HEXAGON_ARCH` | unset | pin one skel arch (`v81`); default offers all |
| `GENIE_SUMMARIZE_EVICTED` | 1 | 0 disables summarising evicted turns (plain drop) |
| (not an env var) | -- | **`poll: false` in the bundle's `genie_config.json`** -- see the poll note below. Worth up to +55% decode and frees 2.7 idle cores. The server now CHECKS this at startup (before the 11-35s load) and warns loudly if the bundle ships `true`; it also warns on a single-length bundle. Both are warnings, never refusals -- a slow server is still a working one. |
| `GENIE_SUMMARY_MAX_TOKENS` | 192 | cap on the retained note. Clamped at runtime to `n_ctx / 8` (floor 32) so the note cannot crowd out the window on a small-context bundle; the server logs the clamp when it bites. |
| `GENIE_WINDOW_MARGIN` | 64 | headroom left between prompt and n_ctx |
| `GENIE_MAX_INFLIGHT` | 2 | requests admitted at once (1 running + queue). Floored at 1 -- it cannot be disabled, since the NPU is single-flight and an unbounded setting only parks threads on the engine lock. Set 1 to protect KV reuse: two interleaved conversations share one resident KV and reset each other's prefix. |
| `GENIE_HOST` / `GENIE_PORT` | 127.0.0.1 / **8080** | bind address. Note the launcher overrides the port: `run-genie-server.ps1` sets **8123** because 8080 usually collides with a llama-server. So the endpoint is `127.0.0.1:8123` when started the normal way, and `127.0.0.1:8080` only if you run `genie_server.py` directly. |
| `GENIE_MODEL_ID` | qwen3-4b-npu | id reported to clients |
| `GENIE_NPU_ROOT` | `../genie-npu` beside this repo | where `bundles/` and `qairt/` live. Set this instead of the two paths above; the newest `qairt/*` is picked automatically. |
| `GENIE_FIRST_TOKEN_TIMEOUT` | 300 | seconds a generation may run before its first token before being called stalled. Generous because prefill at depth legitimately takes tens of seconds. |
| `GENIE_STALL_TIMEOUT` | 120 | seconds between tokens before being called stalled. This is the real wedge signal -- see Supervision. |
| `GENIE_WEDGE_GRACE` | 60 | seconds an abort gets to take effect before the stall is escalated to a wedge |
| `GENIE_FAIL_THRESHOLD` | 3 | consecutive failed generations before `/health` reports `failing` |
| `GENIE_WEDGE_EXIT` | 1 | `0` keeps the process up on a wedge (it stays 503) instead of exiting for a supervisor |
| `GENIE_MAX_RESTARTS` | 5 | launcher only: rapid restarts before it gives up |
| `GENIE_RESTART_COOLDOWN` | 25 | launcher only: seconds between restarts. Not arbitrary -- a force-killed server needs roughly 20s of settling, and restarting sooner was measured costing about half of decode throughput. |
| `GENIE_MAX_TOKENS` | 512 | default cap when a request omits max_tokens |
| `GENIE_STRIP_THINK` | 0 | 1 strips a well-formed `<think>...</think>` pair from non-streamed content. Unrelated to the orphan-close strip below, which is always on because it removes a DUPLICATED answer rather than the model's reasoning. |
| `GENIE_SEED` | unset | pins the sampler seed. Unset means a fresh seed per PROCESS, which is what stops every fresh prompt replaying the same answer -- the bundles ship a fixed `42` and Genie re-seeds from it on every dialog reset. Pin it for reproducibility (comparing bundles, bisecting a bad generation); throughput does not depend on it. Per-REQUEST variation is not available -- see the note below. |
| `GENIE_ORPHAN_HOLD_CHARS` | -1 | how much of a STREAM to withhold while deciding whether the model is about to close a `<think>` block the prefill opened. `-1` holds until that is settled (so a streamed reply arrives as one frame at the end -- correct, not incremental). `0` streams every chunk as it arrives and ships the occasional doubled answer. A positive value is a bounded hold, which was measured LEAKING. Non-streaming and tool paths are unaffected; they buffer anyway and always strip. |
| `GENIE_THINKING` | **0** | Qwen3's reasoning block is **suppressed by default** -- it costs 10-17x on an agent turn (see the tool-calling note below). `1` re-enables it server-wide. Per request either way: `chat_template_kwargs.enable_thinking`, `reasoning_effort` (`"none"` / `"high"`), or `thinking:{"type":"disabled"|"enabled"}` -- an explicit request always beats the server default. |

## Supervision: what happens when the HTP wedges

The Genie query is a blocking call into native code. When the device stops
making progress the calling thread is stuck inside the driver holding the
engine lock, and Python cannot reclaim a thread blocked in native code -- no
timeout, no interrupt, no kill. Later requests park behind that lock until
`GENIE_MAX_INFLIGHT` is exhausted and the rest get a fast `429`, so from
outside this looks like a server that 429s forever while sitting idle.

Detection is by **stalled progress, not elapsed time**. A long generation is
not a wedge -- 2000 tokens at the slowest measured 3.3 t/s is ten minutes of
healthy work -- but it emits tokens the whole way, and a wedge emits nothing.
Time since the last token separates slow from stopped without capping how long
a request may legitimately run.

Escalation, in order:

1. **Stalled** -- no first token in `GENIE_FIRST_TOKEN_TIMEOUT`, or no further
   token in `GENIE_STALL_TIMEOUT`. `/health` goes 503; the server signals
   Genie's abort, which is free and is the mechanism provided for exactly this.
   On a healthy device this is usually where it ends: forced against a live
   generation, the abort landed and the engine returned to `ok` on its own.
2. **Wedged** -- the abort did not take within `GENIE_WEDGE_GRACE`. Nothing
   in-process can help, so the server exits **75** (`EX_TEMPFAIL`) and asks to
   be replaced. `GENIE_WEDGE_EXIT=0` keeps it up and reporting 503 instead.
3. **Restarted** -- `run-genie-server.ps1` restarts on exit 75 and only on 75;
   any other code is a deliberate exit (Ctrl-C, a config error it already
   explained) and repeating it would be pointless. Restarts are capped and
   rate-limited, because looping on a device that wedges every time keeps the
   HTP busy and buries the original failure under identical log stanzas. Only
   restarts following a short life count toward the cap, so a server that ran
   for hours and wedged once does not share a budget with one wedging at
   startup.

Separately, `consecutive_failures` reaching `GENIE_FAIL_THRESHOLD` reports
`failing` on `/health` **without** restarting: the engine is answering, just
badly, and restarting on that would turn a bad bundle into a crash loop.

## Notes / limitations

- **Context window: evict, don't crash.** The compiled window is fixed (read
  from the bundle, reported at `/props`; the bundles here are 4096, 8192
  single-length, 8192 multi-length -- the launcher default -- and 16384) and Genie has NO sliding-window mode -- QAIRT 2.45 exposes no
  such flag on `genie-t2t-run` and no equivalent config key, and overflowing is
  a hard `GenieDialog_query` failure, not a truncation. So the server evicts:
  oldest turns are dropped until the prompt fits, with the system turn and tool
  schemas anchored and tool results never separated from the call that produced
  them. Eviction is logged (never silent). A single message too big to fit even
  alone gets a 400 naming the token counts, not a doomed query.
  `GENIE_WINDOW_MARGIN` (default 64) is the headroom left for generation.

- **Eviction summarises instead of discarding.** Dropping the oldest turns
  outright makes the agent forget it already read a file and read it again --
  burning the window a second time on information it had. So when eviction
  fires, the outgoing turns are condensed by one NPU call into a short note
  folded into the SYSTEM turn (the one thing eviction never touches). A later
  eviction re-summarises the previous note together with the newly evicted
  turns, so notes never stack.

  Measured, same 30-turn conversation with a fact stated at the start:

  | | wall | answer |
  |---|---|---|
  | `GENIE_SUMMARIZE_EVICTED=1` (default) | 12.6s | recalled the key |
  | `=0` | 8.5s | lost it |

  The extra ~4s is paid only when eviction was going to happen anyway. If the
  summarisation call fails, or the note itself will not fit, the server falls
  back to plain eviction -- a summary is never allowed to break a request.
  `GENIE_SUMMARY_MAX_TOKENS` (default 192) bounds the note.

- **A buffered tool stream is still abortable.** Tool responses are buffered
  (a half-emitted `<tool_call>` is worse than a slower one), which means
  nothing is written while the model generates -- so the usual
  disconnect-detection-by-failed-write never fires. Both stream paths emit a
  lightweight probe every 8 chunks (an SSE comment on the OpenAI side, a real
  `ping` event on the Anthropic side) purely so a departed client is noticed.
  Without it an abandoned tool turn runs to `max_tokens` holding the
  single-flight NPU against every other caller.

- **Streaming reports usage too.** OpenAI streams emit a final chunk with an
  empty `choices` list carrying `usage`, but only when the caller sets
  `stream_options.include_usage` -- clients that do not ask see a
  byte-identical stream to before. Anthropic streams carry `output_tokens` in
  `message_delta` as usual. Both include the summarisation overhead below when
  there was any.

- **Summarisation cost is reported, not hidden.** A request that evicts spends
  extra NPU time condensing the outgoing turns. That shows up as
  `usage.genie_context_overhead_tokens` on non-streaming responses (present
  only when non-zero, so an ordinary response is unchanged) and in the server
  log line. The summarisation call deliberately does NOT claim the resident KV
  -- it leaves text in the dialog that is not the caller's conversation, so it
  records "unknown" and the next turn re-prefills rather than resuming from a
  false prefix.

- **KV reuse across turns.** The dialog keeps its KV between queries, so when a
  request's prompt is a byte-exact extension of what the dialog already holds,
  only the new suffix is prefilled. Measured: 1.23s cold, then 0.66s / 0.68s on
  the two following turns of the same conversation, versus 5.48s for an
  unrelated one. The match must be exact -- edited history, an evicted turn, or
  an aborted generation all fall back to a full re-prefill, because resuming on
  mismatched KV would answer from a history that never happened.
  (`GenieDialog_save`/`restore` also exist and work -- measured ~75 KB/token on
  disk, ~128 MB at 1711 tokens -- but they are not used: in-memory continuation
  is free and this server serves one conversation at a time.)

- **The sampler ships with NO repetition penalty, and that is what a
  degenerate loop looks like.** Genie's `token-penalty` block is optional and
  every field in it defaults to 0 (`penalize-last-n`, `repetition-penalty`,
  `presence-penalty`, `frequency-penalty` -- read off
  `examples/Genie/Genie/src/qualla/include/qualla/detail/sampler-utils.hpp`),
  so a bundle without it samples at temp 0.8 with nothing suppressing a repeat.
  Every bundle here arrived that way: the sampler is byte-identical to
  Qualcomm's own reference for this stack **minus** that block. The symptom is
  not an error -- the model answers normally and then emits the same paragraph
  until it hits `max_tokens`, which reads as the model being broken rather than
  as one missing config key.

  Values matter more than presence. Measured on this box, same prompt, 400
  tokens, one restart between each:

  | `repetition-penalty` | repeated sentences | identifiers |
  |---|---|---|
  | none (as shipped) | 0 on a 400-token answer -- **the loop was NOT reproduced at this length** | intact |
  | 2.3 (Qualcomm reference) | 0 | **corrupted** -- one answer spelled two proper nouns as `MCPWeekly` / `MPC Week` / `MP Weekly` and `YaLLABS` / `YaLLLab` |
  | **1.15** (recommended) | 0 | intact -- `src/genie_server.py`, `build_windowed`, `MCP Weekly`, `YawLabs` all byte-exact |

  So `"token-penalty": {"version": 1, "penalize-last-n": 128,
  "repetition-penalty": 1.15, "presence-penalty": 0.0, "frequency-penalty":
  0.3}`. The vendor reference is a starting point, not the answer -- same
  precedent as `poll`. **For an agent workload the 2.3 failure is worse than
  the loop it fixes**: file paths and identifiers are precisely what has to
  survive verbatim, and this server's own eviction summariser is prompted to
  "keep file paths, identifiers, decisions made".

  Two limits worth stating plainly. The block IS honoured at load -- proven by
  A/B, since the same prompt gave different output with and without it, which
  is the check that would have caught it being ignored the way a post-create
  sampler apply is. But **the deep-context loop was never reproduced here**:
  the report that prompted this was a 6.7k-token prompt, and these probes ran
  at ~40 tokens of context. The penalty is the mechanism that suppresses such a
  loop and it is now active; that it fixes THAT case is inference, not
  measurement.

  **That inference turned out to be WRONG for the repetition users actually
  report, and this is the correction (2026-08-27).** The complaint was "it
  repeats on every response" against a bundle that already carried the 1.15
  block above -- so the penalty was live and the repetition continued. Raising
  it made things worse in both directions at once: at
  `1.3 / 0.5 / 0.5` with a 256 window, repeated word-trigrams fell only 12.1% ->
  11.4% while `QnnHtpV73Stub.dll` came back mangled. The knob is exhausted well
  before it fixes this, which is the strongest evidence that this was never what
  the knob was for.

  The real mechanism is in the next note (**"the answer twice"**). The penalty
  section stands as written -- a bundle with no `token-penalty` really does
  sample with nothing suppressing a loop, and 1.15 really is the right value --
  but **do not reach for it when the symptom is a duplicated answer.** Two
  different failures were being treated as one, and the visible one was never
  the sampler's.

- **The answer arrives TWICE, and it is the prefill rather than the sampler.**
  With reasoning suppressed the prompt ends in a CLOSED, empty `<think></think>`
  so the model should answer directly. Qwen3 does not always accept that: it
  writes its reasoning anyway, emits a bare closing tag, and then answers
  properly. The response then carries the answer twice with a stray `</think>`
  between them. **Measured at 1 request in 6** on the shipped 8192 multi bundle,
  and again at 1 in 12 on a second sample -- frequent enough to read as "every
  response" to anyone not counting.

  `GENIE_STRIP_THINK` could never catch it: that pattern needs a MATCHED
  `<think>...</think>` pair, and the opening tag is in the PROMPT rather than in
  the output, so the orphan close never matched and the duplicate shipped.
  `_strip_orphan_think` now drops a leading run ending in a close that was never
  opened, and leaves well-formed pairs to `GENIE_STRIP_THINK`, which is the
  caller's choice rather than the stripper's. Verified: 0 in 6 after, against 1
  in 6 before.

  **Streaming needed a different answer, and it costs something.** SSE cannot
  retract a frame, so a duplicate can only be kept out of a stream by not
  sending it yet -- which means holding the START of every response until the
  orphan is ruled in or out. A bounded hold was tried and LEAKED: set to 1024
  characters, already wide margin over an orphan observed at offset 405, the
  next sample closed at 1080 and went out anyway. Two offsets that far apart are
  not a threshold you can fit, so `GENIE_ORPHAN_HOLD_CHARS` defaults to `-1` and
  holds until the question is answered -- the close tag, or the end of the
  generation.

  The honest statement of that default: **a streamed reply arrives as one frame
  at the end.** Correct, but not incremental. For the agent traffic this server
  exists to serve that is invisible, since the client consumes the finished
  message either way; for a human watching tokens appear it is not, and that
  reader should set `GENIE_ORPHAN_HOLD_CHARS=0` and accept the occasional
  doubled answer. The non-streaming and tool paths already buffer the whole
  generation, so they strip it always and for free regardless of this setting.

- **Every fresh prompt replayed the same answer, because the seed is reset per
  request.** Genie re-seeds its RNG from the config's `seed` on every
  `GenieDialog_reset` (qualla `Sampler::reset` -- "just need to reinit rng"),
  and the server resets whenever a prompt does not continue the resident KV. The
  AI Hub bundles ship `"seed": 42`, so at temp 0.8 the model was nominally
  sampling while the dice were reset before every roll: identical prompt in,
  byte-identical answer out, measured three times running and again across
  separate server processes.

  That is why re-asking never escaped a bad answer, and no repetition penalty
  could have fixed it -- the problem was not which tokens were penalised, it was
  that the same draw was taken every time. The server now writes a fresh seed
  into the config TEXT it hands `GenieDialogConfig_createFromJson`, leaving the
  bundle on disk untouched. `GENIE_SEED` pins it when reproducibility is what
  you want.

  **Limit, because it is a real one:** the seed varies per PROCESS, not per
  request. Inside one server run an identical prompt still replays its identical
  answer. A per-request seed was tried first and is inert on QAIRT 2.45 -- three
  identical generations with a fresh seed on each -- which is the same wall the
  sampling note below describes. And `"seed": -1` does not work either, though
  it looks like it should: the constructor reads -1 as "seed from the clock",
  but `reset()` re-seeds with `_seed` unconditionally, so -1 casts to a fixed
  uint32 and every generation after the first is deterministic again.

  The server checks this at startup and warns when the block is absent, when
  `penalize-last-n` is 0 (the penalties beside it are then applied to an empty
  window and do nothing), or when every penalty in it is 0. It says nothing
  when there is no bundle config at all -- a note about a file you do not have
  is noise in front of the error naming the env vars to set.

  **This is a PER-MACHINE fix, and nothing in this repo can apply it for you.**
  Bundles are large external artifacts deliberately kept out of version
  control, so a fresh clone on another box gets the recommendation above and an
  unpatched bundle. The startup warning is the part that ships: it fires on
  every launch until the block is added, which is the whole reason it exists
  rather than a line in these docs that someone has to remember to read. Keep a
  `genie_config.json.orig` beside the edited one so a measurement can be
  reproduced against the shipped sampler.

- **Sampling is server-level, not per-request.** `temperature` / `top_p` /
  `top_k` are accepted and **not honoured**. Measured directly against QAIRT
  2.45: `GenieDialog_getSampler` returns a valid handle,
  `GenieSamplerConfig_createFromJson({"sampler": {...}})` returns 0, and
  `GenieSampler_applyConfig` returns 0 -- yet generation is byte-identical
  across seeds 1 / 999 / 12345 and temperatures 0.0 / 1.5 / 2.0. The dialog
  binds its sampler at `GenieDialog_create` time. To change sampling, edit
  `dialog.sampler` in `genie_config.json` before the server loads it. The
  server prints this limitation at startup rather than letting it be silent.

  Re-confirmed 2026-08-27 from the other direction: a per-request seed applied
  through the same call sequence produced three byte-identical generations. So
  the finding is not "seeds specifically are ignored" -- the whole post-create
  apply is inert, and `_sampler_params`' temp-0-for-tool-turns is mapped but
  never actually reaches the sampler either.

  **The one thing the server now DOES change is the seed, and it does it at
  create time rather than per request** -- by rewriting `dialog.sampler.seed` in
  the config TEXT passed to `GenieDialogConfig_createFromJson`, never on disk.
  That is the only window in which sampling is settable at all, which is exactly
  why it happens there. See the seed note above for what it fixes and what it
  cannot.

  Two JSON shapes worth knowing, both found by probing: the sampler config
  must be wrapped as `{"sampler": {...}}` (a bare object returns -8 "Missing
  field"), and stop sequences must be `{"stop-sequence": [...]}` (a bare array
  returns -8 "Top level config is not an object" and is silently ignored).

- **It will not start on a port something else is already serving.** Checked
  before the model loads, so a collision costs 0.3s rather than the 11-35s of
  loading followed by a failure. The check exists because on Windows the bind
  does NOT fail: `HTTPServer` sets `allow_reuse_address`, which on POSIX means
  "rebind a TIME_WAIT socket" but on Windows lets a second process bind a port
  another process is actively serving. Both binds succeed and the OLD process
  keeps answering -- so the new server logs a clean startup, reports the right
  HTP allocation for its bundle, and serves nobody, while requests are answered
  by whatever was already there. That happened during the window benchmarking
  and was caught only because `/props` disagreed with the bundle just loaded.
  A server that silently answers from the wrong model is the same failure mode
  as a silent CPU fallback, so it now refuses instead.

- **Single-flight.** The NPU serves one query at a time (concurrent HTP access
  wedges the device), so requests are serialized by a lock. Fine for one agent.
- **Tool calling works, and thinking dominates its latency.** Enabled when the
  bundle's tokenizer carries `<tool_call>` (probed at startup; a bundle without
  it still gets an honest 400). Measured on this box, same prompt and same
  correct call:

  | | wall | completion tokens |
  |---|---|---|
  | thinking on (`GENIE_THINKING=1`) | 10.7 - 41 s | 113 - 300 |
  | thinking off (**default**) | 1.8 - 2.4 s | 17 - 25 |

  Nearly all of the reasoning-path cost is the `<think>` block, and its length
  varies a lot run to run -- so that path is not just slow but unpredictable,
  which is the property a human waiting on an agent step actually notices.

  **This is why the default is off, and it is a deliberate change.** The server
  used to default it ON, on the reasoning that faithfulness to the model is the
  honest default and agent clients could opt out. That made the configuration
  these docs recommend the one nobody got without asking, on a server whose
  reason to exist is being driven by an agent. Suppression is a prompt PREFILL
  (a closed, empty think block the model resumes after), not a filter over the
  output -- so nothing is hidden, and a caller that asks for reasoning gets
  exactly what the model produced. Set `GENIE_THINKING=1`, or send
  `reasoning_effort: "high"` on the request, to get it back.

  One consequence worth knowing: with reasoning suppressed Qwen3 improvises its
  tool-call wrapper more often, so the `<function_call>` and bare-JSON shapes in
  the next note are now the common case rather than the exception. The parser
  accepts all of them.

- **Tool-call wrappers vary.** `<tool_call>` is the trained, in-vocab tag, but
  with thinking suppressed the model also emits `<function_call>` and
  occasionally bare JSON with no wrapper. The parser accepts all of these; a
  block whose payload does not parse is left VISIBLE in the content rather than
  silently dropped, so a malformed call is debuggable instead of invisible.

- **Set `poll: false` in the bundle's `genie_config.json`. It is the single
  biggest free win here.** The QnnHtp backend block ships `"poll": true`, which
  busy-waits. Measured on this box: a resident server with `poll: true` burns
  **270% CPU (2.7 cores) while completely idle**, no requests in flight -- and
  the spinning threads compete with the work, so it is slower as well as
  wasteful:

  | compiled n_ctx | decode, `poll: true` | decode, `poll: false` | idle CPU |
  |---|---|---|---|
  | 4096 | 11.6 t/s | **18.0 t/s** (+55%) | 267% -> 0% |
  | 8192 | 7.9 t/s | **8.8 t/s** (+11%) | 267% -> 0% |
  | 16384 | 3.2 t/s | **3.3 t/s** (+2%) | 267% -> 0% |

  The idle figure is controlled, because the first attempt at it was not. It
  was originally sampled just after a benchmark, which cannot distinguish an
  idle spin from a generation still draining -- a fair objection raised against
  it. Re-run on a server that has answered nothing but `/health`, same bundle
  and same 29 threads, 30-second samples: **267.1%** with `poll: true`, **0.0%**
  with `poll: false`, and **0.1%** with `poll: false` thirty seconds after a
  generation. It is a spin, not drain.

  The decode figure is now measured four times, three of which agree. Interleaved
  legs, single engine, no contention, AC, cool box:

  | | poll=false | poll=true | ratio |
  |---|---|---|---|
  | single server (`bench_endpoint`) | 18.0 | 11.6 | 1.55x |
  | interleaved, on battery | 17.91 | 12.36 | 1.45x |
  | **interleaved, solo, AC** | **17.11** (16.92-17.23) | **11.28** (9.69-12.99) | **1.51x** |
  | via the contention harness | 17.58 | 20.72 | 0.85x -- INVERTED |

  **The inverted one is an artifact of the contention harness and should not be
  read as a real result.** In that harness a GPU server is resident throughout,
  and `poll: true` heats the package enough to drag the whole box down -- its
  legs ran at a clock-under-load median of 56-64% of base against `poll:
  false`'s 73-79%. The two arms were therefore not measured under comparable
  conditions, which a ratio survives and an absolute does not. Removing the
  contention reproduces the original result.

  Note also the spread: `poll: false` holds a 0.31 t/s range across three
  interleaved pairs while `poll: true` spans 3.30. **The busy-wait costs
  predictability as well as throughput**, which matters more than the median for
  an agent workload where a slow turn is a stall a human notices.

- **The contention ABSOLUTES could not be established on this box, and that is
  a property of the hardware rather than a gap in effort.** The poll ratios are
  solid (see `TYPED_ROUTER_BRIEF.md`), but every attempt at absolute
  tokens/sec under two hot engines failed the same way: the package sags to as
  low as **34.8% of base clock** under load, on AC at high charge, and it sags
  further with `poll: true` because the spin adds heat. Six legs produced a
  20x wider spread in one arm than the other. If absolutes are ever needed,
  they want sequential legs with cooling between them, not two engines hot at
  once -- and the per-leg clock has to be sampled DURING the measurement, since
  a pre-flight check certifies nothing about what follows it.

  Prefill improves too (4096: 629 -> 1157 t/s median) and run-to-run noise
  drops sharply (1.85 -> 0.23 t/s at 4096). Nothing measured got worse. The
  penalty shrinks as the window grows because the NPU work per token grows
  while the CPU spin stays constant, so a small-window bundle -- the fast,
  latency-sensitive case -- is hurt most.

  Two consequences beyond throughput. Idle CPU is not free on a laptop, and
  more importantly **an idle NPU server was stealing 2.7 cores from anything
  else on the box**, which contaminates any concurrent benchmark of another
  engine. Every number in this file predating this finding was measured with
  `poll: true` and is therefore pessimistic.

  That second consequence has since been measured, and it is larger than the
  single-engine gain above. `MULTI_ENGINE.md` reports NPU+GPU concurrency at
  **0.78x -- a net loss -- with `poll: true`, against 1.45x with `poll: false`**,
  because the OpenCL backend needs host cores per token to dispatch its kernels
  and the busy-wait was taking them. The same run had earlier retired memory
  bandwidth as the contention mechanism on the strength of the `poll: true`
  numbers; with the spin removed, bandwidth predicts the result again. So this
  one config line decides whether running two engines together is worth doing
  at all.

- **The compiled context window is a per-token tax, paid whether or not you
  use it.** A Genie bundle's KV tensors are graph INPUTS statically shaped to
  the compiled window -- `past_key_0_in: [8, 1, 128, n_ctx-1]`, `uint8` -- so
  every decode step feeds the whole buffer through the HTP no matter how few
  positions are filled. Cost therefore tracks the window the bundle was BUILT
  at, not the context actually in play.

  Three windows, same model, `poll: false`, quiet box, one harness
  (`python src/bench_endpoint.py`). The 8192 and 16384 bundles came from
  `qai-hub-models export` and are the same model as the 4096 prebuilt:
  `precision`, `tool_versions`, `chipset_attributes`, `config.json`, the chat
  template and every tokenizer file are byte-identical, and the only
  `metadata.json` difference is the KV shapes.

  | compiled n_ctx | HTP alloc | prefill (median) | decode (median) | cost per doubling |
  |---|---|---|---|---|
  | 4096 | 328 MB | **1157 t/s** | **18.0 t/s** | -- |
  | 8192 | 647 MB | **458 t/s** | **8.8 t/s** | 2.05x decode, 2.54x prefill |
  | 16384 | 1195 MB | **176 t/s** | **3.3 t/s** | 2.69x decode, 2.59x prefill |

  So **decode is roughly inverse-linear in the window up to 8192 and worse
  beyond it**: the first doubling costs 2.05x (almost exactly the 2x a fixed
  per-token tax predicts), the second 2.69x. Prefill is consistently worse than
  inverse-linear, ~2.55x per doubling. HTP allocation is exactly linear at
  73,983 bytes per token of window, which doubles as a check that a bundle is
  the window it claims -- the 8192 bundle allocated 646,971,904 bytes against a
  646,971,904 prediction.

  Decode is FLAT with depth on both self-exported bundles, which is the tell:
  the 16384 bundle decoded 3.26 t/s holding 469 tokens and 3.27 t/s holding
  10532, and the 8192 bundle 8.77 versus 8.81. Cost is set by the compiled
  window, not by how much of it is live. A 10532-token prefill on the 16k
  bundle takes **60 seconds** of wall time.

  **The 4096 prebuilt is the exception, and the reason is still open.** It is
  NOT flat: measured with shallow and deep runs INTERLEAVED (250, 3300, 250,
  3300, 250, 3300) so that a drift over the run could not be mistaken for a
  depth effect, it gives **18.9 / 18.5 / 18.7 t/s at 250 tokens against
  12.8 / 13.0 / 13.3 at 3300** -- a real ~30% decline that tracks depth rather
  than elapsed time. Both self-exported bundles are flat over far wider spans.

  So a prebuilt appears to pay only for the context actually in use, while a
  single-length export pays for its whole compiled window on every token. That
  is a meaningful artifact-quality difference and worth chasing, because it is
  the difference between 18.9 and 8.8 t/s on a short prompt.

  **ANSWERED: export with SEVERAL `--context-lengths`. It is the difference.**
  The prebuilt advertises `genie.context_lengths = [512, 1024, 2048, 3072,
  4096]` where a single-length export advertises one value. Building an 8192
  bundle with `--context-lengths 512,1024,2048,4096,8192` and measuring it
  against the single-length 8192 bundle -- same model, same tooling, same
  window, same `poll: false`, interleaved depths over three passes:

  | | single-length 8192 | multi-length 8192 | ratio |
  |---|---|---|---|
  | prefill @469 | 463 t/s | **1382 t/s** | 2.98x |
  | prefill @2657 | 461 | **997** | 2.17x |
  | prefill @6157 | 456 | **636** | 1.40x |
  | decode @250 | 8.8 t/s | **18.2 t/s** | 2.07x |
  | decode @3300 | 8.8 | **11.5** | 1.31x |
  | decode @6000 | 8.8 | 8.1 | 0.92x |

  The single-length bundle is flat -- it pays for its whole compiled window on
  every token. The multi-length bundle pays for the context actually in use,
  matching the 4096 prebuilt at shallow depth (18.2 against 18.7) while holding
  twice its window. **It costs +3.8% bundle size (116 MB) and ZERO extra HTP
  memory** -- both 8192 bundles allocate exactly 646,971,904 bytes.

  The size delta says where it goes: part1, the embedding lookup with no
  attention and no KV, grows 0.2 MB; parts 2-4, the transformer layers, grow
  36-44 MB each. Extra compiled graphs, only where context-dependent attention
  lives, sharing one copy of the weights.

  **MECHANISM, read off the artifact rather than inferred from timings.**
  `qnn-context-binary-utility --context_binary <part>.bin` dumps the graphs
  compiled into a context binary, and it is exactly smallest-that-fits graph
  selection:

  | bundle | graphs in part2 | compiled context lengths |
  |---|---|---|
  | single-length 8192 | 2 | `[8192]` |
  | multi-length 8192 | 10 | `[512, 1024, 2048, 4096, 8192]` |
  | 4096 prebuilt | 10 | `[512, 1024, 2048, 3072, 4096]` |

  Named `prompt_ar128_cl<N>` and `token_ar1_cl<N>` -- one prefill and one
  decode graph per compiled length. A single-length bundle has one pair and so
  runs every token against its full window; a multi-length bundle has five and
  runs against the smallest that fits. That is the whole effect, and it is why
  `metadata.json` cannot show it: both bundles declare the SAME 28 inputs, 25
  outputs and 8191 KV shape, identical byte for byte apart from
  `genie.context_lengths`. The extra graphs are inside the binary.

  **I refuted this mechanism earlier and the refutation was wrong.** A targeted
  sweep either side of the 512 boundary showed a smooth slide with no step, and
  I read that as killing graph selection. The sweep was mis-targeted: the graph
  must hold prompt AND generated tokens, so with `--tokens 60` the points at
  requested depths 440 / 470 / 490 / 510 / 540 landed at 496 / 525 / 545 / 565
  / 595 total -- four of the five inside ONE graph (cl1024). It measured within
  a plateau and found it flat, which is what a plateau is. The single crossing
  it did contain, 440 to 470, showed -2.2% against an expected -7.9%, inside a
  run whose noise was 0.53 t/s.

  The general lesson is worth more than the finding: **a boundary test has to
  account for everything that moves the boundary.** I placed the depths against
  the prompt length and forgot the generation, so the experiment I ran was not
  the experiment I designed -- and it returned a clean, confident, wrong
  answer.

  Practical upshot, revised: **always export with several `--context-lengths`,
  and then 8192 is the sweet spot.** The window tax as measured above is a
  property of SINGLE-LENGTH exports, not of Genie or the HTP -- a multi-length
  bundle of the same window is 2-3x faster on short prompts for 3.8% more disk
  and no extra HTP memory. Given that, 8192 buys 2x the context of 4096 at
  near-parity on shallow prompts; 16384 still buys 4x the context for a large
  decode penalty. Prefer the smallest window the workload needs, prefer a
  multi-length build at that window, and prefer eviction + summarisation over a
  bigger bundle when the history compresses -- which is what this server is
  built around.

  So a bigger bundle is a **capability tier, not an upgrade**: it buys window
  that 4096 cannot hold at all, and charges for it on every request including
  the short ones. Pick the smallest window that fits the workload, and prefer
  eviction + summarisation over a bigger bundle when the history is
  compressible -- which is the case this server is built around.

- **`dialog.context.size` cannot buy that tax back.** It is a software-side
  limit, not a graph selector. Setting it to 1024 against the 4096-compiled
  bundle left the HTP allocation byte-identical (343,933,440 both ways) and
  decode unchanged at ~11.0 t/s. The window is fixed at export time
  (`--context-lengths`); lowering the config only lowers the ceiling the
  evictor works against. Note that `genie.context_lengths` in `metadata.json`
  is NOT merely a record of what the model could be exported at -- it names the
  graphs actually compiled into the `.bin`, confirmed with
  `qnn-context-binary-utility` in the mechanism note above. (This bullet said
  the opposite until the graphs were read off the artifact; that reading is what
  the mechanism note supersedes.) What the config cannot do is pick among them:
  selection is per request, smallest-that-fits, and the KV tensor is shaped to
  the largest either way -- `4095` on the 4k bundle.

- **KV is `uint8`, ~72 KB/token** -- read off the bundle's own `metadata.json`
  (`past_key_0_in` / `past_value_0_in` are `dtype: uint8` with a fixed quant
  scale), not assumed. For this model that is 36 layers x 2 x 8 KV heads x 128
  head_dim x 1 byte = 73,728 B/token; the measured HTP allocation delta between
  the two bundles works out to 73,983 B/token, 0.3% off. Any fp16 estimate of
  KV size for this bundle is 2x too high.

- **Throughput is bandwidth-bound.** Decode is ~18 t/s on a quiet box for the
  4096 bundle with `poll: false` (~12 t/s as the bundle ships). The figure
  belongs to the bundle's window and its poll setting, not to the server -- see
  the tables above. It drops
  sharply under memory pressure (the X Elite's 32 GB LPDDR5x is shared by CPU/GPU/NPU),
  so a large resident model elsewhere (e.g. a 26 GB llama-server) will slow it.
- **`finish_reason`** reports `length` only on context-limit; a `max_tokens` cap
  currently reports `stop` (Genie signals a normal sentence-end at the cap).
- Model swaps: point `GENIE_BUNDLE_DIR` at the 1.7B bundle for lower latency, or
  any other X-Elite Genie bundle. Note that the bundle's **compiled window** is
  as big a latency lever as its parameter count -- see the window-tax note
  above before assuming a larger-context bundle is strictly better.
