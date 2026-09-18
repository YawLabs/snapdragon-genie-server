# Genie NPU Server (OpenAI- and Anthropic-compatible)

A local HTTP endpoint speaking the OpenAI Chat Completions API and the
Anthropic Messages API, backed by a Qualcomm Genie context-binary
bundle running on the Snapdragon X Elite NPU (Hexagon v73). The model is loaded
**once** (resident on the HTP via the Genie C API) so requests don't pay the
bundle load that `genie-t2t-run.exe` repeats on every invocation: **11-35s**,
measured on the 8192 multi-length bundle (10.8-15.0s warm, 34.4s after heavy
disk traffic -- the table under [Run](#run)). This line said "~35-50s" and the
server's own header "~8.5s" for the same operation; neither was a measurement
of this bundle, and the table is the one figure to quote.

## Requirements

- **Native ARM64 Python** (aarch64). Genie.dll and its Qnn* deps are
  `aarch64-windows-msvc`; an x64/emulated Python cannot load them. The
  launcher checks this before anything else and exits 1 naming the interpreter
  it tried; `GENIE_PYTHON` points it at another one.
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
  That exit prints the status by its header name (`status=-3
  (ERROR_MEM_ALLOC)`) and names BOTH suspects, because the status does not
  say which it was: a bundle built for another arch or QAIRT, and an HTP held
  by another process or degraded -- with what to check for the second (another
  `genie_server`, `genie-t2t-run.exe` or QNN / ONNX session; `Get-Process
  python, genie*`; a reboot if nothing holds it). `ERROR_MEM_ALLOC`, the one
  status the header ties to memory, lists the held / out-of-memory HTP first;
  it reorders the list and never shortens it. It used to name the arch
  mismatch alone, whatever the status, which on a shared or degraded box sent
  the operator to rebuild a bundle that was fine. Every other exit and error
  that prints a Genie status names it the same way (`status=-6
  (ERROR_QUERY_FAILED)`).

- No pip packages. Pure Python stdlib.

## Run

The Genie bundle and the QAIRT 2.45 runtime are large external artifacts and are
**not** in this repo. The normal way to run is the launcher, which finds them
itself -- it looks for a `genie-npu` directory beside this repo holding
`bundles/` and `qairt/`, and picks the newest QAIRT under it by VERSION NUMBER
(a directory not named like `2.45.0.260326` is ignored, so a stray `latest` or
`backup` beside the SDKs is never handed to the server as one):

```powershell
cd <your clone of this repo>   # the path below is relative to the repo root
powershell -File src\run-genie-server.ps1
```

That serves on `127.0.0.1:8123`, and supervises: see Supervision below.
`-Model qwen3-8b` or `-Model qwen3-8b-8192` picks another bundle (see Model
swaps under the notes).

On a stock Windows client the execution policy blocks every `.ps1`, and the
launcher exits 1 with `... cannot be loaded because running scripts is
disabled on this system`. Loosen it no further than you need to: either
`powershell -ExecutionPolicy Bypass -File src\run-genie-server.ps1`, which
applies to that one process and changes nothing on the machine, or
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, which covers your
account only (and the venv's `Activate.ps1` with it). There is no need to
change the machine-wide policy.

`-Help` (or `-h`, or `--help` under `powershell -File`) prints the models,
every environment variable the launcher reads with its default, and where the
docs are, then exits 0 without touching the environment, the disk, the NPU or
the network. `-?` shows the script's comment-based help (`Get-Help
src\run-genie-server.ps1 -Full` has more); under `powershell -File` it prints
to a console only and is silent when stdout is redirected, still starting
nothing. The launcher is `[CmdletBinding()]`, so a mistyped parameter --
`-Modle qwen3-8b`, `--hlep` -- is a PowerShell binding error with exit 1. It
used to be ignored without a word, and the default 4B bundle was loaded.

If the artifacts live elsewhere, point `GENIE_NPU_ROOT` at the directory
holding them, or set the two paths directly. The launcher checks both exist
before the 11-35s model load and exits naming what it tried, rather than failing
deep inside the server -- and when `qairt\` exists but holds no version
directory (an SDK zip dropped there and never unpacked is the usual shape) it
names the directory it searched:

```powershell
$env:GENIE_NPU_ROOT = "D:\genie-npu"
# or individually:
$env:GENIE_BUNDLE_DIR = "...\qwen3_4b-genie-w4a16-x-elite-ctx8192-multi"
$env:GENIE_SDK_DIR    = "...\qairt\2.45.0.260326"
```

It also checks that the bundle directory holds a `genie_config.json`, before
it even looks for python. A directory one level above the bundle -- which is
what `qai-hub-models fetch ... --extract -o <dir>` leaves, since the bundle
goes in a model-named folder inside `<dir>` -- is refused (exit 1) naming up
to five subdirectories one level down that do hold one, with advice that fits
how the path was chosen (under `-Model`, which rewrites `GENIE_BUNDLE_DIR`
every run, that is to move the files up a level or drop `-Model`). That used
to be a bare `FileNotFoundError` traceback from the server after `Genie.dll`
had loaded.

Running `python src\genie_server.py` directly works too, but then nothing
supervises it, the port defaults to 8080 rather than 8123, and
`GENIE_BUNDLE_DIR` and `GENIE_SDK_DIR` have NO default -- the server exits at
startup naming whichever is unset (and the other's value, when that one is
set). (A relative path in either is made absolute first, so one works from
wherever you launched.)

The server itself takes no arguments -- everything is a `GENIE_*` variable --
and it reads its command line FIRST, before any environment check, bind or
load. `-h`, `--help`, `-help`, `-?` and `/?` print a usage (what it is, the
two required variables, the port it would serve on, the launcher, the
endpoints) and exit 0; the full variable table stays here, under
[Environment](#environment), rather than in a second copy. Any other argument
is refused by name on stderr with exit 2 -- `genie_server.py: unknown
arguments '--port', '8081'. This server takes no arguments -- ...` -- where it
used to be ignored, so `--port 8081` served 8080 and `--help` on a configured
box went straight into the model load. The launcher's own flags (`-Model`,
`-Help`) are the launcher's and never reach the server.

The server checks the setup itself too, so a direct run is covered. All of
these run before the 11-35s model load begins -- the first two before
`Genie.dll` is even loaded -- and each is an exit that names the fix, where
all three used to be a traceback or a misleading status:

- **`GENIE_BUNDLE_DIR` holds no `genie_config.json`** -- `no genie_config.json
  in <dir>`, saying it must be the bundle directory itself and listing the
  subfolders one level down that hold one.
- **A file `genie_config.json` names is not in the bundle** -- the context
  binaries (`ctx-bins`), the tokenizer `path` and the backend `extensions`
  file, each looked up the way Genie resolves it. An incomplete copy or an
  interrupted download of a multi-GB bundle used to go all the way to
  `GenieDialog_create` (`GenieDialogConfig_createFromJson` accepts a config
  whose files are gone), whose exit then blamed an arch mismatch.
- **`Genie.dll` will not load**, told apart by cause: a Python that is not
  ARM64 (`This Python is win-amd64 ...` -- run an ARM64 `python.exe`, or point
  `GENIE_PYTHON` at one under the launcher); an ARM64 Python and a DLL that is
  not an ARM64 image (the wrong QAIRT under `GENIE_SDK_DIR`); no `Genie.dll`
  at all (an incomplete SDK extract, or a path one level off); and a
  `Genie.dll` that is there but will not load because one of ITS dependencies
  will not (the `Qnn*.dll` files beside it, or the Microsoft Visual C++
  runtime for ARM64).

The launcher leaves your shell as it found it: every `GENIE_*` variable it
writes for the server (`GENIE_NPU_ROOT`, `GENIE_BUNDLE_DIR`, `GENIE_MODEL_ID`,
`GENIE_SDK_DIR`, `GENIE_HOST`, `GENIE_PORT`) is put back, or removed again, when
it exits. It used to fill the last four if unset and leave them behind, which
pinned an interactive shell to the first QAIRT it discovered. It also persists
no log: the server's output is the console's, and only the llama launcher
rotates log files.

Startup order is cheap-things-first, so a mistake costs a second rather than a
model load. Before the 11-35s load: the command line (above), the
port-collision check (see the note below), the bundle-config warnings
(`poll`, single-length, sampler penalty), then the **bind itself** -- an
address this machine does not have, or a `GENIE_PORT` outside 0-65535, exits
with `cannot bind HOST:PORT: ...` -- and last the setup checks listed above.
Between the bind and those checks it prints where it WILL answer: `[genie]
port HOST:PORT is reserved; it refuses connections until the model is
resident` (the port actually bound, so `GENIE_PORT=0` shows the one it got). A direct
run used to print no address until the model was resident, so for the whole
load every client tool here said "nothing listening" and there was nothing on
screen to match that against.

The load itself is one blocking native call, `GenieDialog_create`, and nothing
is printed while it runs -- so a load that never returned looked exactly like
one still working. If it passes 60s (the slowest load logged here is 36.8s),
and every 60s after that, the server prints `[genie] still loading after Ns;
a normal load takes ~11-15s, up to ~35s cold. If it never finishes, suspect
the HTP: held by another process or degraded. ...` -- naming what to check
(another `genie_server` or `genie-t2t-run.exe`; `Get-Process python, genie*`),
and that **Ctrl-C is not acted on until the load returns**, so stopping it
means ending the process. No load logged here has hung; the line says what to
suspect, not what happened.

The socket is BOUND before the load and only LISTENS after it, once the model
is resident. A connect made during the load is therefore refused exactly as
one to a closed port is -- on Windows after the stack's ~2s of SYN retries, or
as a timeout for a client that waits less than that -- so "the port answers"
means "the model is resident", and `/health` is reachable only from then on.
That is what anything waiting on the port relies on (`bench_servers` takes the
first successful connect as a server that started): a load that fails -- a bad
bundle, `GenieDialog_create` refusing, the HTP held by another session -- exits
BEFORE the port has ever answered, not after. A second instance started during
the load passes the port-collision check (nothing is accepting yet) and exits
at its OWN bind, with `cannot bind ...` (WinError 10048, or 10013 when the
instance binding second is the wildcard one), ahead of its load, plus a line
naming the collision: `Something already holds that port -- most likely
another instance of this server still loading its bundle, on ANY GENIE_HOST:
the bind is exclusive, so 0.0.0.0 and 127.0.0.1 cannot share a port.` That
guard holds whatever `GENIE_HOST` each instance uses, because
`Server.server_bind` asks for the port with `SO_EXCLUSIVEADDRUSE`. Leaving
`SO_REUSEADDR` off was not enough: two sockets with DIFFERENT addresses on one
port -- `0.0.0.0` and `127.0.0.1` -- bound happily in either order, so an
instance started on the default host during a wildcard instance's load passed
the port check AND the bind, and loaded a second bundle onto the HTP beside
the first. The listen step has an exit of its own, `cannot listen on
HOST:PORT: ...`; in practice that one is POSIX-only, where `SO_REUSEADDR` lets
two sockets bind a port while neither is listening.

Startup then prints `model resident on HTP in <N>s`, the endpoint URL and the
bundle's `n_ctx`, compiled lengths and `poll`, and after that every request
reuses the resident model.

Re-measured 2026-08-26 on the 8192 multi-length bundle (the launcher default),
because the figures here were an older bundle's:

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

Routing looks at the path alone: a query string and a trailing slash are
ignored on GET and POST alike, so `/health?probe=1` (a cache-buster, which is
an ordinary thing for a health checker to send) and `/v1/messages?beta=true`
route like their bare forms, and an error comes back in the envelope of the
API the path names.

- `POST /v1/chat/completions` -- OpenAI chat API. Supports `messages`, `stream`
  (SSE), `max_tokens` / `max_completion_tokens`, `stop` (also Anthropic
  `stop_sequences`), and `tools`. ChatML template is taken from the bundle's own
  `metadata.json` chat_template. An UNUSABLE one does not abort startup any more
  -- corrupt or truncated JSON, `"genie": null` or a non-object `genie`, a
  chat_template missing a delimiter key, or the Jinja STRING where the delimiter
  block belongs each used to leave a raw traceback out of `main()` with nothing
  listening. The server now falls back to standard Qwen ChatML and prints
  `[genie] WARNING: <path> is unusable as a chat template (...)`, because that
  substitution is otherwise invisible: generic ChatML renders every bundle's
  turns plausibly, so a bundle whose real delimiters were dropped would serve
  slightly-wrong prompts forever with nothing in the log to explain the quality.
  A MISSING file, or a bundle carrying no chat_template at all, is the
  documented fallback rather than a degradation and stays quiet.
  Each message's `content`, here and on `/v1/messages`, may be a string, a LIST
  of content blocks, or a single content block on its own
  (`{"type":"text","text":"hi"}` -- clients
  write it by hand, and a `tool_result`'s own `content` arrives that way); text
  and `tool_result` blocks contribute text, images and `tool_use` contribute
  none. A content value that is none of those -- a number, a bool -- is a
  **400** naming the type it got, where it used to flatten to `""` and be
  answered 200 over a turn with no words in it.
  Responses report `GENIE_MODEL_ID` as `model`.
- `POST /v1/messages` -- the Anthropic Messages API, which is what typed
  speaks: real Anthropic SSE (`message_start` .. `message_stop`, `ping`),
  `input_schema` tools converted to the function shape Qwen3 was trained on,
  `tool_use` / `tool_result` content blocks, `stop_sequences`, the
  `stop_reason` mapping under [finish_reason](#notes--limitations) below, and
  **529** `overloaded_error` for backpressure where the OpenAI leg sends 429.
  One deliberate asymmetry: the response ECHOES the request's `model` string
  (falling back to `GENIE_MODEL_ID` when the request names none), as the real
  Messages API does and as a router fanning one request out to several engines
  needs for matching replies -- so a request for `claude-x` is answered with
  `"model": "claude-x"` by an NPU Qwen. That is a routing label, not a claim
  about what ran; `/health`, `/props`, `/v1/models` and the OpenAI leg all
  report `GENIE_MODEL_ID`.
- `GET /v1/models` -- lists the served model id (`GENIE_MODEL_ID`), as one
  superset object satisfying both the OpenAI (`id` / `object`) and Anthropic
  (`type` / `id` / `display_name`) model shapes.
- `GET /v1/models/<id>` -- the retrieve-model route some SDKs call to validate
  a name before the first request. `GENIE_MODEL_ID` returns that same object;
  any other id is a 404 with `code: "model_not_found"` and a message naming the
  model that IS served.
- `GET /props` -- llama.cpp-shaped metadata: `default_generation_settings.n_ctx`
  and `model_alias` / `model_id`, which is what typed reads. Plus a namespaced
  `genie` block carrying what a router cannot otherwise learn over HTTP:

  | field | why a dispatcher needs it |
  |---|---|
  | `engine` | `npu-hexagon-htp` -- which silicon is answering |
  | `single_flight` | `true`; the constraint behind the 429/529, stated rather than discovered from one |
  | `context_lengths` | the graphs compiled into the bundle; `[]` when `metadata.json` gave no list (build unknown) |
  | `multi_length` | `false` means 2-3x slower on short prompts at the SAME `n_ctx`. **`null` means unknown** -- `context_lengths` is `[]` -- and must not be read as `false` (it used to be `false` then, which down-ranked an endpoint whose build was merely unreadable) |
  | `poll` | `true` means an idle 2.7-core busy-wait, ~36% of decode, and about a quarter of the NPU+GPU concurrency win. **`null` means no `poll` key was read** -- absent from the config, or the config unreadable -- which is unknown, NOT the shipped `true` (what QnnHtp does with the key absent is not measured here); the startup note says which of the two it was |

  `n_ctx` alone is not enough to rank this endpoint against a GPU or CPU one,
  and on this engine it is actively misleading: it is the SOFTWARE cap
  (`dialog.context.size`), while throughput is set by the compiled window and by
  how many graphs the bundle carries. Two bundles reporting the same `n_ctx`
  differ 2-3x on a short prompt. The block is additive and namespaced, so a
  client that ignores it sees exactly the response it saw before.
- `GET /health` (alias `GET /healthz`) -- **engine** state, not process
  liveness. `200` when the server can actually generate; `503` with a `state`
  of `failing`, `stalled` or `wedged` when it cannot. The body carries `state`
  (and the same value as `status`), `detail` (why), `model`, `generating`,
  `tokens_in_flight`, `generations`, `consecutive_failures` and
  `token_counts`.

  What those fields count, because each was wrong once:

  | field | meaning |
  |---|---|
  | `generating` | true from the moment a request takes the engine lock -- covering the dialog reset, stop-sequence, sampler and token-cap calls that precede the query, each of which is a call into the same driver and can wedge like one. It used to turn true only at `GenieDialog_query`, so a hang in any of the others read `ok` for as long as it lasted. |
  | `generations` | client generations only. The server's own summarisation calls are supervised identically but not counted, on the success path or the failure path. |
  | `consecutive_failures` | generations that ended in an ERROR status or a throw. A client abort (`ABORTED`) and a full window (`CONTEXT_EXCEEDED`) are endings, not failures, so a user leaning on the stop button cannot flip `/health` to `failing`. The one abort that IS a failure is the WATCHDOG's, sent for a stall: that generation is booked as failed rather than resetting the streak, so a device that stalls on every turn but honours each abort still reaches `failing` (the client of such a turn receives an error -- a 500, or an error frame on a stream -- not a finish; see Supervision). A throw inside a generation DOES count -- it used to be booked as a success and reset the streak. |
  | `token_counts` | `exact` when the Genie tokenizer is attached; `estimated` (chars/4) when `GenieDialog_getTokenizer` failed. It says whether every usage figure, window budget and `length` finish this server reports is a count or a guess -- the startup log carries the matching `WARNING: GenieDialog_getTokenizer failed` line. |

  The distinction is the entire point: a wedged HTP leaves this process
  perfectly able to accept a connection and answer this endpoint while unable
  to serve a single token, so a plain liveness ping reports healthy for as long
  as the outage lasts. The handler touches nothing on the engine, which is what
  lets it answer *during* a wedge -- the engine lock is exactly what the stuck
  thread is holding.

  While it says `stalled` or `wedged`, the two generation routes say so too:
  a `POST /v1/chat/completions` or `/v1/messages` is refused **503** at the
  door (`server_error` / `api_error`, message `engine <stalled|wedged>:
  <the /health detail>. Not queued behind it -- retry on another engine; GET
  /health reports when this one recovers.`) instead of queueing behind the
  stuck call or being told "server busy". Not for `failing`: that state
  clears only when a generation succeeds, so refusing generations would make
  it permanent.

```bash
curl http://127.0.0.1:8123/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b-npu","messages":[{"role":"user","content":"Hi"}],"max_tokens":128}'
```

`python src\genie_smoke.py [BASE]` does that for you, against a server that is
ALREADY running -- it loads nothing itself (`-h` / `--help` / `/?` prints its
usage and exits 0; `--help` used to be taken as the BASE and reported as
`unknown url type: '--help/v1/models'`). It GETs `/v1/models`, then asks for
one non-streamed and one streamed completion, printing `served: <ids>` and the
`model:` that answered each (so you can see which engine you smoked, not infer
it), the content, the finish reason, chunk count and TTFT. BASE defaults to
`http://127.0.0.1:8123`, with `GENIE_PORT` overriding the port (a value that is
not a port number is a `note:` line and 8123, where it used to go unparsed into
the URL and come back as an `http.client.InvalidURL: nonnumeric port`
traceback) -- it used to
default to 8080, which with the documented two-engine stack up is the
llama-server CPU leg: the non-streaming half passed against the wrong engine,
and the streaming half then died in a TypeError, because llama-server opens
every stream with a delta of `{"role": "assistant", "content": null}` and the
null went into the join. (Only a non-empty string is a content delta now, so a
run against llama-server completes and is judged like any other.) **The exit
status is the verdict:** 0 only when both completions carried content and
finished `stop` or `length`; a `FAIL: ...` line and 1 otherwise, including for
a streamed error frame. The streamed half also judges the FRAMING, not just
the payloads: an SSE event is dispatched by a blank line, and a reader that
keeps `data:` lines regardless cannot tell a stream of frames from one frame
that never ends -- so a server writing one newline where the separator's two
belong, unreadable to every SDK, used to be smoked clean. Two `data:` lines
with nothing between them are now counted and reported as `N SSE frame(s) were
not terminated by a blank line`, independently of the content verdict; a
`: keep-alive` comment or an `event:` line between frames is not one. A 4xx/5xx prints `HTTP <code> during <stage>: <body>`
and exits 1 rather than leaving a urllib traceback with the body unread; a
server that is not there (connection refused, a timeout, a reset mid-stream),
one that does not speak HTTP at all, a response that stops mid-body, a 200
whose body is not JSON and a 200 of an unexpected SHAPE print `FAIL:
<ExceptionName> during <stage>: ...` naming the request and exit 1 the same
way, where each used to be a traceback. A REFUSED connection adds a
`Nothing is listening at <BASE>.` line giving both readings, in
`bench_endpoint`'s words: nothing is running there, or a `genie_server` there
is still loading its bundle (it holds the port but refuses connections until
the model is resident -- wait for its `endpoint on` line and re-run rather
than start another). Otherwise it names the two ports a server here is
normally on, since the default moved.

## Environment

Every `GENIE_*` variable the server, its launcher and the bench tools read.
The "default" column is what the CODE does with the variable unset -- where the
launcher supplies a different value, the row says so.

**A malformed value never stops the server from booting.** Every numeric
variable here (`GENIE_PORT`, `GENIE_MAX_TOKENS`, `GENIE_MAX_BODY_BYTES`,
`GENIE_SOCKET_TIMEOUT`, `GENIE_WINDOW_MARGIN`, `GENIE_SUMMARY_MAX_TOKENS`,
`GENIE_FIRST_TOKEN_TIMEOUT`, `GENIE_STALL_TIMEOUT`, `GENIE_WEDGE_GRACE`,
`GENIE_FAIL_THRESHOLD`, `GENIE_MAX_INFLIGHT`, `GENIE_SEED`,
`GENIE_ORPHAN_HOLD_CHARS`) degrades to its default with one line naming the
variable and the value it rejected -- `[genie] WARNING: GENIE_PORT='808O' is
not an integer; using 8080 instead.` -- where a typo used to kill the process
at import with a bare `invalid literal for int()`, before any startup line had
printed. The launcher's three integers and the two bench-tool knobs behave the
same way, each with its own prefix (`[run]`, `[bench]`).

**And neither does a value that parses but is out of range.** Where a variable
has a floor, a value below it degrades the same way rather than being clamped
in silence -- `[genie] WARNING: GENIE_MAX_TOKENS='-1' must be >= 1; using 512
instead.` The value it falls back to is the DEFAULT, not the bound: 512 is a
cap somebody might have wanted, 1 is not. A silent clamp reads as acceptance,
which is how `-1` (llama.cpp's spelling of "no limit", and this repo runs a
llama-server leg beside this one) turned every uncapped completion into a
one-token answer with no line anywhere saying why.

**The four on/off switches read every usual spelling, either way.**
`GENIE_WEDGE_EXIT`, `GENIE_THINKING`, `GENIE_STRIP_THINK` and
`GENIE_SUMMARIZE_EVICTED` are case-insensitive and ignore surrounding spaces:
`1` / `true` / `yes` / `on` turn one on and `0` / `false` / `no` / `off` turn
it off. Anything else is `[genie] WARNING: GENIE_THINKING='maybe' is not an
on/off value (1/true/yes/on or 0/false/no/off); using off instead.` and the
default. Each switch used to parse its own way, and each way read some value
as the OPPOSITE of what was meant, silently: `GENIE_WEDGE_EXIT=False` (what
PowerShell's `$env:GENIE_WEDGE_EXIT = $false` stores), `off` or ` 0` left the
wedge exit ON; `GENIE_THINKING=True` or `on` left thinking off;
`GENIE_STRIP_THINK` took nothing but `1`; and `GENIE_SUMMARIZE_EVICTED` read
anything but `0` as on, `false` included.

| var | default | meaning |
|---|---|---|
| `GENIE_BUNDLE_DIR` | **none -- required.** The launcher sets it to `<GENIE_NPU_ROOT>\bundles\<the -Model bundle>`: the 8192 multi-length 4B unless `-Model` says otherwise | dir with genie_config.json + part*_of_*.bin + tokenizer.json. **Prefer a MULTI-length bundle** -- check `genie.context_lengths` in its metadata.json; a single-length one is 2-3x slower on short prompts. A relative path is made absolute at startup (the server chdirs into the bundle, which used to turn a relative one into a bare FileNotFoundError after it had passed the existence check). Run directly with it unset and the server exits naming it; a directory with no `genie_config.json` (usually the one ABOVE the bundle), or a bundle missing a file its config names, is refused by name before the load too -- see [Run](#run). |
| `GENIE_SDK_DIR` | **none -- required.** The launcher sets it to the newest version directory under `<GENIE_NPU_ROOT>\qairt` | QAIRT 2.45 root (lib/aarch64-windows-msvc, lib/hexagon-v*). Made absolute like the bundle dir. |
| `GENIE_HEXAGON_ARCH` | unset | pin one skel arch (`v81`); default offers all |
| `GENIE_SUMMARIZE_EVICTED` | 1 | `0` (or `false` / `no` / `off`, any case) disables summarising evicted turns (plain drop). An on/off switch -- see above; `false` used to leave it on. |
| (not an env var) | -- | **`poll: false` in the bundle's `genie_config.json`** -- see the poll note below. Worth up to +55% decode and frees 2.7 idle cores. The server now CHECKS this at startup (before the 11-35s load) and warns loudly if the bundle ships `true`; it also warns on a single-length bundle. Both are warnings, never refusals -- a slow server is still a working one. |
| `GENIE_SUMMARY_MAX_TOKENS` | 192 | cap on the retained note. Clamped at runtime to `n_ctx / 8` (floor 32) so the note cannot crowd out the window on a small-context bundle; the server logs the clamp when it bites. |
| `GENIE_WINDOW_MARGIN` | 64 | headroom left between prompt and n_ctx. Must be >= 0: a negative margin is not more headroom, it is a budget past the window. A negative value is rejected with a WARNING line and the default 64 is used -- it is not clamped to 0. |
| `GENIE_MAX_INFLIGHT` | 2 | requests admitted at once (1 running + queue). Floored at 1 -- it cannot be disabled, since the NPU is single-flight and an unbounded setting only parks threads on the engine lock. Set 1 to protect KV reuse: two interleaved conversations share one resident KV and reset each other's prefix. |
| `GENIE_HOST` / `GENIE_PORT` | 127.0.0.1 / **8080** | bind address. Note the launcher overrides the port: `run-genie-server.ps1` sets **8123** unless `GENIE_PORT` is already set, because 8080 is where `run-llama-server.ps1` puts its CPU leg. It also PARSES and range-checks the value (1-65535) before exporting it, so a typo falls back to the launcher's 8123 with `[run] WARNING: GENIE_PORT='808O' is not a port number (1-65535); using 8123.` rather than reaching the server and degrading to ITS default 8080 -- which is the CPU leg. So the endpoint is `127.0.0.1:8123` when started the normal way, and `127.0.0.1:8080` only if you run `genie_server.py` directly. The tools follow the launcher: `bench_endpoint`, `bench_contention --npu` and `bench_servers --ours-port` default to 8123, and `genie_smoke.py` to 8123 with `GENIE_PORT` overriding the port -- it validates it too, printing `note: GENIE_PORT='abc' is not a port number (1-65535); using 8123.` and using 8123, where it used to die in an `http.client.InvalidURL: nonnumeric port` traceback. A base URL without `http://` or `https://`, a host and a port in 1-65535 is refused before any request: `bench_endpoint --base` exits 1, and `bench_contention` refuses an `--npu` or `--gpu` at startup through argparse (exit 2), in the same words. `bench_contention`'s failed ping now says which of two things it saw: nothing listening (not running, or a `genie_server` there still loading -- wait for its `endpoint on` line), or something that took the connection and answered unusably (do not start another on that port). `::1` and `::` bind too (the address family is chosen from the host). A bind that cannot succeed -- an address this machine does not have, a `GENIE_PORT` outside 0-65535 (which now only a direct `python src/genie_server.py` run can reach) -- exits with `cannot bind HOST:PORT: ...` BEFORE the model load. **There is no authentication.** The loopback default is the security model: anyone who can reach the port can use the NPU, read what it generates, and wedge the device for everyone else. Binding `0.0.0.0` is supported and the server warns at startup when you do, but put something in front of it. |
| `GENIE_MODEL_ID` | qwen3-4b-npu | id reported to clients. The launcher sets it from `-Model` (`qwen3-8b-npu`, `qwen3-8b-8192-npu`), or from the bundle directory name when that is one it knows. |
| `GENIE_NPU_ROOT` | `../genie-npu` beside this repo | launcher only: where `bundles/` and `qairt/` live. Set this instead of the two paths above; the newest `qairt/*` BY VERSION NUMBER is picked automatically, and directories not named like a dotted version are ignored. "A dotted version" means what `[version]` can hold: two to four dot-separated components of at most nine ASCII digits each. So an all-digits stray -- a directory named for a full build stamp (`2.45.0.260326153000`), a timestamped backup -- is ignored like `latest` is, instead of killing the launcher with a cast error out of the sort while a valid SDK sits right beside it. |
| `GENIE_PYTHON` | `python` (first on PATH) | launcher only: the interpreter to run the server with. It must be native ARM64 -- Genie.dll is aarch64-only -- and the launcher exits 1 naming the interpreter when it is missing or is not. (That used to be a warning followed by a DLL-load failure that named neither.) The question asked is the interpreter's own BUILD -- `python -c "import sysconfig;print('GENIE_ARCH=' + sysconfig.get_platform())"` -- and the answer is the last line matching `^GENIE_ARCH=`, which must contain arm64 (`win-arm64` accepted, `win-amd64` refused). `platform.machine()` was the old question and is wrong on this box: from CPython 3.12 on Windows it reports the HOST cpu, so an emulated x64 python answered `ARM64` and was accepted, differently from launch to launch (measured here: 10 of 10 accepted-then-refused flaps gone, 10 of 10 now refused). Because only a TAGGED line is read, a `.cmd` shim may print before python AND after it (`exit /b %ERRORLEVEL%`, for want of `@echo off`) and is still accepted -- the old wording promised that and the last-non-blank read broke it. The refusal quotes the build tag (`[run] python arch is 'win-amd64' (...)`), and an interpreter that prints no tagged line is refused with either `[run] (It printed no text on stdout; if it failed, its own error is above.)` or `[run] (It printed N line(s) but no GENIE_ARCH= line; its own error, if any, is above.)`; a silent one used to be accepted, and the server launched under an interpreter that had just failed to run one line. The start line reads `[run] starting Genie server (python win-arm64) on ...`, since the build tag is what was checked. |
| `GENIE_MAX_BODY_BYTES` | 8388608 | largest request body accepted, in bytes (8 MiB), checked against `Content-Length` BEFORE the read and before the single-flight queue. Larger is a 413; orders of magnitude above any legitimate prompt at n_ctx 16384. |
| `GENIE_SOCKET_TIMEOUT` | 120 | seconds any ONE socket read or write may block before the connection is dropped; `0` or less waits forever (the old behaviour). It is **not** a cap on generation time -- a handler never blocks on the socket while it waits for the engine. What it bounds: an idle keep-alive connection is closed after it; a request body that stops arriving is dropped with no response and one log line; a client that stops READING a stream is treated as gone and its generation aborted. Before it existed each of those parked a handler thread for the life of the process, ahead of the `GENIE_MAX_INFLIGHT` queue. |
| `GENIE_FIRST_TOKEN_TIMEOUT` | 300 | seconds a generation may run before its first token before being called stalled. Generous because prefill at depth legitimately takes tens of seconds. Also the limit for a host-side native call that is not a generation (the tokenizer encode that sizes a request) -- see Supervision. Seconds, may be fractional, as may the next two. |
| `GENIE_STALL_TIMEOUT` | 120 | seconds between tokens before being called stalled. This is the real wedge signal -- see Supervision. |
| `GENIE_WEDGE_GRACE` | 60 | seconds an abort gets to take effect before the stall is escalated to a wedge |
| `GENIE_FAIL_THRESHOLD` | 3 | consecutive failed generations before `/health` reports `failing`. Floored at 1. A client's abort and a full-window finish are not failures; an abort the watchdog sent for a stall is -- see `/health` above. |
| `GENIE_WEDGE_EXIT` | 1 | `0` (or `false` / `no` / `off`, any case -- `False` and `off` used to leave the exit ON) keeps the process up on a wedge instead of exiting for a supervisor. The watchdog then keeps watching, and `/health` and every generation request answer 503 `wedged`; the WEDGED stanza is printed once, not every five seconds. |
| `GENIE_MAX_RESTARTS` | 5 | launcher only: engine failures (wedges and native crashes) tolerated within `GENIE_RESTART_WINDOW` before it gives up. `0` is valid (give up on the first). A non-integer or negative value warns and uses the default. |
| `GENIE_RESTART_WINDOW` | 3600 | launcher only: seconds of history the restart cap counts over. Failures older than this age out, and the launcher says so when they do. Minimum 1; junk warns and uses the default. |
| `GENIE_RESTART_COOLDOWN` | 25 | launcher only: seconds between restarts. Not arbitrary -- a force-killed server needs roughly 20s of settling, and restarting sooner was measured costing about half of decode throughput. `0` is valid; a non-integer or negative value warns and uses the default (a negative one used to throw from `Start-Sleep` inside the restart path). |
| `GENIE_MAX_TOKENS` | 512 | default cap when a request sets neither `max_tokens` nor `max_completion_tokens`. Must be >= 1: `0` or `-1` is rejected with a WARNING line and 512 is used (`-1` is llama.cpp's no-limit spelling; here it used to reach the engine as 4294967295, `0` skipped the cap call altogether, and a silent clamp to 1 made every uncapped answer one token long). |
| `GENIE_STRIP_THINK` | 0 | `1` (or `true` / `yes` / `on`, any case; it used to take only `1`) strips a well-formed `<think>...</think>` pair from every BUFFERED response -- non-streaming, and a tool stream on either API, which is generated in full before it is framed. Only an incremental (non-tool) stream is always faithful to the model, because a frame already sent cannot be retracted. Unrelated to the orphan-close strip below, which is always on because it removes a DUPLICATED answer rather than the model's reasoning. |
| `GENIE_MIN_DECODE_STEPS` | 16 | `bench_endpoint`, and through it `bench_servers` and `bench_contention`: fewest decode steps a rate may rest on. Below it the delta is measuring per-request overhead rather than decode -- a 4-step window once reported **0.60 tok/s against a true 17.6**. Such a sample is refused with a line naming the count, not averaged in. `bench_contention` goes one further and refuses a `--tokens` below this floor at startup (exit 2), rather than running a sweep in which every leg is REFUSED; `bench_servers` now does the same, with the same wording, exiting 1 (a `sys.exit`, not argparse's `ap.error`). `bench_endpoint` itself now does too, exiting 1 before its /health check, except under `--prefill-only`, where `--tokens` only sizes the depth budget. It also floors the probe that corrects prefill, which asks for exactly this many steps (16; it was 8). A non-integer value is reported at startup and the default used; the knob is described in `python src/bench_endpoint.py --help`. |
| `GENIE_LOW_CHARGE_PCT` | 25 | `bench_contention` only: pack percentage below which a timing run is flagged as not-a-settled-baseline **even on AC**. Measured on this box: at 13-20% charge, CPU pp512 comes back ~58 against a settled 130, while decode barely moves. Advisory, never fatal -- bandwidth-bound work is largely immune. A non-numeric value warns and falls back to 25; described in `python src/bench_contention.py --help`. |
| `GENIE_SEED` | unset | pins the sampler seed. Unset means a fresh seed per PROCESS, which is what stops every fresh prompt replaying the same answer -- the bundles ship a fixed `42` and Genie re-seeds from it on every dialog reset. Pin it for reproducibility (comparing bundles, bisecting a bad generation); throughput does not depend on it. Per-REQUEST variation is not available -- see the note below. |
| `GENIE_ORPHAN_HOLD_CHARS` | -1 | how much of a STREAM to withhold while deciding whether the model is about to close a `<think>` block the prefill opened. `-1` holds until that is settled (so a streamed reply arrives as one frame at the end -- correct, not incremental). `0` streams every chunk as it arrives and ships the occasional doubled answer. A positive value is a bounded hold, which was measured LEAKING. Non-streaming and tool paths are unaffected; they buffer anyway and always strip. |
| `GENIE_THINKING` | **0** | Qwen3's reasoning block is **suppressed by default** -- it costs 10-17x on an agent turn (see the tool-calling note below). `1` (or `true` / `yes` / `on`, any case; `True` and `on` used to read as off) re-enables it server-wide. Per request either way: `chat_template_kwargs.enable_thinking`, `reasoning_effort` (`"none"` / `"high"`), or `thinking:{"type":"disabled"|"enabled"}` -- an explicit request always beats the server default. |

## Supervision: what happens when the HTP wedges

The Genie query is a blocking call into native code. When the device stops
making progress the calling thread is stuck inside the driver holding the
engine lock, and Python cannot reclaim a thread blocked in native code -- no
timeout, no interrupt, no kill. Until the stall is past its limit nothing can
tell it from a slow prefill, so for those first 120-300s later requests park
behind that lock until `GENIE_MAX_INFLIGHT` is exhausted and the rest get a
fast `429` / `529`. Once `/health` says `stalled` or `wedged`, every new
generation request is refused **503** by name instead (see `/health` under
[Endpoints](#endpoints)) -- where it used to go on parking or being told
"busy", so from outside the server looked like one that 429s forever while
sitting idle.

Detection is by **stalled progress, not elapsed time**. A long generation is
not a wedge -- 2000 tokens at the slowest measured 3.3 t/s is ten minutes of
healthy work -- but it emits tokens the whole way, and a wedge emits nothing.
Time since the last token separates slow from stopped without capping how long
a request may legitimately run.

Escalation, in order:

1. **Stalled** -- no first token in `GENIE_FIRST_TOKEN_TIMEOUT`, or no further
   token in `GENIE_STALL_TIMEOUT`. `/health` goes 503; for a stall INSIDE
   `GenieDialog_query` the server signals Genie's abort, which is free and is
   the mechanism provided for exactly this. On a healthy device this is usually
   where it ends: forced against a live generation, the abort landed and the
   engine returned to `ok` on its own. The turn is still booked as a FAILED
   generation -- a stall that an abort happened to clear is still the engine
   not serving -- so a device that stalls on every turn and honours every abort
   reaches `failing` instead of reading healthy between stalls. Its client is
   told the turn failed, not that it finished: a non-streaming request gets a
   **500** (`server_error` / `api_error`, `the engine stalled: no token
   arrived within its limit, so the watchdog aborted this generation part-way
   and it did not finish. GET /health reports the engine's state.`), and a
   stream ends on its API's error frame (see the streaming-failure note) with
   no `finish_reason` / `stop_reason`. It used to receive an ordinary 200
   `stop` / `end_turn` carrying whatever had been generated -- a fragment an
   agent files as a complete answer. The log line for a stream reads `aborted
   mid-stream (watchdog)`.

   The first-token clock starts when the request takes the engine lock, not at
   `GenieDialog_query`, so a hang in the reset, stop-sequence, sampler or
   token-cap call that precedes the query is a stall too. And the host-side
   native calls that are NOT generations -- `GenieTokenizer_encode`, both when
   it sizes a request and when it re-counts a finished generation against its
   cap -- run on a clock of their own with the same limit: a hang there reports
   `stalled` and then `wedged` with a detail of `no return from
   GenieTokenizer_encode for Ns ...`, where it used to report `ok`.

   So the watchdog tells three stalls apart, and its log lines say which,
   rather than claiming a signal it never sent. All three open on the same bare
   `STALL: <detail>`: the stall is announced BEFORE the attempt, because
   signalling an abort can itself block inside a wedged driver, and a single
   line composed afterwards would say nothing at all in exactly the case that
   matters. What was tried is the line after it:

   | where the stall is | what the watchdog does and prints |
   |---|---|
   | inside `GenieDialog_query` | sends the native ABORT and says so on a second line, `signalling abort to the generation in flight`, again every interval until it takes or the grace runs out |
   | a host-side call with no turn (the tokenizer sizing a request) | nothing to signal: `nothing in flight to abort: the stall is in a host-side call, so only the exit below can clear it` |
   | a turn stuck in a call BEFORE its query (the reset, the stop sequences, the sampler, the token cap) | no native signal either -- ABORT is aimed at a query and this turn is not in one: `no native ABORT was sent: the turn is stalled in a call BEFORE its query, which ABORT cannot reach. It is flagged, so it will not start its query if that call returns; otherwise only the exit below can clear it` |

   The closing clause in the last two rows is the `GENIE_WEDGE_EXIT=1` wording.
   Under `GENIE_WEDGE_EXIT=0` there is no exit to wait for, so it reads
   `nothing in this process can clear it -- with GENIE_WEDGE_EXIT=0 the server
   stays up wedged, answering 503, until you restart it by hand` instead.
   Whoever sets that variable sets it BECAUSE nothing supervises the process,
   which is the one operator the old wording sent off to wait for a restart
   that was never coming.

2. **Wedged** -- the stall outlasted `GENIE_WEDGE_GRACE`, counted from the
   first time the watchdog acted on it. Nothing in-process can help, so the
   server exits **75** (`EX_TEMPFAIL`) and asks to be replaced.
   `GENIE_WEDGE_EXIT=0` keeps it up instead: the watchdog goes on watching, and
   `/health` and every generation request go on answering 503. The WEDGED
   line and the `/health` detail end on what was actually tried: `; an abort was signalled Ns ago and did not
   take` when a native ABORT went out, and `; first seen Ns ago and still stuck.
   No ABORT was sent: the stall is outside GenieDialog_query, where none can be
   delivered` for the other two -- so nobody files "Genie ignores ABORT"
   against a driver that was never asked.

   The stanza is printed ONCE, not every five seconds, so it is the whole
   record of the event -- which is why its second line branches on
   `GENIE_WEDGE_EXIT` rather than describing the other configuration's ending.
   It always opens `The engine cannot be recovered in this process: the stuck
   call is inside the Genie driver, holding the engine lock, and Python cannot
   reclaim a thread blocked in native code.` and then ends either `Exiting 75
   so a supervisor restarts a clean process. (GENIE_WEDGE_EXIT=0 to stay up
   and keep reporting 503.)` or, under `GENIE_WEDGE_EXIT=0`, `NOT exiting:
   GENIE_WEDGE_EXIT=0. This process stays up wedged -- /health and every
   generation request answer 503 -- until you restart it by hand. (Unset
   GENIE_WEDGE_EXIT to exit 75 instead, for a supervisor to restart a clean
   process.)`

   The exit itself is `TerminateProcess` on its own process, not `os._exit`:
   on Windows `os._exit` is `ExitProcess`, which runs every loaded DLL's
   detach code -- `Genie.dll`, the `QnnHtp*` libraries -- and a detach that
   waits on the driver that just wedged hangs the exit (measured with a
   stand-in DLL whose detach sleeps 8s: `os._exit` took 8.02s, `TerminateProcess`
   0.00s). `TerminateProcess` runs no user-mode code, the way a crash leaves.
   It is not a cure for a thread stuck in KERNEL mode -- no exit completes
   until that thread lets go -- and `run-genie-server.ps1` waits on the child
   with no deadline, so that case would still hang. None of this has run
   against a real wedged NPU (see the README's untested list).
3. **Restarted** -- `run-genie-server.ps1` restarts on exit 75 **and on a native
   crash**; any other code is a deliberate exit (Ctrl-C, a config error it
   already explained) and repeating it would be pointless. Restarts are capped
   and rate-limited, because looping on a device that wedges every time keeps
   the HTP busy and buries the original failure under identical log stanzas.
   The cap is a SLIDING WINDOW: more than `GENIE_MAX_RESTARTS` failures within
   `GENIE_RESTART_WINDOW` seconds (default 3600) gives up, exiting 75 so an
   outer supervisor sees the same signal; older failures age out, and the
   launcher prints when they do. Each restart line reads `restart N/M within
   Ws; next in Cs`. (The rule this replaced counted only restarts that followed
   a life shorter than 120s. It could never count a wedge -- the detector itself
   needs at least 180s, stall 120 + grace 60, before the server exits 75 -- so a
   device wedging on every load restarted forever at "restart 1/5" and the
   give-up branch was dead for the case it exists for.) The give-up message
   carries a `pnputil /restart-device` hint; the instance id in it is the dev
   box's, and the line beside it says how to find yours (`Get-PnpDevice
   -FriendlyName '*Hexagon*'`). The hint is now preceded by a machine-wide
   warning -- restarting the Hexagon device, or rebooting, resets the NPU
   under EVERY process on the box, other sessions' servers and benchmarks
   included -- and by a check to run first, `tasklist /m QnnHtp.dll` from the
   elevated shell (this server has already exited, so everything it lists is
   someone else's; warn them). And it is scoped: the restart is verified only
   against the interrupt-delivery crawl (`MODEL_OPTIONS.md`), and untested
   against a wedge or a crash; a reboot is the other way back.

   The crash case is not hypothetical and is why "75 and only 75" was wrong:
   the driver can fault instead of hang, and WER on the dev box records
   `python.exe` dying with `0xC0000005` inside `QnnHtp.dll` at the same offset
   twice (2026-08-24, 2026-08-27). That is a wedge by another name -- the
   engine is gone and only a fresh process brings it back -- but it exits with
   an NTSTATUS rather than 75, so the old rule gave up on precisely the failure
   the loop exists to recover from. A crash is recognised by MAGNITUDE, not by
   sign: an exception code carries a severity, a facility and a code field, so
   it is always enormous (`0xC0000005` access violation, `0xC0000409` stack
   buffer overrun, `0xE06D7363` unhandled C++ exception, `0x80000003`
   breakpoint), while a deliberate failure is `exit(1)` or a sloppy `exit(-1)`.
   That `-1` arrives as `0xFFFFFFFF` -- inside any "negative means crash"
   window while meaning the opposite -- so the test is `$LASTEXITCODE <=
   -65536`, above every deliberate small negative and below every real
   exception code. `STATUS_CONTROL_C_EXIT` is carved out on top of that, so the
   operator's own Ctrl-C never becomes a restart. The log line names which of
   the two happened, since a crash leaves a WER report and a faulting module to
   look up and a wedge leaves nothing but a stuck thread. A streak that mixed
   both is reported as both, rather than as whichever kind happened last.

Separately, `consecutive_failures` reaching `GENIE_FAIL_THRESHOLD` reports
`failing` on `/health` **without** restarting: the engine is answering, just
badly, and restarting on that would turn a bad bundle into a crash loop. What
counts is a generation that ended in an error status or a throw -- including a
`GenieDialog_setStopSequence` that Genie rejects, which now fails the request
(500, or an error frame on a stream) instead of generating under the previous
caller's stop list. Two more statuses on that same path are read rather than
discarded, and each fails the request with a named `RuntimeError` in the same
way. A rejected `GenieDialog_reset` no longer lets the turn prefill on top of a
KV that may still hold the PREVIOUS conversation -- the commit that followed
recorded prompt-plus-generated as the resident text, the next turn's byte-prefix
check passed, and the model answered from a history that never happened, with no
error and no log line. And a rejected `GenieDialog_setMaxNumTokens` no longer
generates under whatever cap the previous request left on the dialog:
`GenieDialog.h` documents `ERROR_GENERAL` for a cap that "could not be applied",
which is value-dependent and so reachable from an ordinary client `max_tokens`,
and a turn silently held at the earlier request's 8 tokens came back short
reporting `finish_reason: "stop"` -- which an agentic client reads as a complete
answer. A client abort and a full window are not failures, so the
stop button cannot produce this state. The watchdog's own abort of a stalled
turn is the exception: that generation counts as failed, so repeated stalls
that each clear on abort add up to `failing` here rather than resetting the
streak every time.

**Shutdown (Ctrl-C)** aborts whatever holds the dialog and then CLOSES the
engine under the engine lock, so the handle is never freed beneath a generation
still inside `GenieDialog_query` -- which at best left a spurious `0xC0000005`
in the WER log beside the real driver faults this section relies on. Closing
is more than the free: under the lock, and ahead of `GenieDialog_free`, a
closed flag is set and the dialog and tokenizer handles are nulled. Handler and
worker threads are daemons and outlive `main()` for as long as the interpreter
takes to leave, and with the free alone each of them was one lock acquisition
away from a native call on the freed handle. After the close none is made: a
request that was queued behind the lock fails with `the engine is closed: the
server is shutting down and the Genie dialog has been freed`, token counts fall
back to the chars/4 estimate, and an abort sends nothing. If the lock is not
released within 5s the driver is stuck, nothing is touched -- a free on a stuck
driver can hang too -- and the server says so: `a generation is still inside
the driver; leaving the dialog for the OS to reclaim`.

The turn that was ALREADY waiting for the engine lock when shutdown began is
refused too, and that took a second step. An abort frees the lock, and the
lock goes to whoever has waited longest -- which under load is a queued
request, not `close()`. `GENIE_MAX_INFLIGHT` is 2, so one running plus one
queued is an ordinary loaded moment, and on Ctrl-C that queued turn used to
win the race: it went on to `GenieDialog_reset` and a fresh
`GenieDialog_query`, `close()` timed out on it and blamed a driver that was
working perfectly, and the process left with a generation running inside
Genie. So `main()`'s finally calls `ENGINE.begin_shutdown()` -- mark closing
under the abort lock, THEN abort the turn in flight -- before `close()`, and
the parked turn raises `the engine is shutting down: no new generation will
start, the Genie dialog is about to be freed` when it gets the lock. The flag
has to precede the abort, because the gap between the two calls is itself the
race.

Both shutdown refusals -- the one above and the closed one -- are answered
**503** (`server_error` / `api_error`) on the non-streaming paths, not 500:
nothing was attempted and nothing about the request was wrong, which is this
server's "503 = shed" contract, so a router can send it to another leg instead
of booking a failure. On a stream whose 200 is already out it arrives as the
usual error frame or event.

The turn that was IN FLIGHT when Ctrl-C arrived -- the one the abort cut short
-- is now told the same thing. `begin_shutdown` marks it before aborting it,
and a turn the abort actually cut is answered **503** on a non-streaming
request (`the server is shutting down: this generation was aborted part-way
and did not finish. Retry on another engine.`), or, on a stream, the API's
error frame after whatever text had gone out, with no `finish_reason` /
`stop_reason` (log line: `aborted mid-stream (shutdown)`). It used to be a 200
carrying the text cut wherever the abort landed, labelled `stop` / `end_turn`
-- a fragment an agent files as complete, beside the 503 the queued turn got.
A turn that finished before the abort landed is reported as the whole answer
it is, and a client's OWN disconnect is unchanged: it left, so nothing is
sent.

## Notes / limitations

- **Context window: evict, don't crash.** The compiled window is fixed (read
  from the bundle, reported at `/props`; the 4B bundles here are 4096, 8192
  single-length, 8192 multi-length -- the launcher default -- and 16384, plus
  the 8B prebuilt at 4096 multi-length) and Genie has NO sliding-window mode -- QAIRT 2.45 exposes no
  such flag on `genie-t2t-run` and no equivalent config key, and overflowing is
  a hard `GenieDialog_query` failure, not a truncation. So the server evicts:
  oldest turns are dropped until the prompt fits, with the system turn and tool
  schemas anchored and tool results never separated from the call that produced
  them. That holds at the TAIL too, which is where it used to break: a
  conversation that ends on tool results -- every agent step does -- ends on a
  UNIT, the assistant turn that made the calls plus every result after it, and
  eviction may cut in front of that unit and no later. When the unit itself
  does not fit the request is a **400 naming the token counts**, never a 200
  over a prompt that opens on a bare `<tool_response>` with its call gone.
  Eviction is logged (never silent). A single message too big to fit even
  alone gets the same 400, not a doomed query.
  `GENIE_WINDOW_MARGIN` (default 64) is the headroom left for generation.

  **Every system message reaches the model, and only the LEADING ones are
  anchored.** A client that sends several -- a trailing per-turn reminder, a
  framework's injection -- used to have every one after the first dropped
  without a word. Now the leading system messages (those before the first turn
  of any other role) are folded, in order and blank-line separated, into the
  one anchored system turn. A `role: "system"` message LATER in the conversation
  is rendered inline where it was sent, as its own `<|im_start|>system` block
  -- which is what the bundle's Jinja does with it -- and is an ordinary turn
  from there on: evicted in order with the turns around it, never left at the
  head of what survives an eviction (there it would read as part of the system
  turn, behind the retained note), and handed to the summariser as `system:
  ...` when it goes. An empty one renders nothing.

  `role: "developer"` is the SAME role as `system`, everywhere the paragraph
  above says system: OpenAI renamed it, its SDKs emit the new spelling, and one
  that LEADS is folded into the anchored system turn while one that comes later
  is its own inline block, evicted in order and summarised as `system: ...` like
  any other turn. It used to fall through the renderer's unknown-role path and
  become a USER turn, which is wrong twice and silently: the agent's
  instructions were evictable, so the model got dumber as the conversation grew,
  and with no message of role `system` left the template's `default_system`
  ("You are a helpful AI assistant.") went in FRONT of them, contradicting the
  instructions with a prompt nobody sent. Any OTHER unrecognised role is still
  rendered as a user turn -- dropping it would lose words someone typed, which
  is the failure above.

  Folding ALL of them into the system turn, wherever they sat, is the obvious
  fix for the dropped messages and the wrong one, because that turn is the one
  thing eviction cannot shrink: a client that keeps a per-turn reminder in its
  history then grows it by a message per exchange until nothing else fits.
  Measured at `n_ctx=2048` with 400-character reminders: by 18 exchanges the
  stale reminders had pushed 34 of the 37 real turns out of the window, and
  from 20 on nothing fitted at all -- a 400 on every later request, since the
  client resends the same history. Inline, a per-turn reminder does not cost
  KV reuse either: it never touches the system turn, so a growing conversation
  that carries one still extends the resident prompt byte-for-byte (see the
  KV-reuse note below).

- **Eviction summarises instead of discarding.** Dropping the oldest turns
  outright makes the agent forget it already read a file and read it again --
  burning the window a second time on information it had. So when eviction
  fires, the outgoing turns are condensed by one NPU call into a short note
  folded into the SYSTEM turn (the one thing eviction never touches), under the
  heading `[genie_server note v1: summary of earlier turns]`.

  **The note carries forward, and for a stateless client that is the server's
  job.** No response returns the note and clients resend their history
  verbatim, so the next request arrives with no trace of it. The last note is
  therefore kept server-side -- ONE slot, like the resident KV, keyed by a
  digest of exactly the evicted turns it stands for, so one conversation can
  never be handed another's. On the next over-window request:

  | what was evicted this time | what happens |
  |---|---|
  | the same turns as last time | the stored note is reused as it is: **no NPU call, no overhead tokens, no dialog reset** -- so KV reuse survives eviction |
  | those turns plus newer ones | ONLY the newly evicted turns are summarised, with the stored note as the prior, and the result replaces it -- notes never stack and never restart from scratch |
  | anything else (another conversation, edited history) | summarised fresh |

  If a re-summary fails, the previous note is kept rather than lost. A
  summarisation CUT SHORT BY AN ABORT -- the watchdog's, for a stall, or
  shutdown's -- is a failed one too: Genie reports it as an ordinary `stop` with
  whatever had been produced, and a fragment is not a summary. The partial text
  is discarded and nothing is remembered as standing for those turns (the log
  says `the summary was cut short by an abort -- discarded, ...`), so the next
  over-window request offers the same turns to the summariser again; with a
  note already in hand, that good note is kept and the record of what it covers
  is not advanced. Two over-window conversations interleaving evict each
  other's note exactly as they evict each other's KV -- the same trade as
  `GENIE_MAX_INFLIGHT`. (Until
  2026-09-16 this paragraph claimed the carry-forward and the code did not do
  it: every over-window request re-summarised from the last few thousand
  characters of what it evicted, so a fact stated before that tail was gone
  after a few file reads.) A note a CLIENT sends back is honoured only as the
  last block of its system turn with the heading on a line of its own; a system
  prompt that merely quotes the heading is left alone. A client that sends no
  system prompt keeps the template's default one after its first eviction, with
  the note appended to it.

  Measured, same 30-turn conversation with a fact stated at the start:

  | | wall | answer |
  |---|---|---|
  | `GENIE_SUMMARIZE_EVICTED=1` (default) | 12.6s | recalled the key |
  | `=0` | 8.5s | lost it |

  That table was measured under the note's OLD heading (`[earlier context]`)
  and before the carry-forward above; it has not been re-measured, because the
  pass that changed them ran no NPU work. Treat it as the cost and the benefit
  of one summarisation, which is what it measured.

  The extra ~4s is paid only when eviction was going to happen anyway. If the
  summarisation call fails, or the note itself will not fit, the server falls
  back to plain eviction -- a summary is never allowed to break a request.
  `GENIE_SUMMARY_MAX_TOKENS` (default 192) bounds the note. The summariser's
  INPUT is bounded too: the transcript offered to it is capped at
  `min(6000, (n_ctx // 2) * 3)` characters (so unchanged at 4096 and above),
  and the prompt actually built is token-counted against `n_ctx` less the
  note's cap and `GENIE_WINDOW_MARGIN`, halving the transcript until it fits.
  If it cannot fit -- a tiny window, a long prior note -- no NPU call is made
  and plain eviction follows.

  The log says which of these happened, one line per eviction beside the
  `dropped N oldest message(s)` line:

  ```
  [genie] context window: summarised N evicted message(s) into a C-char note (T tokens of NPU time)
  [genie] context window: reused the retained C-char note for M evicted message(s) (no NPU call)
  [genie] context window: kept the previous C-char note; M newly evicted message(s) could NOT be summarised into it
  [genie] context window: evicted turns NOT summarised -- the summarisation prompt does not fit n_ctx=N either
  ```

  N in the first line counts only the NEWLY summarised turns.

- **A response that writes nothing while it generates is still abortable --
  all three kinds.** A failed write is the only way a stream learns its reader
  has gone, and three situations write nothing for a whole generation: a tool
  turn (buffered, because a half-emitted `<tool_call>` is worse than a slower
  one), a stream whose orphan gate is still holding -- which under the defaults
  (`GENIE_ORPHAN_HOLD_CHARS=-1`, thinking off) is every plain stream, normally
  for its whole length -- and every non-streaming response. Each is probed
  every 8th chunk: a stream
  writes its API's keep-alive (an SSE comment on the OpenAI side, a real `ping`
  event on the Anthropic side) purely so the write can fail; a non-streaming
  response has no frame to write, so it asks the socket instead (`select` plus
  `MSG_PEEK`, which never blocks and leaves a pipelined next request unread).
  Without it an abandoned turn runs to `max_tokens` holding the single-flight
  NPU against every other caller.

  What "gone" covers: a closed or reset connection; a client that half-closes
  its sending side and waits (no HTTP/1.1 library does that on keep-alive, and
  it is indistinguishable from here); and a client that has not READ a stream
  for `GENIE_SOCKET_TIMEOUT` seconds. A departed client gets no response, its
  generation is aborted, and only ITS generation: the abort is aimed at the
  turn the leaving request is running, so a finished stream's late write
  failure can no longer truncate the next client's answer. One departure is ONE
  native ABORT for that turn: the emitter that noticed and the generator it
  then walks away from both abort, and the second is not sent to Genie again,
  because it would land just as the first was making the query return -- a
  signal at an idle dialog, which might stick and cut the NEXT request short.
  (The watchdog is different on purpose: it re-signals a stall every interval.)
  One that left before its query started costs nothing at all -- no lock wait,
  no prefill -- where it used to cost a full prefill plus a token. That holds
  for every response kind: a stream finds out when its first frame fails to
  write, and a non-streaming response, which has no first frame, asks the
  socket before the generation is created.

- **Streaming reports usage too.** OpenAI streams emit a final chunk with an
  empty `choices` list carrying `usage`, but only when the caller sets
  `stream_options.include_usage` -- clients that do not ask see a
  byte-identical stream to before. Anthropic streams carry `output_tokens` in
  `message_delta` as usual. Both include the summarisation overhead below when
  there was any. Every path counts the RAW generation -- what the model
  produced, not what survived the think/orphan strip -- because the stripped
  tokens cost the same NPU time; the OpenAI tool stream was the last path still
  counting post-strip and no longer does. A stream that FAILED carries no usage
  frame at all (see the error-frame note below).

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

  What drops the record, so the next turn re-prefills: any dialog reset; an
  aborted turn, whatever status Genie returned for it; a throw anywhere after
  the reset; the server's own summarisation call; and **a request that supplied
  stop sequences, unless its generation certainly ran to its token cap.** Genie
  strips the matched text from what it hands back, but the tokens that began
  the match were fed and sit in the KV -- so after a hit the KV holds tokens
  the recorded text does not, and a continuation would resume one step out of
  line. Whether a sequence fired is not observable, so the record goes whenever
  one could have. A prompt that cannot be encoded (a lone surrogate) is refused
  before any engine call and leaves the record intact.

  Two things that silently defeated the byte-exact match after a tool call,
  both fixed 2026-09-16: tool arguments and schemas are now rendered as raw
  UTF-8 (`ensure_ascii=False`, as the template's `tojson` does) and come back
  in responses the same way, so a client echoing non-ASCII arguments echoes
  what the dialog holds; and a call-only assistant turn no longer gets a
  newline before `<tool_call>` when thinking is off -- which, thinking being
  off by default, had broken reuse after EVERY tool call. And one thing that
  defeats it by design: a LEADING system message that changes each turn changes
  the system turn, so nothing after it can match. A per-turn reminder sent
  LATER in the conversation does not -- it renders inline where it sits (see
  the system-message note above), so as long as the client keeps the earlier
  ones in its history the next prompt is still a byte-exact extension.
  (`GenieDialog_save`/`restore` also exist and work -- measured ~75 KB/token on
  disk, ~128 MB at 1711 tokens -- but they are not used: in-memory continuation
  is free and this server serves one conversation at a time.)

- **The sampler ships with NO repetition penalty, and that is what a
  degenerate loop looks like.** Genie's `token-penalty` block is optional and
  every field in it defaults to 0 (`penalize-last-n`, `repetition-penalty`,
  `presence-penalty`, `frequency-penalty` -- established here by running bundles
  with the block present and absent and diffing the output),
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
  `GenieDialog_reset`,
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
  is noise in front of the error naming the env vars to set. A
  `genie_config.json` that is PRESENT but does not parse gets its own line --
  `WARNING: could not parse genie_config.json (<the parser's message, with line
  and column>). Nothing can be read from it, so the poll and sampler checks
  were SKIPPED, not passed` -- where it used to print the false `note: no poll
  key found`. The single-length finding comes from `metadata.json` and is still
  reported. With `GENIE_BUNDLE_DIR` unset, none of these readers falls back to
  a `genie_config.json` or `metadata.json` that happens to sit in the current
  directory.

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
  cannot. The per-request apply is still made (it costs one no-op and starts
  working unchanged if a later QAIRT honours it), and the baseline it restores
  to after a tool turn's temp-0 is the sampler the dialog was CREATED with --
  carrying the per-process seed, not the bundle's on-disk `42` -- so the day
  that call takes effect it cannot re-arm the fixed-seed replay.

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
  as a silent CPU fallback, so it now refuses instead. The socket is then
  BOUND -- not yet listening -- before the load rather than after it, which
  closes the other half of the race. A second instance started while the first
  is still loading used to find the port free, load beside it and race it to
  the bind; it still passes this check, since nothing is accepting yet, but is
  stopped by its own BIND failing (`cannot bind ...`, WinError 10048, or 10013
  when the second binder is the wildcard one), ahead of its load, with a line
  saying the holder may be on ANY `GENIE_HOST`. Turning `allow_reuse_address`
  off was only most of the fix: two sockets with DIFFERENT addresses on one
  port -- `0.0.0.0` and `127.0.0.1` -- still bound happily in either order, so
  a second instance on the default host slipped through both guards during a
  wildcard instance's load. `Server.server_bind` now sets
  `SO_EXCLUSIVEADDRUSE`, which is the only option that makes a port this
  process holds unavailable to every other address on it, so the guard is
  cross-host. (It replaces `SO_REUSEADDR` rather than joining it -- both at
  once fails the bind with `WSAEINVAL` -- and costs nothing on restart, since
  TIME_WAIT applies to accepted connections, not to a listening socket.) The
  listen comes after the load, so an answering port always means a resident
  model (see startup order under [Run](#run)).

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
  legs, single engine, no contention, cool box, **on AC with a SETTLED pack**
  (above ~40% and charge draw under 5 W -- see the precondition note below; bare
  "on AC" is not enough and this line said only that until 2026-08-28):

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

- **The measurement precondition is a SETTLED PACK, not "on AC". Plugging in is
  not enough and the difference is 2x on prefill.** Measured across two sessions
  on this box:

  | pack | condition | CPU pp512 |
  |---|---|---|
  | 13-20% | on AC, charging | **58** |
  | 33% | on AC, charging | 124 |
  | 41.6% | on AC, charging | 113 |
  | 100% | settled, 4.6 W draw | **132** |

  Below roughly 20-25% the system protects the charge and starves compute --
  half the prefill, while plugged in and while `PowerOnline` reads `True`. So
  someone who plugs in at 15%, satisfies a bare "measure on AC" instruction and
  starts, publishes a number that is 2x wrong with nothing to warn them. The
  condition that actually holds is **on AC, above ~40%, charge draw under 5 W**.

  Two things this does NOT explain, so do not reach for the pack when you see
  them. The CPU clock oscillates 40-58 points *under load* on a fully settled
  100% pack, and every leg converges on the same 48.9-56.2% floor band whatever
  charging did. And a low CPU clock during a GPU-bound leg is ordinary idle
  downclocking, not throttling -- the CPU has nothing to do.

  **Decode is largely immune to all of it; prefill is not.** Genie decode
  measured 17.70 t/s charging at 33% and 17.91 settled at 100% -- 1.2% apart --
  while prefill separates backends 2x. Check the pack before quoting a prefill
  figure; a decode figure survives a messier box. Every bench tool here records
  pack, draw and clock alongside its numbers now -- `bench_endpoint` and
  `bench_contention` per measurement, `bench_servers` per row of its JSON,
  `bench.py` per GEMM case -- all through the one sampler
  (`bench_endpoint.box_state`) and under the one key, `clock_pct`, which is the
  `% Processor Performance` counter. So a run's conditions are in its output
  rather than in someone's memory, and the columns are comparable across tools.
  The sampler's row is culture-proof -- its doubles are formatted with the
  invariant culture -- so it reads the same on a comma-decimal Windows (de-DE,
  fr-FR), where `79,2` used to add a field, fail the parse and blind every
  consumer at once, `bench_contention`'s on-battery abort included. That abort
  needed its own reader fixed as well, since the cool gate reads the clock
  counter directly rather than through the shared sampler: both of
  `bench_contention`'s reads are invariant-formatted too, and the abort fires
  under `CurrentCulture='de-DE'` where it could not before. That abort is no
  longer the only thing that catches a run taken on battery, either: it never
  runs at all with `--cool-floor 0`, with a clock that never dipped below the
  floor (the power source is read only when the gate is waiting), or with a
  counter that cannot be read -- so the per-round `power_samples[].on_ac` is
  checked once more at the end of every sweep and a run that was on battery is
  warned about there. Only the gate stops a sweep, so the exit code is unchanged.
  (`bench_servers` used to compute its own clock from `Win32_Processor`
  `CurrentClockSpeed / MaxClockSpeed`; sampled at the same instant the two
  instruments differed by about 7 points, which is why the README's "42-79% of
  base" for the cross-server run is not the same quantity as the 34.8% and
  48.9% figures in this file.)

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
  single-engine gain above. `MULTI_ENGINE.md` reports NPU+GPU concurrency
  costing about a quarter of its win under `poll: true` -- **1.70x against
  1.26x** over the best single engine -- because the OpenCL backend needs host
  cores per token to dispatch its kernels and the busy-wait was taking them.
  **The stronger claim this paragraph used to make, that `poll: true` turns
  concurrency into a 0.78x NET LOSS, was refuted by a controlled re-run**: both
  settings are a gain, and two hot engines are worth running either way. The
  same run had earlier retired memory bandwidth as the contention mechanism on
  the strength of the `poll: true` numbers; with the spin removed, bandwidth
  predicts the result again. So this config line changes how much running two
  engines together is worth
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

  | compiled n_ctx | HTP alloc (MiB) | prefill (median) | decode (median) | cost per doubling |
  |---|---|---|---|---|
  | 4096 | 328 (343,933,440 B) | **1157 t/s** | **18.0 t/s** | -- |
  | 8192 | 617 (646,971,904 B) | **458 t/s** | **8.8 t/s** | 2.05x decode, 2.54x prefill |
  | 16384 | 1195 (1,253,048,832 B) | **176 t/s** | **3.3 t/s** | 2.69x decode, 2.59x prefill |

  (Units fixed 2026-09-16: the 8192 row read "647 MB" -- decimal megabytes --
  between two rows that were MiB under the same "MB" heading. All three are MiB
  now, with the bytes beside them, as in `IMPLEMENTATION_PLAN.md`.)

  So **decode is roughly inverse-linear in the window up to 8192 and worse
  beyond it**: the first doubling costs 2.05x (almost exactly the 2x a fixed
  per-token tax predicts), the second 2.69x. Prefill is consistently worse than
  inverse-linear, ~2.55x per doubling. HTP allocation is exactly linear at
  73,984 bytes per token of window (909,115,392 B over the 12,288 positions
  between the 4096 and 16384 bundles; this line said 73,983 until 2026-09-16,
  which does not reproduce the next figure), which doubles as a check that a
  bundle is the window it claims -- the 8192 bundle allocated 646,971,904 bytes
  against a 646,971,904 prediction.

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
  the bundles works out to 73,984 B/token, 0.35% off. Any fp16 estimate of
  KV size for this bundle is 2x too high.

- **Throughput is bandwidth-bound.** Decode is ~18 t/s on a quiet box for the
  4096 bundle with `poll: false` (~12 t/s as the bundle ships). The figure
  belongs to the bundle's window and its poll setting, not to the server -- see
  the tables above. It drops
  sharply under memory pressure (the X Elite's 32 GB LPDDR5x is shared by CPU/GPU/NPU),
  so a large resident model elsewhere (e.g. a 26 GB llama-server) will slow it.
- **`finish_reason` is `length` when the generation reached its `max_tokens`
  cap or the context limit** (Anthropic `stop_reason: "max_tokens"`). Genie
  reports SUCCESS at the cap -- a normal sentence-end -- so the server counts
  what came back: the number of token callbacks, with a tokenizer re-count of
  the text as the second opinion when callbacks fall short (a token whose bytes
  are a partial character may not get a callback of its own). Without a
  tokenizer it is callbacks only. Until 2026-09-16 a capped generation reported
  `stop`, and a client could not tell a cut answer from a complete one. On the
  Anthropic side the full mapping is: `tool_use` when the reply carries a call,
  else `max_tokens` at the cap, else `stop_sequence` when the request SUPPLIED
  stop sequences, else `end_turn`. That third one is a heuristic, not a
  detection -- Genie strips the matched text, so whether a sequence fired is
  not observable -- and `stop_sequence` (the field) is always `null`. The cap
  outranks it: a capped generation is `max_tokens` whatever stop list it
  carried. None of these is ever a turn the server cut short: a stall the
  watchdog aborted and a turn shutdown aborted are errors, not finishes (see
  the streaming-failure note below), so `stop` / `end_turn` / `stop_sequence`
  can no longer be a stall.

  **How far the cap detection is verified:** only against device-free stubs.
  How many token callbacks real Genie makes per token, and the exact count it
  reaches at the cap, have not been measured on the NPU, so `length` at the
  cap is the stubbed contract rather than an observed one -- see the README's
  untested list.

- **The output cap: both spellings, resolved once, applied every turn.**
  `max_tokens` and `max_completion_tokens` are both honoured. For either one
  `0`, `0.0`, `"0"`, `false` and `null` all mean "not set"; a legacy
  `max_tokens` that is SET wins; one that is absent or 0 defers to the modern
  spelling (`{"max_tokens": 0, "max_completion_tokens": 16}` is 16 -- it used
  to fall through to the default); neither set means `GENIE_MAX_TOKENS`. A
  negative or non-integer value is a 400 naming the field the client sent. The
  value is NOT clamped to the window: one too large fails the fit check and
  gets the overflow 400, which quotes the client's own number back. The cap is
  written to the resident dialog on EVERY turn, because it lives there -- a
  request that set none used to run under whatever the previous request had
  left behind.

- **A streaming failure is an error-shaped frame, never content.** Once the
  200 has gone out a failure cannot be a status, so it is sent the way each API
  spells it, after whatever text was generated before it:

  ```
  OpenAI      data: {"error": {"message": "...", "type": "server_error"}}
              data: [DONE]
  Anthropic   event: error
              data: {"type": "error", "error": {"type": "api_error", "message": "..."}}
              event: message_stop
              data: {"type": "message_stop"}
  ```

  (Each Anthropic event goes out as ONE write, `event:` and `data:` together;
  it used to be two, so a client could be handed half an event.)

  No `finish_reason`, no `stop_reason` / `message_delta`, and no usage frame on
  a failed stream: those say how a turn ENDED, and it did not. This used to be a
  content delta reading `\n[error: ...]` followed by `finish_reason: "stop"`, so
  an agent stored the error string as the model's answer and carried on.
  (`src/genie_smoke.py` still checks for that text, only as a guard against a
  server old enough to send it.) Non-streaming failures are an ordinary 500 in
  the endpoint's envelope.

  A turn the SERVER aborted part-way is one of these failures now, not a
  finish: the watchdog cutting a stalled turn (500 non-streaming) and shutdown
  cutting the turn in flight (503 non-streaming) both end a stream on the
  frames above, after whatever text had already gone out. Both used to end on
  `finish_reason: "stop"` / `stop_reason: "end_turn"` over a fragment. A
  client that disconnects is not sent anything, as before.

- **Malformed requests are refused at the door, in the envelope of the API they
  were sent to** -- `{"error": {...}}` on the OpenAI leg, `{"type": "error",
  "error": {...}}` on `/v1/messages` -- and before anything that costs NPU time
  (building the prompt can evict, and evicting can summarise). The 400s:
  `messages` missing or not a list of objects; a message `content` that is
  neither a string, a list of content blocks nor a single content block
  (`message content must be a string, a list of content blocks, or one content
  block -- not int`), which used to flatten to nothing and be answered 200 over
  a turn with no words in it; `tools` that is not a list of
  objects (`tools must be a list of objects` -- a string used to be rendered as
  one "function signature" per character, with a 200); a request with `tools`
  against a bundle whose tokenizer has no `<tool_call>`; a bad output cap; a
  prompt that does not fit even after eviction; text that is not valid Unicode,
  e.g. a lone surrogate `\ud83d` from a string cut through an emoji (`...not
  valid Unicode...`); and any wrong-typed field that raises before generation
  -- `temperature: "hot"`, `stop: 5`, `stream_options: "yes"` -- as `malformed
  request (ExceptionType: detail)`. Body-level refusals come first: bad or
  negative `Content-Length` (400), a chunked body (411), a body over
  `GENIE_MAX_BODY_BYTES` (413), JSON that does not parse or is not an object
  (400). Any other unhandled exception is a 500 (`server_error` / `api_error`)
  rather than a dropped connection; once headers are out, the connection is
  closed and the reason logged. The exceptions are **503**s in the same
  envelope: a turn refused because shutdown had begun, or cut short by it
  (see Shutdown above), and a generation request refused because `/health`
  says the engine is `stalled` or `wedged` (see `/health` under Endpoints).

- **The log is silent on success and never silent on a refusal.** There is no
  per-request access line (an agent makes hundreds). What IS printed is one
  stdout line for every request not served as asked: `[genie] <code> <METHOD>
  <path>: <message>` for every non-2xx the server writes (400 / 404 / 411 / 413
  / 429 / 503 / 529 / 500, and the stdlib's own refusals such as 501), plus
  `engine failure mid-stream`, `dropped ... stopped sending its N-byte body`
  and `failed mid-response`. The 503 there is a generation refused, not
  `/health`: a turn the closing engine would not start or cut short, or one
  refused because the engine is `stalled` or `wedged`. When shutdown refuses
  a STREAM whose 200 is already out, the line reads `refused mid-stream`
  rather than `engine failure mid-stream` -- nothing about the engine failed,
  it declined -- and a stream the server cut part-way reads `aborted
  mid-stream (shutdown)` or `aborted mid-stream (watchdog)`. One line, ASCII,
  message capped at 300 characters. A `/health` 503
  is deliberately NOT logged per poll -- it is the answer, not a refusal, and
  the watchdog announces the state change once.

- **Generated text is decoded incrementally.** A multibyte character split
  across two Genie callbacks is emitted whole rather than as a pair of U+FFFD,
  in the stream and in the KV record alike; a partial sequence left dangling at
  the end is flushed as a single U+FFFD.
- Model swaps: `run-genie-server.ps1 -Model qwen3-8b` serves the Qwen3-8B
  w4a16 prebuilt (multi-length 4096, id `qwen3-8b-npu`; expect roughly half
  the 4B's decode -- bandwidth-bound, ~2x the weight bytes per token), and
  `-Model qwen3-8b-8192` the self-exported 8192 multi-length build of the same
  model (id `qwen3-8b-8192-npu`; the 4096 prebuilt stays the default 8B only
  until a same-harness AC comparison exists -- see `MODEL_OPTIONS.md`). An
  explicit `-Model` beats a `GENIE_BUNDLE_DIR` / `GENIE_MODEL_ID` lingering in
  the shell, and moves the id with the bundle. For
  anything else, point `GENIE_BUNDLE_DIR` at the bundle -- e.g. the 1.7B for
  lower latency. Note that the bundle's **compiled window** is as big a
  latency lever as its parameter count -- see the window-tax note above
  before assuming a larger-context bundle is strictly better. Qwen3.5-9B
  deliberately has NO entry here: it cannot be a Genie bundle today (no
  upstream export; different architecture) and serves through
  `run-llama-server.ps1` instead -- the whole model matrix, and a
  box-state failure mode that can make any Genie bundle crawl at ~0.3 t/s
  under `poll: false`, is in `MODEL_OPTIONS.md`.

## Reproducing the cross-server comparison

Two tools, with different reach. `src/bench_servers.py` measures decode for
this server against **`geniex serve` only**, on the *same* bundle: its method
is a delta between two capped requests, and GenieAPIService honours no output
cap under either spelling, so the delta cannot be formed against it at all.
`src/probe_server_semantics.py` is the one that takes ANY base URL, so it is
the tool to point at GenieAPIService -- or at anything else OpenAI-shaped --
for the seed-replay, overflow and stop-sequence probes. Both default to a
`geniex` model id that only exists once you
have imported that bundle, so the setup is written down here rather than left
in someone's shell history.

The findings these produced are in the README; this is only how to re-run them,
which is worth doing whenever either vendor ships a release.

**`bench_servers.py` -- what a reader will hit.** It is a hardware tool: it
needs the NPU, `pip install tokenizers`, and BOTH `GENIE_BUNDLE_DIR` and
`GENIE_SDK_DIR` set -- it refuses at startup otherwise, where an unset SDK dir
used to cost three 240-second port waits before saying anything. (`GENIEX_EXE`
overrides where it looks for `geniex.exe`; the default is `%LOCALAPPDATA%\GenieX
CLI\geniex.exe`.) A `geniex.exe` that is neither an existing file nor on PATH
is refused there too, BEFORE the first launch: that is the whole `geniex` arm,
so every launch of it would fail the same way and the sweep would run
one-armed to its end and finish `complete`. So is a `--tokens` below
`GENIE_MIN_DECODE_STEPS`, for the same reason one depth further down -- every
measurement would be refused as too short a window to be a rate, after the box
had paid for it.

- **It starts and stops its OWN servers and touches nothing else.** A server
  already listening on `--ours-port` (default **8123**, which is also the
  launcher's -- i.e. your resident `genie_server`) or `--geniex-port` (default
  18181) is REFUSED with its pid and port named, never stopped. Stop it
  yourself, or pass another port. It used to `taskkill /IM geniex.exe` and
  force-stop whatever owned both ports before pass 1, which killed a
  co-tenant's normally-launched server mid-session with no notice on either
  side. Before pass 1 and before every start it also lists the box's
  processes (`Win32_Process`, read-only). A `genie_server` (a Python process
  whose script is `genie_server.py`, run directly or under `-m pdb` / `-m
  cProfile`), a `geniex` or a `GenieAPIService` that this run did not start,
  on ANY port, is refused by pid and command line: `NOT starting: N other NPU
  server(s) running on this box that this run did not start: ... Stop them
  yourself, or pass --allow-other-npu-servers to measure beside them
  deliberately`. Found mid-sweep, it ends the run as `outcome: "refused:
  ..."` with the rows so far written (exit 1). `--allow-other-npu-servers`
  goes on beside them, prints a `WARNING: running beside N other NPU
  server(s)` block naming them, and the results file records them. A listing
  that cannot be taken (off-Windows, PowerShell failed) prints a note --
  `(could not list this box's processes, so NPU servers on ports other than
  the arms' were NOT looked for; the arm ports still are)` -- and is not a
  refusal.
- Each arm's stdout and stderr go to `bench_servers-<arm>.log` beside `--out`.
  A child that exits before its port answers fails that arm-run at once, with
  its exit code and the log's last lines. For `genie_server` that covers the
  model load as well -- it does not listen until the model is resident, so a
  bad bundle or a held HTP exits before the port has answered, and a load that
  HANGS costs the 240 s port wait and no more. In general, though, an answering
  port is a TCP connect and not a loaded model: `geniex` binds at once and
  loads on its first request. So a child that exits AFTER its port answered is
  looked for too, at the first request that fails. Gone by the warmup, it is
  recorded in `failed_starts` with its exit code and log tail, and its depths
  are skipped; gone later, the arm-run stops there and is recorded in
  `died_mid_run` with the depth it died at.
- Log tails in those failure lines are printed as ASCII, anything else
  backslash-escaped (`\u26a0`, `\x97`) -- geniex writes UTF-8 with emoji in
  exactly its failure messages -- and stdout escapes any character it cannot
  encode, so a piped or `| Tee-Object` run can no longer end on a
  `UnicodeEncodeError` over a line whose only job was to say why an arm-run was
  skipped.
- A listener this run did not start is found by asking who is LISTENING on the
  port. A foreign `genie_server` that is still loading holds its port bound but
  not listening, so it cannot be named; the arm this run then launches exits 1
  at its own bind (`cannot bind ... 10048`, or 10013 when the arm is the one
  binding the wildcard address second), and that is reported as a failed start
  with the log tail. That backstop is now cross-host: the bind is exclusive, so
  a foreign mid-load instance on `0.0.0.0` can no longer let this run's
  `127.0.0.1` arm bind beside it and measure against the wrong server.
- `--out` (default `sweep-results.json`, in the current directory; `''` for
  none) is NOT overwritten without `--force`, and an existing file, a
  directory (`NOT starting: --out X is a directory -- it names the results
  FILE.`, `--force` or not) or a missing directory is refused BEFORE the sweep
  rather than after twenty minutes of it. The path is looked at again when the
  record is written: a file that appeared there during the sweep, or replaced
  the one `--force` was given, is left alone -- `--force` or not -- and the
  record goes to a timestamped file beside it (`sweep-results.<UTC
  stamp>.json`, i.e. `<out-root>.<YYYYMMDDTHHMMSSZ><ext>`), as it does when
  writing `--out` fails for any other reason (the directory gone, the volume
  full). The closing lines say where it went (one sentence, then `wrote
  <actual path>`), and the run exits 1 whatever `outcome` says. If the
  fallback write fails too, the output says the numbers exist only in the
  terminal. The repo's `.gitignore` covers the default name, the timestamped
  fallback and the logs.
- Every arm gets the cap under BOTH spellings (`max_tokens` and
  `max_completion_tokens`), `cache_prompt: false`, thinking off
  (`chat_template_kwargs.enable_thinking=false`, `reasoning_effort: "none"`)
  and `stream: false` -- the same body `bench_endpoint` sends.
- Depths are resolved against the bundle's window (`dialog.context.size` from
  its `genie_config.json`, or `--n-ctx`, else an assumed 4096) less `--tokens`
  less 256: an over-budget depth is dropped with a printed note, a non-integer
  is a one-line refusal. The graph boundaries INSIDE the window are still yours
  to respect -- keep depth + `--tokens` inside one compiled length.
- A sample is kept only if the long run produced at least 90% of `--tokens`
  AND at least `GENIE_MIN_DECODE_STEPS` (16) steps.
- `--repeat` and `--depth` are accepted as aliases of `--passes` and
  `--depths`, because the other bench CLIs spell them that way. Progress reads
  `[run R/T, pass P/N]`: a pass is one A/B pair, a run is one arm's turn in it.
- **Exit status is 0 only when the sweep completed** and its record is at
  `--out`. Interrupted, refused (another NPU server included) or errored is 1,
  and so is a record that went to the timestamped fallback -- and a Ctrl-C
  or a bug mid-run still stops the servers this run started and still writes
  the rows gathered so far, with `outcome` saying
  which it was. That includes a Ctrl-C during the port wait: the child is
  registered the moment it exists, before the wait, so it is stopped rather
  than left holding the Hexagon and its port. `outcome` is about the LOOP: it
  is still `complete`, and the exit code still 0, when an arm-run failed to
  start or died mid-run, as long as that arm still produced rows, because the
  sweep went on without it. Those are on the screen as they happen, in
  `failed_starts` / `died_mid_run`, and in a closing line printed after the
  medians -- `NOT every arm-run ran to its end: of N, X failed to start (arm)
  and Y died mid-run (arm) ...` -- so a table with rows missing cannot pass for
  a finished A/B.

  A loop that ran to its end and left an ARM with no rows AT ALL is not a
  comparison whatever the loop did, so that one is `outcome: "incomplete: no
  rows for <arms>"` and exits 1. It is said after the medians, where the
  numbers are read: ``NOT an A/B: `geniex` produced no rows at all, so there is
  nothing to compare -- this run exits non-zero.`` That covers an arm every
  launch of which failed, an arm that died in every warmup, AND an arm every
  sample of which was REFUSED -- which the `NOT every arm-run ran to its end`
  line cannot see, because no arm-run ended early.

The results file says which sweep it was. Top level: `tool`, `outcome`,
`started`, `finished`, `bundle_dir`, `n_ctx`, `n_ctx_source`, `arms`, `passes`,
`order`, `depths`, `tokens`, `timeout`, `acceptance`, `cap_spellings`,
`request_settings`, `depth_means`, `token_counter`, `clock_instrument`, `logs`,
`failed_starts`, `died_mid_run`, `other_npu_servers`, `rows`. `failed_starts`
is a list of `{arm, pass, run, reason}` and also holds an arm whose server
died during the warmup after its port had answered; `died_mid_run` is a list
of `{arm, pass, run, depth, reason}`. `other_npu_servers` is `{allowed, seen,
unscanned_runs}`: `seen` lists `{pid, ppid, name, cmdline, kind, run}` for
every server the run went on beside (run 0 = the startup check), and
`unscanned_runs` lists the runs whose process listing could not be taken --
"none seen" and "not looked" are different facts about a number. Each row: `arm`, `pass`, `run`, `depth`, `rate`,
`steps`, `secs`, `prompt_tokens`, `on_ac`, `charge_pct`, `charge_w`,
`clock_pct`. The old row key `clock` (the `Win32_Processor` ratio, `-1` on a
failed read) is gone; `clock_pct` is the performance counter and is `null`
when it could not be read.

**`probe_server_semantics.py`** takes three positionals, `[BASE] [MODEL]
[CAP]`, defaulting to geniex's (`http://127.0.0.1:18181`,
`qualcomm/qwen3-4b-ours`, `max_completion_tokens`). For this server:

```powershell
python src\probe_server_semantics.py http://127.0.0.1:8123 qwen3-4b-npu max_tokens
```

`-h` / `--help` / `/?` print the usage and exit 0 without `GENIE_BUNDLE_DIR`.
A BASE that is not an http(s) URL (`127.0.0.1:8123`, `localhost:8123`), a
fourth positional and an unknown option are refused by name, exit 1, before
anything is loaded. Before PROBE 1 it GETs `BASE/v1/models` once. If nothing
takes that connection, it exits 1 with `cannot reach <BASE>/v1/models (...)
-- nothing was probed`, plus the 18181 / 8123 hint when the connect was
refused, instead of eight refused rows over ~20 s and exit 0; any answer, an
HTTP error included, is enough to go on. The first line of output names the
base, model and cap key (`probing <BASE> -- model <MODEL>, cap sent as
<CAP>`), and a completed run exits 0.

`GENIE_BUNDLE_DIR` must be the bundle the server under test is serving, and
must hold `genie_config.json` as well as `tokenizer.json`: PROBE 2's three
prompt sizes are derived from `dialog.context.size` (0.85x / 1.1x / 2.5x of it
-- 6963 / 9011 / 20480 at 8192) rather than assuming 8192, so re-running
against the 4096 8B prebuilt keeps an in-window control row instead of putting
all three past the window. A missing variable, tokenizer, config or key is a
named exit, not a traceback; blank completions in PROBE 1 read `EMPTY ... no
seed verdict` rather than REPLAYS, and a failed run in PROBE 3 reads `NO
VERDICT`.

**geniex serve -- one command.** It accepts an AI Hub bundle directory
(`metadata.json` + `part*.bin`) directly, and COPIES it into its own cache
(~3 GB), so later edits to `poll` or the window must be made on the copy at
`%LOCALAPPDATA%\..\.cache\geniex\models\qualcomm\<name>`, not on the source bundle:

```powershell
geniex pull qwen3-4b-ours --model-hub localfs --model-type llm `
  --local-path $env:GENIE_BUNDLE_DIR
geniex serve --host 127.0.0.1:18181
# the served id is namespaced: qualcomm/qwen3-4b-ours
```

**GenieAPIService -- four things, none of them optional.** From
[qualcomm/qai-appbuilder](https://github.com/qualcomm/qai-appbuilder):

1. **Take the v2.3.7 asset, not `GenieAPIService_Stable`.** Stable's model
   detector tries QNN, MNN and GGUF against a valid Genie config and rejects it
   with no reason given; v2.3.7 lets you declare the backend instead of
   guessing. (The newest Windows-ARM64 asset, v2.48.40, downloaded corrupt --
   truncated at 38 MiB with no central directory, identically over three
   fetches. Check it before assuming it is fixed.)
2. **Graft the local QAIRT 2.45 runtime over the bundled 2.44.** Bundles
   compiled by 2.45 against a 2.44 runtime is the unsupported direction. Eight
   files, from `$env:GENIE_SDK_DIR`: `Genie.dll`, `QnnHtp.dll`,
   `QnnHtpNetRunExtensions.dll`, `QnnHtpPrepare.dll`, `QnnHtpV73Stub.dll` and
   `QnnSystem.dll` from `lib/aarch64-windows-msvc`, plus `libQnnHtpV73Skel.so`
   and `libqnnhtpv73.cat` from `lib/hexagon-v73/unsigned`. The service imports
   29 Genie symbols and 2.45 exports all of them, so the swap is ABI-clean.
   Back the originals up first; two filenames differ only in case.
3. **Put the model files BESIDE `config.json`,** not wherever the config
   points. This is the step that is not documented anywhere and the one that
   makes the difference between loading and the bare "Load Model Failed":
   hardlink `part*_of_4.bin`, `tokenizer.json`, `htp_backend_ext_config.json`
   and friends into `config/<name>/`, then use bare filenames for `ctx-bins`,
   `tokenizer.path` and `extensions`. Hardlinks cost no disk. The bundle's own
   `config.json` collides with the service's, so rename it.
4. **Declare the model in `service_config.json`** -- this is what bypasses the
   detector:

   ```json
   {"name": "qwen3-4b-ours", "path": "qwen3-4b-ours", "backend": "qnn",
    "device": "npu", "context_size": 8192, "enabled": true}
   ```

   Then `GenieAPIService.exe -c config\<name>\config.json -p 8910 -l`.

A sanity check worth running before believing any failure is yours: the
`genie-t2t-run.exe` shipped in the same package, with the same grafted DLLs,
should generate from the bundle in one shot. If it does and the service does
not, the graft is fine and the service's config is the problem -- which is
exactly how the beside-the-config requirement was found.

**Keep only one server resident.** The HTP is single-flight; two loaded engines
contend, and a benchmark taken that way measures the contention.
