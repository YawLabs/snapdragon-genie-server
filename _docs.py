import io
import sys

edits = 0


def rep(path, old, new):
    global edits
    s = io.open(path, encoding="utf-8").read()
    if s.count(old) != 1:
        sys.exit("FAIL %s: %d matches for %r" % (path, s.count(old), old[:90]))
    io.open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    edits += 1
    print("  ok: %s" % path)


R = "README.md"
G = "docs/GENIE_SERVER.md"
T = "docs/TYPED_ROUTER_BRIEF.md"
M = "docs/MULTI_ENGINE.md"

# --- README: test count, and the launcher no longer needs editing ----------
rep(R, '''78 tests, ~4s, and **none of them need the NPU, a Genie bundle, or the QAIRT
SDK** -- they drive the handlers with a fake socket and a stub engine, so they
run anywhere.''',
'''Lint with the same config CI would have used, if there were CI:

```powershell
python -m ruff check src tests
```

203 tests, ~2s, and **none of them need the NPU, a Genie bundle, or the QAIRT
SDK** -- they drive the handlers with a fake socket and a stub engine, so they
run anywhere.

That device-free property is load-bearing rather than incidental, and it has a
cost worth stating: the ctypes bindings and every Genie call are NOT covered.
A regression there is invisible until the server is actually started, so
starting it remains part of checking a change that touches the engine.''')

rep(R, '''src/run-genie-server.ps1  launcher; edit the bundle/SDK paths at the top''',
'''src/run-genie-server.ps1  launcher + supervisor; finds the bundle/SDK itself''')

# --- GENIE_SERVER: /health is no longer a liveness ping --------------------
rep(G, '''- `GET /health` -- liveness.''',
'''- `GET /health` -- **engine** state, not process liveness. `200` when the
  server can actually generate; `503` with a `state` of `failing`, `stalled` or
  `wedged` when it cannot. The body carries `detail` (why), `generating`,
  `tokens_in_flight`, `generations` and `consecutive_failures`.

  The distinction is the entire point: a wedged HTP leaves this process
  perfectly able to accept a connection and answer this endpoint while unable
  to serve a single token, so a plain liveness ping reports healthy for as long
  as the outage lasts. The handler touches nothing on the engine, which is what
  lets it answer *during* a wedge -- the engine lock is exactly what the stuck
  thread is holding.''')

# --- GENIE_SERVER: the env the supervisor reads ----------------------------
rep(G, '''| `GENIE_MODEL_ID` | qwen3-4b-npu | id reported to clients |''',
'''| `GENIE_MODEL_ID` | qwen3-4b-npu | id reported to clients |
| `GENIE_NPU_ROOT` | `../genie-npu` beside this repo | where `bundles/` and `qairt/` live. Set this instead of the two paths above; the newest `qairt/*` is picked automatically. |
| `GENIE_FIRST_TOKEN_TIMEOUT` | 300 | seconds a generation may run before its first token before being called stalled. Generous because prefill at depth legitimately takes tens of seconds. |
| `GENIE_STALL_TIMEOUT` | 120 | seconds between tokens before being called stalled. This is the real wedge signal -- see Supervision. |
| `GENIE_WEDGE_GRACE` | 60 | seconds an abort gets to take effect before the stall is escalated to a wedge |
| `GENIE_FAIL_THRESHOLD` | 3 | consecutive failed generations before `/health` reports `failing` |
| `GENIE_WEDGE_EXIT` | 1 | `0` keeps the process up on a wedge (it stays 503) instead of exiting for a supervisor |
| `GENIE_MAX_RESTARTS` | 5 | launcher only: rapid restarts before it gives up |
| `GENIE_RESTART_COOLDOWN` | 25 | launcher only: seconds between restarts. Not arbitrary -- a force-killed server needs roughly 20s of settling, and restarting sooner was measured costing about half of decode throughput. |''')

# --- GENIE_SERVER: the supervision section ---------------------------------
rep(G, '''## Notes / limitations''',
'''## Supervision: what happens when the HTP wedges

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

## Notes / limitations''')

# --- router brief: /health is now a failover signal ------------------------
rep(T, '''| `GET /health` | liveness only -- see the caveat below |''',
'''| `GET /health` | **engine** state. 200 = can generate; 503 + `state` (`failing` / `stalled` / `wedged`) + `detail` = cannot. Answers during a wedge, so it is usable as a failover signal. |''')

rep(T, '''4. Health checks that survive the HTP wedge -- `/health` answering is not
   proof the device will execute, since `1003` fails at execute time, not at
   load.''',
'''4. ~~Health checks that survive the HTP wedge~~ -- **done 2026-08-24.**
   `/health` used to be a liveness ping, and the objection here was right: it
   answering was not proof the device would execute, since `1003` fails at
   execute time rather than at load. It now reports engine state and returns
   503 when the engine cannot serve, deliberately touching nothing on the
   engine so it still answers while a wedged thread holds the lock. A stall is
   aborted; if that does not take, the process exits 75 and the launcher
   restarts it. A dispatcher can treat a 503 from this endpoint as "shed to
   another engine" and a 200 as a real capability claim.''')

print("%d edits" % edits)
