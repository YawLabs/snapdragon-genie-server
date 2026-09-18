#!/usr/bin/env python3
"""Interleaved decode-rate A/B between two OpenAI-compatible servers.

`bench_endpoint.py` measures ONE server and trusts its `usage` block. This
measures TWO against each other and trusts neither's: a usage block is that
server's own convention (which tokens it counts, whether reasoning that
arrived in a side channel is included), and putting each arm on its own
convention would be two instruments rather than one measurement. So tokens
are counted locally from the returned text, with the bundle's own
tokenizer.json, identically for every arm. GenieAPIService, which reports
usage as all zeros, is the extreme case of the same problem -- but it cannot
be an arm here at all (see below). The two arms this tool launches are
genie_server.py and `geniex serve`, and geniex does report usage; the local
count is about having ONE instrument, not about a missing one.

Three things here are not incidental:

* **Interleaved passes.** A/B/B/A/A/B rather than all-A-then-all-B. The
  Hexagon is single-flight, so an arm's server must be stopped before the
  other starts, and a box that drifts over a twenty-minute run would otherwise
  hand the whole drift to whichever arm ran second. Interleaved, drift lands on
  both and shows up as spread instead of as a difference.

* **Decode as a two-request delta.** Same prompt at max_tokens=1 and
  max_tokens=1+N, subtracted. Prefill is identical in both and cancels, taking
  per-request HTTP overhead with it -- which is the only reason two different
  HTTP stacks are comparable. A sample is DISCARDED unless the long run
  produced at least 90% of the N extra steps it asked for (an early stop makes
  the subtraction meaningless, and averaging it in is how a wrong number gets
  published) AND at least bench_endpoint's MIN_DECODE_STEPS floor (16 unless
  GENIE_MIN_DECODE_STEPS says otherwise), below which the residue of overhead
  that does not cancel is divided by a handful of tokens and comes out shaped
  like a rate. bench_endpoint applies only the absolute floor; this tool
  applies both, because --tokens here is the window it was promised and a
  window well short of it is a different generation from the one requested.

* **Boundary-safe depths.** Genie picks the smallest compiled graph that fits
  at prefill time, so a depth where prompt + generated straddles a boundary
  runs the two calls on DIFFERENT graphs and the subtraction stops cancelling.
  The resulting noise reads convincingly as thermal decay. Keep
  depth + tokens inside one compiled length. Depths are resolved with
  bench_endpoint.resolve_depths against the bundle's window (genie_config.json
  `dialog.context.size`, or --n-ctx), so a depth past the budget is dropped
  with a note and a non-integer is a sentence rather than a traceback -- but
  the window is the outer bound only; the graph boundaries inside it are
  still yours to respect.

"Depth" here is the number of tokens in the UNTEMPLATED user message, counted
with the bundle tokenizer (prompt_depth.prompt_at builds it). bench_endpoint's
`depth=` is the server's own prompt_tokens for a TEMPLATED prompt sized by a
chars/4 estimate, so d250 here and depth=250 there are not the same prompt.
When a server reports a non-zero prompt_tokens it is printed beside the
target and stored in the row, so the two can be read against each other.

Not every server can be an arm. One that honours no output cap cannot have the
delta formed against it at all -- measured on GenieAPIService v2.3.7, which
returns 125 tokens for a requested 16 under both `max_tokens` and
`max_completion_tokens`.

Co-tenants. This tool starts and stops its OWN servers and touches nothing
else. Before pass 1, and again before every start, it looks at who owns the
arm ports, and a listener this run did not start -- a resident genie_server
on the launcher's default 8123, a geniex left over from another session --
ends the run with the pid and port named, never with a kill. Stop it yourself, or pick another
--ours-port / --geniex-port. (It used to `taskkill /IM geniex.exe` and
force-stop whatever owned both ports before pass 1, which killed a
co-tenant's normally-launched server mid-session with no notice on either
side.) At the same two moments it lists the box's processes, because the arm
ports are not the only way to share the Hexagon: a genie_server, geniex or
GenieAPIService this run did not start, on ANY port, is refused the same way,
by pid and command line, rather than having an arm's ~3 GB bundle loaded
beside it. --allow-other-npu-servers goes on beside them deliberately, and the
results file then lists them. A listing that cannot be taken is said so, and
recorded, rather than read as "none running".

Each arm's stdout and stderr go to bench_servers-<arm>.log beside --out, and
a child that exits before its port answers fails the pass at once, with its
exit code and the log's last lines, instead of after the 240 s port wait it
used to cost -- three times per run under the default --passes when
GENIE_SDK_DIR was simply unset. An answering port is a TCP connect and not,
in general, a loaded model -- it is one for genie_server, which does not
listen until its model is resident, and it is not for geniex, which binds at
once and loads on its first request -- so a child that exits AFTER it is
looked for too, at the first request that fails: gone by the warmup, it is a
failed start like the other and its depths are skipped; gone later, the
arm-run stops there and is listed under `died_mid_run` with the depth it died
at. Either way the exit code and the log tail are on the screen and in the
file.

The results file (--out, refused if it exists unless --force, and refused
BEFORE the sweep) is looked at again when it is written: a file that appeared
there during the sweep, or replaced the one --force was given, is left alone
-- --force or not -- and so is the record: it goes to a timestamped file
beside --out, as it does when writing --out fails for any reason, the closing
lines say where it went, and the run exits non-zero whatever `outcome` says.
It says which sweep it was: the arms, the bundle, the window,
both timestamps, the passes, the acceptance rule, every arm-run that failed
to start or died mid-run and why, the other NPU servers it ran beside, and
`outcome` -- "complete", "interrupted",
"refused: ...", "error: ..." or "incomplete: no rows for <arms>". `outcome`
is how the LOOP ended, with one exception: a loop that ran to its end and
left an ARM empty is "incomplete: no rows for ...", because that is not a
comparison whatever the loop did. "complete" with an entry in
`failed_starts` or `died_mid_run` is a sweep that ran to its end without
those arm-runs while both arms were still measured, and the closing lines say
so. A Ctrl-C or a bug mid-run still stops the arm this run started -- from
the moment it is launched, port wait included -- and still writes the rows
gathered so far; anything but "complete" exits non-zero.

Needs the NPU, a bundle, GENIE_SDK_DIR (genie_server.py, the `ours` arm,
exits at load without it), a geniex.exe (GENIEX_EXE, the `geniex` arm) and
`pip install tokenizers`. The first three, the output path, --tokens against
the decode floor, the depths, the arm ports and the other NPU servers on the
box are all checked BEFORE the first launch, because the cheapest refusal is
the one that has not yet loaded a 3 GB bundle onto the HTP. Unlike tests/,
this is a hardware tool.

The --geniex-model default names a model that must be IMPORTED first; standing
both servers up on one bundle is four steps for one of them and one command for
the other, all written down in "Reproducing the cross-server comparison" in
docs/GENIE_SERVER.md. Do that before wondering why an arm will not start.
"""
import argparse
import json
import ntpath
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback

# The HTTP plumbing (POST + monotonic timing + a one-line failure reason), the
# depth resolver, the decode-window floor and the box-state sampler are
# bench_endpoint's. Sharing them is the point: the cross-server table in the
# README sits next to bench_endpoint's tables, and every place the two tools
# measured the same quantity with different code was a place the columns
# quietly stopped being comparable (the clock column was the worst -- two
# instruments, three JSON keys).
import bench_endpoint as be

# The prompt builder and the token counter are prompt_depth's, shared with
# probe_server_semantics.py so that "depth N" is one prompt in both tools.
# Tokens are counted LOCALLY for the reason in the module docstring -- one
# instrument for both arms -- and not because an arm lacks a usage block: this
# file's old `pip install tokenizers` message said "one of the servers under
# test reports usage as all zeros", which is GenieAPIService, which cannot be
# an arm here at all.
from prompt_depth import load_tokenizer
from prompt_depth import ntok as _ntok
from prompt_depth import prompt_at as _prompt_at

BUNDLE_DIR = os.environ.get("GENIE_BUNDLE_DIR", "")
SDK_DIR = os.environ.get("GENIE_SDK_DIR", "")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENIEX = os.environ.get(
    "GENIEX_EXE",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "GenieX CLI", "geniex.exe"))

# Fraction of the requested N extra steps a sample must actually have produced.
# See the module docstring: the absolute floor is bench_endpoint's, this is
# the tool's own.
EARLY_STOP_TOLERANCE = 0.9

# Seconds to wait for the port to answer after a launch. genie_server loads a
# ~3 GB bundle onto the HTP first -- it binds its port before that load and
# listens only after it, so this is also the bound on a load that HANGS --
# and geniex copies the bundle into its cache on first use. A child that DIES
# before its port answers is detected long before this (see wait_port); one
# that dies after it, at its first failed request (see death_of).
START_SECS = 240

# After a request fails, how long the child is given to finish exiting before
# it is taken to be alive. A dying server's connections are reset BEFORE its
# process handle reports the exit -- measured here with a child that
# os._exit()s holding a connection: poll() was still None right after the
# reset 40 times of 40, and turned a few milliseconds later -- so an immediate
# poll() reads a dead arm as a live one. Seconds rather than milliseconds
# because a real arm has an HTP session to tear down on its way out.
DEATH_GRACE_SECS = 5

# After a child is gone, how long its port may keep answering before that is
# treated as someone else's listener rather than a socket winding down.
FREE_SECS = 30

# Settle time after a stop, so the next arm does not race the HTP release.
SETTLE_SECS = 4


def env_problem(bundle_dir, sdk_dir):
    """Why this run cannot start, or None. Checked before anything is spent.

    Both variables are prerequisites of the `ours` arm: genie_server.py
    sys.exits at load without GENIE_SDK_DIR. This tool read SDK_DIR from the
    environment and then never looked at it, so the one failure it already
    had in hand surfaced 240 s later, per pass, as a reasonless "FAILED to
    start" -- the child's exit message was going to DEVNULL.
    """
    if not bundle_dir:
        return "set GENIE_BUNDLE_DIR to the bundle both servers will serve"
    if not sdk_dir:
        return ("set GENIE_SDK_DIR (the QAIRT SDK root) -- the `ours` arm is "
                "genie_server.py, which exits at load without it")
    if not os.path.isdir(sdk_dir):
        return "GENIE_SDK_DIR is not a directory: %s" % sdk_dir
    return None


def bundle_n_ctx(bundle_dir):
    """`dialog.context.size` from the bundle's genie_config.json, or None.

    The same key genie_server's read_context_size() serves at /props, read
    from the file rather than from a server because no server is up when the
    depths are resolved -- this tool starts its own, and both arms serve this
    one bundle. --n-ctx overrides it for a bundle whose config is elsewhere.
    """
    try:
        with open(os.path.join(bundle_dir, "genie_config.json"),
                  encoding="utf-8") as f:
            size = int(json.load(f)["dialog"]["context"]["size"])
        return size if size > 0 else None
    except Exception:
        return None


TOK = None


def ntok(text):
    return _ntok(TOK, text)


def prompt_at(depth):
    """A prompt of exactly `depth` tokens, measured rather than estimated."""
    return _prompt_at(TOK, depth)


# --------------------------------------------------------------------------
# Ports and processes. Everything here is scoped to what THIS run started.
# --------------------------------------------------------------------------

def port_open(port):
    """True if something accepts a connection on 127.0.0.1:`port`."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


def port_owner(port):
    """PID of the process listening on `port`, or None if nothing is -- or if
    it cannot be told (off-Windows, PowerShell failed). The caller pairs this
    with port_open, so an owner that cannot be named still cannot be mistaken
    for a free port.

    LISTENING is the filter (-State Listen), and a genie_server that is still
    loading holds its port in state Bound: it binds before its load and
    listens after it. So neither this nor port_open sees a foreign
    genie_server mid-load, and foreign_listener cannot name one. Nothing is
    measured against the wrong server for it: the arm this run then launches
    exits 1 at its own bind -- "cannot bind HOST:PORT: ... 10048", or 10013
    when the arm this run launches is the one binding the WILDCARD address
    second, plus a line saying the holder may be on any GENIE_HOST -- ahead of
    its load, and wait_port reports that with the exit code and the log. The
    behaviour relied on here is unchanged (exit 1 at the bind, reported as a
    failed start with the log tail); it is now enforced across hosts as well,
    since genie_server asks for its port exclusively, so a foreign mid-load
    instance on 0.0.0.0 can no longer let this run's 127.0.0.1 arm bind beside
    it and load a second bundle onto the HTP.
    """
    if sys.platform != "win32":
        return None
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-NetTCPConnection -LocalPort %d -State Listen "
             "-ErrorAction SilentlyContinue | Select-Object -First 1)"
             ".OwningProcess" % port],
            capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def foreign_listener(arms):
    """A refusal naming the pid and port of a listener this run did not
    start, or None when every arm port is free.

    Called only when none of this run's children is up, so ANY listener is
    someone else's. The message says what to do and never what this tool
    will do about it, because it does nothing: the resident server on 8123
    is the launcher's normal state (run-genie-server.ps1), and a benchmark
    that kills it is a benchmark that ends someone else's session.
    """
    for name, arm in arms.items():
        pid = port_owner(arm["port"])
        if pid is not None:
            return ("port %d (the %s arm) is already served by pid %d, which "
                    "this run did not start -- refusing to touch it. Stop it "
                    "yourself, or pass --%s-port to use a free port."
                    % (arm["port"], name, pid, name))
        if port_open(arm["port"]):
            return ("port %d (the %s arm) is answering but its owner could "
                    "not be identified -- refusing to start a second server "
                    "behind it (on Windows both binds succeed and the OLD "
                    "process keeps answering). Free the port, or pass "
                    "--%s-port." % (arm["port"], name, name))
    return None


# The interpreters a genie_server runs under, by image name without ".exe":
# python, python3, python3.12, pythonw. Not py.exe -- the launcher's own
# python child carries the same command line and is what gets listed.
_PYTHON_IMAGE = re.compile(r"^pythonw?[0-9.]*$", re.IGNORECASE)

# Interpreter options that take their value as the NEXT argument, so the value
# is not mistaken for the script: `python -X utf8 src/genie_server.py`.
_PYTHON_OPTS_WITH_VALUE = ("-X", "-W")


def _win_argv(cmdline):
    """`cmdline` split the way a Windows program splits its own, near enough:
    whitespace separates, a double quote groups (an unterminated one runs to
    the end of the line), and a backslash is a path separator, never an
    escape. Not shlex: that takes an apostrophe for a quote as well, and
    C:\\Users\\O'Brien is a path, not the start of a string. The one rule
    left out, a backslash-escaped quote, does not occur in a script path.
    """
    argv, cur, quoted, grouped = [], [], False, False
    for ch in cmdline or "":
        if ch == '"':
            quoted, grouped = not quoted, True
        elif ch in " \t" and not quoted:
            if cur or grouped:
                argv.append("".join(cur))
            cur, grouped = [], False
        else:
            cur.append(ch)
    if cur or grouped:
        argv.append("".join(cur))
    return argv


def _is_genie_server_py(arg):
    return ntpath.basename(arg).lower() == "genie_server.py"


def _runs_genie_server(cmdline):
    """True when a Python command line runs genie_server.py.

    As its script -- the first argument after the interpreter's own options
    -- or through a module that runs a script it is handed, which is how a
    server under `-m pdb` or `-m cProfile -o out.prof` looks: there, any
    later argument naming genie_server.py counts. `-c` runs the code it is
    given, so what follows it is that code's argv, not a script.
    """
    argv = _win_argv(cmdline)[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-c":
            return False
        if a == "-m":
            return any(_is_genie_server_py(x) for x in argv[i + 2:])
        if a in _PYTHON_OPTS_WITH_VALUE:
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            return _is_genie_server_py(a)
    return False


def npu_server_kind(name, cmdline, geniex=None):
    """'genie_server', 'geniex' or 'GenieAPIService' for a process that is one
    of the servers that load a bundle onto the HTP, else None.

    By image name for the two executables, and for genie_server by the SCRIPT
    a Python process runs (see _runs_genie_server) -- not by the words in its
    command line, which matched a shell whose command merely mentioned
    genie_server.py, and a pytest run selecting its tests. `geniex` is
    GENIEX's own path, so a GENIEX_EXE under another file name is still
    recognised as geniex.
    """
    image = ntpath.basename(name or "").lower()
    stem = image[:-4] if image.endswith(".exe") else image
    if stem == "geniex" or (geniex and image == ntpath.basename(geniex).lower()):
        return "geniex"
    if stem == "genieapiservice":
        return "GenieAPIService"
    if _PYTHON_IMAGE.match(stem) and _runs_genie_server(cmdline):
        return "genie_server"
    return None


def list_processes():
    """Every process on the box as {"pid", "ppid", "name", "cmdline"}, or None
    when they cannot be listed (off-Windows, PowerShell failed or timed out).

    Read-only: Win32_Process through Get-CimInstance, which has the command
    line that Get-Process lacks. None rather than [] on a failure, so the
    caller can say it did not look instead of saying there was nothing.
    """
    if sys.platform != "win32":
        return None
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "ConvertTo-Json -Compress -Depth 2 -InputObject @(Get-CimInstance "
             "Win32_Process -ErrorAction Stop | Select-Object ProcessId,"
             "ParentProcessId,Name,CommandLine)"],
            capture_output=True, text=True, errors="replace", timeout=60)
        if r.returncode != 0:
            return None
        rows = json.loads(r.stdout)
        if isinstance(rows, dict):
            rows = [rows]
        return [{"pid": int(p["ProcessId"]),
                 "ppid": int(p.get("ParentProcessId") or 0),
                 "name": p.get("Name") or "",
                 "cmdline": p.get("CommandLine") or ""} for p in rows]
    except Exception:
        return None


def other_npu_servers(procs, own_pids, geniex=None):
    """The NPU servers in `procs` that this run did not start, pid order.

    One is this run's if its pid, or any ancestor's, is in `own_pids` -- this
    process and every child it has launched, so an arm whose tree kill left a
    grandchild behind is not reported as a stranger. Each entry is the
    process dict plus its `kind`.
    """
    parent = {p["pid"]: p["ppid"] for p in procs}

    def ours(pid):
        seen = set()
        while pid and pid not in seen:
            if pid in own_pids:
                return True
            seen.add(pid)
            pid = parent.get(pid)
        return False

    found = []
    for p in procs:
        kind = npu_server_kind(p["name"], p["cmdline"], geniex)
        if kind and not ours(p["pid"]):
            found.append(dict(p, kind=kind))
    return sorted(found, key=lambda p: p["pid"])


def npu_servers_text(found):
    """One indented line per server: pid, kind and a cut-down command line."""
    return "\n".join("  pid %-6d %-15s %s" % (p["pid"], p["kind"],
                                              _ascii((p["cmdline"] or p["name"])[:160]))
                     for p in found)


def npu_refusal(found):
    """The refusal for other NPU servers on the box, naming each by pid."""
    return ("%d other NPU server(s) running on this box that this run did not "
            "start:\n%s\nrefusing to load an arm beside them: each arm puts a "
            "~3 GB bundle on the Hexagon, two loaded engines contend, and "
            "concurrent HTP access can wedge the device for every session "
            "here. Stop them yourself, or pass --allow-other-npu-servers to "
            "measure beside them deliberately (the results file then lists "
            "them)." % (len(found), npu_servers_text(found)))


def _ascii(text):
    """`text` with every non-ASCII character backslash-escaped."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def forgiving_stdout():
    """Make print() unable to raise over a character stdout cannot encode.

    Piped or redirected (`| Tee-Object`, an agent's shell) this process's
    stdout is the ANSI code page with errors="strict", and this tool prints
    text it did not write: the tail of a child's log, a server's own error
    message in a "warmup failed" or SKIP line (geniex's are emoji-prefixed).
    One such character raised UnicodeEncodeError out of a print(), main()'s
    `except Exception` took it for a bug, and a twenty-minute sweep ended
    over a line whose only job was to say why one arm-run was being skipped.
    stderr already escapes what it cannot encode. Best-effort: a stdout that
    cannot be reconfigured (a test capture, an IDE's stream) is left alone.
    """
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except Exception:
        pass


def log_tail(path, lines=12):
    """The last lines of a child's log, ready to append to a failure line.

    "" when there is no log to read; a note when the log is empty, because an
    empty log after a launch failure is itself a fact worth stating (the
    child never got as far as printing).

    The lines come back as ASCII, anything else backslash-escaped, so the
    failure line is printable whatever stdout's encoding is -- it is printed,
    and it is the one line that must not be able to end the run (see
    forgiving_stdout, which covers the prints this does not). geniex writes
    UTF-8 with emoji in exactly its failure messages, and a redirected Python
    child writes the ANSI code page: a byte that is not UTF-8 is shown as the
    byte it was rather than as U+FFFD, which says nothing and was itself
    unencodable on a cp1252 pipe.
    """
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="backslashreplace") as f:
            tail = f.read().splitlines()[-lines:]
    except OSError:
        return ""
    if not tail:
        return " (log %s is empty)" % path
    return ";\n      last %d line(s) of %s:\n      %s" % (
        len(tail), path, "\n      ".join(_ascii(ln) for ln in tail))


def wait_port(port, secs=START_SECS, proc=None, log=None):
    """None once something answers on `port`; otherwise a string saying why.

    The child is polled BEFORE each socket probe, so a server that exits
    before its port answers fails this in one loop turn with its exit code
    and the tail of its log. For genie_server that is every start failure
    there is: a missing SDK, a non-ARM64 python, a port it cannot bind, and
    the load itself -- a bad bundle, GenieDialog_create refusing, the HTP held
    by another session -- because it binds its port before load_engine() and
    listens only once the engine is in hand. A connect to that
    bound-not-listening port fails exactly as one to a closed port does
    (measured on this box: ConnectionRefusedError 10061 after ~2 s, which
    under port_open's 1 s timeout arrives as a TimeoutError -- an OSError
    either way, so port_open says False). Every one of those used to cost the
    full timeout and print nothing, because both streams went to DEVNULL and
    the Popen was dropped on the floor.

    "Answers" is a TCP connect, the one readiness signal both arms give: there
    is no HTTP route the two have in common to ask instead. For genie_server
    it means the model is resident, for the reason above. For geniex it is
    NOT proof that a model is loaded: geniex binds at once and loads on its
    first request, so that arm can still die in its load AFTER this returned
    None. That is main()'s to catch, with death_of(), at the first request
    that fails -- for either arm, since a server can die after a good start
    too -- and it must not be read as "whatever gets past here started".
    """
    end = time.time() + secs
    while True:
        if proc is not None and proc.poll() is not None:
            return ("exited with code %s before port %d answered%s"
                    % (proc.returncode, port, log_tail(log)))
        if port_open(port):
            return None
        if time.time() >= end:
            return ("nothing listening on port %d after %d s (the process is "
                    "still alive)%s" % (port, secs, log_tail(log)))
        time.sleep(2)


def wait_port_free(port, secs=FREE_SECS):
    """True once nothing answers on `port`, False if it still does after `secs`."""
    end = time.time() + secs
    while port_open(port):
        if time.time() >= end:
            return False
        time.sleep(1)
    return True


def death_of(child, grace=DEATH_GRACE_SECS):
    """Why `child` is no longer running, or None while it still is.

    Asked after a request to it FAILED, which is the only moment a child that
    died after its port answered can be told from a request that merely went
    wrong: both print the same ConnectionResetError. The reason carries what
    wait_port's does -- the exit code and the tail of the arm's log, which is
    where "GenieDialog_create failed" actually is. The wait is
    DEATH_GRACE_SECS long for the reason given there; a child that is alive
    costs the whole grace, which is paid only beside a request that has
    already failed.
    """
    proc = child["proc"]
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    if proc.poll() is None:
        return None
    return ("exited with code %s after port %d had answered%s"
            % (proc.returncode, child["port"], log_tail(child["log"])))


def _terminate(proc):
    """End one child this run started, tree and all, and wait for it.

    Never raises: this runs from main()'s `finally`, where an exception would
    replace the results file with a traceback. A child that outlives both the
    tree kill and Popen.kill() is reported and left; stop_all's port check
    then says whether it is still serving.
    """
    if sys.platform == "win32":
        # /T for the tree, scoped to OUR pid -- never by image name.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("   WARNING: pid %d did not exit after a tree kill and a "
                  "kill() -- left running" % proc.pid, flush=True)


def start(name, arm, log_dir, children):
    """Launch one arm, register it in `children`, and wait for its port.

    Returns (child, None) when the port answers, (child, reason) when it does
    not, and (None, reason) when the process could not be launched at all (a
    missing geniex.exe used to traceback out of main with nothing written).

    `children[name]` is set the moment the process exists and BEFORE the port
    wait, because the wait is where this function spends its time -- up to
    START_SECS on a child that is alive and deaf, which is exactly when
    someone reaches for Ctrl-C -- and an exception out of it never reaches the
    `return`. The handle used to be handed back for main() to store, so a
    KeyboardInterrupt in wait_port's sleep unwound to main()'s `finally` with
    `children` still empty: the screen said "stopping the arms", stop_all had
    nothing to stop, and the child kept the Hexagon and its port -- which the
    next run then refused as a listener it did not start. The child is still
    returned, failure or not, for the caller's own use of it.

    Its output goes to bench_servers-<arm>.log in `log_dir`, appended with a
    header per launch, so a failed pass has something to read and a whole run
    can be re-read afterwards.
    """
    if name == "ours":
        cmd = [sys.executable, os.path.join("src", "genie_server.py")]
        kw = {"cwd": REPO, "env": {**os.environ, "GENIE_PORT": str(arm["port"])}}
    else:
        cmd = [GENIEX, "serve", "--host", "127.0.0.1:%d" % arm["port"],
               "--keepalive", "3600"]
        kw = {}
    log_path = os.path.join(log_dir, "bench_servers-%s.log" % name)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n=== %s  %s ===\n"
                  % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
        log.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **kw)
        except OSError as e:
            return None, "could not launch %s: %s" % (cmd[0], e)
        child = {"proc": proc, "log": log_path, "port": arm["port"]}
        children[name] = child
    return child, wait_port(arm["port"], proc=proc, log=log_path)


def stop_all(children):
    """Stop every server THIS RUN started, and nothing else.

    `children` maps arm name to the child start() registered there; it is
    emptied here.
    Single-flight means no two arms may hold the Hexagon, so this runs
    before every start. A port that still answers after its child is gone is
    reported, not killed: at that point it is someone else's listener.
    """
    stopped = False
    for name, child in list(children.items()):
        proc = child["proc"]
        if proc.poll() is None:
            print("   stopping %s (pid %d)" % (name, proc.pid), flush=True)
            _terminate(proc)
            stopped = True
        del children[name]
        if not wait_port_free(child["port"]):
            print("   WARNING: port %d still answers after %s (pid %d) was "
                  "stopped -- something else is listening there"
                  % (child["port"], name, proc.pid), flush=True)
    if stopped:
        time.sleep(SETTLE_SECS)


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

def request_body(model, prompt, cap):
    """The request every arm gets, generation settings identical to
    bench_endpoint.chat's.

    The two tools measured decode under different settings: this one sent
    no thinking field, so an arm inheriting GENIE_THINKING=1 measured
    reasoning while bench_endpoint pinned it off. Now both send the same
    thinking-off spellings (Qwen3's chat_template_kwargs and OpenAI's
    reasoning_effort; a server that knows neither keeps its default), the
    same cache_prompt=false (llama-server would otherwise reuse the prefix
    and break the subtraction) and BOTH cap spellings -- geniex ignores the
    legacy `max_tokens` and honours only `max_completion_tokens`, genie_server
    takes either (a legacy `max_tokens` that is set wins there; the modern
    spelling is consulted only when it is absent or 0), so sending both with
    ONE value is honoured by each whichever it reads. The one key
    bench_endpoint does not send is the explicit `stream: false` this tool
    has always sent; it selects the response shape, not a generation setting.
    tests/test_bench_servers.py holds this body against the one bench_endpoint
    sends, so the two cannot drift apart unnoticed.
    """
    return {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cap,
            "max_completion_tokens": cap,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
            "stream": False}


def chat(arm, prompt, cap, timeout):
    """(wall, tokens, reported) for one completion; raises on a failed request.

    `tokens` is the LOCAL count of everything the server generated: content
    plus any reasoning side channel, which is still decoded work. The text is
    pulled out by bench_endpoint's _content_of rather than by a second copy of
    it, so the two tools cannot disagree about what a server "generated";
    only the counting is this tool's own. `reported` is the server's own
    prompt_tokens when it gave a positive one, else None. Timing and the
    failure reason are bench_endpoint.post_timed's, so the clock is the same
    monotonic one and a failed request names the server's message.
    """
    body, wall = be.post_timed("http://127.0.0.1:%d" % arm["port"],
                               "/v1/chat/completions",
                               request_body(arm["model"], prompt, cap), timeout)
    if body is None:
        raise RuntimeError("request failed: %s" % wall)
    text = be._content_of(body)      # "" for a body that is not completion-shaped
    usage = body.get("usage") if isinstance(body, dict) else None
    reported = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if not isinstance(reported, int) or reported <= 0:
        reported = None
    return wall, ntok(text), reported


def sample(arm, depth, n, timeout):
    """(rate, note, detail) for one delta sample at `depth`.

    rate is None when the sample is refused, and `note` says why in the words
    the pass log prints; `detail` (steps, seconds, the server's prompt_tokens)
    goes into the row so the stored rows carry what the screen showed. The
    acceptance rule is the module docstring's: 90% of N AND bench_endpoint's
    absolute floor.
    """
    p = prompt_at(depth)
    w1, t1, _ = chat(arm, p, 1, timeout)
    w2, t2, reported = chat(arm, p, 1 + n, timeout)
    steps, secs = t2 - t1, w2 - w1
    detail = {"steps": steps, "secs": round(secs, 3), "prompt_tokens": reported}
    if steps < n * EARLY_STOP_TOLERANCE:
        return None, "early stop (%d of %d steps)" % (steps, n), detail
    if steps < be.MIN_DECODE_STEPS:
        return None, ("%d-step window is under the %d-step floor "
                      "(GENIE_MIN_DECODE_STEPS) -- overhead, not decode"
                      % (steps, be.MIN_DECODE_STEPS)), detail
    if secs <= 0:
        return None, "non-positive delta -- prompt caching?", detail
    return steps / secs, "%d/%.2fs" % (steps, secs), detail


def incomplete_runs(failed_starts, died_mid_run, runs):
    """One line naming the arm-runs that did not run to their end, or "".

    The reasons were printed as they happened and are in the results file;
    this is the count, per arm, next to the table it qualifies -- "no data"
    and a small n there are these arm-runs, not a slow arm.
    """
    if not failed_starts and not died_mid_run:
        return ""
    parts = []
    for what, entries in (("failed to start", failed_starts),
                          ("died mid-run", died_mid_run)):
        if entries:
            arms = sorted({e["arm"] for e in entries})
            parts.append("%d %s (%s)" % (len(entries), what, ", ".join(arms)))
    return ("NOT every arm-run ran to its end: of %d, %s -- the reasons are "
            "above and in the results file, and rows missing from the table "
            "are these, not measurements" % (runs, " and ".join(parts)))


def arms_with_no_rows(rows, arms):
    """The requested arms that produced no measurement at all, in run order.

    An A/B needs both sides. A sweep can finish its loop with one side empty
    -- every launch of it refused, every depth of it refused, or it died in
    each of its warmups -- and that is not a comparison, however complete the
    loop was. Separate from incomplete_runs(), which counts arm-runs that
    ended early: an arm can lose one run of three and still be measured, and
    an arm can fail no run at all and still have nothing (every sample
    refused).
    """
    return [name for name in arms
            if not any(r["arm"] == name for r in rows)]


def box_sample():
    """One bench_endpoint.box_state reading as the dict a row stores.

    The keys are bench_endpoint's BOX_SAMPLES keys, so a `clock_pct` here is
    the same instrument (Get-Counter '% Processor Performance') under the
    same name as in that tool's output. This module used to compute its own
    clock from Win32_Processor CurrentClockSpeed/MaxClockSpeed and store it
    as "clock", with -1 for a failed read -- a third instrument under a third
    key, sampled on this box at the same instant as the counter and 7 points
    apart, with a fake value where the other tools store None.
    """
    ac, pct, watts, clock = be.box_state()
    return {"on_ac": ac, "charge_pct": pct, "charge_w": watts, "clock_pct": clock}


def clock_text(box):
    """'clock 88% of base', or the honest 'clock unreadable' -- never -1."""
    clock = box.get("clock_pct")
    if clock is None:
        return "clock unreadable"
    return "clock %.0f%% of base" % clock


# --------------------------------------------------------------------------
# The results file
# --------------------------------------------------------------------------

def out_state(path):
    """What is at `path` now, to tell at write time whether it changed while
    the sweep ran: None when nothing is, else (file id, size, mtime in ns)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def _timestamped_beside(path):
    """`path` with a UTC timestamp in front of its extension.

    Beside the path that was asked for, not in a temp directory: the operator
    is looking at the directory they typed, and this has to turn up in it.
    bench_contention's, as is the whole write-time contract below.
    """
    root, ext = os.path.splitext(path)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return "%s.%s%s" % (root, stamp, ext)


def _dump_json(path, record, mode):
    """The one place the record is written, so the asked-for path and the
    fallback are written identically -- and so a test can make a write fail
    without a read-only directory. `mode` "x" creates or fails, never
    replacing what is there."""
    with open(path, mode, encoding="utf-8") as f:
        json.dump(record, f, indent=1)


def write_results(path, record, at_start):
    """Write the sweep's record, never over a file it was not given and never
    discarding it. (path_written, note), bench_contention._write_record's
    contract: `note` is None only when the record landed on `path`; anything
    else is one sentence for the operator AND a non-zero exit, because a
    wrapper that branches on the exit code must not read a fallback write as
    the file it named.

    `at_start` is out_state(path) from before the sweep. The startup check
    refuses an existing --out without --force, and could not see the two
    outcomes this replaces: a file that APPEARED during the sweep (another
    run writing the same default sweep-results.json) was overwritten without
    a word, and an OSError at the write -- the directory gone, the volume
    full, --out now a directory -- was a traceback with no record written
    anywhere, after twenty minutes of the box. Now a file that is not what
    was there at the start is left alone, --force or not: --force is consent
    to replace the file that was there when the run began, not whatever is
    there when it ends. Where nothing is, the file is created exclusively, so
    one that appears between this check and the open fails the open rather
    than being overwritten. Either failure, and any OSError, writes a
    timestamped file beside `path` instead (itself created exclusively).
    """
    now = out_state(path)
    if now is not None and now != at_start:
        why = ("%s %s during the sweep (another run writing the same path?) and "
               "was NOT overwritten" % (path, "appeared" if at_start is None
                                        else "was replaced"))
    else:
        try:
            _dump_json(path, record, "x" if now is None else "w")
            return path, None
        except FileExistsError:
            why = ("%s appeared during the sweep (another run writing the same "
                   "path?) and was NOT overwritten" % path)
        except OSError as exc:
            why = "%s could not be written (%s)" % (path, exc)
    fallback = _timestamped_beside(path)
    try:
        _dump_json(fallback, record, "x")
    except OSError as exc:
        return None, ("%s, and the fallback %s failed too (%s), so this run's "
                      "numbers exist only in the terminal above"
                      % (why, fallback, exc))
    return fallback, "%s, so this run's results went to %s" % (why, fallback)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Two spellings where another bench CLI uses the other one: bench_endpoint
    # says --repeat and --depths, bench_contention says --repeat and --depth.
    # The docs quote --passes and --depths for this tool, so both stay.
    ap.add_argument("--depths", "--depth", dest="depths", default="250,1500,3000",
                    help="comma-separated prompt depths (untemplated tokens, "
                         "bundle tokenizer); keep depth + --tokens inside ONE "
                         "compiled length. --depth is the same flag.")
    ap.add_argument("--tokens", type=int, default=120,
                    help="N, the extra steps of each delta sample (default "
                         "%(default)s, as in bench_contention; bench_endpoint "
                         "says 200). The README's cross-server table was "
                         "measured at 120, with depths chosen so depth + 120 "
                         "stays inside one compiled length.")
    ap.add_argument("--passes", "--repeat", dest="passes", type=int, default=3,
                    help="A/B pairs; each pass runs BOTH arms once, alternating "
                         "which goes first. --repeat is the same flag.")
    ap.add_argument("--timeout", type=float, default=900,
                    help="per-request timeout, seconds (default %(default)s; "
                         "bench_endpoint and bench_contention say 1800). The "
                         "deepest default prompt here is 3000 tokens against "
                         "bench_endpoint's 12000.")
    ap.add_argument("--n-ctx", type=int, dest="n_ctx",
                    help="the bundle's window, when genie_config.json is not "
                         "beside the tokenizer; otherwise read from there.")
    ap.add_argument("--ours-port", type=int, default=8123,
                    help="port the `ours` arm is started on (default "
                         "%(default)s, which is also run-genie-server.ps1's: a "
                         "server already listening there is refused by pid, "
                         "never stopped -- pass a free port instead)")
    ap.add_argument("--geniex-port", type=int, default=18181,
                    help="port the geniex arm is started on (default "
                         "%(default)s; the same refusal applies)")
    # One model flag per arm (bench_endpoint has one --model, bench_contention
    # --npu-model/--gpu-model): the two servers name the same bundle differently.
    ap.add_argument("--ours-model", default="qwen3-4b-npu",
                    help="model id genie_server.py serves (default %(default)s)")
    ap.add_argument("--geniex-model", default="qualcomm/qwen3-4b-ours",
                    help="model id geniex serves -- namespaced, and only there "
                         "once the bundle has been imported (default %(default)s)")
    ap.add_argument("--out", default="sweep-results.json",
                    help="results JSON (default %(default)s); '' for none. "
                         "Arm logs go beside it.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists. Only the file "
                         "that was there at the start: one that appears or is "
                         "replaced during the sweep is left alone, and the "
                         "results go to a timestamped file beside it")
    ap.add_argument("--allow-other-npu-servers", action="store_true",
                    dest="allow_other_npu_servers",
                    help="run although a genie_server, geniex or "
                         "GenieAPIService this run did not start is already "
                         "running on the box (they are refused by pid "
                         "otherwise): the arms then share the Hexagon with "
                         "them, and the results file lists them")
    return ap


def main():
    global TOK
    a = _parser().parse_args()
    forgiving_stdout()

    # Every refusal that can be known before the first launch is made before
    # it: the env, geniex.exe, the decode floor, the output path, the depths,
    # the ports. A run costs twenty minutes of a shared box and its first
    # minute is a model load.
    problem = env_problem(BUNDLE_DIR, SDK_DIR)
    if problem:
        sys.exit(problem)
    # start() turns a geniex.exe that is not there into a per-launch "FAILED
    # to start -- skipping", which is right for one arm-run of many and wrong
    # as the answer to "is there a geniex arm at all": every launch of it
    # fails the same way, and the sweep used to run the `ours` arm to its end,
    # print "geniex  no data" and finish "complete". isfile OR which, because
    # Popen resolves a bare name on PATH and GENIEX_EXE may be one.
    if not os.path.isfile(GENIEX) and not shutil.which(GENIEX):
        sys.exit("NOT starting: no geniex.exe at %s -- that is the whole "
                 "`geniex` arm, so every launch of it would fail and the "
                 "sweep would spend the box measuring `ours` against nothing. "
                 "Point GENIEX_EXE at the executable (or install GenieX CLI)."
                 % GENIEX)
    if a.tokens < be.MIN_DECODE_STEPS:
        # bench_contention._validate refuses this with the same words. The
        # floor is bench_endpoint's, read from GENIE_MIN_DECODE_STEPS at its
        # import; sample() applies it to every measurement, and a measurement
        # asks for --tokens decode steps and can come back with fewer, never
        # more, so below the floor EVERY depth of EVERY arm-run is refused --
        # after its two requests have been paid for.
        sys.exit("--tokens %d is below the decode floor of %d steps "
                 "(GENIE_MIN_DECODE_STEPS, read by bench_endpoint; default "
                 "16): every measurement would be REFUSED as too short a "
                 "window to be a rate, so the sweep would spend the box and "
                 "report both arms as no data. Raise --tokens or lower the "
                 "variable" % (a.tokens, be.MIN_DECODE_STEPS))
    if a.out and os.path.isdir(a.out):
        # Ahead of the exists check, whose way out is --force: --force cannot
        # turn a directory into a file, and the write would fail at the end.
        sys.exit("NOT starting: --out %s is a directory -- it names the "
                 "results FILE." % a.out)
    if a.out and os.path.exists(a.out) and not a.force:
        sys.exit("NOT starting: %s already exists and would be overwritten. "
                 "Re-run with --force, or pass a different --out." % a.out)
    # What --force (or nothing) agreed to replace, so the write at the end can
    # tell that file from one that appeared or was replaced during the sweep.
    out_at_start = out_state(a.out) if a.out else None
    # The arm logs are opened here at the first launch and the results land
    # here at the end, so a directory that is not there is found out now.
    log_dir = os.path.dirname(os.path.abspath(a.out)) if a.out else os.getcwd()
    if not os.path.isdir(log_dir):
        sys.exit("NOT starting: %s is not a directory -- --out and the arm "
                 "logs are written there." % log_dir)

    TOK = load_tokenizer(BUNDLE_DIR)
    ctx = a.n_ctx or bundle_n_ctx(BUNDLE_DIR)
    # bench_endpoint's budget rule, verbatim: room for the generation plus a
    # margin for what the chat template adds around the untemplated prompt.
    limit = (ctx or 4096) - a.tokens - 256
    if limit < 1:
        sys.exit("--tokens %d leaves no room in an n_ctx=%s window -- lower it"
                 % (a.tokens, ctx))
    try:
        depths, note = be.resolve_depths(a.depths, limit, ctx, a.tokens)
    except ValueError as e:
        sys.exit(str(e))
    if note:
        print(note, flush=True)
    if ctx is None:
        print("  (no genie_config.json in the bundle and no --n-ctx -- the depth "
              "budget assumes a 4096 window)", flush=True)

    arms = {
        "ours": {"port": a.ours_port, "model": a.ours_model},
        "geniex": {"port": a.geniex_port, "model": a.geniex_model},
    }
    problem = foreign_listener(arms)
    if problem:
        sys.exit(problem)

    # The arm ports are not the only way to share the Hexagon. A genie_server,
    # geniex or GenieAPIService another session loaded on ANY other port holds
    # a bundle there too, and foreign_listener, which looks only at this run's
    # two ports, never saw it: the sweep then loaded a second ~3 GB bundle
    # beside it for twenty minutes. So the box's processes are looked at as
    # well, here and before every start. `launched` is this process and every
    # child it has started, whose descendants are this run's own.
    launched = {os.getpid()}
    npu_seen = []      # what --allow-other-npu-servers let the run go on beside
    unscanned = []     # runs whose look could not be taken (0 = at startup)

    def npu_check(run):
        """A refusal naming the other NPU servers, or None."""
        procs = list_processes()
        if procs is None:
            unscanned.append(run)
            print("   (could not list this box's processes, so NPU servers on "
                  "ports other than the arms' were NOT looked for; the arm "
                  "ports still are)", flush=True)
            return None
        found = other_npu_servers(procs, launched, GENIEX)
        if not found:
            return None
        if not a.allow_other_npu_servers:
            return npu_refusal(found)
        print("   WARNING: running beside %d other NPU server(s) "
              "(--allow-other-npu-servers) -- the arms share the Hexagon with:\n%s"
              % (len(found), npu_servers_text(found)), flush=True)
        npu_seen.extend(dict(p, run=run) for p in found)
        return None

    problem = npu_check(0)
    if problem:
        sys.exit("NOT starting: %s" % problem)

    order = []
    for i in range(a.passes):
        order += ["ours", "geniex"] if i % 2 == 0 else ["geniex", "ours"]
    print("interleaved: %s" % " -> ".join(order), flush=True)

    children = {}
    rows = []
    failed_starts = []
    died_mid_run = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    outcome = "complete"
    # try/finally, so neither a Ctrl-C nor a bug mid-run leaves an arm of ours
    # holding the Hexagon. Both are caught rather than let through, because
    # the rows gathered so far cost real box time: they are still summarised
    # and written below, with `outcome` saying how the run ended, and the
    # process then exits non-zero.
    try:
        for idx, name in enumerate(order):
            stop_all(children)
            # `run` counts arm-runs (one per entry of `order`), `pass` counts
            # A/B pairs -- the unit --passes and the README's "three passes
            # each" mean. The two used to be one number, so `--passes 3`
            # printed "[pass 6/6]" and stored pass=1..6.
            run, pas = idx + 1, idx // 2 + 1
            box = box_sample()
            print("\n[run %d/%d, pass %d/%d] %s (%s)"
                  % (run, len(order), pas, a.passes, name, clock_text(box)),
                  flush=True)
            problem = foreign_listener(arms) or npu_check(run)
            if problem:
                print("   %s" % problem, flush=True)
                outcome = "refused: %s" % problem
                break
            child, problem = start(name, arms[name], log_dir, children)
            if child is not None:
                launched.add(child["proc"].pid)
            if problem:
                print("   FAILED to start -- skipping: %s" % problem, flush=True)
                # In the artifact too: `outcome` says how the LOOP ended, so
                # without this a "complete" file with one arm's rows missing
                # gives no reason for the gap.
                failed_starts.append({"arm": name, "pass": pas, "run": run,
                                      "reason": problem})
                continue
            try:
                chat(arms[name], prompt_at(64), 16, a.timeout)   # warmup, discarded
            except Exception as e:
                print("   warmup failed: %s" % str(e)[:120], flush=True)
                # An answering port is not always a loaded model (geniex's is
                # not; see wait_port), so this is where an arm that died in
                # its load shows up -- as a reset connection, which says
                # nothing. A child that is GONE never served a request: that
                # is a failed start, recorded as one, and its depths are not
                # walked (each would print a SKIP
                # with the same non-reason, and the file would say "complete"
                # with this arm's rows missing and failed_starts empty). A
                # child that is still alive is left to the depths as before:
                # a warmup can fail for reasons a sample then states.
                problem = death_of(child)
                if problem:
                    print("   FAILED to start -- skipping: %s" % problem, flush=True)
                    failed_starts.append({"arm": name, "pass": pas, "run": run,
                                          "reason": problem})
                    continue
            for d in depths:
                detail = {}
                problem = None
                try:
                    rate, note, detail = sample(arms[name], d, a.tokens, a.timeout)
                except Exception as e:
                    rate, note = None, "%s %s" % (type(e).__name__, str(e)[:90])
                    problem = death_of(child)
                reported = detail.get("prompt_tokens")
                print("   d%-5d %s (%s)%s"
                      % (d, ("%6.2f t/s" % rate) if rate else "  SKIP  ", note,
                         "  server says prompt=%d" % reported if reported else ""),
                      flush=True)
                if problem:
                    # The same death, later: the arm served some requests and
                    # then went away. Not a failed start, so it has its own
                    # list -- with the depth it died at, because the rows
                    # above it are real and the ones from here on are absent
                    # for THIS reason and not for a slow or refused sample.
                    print("   %s DIED mid-run -- skipping its remaining depths: %s"
                          % (name, problem), flush=True)
                    died_mid_run.append({"arm": name, "pass": pas, "run": run,
                                         "depth": d, "reason": problem})
                    break
                if rate:
                    row = {"arm": name, "pass": pas, "run": run, "depth": d,
                           "rate": round(rate, 3)}
                    row.update(detail)
                    row.update(box_sample())
                    rows.append(row)
    except KeyboardInterrupt:
        outcome = "interrupted"
        print("\ninterrupted -- stopping the arms; the %d sample(s) so far are "
              "still written" % len(rows), flush=True)
    except Exception as e:
        outcome = "error: %s: %s" % (type(e).__name__, e)
        traceback.print_exc()
        print("\nthe run died -- stopping the arms; the %d sample(s) so far are "
              "still written" % len(rows), flush=True)
    finally:
        stop_all(children)
    finished = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n=== medians, full range in brackets ===", flush=True)
    for d in depths:
        out = []
        for name in arms:
            got = [r["rate"] for r in rows if r["arm"] == name and r["depth"] == d]
            out.append("%-7s %6.2f [%.2f-%.2f n=%d]"
                       % (name, statistics.median(got), min(got), max(got),
                          len(got)) if got else "%-7s no data" % name)
        print("d%-5d  %s" % (d, "   ".join(out)), flush=True)
    # Overlapping ranges mean the arms are indistinguishable at that depth.
    # Say so rather than quoting a ratio of two medians as though it were one.
    for d in depths:
        o = [r["rate"] for r in rows if r["arm"] == "ours" and r["depth"] == d]
        g = [r["rate"] for r in rows if r["arm"] == "geniex" and r["depth"] == d]
        if o and g:
            overlap = not (min(o) > max(g) or min(g) > max(o))
            print("d%-5d ranges %s" % (d, "OVERLAP -- indistinguishable"
                                       if overlap else "are disjoint"), flush=True)
    # Said once more at the end, where the numbers are read: `outcome` is
    # about the loop, so "complete" alone would let a table with an arm's rows
    # missing pass for a finished A/B.
    gaps = incomplete_runs(failed_starts, died_mid_run, len(order))
    if gaps:
        print("\n%s" % gaps, flush=True)
    # An arm with nothing in it is not a slow arm, and the loop reaching its
    # end does not make the run an A/B. `outcome` is overwritten here (and not
    # only printed) because it is what the results file carries and what the
    # exit status is read off, and "complete" on a one-armed sweep is the
    # reading a wrapper script takes for a finished comparison.
    empty = arms_with_no_rows(rows, arms)
    if empty:
        print("\nNOT an A/B: %s produced no rows at all, so there is nothing "
              "to compare -- this run exits non-zero. The reason is above, on "
              "the arm-run or the depth it happened at."
              % " and ".join("`%s`" % n for n in empty), flush=True)
        if outcome == "complete":
            outcome = "incomplete: no rows for %s" % ", ".join(empty)
    write_note = None
    if a.out:
        # Everything a reader needs to tell two sweeps apart. The file used to
        # hold rows, depths and tokens only, so two sweeps against different
        # bundles were indistinguishable once the terminal scrolled away.
        result = {
            "tool": "bench_servers",
            "outcome": outcome,
            "started": started, "finished": finished,
            "bundle_dir": BUNDLE_DIR,
            "n_ctx": ctx,
            "n_ctx_source": "--n-ctx" if a.n_ctx else
                            ("genie_config.json" if ctx else "assumed 4096"),
            "arms": arms,
            "passes": a.passes, "order": order,
            "depths": depths, "tokens": a.tokens, "timeout": a.timeout,
            "acceptance": {"min_fraction_of_cap": EARLY_STOP_TOLERANCE,
                           "min_steps": be.MIN_DECODE_STEPS},
            # What every arm was sent besides its model, prompt and cap: the
            # thinking-off spellings, cache_prompt, and the fact that the cap
            # went out under BOTH names (the per-arm "cap" key this replaced
            # was never in the file at all).
            "cap_spellings": ["max_tokens", "max_completion_tokens"],
            "request_settings": {
                k: v for k, v in request_body("", "", 0).items()
                if k not in ("model", "messages", "max_tokens",
                             "max_completion_tokens")},
            "depth_means": "tokens of the untemplated user message under the "
                           "bundle tokenizer; prompt_tokens is what the server "
                           "reported, when it did",
            "token_counter": "local, %s" % os.path.join(BUNDLE_DIR, "tokenizer.json"),
            "clock_instrument": "Get-Counter '\\Processor Information(_Total)\\% "
                                "Processor Performance' via bench_endpoint."
                                "box_state; None means unreadable",
            "logs": {name: os.path.join(log_dir, "bench_servers-%s.log" % name)
                     for name in arms},
            "failed_starts": failed_starts,
            "died_mid_run": died_mid_run,
            # Who else held the Hexagon: every NPU server the run went on
            # beside under --allow-other-npu-servers, with the run it was
            # seen before (0 = at startup), and the runs whose look could not
            # be taken at all -- "none seen" and "not looked" are different
            # facts about a number.
            "other_npu_servers": {"allowed": a.allow_other_npu_servers,
                                  "seen": npu_seen,
                                  "unscanned_runs": unscanned},
            "rows": rows,
        }
        # write_results decides WHERE, and says so when it is not where the
        # operator asked. It never discards the record: the per-row JSON is
        # not on the screen, and this is the only copy of it.
        written, write_note = write_results(a.out, result, out_at_start)
        if write_note:
            print("\n%s." % write_note, flush=True)
        if written:
            print("\nwrote %s (%d samples, %s)" % (written, len(rows), outcome),
                  flush=True)
    # Non-zero whenever the loop did not run to its end, so a wrapper script
    # cannot take an interrupted, refused or dead run for a finished one --
    # and non-zero too when the loop DID run to its end and left an arm with
    # no rows, which is the same lie in a different shape (see the `empty`
    # block above, which is what puts that in `outcome`). Losing SOME of an
    # arm's runs does not change this: the loop went on without them, the arm
    # still has rows, and the gaps are on the screen (as they happened, and
    # again in the closing lines) and in `failed_starts` or `died_mid_run`.
    # And non-zero when the record is not at --out, whatever the sweep did:
    # a wrapper reading --out after a 0 would be reading someone else's file.
    if write_note:
        return 1
    return 0 if outcome == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
