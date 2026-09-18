"""What the two PowerShell launchers promise, pinned from Python.

Two kinds of test live here, and a few literals read both ways.

THE EXIT-75 CONTRACT. A wedged HTP cannot be recovered inside the server
process, so the server exits EXIT_WEDGED and asks to be replaced; the launcher
restarts on exactly that code, treats every other non-crash exit as deliberate,
and re-emits it when it gives up so an outer supervisor sees the same signal.
The number crosses a process boundary as a bare literal on both sides -- a
PowerShell script and a Python module have no definition they can share -- so
this file is the shared definition. It pins the Python constant to 75 and
reads the launcher to confirm it gates on, and re-emits, the same 75. Those
tests need nothing but Python.

THE LAUNCHERS' OWN LOGIC. run-genie-server.ps1 and run-llama-server.ps1 carry
real decisions -- the restart cap, newest-QAIRT discovery, env save/restore,
port and host parsing, log naming and rotation, the placement report -- and
until this file nothing executed a line of either. They are tested here WITHOUT
being run: a launcher run loads a 3 GB bundle onto the NPU or starts a
llama-server, and this suite is device-free. Instead the harness parses the
script with PowerShell's own parser, lifts individual statements and functions
out of the AST by how they begin, and runs only those, against inputs the test
sets up -- a scripted stand-in for the interpreter, a fake clock, a temp
directory. `piece("while ($true)")` is the supervise loop exactly as written,
not a copy of it, so an edit to the launcher is an edit to what is tested; a
statement that moves or is reworded fails loudly ("no statement starts with")
rather than silently testing nothing.

LITERALS READ FROM PYTHON. Where what a launcher promises comes down to a
literal -- the regex that decides which directory names are QAIRT versions,
the API the finally restores the environment through -- the literal is read
out of the script and judged here as well, with full-line comments dropped so
that a comment explaining what the code used to do cannot satisfy, or fail, a
pin. Those need nothing but Python either. One of them is the ONLY test that
can fail for its bug on this box: restoring an unset var through
[Environment]::SetEnvironmentVariable deletes it on Windows PowerShell 5.1 and
leaves it set-but-empty from .NET 9 on, so the harness run below passes either
way wherever 5.1 is the PowerShell that runs it.

The harness tests need a `powershell` (or `pwsh`) on PATH and are skipped without
one. They are the only place in the suite that starts a subprocess, and each
costs about half a second of interpreter start-up, which is why several
no-exit cases share one process through a module-scoped fixture while anything
that ends in `exit` -- which ends the harness too -- gets its own.

WHOLE-SCRIPT RUNS, the one exception to "run only pieces". What a help flag,
a stray argument or a missing binary does is decided by PowerShell's own
parameter binding before a single statement runs, so those are run as the
operator runs them -- `powershell -File <launcher> <args>`, or `& <launcher>`
in-shell -- and only ever under an environment that stops the script at its
first real check should the path under test regress: a GENIE_NPU_ROOT with no
bundle and a GENIE_PYTHON that does not exist, or a LLAMA_BIN_DIR with no
llama-server.exe in it. See launcher_env().

Device-free: no NPU, no bundle, no server, and no launcher run that could get
as far as starting one.
"""

import base64
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import genie_server
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
LAUNCHER = SRC / "run-genie-server.ps1"
LLAMA_LAUNCHER = SRC / "run-llama-server.ps1"

# The launcher line that depends on the value: in the supervise loop of
# src/run-genie-server.ps1,
#     if ($code -ne 75 -and -not $crashed) { ... "not a wedge, not restarting" ... exit $code }
# and, in the give-up branch a few lines below it, `exit 75`. Change
# EXIT_WEDGED to anything else and a wedge takes the "not a wedge" branch with
# nothing else failing: 76 is far above the native-crash window (-le -65536),
# so Test-NativeCrash stays false and the launcher exits instead of restarting.
WEDGE_GATE = re.compile(r"if \(\$code -ne (\d+) -and -not \$crashed\)")
LITERAL_EXIT = re.compile(r"^\s*exit (\d+)\s*$", re.M)


def test_exit_wedged_is_75():
    assert genie_server.EXIT_WEDGED == 75, "EX_TEMPFAIL, and the launcher gate is a literal 75"


def test_the_launcher_gates_restarts_on_the_same_code():
    text = LAUNCHER.read_text(encoding="utf-8")
    m = WEDGE_GATE.search(text)
    assert m, "the restart gate in run-genie-server.ps1 moved; update WEDGE_GATE here"
    assert int(m.group(1)) == genie_server.EXIT_WEDGED


def test_the_launcher_reemits_the_same_code_when_it_gives_up():
    text = LAUNCHER.read_text(encoding="utf-8")
    literal_exits = {int(c) for c in LITERAL_EXIT.findall(text)}
    assert genie_server.EXIT_WEDGED in literal_exits, (
        "the give-up branch must exit %d for an outer supervisor; literal exits found: %s"
        % (genie_server.EXIT_WEDGED, sorted(literal_exits)))


# --- literals read from Python -------------------------------------------------

def launcher_code(launcher=LAUNCHER):
    """The launcher's text without its comments: full-line ones, and the
    `<# ... #>` comment-based help block at the top.

    The comments here are long and say what the code USED to do, by name, so a
    pin on the raw text would be met (or broken) by prose. The help block is
    prose too -- Get-Help's, not the code's.
    """
    text = re.sub(r"<#.*?#>", "", launcher.read_text(encoding="utf-8"), flags=re.S)
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


# The filter in newest-QAIRT discovery: which directory names under <root>\qairt
# are versions at all. Quoted with '...' in the launcher, so no escaping to undo.
QAIRT_NAME_FILTER = re.compile(r"\$_\.Name -match '([^']+)'")
INT32_MAX = 2 ** 31 - 1


def test_the_qairt_name_filter_admits_only_what_a_version_can_hold():
    # Every name the filter admits is cast with [version], whose components
    # are Int32. The filter was ^\d+(\.\d+){1,3}$ -- any number of digits --
    # so ONE all-digits stray with a long part (a full build stamp, a
    # timestamped backup) threw "Value was either too large or too small for an
    # Int32" out of Sort-Object and killed the launcher, a valid SDK beside it.
    found = QAIRT_NAME_FILTER.findall(launcher_code())
    assert len(found) == 1, "the QAIRT name filter moved; update QAIRT_NAME_FILTER here"
    admits = re.compile(found[0])
    for name in ("2.45.0.260326", "2.100.0.1", "2.9", "2.9.0",
                 "999999999.999999999.999999999.999999999"):
        assert admits.match(name), name
        assert all(int(part) <= INT32_MAX for part in name.split(".")), name
    for name in ("2.45.0.99999999999",      # the verifier's reproduction
                 "2.45.0.2147483648",       # Int32 max + 1: ten digits is already too many
                 "2.45.0.2603261530",       # a yymmddHHMM stamp
                 "2.45.0.260326153000",     # a full yymmddHHMMSS stamp
                 "1.2.3.4.5", "2", "latest", "v2.50", "2.45.0.260326-arm64", "2..45", ""):
        assert not admits.match(name), name
    # .NET's \d (like Python's) admits every Unicode digit; [version] parses
    # only ASCII ones, so a name in Arabic-Indic digits was the same cast error.
    assert not admits.match("\u0662.\u0664\u0665")
    # The property, not just the examples: no admitted component can overflow.
    assert not admits.match("2." + "9" * 10)
    assert admits.match("2." + "9" * 9)


def test_the_qairt_sort_key_cannot_throw_on_a_name_it_cannot_parse():
    # Belt to the filter's braces. A bare `Sort-Object { [version]($_.Name) }`
    # throws a terminating cast error out of the pipeline for ANY name
    # [version] rejects, and $ErrorActionPreference is Stop, so the ordering
    # rested entirely on the filter above never letting one through -- which is
    # the assumption that failed. The key catches, and hands back the lowest
    # version there is, so under -Descending an unparseable name sorts LAST.
    # The behaviour is run in
    # test_the_sort_key_ranks_a_name_it_cannot_parse_below_every_version.
    code = launcher_code()
    assert not re.search(r"Sort-Object \{ \[version\]", code), "a bare cast is back in the sort key"
    assert re.search(r"Sort-Object \{ try \{ \[version\]\(\$_\.Name\) \} "
                     r'catch \{ \[version\]"0\.0" \} \} -Descending', code)


def test_the_finally_restores_through_the_env_provider_not_the_dotnet_api():
    # `[Environment]::SetEnvironmentVariable($name, $savedEnv[$name], "Process")`
    # reads as "null deletes". It never gets null: PowerShell's method binder
    # passes "" for $null to a [string] parameter, and "" deletes only up to
    # .NET 8 -- from .NET 9 (pwsh 7.5+) it SETS an empty value, so an in-shell
    # run left GENIE_HOST / GENIE_PORT / GENIE_MODEL_ID set but empty for the
    # next thing in that shell to read. Remove-Item on the env: provider is a
    # delete on every version. The dynamic half of this lives in
    # test_every_env_write_is_inside_the_try_and_undone_by_the_finally; on 5.1
    # it cannot tell the two apart, which is why this pin exists.
    code = launcher_code()
    assert "SetEnvironmentVariable" not in code
    head, sep, finally_body = code.rpartition("} finally {")
    assert sep, "the launcher's try/finally moved; update this test"
    assert re.search(r'Remove-Item -LiteralPath "env:\$name"', finally_body)
    assert re.search(r'Set-Item -LiteralPath "env:\$name" -Value \$savedEnv\[\$name\]',
                     finally_body)
    # ...and the delete is chosen by null-ness, not by truthiness: a saved ""
    # (representable from .NET 9 on) is a value that was found, not an absence.
    assert re.search(r"if \(\$null -eq \$savedEnv\[\$name\]\)", finally_body)


def test_the_arch_that_is_compared_is_never_the_raw_capture():
    # The interpreter's stdout is an ARRAY the moment a wrapper prints a second
    # line, and `-notmatch` on an array returns the non-matching elements --
    # truthy -- so a native ARM64 python behind a chatty .cmd shim was refused.
    # One TAGGED line is the answer, because a shim without `@echo off` echoes
    # its own command lines both before python's print and after it. The
    # behaviour is tested through the harness below; this holds the shape where
    # there is no PowerShell to run it.
    code = launcher_code()
    assert not re.search(r"^\s*\$arch = & ", code, re.M)
    assert re.search(r"^\s*\$arch = .*GENIE_ARCH=.*Select-Object -Last 1", code, re.M)


# The one line the interpreter is asked for, lifted out of the launcher and run
# here. `& $python -c "..."`, a double-quoted PowerShell string with no `$` in
# it, so what is between the quotes is what python receives.
ARCH_PROBE = re.compile(r'^\s*\$archOut = & \$python -c "([^"]+)"', re.M)


def test_the_arch_probe_asks_the_interpreter_for_its_own_build():
    # platform.machine() is the wrong question and was the first one asked: from
    # CPython 3.12 on Windows it reports the HOST cpu (WMI), so an emulated x64
    # python on this box answers ARM64 -- the exact interpreter the refusal
    # exists to stop -- and answers it differently run to run, because the
    # WMI-cold first call falls back to PROCESSOR_ARCHITECTURE and says AMD64.
    # sysconfig.get_platform() is the interpreter's own build tag (win-arm64
    # against win-amd64) and no host can move it. The harness below cannot see
    # this: it substitutes a scriptblock for $python, so the probe is run for
    # real here instead, under the interpreter running the suite.
    m = ARCH_PROBE.search(launcher_code())
    assert m, "no `$archOut = & $python -c \"...\"` line in the launcher"
    probe = m.group(1)
    assert "platform.machine" not in probe
    assert "sysconfig.get_platform" in probe
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          encoding="utf-8", errors="replace", timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["GENIE_ARCH=" + sysconfig.get_platform()], done.stdout


# What the launcher then compares that tag against. PowerShell's -match is
# case-insensitive, hence re.I here.
ARCH_GATE = re.compile(r'^\s*if \(\$arch -notmatch "([^"]+)"\)', re.M)


def test_the_arch_gate_admits_a_win_arm64_build_and_refuses_a_win_amd64_one():
    m = ARCH_GATE.search(launcher_code())
    assert m, "no `if ($arch -notmatch \"...\")` line in the launcher"
    accepts = re.compile(m.group(1), re.I)
    assert accepts.search("win-arm64"), "the native build this box serves on"
    assert not accepts.search("win-amd64"), "what every emulated x64 build prints"
    assert not accepts.search("win32")


# --- the piece harness --------------------------------------------------------

POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
needs_powershell = pytest.mark.skipif(
    POWERSHELL is None, reason="no powershell/pwsh on PATH; the launchers are Windows scripts")

# Parses the launcher and defines Get-Piece. Nothing in the launcher runs as a
# result of this; ParseFile only builds the AST. Harness failures exit with
# their own codes and a HARNESS-ERROR line so they can never be mistaken for
# the launcher's `exit 1`.
PRELUDE = r"""
$ErrorActionPreference = "Stop"
$__tok = $null; $__err = $null
$__ast = [System.Management.Automation.Language.Parser]::ParseFile('@@LAUNCHER@@', [ref]$__tok, [ref]$__err)
if ($__err.Count -gt 0) { Write-Host "HARNESS-ERROR: the launcher does not parse"; exit 97 }
$__stmts = $__ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.StatementAst] }, $true)
function Get-Piece([string]$prefix) {
    # First in document order; at one offset the longest, i.e. the outermost
    # (an assignment and the `if` expression it assigns can start together).
    $hit = $null
    foreach ($s in $__stmts) {
        if (-not $s.Extent.Text.StartsWith($prefix, [StringComparison]::Ordinal)) { continue }
        if (-not $hit -or $s.Extent.StartOffset -lt $hit.Extent.StartOffset -or
            ($s.Extent.StartOffset -eq $hit.Extent.StartOffset -and
             $s.Extent.Text.Length -gt $hit.Extent.Text.Length)) { $hit = $s }
    }
    if (-not $hit) { Write-Host "HARNESS-ERROR: no statement starts with: $prefix"; exit 96 }
    $hit.Extent.Text
}
"""

# Every `$env:NAME = ...` assignment in the script, and whether it sits inside
# the body of the script's first try. AST, not a regex: the launchers mention
# `$env:GENIE_NPU_ROOT = '<dir>'` inside a Write-Host string and in comments,
# and one real write sits mid-line behind an `if`.
ENV_WRITES = r"""
$__try = $__ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.TryStatementAst] }, $true) |
    Sort-Object { $_.Extent.StartOffset } | Select-Object -First 1
$__writes = $__ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $n.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
        $n.Left.VariablePath.DriveName -eq 'env' }, $true)
foreach ($w in $__writes) {
    $inside = $false
    if ($__try) {
        $inside = ($w.Extent.StartOffset -gt $__try.Body.Extent.StartOffset -and
                   $w.Extent.EndOffset -lt $__try.Body.Extent.EndOffset)
    }
    Write-Host ("WRITE " + ($w.Left.VariablePath.UserPath -replace '^env:', '') + " " + $inside)
}
"""


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def piece(prefix):
    """PowerShell that runs, in the harness's own scope, the launcher statement starting so."""
    return ". ([scriptblock]::Create((Get-Piece %s)))\n" % ps_quote(prefix)


def run_pieces(launcher, body, env=None):
    """Run `body` under the prelude for `launcher`. Returns (exit code, stdout).

    The child's environment has every GENIE_* / LLAMA_* var removed before
    `env` is applied: both launchers read those, and a developer's exported
    LLAMA_PORT or GENIE_MAX_RESTARTS must not decide whether a test passes.
    -EncodedCommand rather than a temp .ps1 so no execution policy applies.
    """
    script = PRELUDE.replace("@@LAUNCHER@@", str(launcher).replace("'", "''")) + body
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    child_env = {k: v for k, v in os.environ.items()
                 if not k.upper().startswith(("GENIE_", "LLAMA_"))}
    child_env.update(env or {})
    done = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, encoding="utf-8", errors="replace", timeout=120, env=child_env)
    assert "HARNESS-ERROR" not in done.stdout, done.stdout + done.stderr
    return done.returncode, done.stdout


def results(out):
    """The `RESULT key=value` lines a harness body printed, as a dict."""
    found = {}
    for line in out.splitlines():
        if line.startswith("RESULT "):
            key, _, value = line[len("RESULT "):].partition("=")
            found[key] = value
    return found


def section(out, name):
    """The output printed between `BEGIN name` and `END name`."""
    m = re.search(r"^BEGIN %s\r?\n(.*?)^END %s\r?$" % (re.escape(name), re.escape(name)),
                  out, re.S | re.M)
    assert m, "no section %r in:\n%s" % (name, out)
    return m.group(1)


def unsigned(code):
    """An NTSTATUS as a Windows process exit code reads back: unsigned 32-bit."""
    return code & 0xFFFFFFFF


# --- run-genie-server.ps1: the supervise loop ------------------------------------

ACCESS_VIOLATION = -1073741819      # 0xC0000005, the QnnHtp.dll crash the launcher cites
CONTROL_C = -1073741510             # 0xC000013A, STATUS_CONTROL_C_EXIT


def supervise(lives, env=None):
    """Run the launcher's real supervise loop over scripted server lives.

    `lives` is [(seconds the server stayed up, its exit code), ...]. The
    interpreter is a scriptblock that advances a fake clock by the life's
    length and sets $LASTEXITCODE; Get-Date and Start-Sleep are shadowed by
    functions on the same clock (a PowerShell function outranks a cmdlet), so
    hours of restart history run in the time it takes to start PowerShell.
    Running out of scripted lives exits 99: a loop that never gives up must
    fail its test, not hang it.
    """
    body = (
        "$here = 'C:\\harness-has-no-server'\n"
        "$arch = 'win-arm64'; $bindHost = '127.0.0.1'; $port = '8123'\n"
        + piece("function Get-EnvInt")
        + piece("$maxRestarts")
        + piece("$cooldown")
        + piece("$window")
        + piece("$failures = @()")
        + piece("$STATUS_CONTROL_C_EXIT")
        + piece("function Test-NativeCrash")
        + "$global:clock = [datetime]'2026-01-01T00:00:00'\n"
        "function Get-Date { $global:clock }\n"
        "function Start-Sleep { param($Seconds = 0)\n"
        "    $global:clock = $global:clock.AddSeconds($Seconds); Write-Host \"STUB: slept $Seconds\" }\n"
        "$global:secs = @(%s)\n"
        "$global:codes = @(%s)\n"
        "$global:life = 0\n"
        "$python = {\n"
        "    if ($global:life -ge $global:secs.Count) { Write-Host 'STUB: script exhausted'; exit 99 }\n"
        "    $global:clock = $global:clock.AddSeconds($global:secs[$global:life])\n"
        "    $global:LASTEXITCODE = $global:codes[$global:life]\n"
        "    $global:life++\n"
        "}\n"
        % (", ".join("%d" % s for s, _ in lives), ", ".join("%d" % c for _, c in lives))
        + piece("while ($true)"))
    return run_pieces(LAUNCHER, body, env)


@needs_powershell
def test_a_device_that_wedges_every_few_minutes_reaches_the_give_up():
    # THE regression this cap was rewritten for. The old rule counted only
    # lives under 120s and reset the count on any longer one, but the wedge
    # detector needs at least 180s before the server exits 75 -- so no wedge
    # life could ever count, this exact history restarted forever at "restart
    # 1/5", and the give-up branch (with the pnputil hint it carries) was dead.
    code, out = supervise([(200, 75)] * 8)
    assert "STUB: script exhausted" not in out, out
    assert code == genie_server.EXIT_WEDGED, out
    assert "restart 5/5 within 3600s" in out
    assert "the engine wedged 6 times within 3600s" in out
    # The recovery command is this box's PnP instance id, so it has to come
    # with the way to find one's own.
    assert "pnputil /restart-device" in out
    assert "Get-PnpDevice -FriendlyName '*Hexagon*'" in out
    # ...and with its blast radius, BEFORE it: a device restart resets the NPU
    # under every process on the machine -- on this shared box, another
    # session's server or benchmark -- and the hint used to hand the command
    # out with no word of that. The check it offers is one read-only line.
    warning = out.index("resets the NPU under EVERY")
    assert warning < out.index("pnputil /restart-device"), out
    assert "another session's server or benchmark" in out
    assert "tasklist /m QnnHtp.dll" in out
    assert out.index("tasklist /m QnnHtp.dll") < out.index("pnputil /restart-device")
    # "Verified" is scoped to what was verified: the crawl, which never reaches
    # this branch. It used to read as a verified fix for the wedge in hand.
    assert "verified only" in out and "not against" in out
    assert "a wedge or a crash" in out
    assert "(Verified fix for the" not in out


@needs_powershell
def test_failures_older_than_the_window_age_out_and_the_loop_says_so():
    # A wedge every ~34 minutes: never more than two inside an hour, so the
    # loop must keep going -- and must SAY the count dropped, because "restart
    # 2/5" under the seventh identical stanza is otherwise unexplained. The
    # last life is a deliberate exit, which ends the loop with its own code.
    code, out = supervise([(2000, 75)] * 7 + [(5, 3)])
    assert code == 3, out
    assert "server exited 3 -- not a wedge, not restarting." in out
    assert "1 earlier failure(s) aged out of the 3600s window; 1 still count." in out
    assert max(int(n) for n in re.findall(r"restart (\d+)/5", out)) == 2


@needs_powershell
def test_a_mixed_window_is_summarised_as_mixed_and_junk_integers_fall_back():
    # A negative cooldown used to reach Start-Sleep and throw inside the
    # restart path -- the supervisor died at the one moment it was meant to
    # act. Junk now falls back loudly to the default; a valid value is used.
    code, out = supervise(
        [(130, ACCESS_VIOLATION), (130, 75), (130, 75)],
        env={"GENIE_MAX_RESTARTS": "1", "GENIE_RESTART_COOLDOWN": "-5",
             "GENIE_RESTART_WINDOW": "soon"})
    assert "GENIE_RESTART_COOLDOWN='-5' is not an integer >= 0; using 25." in out
    assert "GENIE_RESTART_WINDOW='soon' is not an integer >= 1; using 3600." in out
    assert "GENIE_MAX_RESTARTS" not in out.split("starting Genie server")[0]
    assert "server crashed: exit 0xC0000005" in out
    assert "restart 1/1 within 3600s; next in 25s." in out
    assert "STUB: slept 25" in out
    assert "the engine crashed and wedged 2 times within 3600s" in out
    assert code == genie_server.EXIT_WEDGED


@needs_powershell
def test_ctrl_c_is_not_a_crash_to_restart():
    # STATUS_CONTROL_C_EXIT is an NTSTATUS like any crash code; restarting on
    # it would turn the operator's own stop into a relaunch.
    code, out = supervise([(30, CONTROL_C), (30, 75)])
    assert "not a wedge, not restarting" in out
    assert "restart" not in out.replace("not restarting", "")
    assert code == unsigned(CONTROL_C)


# --- run-genie-server.ps1: SDK discovery, the interpreter, the environment --------

DISCOVERY = (piece("$sdkDir = $env:GENIE_SDK_DIR") + piece("$qairt = Join-Path")
             + piece("$qairtNote = $null") + piece("if (-not $sdkDir)"))


@needs_powershell
def test_newest_qairt_is_a_version_compare_and_ignores_names_that_are_not_versions(tmp_path):
    # A string sort ranks 2.9 above 2.100 and "latest" above every number.
    full = tmp_path / "full"
    for name in ("2.45.0.260326", "2.100.0.1", "2.9.0.1", "latest", "backup-2.200.0"):
        (full / "qairt" / name).mkdir(parents=True)
    empty = tmp_path / "empty"
    (empty / "qairt" / "latest").mkdir(parents=True)     # a directory, but not an SDK
    absent = tmp_path / "absent"
    absent.mkdir()
    body = ""
    for label, root, explicit in (("full", full, ""), ("empty", empty, ""),
                                  ("absent", absent, ""), ("explicit", full, "D:\\my-sdk")):
        body += ("$env:GENIE_NPU_ROOT = %s\n$env:GENIE_SDK_DIR = %s\n"
                 % (ps_quote(root), ps_quote(explicit) if explicit else "$null")
                 + DISCOVERY
                 + "Write-Host ('RESULT %s.sdk=' + $sdkDir)\n" % label
                 + "Write-Host ('RESULT %s.note=' + $qairtNote)\n" % label)
    code, out = run_pieces(LAUNCHER, body)
    got = results(out)
    assert got["full.sdk"] == str(full / "qairt" / "2.100.0.1")
    assert got["full.note"] == ""
    assert got["empty.sdk"] == ""
    assert got["empty.note"].startswith("searched %s for a QAIRT version directory" % (empty / "qairt"))
    assert got["empty.note"].endswith("none found")
    assert got["absent.sdk"] == ""
    assert got["absent.note"] == "no %s directory to search for a QAIRT version" % (absent / "qairt")
    assert got["explicit.sdk"] == "D:\\my-sdk"      # an explicit GENIE_SDK_DIR skips discovery
    assert code == 0


@needs_powershell
def test_an_empty_qairt_directory_is_named_in_the_failure(tmp_path):
    # "tried: (unset)" plus "set GENIE_NPU_ROOT" was advice for a root the
    # operator had already set correctly. The line that helps names the
    # directory that was scanned.
    (tmp_path / "qairt").mkdir()
    bundle = tmp_path / "bundles" / "some-bundle"
    bundle.mkdir(parents=True)
    body = ("$env:GENIE_NPU_ROOT = %s\n$env:GENIE_BUNDLE_DIR = %s\n"
            "$modelExplicit = $false; $Model = 'qwen3-4b'\n"
            % (ps_quote(tmp_path), ps_quote(bundle))
            + DISCOVERY + piece("foreach ($pair in"))
    code, out = run_pieces(LAUNCHER, body)
    assert code == 1
    assert "GENIE_SDK_DIR is not set to an existing directory." in out
    assert "tried: (unset)" in out
    assert "searched %s for a QAIRT version directory" % (tmp_path / "qairt") in out
    assert "Unpack the SDK there" in out


# Names made only of digits and dots that [version] cannot hold: its components
# are Int32, so ten digits is already too many.
OVERFLOWING = ("2.45.0.99999999999",        # the reviewer's reproduction
               "2.45.0.2147483648",         # Int32 max + 1
               "2.45.0.260326153000")       # a full yymmddHHMMSS build stamp


@needs_powershell
def test_a_stray_all_digits_directory_does_not_kill_discovery(tmp_path):
    # The filter was ^\d+(\.\d+){1,3}$, so each of these passed it and then
    # threw "Value was either too large or too small for an Int32" out of
    # Sort-Object -- a terminating error under $ErrorActionPreference = "Stop",
    # so the launcher died without a [run] line, the valid SDK sitting beside
    # the stray. All three sort ABOVE 2.44.0.1 as numbers; they are not
    # versions, so they are ignored like "latest" is.
    for name in (*OVERFLOWING, "2.44.0.1"):
        (tmp_path / "qairt" / name).mkdir(parents=True)
    body = ("$env:GENIE_NPU_ROOT = %s\n$env:GENIE_SDK_DIR = $null\n" % ps_quote(tmp_path)
            + DISCOVERY
            + "Write-Host ('RESULT sdk=' + $sdkDir)\nWrite-Host ('RESULT note=' + $qairtNote)\n")
    code, out = run_pieces(LAUNCHER, body)
    got = results(out)
    assert got.get("sdk") == str(tmp_path / "qairt" / "2.44.0.1"), out
    assert got["note"] == ""
    assert code == 0


@needs_powershell
def test_the_sort_key_ranks_a_name_it_cannot_parse_below_every_version(tmp_path):
    # The sort key on its own, with the filter taken out of the way: the
    # discovery statement is run with its name filter swapped for one that
    # admits everything, so every name reaches [version]. Whatever a future
    # edit does to the filter, a name the cast rejects must rank below the
    # real versions rather than throw. The filter literal is read out of the
    # launcher, so rewording it cannot quietly turn the swap into a no-op --
    # the harness checks that the text changed.
    found = QAIRT_NAME_FILTER.findall(launcher_code())
    assert len(found) == 1, "the QAIRT name filter moved; update QAIRT_NAME_FILTER here"
    for name in (*OVERFLOWING, "latest", "zz-backup", "2.9.0", "2.44.0.1"):
        (tmp_path / "qairt" / name).mkdir(parents=True)
    body = ("$env:GENIE_NPU_ROOT = %s\n$env:GENIE_SDK_DIR = $null\n" % ps_quote(tmp_path)
            + piece("$sdkDir = $env:GENIE_SDK_DIR") + piece("$qairt = Join-Path")
            + piece("$qairtNote = $null")
            + "$__found = Get-Piece 'if (-not $sdkDir)'\n"
            "$__open = $__found.Replace(%s, '.')\n" % ps_quote(found[0])
            + "if ($__open -ceq $__found) { Write-Host 'HARNESS-ERROR: the filter was not swapped'; exit 95 }\n"
            ". ([scriptblock]::Create($__open))\n"
            "Write-Host ('RESULT sdk=' + $sdkDir)\n")
    code, out = run_pieces(LAUNCHER, body)
    assert results(out).get("sdk") == str(tmp_path / "qairt" / "2.44.0.1"), out
    assert code == 0


@needs_powershell
def test_a_named_interpreter_that_does_not_exist_is_refused_by_name():
    body = piece("$python = if") + piece("if (-not (Get-Command $python")
    code, out = run_pieces(LAUNCHER, body + "Write-Host 'RESULT reached=the-end'\n",
                           env={"GENIE_PYTHON": "no-such-python-for-this-test"})
    assert code == 1
    assert "python not found: 'no-such-python-for-this-test'. Set GENIE_PYTHON" in out
    assert "reached" not in results(out)


@needs_powershell
def test_a_non_arm64_interpreter_is_a_refusal_not_a_warning():
    # Genie.dll is pure ARM64, so nothing after this check can succeed under an
    # x64 python; it used to be a Write-Warning followed by the DLL-load crash.
    # The two values are what sysconfig.get_platform() prints for the two
    # builds -- a native ARM64 python and an emulated x64 one.
    body = ("$python = 'python'\n$arch = 'win-arm64'\n" + piece("if ($arch -notmatch")
            + "Write-Host 'RESULT arm64=accepted'\n$arch = 'win-amd64'\n"
            + piece("if ($arch -notmatch") + "Write-Host 'RESULT amd64=accepted'\n")
    code, out = run_pieces(LAUNCHER, body)
    got = results(out)
    assert got.get("arm64") == "accepted"
    assert "amd64" not in got
    assert code == 1
    assert "python arch is 'win-amd64'" in out
    assert "GENIE_PYTHON" in out


# The capture, the reduction to one TAGGED line, and the check, as the launcher
# has them. `$python` is called with `&`, so a scriptblock stands in for it:
# what the block emits is captured exactly as a native command's stdout lines
# are -- a scalar for one line, an array for more, nothing for none.
ARCH_CHECK = (piece("$archOut = &") + piece("$archSaid = ") + piece("$arch = ")
              + piece("if ($arch -notmatch"))


def interpreter_printing(lines):
    return "$python = { %s }\n" % "; ".join(ps_quote(line) for line in lines)


@needs_powershell
def test_a_chatty_wrapper_does_not_get_a_native_arm64_interpreter_refused():
    # GENIE_PYTHON takes anything Get-Command resolves, a .cmd shim included.
    # One that prints a line before python does makes the capture an ARRAY, and
    # `$array -notmatch "arm64|aarch64"` does not answer "did it match": it
    # returns the elements that did NOT, and a non-empty array is true. So the
    # native ARM64 interpreter behind the shim was refused, by a message that
    # read "python arch is 'activating env foo win-arm64'". While the check was
    # a Write-Warning that was noise; as an `exit 1` it has to be right.
    cases = (("quiet", ["GENIE_ARCH=win-arm64"]),
             ("banner", ["activating env foo", "GENIE_ARCH=win-arm64"]),
             # No `@echo off`: cmd echoes every command line it runs, so the
             # shim talks BEFORE python and -- `exit /b %ERRORLEVEL%`, the way
             # a shim hands exit 75 back to the supervise loop -- AFTER it too.
             # This exact capture got a native ARM64 python refused while the
             # last non-blank line was read as the machine: the answer was
             # `C:\shims>exit /b 0`.
             ("echoed", ["", "C:\\shims>C:\\py\\python.exe -c import sysconfig",
                         "GENIE_ARCH=win-arm64", "", "C:\\shims>exit /b 0"]),
             ("padded", ["  GENIE_ARCH=aarch64  "]))
    body = ""
    for label, lines in cases:
        body += (interpreter_printing(lines) + ARCH_CHECK
                 + "Write-Host ('RESULT %s=' + $arch)\n" % label)
    code, out = run_pieces(LAUNCHER, body)
    assert results(out) == {"quiet": "win-arm64", "banner": "win-arm64",
                            "echoed": "win-arm64", "padded": "aarch64"}, out
    assert "python arch is" not in out
    assert code == 0


@needs_powershell
@pytest.mark.parametrize("printed, read_as, noted", [
    # TAGGED, not merely last: a banner that happens to mention arm64 must not
    # vouch for an x64 python, and the last line can be a shim's echo of its
    # own `exit /b`. The refusal quotes the tagged line it did read.
    (["activating the win-arm64 toolchain env", "GENIE_ARCH=win-amd64"], "win-amd64", None),
    # Output, but nothing tagged: a wrapper talking over a python that never
    # reached the print, or an interpreter answering some older probe. It has
    # not said it is ARM64, so it is refused like any other -- the raw capture
    # used to be ACCEPTED here, an empty capture on the left of -notmatch being
    # an empty collection, i.e. false.
    (["activating env foo", "ARM64"], "",
     "It printed 2 line(s) but no GENIE_ARCH= line"),
    # Nothing at all: it died with its error on stderr, or a wrapper swallowed
    # stdout. A [string] cast of the empty pipeline is $null on 5.1, so .Trim()
    # on it threw "You cannot call a method on a null-valued expression" in
    # place of any [run] line.
    ([], "", "It printed no text on stdout"),
    (["", "   "], "", "It printed no text on stdout"),
])
def test_the_tagged_line_is_the_machine_and_a_wrong_one_is_still_refused(printed, read_as, noted):
    body = interpreter_printing(printed) + ARCH_CHECK + "Write-Host 'RESULT reached=the-end'\n"
    code, out = run_pieces(LAUNCHER, body)
    assert code == 1, out
    assert "reached" not in results(out)
    assert "python arch is '%s'" % read_as in out
    if noted is None:
        assert "[run] (It printed" not in out, "the tagged line speaks for itself"
    else:
        assert noted in out
        assert out.count("[run] (It printed") == 1, "one explanation, the one that applies"


@needs_powershell
def test_genie_port_is_parsed_and_range_checked_with_a_run_line():
    # GENIE_PORT used to be copied through unparsed, and the server degrades a
    # value it cannot read to its OWN default, 8080 -- where run-llama-server
    # puts its CPU leg -- while this launcher's start line announced the typo
    # itself as the port and every tool here went on looking at 8123. Parsed
    # here, the launcher's default is what a bad value falls back to, loudly.
    cases = (("typo", "808O"), ("zero", "0"), ("high", "70000"),
             ("good", "8200"), ("padded", "  8200  "), ("empty", ""))
    body = "$arch = 'win-arm64'; $bindHost = '127.0.0.1'\n"
    for label, value in cases:
        body += ("Write-Host 'BEGIN port.%s'\n" % label
                 + "$env:GENIE_PORT = %s\n" % ps_quote(value)
                 + piece('$port = "8123"') + piece("if ($env:GENIE_PORT)")
                 + "Write-Host ('RESULT port.%s=' + $port)\n" % label
                 + piece('Write-Host "[run] starting Genie server')
                 + "Write-Host 'END port.%s'\n" % label)
    code, out = run_pieces(LAUNCHER, body)
    assert code == 0, out
    assert results(out) == {"port.typo": "8123", "port.zero": "8123", "port.high": "8123",
                            "port.good": "8200", "port.padded": "8200",
                            "port.empty": "8123"}, out
    assert ("[run] WARNING: GENIE_PORT='808O' is not a port number (1-65535); using 8123."
            in section(out, "port.typo")), out
    for label in ("zero", "high"):
        assert "WARNING" in section(out, "port." + label), label
    for label in ("good", "padded", "empty"):
        assert "WARNING" not in section(out, "port." + label), label
    # The line that announces the start says the port that will be served, not
    # the value that was rejected.
    def announced(label):
        return [line for line in section(out, "port." + label).splitlines()
                if "starting Genie server" in line]

    assert announced("typo") == [
        "[run] starting Genie server (python win-arm64) on 127.0.0.1:8123"], out
    assert announced("good") == [
        "[run] starting Genie server (python win-arm64) on 127.0.0.1:8200"], out


@needs_powershell
def test_every_env_write_is_inside_the_try_and_undone_by_the_finally():
    # An in-shell `.\run-genie-server.ps1` runs in the calling shell's process,
    # so whatever it leaves in $env: is fed to the next run: a lingering
    # GENIE_BUNDLE_DIR serves the wrong model, a lingering GENIE_SDK_DIR pins
    # the SDK because discovery only runs while it is unset. Two halves:
    #   static  -- every `$env:X =` sits inside the try whose finally restores,
    #              and is named in $EnvWritten, the list the finally walks;
    #   dynamic -- the save loop and the finally body, run for real, put a set
    #              var back and DELETE one that was unset (not leave it empty).
    body = (ENV_WRITES + piece("$EnvWritten =")
            + "Write-Host ('RESULT written=' + ($EnvWritten -join ','))\n"
            "$env:GENIE_HOST = 'as-found'; $env:GENIE_PORT = $null\n"
            + piece("$savedEnv = @{}") + piece("foreach ($name in $EnvWritten)")
            + "$env:GENIE_HOST = 'overwritten'; $env:GENIE_PORT = '9'; $env:GENIE_SDK_DIR = 'D:\\pinned'\n"
            "foreach ($s in $__try.Finally.Statements) { . ([scriptblock]::Create($s.Extent.Text)) }\n"
            "Write-Host ('RESULT host=' + $env:GENIE_HOST)\n"
            "Write-Host ('RESULT port-deleted=' + ($null -eq [Environment]::GetEnvironmentVariable('GENIE_PORT', 'Process')))\n"
            "Write-Host ('RESULT sdk-deleted=' + ($null -eq [Environment]::GetEnvironmentVariable('GENIE_SDK_DIR', 'Process')))\n")
    code, out = run_pieces(LAUNCHER, body)
    writes = [line.split() for line in out.splitlines() if line.startswith("WRITE ")]
    assert writes, out
    outside = [name for _, name, inside in writes if inside != "True"]
    assert outside == [], "env writes the finally does not enclose: %s" % outside
    got = results(out)
    assert {name for _, name, _ in writes} == set(got["written"].split(","))
    assert {"GENIE_NPU_ROOT", "GENIE_SDK_DIR", "GENIE_HOST", "GENIE_PORT",
            "GENIE_BUNDLE_DIR", "GENIE_MODEL_ID"} == set(got["written"].split(","))
    assert got["host"] == "as-found"
    assert got["port-deleted"] == "True"
    assert got["sdk-deleted"] == "True"
    assert code == 0


# --- run-genie-server.ps1: which bundle, advertised under which id ----------------

# The three tiers the launcher serves, as its own $Bundles map spells them.
# Read here, not imported: a Python copy of the map would agree with itself
# while the launcher drifted.
BUNDLE_4B = "qwen3_4b-genie-w4a16-x-elite-ctx8192-multi"
BUNDLE_8B = "qwen3_8b-genie-w4a16-qualcomm_snapdragon_x_elite"
BUNDLE_8192 = "qwen3_8b-genie-w4a16-x-elite-ctx8192-multi"


def bundle_ids(tmp_path, cases):
    """Run the launcher's real bundle/id block over `cases`.

    `cases` is [(label, $Model, -Model was passed, GENIE_BUNDLE_DIR leaf or
    None, GENIE_MODEL_ID or None), ...]. GENIE_NPU_ROOT is rooted at tmp_path
    because Windows PowerShell 5.1's Join-Path throws DriveNotFound on a drive
    letter that does not exist; nothing under it needs to be created, since
    this block only computes paths.
    """
    body = piece("$Bundles = @{")
    for label, model, explicit, leaf, model_id in cases:
        body += ("Write-Host 'BEGIN %s'\n" % label
                 + "$Model = %s\n" % ps_quote(model)
                 + "$modelExplicit = $%s\n" % ("true" if explicit else "false")
                 + "$env:GENIE_NPU_ROOT = %s\n" % ps_quote(tmp_path)
                 + "$env:GENIE_BUNDLE_DIR = %s\n"
                 % (ps_quote(tmp_path / "bundles" / leaf) if leaf else "$null")
                 + "$env:GENIE_MODEL_ID = %s\n" % (ps_quote(model_id) if model_id else "$null")
                 + piece("$DefaultBundle =") + piece("if ($modelExplicit)")
                 + piece("if (-not $modelExplicit)")
                 + "Write-Host ('RESULT %s.dir=' + (Split-Path -Leaf $env:GENIE_BUNDLE_DIR))\n" % label
                 + "Write-Host ('RESULT %s.id=' + $env:GENIE_MODEL_ID)\n" % label
                 + "Write-Host 'END %s'\n" % label)
    code, out = run_pieces(LAUNCHER, body)
    assert code == 0, out
    return out


@needs_powershell
def test_an_explicit_model_moves_the_id_with_the_bundle_past_a_stale_export(tmp_path):
    # The failure this block exists for: "the right bundle advertised under the
    # wrong id". Every bench table, router and /v1/models consumer in the repo
    # keys on that id, so a shell still exporting the 4B's pair must not make
    # `-Model qwen3-8b` serve the 8B under the 4B's name -- and each tier needs
    # its OWN id, or two of the three measure as one row.
    out = bundle_ids(tmp_path, [
        ("explicit-8b", "qwen3-8b", True, BUNDLE_4B, "qwen3-4b-npu"),
        ("explicit-8192", "qwen3-8b-8192", True, None, None),
        ("explicit-4b", "qwen3-4b", True, None, None),
    ])
    got = results(out)
    assert got["explicit-8b.dir"] == BUNDLE_8B
    assert got["explicit-8b.id"] == "qwen3-8b-npu"
    assert got["explicit-8192.dir"] == BUNDLE_8192
    assert got["explicit-8192.id"] == "qwen3-8b-8192-npu"
    assert got["explicit-4b.dir"] == BUNDLE_4B
    assert got["explicit-4b.id"] == "qwen3-4b-npu"
    ids = {got[k] for k in got if k.endswith(".id")}
    assert len(ids) == 3, "two tiers sharing one id is two measurement runs sharing one label"


@needs_powershell
def test_without_a_model_flag_the_id_is_filled_in_from_the_bundle_it_found(tmp_path):
    # Env-path hygiene: with no -Model and no GENIE_MODEL_ID the server falls
    # back to its own default id (qwen3-4b-npu) whatever the env points at, so
    # the launcher fills the id in for any bundle the map recognises -- and the
    # bundle dir the env set is left exactly as it was.
    out = bundle_ids(tmp_path, [
        ("default", "qwen3-4b", False, None, None),
        ("env-8b", "qwen3-4b", False, BUNDLE_8B, None),
    ])
    got = results(out)
    assert got["default.dir"] == BUNDLE_4B
    assert got["default.id"] == "qwen3-4b-npu"
    assert got["env-8b.dir"] == BUNDLE_8B, "-Model was not passed; the env keeps the bundle"
    assert got["env-8b.id"] == "qwen3-8b-npu", "filled from the bundle, not from the default"
    assert "does not match bundle" not in out, "nothing disagreed"


@needs_powershell
def test_a_bundle_the_map_does_not_know_is_said_out_loud_or_left_alone(tmp_path):
    # An unrecognised bundle cannot be filled in, so it is named -- unless its
    # leaf is a 4B, where the server's own default id is already right and a
    # note would be noise.
    unknown = "qwen3_14b-genie-w4a16-x-elite-ctx8192-multi"
    variant = "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite"
    out = bundle_ids(tmp_path, [
        ("unknown", "qwen3-4b", False, unknown, None),
        ("variant", "qwen3-4b", False, variant, None),
    ])
    got = results(out)
    assert got["unknown.id"] == "", "nothing to fill it in from"
    unknown_out = section(out, "unknown")
    assert "GENIE_MODEL_ID is unset" in unknown_out
    assert "default id (qwen3-4b-npu) for bundle '%s'" % unknown in unknown_out
    assert "anything routes on the reported model id" in unknown_out
    assert got["variant.id"] == ""
    assert "GENIE_MODEL_ID is unset" not in section(out, "variant"),         "a 4B bundle already gets the right default id"


@needs_powershell
def test_an_id_that_disagrees_with_the_bundle_is_warned_about_and_still_wins(tmp_path):
    # A stale export serving the right bundle under the wrong id is the silent
    # failure the whole block exists to catch, and it is the one case where
    # the launcher warns without overriding: env wins, loudly.
    out = bundle_ids(tmp_path, [
        ("stale", "qwen3-4b", False, BUNDLE_8B, "qwen3-4b-npu"),
        ("agrees", "qwen3-4b", False, BUNDLE_8B, "qwen3-8b-npu"),
    ])
    got = results(out)
    assert got["stale.id"] == "qwen3-4b-npu", "the env value is still what gets advertised"
    stale = section(out, "stale")
    assert "GENIE_MODEL_ID 'qwen3-4b-npu' does not match bundle" in stale
    assert "'%s' (this launcher knows it as 'qwen3-8b-npu')" % BUNDLE_8B in stale
    assert "unset GENIE_MODEL_ID if it is stale" in stale
    # The same bundle with the id the map agrees on: silence, so the warning
    # above means something.
    agrees = section(out, "agrees")
    assert got["agrees.id"] == "qwen3-8b-npu"
    assert "does not match" not in agrees


# --- run-llama-server.ps1: everything that does not end in `exit` ------------------

@pytest.fixture(scope="module")
def llama(tmp_path_factory):
    """One PowerShell process for every no-exit llama-launcher case.

    Returns (stdout, logs dir). Cases print between BEGIN/END markers so a test
    can assert on what ITS inputs printed, and RESULT lines for the values.
    """
    if POWERSHELL is None:
        pytest.skip("no powershell/pwsh on PATH")
    logs = tmp_path_factory.mktemp("llama-logs")
    # A previous run's pair for the rotation case, plus a stale .prev that the
    # rotation must overwrite rather than trip on.
    old = logs / "llama-server-rot-cpu-8080.log"
    old.write_text("old stdout", encoding="ascii")
    Path(str(old) + ".err").write_text("old stderr: the crash reason", encoding="ascii")
    (logs / "llama-server-rot-cpu-8080.prev.log").write_text("two runs ago", encoding="ascii")

    body = ENV_WRITES
    for value in ("abc", "0", "70000", "8124", ""):
        body += ("Write-Host 'BEGIN port.%s'\n$portDefault = '8080'\n$env:LLAMA_PORT = %s\n"
                 % (value, ps_quote(value) if value else "$null")
                 + piece("$port = $portDefault") + piece("if ($env:LLAMA_PORT)")
                 + "Write-Host ('RESULT port.%s=' + $port)\nWrite-Host 'END port.%s'\n" % (value, value))
    for bind in ("127.0.0.1", "0.0.0.0", "::", "::1", "fe80::1"):
        body += ("$bindHost = %s\n" % ps_quote(bind)
                 + piece("$probeHost =") + piece("$urlHost =")
                 + "Write-Host ('RESULT url.%s=' + $urlHost)\n" % bind)
    select = (piece("if ($hf -and $gguf)") + piece("$src = if")
              + piece('if ($src -match "Q8_0")') + piece('if ($Leg -eq "gpu" -and $src')
              + piece('if ($Leg -eq "cpu" -and $src'))
    for label, hf, gguf in (("both", "some/Repo-GGUF:Q8_0", "C:\\models\\Other-Q4_0.gguf"),
                            ("hf-only", "some/Repo-GGUF:Q8_0", "")):
        body += ("Write-Host 'BEGIN select.%s'\n$Leg = 'cpu'\n$hf = %s\n$gguf = %s\n"
                 % (label, ps_quote(hf), ps_quote(gguf) if gguf else "$null")
                 + select
                 + "Write-Host ('RESULT select.%s.src=' + $src)\n" % label
                 + "Write-Host ('RESULT select.%s.hf=' + $hf)\n" % label
                 + "Write-Host 'END select.%s'\n" % label)
    for label, src in (("hf", "unsloth/Qwen3.5-9B-GGUF:Q4_0"),
                       ("gguf", "C:\\my models\\Some Model-Q4_K_M.gguf")):
        body += ("$src = %s\n$logDir = 'C:\\logs'\n$Leg = 'gpu'\n$port = '8124'\n" % ps_quote(src)
                 + piece("$stem =") + piece("$log = Join-Path")
                 + "Write-Host ('RESULT log.%s=' + (Split-Path -Leaf $log))\n" % label)
    for label, value in (("default", ""), ("set", "4096")):
        body += ("$env:LLAMA_CACHE_RAM = %s\n" % (ps_quote(value) if value else "$null")
                 + piece("$cacheRam =") + "Write-Host ('RESULT cache.%s=' + $cacheRam)\n" % label)
    body += piece("$cpuDefaultHf =")
    for label, hf in (("default", "$cpuDefaultHf"), ("other", "'some/Repo-GGUF:Q4_0'")):
        body += ("$hf = %s\n" % hf + piece("$hfNote =")
                 + "Write-Host ('RESULT note.%s=' + $hfNote)\n" % label)
    for label, log in (("rot", old), ("fresh", logs / "llama-server-fresh-cpu-8080.log")):
        body += ("$log = %s\n" % ps_quote(log)
                 + piece("$prev =") + piece("$keptPrev =") + piece("foreach ($suffix in")
                 + "Write-Host ('RESULT kept.%s=' + $keptPrev)\n" % label)
    # Add-Quotes is a pure function; it lifts and runs with the rest.
    body += piece("function Add-Quotes")
    for label, value in (("plain", "C:\\models\\model.gguf"),
                         ("spaced", "C:\\my models\\Some Model-Q4_K_M.gguf"),
                         ("slots", "C:\\my models\\cache_slots\\gpu\\"),
                         ("slots-plain", "C:\\models\\cache_slots\\gpu\\"),
                         ("alias", "qwen3.5 9b gpu")):
        body += ("Write-Host ('RESULT quote.%s=[' + (Add-Quotes %s) + ']')\n"
                 % (label, ps_quote(value)))
    code, out = run_pieces(LLAMA_LAUNCHER, body)
    assert code == 0, out
    return out, logs


@needs_powershell
def test_llama_port_is_parsed_and_range_checked_with_a_run_line(llama):
    # LLAMA_PORT=abc used to die at a raw [int] cast with no [run] line.
    out, _ = llama
    got = results(out)
    assert got["port.abc"] == "8080"
    assert got["port.0"] == "8080"
    assert got["port.70000"] == "8080"
    assert got["port.8124"] == "8124"
    assert got["port."] == "8080"
    assert "[run] WARNING: LLAMA_PORT='abc' is not a port number (1-65535); using 8080." in \
        section(out, "port.abc")
    assert "WARNING" in section(out, "port.70000")
    assert "WARNING" not in section(out, "port.8124")
    assert "WARNING" not in section(out, "port.")


@needs_powershell
def test_llama_ipv6_literal_is_bracketed_for_the_probe_url(llama):
    # "http://::1:8124/health" does not parse; the probe threw every second
    # into an empty catch and a healthy server was killed at the deadline.
    got = results(llama[0])
    assert got["url.::1"] == "[::1]"
    assert got["url.fe80::1"] == "[fe80::1]"
    assert got["url.127.0.0.1"] == "127.0.0.1"
    assert got["url.0.0.0.0"] == "127.0.0.1"     # a wildcard bind is probed on loopback
    assert got["url.::"] == "127.0.0.1"
    # ...and both places a URL is built use it.
    text = LLAMA_LAUNCHER.read_text(encoding="utf-8")
    assert re.findall(r"http://\$\{(\w+)\}", text) == ["urlHost", "urlHost"]


@needs_powershell
def test_llama_gguf_wins_out_loud_and_warnings_follow_the_selected_source(llama):
    out, _ = llama
    got = results(out)
    both = section(out, "select.both")
    assert got["select.both.src"] == "C:\\models\\Other-Q4_0.gguf"
    assert got["select.both.hf"] == ""           # the dropped spec cannot reach -hf later
    assert "both LLAMA_HF and LLAMA_GGUF are set; using LLAMA_GGUF" in both
    assert "some/Repo-GGUF:Q8_0" in both         # names what was discarded
    # The discarded spec is a Q8_0; matching the concatenation used to warn
    # about a quant that was never going to be loaded.
    assert "Q8_0 model is selected" not in both
    # The same spec, selected, does warn -- so the silence above means something.
    assert got["select.hf-only.src"] == "some/Repo-GGUF:Q8_0"
    assert "Q8_0 model is selected" in section(out, "select.hf-only")
    assert "both LLAMA_HF" not in section(out, "select.hf-only")


@needs_powershell
def test_llama_log_is_named_for_the_selected_model(llama):
    got = results(llama[0])
    assert got["log.hf"] == "llama-server-Qwen3.5-9B-GGUF-Q4_0-gpu-8124.log"
    assert got["log.gguf"] == "llama-server-Some-Model-Q4_K_M-gpu-8124.log"


@needs_powershell
def test_llama_cache_ram_and_the_download_note(llama):
    got = results(llama[0])
    assert got["cache.default"] == "16384"
    assert got["cache.set"] == "4096"
    assert "5.4 GB" in got["note.default"]
    assert "5.4" not in got["note.other"]        # the figure is the cpu-leg default's only


@needs_powershell
def test_llama_previous_logs_survive_the_relaunch(llama):
    # -RedirectStandardOutput/-Error truncate on open, so the relaunch after a
    # crash destroyed the crash reason the launcher's own messages point at.
    out, logs = llama
    got = results(out)
    assert got["kept.rot"] == "True"
    assert got["kept.fresh"] == "False"
    prev = logs / "llama-server-rot-cpu-8080.prev.log"
    assert prev.read_text(encoding="ascii") == "old stdout"
    assert Path(str(prev) + ".err").read_text(encoding="ascii") == "old stderr: the crash reason"
    assert not (logs / "llama-server-rot-cpu-8080.log").exists()
    assert not (logs / "llama-server-rot-cpu-8080.log.err").exists()


@needs_powershell
def test_llama_quotes_a_spaced_path_and_survives_its_trailing_backslash(llama):
    # Both rules are probe-verified fixes for silent argv corruption, and the
    # values are operator-supplied paths (LLAMA_SLOT_DIR, LLAMA_GGUF,
    # LLAMA_ALIAS) -- or, unset, ones derived from the checkout's own location,
    # so a clone under "C:\my models" hits this with no config at all.
    got = results(llama[0])
    # No whitespace, nothing to fix: quoting everything would put literal quote
    # characters into argv entries that never needed them.
    assert got["quote.plain"] == "[C:\\models\\model.gguf]"
    # A space shatters an unquoted path into two argv entries (PS 5.1 joins
    # -ArgumentList with spaces and no quoting).
    assert got["quote.spaced"] == '["C:\\my models\\Some Model-Q4_K_M.gguf"]'
    # ...and a quoted path ENDING in a backslash reaches the child CRT as an
    # escaped quote: the value never closes and every later flag -- ctx-size,
    # the samplers, --reasoning off -- is swallowed into --slot-save-path.
    # Doubling the trailing run is what closes it.
    assert got["quote.slots"] == '["C:\\my models\\cache_slots\\gpu\\\\"]'
    # The doubling exists only because of the quote; unquoted it would corrupt
    # a path that was already fine.
    assert got["quote.slots-plain"] == "[C:\\models\\cache_slots\\gpu\\]"
    assert got["quote.alias"] == '["qwen3.5 9b gpu"]'


@needs_powershell
def test_llama_every_operator_supplied_path_reaches_argv_through_add_quotes(llama):
    # The function is only worth what its call sites are: --slot-save-path, the
    # alias and the gguf path are the three values an operator can put a space
    # in, and each is documented in docs/MODEL_OPTIONS.md as an env var.
    text = LLAMA_LAUNCHER.read_text(encoding="utf-8")
    assert re.findall(r"\(Add-Quotes \$(\w+)\)", text) == ["slotDir", "alias", "gguf"]


@needs_powershell
def test_llama_launcher_writes_no_env_vars(llama):
    # Its header promises "INPUTS ONLY -- this script never writes them back".
    assert [line for line in llama[0].splitlines() if line.startswith("WRITE ")] == []


# --- run-llama-server.ps1: the port it refuses to bind -----------------------------

def listener_table(cases, port="8080", holder=None):
    """Run the launcher's real listener check against a scripted table.

    `cases` is [(label, the address we are about to bind, [(listening address,
    pid), ...]), ...]. Get-NetTCPConnection is shadowed by a function -- a
    PowerShell function outranks a cmdlet, the same trick supervise() uses for
    Get-Date -- so no socket is opened and whatever this box happens to be
    serving cannot decide the result. [CmdletBinding()] is what makes the
    launcher's -ErrorAction bind to a function.

    Get-CimInstance, which the refusal asks who the holder is, is shadowed the
    same way, ALWAYS: the scripted pid is a real pid on this box as often as
    not, and a real process must not decide what the refusal says. `holder` is
    (process name, command line) for the one it answers with, "throw" for a
    lookup that fails, or None for a pid with no process behind it (it exited).

    The refusal ends in `exit 1`, which ends the harness too, so a case that
    conflicts must be the last one in its call.
    """
    if holder == "throw":
        answer = "    throw 'Access denied'\n"
    elif holder is None:
        answer = ""
    else:
        answer = ("    [pscustomobject]@{ ProcessId = 0; Name = %s; CommandLine = %s }\n"
                  % (ps_quote(holder[0]), ps_quote(holder[1]) if holder[1] is not None else "$null"))
    body = ("function Get-CimInstance {\n"
            "    [CmdletBinding()] param([string]$ClassName, [string]$Filter)\n"
            "    Write-Host \"STUB: Get-CimInstance $ClassName $Filter\"\n"
            + answer + "}\n")
    for label, bind, rows in cases:
        table = "".join(
            "    [pscustomobject]@{ LocalAddress = %s; LocalPort = %s; OwningProcess = %d }\n"
            % (ps_quote(addr), port, pid) for addr, pid in rows)
        body += ("Write-Host 'BEGIN %s'\n" % label
                 + "function Get-NetTCPConnection {\n"
                 "    [CmdletBinding()] param([int]$LocalPort, [string]$State)\n"
                 "    @(\n" + table + "    )\n}\n"
                 + "$bindHost = %s\n$port = %s\n" % (ps_quote(bind), ps_quote(port))
                 + piece("$wildcards =") + piece("$conflict = Get-NetTCPConnection")
                 + piece("if ($conflict)")
                 + "Write-Host 'RESULT bound.%s=yes'\n" % label
                 + "Write-Host 'END %s'\n" % label)
    return run_pieces(LLAMA_LAUNCHER, body)


@needs_powershell
@pytest.mark.parametrize("label, bind, addr", [
    # Our exact address, the shape a loopback dial would also have caught.
    ("exact", "127.0.0.1", "127.0.0.1"),
    # A wildcard listener owns our address too -- and "::" is in $wildcards
    # because a v6 wildcard on Windows takes the v4 loopback with it.
    ("listener-wildcard", "127.0.0.1", "::"),
    # We are the wildcard: a listener on ONE interface is ours as well, which
    # is the shape genie_server's loopback dial cannot see at all.
    ("we-are-wildcard", "0.0.0.0", "192.168.1.10"),
])
def test_llama_refuses_a_port_that_something_is_already_serving(label, bind, addr):
    # On Windows two processes can both hold a port and the OLD one keeps
    # answering, so a clean startup log proves nothing about who the requests
    # reach -- it happened here (GENIE_SERVER.md), and it silently attributes
    # one engine's numbers to another.
    code, out = listener_table([(label, bind, [(addr, 9876)])],
                               holder=("llama-server.exe", "llama-server.exe -hf some/Repo-GGUF:Q4_0 --port 8080"))
    assert code == 1, out
    assert ("[run] something is already listening on %s:8080 (pid 9876) "
            "-- refusing to double-bind." % addr) in out
    assert "RESULT bound.%s=yes" % label not in out, "it must not start anyway"
    # Named, so the operator can tell whose it is -- and the lookup is for
    # THAT pid.
    assert "STUB: Get-CimInstance Win32_Process ProcessId = 9876" in out
    assert ("[run] pid 9876 is llama-server.exe -- llama-server.exe -hf some/Repo-GGUF:Q4_0 "
            "--port 8080") in out
    # The way round that harms nobody comes first; a kill is the operator's
    # own call. It used to print a ready-to-paste `Stop-Process 9876` for a
    # pid it had never looked at, and on this shared box the likeliest holder
    # of 8080 is another session's llama-server.
    assert "Stop-Process" not in out
    assert "LLAMA_PORT" in out
    assert "another session's server" in out
    assert "only once you have confirmed it is yours" in out


@needs_powershell
@pytest.mark.parametrize("holder, said", [
    # The pid has no process behind it any more (Win32_Process answers nothing).
    (None, "[run] pid 9876: its name could not be read"),
    # The lookup itself fails -- the refusal must still be a refusal, with its
    # own words, not a CIM error in place of them.
    ("throw", "[run] pid 9876: its name could not be read"),
    # Another user's or an elevated process: the name reads, the command line
    # does not. Say what can be said.
    (("svchost.exe", None), "[run] pid 9876 is svchost.exe\n"),
    # A long command line is cut, not dumped: this is one line of a refusal.
    (("llama-server.exe", "llama-server.exe " + "x" * 400),
     "[run] pid 9876 is llama-server.exe -- llama-server.exe " + "x" * 283 + " ...\n"),
])
def test_llama_names_a_holder_it_cannot_fully_identify_without_offering_a_kill(holder, said):
    code, out = listener_table([("exact", "127.0.0.1", [("127.0.0.1", 9876)])], holder=holder)
    assert code == 1, out
    assert said in out.replace("\r\n", "\n"), out
    assert "Stop-Process" not in out
    assert "only once you have confirmed it is yours" in out


@needs_powershell
def test_llama_binds_past_a_listener_it_cannot_collide_with():
    # The other half: refusing on any listener at all would make the launcher
    # unusable on a box serving that port on a real interface, and "nothing
    # listening" must not read as a conflict either -- one shares a process
    # because neither reaches the `exit 1`.
    code, out = listener_table([
        ("elsewhere", "127.0.0.1", [("192.168.1.10", 9876)]),
        ("nothing", "127.0.0.1", []),
    ])
    assert code == 0, out
    assert results(out) == {"bound.elsewhere": "yes", "bound.nothing": "yes"}
    assert "double-bind" not in out


# --- run-llama-server.ps1: startup failure and the placement report ----------------

def startup_failure(has_exited, exit_code, log="C:\\logs\\x.log", hf=None, setup=""):
    """Run the launcher's real startup-failure block over a scripted child.

    `log` is the log path the block reads (its .err is where a refused resume
    shows up); `hf` the -hf spec, if the model came from one; `setup` any
    PowerShell to run first, such as pointing the HF cache somewhere.
    """
    body = ("function Drain-Logs { }\nfunction Flush-Carry { }\n"
            "function Stop-Process { param($Id, [switch]$Force, [switch]$Confirm)\n"
            "    Write-Host \"STUB: Stop-Process $Id\" }\n"
            + setup + piece("function Get-HfPartials")
            + "$log = %s; $timeout = 1800; $up = $false\n$hf = %s\n"
            "$proc = [pscustomobject]@{ HasExited = $%s; ExitCode = %d; Id = 4242 }\n"
            % (ps_quote(log), ps_quote(hf) if hf else "$null",
               "true" if has_exited else "false", exit_code)
            + piece("if (-not $up)") + "Write-Host 'RESULT fell=through'\n")
    return run_pieces(LLAMA_LAUNCHER, body)


@needs_powershell
def test_llama_a_server_that_dies_on_load_exits_with_its_own_code():
    # Both startup failures used to leave as 1, so an outer script could not
    # tell a crash-on-load from an expired health timeout.
    code, out = startup_failure(True, 3)
    assert code == 3, out
    assert "llama-server exited 3 during startup" in out
    assert "STUB: Stop-Process" not in out


@needs_powershell
def test_llama_a_health_timeout_stops_the_server_and_exits_1():
    code, out = startup_failure(False, 0)
    assert code == 1, out
    assert "server not healthy after 1800s (LLAMA_HEALTH_TIMEOUT" in out
    assert "STUB: Stop-Process 4242" in out


@needs_powershell
def test_llama_placement_reports_what_it_saw_and_names_what_it_could_not(tmp_path):
    # The cpu leg never passes -lv 5, so the device line cannot exist there;
    # the old report asserted "CPU-only load (KleidiAI)" on every cpu start,
    # against a real log on this box that shows the OpenCL backend engaged.
    def write(name, err_text):
        log = tmp_path / (name + ".log")
        log.write_text("", encoding="ascii")
        Path(str(log) + ".err").write_text(err_text, encoding="ascii")
        return log

    cases = (
        ("cpu-opencl", "cpu", write("a", "srv  load: verbosity = 3\nggml_opencl: backend=OpenCL\n")),
        ("cpu-plain", "cpu", write("b", "srv  load: verbosity = 3\n")),
        ("gpu-placed", "gpu", write("c", "llama_model_load: using device GPUOpenCL (Adreno)\n")),
        ("gpu-missing", "gpu", write("d", "srv  load: verbosity = 5\n")),
        ("unreadable", "cpu", tmp_path / "no-such-dir" / "e.log"),
        # The gpu leg's log naming a DIFFERENT device: the build answered, at
        # CPU speed, on the quant this repo measures at half speed there.
        # Reachable without a broken build -- LLAMA_EXTRA_ARGS is appended
        # AFTER the leg's own --device GPUOpenCL, so it is last-wins.
        ("gpu-other-device", "gpu",
         write("f", "llama_model_load: using device CPU (fallback)\n")),
        ("gpu-unreadable", "gpu", tmp_path / "no-such-dir" / "g.log"),
    )
    report = (piece("function Find-InLog") + piece("$logReadError = $null") + piece("$dev =")
              + piece("if ($logReadError)") + piece('if ($Leg -eq "gpu") {'))
    body = ""
    for label, leg, log in cases:
        body += ("Write-Host 'BEGIN %s'\n$Leg = '%s'\n$log = %s\n" % (label, leg, ps_quote(log))
                 + report + "Write-Host 'END %s'\n" % label)
    code, out = run_pieces(LLAMA_LAUNCHER, body)
    assert code == 0, out
    assert "CPU-only" not in out

    cpu_opencl = section(out, "cpu-opencl")
    assert "placement: not checked -- the device line needs -lv 5" in cpu_opencl
    assert "The log mentions the OpenCL backend" in cpu_opencl
    assert "--device none" in cpu_opencl

    cpu_plain = section(out, "cpu-plain")
    assert "placement: not checked" in cpu_plain
    assert "OpenCL backend" not in cpu_plain

    gpu_placed = section(out, "gpu-placed")
    assert "[run] placement: llama_model_load: using device GPUOpenCL (Adreno)" in gpu_placed
    assert "WARNING" not in gpu_placed

    gpu_missing = section(out, "gpu-missing")
    assert "placement: no 'using device' line found." in gpu_missing
    assert "WARNING: gpu leg requested but no 'using device GPUOpenCL' in the log." in gpu_missing

    # A log that cannot be read is its own verdict, not "no line found".
    unreadable = section(out, "unreadable")
    assert "placement: could not read the log -- " in unreadable
    assert "not checked" not in unreadable

    # The warning is what separates the right engine from the wrong one, and
    # the only thing that can say so is the device NAME: a gpu leg whose log
    # names a device is still not a gpu leg unless that device is GPUOpenCL.
    other = section(out, "gpu-other-device")
    assert "[run] placement: llama_model_load: using device CPU (fallback)" in other
    assert "WARNING: gpu leg requested but no 'using device GPUOpenCL' in the log." in other
    assert "Q4_K_M" in other

    # Pinning today's behaviour, not blessing it: on the gpu leg an unreadable
    # log gets BOTH lines -- the honest "could not read" and then a claim about
    # what is not in a file the launcher never opened. The warning is the
    # cautious way round (unverified placement is warned about, not waved
    # through), so it stays; the second line's wording is the part to revisit.
    gpu_unreadable = section(out, "gpu-unreadable")
    assert "placement: could not read the log -- " in gpu_unreadable
    assert "WARNING: gpu leg requested but no 'using device GPUOpenCL' in the log." in gpu_unreadable


# --- run-genie-server.ps1: the bundle dir must BE a bundle ------------------------

def bundle_preflight(bundle_dir, model="qwen3-4b", explicit=False):
    """Run the launcher's real genie_config.json pre-flight against `bundle_dir`."""
    body = (piece("$Bundles = @{")
            + "$Model = %s\n$modelExplicit = $%s\n" % (ps_quote(model), "true" if explicit else "false")
            + piece("$DefaultBundle =")
            + "$env:GENIE_BUNDLE_DIR = %s\n" % ps_quote(bundle_dir)
            + piece("$bundleConfig =") + piece("if (-not (Test-Path -LiteralPath $bundleConfig")
            + "Write-Host 'RESULT reached=the-end'\n")
    return run_pieces(LAUNCHER, body)


def make_bundle(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "genie_config.json").write_text("{}", encoding="ascii")
    return path


@needs_powershell
def test_a_real_bundle_dir_passes_the_preflight(tmp_path):
    code, out = bundle_preflight(make_bundle(tmp_path / "some-bundle"))
    assert code == 0, out
    assert results(out) == {"reached": "the-end"}
    assert "[run]" not in out


@needs_powershell
def test_the_directory_above_the_bundles_is_refused_naming_the_ones_below(tmp_path):
    # The likeliest wrong value exists as a directory, so the existence check
    # passed it and the server died in load_engine with a bare FileNotFoundError
    # traceback -- after Genie.dll had loaded. `qai-hub-models fetch --extract
    # -o <dir>` puts the bundle in a subdirectory it names itself, so <dir>, or
    # bundles\, is exactly one level off; the fix is the directories below.
    above = tmp_path / "bundles"
    one = make_bundle(above / "qwen3_4b-a")
    two = make_bundle(above / "qwen3_8b-b")
    (above / "not-a-bundle").mkdir()
    code, out = bundle_preflight(above)
    assert code == 1, out
    assert "reached" not in results(out)
    assert "[run] GENIE_BUNDLE_DIR has no genie_config.json: %s" % above in out
    assert "It must be ONE bundle directory" in out
    assert "One level down, these hold one:" in out
    lines = out.splitlines()
    assert "        %s" % one in lines and "        %s" % two in lines
    assert "not-a-bundle" not in out
    assert "Set GENIE_BUNDLE_DIR to the one you mean." in out


@needs_powershell
def test_under_model_the_nested_bundle_fix_is_one_the_next_run_keeps(tmp_path):
    # -Model rewrites GENIE_BUNDLE_DIR on every run, so "set GENIE_BUNDLE_DIR"
    # is advice the next run undoes; the fix under -Model is to move the files.
    bundle_dir = tmp_path / "bundles" / BUNDLE_8B
    make_bundle(bundle_dir / "qwen3_8b-extracted")
    code, out = bundle_preflight(bundle_dir, model="qwen3-8b", explicit=True)
    assert code == 1, out
    assert "-Model qwen3-8b always serves <root>\\bundles\\%s itself" % BUNDLE_8B in out
    assert "move" in out and "drop -Model" in out
    assert "Set GENIE_BUNDLE_DIR to the one you mean." not in out


@needs_powershell
@pytest.mark.parametrize("shape", ["empty", "config-is-a-directory"])
def test_a_dir_with_no_config_anywhere_near_is_refused(tmp_path, shape):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    if shape == "config-is-a-directory":
        (bundle_dir / "genie_config.json").mkdir()     # a name is not a file
    code, out = bundle_preflight(bundle_dir)
    assert code == 1, out
    assert "[run] GENIE_BUNDLE_DIR has no genie_config.json" in out
    assert "No directory one level down holds one either" in out


@needs_powershell
def test_the_preflight_runs_before_python_is_even_looked_for(tmp_path):
    # Whole-script, on the default route: the launcher's own
    # <root>\bundles\<default bundle>, extracted one level too deep. The
    # interpreter does not exist, so "python not found" is how far a run gets
    # once the bundle is right -- the control that proves the refusal came
    # first rather than instead.
    root = tmp_path / "npu-root"
    nested = make_bundle(root / "bundles" / BUNDLE_4B / "qwen3_4b-extracted")
    (root / "qairt" / "2.45.0.1").mkdir(parents=True)
    env = {"GENIE_NPU_ROOT": str(root), "GENIE_PYTHON": "no-such-python-for-this-test"}
    code, out, err = run_launcher(LAUNCHER, [], env)
    assert code == 1, out + err
    assert "[run] GENIE_BUNDLE_DIR has no genie_config.json: %s" % (root / "bundles" / BUNDLE_4B) in out
    assert "        %s" % nested in out.splitlines()
    assert "python not found" not in out
    code, out, err = run_launcher(LAUNCHER, [], dict(env, GENIE_BUNDLE_DIR=str(nested)))
    assert code == 1, out + err
    assert "has no genie_config.json" not in out
    assert "[run] python not found: 'no-such-python-for-this-test'" in out


# --- run-llama-server.ps1: a download llama.cpp can never resume ------------------

HF_SPEC = "unsloth/Qwen3.5-9B-GGUF:Q4_0"
HF_REPO_DIR = "models--unsloth--Qwen3.5-9B-GGUF"
# Where llama.cpp looks for its HF cache, in its own order (common/hf-cache.cpp).
HF_CACHE_VARS = ("LLAMA_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                 "HF_HOME", "XDG_CACHE_HOME", "USERPROFILE")
BLOB = "17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24"
RESUME_REFUSED = ("common_pull_file: server did not respond with 206 Partial Content "
                  "for a resume request. Status: 416\n")


def hf_cache_env(**values):
    """PowerShell that sets exactly these HF cache variables and unsets the rest,
    so the real cache on this box can never decide a result."""
    return "".join("$env:%s = %s\n" % (name, ps_quote(values[name]) if name in values else "$null")
                   for name in HF_CACHE_VARS)


def partial_download(hub, repo_dir=HF_REPO_DIR, size=4096, suffix=".downloadInProgress"):
    blobs = hub / repo_dir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    blob = blobs / (BLOB + suffix)
    blob.write_bytes(b"\0" * size)
    return blob


@needs_powershell
def test_llama_finds_partial_downloads_where_llama_cpp_would_resume_them(tmp_path):
    # llama.cpp resolves its cache LLAMA_CACHE first, then HF_HUB_CACHE,
    # HUGGINGFACE_HUB_CACHE, HF_HOME\hub, XDG_CACHE_HOME\huggingface\hub and
    # USERPROFILE\.cache\huggingface\hub; a partial is blobs\<etag> plus
    # .downloadInProgress. Looking anywhere else names the wrong file, or none.
    cases = []
    lc, hh = tmp_path / "lc", tmp_path / "hh"
    found_lc = partial_download(lc)
    partial_download(hh / "hub")      # a decoy: LLAMA_CACHE wins over HF_HOME
    cases.append(("llama-cache", HF_SPEC, {"LLAMA_CACHE": str(lc), "HF_HOME": str(hh)},
                  found_lc, lc / HF_REPO_DIR / "blobs"))
    hc = tmp_path / "hc"
    cases.append(("hf-hub-cache", HF_SPEC, {"HF_HUB_CACHE": str(hc)}, partial_download(hc),
                  hc / HF_REPO_DIR / "blobs"))
    cases.append(("hf-home", HF_SPEC, {"HF_HOME": str(hh)}, hh / "hub" / HF_REPO_DIR / "blobs" /
                  (BLOB + ".downloadInProgress"), hh / "hub" / HF_REPO_DIR / "blobs"))
    xdg = tmp_path / "xdg"
    cases.append(("xdg", HF_SPEC, {"XDG_CACHE_HOME": str(xdg)},
                  partial_download(xdg / "huggingface" / "hub"),
                  xdg / "huggingface" / "hub" / HF_REPO_DIR / "blobs"))
    up = tmp_path / "up"
    cases.append(("userprofile", HF_SPEC, {"USERPROFILE": str(up)},
                  partial_download(up / ".cache" / "huggingface" / "hub"),
                  up / ".cache" / "huggingface" / "hub" / HF_REPO_DIR / "blobs"))
    done = tmp_path / "done"
    partial_download(done, suffix="")            # a FINISHED blob is not a partial
    cases.append(("finished-only", HF_SPEC, {"LLAMA_CACHE": str(done)}, None,
                  done / HF_REPO_DIR / "blobs"))
    other = tmp_path / "other"
    partial_download(other, repo_dir="models--someone--Other-GGUF")
    cases.append(("other-repo", HF_SPEC, {"LLAMA_CACHE": str(other)}, None,
                  other / HF_REPO_DIR / "blobs"))
    cases.append(("no-spec", "", {"LLAMA_CACHE": str(lc)}, None, None))
    body = piece("function Get-HfPartials")
    for label, spec, env, _, _ in cases:
        body += (hf_cache_env(**env)
                 + "$__p = @(Get-HfPartials %s)\n" % ps_quote(spec)
                 + "Write-Host ('RESULT %s.found=' + (($__p | ForEach-Object { $_.FullName }) -join '|'))\n" % label
                 + "Write-Host ('RESULT %s.blobs=' + $hfBlobs)\n" % label)
    code, out = run_pieces(LLAMA_LAUNCHER, body)
    assert code == 0, out
    got = results(out)
    for label, _, _, found, blobs in cases:
        assert got[label + ".found"] == (str(found) if found else ""), (label, out)
        assert got[label + ".blobs"] == (str(blobs) if blobs else ""), (label, out)


@needs_powershell
def test_llama_names_an_unfinished_download_before_the_start(tmp_path):
    # Before the start, every partial is named with its size: the full size
    # cannot be known offline (the blob is named for its hash), so the note
    # says what to look for rather than guessing -- and says it only for an
    # -hf source, the one llama-server will try to resume.
    cache = tmp_path / "cache"
    blob = partial_download(cache, size=5000)
    note = piece("$hfPartials =") + piece("if ($hfPartials.Count")
    body = piece("function Get-HfPartials") + hf_cache_env(LLAMA_CACHE=str(cache))
    for label, hf in (("partial", HF_SPEC), ("gguf", None)):
        body += ("Write-Host 'BEGIN %s'\n$hf = %s\n" % (label, ps_quote(hf) if hf else "$null")
                 + note + "Write-Host 'END %s'\n" % label)
    body += (hf_cache_env(LLAMA_CACHE=str(tmp_path / "empty-cache"))
             + "Write-Host 'BEGIN clean'\n$hf = %s\n" % ps_quote(HF_SPEC) + note + "Write-Host 'END clean'\n")
    code, out = run_pieces(LLAMA_LAUNCHER, body)
    assert code == 0, out
    seen = section(out, "partial")
    assert "[run] note: the HF cache holds an unfinished download for unsloth/Qwen3.5-9B-GGUF:" in seen
    assert "[run]   %s  (5000 bytes, last written 20" % blob in seen
    assert "Status: 416" in seen
    assert section(out, "gguf") == "", "a local GGUF is never resumed from the cache"
    assert section(out, "clean") == ""


@needs_powershell
def test_llama_a_refused_resume_is_named_with_its_file_and_nothing_is_deleted(tmp_path):
    # HTTP 416 on a resume means the partial is already at or past the full
    # size -- on this box one was 277 MiB OVER, two downloads having appended
    # to it at once -- and this llama.cpp build retries it unchanged, so the
    # leg died the same way on every run while the launcher said only "see its
    # stderr above". Now the file is named, with what to do; and the launcher
    # does not do it: the cache is shared, and another session may be
    # mid-download.
    cache = tmp_path / "cache"
    blob = partial_download(cache, size=6000)
    log = tmp_path / "llama.log"
    Path(str(log) + ".err").write_text("load_model: loading\n" + RESUME_REFUSED * 3
                                       + "download failed after 3 attempts\n", encoding="ascii")
    code, out = startup_failure(True, 1, log=log, hf=HF_SPEC,
                                setup=hf_cache_env(LLAMA_CACHE=str(cache)))
    assert code == 1, out
    assert "llama-server exited 1 during startup" in out
    assert "[run] cause: llama-server could not RESUME a partial download. HTTP 416" in out
    assert "at or past the full file's size" in out
    assert "[run]   %s  (6000 bytes)" % blob in out
    assert "The launcher deletes nothing." in out
    assert "move that file aside (or delete it)" in out
    assert "Do not just drop the" in out
    # Detection only: the file is exactly as it was.
    assert blob.exists() and blob.stat().st_size == 6000


@needs_powershell
def test_llama_a_refused_resume_with_no_partial_in_sight_says_where_it_looked(tmp_path):
    log = tmp_path / "llama.log"
    Path(str(log) + ".err").write_text(RESUME_REFUSED, encoding="ascii")
    cache = tmp_path / "cache"
    code, out = startup_failure(True, 1, log=log, hf=HF_SPEC,
                                setup=hf_cache_env(LLAMA_CACHE=str(cache)))
    assert code == 1, out
    assert "cause: llama-server could not RESUME" in out
    assert "(no *.downloadInProgress found in %s" % (cache / HF_REPO_DIR / "blobs") in out


@needs_powershell
def test_llama_any_other_startup_death_gets_no_resume_diagnosis(tmp_path):
    log = tmp_path / "llama.log"
    Path(str(log) + ".err").write_text("llama_model_load: error loading model\n", encoding="ascii")
    partial_download(tmp_path / "cache")     # a partial exists, but nothing refused it
    code, out = startup_failure(True, 3, log=log, hf=HF_SPEC,
                                setup=hf_cache_env(LLAMA_CACHE=str(tmp_path / "cache")))
    assert code == 3, out
    assert "cause:" not in out


# What the partial-download code may call: it reads and reports, and nothing it
# runs can move, delete or rewrite a file -- the pin that holds the "detection
# only" promise as the code around it changes.
HF_EFFECTS = r"""
$__scopes = @($__ast.FindAll({ param($n)
        ($n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-HfPartials') -or
        ($n -is [System.Management.Automation.Language.IfStatementAst] -and
         @('$hfPartials.Count -gt 0', '$resumeRefused') -contains $n.Clauses[0].Item1.Extent.Text) }, $true))
Write-Host ("RESULT scopes=" + $__scopes.Count)
foreach ($s in $__scopes) {
    foreach ($c in $s.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true)) {
        Write-Host ("CALLS " + $c.GetCommandName())
    }
}
"""


@needs_powershell
def test_llama_the_partial_download_code_can_only_look():
    code, out = run_pieces(LLAMA_LAUNCHER, HF_EFFECTS)
    assert code == 0, out
    assert results(out)["scopes"] == "3", out
    calls = {line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("CALLS ")}
    assert "Get-ChildItem" in calls, "the scan is what is being pinned; it must be found"
    assert calls <= {"Get-ChildItem", "Get-HfPartials", "Write-Host", "ForEach-Object"}, calls


# --- whole-script runs: help, stray arguments, a missing binary --------------------

def launcher_env(tmp_path, launcher):
    """An environment in which `launcher`, run whole, stops at its first check.

    run-genie-server.ps1: a GENIE_NPU_ROOT with nothing in it, bundle and SDK
    dirs that do not exist, and a GENIE_PYTHON that is no program at all -- so
    a run that gets past what it is meant to stop at exits 1 at the bundle
    check, and could not start python even past that. run-llama-server.ps1:
    the same root and a LLAMA_BIN_DIR with no llama-server.exe in it.
    """
    root = tmp_path / "npu-root"
    root.mkdir(exist_ok=True)
    env = {"GENIE_NPU_ROOT": str(root)}
    if launcher == LAUNCHER:
        env.update(GENIE_BUNDLE_DIR=str(root / "no-bundle"), GENIE_SDK_DIR=str(root / "no-sdk"),
                   GENIE_PYTHON="no-such-python-for-this-test")
    else:
        env.update(LLAMA_BIN_DIR=str(tmp_path / "no-bin"))
    return env


def run_launcher(launcher, args, env, in_shell=False):
    """Run the WHOLE launcher; returns (exit code, stdout, stderr).

    `powershell -File` is the documented route. `in_shell` runs
    `& '<launcher>' <args>` inside a PowerShell instead, which binds some
    spellings differently (see the launchers' headers). GENIE_* / LLAMA_* are
    stripped from the inherited environment first, as in run_pieces, so `env`
    -- normally from launcher_env() -- is all the launcher sees of them.
    """
    assert "GENIE_NPU_ROOT" in env, "never run a launcher whole against the real root"
    child_env = {k: v for k, v in os.environ.items()
                 if not k.upper().startswith(("GENIE_", "LLAMA_"))}
    child_env.update(env)
    if in_shell:
        script = "& %s %s\nexit $LASTEXITCODE\n" % (ps_quote(launcher), " ".join(args))
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        cmd = [POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
    else:
        cmd = [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", str(launcher), *args]
    done = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=120, env=child_env)
    return done.returncode, done.stdout, done.stderr


# Every environment variable each launcher reads, from its code: the usage has
# to mention all of them, so a new knob cannot ship undocumented at -Help.
def env_vars_read(launcher):
    code = launcher_code(launcher)
    return (set(re.findall(r"\$env:((?:GENIE|LLAMA)_\w+)", code))
            | set(re.findall(r'Get-EnvInt "(GENIE_\w+)"', code)))


HELP_SPELLINGS = [["-Help"], ["-h"], ["-help"], ["--help"]]


@needs_powershell
@pytest.mark.parametrize("args", [*HELP_SPELLINGS, ["-Model", "qwen3-8b", "-h"]], ids=" ".join)
def test_genie_help_prints_usage_and_starts_nothing(tmp_path, args):
    # Every one of these used to start the server -- on a provisioned box, an
    # 11-35 s bundle load onto an NPU another session may be benchmarking.
    # --help counts because `powershell -File` hands it over as -help.
    code, out, err = run_launcher(LAUNCHER, args, launcher_env(tmp_path, LAUNCHER))
    assert code == 0, out + err
    assert out.startswith("usage: powershell -File src\\run-genie-server.ps1 [-Model <name>] [-Help]"), out
    # Each -Model, with the bundle and id it serves, read out of $Bundles.
    for name, bundle, model_id in (("qwen3-4b", BUNDLE_4B, "qwen3-4b-npu"),
                                   ("qwen3-8b", BUNDLE_8B, "qwen3-8b-npu"),
                                   ("qwen3-8b-8192", BUNDLE_8192, "qwen3-8b-8192-npu")):
        assert re.search(r"^ +%s +%s -> %s" % (re.escape(name), re.escape(bundle), re.escape(model_id)),
                         out, re.M), name
    for var in sorted(env_vars_read(LAUNCHER)):
        assert var in out, "%s is read by the launcher but missing from -Help" % var
    assert "docs/GENIE_SERVER.md" in out
    # Nothing past the usage: no bundle check, no discovery, no interpreter.
    assert "[run]" not in out


@needs_powershell
@pytest.mark.parametrize("args", HELP_SPELLINGS, ids=" ".join)
def test_llama_help_prints_usage_and_starts_nothing(tmp_path, args):
    code, out, err = run_launcher(LLAMA_LAUNCHER, args, launcher_env(tmp_path, LLAMA_LAUNCHER))
    assert code == 0, out + err
    assert out.startswith("usage: powershell -File src\\run-llama-server.ps1 [-Leg cpu|gpu] [-Help]"), out
    for var in sorted(env_vars_read(LLAMA_LAUNCHER)):
        assert var in out, "%s is read by the launcher but missing from -Help" % var
    assert "-Leg cpu" in out and "-Leg gpu" in out
    assert "docs/MODEL_OPTIONS.md" in out
    assert "[run]" not in out


@needs_powershell
@pytest.mark.parametrize("launcher, synopsis", [
    (LAUNCHER, "Launch the Genie NPU OpenAI-compatible server"),
    (LLAMA_LAUNCHER, "Launch a Qwen3.5-9B llama-server leg"),
], ids=["genie", "llama"])
def test_question_mark_is_powershells_own_help_and_never_runs_the_script(tmp_path, launcher, synopsis):
    # -? never reaches a script that has a comment-based help block (or
    # [CmdletBinding()]): PowerShell answers it. Without either, it fell into
    # $args and the launcher ran -- a reviewer's in-shell `-?` loaded the real
    # bundle onto the NPU that way.
    env = launcher_env(tmp_path, launcher)
    code, out, err = run_launcher(launcher, ["-?"], env, in_shell=True)
    flat = "".join(out.split())             # Get-Help wraps at the console width
    assert "".join(synopsis.split()) in flat, out + err
    assert "[-Help]" in flat, "the switch shows in the SYNTAX line"
    assert "[run]" not in out
    # Under -File, PowerShell prints that help to a console only; with stdout
    # captured it prints nothing at all (probe-verified on 5.1). What matters
    # is the same: exit 0 and nothing started.
    code, out, err = run_launcher(launcher, ["-?"], env)
    assert code == 0, out + err
    assert "[run]" not in out


@needs_powershell
@pytest.mark.parametrize("launcher, args, named", [
    # A typo'd parameter name used to land in $args unremarked: `-Modle
    # qwen3-8b` served the 4B, and a stray switch started the server.
    (LAUNCHER, ["-Modle", "qwen3-8b"], "Modle"),
    (LAUNCHER, ["--hlep"], "hlep"),
    # /? is not a PowerShell help spelling; it is a positional value, and the
    # ValidateSet refuses it with the valid names.
    (LAUNCHER, ["/?"], "qwen3-4b,qwen3-8b,qwen3-8b-8192"),
    (LLAMA_LAUNCHER, ["-Legg", "gpu"], "Legg"),
], ids=["genie-typo", "genie-stray", "genie-slash", "llama-typo"])
def test_a_stray_argument_is_a_binding_error_not_a_default_run(tmp_path, launcher, args, named):
    code, out, err = run_launcher(launcher, args, launcher_env(tmp_path, launcher))
    assert code != 0, out + err
    # PowerShell wraps its error text at the console width, mid-word.
    assert named in "".join((out + err).split()), out + err
    assert "[run]" not in out, "the launcher ran on past an argument it did not understand"


# Where the -Help branch sits, against everything that reads or changes state:
# every env write, the try whose finally restores them, every filesystem,
# network and process call, and the interpreter itself. All of them come AFTER
# the branch ends, so asking for usage cannot touch anything.
HELP_FIRST = r"""
$__help = @($__ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.IfStatementAst] -and
        $n.Clauses[0].Item1.Extent.Text -eq '$Help' }, $true))
Write-Host ("RESULT helps=" + $__help.Count)
Write-Host ("RESULT help-end=" + $__help[0].Extent.EndOffset)
$__watch = @('New-Item', 'Move-Item', 'Remove-Item', 'Set-Item', 'Start-Process', 'Stop-Process',
             'Get-ChildItem', 'Get-Command', 'Test-Path', 'Get-NetTCPConnection',
             'Get-CimInstance', 'Select-String', 'Invoke-WebRequest')
foreach ($c in $__ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true)) {
    $__name = $c.GetCommandName()
    if ($c.InvocationOperator -eq 'Ampersand' -and $c.CommandElements[0].Extent.Text -eq '$python') {
        Write-Host ("EFFECT " + $c.Extent.StartOffset + " python")
    } elseif ($__watch -contains $__name) {
        Write-Host ("EFFECT " + $c.Extent.StartOffset + " " + $__name)
    }
}
foreach ($w in $__ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $n.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
        $n.Left.VariablePath.DriveName -eq 'env' }, $true)) {
    Write-Host ("EFFECT " + $w.Extent.StartOffset + " env-write")
}
foreach ($t in $__ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.TryStatementAst] }, $true)) {
    Write-Host ("EFFECT " + $t.Extent.StartOffset + " try")
}
"""


@needs_powershell
@pytest.mark.parametrize("launcher, must_see", [
    (LAUNCHER, {"env-write", "try", "python", "Get-ChildItem", "Get-Command", "Test-Path"}),
    (LLAMA_LAUNCHER, {"New-Item", "Move-Item", "Start-Process", "Test-Path", "Get-NetTCPConnection"}),
], ids=["genie", "llama"])
def test_the_help_branch_comes_before_anything_that_reads_or_changes_state(launcher, must_see):
    code, out = run_pieces(launcher, HELP_FIRST)
    assert code == 0, out
    got = results(out)
    assert got["helps"] == "1", out
    help_end = int(got["help-end"])
    effects = [line.split(" ", 2)[1:] for line in out.splitlines() if line.startswith("EFFECT ")]
    kinds = {kind for _, kind in effects}
    assert must_see <= kinds, "the scan lost track of %s" % (must_see - kinds)
    early = [(int(at), kind) for at, kind in effects if int(at) < help_end]
    assert early == [], "runs before -Help can answer: %s" % early


@needs_powershell
def test_llama_a_missing_binary_names_the_fork_how_to_build_it_and_whether_stock_will_do(tmp_path):
    # It used to say only "Set LLAMA_BIN_DIR to a directory containing the fork
    # build" -- which fork, from where, built how, and whether a stock
    # llama.cpp would do were in no tracked file.
    env = launcher_env(tmp_path, LLAMA_LAUNCHER)
    code, out, err = run_launcher(LLAMA_LAUNCHER, [], env)
    assert code == 1, out + err
    assert "[run] llama-server.exe not found at: %s" % (
        Path(env["LLAMA_BIN_DIR"]) / "llama-server.exe") in out
    assert "https://github.com/YawLabs/llama.cpp" in out
    assert "cmake --preset arm64-windows-llvm-release -DGGML_OPENCL=ON" in out
    assert "cmake --build build-arm64-windows-llvm-release" in out
    assert "Stock llama.cpp is UNTESTED here" in out
    assert "LLAMA_BIN_DIR" in out
    # The build it describes lands where the launcher looks by default:
    # <parent of this repo>\llama-qnn-fork\build-<preset>\bin.
    m = re.search(r'Join-Path \$yaw "llama-qnn-fork\\(build-[\w-]+)\\bin"', launcher_code(LLAMA_LAUNCHER))
    assert m, "the default LLAMA_BIN_DIR moved; update this test"
    assert "cmake --build %s" % m.group(1) in out
    assert "llama-qnn-fork\\" in out
    # It stopped at the check: no slot dir, no log dir, no child.
    assert sorted(p.name for p in (tmp_path / "npu-root").iterdir()) == []
