# Launch the Genie NPU OpenAI-compatible server.
# Runs `python` from PATH, or the interpreter GENIE_PYTHON names. Either way it
# must be a native ARM64 build -- Genie.dll is aarch64-only -- and the launcher
# refuses anything else rather than walking into the DLL-load failure.
#
#   powershell -File src\run-genie-server.ps1                        # Qwen3-4B (default)
#   powershell -File src\run-genie-server.ps1 -Model qwen3-8b        # Qwen3-8B, AI Hub prebuilt (4096)
#   powershell -File src\run-genie-server.ps1 -Model qwen3-8b-8192   # Qwen3-8B, self-exported 8192 multi-length
#
# NOT here: Qwen3.5-9B. It has no Genie export upstream (open qai-hub-models
# feature request); it serves through src\run-llama-server.ps1 instead. See
# docs/MODEL_OPTIONS.md for the whole model matrix.
param(
    [ValidateSet("qwen3-4b", "qwen3-8b", "qwen3-8b-8192")]
    [string]$Model = "qwen3-4b"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# One entry per servable NPU model: the bundle dir under <root>\bundles and the
# id reported to clients. The 8B is the AI Hub prebuilt (fetched with
# `qai-hub-models fetch qwen3_8b -r genie -p w4a16 -c qualcomm-snapdragon-x-elite`),
# multi-length [512..4096] out of the box -- apply the per-machine config fixes
# (poll: false, token-penalty) before first serve; see docs/MODEL_OPTIONS.md.
# qwen3-8b is the AI Hub PREBUILT (4096); qwen3-8b-8192 is the self-exported
# 8192 multi-length build (2026-09-04) -- twice the window, and multi-length
# so a short prompt still runs against the smallest graph that fits. Both are
# 8B w4a16 and both are multi-length, so no window tax applies to either (the
# per-token tax is a SINGLE-length property). The 4096 prebuilt stays the
# default 8B only until a same-harness AC comparison exists -- the 8192-multi's
# decode was measured on battery and its prefill never on AC
# (docs/MODEL_OPTIONS.md, "The 8B 8192 multi-length tier").
$Bundles = @{
    "qwen3-4b"      = @{ dir = "qwen3_4b-genie-w4a16-x-elite-ctx8192-multi";       id = "qwen3-4b-npu" }
    "qwen3-8b"      = @{ dir = "qwen3_8b-genie-w4a16-qualcomm_snapdragon_x_elite"; id = "qwen3-8b-npu" }
    "qwen3-8b-8192" = @{ dir = "qwen3_8b-genie-w4a16-x-elite-ctx8192-multi";       id = "qwen3-8b-8192-npu" }
}
$DefaultBundle = $Bundles[$Model].dir
# For the 4B, swap to ...qualcomm_snapdragon_x_elite (4096, also multi-length)
# for the best decode at depth, or ...ctx16384 for the largest window -- but
# that one is SINGLE-length and therefore slow at every depth.
# Every GENIE_* var this script WRITES, saved before the first write and put
# back -- or removed again -- in the finally at the bottom, so an in-shell run
# leaves the environment exactly as it found it. Two of these can produce a
# WRONG serve if they linger (bundle dir and model id; see the -Model note
# below) and were restored first. The other four used to keep a fill-if-unset
# contract, i.e. they leaked, and GENIE_SDK_DIR showed why that was not free:
# the newest-QAIRT discovery below runs only while the var is unset, so the
# first in-shell run pinned the calling shell to that SDK for the rest of the
# session, and a runtime installed later was never picked up. Env is still how
# genie_server.py reads its configuration, so the values are computed into
# locals, written for the child, and undone. The llama launcher makes the same
# promise for LLAMA_* (locals only, nothing written).
#
# The try opens BEFORE the first write on purpose. It used to open after the
# -Model override, and a terminating error in that gap left the override
# behind in the calling shell -- a finally only protects what it encloses.
$EnvWritten = @("GENIE_NPU_ROOT", "GENIE_BUNDLE_DIR", "GENIE_MODEL_ID",
                "GENIE_SDK_DIR", "GENIE_HOST", "GENIE_PORT")
$savedEnv = @{}
foreach ($name in $EnvWritten) {
    $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$modelExplicit = $PSBoundParameters.ContainsKey("Model")

try {

# Resolved rather than hardcoded: a checked-in absolute path is one developer's
# machine, and every other clone gets a "not found" naming a stranger's home
# directory. Set GENIE_NPU_ROOT (or the two vars directly) to point elsewhere.
if (-not $env:GENIE_NPU_ROOT) { $env:GENIE_NPU_ROOT = (Join-Path (Split-Path -Parent (Split-Path -Parent $here)) "genie-npu") }

# An EXPLICIT -Model beats a lingering env var, and it moves the model id with
# the bundle: a shell that still exports the 4B's GENIE_BUNDLE_DIR (or a
# GENIE_MODEL_ID from an earlier session) must not hand -Model qwen3-8b the
# wrong bundle -- or worse, the right bundle advertised under the wrong id.
# Without -Model the old contract holds: env vars win over the defaults.
#
# The override is UNDONE in the finally at the bottom, with everything else in
# $EnvWritten. An interactive `.\run-genie-server.ps1 -Model qwen3-8b` runs in
# the calling shell's own process, and without the restore the two vars would
# linger there and turn a later plain run into a silent 8B serve -- the
# launcher manufacturing the exact stale-env failure this block exists to
# defeat.
if ($modelExplicit) {
    $env:GENIE_BUNDLE_DIR = Join-Path $env:GENIE_NPU_ROOT "bundles\$DefaultBundle"
    $env:GENIE_MODEL_ID = $Bundles[$Model].id
} elseif (-not $env:GENIE_BUNDLE_DIR) {
    $env:GENIE_BUNDLE_DIR = Join-Path $env:GENIE_NPU_ROOT "bundles\$DefaultBundle"
}
# Env-path id hygiene: with no -Model and no GENIE_MODEL_ID, the server falls
# back to its own default id (qwen3-4b-npu) whatever bundle the env points at
# -- the right bundle advertised under the wrong id. Fill the id in when the
# bundle dir is one this map knows; say so out loud when it is not. And when
# an id IS set but disagrees with what the map says the bundle is, warn --
# env still wins, but a stale export serving the right bundle under the wrong
# id is exactly the silent failure this block exists to catch.
if (-not $modelExplicit) {
    $leaf = Split-Path -Leaf $env:GENIE_BUNDLE_DIR
    $known = $Bundles.GetEnumerator() | Where-Object { $_.Value.dir -eq $leaf } | Select-Object -First 1
    if (-not $env:GENIE_MODEL_ID) {
        if ($known) {
            $env:GENIE_MODEL_ID = $known.Value.id
        } elseif ($leaf -notmatch "qwen3_4b") {
            Write-Host "[run] note: GENIE_MODEL_ID is unset, so the server will advertise its"
            Write-Host "[run] default id (qwen3-4b-npu) for bundle '$leaf'. Set GENIE_MODEL_ID"
            Write-Host "[run] if anything routes on the reported model id."
        }
    } elseif ($known -and $env:GENIE_MODEL_ID -ne $known.Value.id) {
        Write-Host "[run] note: GENIE_MODEL_ID '$($env:GENIE_MODEL_ID)' does not match bundle"
        Write-Host "[run] '$leaf' (this launcher knows it as '$($known.Value.id)'). The env value"
        Write-Host "[run] wins and will be advertised -- unset GENIE_MODEL_ID if it is stale."
    }
}

# QAIRT runtime: an explicit GENIE_SDK_DIR wins; otherwise the newest version
# directory under <root>\qairt, so a runtime upgrade does not need an edit
# here. "Newest" is a VERSION compare over names shaped like 2.45.0.260326,
# not a string sort -- a string sort ranks 2.9 above 2.100 and any alphabetic
# name ("latest", "backup") above every number. Names that are not a dotted
# version are ignored, so a stray directory beside the SDKs is never handed to
# the server as one.
#
# "A dotted version" means one [version] can HOLD: two to four components of at
# most nine ASCII digits each. [version]'s components are Int32 and the filter
# used to be \d+, so an all-digits stray with one longer part -- a directory
# named for a full build stamp (2.45.0.260326153000), a timestamped backup --
# passed the filter and then killed the launcher with a bare cast error out of
# Sort-Object, a valid SDK sitting right beside it. Nine digits cannot overflow
# an Int32. [0-9] rather than \d because .NET's \d admits every Unicode digit
# and [version] parses only the ASCII ones. The sort key catches all the same,
# and ranks a name it cannot parse below every real version, so the ordering
# never rests on the filter being airtight.
#
# Computed into a local; the env write happens below, after the checks, so a
# failed run writes nothing. What was searched is remembered so the failure
# can name it.
$sdkDir = $env:GENIE_SDK_DIR
$qairt = Join-Path $env:GENIE_NPU_ROOT "qairt"
$qairtNote = $null
if (-not $sdkDir) {
    if (Test-Path $qairt) {
        $newest = Get-ChildItem $qairt -Directory -ErrorAction SilentlyContinue |
                  Where-Object { $_.Name -match '^[0-9]{1,9}(\.[0-9]{1,9}){1,3}$' } |
                  Sort-Object { try { [version]($_.Name) } catch { [version]"0.0" } } -Descending |
                  Select-Object -First 1
        if ($newest) { $sdkDir = $newest.FullName }
        else { $qairtNote = "searched $qairt for a QAIRT version directory (e.g. 2.45.0.260326): none found" }
    } else {
        $qairtNote = "no $qairt directory to search for a QAIRT version"
    }
}

# Fail with something a stranger can act on, naming what to set and what was
# actually tried -- the server's own check would otherwise report a path the
# user never chose. When discovery ran and came up empty, "tried: (unset)" plus
# "set GENIE_NPU_ROOT" was advice for a root the operator had already set
# correctly; the line that helps names the directory that was scanned (an SDK
# zip dropped there but never unpacked is the usual shape).
foreach ($pair in @(@("GENIE_BUNDLE_DIR", $env:GENIE_BUNDLE_DIR),
                    @("GENIE_SDK_DIR",    $sdkDir))) {
    if (-not $pair[1] -or -not (Test-Path $pair[1])) {
        Write-Host ""
        Write-Host ("[run] $($pair[0]) is not set to an existing directory.")
        Write-Host ("      tried: " + $(if ($pair[1]) { $pair[1] } else { "(unset)" }))
        if ($pair[0] -eq "GENIE_SDK_DIR" -and $qairtNote) {
            Write-Host ("      $qairtNote.")
            Write-Host ("      Unpack the SDK there (a version-named directory holding lib\), or")
            Write-Host ("      point GENIE_SDK_DIR at an unpacked one.")
        }
        Write-Host ("      The Genie bundle and the QAIRT runtime are large external")
        Write-Host ("      artifacts and are deliberately NOT in this repo. Set either")
        Write-Host ("        `$env:GENIE_NPU_ROOT = '<dir holding bundles\ and qairt\>'")
        Write-Host ("      or $($pair[0]) directly. See docs/GENIE_SERVER.md.")
        if ($modelExplicit -and $pair[0] -eq "GENIE_BUNDLE_DIR") {
            # "Set GENIE_BUNDLE_DIR directly" is a dead end while -Model is on
            # the command line -- the override above rewrites it every run. Do
            # not hand out advice the next run will undo.
            Write-Host ("      NOTE: -Model $Model overrides GENIE_BUNDLE_DIR each run. For a")
            Write-Host ("      bundle under a different directory name, drop -Model and set")
            Write-Host ("      GENIE_BUNDLE_DIR (plus GENIE_MODEL_ID), or edit the `$Bundles map.")
        }
        exit 1
    }
}
$env:GENIE_SDK_DIR = $sdkDir
$bindHost = if ($env:GENIE_HOST) { $env:GENIE_HOST } else { "127.0.0.1" }
# Port: parsed and range-checked HERE, in the shape run-llama-server.ps1 uses
# for LLAMA_PORT, so the LAUNCHER's 8123 is what a typo degrades to. It used to
# be copied through unparsed, and the server degrades a value it cannot read to
# its OWN default, 8080 -- which is where run-llama-server.ps1 puts its CPU leg
# and typed's local endpoint, the one port this 8123 default exists to avoid.
# So GENIE_PORT=808O either collided with the CPU leg or served the NPU where
# clients expect the 9B, while the line that announces the start below printed
# the typo itself as the port and every tool here kept looking at 8123.
# Not Get-EnvInt: that one is defined further down (a function exists only once
# its definition has run) and takes a minimum, not a range.
$port = "8123"
if ($env:GENIE_PORT) {
    $p = 0
    if ([int]::TryParse($env:GENIE_PORT, [ref]$p) -and $p -ge 1 -and $p -le 65535) { $port = "$p" }
    else { Write-Host "[run] WARNING: GENIE_PORT='$($env:GENIE_PORT)' is not a port number (1-65535); using 8123." }
}
$env:GENIE_HOST = $bindHost
$env:GENIE_PORT = $port

# Which python, and is it ARM64. Bare `python` on PATH used to be the only
# choice and a mismatch only a Write-Warning, after which the launcher walked
# into a guaranteed failure: Genie.dll is pure ARM64 (PE machine 0xAA64, not
# ARM64X), so an x64 interpreter -- an x64 venv, the Store stub -- cannot load
# it, and the server died in load_engine with an OSError that named neither
# the interpreter nor the fix. GENIE_PYTHON names the interpreter explicitly;
# the arch check is a refusal because nothing after it can succeed.
$python = if ($env:GENIE_PYTHON) { $env:GENIE_PYTHON } else { "python" }
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    Write-Host "[run] python not found: '$python'. Set GENIE_PYTHON to a native ARM64 interpreter."
    exit 1
}
#
# The probe asks the interpreter for ITS OWN BUILD, not for the machine it is
# running on. platform.machine() was the obvious question and is the wrong one:
# from CPython 3.12 on Windows it reports the HOST cpu (it asks WMI), so an
# emulated x64 python on this Snapdragon box answers ARM64 -- and answers it
# unreliably, because the WMI-cold first call falls back to
# PROCESSOR_ARCHITECTURE and says AMD64, so the SAME x64 interpreter was
# refused on one launch and accepted on the next. Accepted, it got the
# "starting Genie server (python ARM64)" line below and then the DLL-load
# OSError this refusal exists to replace. The x64 3.8-3.11 builds answer AMD64
# (pre-3.12 platform reads the env var), which is why the old probe looked
# right. sysconfig.get_platform() is the interpreter's own build tag --
# win-arm64 against win-amd64 -- and no host can move it.
#
# ONE line of what the interpreter printed is the answer, and it is TAGGED: the
# last line matching ^GENIE_ARCH=. GENIE_PYTHON takes anything Get-Command
# resolves, a .cmd shim included, and a shim talks. For want of `@echo off` cmd
# echoes every command line it runs -- as does an activation banner -- so there
# can be output BEFORE python's print and, just as easily, AFTER it:
# `exit /b %ERRORLEVEL%`, which is how a shim hands the server's exit 75 back
# to the supervise loop below, echoes itself once python has already printed.
# Neither the first line nor the last non-blank one is the answer, then; taking
# the last non-blank one refused a native ARM64 python behind exactly the shim
# shape this comment says is supported, quoting `C:\shims>exit /b 0` as the
# machine. The tag is also what stops a banner that happens to mention ARM64
# from vouching for an x64 interpreter: only a tagged line is ever read.
#
# Reduced to one STRING before the compare, because `-notmatch` on an array
# does not answer "did it match": it returns the elements that did NOT, and a
# non-empty array is true, so any second line refused a native ARM64 python by
# a message that blamed the architecture while printing ARM64.
#
# The line is taken by INTERPOLATION ("$(...)"), not a [string] cast, because
# there may be no line at all: an interpreter that dies with its error on
# stderr, or a wrapper that swallows stdout, leaves the pipeline empty, and on
# Windows PowerShell 5.1 [string] of an empty pipeline is still $null. .Trim()
# on that threw "You cannot call a method on a null-valued expression" -- the
# launcher dying of its own arch check, where an interpreter that never said
# ARM64 belongs in the refusal below like any other. An expandable string is
# a string whatever went into it.
$archOut = & $python -c "import sysconfig;print('GENIE_ARCH=' + sysconfig.get_platform())"
$archSaid = @(@($archOut) | Where-Object { "$_".Trim() })
$arch = "$($archSaid | Where-Object { "$_".Trim() -match '^GENIE_ARCH=' } | Select-Object -Last 1)".Trim() -replace '^GENIE_ARCH=', ''
if ($arch -notmatch "arm64|aarch64") {
    Write-Host "[run] python arch is '$arch' ($python) -- Genie.dll is aarch64-only and cannot"
    Write-Host "[run] load in this interpreter. Set GENIE_PYTHON to a native ARM64 python."
    if (-not $arch) {
        # No tagged line at all: the probe never reached its print. '' is not an
        # architecture, so say which of the two shapes it looks like -- the raw
        # capture used to sail through here (an EMPTY capture on the left of
        # -notmatch is an empty collection, which is false) and the server was
        # launched under an interpreter that had just failed to run one line.
        if ($archSaid.Count) {
            Write-Host "[run] (It printed $($archSaid.Count) line(s) but no GENIE_ARCH= line; its own error, if any, is above.)"
        } else {
            Write-Host "[run] (It printed no text on stdout; if it failed, its own error is above.)"
        }
    }
    exit 1
}
# Supervise. A wedged HTP cannot be recovered inside the server process -- the
# stuck call is in the driver and Python cannot reclaim a thread blocked in
# native code -- so the server exits 75 (EX_TEMPFAIL) and asks to be replaced.
# Without something to act on that exit, the detection is only a better error
# message; this loop is what turns it into recovery.
#
# A wedge is not the only way the device takes the server down, though, and it
# was not the one actually observed. The driver can fault instead of hang: WER
# on this box records python.exe dying with 0xC0000005 inside QnnHtp.dll at the
# identical offset twice (2026-08-24 and 2026-08-27), alongside the same crash
# from test-qnn-lifecycle.exe. A crash is a wedge by another name -- the engine
# is gone and only a fresh process brings it back -- but it exits with an
# NTSTATUS, not 75, so treating "not 75" as "deliberate" made this loop give up
# on precisely the failure it exists to recover from.
#
# Restarts are rate-limited and capped. A device that wedges immediately on
# every load is not going to be fixed by looping on it, and a tight restart
# loop against a sick NPU is worse than being down: it keeps the HTP busy and
# buries the original failure under identical log stanzas.
#
# Capped by failures inside a SLIDING WINDOW, not by the length of the last
# life. The old rule counted only lives shorter than 120s and reset the count
# on any longer one. That sounded right -- hours of uptime and then one wedge
# is not a sick device -- and was wrong for both failures this loop handles:
# the wedge detector itself needs at least 180s (stall 120s + grace 60s, or
# first-token 300s + 60s) before the server exits 75, so NO wedge life could
# ever be short enough to count, and a driver that faults a few minutes in
# (the QnnHtp.dll pattern above, typically on the first real request)
# restarted forever at "restart 1/5". The give-up branch, and the pnputil hint
# it carries, were dead for the exact case the loop exists for. Now every
# failure is timestamped, failures older than GENIE_RESTART_WINDOW seconds age
# out (and the loop says so when they do), and more than GENIE_MAX_RESTARTS
# inside the window is the give-up. An hour by default: a wedge costs 3-6
# minutes to detect plus the load and the cooldown, so the six it takes to
# exceed the default cap of 5 fit inside it (about 40 minutes at the slow end),
# while a box that wedged once at lunch and once at dinner never accumulates.
#
# The three integers are parsed, floored and defaulted here, in the shape the
# llama launcher uses for LLAMA_HEALTH_TIMEOUT. A raw [int] cast made junk a
# bare cast error, and a negative cooldown reached Start-Sleep and threw "less
# than the minimum allowed range" -- inside the restart path, so the supervisor
# died at the one moment it was supposed to act. Junk falls back loudly to the
# default. 0 is a valid cap (give up on the first failure) and a valid cooldown
# (the operator may be measuring exactly that).
function Get-EnvInt([string]$name, [int]$default, [int]$min) {
    $raw = [Environment]::GetEnvironmentVariable($name, "Process")
    if (-not $raw) { return $default }
    $v = 0
    if (-not [int]::TryParse($raw, [ref]$v) -or $v -lt $min) {
        Write-Host "[run] WARNING: $name='$raw' is not an integer >= $min; using $default."
        return $default
    }
    return $v
}
$maxRestarts = Get-EnvInt "GENIE_MAX_RESTARTS" 5 0
# 25s, not a token pause: a force-killed server needs roughly 20s of settling
# before the next bundle load, and restarting sooner was measured costing about
# half of decode throughput. A restart that silently comes back at half speed
# is a bad way to recover from an incident -- the server looks healthy and
# every number it produces is wrong.
$cooldown    = Get-EnvInt "GENIE_RESTART_COOLDOWN" 25 0
$window      = Get-EnvInt "GENIE_RESTART_WINDOW" 3600 1
# One @{ at; kind } per failure still inside the window. The give-up summary is
# derived from it, so a failure that ages out takes its kind with it.
$failures = @()

# Ctrl-C arrives as an NTSTATUS too (STATUS_CONTROL_C_EXIT), so it has to be
# carved out by hand or the operator's own stop becomes a restart -- the one
# way this change could make things worse than the bug it fixes.
$STATUS_CONTROL_C_EXIT = -1073741510   # 0xC000013A

function Test-NativeCrash([int]$code) {
    # What separates a crash from a deliberate failure is MAGNITUDE, not sign.
    # An exception code carries a severity, a facility and a code field, so it
    # is always enormous: 0xC0000005 access violation, 0xC0000409 stack buffer
    # overrun, 0xE06D7363 unhandled C++ exception, 0x80000003 breakpoint. A
    # program that means to fail writes exit(1), or sloppily exit(-1), and -1
    # arrives as 0xFFFFFFFF -- numerically inside any "negative means crash"
    # window while meaning the exact opposite. So the window stops at -65536,
    # well above every deliberate small negative and far below every real
    # exception code.
    #
    # Written in decimal deliberately: PowerShell 5.1 parses the literal
    # 0xC0000000 as a SIGNED Int32 (-1073741824), so the obvious hex spelling
    # of a bound like this silently matches every code including 0 and 75.
    if ($code -eq $STATUS_CONTROL_C_EXIT) { return $false }
    return ($code -le -65536)
}

while ($true) {
    Write-Host "[run] starting Genie server (python $arch) on ${bindHost}:${port}"
    $started = Get-Date
    & $python (Join-Path $here "genie_server.py")
    $code = $LASTEXITCODE
    $ranFor = [int](((Get-Date) - $started).TotalSeconds)

    $crashed = Test-NativeCrash $code
    if ($code -ne 75 -and -not $crashed) {
        # Anything else is a deliberate exit: Ctrl-C, a config error the server
        # already explained, a port collision. Restarting would just repeat it.
        Write-Host "[run] server exited $code -- not a wedge, not restarting."
        exit $code
    }

    # Name which of the two happened. They need different next steps from the
    # operator -- a crash left a WER report and a faulting module to look up, a
    # wedge left nothing but a stuck thread -- and "wedged" printed over a crash
    # sends them hunting for a hang that never happened.
    $what = if ($crashed) { "crashed" } else { "wedged" }
    if ($crashed) {
        $hex = [BitConverter]::ToUInt32([BitConverter]::GetBytes($code), 0)
        Write-Host ("[run] server crashed: exit 0x{0:X8}. The faulting module is in" -f $hex)
        Write-Host "[run] Event Viewer (Application, Windows Error Reporting) -- if it"
        Write-Host "[run] names QnnHtp.dll the fault was inside the QNN driver, not here."
    }

    # Age out failures older than the window, and say so: a count that drops
    # back silently is indistinguishable from one that never moved, and the
    # operator reading "restart 1/5" under the fifth identical stanza deserves
    # to know why the loop is still going.
    $now = Get-Date
    $aged = @($failures | Where-Object { ($now - $_.at).TotalSeconds -gt $window })
    if ($aged.Count -gt 0) {
        $failures = @($failures | Where-Object { ($now - $_.at).TotalSeconds -le $window })
        Write-Host "[run] $($aged.Count) earlier failure(s) aged out of the ${window}s window; $($failures.Count) still count."
    }
    $failures += @{ at = $now; kind = $what }
    $restarts = $failures.Count

    if ($restarts -gt $maxRestarts) {
        # The summary covers every failure in the window, not just the last
        # one: a window can MIX, and "wedged 5 times" over three wedges and two
        # crashes would send the operator looking for a hang on the strength
        # of which kind happened to come last.
        $kinds = @($failures | ForEach-Object { $_.kind } | Sort-Object -Unique)
        $summary = if ($kinds.Count -gt 1) { "crashed and wedged" } else { $kinds[0] }
        Write-Host "[run] the engine $summary $restarts times within ${window}s."
        Write-Host "[run] Giving up rather than looping on a sick device. The HTP may"
        Write-Host "[run] need a reset before this comes back: from an elevated"
        Write-Host "[run] PowerShell,  pnputil /restart-device ""ACPI\QCOM0D0A\2&DABA3FF&0"""
        Write-Host "[run] -- that instance id is the DEV BOX's Hexagon NPU node and is per"
        Write-Host "[run] machine; on another box take yours from"
        Write-Host "[run] Get-PnpDevice -FriendlyName '*Hexagon*'. (Verified fix for the"
        Write-Host "[run] interrupt-delivery crawl -- see docs/MODEL_OPTIONS.md), or reboot."
        Write-Host "[run] Raise GENIE_MAX_RESTARTS (or shorten GENIE_RESTART_WINDOW) to retry more."
        exit 75
    }

    Write-Host "[run] engine $what after ${ranFor}s -- restart $restarts/$maxRestarts within ${window}s; next in ${cooldown}s."
    Start-Sleep -Seconds $cooldown
}

} finally {
    # Put every var in $EnvWritten back exactly as it was found (see the note
    # above the try): the -Model override, the id the hygiene block filled in,
    # and the root / SDK / host / port fills alike. A var that was unset is
    # REMOVED again rather than left empty, and through the env: provider, not
    # [Environment]::SetEnvironmentVariable. That API deletes on $null, but
    # $null never reaches it from PowerShell: the method binder hands a [string]
    # parameter "" instead (which is why [NullString] exists), and "" is a
    # delete only up to .NET 8. From .NET 9 -- pwsh 7.5 and later -- it SETS an
    # empty value, so an in-shell run there left all six set but empty, and a
    # bare `python src\genie_server.py` in that shell read GENIE_HOST="" as the
    # wildcard bind and GENIE_MODEL_ID="" as its id. Remove-Item on the provider
    # is a delete on every version. It is told to be quiet about a var that is
    # already gone -- one this run never got as far as writing -- because
    # $ErrorActionPreference is Stop, and a throw in here would abandon the
    # rest of the restore.
    foreach ($name in $EnvWritten) {
        if ($null -eq $savedEnv[$name]) {
            Remove-Item -LiteralPath "env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item -LiteralPath "env:$name" -Value $savedEnv[$name]
        }
    }
}
