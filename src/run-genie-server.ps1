# Launch the Genie NPU OpenAI-compatible server.
# Uses the box's native ARM64 python (must be aarch64 to load Genie.dll).
#
#   powershell -File src\run-genie-server.ps1                   # Qwen3-4B (default)
#   powershell -File src\run-genie-server.ps1 -Model qwen3-8b   # Qwen3-8B tier
#
# NOT here: Qwen3.5-9B. It has no Genie export upstream (open qai-hub-models
# feature request); it serves through src\run-llama-server.ps1 instead. See
# docs/MODEL_OPTIONS.md for the whole model matrix.
param(
    [ValidateSet("qwen3-4b", "qwen3-8b")]
    [string]$Model = "qwen3-4b"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# One entry per servable NPU model: the bundle dir under <root>\bundles and the
# id reported to clients. The 8B is the AI Hub prebuilt (fetched with
# `qai-hub-models fetch qwen3_8b -r genie -p w4a16 -c qualcomm-snapdragon-x-elite`),
# multi-length [512..4096] out of the box -- apply the per-machine config fixes
# (poll: false, token-penalty) before first serve; see docs/MODEL_OPTIONS.md.
$Bundles = @{
    "qwen3-4b" = @{ dir = "qwen3_4b-genie-w4a16-x-elite-ctx8192-multi";        id = "qwen3-4b-npu" }
    "qwen3-8b" = @{ dir = "qwen3_8b-genie-w4a16-qualcomm_snapdragon_x_elite"; id = "qwen3-8b-npu" }
}
$DefaultBundle = $Bundles[$Model].dir
# For the 4B, swap to ...qualcomm_snapdragon_x_elite (4096, also multi-length)
# for the best decode at depth, or ...ctx16384 for the largest window -- but
# that one is SINGLE-length and therefore slow at every depth.
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
# The override is UNDONE in the finally at the bottom. An interactive
# `.\run-genie-server.ps1 -Model qwen3-8b` runs in the calling shell's own
# process, and without the restore the two vars would linger there and turn a
# later plain run into a silent 8B serve -- the launcher manufacturing the
# exact stale-env failure this block exists to defeat.
$modelExplicit = $PSBoundParameters.ContainsKey("Model")
$savedBundleDir = $env:GENIE_BUNDLE_DIR
$savedModelId = $env:GENIE_MODEL_ID
if ($modelExplicit) {
    $env:GENIE_BUNDLE_DIR = Join-Path $env:GENIE_NPU_ROOT "bundles\$DefaultBundle"
    $env:GENIE_MODEL_ID = $Bundles[$Model].id
} elseif (-not $env:GENIE_BUNDLE_DIR) {
    $env:GENIE_BUNDLE_DIR = Join-Path $env:GENIE_NPU_ROOT "bundles\$DefaultBundle"
}
# Env-path id hygiene: with no -Model and no GENIE_MODEL_ID, the server falls
# back to its own default id (qwen3-4b-npu) whatever bundle the env points at
# -- the right bundle advertised under the wrong id. Fill the id in when the
# bundle dir is one this map knows; say so out loud when it is not.
if (-not $modelExplicit -and -not $env:GENIE_MODEL_ID) {
    $leaf = Split-Path -Leaf $env:GENIE_BUNDLE_DIR
    $known = $Bundles.GetEnumerator() | Where-Object { $_.Value.dir -eq $leaf } | Select-Object -First 1
    if ($known) {
        $env:GENIE_MODEL_ID = $known.Value.id
    } elseif ($leaf -notmatch "qwen3_4b") {
        Write-Host "[run] note: GENIE_MODEL_ID is unset, so the server will advertise its"
        Write-Host "[run] default id (qwen3-4b-npu) for bundle '$leaf'. Set GENIE_MODEL_ID"
        Write-Host "[run] if anything routes on the reported model id."
    }
}

try {

if (-not $env:GENIE_SDK_DIR) {
    # Newest QAIRT under <root>/qairt, so a runtime upgrade does not need an
    # edit here. Falls through to the error below if none is installed.
    $qairt = Join-Path $env:GENIE_NPU_ROOT "qairt"
    if (Test-Path $qairt) {
        $newest = Get-ChildItem $qairt -Directory -ErrorAction SilentlyContinue |
                  Sort-Object Name -Descending | Select-Object -First 1
        if ($newest) { $env:GENIE_SDK_DIR = $newest.FullName }
    }
}

# Fail with something a stranger can act on, naming what to set and what was
# actually tried -- the server's own check would otherwise report a path the
# user never chose.
foreach ($pair in @(@("GENIE_BUNDLE_DIR", $env:GENIE_BUNDLE_DIR),
                    @("GENIE_SDK_DIR",    $env:GENIE_SDK_DIR))) {
    if (-not $pair[1] -or -not (Test-Path $pair[1])) {
        Write-Host ""
        Write-Host ("[run] $($pair[0]) is not set to an existing directory.")
        Write-Host ("      tried: " + $(if ($pair[1]) { $pair[1] } else { "(unset)" }))
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
if (-not $env:GENIE_HOST)  { $env:GENIE_HOST = "127.0.0.1" }
if (-not $env:GENIE_PORT)  { $env:GENIE_PORT = "8123" }

# Confirm python is ARM64 (Genie.dll is aarch64-windows-msvc).
$arch = & python -c "import platform;print(platform.machine())"
if ($arch -notmatch "ARM64|aarch64") {
    Write-Warning "python arch is '$arch' -- Genie.dll is aarch64; use a native ARM64 python or this will fail to load."
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
$maxRestarts = if ($env:GENIE_MAX_RESTARTS) { [int]$env:GENIE_MAX_RESTARTS } else { 5 }
# 25s, not a token pause: a force-killed server needs roughly 20s of settling
# before the next bundle load, and restarting sooner was measured costing about
# half of decode throughput. A restart that silently comes back at half speed
# is a bad way to recover from an incident -- the server looks healthy and
# every number it produces is wrong.
$cooldown    = if ($env:GENIE_RESTART_COOLDOWN) { [int]$env:GENIE_RESTART_COOLDOWN } else { 25 }
$restarts = 0
# Which kinds the current streak has seen, for the give-up summary below.
$sawCrash = $false
$sawWedge = $false

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
    Write-Host "[run] starting Genie server (python $arch) on $($env:GENIE_HOST):$($env:GENIE_PORT)"
    $started = Get-Date
    & python (Join-Path $here "genie_server.py")
    $code = $LASTEXITCODE
    $ranFor = ((Get-Date) - $started).TotalSeconds

    $crashed = Test-NativeCrash $code
    if ($code -ne 75 -and -not $crashed) {
        # Anything else is a deliberate exit: Ctrl-C, a config error the server
        # already explained, a port collision. Restarting would just repeat it.
        Write-Host "[run] server exited $code -- not a wedge, not restarting."
        exit $code
    }

    # Count only restarts that follow a SHORT life. A server that ran for hours
    # and then wedged once is a different animal from one wedging on startup,
    # and folding them together would exhaust the budget on a healthy box that
    # simply had a long uptime.
    if ($ranFor -lt 120) { $restarts++ } else { $restarts = 1; $sawCrash = $false; $sawWedge = $false }

    # Name which of the two happened. They need different next steps from the
    # operator -- a crash left a WER report and a faulting module to look up, a
    # wedge left nothing but a stuck thread -- and "wedged" printed over a crash
    # sends them hunting for a hang that never happened.
    #
    # Tracked as two flags rather than one label because a streak can MIX. The
    # summary below covers every failure in the streak, not just the last one,
    # and reporting "wedged 5 times" over three wedges and two crashes would
    # send the operator looking for a hang on the strength of which kind
    # happened to come last. The flags reset with the counter, since a fresh
    # streak is a fresh question.
    if ($crashed) { $what = "crashed"; $sawCrash = $true }
    else          { $what = "wedged";  $sawWedge = $true }
    if ($crashed) {
        $hex = [BitConverter]::ToUInt32([BitConverter]::GetBytes($code), 0)
        Write-Host ("[run] server crashed: exit 0x{0:X8}. The faulting module is in" -f $hex)
        Write-Host "[run] Event Viewer (Application, Windows Error Reporting) -- if it"
        Write-Host "[run] names QnnHtp.dll the fault was inside the QNN driver, not here."
    }

    if ($restarts -gt $maxRestarts) {
        $summary = if ($sawCrash -and $sawWedge) { "crashed and wedged" }
                   elseif ($sawCrash)            { "crashed" }
                   else                          { "wedged" }
        Write-Host "[run] the engine $summary $restarts times in quick succession."
        Write-Host "[run] Giving up rather than looping on a sick device. The HTP"
        Write-Host "[run] may need a reset (reboot, or reload the driver) before"
        Write-Host "[run] this will come back. Raise GENIE_MAX_RESTARTS to retry more."
        exit 75
    }

    Write-Host "[run] engine $what after ${ranFor}s -- restart $restarts/$maxRestarts in ${cooldown}s."
    Start-Sleep -Seconds $cooldown
}

} finally {
    # Undo the -Model override (see the comment on it above). Restores also
    # the id the hygiene block filled in, so an in-shell run leaves the two
    # vars exactly as it found them. Everything else this script sets
    # (GENIE_NPU_ROOT, GENIE_SDK_DIR, GENIE_HOST, GENIE_PORT) keeps its
    # long-standing fill-if-unset contract.
    $env:GENIE_BUNDLE_DIR = $savedBundleDir
    $env:GENIE_MODEL_ID = $savedModelId
}
