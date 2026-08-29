# Launch the Genie NPU OpenAI-compatible server.
# Uses the box's native ARM64 python (must be aarch64 to load Genie.dll).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# EDIT THESE to where you extracted the Genie bundle and the QAIRT 2.45 runtime.
# They are large external artifacts and are intentionally NOT in this repo.
# (Env vars already set in the shell win over these defaults.)
# Default is the 8192 MULTI-LENGTH bundle. Two reasons, both measured:
#   * multi-length beats single-length by 2-3x on short prompts at the same
#     window, because the binary carries one prefill and one decode graph
#     per compiled context length and runs against the smallest that fits.
#     Verified with qnn-context-binary-utility: 10 graphs against 2. Costs
#     +3.8% disk and no extra HTP memory.
#   * at 8192 it holds twice the 4096 prebuilt's context for near-identical
#     shallow decode (18.2 vs 18.7 t/s at 250 tokens), which matters because
#     a realistic agent preamble already measured 626 tokens and source runs
#     10-13 tokens per line.
$DefaultBundle = "qwen3_4b-genie-w4a16-x-elite-ctx8192-multi"
# Swap to ...qualcomm_snapdragon_x_elite (4096, also multi-length) for the
# best decode at depth, or ...ctx16384 for the largest window -- but that one
# is SINGLE-length and therefore slow at every depth.
# Resolved rather than hardcoded: a checked-in absolute path is one developer's
# machine, and every other clone gets a "not found" naming a stranger's home
# directory. Set GENIE_NPU_ROOT (or the two vars directly) to point elsewhere.
if (-not $env:GENIE_NPU_ROOT) { $env:GENIE_NPU_ROOT = (Join-Path (Split-Path -Parent (Split-Path -Parent $here)) "genie-npu") }

if (-not $env:GENIE_BUNDLE_DIR) {
    $env:GENIE_BUNDLE_DIR = Join-Path $env:GENIE_NPU_ROOT "bundles\$DefaultBundle"
}
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
    if ($ranFor -lt 120) { $restarts++ } else { $restarts = 1 }

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

    if ($restarts -gt $maxRestarts) {
        Write-Host "[run] the engine $what $restarts times in quick succession."
        Write-Host "[run] Giving up rather than looping on a sick device. The HTP"
        Write-Host "[run] may need a reset (reboot, or reload the driver) before"
        Write-Host "[run] this will come back. Raise GENIE_MAX_RESTARTS to retry more."
        exit 75
    }

    Write-Host "[run] engine $what after ${ranFor}s -- restart $restarts/$maxRestarts in ${cooldown}s."
    Start-Sleep -Seconds $cooldown
}
