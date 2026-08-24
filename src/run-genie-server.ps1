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
Write-Host "[run] starting Genie server (python $arch) on $($env:GENIE_HOST):$($env:GENIE_PORT)"
& python (Join-Path $here "genie_server.py")
