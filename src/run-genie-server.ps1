# Launch the Genie NPU OpenAI-compatible server.
# Uses the box's native ARM64 python (must be aarch64 to load Genie.dll).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# EDIT THESE to where you extracted the Genie bundle and the QAIRT 2.45 runtime.
# They are large external artifacts and are intentionally NOT in this repo.
# (Env vars already set in the shell win over these defaults.)
if (-not $env:GENIE_BUNDLE_DIR) { $env:GENIE_BUNDLE_DIR = "C:\Users\jeff\yaw\genie-npu\bundles\qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite" }
if (-not $env:GENIE_SDK_DIR)    { $env:GENIE_SDK_DIR    = "C:\Users\jeff\yaw\genie-npu\qairt\2.45.0.260326" }
if (-not $env:GENIE_HOST)  { $env:GENIE_HOST = "127.0.0.1" }
if (-not $env:GENIE_PORT)  { $env:GENIE_PORT = "8123" }

# Confirm python is ARM64 (Genie.dll is aarch64-windows-msvc).
$arch = & python -c "import platform;print(platform.machine())"
if ($arch -notmatch "ARM64|aarch64") {
    Write-Warning "python arch is '$arch' -- Genie.dll is aarch64; use a native ARM64 python or this will fail to load."
}
Write-Host "[run] starting Genie server (python $arch) on $($env:GENIE_HOST):$($env:GENIE_PORT)"
& python (Join-Path $here "genie_server.py")
