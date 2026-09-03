# Launch a Qwen3.5-9B llama-server leg beside the Genie NPU server.
#
#   powershell -File src\run-llama-server.ps1            # CPU leg (default)
#   powershell -File src\run-llama-server.ps1 -Leg gpu   # Adreno leg
#
# Why this leg exists: the Genie/HTP chain cannot serve this model. Qwen3.5-9B
# has no Genie export upstream (qai-hub-models has no qwen3_5_9b target -- open
# feature request, qualcomm/ai-hub-models#287 -> #221 -- and the Qwen3.5
# entries that DO exist are fetch-only GGUFs for the GenieX llama.cpp runtime,
# not HTP context binaries). So the 9B serves through llama.cpp, exactly the
# "one Genie server, N llama-servers" shape MULTI_ENGINE.md describes. See
# docs/MODEL_OPTIONS.md for the whole decision.
#
# THE QUANT IS PER-LEG, AND THE RANKING INVERTS BETWEEN THEM. Measured on this
# box (Qwen3-4B, d0): on the Adreno OpenCL leg Q4_K_M decodes 19.39 t/s against
# Q8_0's 9.14 -- the OpenCL SOA_Q/Adreno kernels target Q4, so the bigger file
# is also the slower one there. On the CPU leg the KleidiAI int8 kernels favour
# Q8_0 (the build itself warns: no KleidiAI kernel for q4_K). Hence: cpu leg =
# Q8_0, gpu leg = Q4_K_M. Do not "upgrade" the GPU leg to Q8_0 for quality; it
# costs half the throughput.
#
# Env vars (LLAMA_*) are INPUTS ONLY -- this script never writes them back.
# That is deliberate: an interactive `.\run-llama-server.ps1` runs in the
# calling shell's process, so a launcher that exported its own defaults would
# leave the cpu leg's port and model in the environment and silently feed them
# to a later `-Leg gpu` run in the same window. Locals cannot leak.
param(
    [ValidateSet("cpu", "gpu")]
    [string]$Leg = "cpu"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Large external artifacts resolve like the Genie launcher's: relative to the
# directory holding this repo. Env vars already set in the shell win.
$yaw = Split-Path -Parent (Split-Path -Parent $here)
$root = if ($env:GENIE_NPU_ROOT) { $env:GENIE_NPU_ROOT } else { Join-Path $yaw "genie-npu" }
# The fork build, not genie-npu\llama-bin: it carries the agent-mode flags
# this launcher uses (--agent, --cache-idle-slots, --ctx-checkpoints,
# --reasoning) and its llama-server initialises OpenCL (an earlier build's
# server could not reach the Adreno at all -- see MULTI_ENGINE.md).
$binDir = if ($env:LLAMA_BIN_DIR) { $env:LLAMA_BIN_DIR } else { Join-Path $yaw "llama-qnn-fork\build-arm64-windows-llvm-release\bin" }
$server = Join-Path $binDir "llama-server.exe"
if (-not (Test-Path $server)) {
    Write-Host "[run] llama-server.exe not found at: $server"
    Write-Host "[run] Set LLAMA_BIN_DIR to a directory containing the fork build."
    exit 1
}

# --- per-leg defaults (locals; an explicitly exported LLAMA_* wins) ---------
$hf = $env:LLAMA_HF
$gguf = $env:LLAMA_GGUF
if ($Leg -eq "cpu") {
    # The operator's known-good CPU serving config, encoded rather than
    # reinvented. Q8_0 via -hf: llama-server fetches into the HF cache itself
    # (~9.5 GB once). 8080 because this leg IS typed's local endpoint; the
    # Genie NPU server stays on 8123.
    if (-not ($hf -or $gguf)) { $hf = "unsloth/Qwen3.5-9B-GGUF:Q8_0" }
    $port  = if ($env:LLAMA_PORT) { $env:LLAMA_PORT } else { "8080" }
    # No -a by default on this leg, deliberately: without it the server
    # reports the -hf spec ("unsloth/Qwen3.5-9B-GGUF:Q8_0"), which is what
    # the hand-run instances have always advertised -- a client keyed on that
    # id would break if the launcher renamed it.
    $alias = $env:LLAMA_ALIAS
} else {
    # GPU leg: local Q4_K_M (the measured Adreno fast path), its own port so
    # both legs can serve at once. Fetch the file with:
    #   hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir <root>\gguf
    if (-not ($hf -or $gguf)) { $gguf = Join-Path $root "gguf\Qwen3.5-9B-Q4_K_M.gguf" }
    $port  = if ($env:LLAMA_PORT) { $env:LLAMA_PORT } else { "8124" }
    # A fresh endpoint nothing routes on yet, so it gets a clean self-id.
    $alias = if ($env:LLAMA_ALIAS) { $env:LLAMA_ALIAS } else { "qwen3.5-9b-gpu" }
}
# An exported model spec still applies to whichever leg runs next, so say so
# out loud when it lands the known-wrong quant on an engine. Warn, not refuse:
# the operator may be measuring exactly this.
$src = "$hf$gguf"
if ($Leg -eq "gpu" -and $src -match "Q8_0") {
    Write-Host "[run] WARNING: Q8_0 on the Adreno leg -- measured HALF the throughput of Q4_K_M there. An exported LLAMA_HF/LLAMA_GGUF is overriding the gpu-leg default."
}
if ($Leg -eq "cpu" -and $src -match "Q4_K") {
    Write-Host "[run] WARNING: Q4_K on the CPU leg -- KleidiAI has no q4_K kernel (Q4_0/Q8_0 only), so this serves unaccelerated. An exported LLAMA_HF/LLAMA_GGUF is overriding the cpu-leg default."
}
if ($gguf -and -not (Test-Path $gguf)) {
    Write-Host "[run] GGUF does not exist: $gguf"
    Write-Host "[run] Fetch it:  hf download unsloth/Qwen3.5-9B-GGUF $(Split-Path -Leaf $gguf) --local-dir $(Split-Path -Parent $gguf)"
    exit 1
}
$bindHost = if ($env:LLAMA_HOST) { $env:LLAMA_HOST } else { "127.0.0.1" }
# The bind address is not always a dialable address: probing 0.0.0.0 fails on
# Windows, which would make the port pre-check inert AND the health loop blind
# -- ending with this launcher killing a perfectly healthy server at timeout.
$probeHost = if ($bindHost -eq "0.0.0.0" -or $bindHost -eq "::") { "127.0.0.1" } else { $bindHost }
$ctx     = if ($env:LLAMA_CTX)     { $env:LLAMA_CTX }     else { "64000" }
$threads = if ($env:LLAMA_THREADS) { $env:LLAMA_THREADS } else { "6" }
# Slot KV cache on disk, one dir per leg (slots from different models must not
# mix). Anchored under genie-npu rather than the CWD-relative `cache_slots` of
# the original hand-run command.
$slotDir = if ($env:LLAMA_SLOT_DIR) { $env:LLAMA_SLOT_DIR } else { Join-Path $root "cache_slots\$Leg" }
if (-not (Test-Path $slotDir)) { New-Item -ItemType Directory -Force $slotDir | Out-Null }

# Refuse a port something is already serving, for the same reason the Genie
# server does: on Windows two processes can both hold a port and the OLD one
# keeps answering, so a clean startup log proves nothing about who your
# requests reach (it happened -- see GENIE_SERVER.md). This names the PID
# instead of fighting it.
$busy = $false
try {
    $c = New-Object Net.Sockets.TcpClient
    $c.Connect($probeHost, [int]$port)
    $busy = $true; $c.Close()
} catch { }
if ($busy) {
    $owner = (Get-NetTCPConnection -LocalPort ([int]$port) -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
    Write-Host "[run] something is already serving ${probeHost}:${port} (pid $owner) -- refusing to double-bind."
    Write-Host "[run] Stop it first (Stop-Process $owner) or set LLAMA_PORT."
    exit 1
}

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory $logDir | Out-Null }
$log = Join-Path $logDir "llama-server-qwen3.5-9b-$Leg.log"

# Start-Process -ArgumentList under PS 5.1 joins elements with spaces and NO
# quoting, so a path containing a space shatters into two argv entries
# (probe-verified on this box). Quote anything path-shaped on its way in.
function Add-Quotes([string]$s) { if ($s -match "\s") { '"' + $s + '"' } else { $s } }

$srvArgs = @(
    "--jinja",
    "--ctx-size", $ctx,
    "--slot-save-path", (Add-Quotes $slotDir),
    "--slots",
    "--cache-ram", "16384",
    # Reasoning OFF for the same reason the Genie server suppresses Qwen3's
    # think block by default: 10-17x on an agent turn. preserve_thinking keeps
    # prior-turn thinking in the template so history replays byte-stable.
    "--reasoning", "off",
    "--chat-template-kwargs", '{\"preserve_thinking\":true}',
    "--cache-idle-slots",
    "--batch-size", "512",
    "--ubatch-size", "128",
    "--kv-unified",
    # Qwen's recommended non-thinking sampler; presence 1.5 is the repetition
    # control here (repeat-penalty stays 1.0 = off).
    "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0.0",
    "--presence-penalty", "1.5", "--repeat-penalty", "1.0",
    "--host", $bindHost, "--port", $port,
    "--parallel", "1",
    "--ctx-checkpoints", "8",
    "--checkpoint-min-step", "512",
    # Agent mode: CORS proxy + built-in tools. Local serving only -- the help
    # text itself says do not enable on an exposed host.
    "--agent"
)
if ($Leg -eq "cpu") {
    $srvArgs += @(
        # About half the cores: the all-cores default measured 2-5x slower on
        # this hardware (t6 26.2 t/s vs t12 11.9 at d0, on the 4B).
        "--threads", $threads, "--threads-batch", $threads,
        "--flash-attn", "on",
        # q8_0 KV halves cache bytes vs f16 -- at a 64000 window that is what
        # makes the window affordable on a shared 32 GB pool.
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"
    )
} else {
    # -fa auto and f16 KV on the GPU leg: the OpenCL backend's flash-attn and
    # quantised-KV support is partial, and forcing either can silently push
    # attention ops back to the CPU. Let the backend pick.
    #
    # -lv 5 because the placement check below reads the log, and at default
    # verbosity this build prints NO device line at all -- the check would
    # report "CPU-only" against a perfectly placed load.
    $srvArgs += @("--device", "GPUOpenCL", "-ngl", "99", "--flash-attn", "auto", "-lv", "5")
}
if ($alias) { $srvArgs += @("-a", (Add-Quotes $alias)) }
if ($gguf) { $srvArgs = @("-m", (Add-Quotes $gguf)) + $srvArgs }
else       { $srvArgs = @("-hf", $hf) + $srvArgs }
if ($env:LLAMA_EXTRA_ARGS) { $srvArgs += ($env:LLAMA_EXTRA_ARGS -split " ") }

Write-Host "[run] starting llama-server ($Leg leg) on ${bindHost}:${port}"
Write-Host "[run] model: $(if ($gguf) { $gguf } else { $hf + ' (HF cache; ~9.5 GB on first fetch)' })"
Write-Host "[run] log:   $log"
$proc = Start-Process -FilePath $server -ArgumentList $srvArgs `
    -RedirectStandardOutput $log -RedirectStandardError ($log + ".err") `
    -NoNewWindow -PassThru
# Load-bearing, not decoration: a PS 5.1 Process object from Start-Process has
# no cached process handle, and without one .ExitCode reads $null forever
# after the child dies (probe-verified here) -- so the crash paths below would
# print "exited ." and return success on a dead server. Touching .Handle once
# while the child is alive is what makes the real code readable later.
try { $null = $proc.Handle } catch { }

# Reads share-tolerantly (FileShare ReadWrite) because the redirect writer
# still holds the files -- [IO.File]::ReadAllText here threw "in use by
# another process", killed the streaming loop, and the finally then stopped a
# HEALTHY server. llama-server logs to STDERR, so both files are streamed.
function Read-NewText([string]$path, [long]$pos) {
    try {
        $fs = [IO.FileStream]::new($path, [IO.FileMode]::Open,
              [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        try {
            if ($fs.Length -le $pos) { return @($pos, "") }
            $fs.Position = $pos
            $sr = [IO.StreamReader]::new($fs)
            return @($fs.Length, $sr.ReadToEnd())
        } finally { $fs.Dispose() }
    } catch { return @($pos, "") }
}

# Wait for health. Generous by default because a first -hf run downloads the
# model before loading it. LLAMA_HEALTH_TIMEOUT (seconds) overrides.
$timeout = if ($env:LLAMA_HEALTH_TIMEOUT) { [int]$env:LLAMA_HEALTH_TIMEOUT } else { 1800 }
$up = $false
foreach ($i in 1..$timeout) {
    Start-Sleep -Milliseconds 1000
    if ($proc.HasExited) { break }
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 `
             "http://${probeHost}:${port}/health"
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
}
if (-not $up) {
    # Two different failures, two different messages: a server that DIED gets
    # its exit code and its last stderr lines surfaced (a bad -hf spec dies in
    # seconds -- calling that "not healthy after 1800s" sends the operator
    # hunting a timeout that never ran); one that is genuinely still silent
    # after the full window gets the timeout message.
    if ($proc.HasExited) {
        Write-Host "[run] llama-server exited $($proc.ExitCode) during startup. Last stderr:"
        $tail = Read-NewText ($log + ".err") 0
        if ($tail[1]) { ($tail[1] -split "`n") | Select-Object -Last 8 | ForEach-Object { Write-Host "  $_" } }
        Write-Host "[run] full logs: $log(.err)"
    } else {
        Write-Host "[run] server not healthy after ${timeout}s -- see $log"
        Stop-Process -Id $proc.Id -Force -Confirm:$false
    }
    exit 1
}

# Report placement rather than assuming it. A llama-server build has been
# observed on this box silently loading on CPU while asked for the GPU -- no
# error, requests answered, at CPU speed by the wrong engine. The device line
# in the log is the one signal worth trusting; a throughput number is not (a
# CPU-vs-GPU pair measured 0.2% apart here).
$dev = Select-String -Path $log, ($log + ".err") -Pattern "using device" -SimpleMatch -ErrorAction SilentlyContinue | Select-Object -First 2
if ($dev) { $dev | ForEach-Object { Write-Host "[run] placement: $($_.Line.Trim())" } }
else      { Write-Host "[run] placement: no 'using device' line found -- CPU-only load (KleidiAI)." }
if ($Leg -eq "gpu") {
    $onGpu = Select-String -Path $log, ($log + ".err") -Pattern "using device GPUOpenCL" -SimpleMatch -Quiet -ErrorAction SilentlyContinue
    if (-not $onGpu) {
        Write-Host "[run] WARNING: gpu leg requested but no 'using device GPUOpenCL' in the log."
        Write-Host "[run] Requests may be served by the CPU on the WRONG quant for it (Q4_K_M)."
        Write-Host "[run] Verify with the GPU engine-utilisation counter before trusting numbers."
    }
}
Write-Host "[run] up: http://${probeHost}:${port}/v1/chat/completions  (Ctrl-C stops it)"

# Stream the logs until the server exits or the operator Ctrl-Cs. finally runs
# on Ctrl-C in PowerShell, so the child does not outlive the launcher. One
# more drain after the loop, because a crash lands its most important lines --
# the reason -- in the gap between the last poll and the exit.
try {
    $posOut = 0L; $posErr = 0L
    while ($true) {
        $exited = $proc.HasExited
        foreach ($pair in @(@($log, "out"), @(($log + ".err"), "err"))) {
            $pos = if ($pair[1] -eq "out") { $posOut } else { $posErr }
            $r = Read-NewText $pair[0] $pos
            if ($r[1]) { Write-Host $r[1] -NoNewline }
            if ($pair[1] -eq "out") { $posOut = $r[0] } else { $posErr = $r[0] }
        }
        if ($exited) { break }
        Start-Sleep -Milliseconds 500
    }
    $code = $proc.ExitCode
    if ($null -eq $code) { $code = 1 }   # unreadable exit code is not success
    Write-Host "[run] llama-server exited $code."
    exit $code
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -Confirm:$false }
}
