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
# The bind address is not always a dialable address: the health probe cannot
# dial 0.0.0.0 on Windows, so probe loopback when binding wildcard.
$probeHost = if ($bindHost -eq "0.0.0.0" -or $bindHost -eq "::") { "127.0.0.1" } else { $bindHost }
$ctx     = if ($env:LLAMA_CTX)     { $env:LLAMA_CTX }     else { "64000" }
$threads = if ($env:LLAMA_THREADS) { $env:LLAMA_THREADS } else { "6" }
# Health timeout: parsed and clamped HERE, before any child process exists. A
# junk value used to throw at the [int] cast AFTER Start-Process -- the
# launcher died and the just-started server was orphaned, unsupervised. And a
# 0 could never mean "no wait": PS ranges descend (1..0 is TWO iterations),
# so it produced a ~2s wait and a kill. Floor of 1s; junk falls back loudly.
$timeout = 0
if ($env:LLAMA_HEALTH_TIMEOUT) {
    if (-not [int]::TryParse($env:LLAMA_HEALTH_TIMEOUT, [ref]$timeout) -or $timeout -lt 1) {
        Write-Host "[run] WARNING: LLAMA_HEALTH_TIMEOUT='$($env:LLAMA_HEALTH_TIMEOUT)' is not a positive integer (seconds); using 1800."
        $timeout = 1800
    }
} else { $timeout = 1800 }
# Slot KV cache on disk, one dir per leg (slots from different models must not
# mix). Anchored under genie-npu rather than the CWD-relative `cache_slots` of
# the original hand-run command.
$slotDir = if ($env:LLAMA_SLOT_DIR) { $env:LLAMA_SLOT_DIR } else { Join-Path $root "cache_slots\$Leg" }
if (-not (Test-Path $slotDir)) { New-Item -ItemType Directory -Force $slotDir | Out-Null }

# Refuse a port something is already serving, for the same reason the Genie
# server does: on Windows two processes can both hold a port and the OLD one
# keeps answering, so a clean startup log proves nothing about who your
# requests reach (it happened -- see GENIE_SERVER.md). Checked via the
# listener table, not a dial: a dial to loopback cannot see a listener bound
# to a single non-loopback interface, and a wildcard bind coexists with such
# a listener silently. A conflict is: we bind wildcard and ANYTHING listens
# on the port, or something listens on wildcard, or on our exact address.
$wildcards = @("0.0.0.0", "::")
$conflict = Get-NetTCPConnection -LocalPort ([int]$port) -State Listen -ErrorAction SilentlyContinue |
    Where-Object { ($bindHost -in $wildcards) -or ($_.LocalAddress -in $wildcards) -or ($_.LocalAddress -eq $bindHost) } |
    Select-Object -First 1
if ($conflict) {
    Write-Host "[run] something is already listening on $($conflict.LocalAddress):${port} (pid $($conflict.OwningProcess)) -- refusing to double-bind."
    Write-Host "[run] Stop it first (Stop-Process $($conflict.OwningProcess)) or set LLAMA_PORT."
    exit 1
}

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory $logDir | Out-Null }
# Port in the name, not just the leg: the refusal above says "set LLAMA_PORT"
# to run a second instance, and two same-leg instances sharing one log file
# would have the second TRUNCATE the first's live log (verified: the redirect
# open succeeds against the in-use file).
$log = Join-Path $logDir "llama-server-qwen3.5-9b-$Leg-$port.log"

# Start-Process -ArgumentList under PS 5.1 joins elements with spaces and NO
# quoting, so a path containing a space shatters into two argv entries
# (probe-verified on this box). Quote anything path-shaped on its way in --
# and double any TRAILING backslashes first, because '...\' + '"' reaches the
# child CRT as an escaped quote: the region never closes and every following
# flag is swallowed into the value (also probe-verified).
function Add-Quotes([string]$s) {
    if ($s -notmatch "\s") { return $s }
    '"' + ($s -replace '(\\+)$', '$1$1') + '"'
}

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
    # verbosity this build prints NO device line at all (verified: a default-
    # verbosity launch had no 'using device' line to find) -- the check would
    # report "CPU-only" against a perfectly placed load. The debug flood this
    # buys is kept OUT of the console by the D-line filter in the streamer;
    # the log file keeps everything for forensics.
    $srvArgs += @("--device", "GPUOpenCL", "-ngl", "99", "--flash-attn", "auto", "-lv", "5")
}
if ($alias) { $srvArgs += @("-a", (Add-Quotes $alias)) }
if ($gguf) { $srvArgs = @("-m", (Add-Quotes $gguf)) + $srvArgs }
else       { $srvArgs = @("-hf", $hf) + $srvArgs }
if ($env:LLAMA_EXTRA_ARGS) { $srvArgs += ($env:LLAMA_EXTRA_ARGS -split " ") }

Write-Host "[run] starting llama-server ($Leg leg) on ${bindHost}:${port}"
Write-Host "[run] model: $(if ($gguf) { $gguf } else { $hf + ' (HF cache; ~9.5 GB on first fetch)' })"
Write-Host "[run] log:   $log(.err)"
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
# HEALTHY server. Returns the POSITION ReadToEnd actually consumed, not a
# pre-read Length snapshot: Length is sampled before the read while ReadToEnd
# drains to the live EOF, so a mid-read append came back inside the text AND
# below the returned position -- and was printed twice on the next poll
# (probe-verified). llama-server logs to STDERR, so both files are streamed.
function Read-NewText([string]$path, [long]$pos) {
    try {
        $fs = [IO.FileStream]::new($path, [IO.FileMode]::Open,
              [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        try {
            if ($fs.Length -le $pos) { return @($pos, "") }
            $fs.Position = $pos
            $sr = [IO.StreamReader]::new($fs)
            $text = $sr.ReadToEnd()
            return @($fs.Position, $text)
        } finally { $fs.Dispose() }
    } catch { return @($pos, "") }
}

# Console streamer over both log files, line-buffered so the gpu leg's
# debug-level filter sees whole lines (a chunk boundary mid-line would
# otherwise leak fragments). At -lv 5 the server writes 15-20 "D"-severity
# lines per decoded token; those stay in the file and out of the console.
$stream = @{ outPos = 0L; errPos = 0L; outCarry = ""; errCarry = "" }
$filterDebug = ($Leg -eq "gpu")
function Drain-Logs {
    foreach ($k in @("out", "err")) {
        $path = if ($k -eq "out") { $log } else { $log + ".err" }
        $r = Read-NewText $path $stream["${k}Pos"]
        $stream["${k}Pos"] = $r[0]
        if (-not $r[1]) { continue }
        $data = $stream["${k}Carry"] + $r[1]
        $nl = $data.LastIndexOf("`n")
        if ($nl -lt 0) { $stream["${k}Carry"] = $data; continue }
        $stream["${k}Carry"] = $data.Substring($nl + 1)
        $lines = $data.Substring(0, $nl + 1)
        if ($filterDebug) { $lines = $lines -replace '(?m)^\S+ D .*\r?\n', '' }
        if ($lines) { Write-Host $lines -NoNewline }
    }
}
function Flush-Carry {
    foreach ($k in @("out", "err")) {
        if ($stream["${k}Carry"]) { Write-Host $stream["${k}Carry"]; $stream["${k}Carry"] = "" }
    }
}

# Wait for health on a wall-clock deadline (an iteration-counted loop ran up
# to ~3x the stated timeout: each pass is 1s of sleep PLUS up to 2s of probe
# timeout). The server's own output streams throughout, so a first -hf run's
# ~9.5 GB download and the model load are visible progress, not silence.
$deadline = (Get-Date).AddSeconds($timeout)
$up = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 1000
    if ($proc.HasExited) { break }
    Drain-Logs
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 `
             "http://${probeHost}:${port}/health"
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
}
if (-not $up) {
    # Everything the server said is already on the console (streamed above);
    # distinguish the two failures and name the knob instead of misreporting
    # a death-in-seconds as an expired timeout.
    Drain-Logs; Flush-Carry
    if ($proc.HasExited) {
        $code = $proc.ExitCode
        if ($null -eq $code) { $code = 1 }
        Write-Host "[run] llama-server exited $code during startup -- see its stderr above; full logs: $log(.err)"
    } else {
        Write-Host "[run] server not healthy after ${timeout}s (LLAMA_HEALTH_TIMEOUT, default 1800)."
        Write-Host "[run] If the lines above show a first-run -hf download still in progress, raise"
        Write-Host "[run] LLAMA_HEALTH_TIMEOUT or pre-download with 'hf download'. Stopping the server; logs: ${log}.err"
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
# more drain after the exit flag is seen, because a crash lands its most
# important lines -- the reason -- in the gap between the last poll and the
# exit; the carry flush gets a final unterminated line out too.
try {
    while ($true) {
        $exited = $proc.HasExited
        Drain-Logs
        if ($exited) { break }
        Start-Sleep -Milliseconds 500
    }
    Flush-Carry
    $code = $proc.ExitCode
    if ($null -eq $code) { $code = 1 }   # unreadable exit code is not success
    Write-Host "[run] llama-server exited $code."
    exit $code
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -Confirm:$false }
}
