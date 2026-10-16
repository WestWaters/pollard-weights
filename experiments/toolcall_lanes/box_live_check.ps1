<#
box_live_check.ps1 -- pollard-toolcall's in-process lanes, live, on the Windows box (RTX 5070 Ti, sm_120).

Runs the full tool-call gate (60 probes) on, in order, ONE model at a time:

  1. EXL3   C:\pollard\Qwen05-EXL3 (Qwen2.5-0.5B-Instruct, 4.0 bpw, head 6)  -- exllamav3 in-process
            vs C:\pollard\ptest\q05b-hf (its bf16 original)                     -- transformers in-process
  2. GPTQ   C:\pollard\bench\qwen7b-gptq-w4 (Qwen2.5-7B-Instruct, w4)          -- vLLM if a CUDA vLLM is installed,
            else transformers + gptqmodel in-process (the venv's `pip install vllm` has no compiled kernels)
            vs the Qwen2.5-7B-Instruct bf16 snapshot (15.2 GB: does not fit 16 GB of VRAM, so transformers runs it
            on the CPU with -Threads threads -- slow, it is the reference and runs once; cached afterwards)
  3. MX     the first compressed-tensors / FP8 / MXFP4 / NVFP4 rung found under C:\pollard (none on disk as of
            2026-10-15: logged and skipped)
  -Big adds EXL3 C:\pollard\ptest\q3b-sm4-calib3 (Qwen2.5-3B, smoothed 4 bpw) vs C:\pollard\ptest\q3b-hf.

It REFUSES to start (and refuses each next model) while the GPU is in use: any process on nvidia-smi that is not the
desktop shell or one of the always-on apps in $Ignore (a game -- RobloxPlayerBeta.exe, ImpostorLegacy.exe -- counts),
or GPU utilisation above 15 %, or more than 2.5 GB of VRAM in use. pollard-toolcall checks again itself and records a
rung as not-measured rather than take a busy GPU.

Results: $Out\results.json (what pollard-card / pollard-verify read), one <name>.json + <name>.log per rung.

ONE-TIME SETUP, over SSH as jwate (not under SYSTEM -- git refuses SYSTEM in jwate's clone):
    powershell -NoProfile -ExecutionPolicy Bypass -File <this file> -Setup
  creates the worktree C:\pollard\pw-tools2 on origin/feat/tool-aware-all-lanes (C:\pollard\pw is not touched).
  Bootstrap the very first copy of this file with scp from the Mac, or run the two git lines in Setup by hand.

RUN (when the GPU is free -- the son is not gaming), detached so it survives the SSH session:
    schtasks /create /tn PollardToolcallLanes /ru SYSTEM /sc once /sd 01/01/2026 /st 00:00 /f ^
      /tr "powershell -NoProfile -ExecutionPolicy Bypass -File C:\pollard\pw-tools2\experiments\toolcall_lanes\box_live_check.ps1"
    schtasks /run /tn PollardToolcallLanes
  (the start date is in the PAST on purpose: a future /st fires on its own -- see the 10-09 incident; /run starts it)
  check:   type C:\pollard\ptest\toolcall_lanes\status.txt
  done:    C:\pollard\ptest\toolcall_lanes\DONE.txt exists  ->  schtasks /delete /tn PollardToolcallLanes /f
#>
param(
    [string]$Repo = "C:\pollard\pw-tools2",
    [string]$Py = "C:\pollard\venv312\Scripts\python.exe",
    [string]$Out = "C:\pollard\ptest\toolcall_lanes",
    [int]$Limit = 0,                  # 0 = all 60 probes; N = the first N of each category
    [int]$Threads = 19,               # CPU threads for a CPU-side reference (60/40: 19 of 32)
    [switch]$Big,
    [switch]$Setup
)
$ErrorActionPreference = "Continue"

if ($Setup) {
    $git = if (Test-Path "C:\Program Files\Git\cmd\git.exe") { "C:\Program Files\Git\cmd\git.exe" } else { "git" }
    & $git -C C:\pollard\pw fetch origin feat/tool-aware-all-lanes
    if (-not (Test-Path $Repo)) {
        & $git -C C:\pollard\pw worktree add --detach $Repo origin/feat/tool-aware-all-lanes
    } else {
        & $git -C $Repo checkout --detach origin/feat/tool-aware-all-lanes
    }
    & $git -C $Repo log --oneline -1
    exit 0
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Status = Join-Path $Out "status.txt"
function Say($m) { $line = "$(Get-Date -Format s)  $m"; Write-Host $line; Add-Content -Path $Status -Value $line }
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Out "DONE.txt")

# ---- the toolchain (pollard_exl3_blackwell_toolchain): CUDA 12.8 to match torch cu128, MSVC for exllamav3's JIT
#      extension (already built in jwate's torch_extensions cache -- point SYSTEM at it so nothing recompiles)
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_PATH = $cuda; $env:CUDA_HOME = $cuda
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:TORCH_EXTENSIONS_DIR = "C:\Users\jwate\AppData\Local\torch_extensions\torch_extensions\Cache"
$msvc = Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
$sdk = "C:\Program Files (x86)\Windows Kits\10"
$sdkv = (Get-ChildItem "$sdk\bin" -Directory -ErrorAction SilentlyContinue | Where-Object Name -like "10.*" |
         Sort-Object Name -Descending | Select-Object -First 1).Name
if ($msvc -and $sdkv) {
    $m = $msvc.FullName
    $env:PATH = "$m\bin\Hostx64\x64;$sdk\bin\$sdkv\x64;$cuda\bin;$env:PATH"
    $env:INCLUDE = "$m\include;$sdk\Include\$sdkv\ucrt;$sdk\Include\$sdkv\um;$sdk\Include\$sdkv\shared"
    $env:LIB = "$m\lib\x64;$sdk\Lib\$sdkv\ucrt\x64;$sdk\Lib\$sdkv\um\x64"
} else {
    $env:PATH = "$cuda\bin;$env:PATH"
}
$env:HF_HOME = "C:\Users\jwate\.cache\huggingface"     # SYSTEM has its own profile
$env:POLLARD_HOME = Join-Path $Out "home"               # reference cache here; the box workspace is not touched
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$Repo\tools"
# always-on desktop apps seen on this box's GPU table (2026-10-15) -- NOT games; a game still blocks the run
$Ignore = "discord,medal,steamwebhelper,voicemod,opera.exe,spotify,razer central,mscopilot,snippingtool," +
          "widgetboard,googleplaygamesservices"
$env:POLLARD_GPU_IGNORE = $Ignore

function GpuBusy {
    $users = & $Py -c "import pollard_toolcall as T; print(','.join(T.gpu_users()))" 2>$null
    $q = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) -split ","
    $util = [int]($q[0].Trim()); $used = [int]($q[1].Trim())
    if ($users) { return "GPU in use by: $users" }
    if ($util -gt 15) { return "GPU utilisation $util %" }
    if ($used -gt 2560) { return "$used MiB of VRAM in use" }
    return $null
}

function Gate($name, $rung, $ref) {
    # returns $false only when the GPU was busy (stop the queue); a missing model is skipped and the queue goes on
    if (-not (Test-Path $rung)) { Say "skip $name : $rung not on disk"; return $null }
    $busy = GpuBusy
    if ($busy) { Say "REFUSED before $name : $busy -- not ours to take"; return $false }
    Say "start $name : $rung  vs  $(if ($ref) { $ref } else { '(no reference: absolute rate)' })"
    $a = @("$Repo\tools\pollard_toolcall.py", "--model", $rung, "--threads", "$Threads",
           "--results", (Join-Path $Out "results.json"), "--rung", $name, "--out", (Join-Path $Out "$name.json"),
           "--verbose")
    if ($ref) { $a += @("--ref", $ref) }
    if ($Limit -gt 0) { $a += @("--limit", "$Limit") }
    & $Py @a *> (Join-Path $Out "$name.log")
    Say "end   $name : exit $LASTEXITCODE  (log $name.log)"
    return $true
}

$busy = GpuBusy
if ($busy) { Say "REFUSED to start: $busy"; exit 2 }
$vers = & $Py -c "import importlib.metadata as m; print(' '.join(p + '=' + m.version(p) for p in ('exllamav3', 'torch', 'transformers', 'gptqmodel', 'compressed-tensors')))" 2>$null
Say "pollard-toolcall in-process lanes  repo=$Repo  limit=$Limit  $vers"

$ok = Gate "Qwen2.5-0.5B-Instruct-EXL3-4.0bpw" "C:\pollard\Qwen05-EXL3" "C:\pollard\ptest\q05b-hf"
if ($ok -ne $false -and $Big) {
    $ok = Gate "Qwen2.5-3B-Instruct-EXL3-4.0bpw-sm-calib3" "C:\pollard\ptest\q3b-sm4-calib3" "C:\pollard\ptest\q3b-hf"
}
if ($ok -ne $false) {
    $ref7 = Get-ChildItem "C:\Users\jwate\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots" -Directory |
            Select-Object -First 1
    $ok = Gate "Qwen2.5-7B-Instruct-GPTQ-w4" "C:\pollard\bench\qwen7b-gptq-w4" $ref7.FullName
}
if ($ok -ne $false) {
    $mx = & $Py -c @"
import os, pollard_toolcall as T
for root in (r'C:\pollard', r'C:\Users\jwate\pollard'):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ('venv312', 'llama.cpp', 'ik_llama.cpp', '.git', 'node_modules')]
        if dp.count(os.sep) - root.count(os.sep) > 4: dn[:] = []; continue
        if 'config.json' in fn and T.lane_of(dp) == 'mx':
            print(dp); raise SystemExit
"@ 2>$null
    if ($mx) { $ok = Gate ("MX-" + (Split-Path $mx -Leaf)) $mx $null }   # its bf16 source is not known here
    else { Say "skip MX : no compressed-tensors / FP8 / MXFP4 / NVFP4 rung on disk" }
}
Say "done"
Set-Content -Path (Join-Path $Out "DONE.txt") -Value (Get-Date -Format s)
