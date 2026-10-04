# End-to-end tPD-quant pipeline on the box. GPU stages run ONLY when Mario says the GPU is free.
#
#   powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage prep            # CPU only, any time
#   powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage preflight       # GPU, ~2 min
#   powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage decomp          # GPU, hours
#   powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage arms            # GPU, ~15-30 min
#   powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage gpu             # preflight+decomp+arms
#
# Detached (survives the SSH session; see README):
#   schtasks /Create /TN tpd_quant /SC ONCE /ST 23:59 /F /TR "powershell -ExecutionPolicy Bypass -File C:\pollard\tpd_quant\run_box.ps1 -Stage gpu"
#   schtasks /Run /TN tpd_quant
param(
    [ValidateSet("prep", "preflight", "decomp", "arms", "gpu")] [string]$Stage = "prep",
    [int]$Seed = 0,                 # decomposition seed (run a 2nd decomposition with -Seed 1 if wanted)
    [int]$Batch = 16,               # drop to 8 if preflight says TOO CLOSE / OOM
    [int]$HumanEval = 0,            # N HumanEval problems in the arms stage (executes model code)
    [switch]$Force,                 # skip the "someone else is on the GPU" guard
    [string]$Kit = $PSScriptRoot,
    [string]$Py = "C:\pollard\tpd-venv\Scripts\python.exe",
    [string]$Data = "C:/pollard/tpd_data",
    [string]$Out = "C:/pollard/tpd_out"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Log = Join-Path $Out ("run_box_{0}_s{1}_{2}.log" -f $Stage, $Seed, (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $Log | Out-Null
$RunId = "qwen3code-s$Seed"
$env:PYTHONUNBUFFERED = "1"
$env:WANDB_MODE = "disabled"

function Assert-GpuFree {
    if ($Force) { Write-Host "GPU guard skipped (-Force)"; return }
    $apps = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null
    $used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    Write-Host "nvidia-smi: memory.used=${used} MiB; compute apps: $apps"
    if ($apps -or $used -gt 1500) {
        throw "GPU is in use (son's session?). Not starting. Re-run when Mario says the GPU is free, or pass -Force."
    }
}

function Run-Prep {
    if (Test-Path "$Data/eval_sets.pt") { Write-Host "prep: $Data/eval_sets.pt exists, skipping"; return }
    & $Py "$Kit\prep_data.py" --out $Data
    if ($LASTEXITCODE) { throw "prep_data failed" }
}

function Run-Preflight {
    Assert-GpuFree
    & $Py "$Kit\run_tpd.py" --data-root $Data --out-dir $Out --run-id "$RunId-preflight" --seed $Seed --batch $Batch --preflight
    if ($LASTEXITCODE) { throw "preflight failed (OOM? retry with -Batch 8)" }
}

function Run-Decomp {
    Assert-GpuFree
    & $Py "$Kit\run_tpd.py" --data-root $Data --out-dir $Out --run-id $RunId --seed $Seed --batch $Batch
    if ($LASTEXITCODE) { throw "decomposition failed" }
}

function Run-Arms {
    Assert-GpuFree
    $ckpt = Get-ChildItem "$Out/spd/$RunId/model_*.pth" | Sort-Object { [int]($_.BaseName -replace 'model_', '') } | Select-Object -Last 1
    if (-not $ckpt) { throw "no checkpoint in $Out/spd/$RunId" }
    Write-Host "arms: using $($ckpt.FullName)"
    & $Py "$Kit\quant_arms.py" --model Qwen/Qwen3-0.6B --decomp $ckpt.FullName --eval-sets "$Data/eval_sets.pt" `
        --out "$Out/results/$RunId" --seeds 0 1 --humaneval $HumanEval
    if ($LASTEXITCODE) { throw "quant_arms failed" }
}

try {
    switch ($Stage) {
        "prep"      { Run-Prep }
        "preflight" { Run-Prep; Run-Preflight }
        "decomp"    { Run-Decomp }
        "arms"      { Run-Arms }
        "gpu"       { Run-Prep; Run-Preflight; Run-Decomp; Run-Arms }
    }
    Write-Host "== $Stage done"
} finally {
    Stop-Transcript | Out-Null
}
