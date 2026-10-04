# tPD kit setup for the Blackwell box (RTX 5070 Ti). CPU/network only - does not use the GPU.
#
# Creates a SEPARATE venv (C:\pollard\tpd-venv). Never touches C:\pollard\venv312.
#   - tPD pins requires-python ==3.13.* -> uv fetches a managed CPython 3.13 if none is installed.
#   - tPD's uv.lock pins torch 2.8.0 from PyPI, which on Windows is the CPU build. So torch/torchvision
#     come from the cu128 index (same wheels index as venv312), everything else from the lock.
#     A plain `pip install -e .` is NOT ok: it resolves wandb 0.30 which lacks `wandb_gql` and
#     `import spd` dies (seen in the Mac smoke test).
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-Commit <sha>]
param(
    [string]$TpdDir = "C:\pollard\tpd",
    [string]$Venv = "C:\pollard\tpd-venv",
    # Commit the kit was smoke-tested against (2026-06-16, "Disable Dependabot on public release").
    [string]$Commit = "edbfb6c66058d0d9912607b46d2b59319c1e49f8"
)
$ErrorActionPreference = "Stop"
if ($Venv -match "venv312") { throw "Refusing: tPD must not be installed into venv312." }

# 1. uv (user-level install; no admin)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "== installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

# 2. clone / update tPD
if (-not (Test-Path "$TpdDir\.git")) {
    Write-Host "== cloning tPD -> $TpdDir"
    git clone https://github.com/Antovigo/targeted-parameter-decomposition $TpdDir
} else {
    git -C $TpdDir fetch --quiet origin
}
if ($Commit) { git -C $TpdDir checkout --quiet $Commit }
Write-Host ("== tPD at " + (git -C $TpdDir log -1 --format="%h %cd %s"))

# 3. venv (Python 3.13)
if (-not (Test-Path "$Venv\Scripts\python.exe")) { uv venv --python 3.13 $Venv }
$Py = "$Venv\Scripts\python.exe"

# 4. torch + torchvision cu128 (versions = tPD's lock: torch 2.8.0 / torchvision 0.23.0; sm_120 OK)
uv pip install --python $Py "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128

# 5. everything else exactly as tPD's uv.lock (minus torch/torchvision), then tPD itself
Push-Location $TpdDir
uv export --frozen --no-dev --no-hashes --no-emit-project -o "$Venv\tpd-lock-reqs.txt"
Pop-Location
Get-Content "$Venv\tpd-lock-reqs.txt" |
    Where-Object { $_ -notmatch '^(torch|torchvision)==' } |
    Set-Content "$Venv\tpd-lock-reqs-notorch.txt"
uv pip install --python $Py -r "$Venv\tpd-lock-reqs-notorch.txt"
uv pip install --python $Py --no-deps -e $TpdDir

# 6. verify (no GPU allocation: only reports the build)
& $Py -c @"
import torch, transformers, wandb, spd.run_spd
print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| arch', torch.cuda.get_arch_list())
assert torch.version.cuda and torch.version.cuda.startswith('12.8'), 'torch is not the cu128 build'
assert 'sm_120' in torch.cuda.get_arch_list(), 'no sm_120 (Blackwell) kernels in this torch'
print('transformers', transformers.__version__, '| wandb', wandb.__version__, '| spd import OK')
"@
Write-Host "== setup OK: $Py"
