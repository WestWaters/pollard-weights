# Setup on the box (one time, CPU/network only)

Everything goes into a **separate** venv, `C:\pollard\tpd-venv`. `C:\pollard\venv312` is never touched.

## 1. Copy the kit to the box

From the Mac (the reverse tunnel is already up):

```bash
scp -P 3333 -i ~/.ssh/pollard_5070ti -r ~/Desktop/Pollard-Weights/experiments/tpd_quant jwate@localhost:C:/pollard/tpd_quant
```

## 2. Run the setup script

```powershell
cd C:\pollard\tpd_quant
powershell -ExecutionPolicy Bypass -File setup.ps1
```

What it does:

1. Installs `uv` if it isn't on PATH (user-level installer, no admin needed).
2. Clones `https://github.com/Antovigo/targeted-parameter-decomposition` to `C:\pollard\tpd` and checks out
   `edbfb6c` (the commit the kit was smoke-tested against). Pass `-Commit ""` to stay on the latest `main`.
3. `uv venv --python 3.13 C:\pollard\tpd-venv`. tPD declares `requires-python = "==3.13.*"`, so venv312
   couldn't be used even if we wanted to. uv downloads a managed CPython 3.13 if the box doesn't have one.
4. Installs `torch==2.8.0` and `torchvision==0.23.0` from `https://download.pytorch.org/whl/cu128` (the same
   wheels index venv312 uses). These are the versions tPD's `uv.lock` pins, but the lock gets them from PyPI,
   and on Windows that's the CPU build.
5. Installs everything else from tPD's `uv.lock` (via `uv export`, with the torch lines removed), then
   `tPD` itself with `--no-deps -e`.
6. Checks the result: the cu128 build, `sm_120` in the arch list, and that `import spd.run_spd` works. It
   doesn't allocate anything on the GPU.

Why the lock matters: an unpinned `pip install -e .` pulls wandb 0.30, which no longer ships `wandb_gql`,
and `import spd` crashes. The Mac smoke test hit exactly this. The lock pins wandb 0.23.1, transformers
4.57.3 and datasets 4.4.2.

## 3. Build the data (CPU only, can run any time)

```powershell
powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage prep
```

This produces `C:/pollard/tpd_data/{target_code,nontarget_mix}/data/*.parquet` and `eval_sets.pt`. It
downloads about 0.5 GB of text: `codeparrot/codeparrot-clean-valid`, `Salesforce/wikitext`
(wikitext-103-raw-v1) and `HuggingFaceH4/ultrachat_200k`. Tokenizing is single-process and takes a few
minutes. All three datasets are ungated, so no HF token is needed.

## Manual equivalent (if the script fails partway)

```powershell
uv venv --python 3.13 C:\pollard\tpd-venv
$Py = "C:\pollard\tpd-venv\Scripts\python.exe"
uv pip install --python $Py torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
cd C:\pollard\tpd
uv export --frozen --no-dev --no-hashes --no-emit-project -o C:\pollard\tpd-venv\reqs.txt
(Get-Content C:\pollard\tpd-venv\reqs.txt) | ? { $_ -notmatch '^(torch|torchvision)==' } | Set-Content C:\pollard\tpd-venv\reqs-notorch.txt
uv pip install --python $Py -r C:\pollard\tpd-venv\reqs-notorch.txt
uv pip install --python $Py --no-deps -e C:\pollard\tpd
```
