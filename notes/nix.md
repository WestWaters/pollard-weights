# Nix installation

The flake packages the core Python tools with numpy and gguf. It does not run
`install.sh`, modify your system or fetch models. `flake.lock` pins Nixpkgs and
the Python dependency set. The package source is limited to packaging metadata,
tools, tests and Python experiments, excluding runtime logs and local caches.
Use a clean checkout: Nix flake inputs themselves can enter the world-readable
store before the package source filter runs. Never keep secrets in tracked files.

From this checkout:

```sh
nix build                         # result/bin contains the commands
nix run . -- --help                # pollard
nix run .#calc -- --help           # pollard-calc
nix profile install .             # install the core commands in your profile
nix flake check                   # core tests and installed-command checks
```

To add the package to a NixOS configuration, add this repository as a flake input
and use `inputs.pollard.packages.${pkgs.stdenv.hostPlatform.system}.default` in
`environment.systemPackages`. No overlay is required; `overlays.default` is also
available for configurations that prefer it.

For source development:

```sh
nix develop
python tools/pollard_calc.py --help
pytest tests --ignore=tests/studio
nix fmt -- flake.nix nix/package.nix
```

The development shell supplies core Python dependencies and pytest, not an
editable installation. Run source files with Python to test local edits.

## Runtime boundaries

This is a core CLI package, not a CUDA-enabled inference or quantization stack.
Like a core pip installation, entry points for optional tools are present, but
Studio and the Torch/GPTQ/EXL3/MX/MLX paths need their respective dependencies.
Their presence in `result/bin` is not a claim that those paths work. The Nix
checks exclude Studio's tests and skip optional tensor fixtures when Torch is
absent. Studio and GPU extras require separate test environments; installing
this core package does not validate them.

llama.cpp is separate too: supply a compatible build with the executables your
chosen lane requires on PATH. This flake does not enable RPC, choose a GPU
backend, or guarantee compatibility with the project's current runtime recipes.

Pollard launches several helpers with its own `sys.executable`. A host Python
environment containing Torch does not automatically add Torch to this package's
Nix Python environment. Do not use `pip install --user` or force another
interpreter on the wrappers as a workaround. GPU extras need an explicitly
packaged and tested Python environment, with driver/runtime compatibility
checked separately. That is outside this core installation change.

The flake exposes Linux x86_64/ARM64 and macOS ARM64 outputs. Intel macOS is
excluded because the pinned Nixpkgs branch no longer supports it. Native build
results should be reported separately from evaluation of another platform's
derivation; evaluation alone is not a successful build.
