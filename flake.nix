{
  description = "Pollard Weights core CLI and development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      overlays.default = final: prev: {
        pollard-weights = final.callPackage ./nix/package.nix { };
      };

      packages = forAllSystems (pkgs: rec {
        pollard-weights = pkgs.callPackage ./nix/package.nix { };
        default = pollard-weights;
      });

      apps = forAllSystems (pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/pollard";
          meta.description = "Pollard Weights automatic planning CLI";
        };
        calc = {
          type = "app";
          program = "${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/pollard-calc";
          meta.description = "Pollard model memory calculator";
        };
      });

      checks = forAllSystems (pkgs: {
        package = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
        installed-cli =
          pkgs.runCommand "pollard-installed-cli"
            {
              nativeBuildInputs = [
                self.packages.${pkgs.stdenv.hostPlatform.system}.default
                pkgs.python3
              ];
            }
            ''
              mkdir -p work
              cd work
              unset PYTHONPATH
              ${pkgs.python3}/bin/python ${./nix/check-installed.py}
              touch "$out"
            '';
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: [
              ps.numpy
              ps.gguf
              ps.pytest
              ps.setuptools
              ps.build
            ]))
            pkgs.git
            pkgs.nixfmt
          ];
        };
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt);
    };
}
