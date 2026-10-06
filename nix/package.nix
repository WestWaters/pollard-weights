{
  lib,
  python3,
  git,
}:
let
  project = (builtins.fromTOML (builtins.readFile ../pyproject.toml)).project;
in
python3.pkgs.buildPythonApplication {
  pname = project.name;
  inherit (project) version;
  pyproject = true;

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../pyproject.toml
      ../README.md
      ../LICENSE
      (lib.fileset.fileFilter (
        file:
        builtins.any file.hasExt [
          "py"
          "html"
          "css"
          "js"
          "png"
        ]
      ) ../tools)
      (lib.fileset.fileFilter (file: file.hasExt "py") ../tests)
      (lib.fileset.fileFilter (file: file.hasExt "py") ../experiments)
    ];
  };
  build-system = [ python3.pkgs.setuptools ];
  dependencies = with python3.pkgs; [
    numpy
    gguf
  ];

  nativeCheckInputs = [
    python3.pkgs.pytestCheckHook
    git
  ];
  # The core package excludes GUI dependencies; Studio has its own pip extra.
  # Optional tensor fixtures skip here when torch is absent.
  pytestFlags = [
    "tests"
    "--ignore=tests/studio"
  ];
  pythonImportsCheck = [
    "pollard_calc"
    "pollard_serve_eval"
  ];

  meta = {
    inherit (project) description;
    homepage = "https://github.com/WestWaters/pollard-weights";
    license = lib.licenses.asl20;
    mainProgram = "pollard";
    platforms = lib.platforms.unix;
  };
}
