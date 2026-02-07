{
  description = "Alerta Escuela Equity Audit - independent equity audit of Peru's dropout prediction system";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Runtime libraries for Python manylinux wheels
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib      # libstdc++.so.6
        libgcc.lib            # libgomp.so.1 (OpenMP for LightGBM)
        zlib                  # libz.so (various wheels)
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          ruff
          cmake
        ];

        env = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        };

        shellHook = ''
          echo "Alerta Escuela Equity Audit"
          echo "Python: $(python3 --version)"
          echo "uv: $(uv --version)"
          echo ""
          echo "Run 'uv sync' to install Python dependencies"
        '';
      };
    };
}
