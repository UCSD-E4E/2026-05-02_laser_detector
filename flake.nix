{
  description = "laser-detector dev shell — uv + Python 3.13 with host CUDA driver libs";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Shared libs that the PyPI wheels (torch, torchvision, opencv) dlopen at
      # import time. On NixOS these aren't on a default search path, so we add
      # them via LD_LIBRARY_PATH. Deliberately NOT including glibc — mixing it
      # into LD_LIBRARY_PATH causes GLIBC version mismatches with the nix python.
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib # libstdc++, libgcc_s, libgomp
        zlib # torch, numpy
        glib # libgthread-2.0 (opencv)
        libGL # libGL (opencv-python, non-headless)
        # X11/XCB libs that opencv-python's highgui links against (needed even
        # for headless `import cv2`, which dlopens them at load time).
        libxcb
        libx11
        libxext
        libsm
        libice
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.uv
          pkgs.python313
        ];

        # Use the nix-provided interpreter; never let uv download a standalone
        # CPython — those are generic builds that need nix-ld/patchelf to run.
        env = {
          UV_PYTHON = "${pkgs.python313}/bin/python3.13";
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON_PREFERENCE = "only-system";
        };

        shellHook = ''
          # libcuda.so is provided by the host NVIDIA driver, not by a wheel.
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          # Keep the venv (and triton's JIT cache) on the same fast local fs as
          # uv's download cache (XDG_CACHE_HOME, already on /local here). Same fs
          # => uv hardlinks instead of copying GBs onto NFS, and imports read
          # from local NVMe rather than the shared NFS server. The venv is
          # disposable (reproducible via `uv sync`), so local/ephemeral is fine.
          CACHE_BASE="''${XDG_CACHE_HOME:-$HOME/.cache}"
          export UV_PROJECT_ENVIRONMENT="$CACHE_BASE/uv-venvs/2026-05-02_laser_detector"
          export TRITON_CACHE_DIR="$CACHE_BASE/triton"
          mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$TRITON_CACHE_DIR"

          echo "laser-detector devShell — uv $(uv --version 2>/dev/null), $(python3.13 --version)"
          echo "venv: $UV_PROJECT_ENVIRONMENT"
          echo "First time / after rebuild:  uv sync"
          echo "Verify torch+driver:         uv run python -c 'import torch, cv2; print(torch.__version__, torch.cuda.is_available())'"
        '';
      };
    };
}
