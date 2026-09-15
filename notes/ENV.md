# Environment decisions (2026-09-15)

## Dev env (active)
- System Python 3.14.4, torch 2.10.0+cu128 (user site, ~/.local), triton 3.6.0.
- exllamav3 v1.5.0 fork: `pip3 install --user --break-system-packages --no-build-isolation -e .`
  (PEP 668 blocks plain installs; the wheel build failed but the PEP 660 editable link works
  and the extension JIT-loads fine).
- Reference package: `pip3 install --user --break-system-packages dflash datasets`.
- Used for: engine dev, all pytest runs so far (25/25 DFlash2 + existing generator tests green).

## Repro env (for final validation + container parity)
- `~/exl3-qwen38-dflash2/env` — uv venv, Python 3.13.
- IMPORTANT: `--torch-backend=auto` resolved to torch 2.10.0+cu130 on this host (CUDA 13.0
  toolkit visible) which libcudart-mismatches the exllamav3 +cu128 wheel. Fixed by pinning
  `--index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch`.
- The container Dockerfile pins the cu128 index for the same reason.

## GPU
- RTX 3090 24 GB, driver 595.84. CUDA toolkit at ~/cuda-13.0 (nvcc 13.0) used only for the
  dev-env JIT build.
