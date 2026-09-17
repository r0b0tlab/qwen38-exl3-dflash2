# Environment decisions

## Runtime (2026-09-17, native community engine)

- `~/exl3-qwen38-dflash2/env` — uv venv, Python 3.13.13.
- torch 2.10.0+cu130. The community extension links `libcudart.so.13`; put
  `/home/am/cuda-13.0/lib64` (or the CUDA 13 runtime lib) on `LD_LIBRARY_PATH` if
  import fails with `libcudart.so.13`.
- Engine: `PYTHONPATH=$PWD/exllamav3` against r0b0tlab/exllamav3 `community` @ `355c6ee`.
  Six EXL3 GEMM kernel shapes (upstream 1–4 plus dense M=32/64).
- 3090 knobs (defaults, do not override unless A/B testing):
  - `EXL3_HGEMM_F16ACC=auto` — fp16-accumulator MMA, measured 2.00x on sm_86
  - `EXL3_INT8_GEMV=2` with `EXL3_INT8_GEMV_MAX_K=5` (Ampere is DRAM-bound at K>=6)
  - `EXL3_QC_STAGING=1`
- Do not overlay `dflash2-pathway` Python onto the v1.5.0 wheel. Native DFlash2 CUDA
  lives in the community extension.

## Dev env (overlay era, 2026-09-15, superseded)

- System Python 3.14.4, torch 2.10.0+cu128 (user site), triton 3.6.0.
- Used to build the `dflash2-pathway` overlay. Kept only as the conversion-time
  record; new work uses the venv above.

## GPU

- RTX 3090 24 GB, driver 595.84. CUDA toolkit at ~/cuda-13.0 (nvcc 13.0) for the
  community extension build.
