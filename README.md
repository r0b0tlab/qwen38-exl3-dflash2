# Qwen3.8-27B EXL3 + DFlash2 on the RTX 3090

The published EXL3 quants — [`r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw)
(target) and [`r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw)
(draft) — tuned for a 24 GB RTX 3090, running at the model's full advertised 262,144-token context
with block-diffusion speculative decoding through **native DFlash2 in ExLlamaV3**. Engine pin:
[`r0b0tlab/exllamav3`](https://github.com/r0b0tlab/exllamav3) branch `community` @ `355c6ee`
(upstream native DFlash2 CUDA, GDN rewind, dense M=32/64 TILEBLOCKS_M). Quants of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), licensed per
the parents.

This runtime does **not** overlay the old `dflash2-pathway` Python tree on the v1.5.0 wheel.

## Results

RTX 3090, GSM8K greedy, max 512 new tokens, ctx 8192. Current engine vs the previous
`dflash2-pathway` overlay on the v1.5.0 wheel:

| | autoregressive | MTP head | DFlash2 (native) | overlay (was) |
| --- | --- | --- | --- | --- |
| Acceptance length | 1.00 | 4.12 | **5.66** | 5.47 |
| Decode tok/s | 42.8 | 116.3 | **162.9** | 152.4 |

DFlash2 decode is **+6.9%** vs the overlay (162.9 vs 152.4 tok/s) at slightly higher
acceptance. At 150k-token depth (open-ended text): 25.3 tok/s, acceptance 1.82, prefill
594 tok/s, peak 23.13 GB. Full numbers: `notes/ACCEPTANCE.md`, `notes/RESULTS.md`.

## Resource requirements (RTX 3090, 24 GB)

**On disk:**

| artifact | size |
| --- | --- |
| target EXL3 4.00 bpw (6 bpw head, vision 6, MTP 4) | 15.4 GiB |
| draft DFlash2 EXL3 4.00 bpw (conv/selector kept fp16) | 1.2 GiB |
| BF16 sources (needed for conversion only) | 48.4 GiB + 3.4 GiB |

**VRAM at 262,144-token context with DFlash2 active, one sequence:**

| state | usage |
| --- | --- |
| loaded and idle (torch) | 21.7 GB |
| resident serving, mean over the Q200v2 run (nvidia-smi, 2 s cadence) | 20.7 GiB |
| peak during long-context requests (nvidia-smi) | 22.7 GiB |
| peak, native 150k prefill + decode (torch) | 23.13 GB |

Budget math behind those numbers:

- weights as loaded: ~15.5 GiB target + ~1.2 GiB draft
- KV cache (16 full-attention layers; `16 × 4 kv heads × 256 dim × 2 (K+V) × bytes` per
  token) at 262,144 tokens: fp16 ≈ 16.0 GiB, 8-bit ≈ 8.0, 6-bit ≈ 6.0, 4-bit ≈ 4.0,
  3-bit ≈ 3.0 GiB — cq3 is the validated setting at this context
- linear-attention (GDN) recurrent states: ≈ 1.22 GiB **per sequence slot** with
  speculative decoding (48 layers × 8 fp32 history rows); ≈ 0.15 GiB/slot without a
  draft attached
- prefill staging (`EXL3_QC_STAGING=1`) plus CUDA context/allocator: ≈ 2.5 GiB

Concurrency: each extra sequence costs its own ~1.2 GiB of verify history; four
concurrent sequences measured at 19.5 GiB with short contexts (1024 tokens/slot), and at
the full 262k context the card fits a single sequence.

## Layout

- `container/` — 3090 runtime image (CUDA 13, native community engine, no overlay).
- `scripts/` — VRAM budget gate + acceptance/long-context/sampled benchmarks.
- `notes/` — design doc, benchmark logs.
- **Weights** (HuggingFace): [r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw](https://huggingface.co/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw)
  and [r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw)
  — what the container pulls on first run.
- `patches/` — historical `dflash2-pathway` overlay; superseded, do not apply.

## Reproduce

```bash
# Native engine (community fork of ExLlamaV3 v1.5.0+)
git clone -b community https://github.com/r0b0tlab/exllamav3 exllamav3
# pin: 355c6ee
```

## Quickstart (host)

```bash
# 1. Environment (see notes/ENV.md): Python 3.13 venv, torch 2.10.0+cu130.
# 2. Convert the target (one time, ~hours on the 3090):
cd exllamav3 && python convert.py \
  -i ../models/qwen38-27b-hf -o ../models/qwen38-27b-exl3 \
  -w ../work/target -b 4.00 -vb 6 -mb 4 -hb 6
# 3. Convert the DFlash2 draft (minutes):
python convert.py -i ../models/dflash2-hf -o ../models/dflash2-exl3 \
  -w ../work/draft -b 4.00
# 4. Chat with speculative decoding at full context:
python examples/chat.py -m ../models/qwen38-27b-exl3 -mode chatml \
  -dm ../models/dflash2-exl3 -cs 262144 -cq 3
```

Published drafts store selector codebooks under `*.weight`; community `355c6ee` accepts
that key as well as the bare native name.

## Quickstart (container)

Click-run — pulls the prebuilt image and downloads the EXL3 models into a named volume on
first start (no local files needed; ~17 GB on the first run):

```bash
docker run --gpus all -v qwen38-models:/models \
  ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0-native
```

With local models, or to build the image yourself:

```bash
# requires the engine clone in exllamav3/ and the 3090 extension in container/
docker build -t qwen38-exl3-dflash2:1.5.0-native -f container/Dockerfile .

docker run -it --rm --gpus all -v "$PWD/models:/models:ro" qwen38-exl3-dflash2:1.5.0-native

docker run --rm --gpus all -v "$PWD/models:/models:ro" \
  -e PROMPT="Explain photosynthesis in one sentence." -e EXTRA_ARGS="-basic -tps" \
  qwen38-exl3-dflash2:1.5.0-native
```

## What native DFlash2 adds (vs the old overlay)

- CUDA grouped-dynamic-conv, top-k, and selector-walk kernels (`exllamav3_ext`).
- GDN rewind on rejection (upstream `7b1ba0b`).
- Dense GEMM shapes 5/6: M=32/64 tiles share one decoded B fragment (verify/prefill
  m>=32). Decode at the DFlash2 window (m=8) is unchanged by this path; GSM8K still
  gained from the CUDA drafter + rewind.
- 3090 defaults: `EXL3_HGEMM_F16ACC=auto` (fp16-accumulator MMA), int8 GEMV cap K=5.

## Validation gates

| Gate | Where | Result |
| --- | --- | --- |
| Native DFlash2 load + generate | host smoke | PASS — 6 kernel shapes, CUDA selector |
| Acceptance >= 4.0 and > MTP + 0.3 | `scripts/acceptance_check.py` n=40 | PASS — 5.657 vs 4.120, 162.9 tok/s |
| 262k-context load + 150k prefill + decode | `scripts/long_context_check.py` | PASS — 594 tok/s prefill, 25.3 tok/s decode, 23.13 GB |
| T=1.0 sampled distribution sanity | `scripts/sampled_sanity_check.py` | PASS — notes/LOSSLESS.md |

Q200v2 and NIAH were not re-run: weights are unchanged from the overlay campaign.

## Licenses

- Engine: ExLlamaV3 MIT. DFlash2 math from `z-lab/dflash` (MIT). This repository's original
  code (scripts, container, docs): MIT, see `LICENSE`.
- Weights: Qwen3.8-27B under the Qwen license; DFlash2 draft — see its HF repo (Apache-2.0 per the
  llama.cpp port).
