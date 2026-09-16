---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- quantization
- 4-bit
- speculative-decoding
- dflash2
- rtx3090
---

# Qwen3.8-27B — EXL3 4.00 bpw

EXL3 (ExLlamaV3) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
tuned for a 24 GB RTX 3090 at the model's full 262,144-token context.

Conversion: ExLlamaV3 v1.5.0 (fork branch `dflash2-pathway`), single pass with
`convert.py -b 4.00 -vb 6 -mb 4 -hb 6` — 4.00 bpw decoder, 6 bpw lm_head, vision tower
6 bpw, native MTP head 4 bpw.

Pair with the DFlash2 draft: [r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw).

## Resource requirements (RTX 3090, 24 GB)

- Weights on disk: **15.4 GiB** (this repo); pair with the draft repo (1.2 GiB).
- VRAM at the full 262,144-token context with the DFlash2 draft, one sequence: ~21.7 GB
  loaded, 20.7 GiB mean while serving, up to 22.7 GiB during long-context requests.
- KV cache (16 full-attention layers) at 262,144 tokens: fp16 ~16.0 GiB, 8-bit ~8.0,
  6-bit ~6.0, 4-bit ~4.0, 3-bit ~3.0 GiB (cq3 is the validated setting).
- Speculative decoding reserves ~1.22 GiB of fp32 GDN verify history per sequence slot
  (48 layers x 8 history rows); without a draft that drops to ~0.15 GiB/slot.

## Use

- **ExLlamaV3** (fork clone, branch `dflash2-pathway`):

  ```bash
  python examples/chat.py -m <this-dir> -mode chatml \
    -dm <draft-dir> -cs 262144 -cq 3
  ```

- **Container (click-run)** — auto-downloads this repo on first run:

  ```bash
  docker run --gpus all -v qwen38-models:/models \
    ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0
  ```

## Validation (single RTX 3090, 24 GB)

- Acceptance length, GSM8K greedy: **5.474** (DFlash2) vs 4.101 (MTP head) vs 1.000
  (autoregressive); 152.4 tok/s decode at 8k context.
- 262k-context load (cq3 cache, 23.2 GB peak); 150k-token prefill at 599 tok/s.
- Multi-needle NIAH at 262,080 tokens: PASS (needles at 33/66%) and PASS (33/66/90%).
- Q200v2 text-180: gsm8k 98.75 %, humaneval 100 %, ifeval 84.62 % (hard_reasoning manual).

Full metrics, harnesses and the served model: https://github.com/r0b0tlab/qwen38-exl3-dflash2
