---
license: apache-2.0
base_model: incoai/Qwen3.8-27B-DFlash2
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- quantization
- 4-bit
- speculative-decoding
- dflash2
---

# Qwen3.8-27B-DFlash2 — EXL3 4.00 bpw (draft)

EXL3 (ExLlamaV3) quantization of the [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
block-diffusion draft for use with [r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw](https://huggingface.co/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw).

Backbone Linears quantized at 4.00 bpw; the dynamic-convolution `base_kernel` tensors and the
candidate-selector codebooks stay fp16 (uncalibrated). Quantization is acceptance-neutral on
this stack (overlay measurement: mean acceptance 5.474 EXL3 vs 5.463 BF16 draft). Native
engine GSM8K: **5.657** AL / **162.9 tok/s**.

Requires [`r0b0tlab/exllamav3`](https://github.com/r0b0tlab/exllamav3) branch `community`
@ `355c6ee` — native `DFlash2DraftModel`, selected automatically via `-dm <dir>`. Do not
use the old `dflash2-pathway` overlay.

## Resource requirements

- Weights on disk: **1.2 GiB** (this repo). Target repo: 15.4 GiB.
- This draft adds ~1.22 GiB of fp32 GDN verify states (per sequence slot) to the target's
  footprint; its own KV is windowed (5 sliding-window layers, tens of MB). Target at full
  262,144-token context with cq3: ~21.7 GB loaded, peak ~22.7 GiB.

## Use

```bash
python examples/chat.py -m <target-dir> -mode chatml \
  -dm <this-dir> -cs 262144 -cq 3
```

Container (click-run, auto-downloads both repos on first run):

```bash
docker run --gpus all -v qwen38-models:/models \
  ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0-native
```

Metrics and harnesses: https://github.com/r0b0tlab/qwen38-exl3-dflash2
