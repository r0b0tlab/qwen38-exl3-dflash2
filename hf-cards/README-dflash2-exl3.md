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
this stack (mean acceptance 5.474 EXL3 vs 5.463 BF16 draft).

Requires the ExLlamaV3 fork with the DFlash2 pathway (branch `dflash2-pathway`,
https://github.com/r0b0tlab/exllamav3) — arch `DFlash2DraftModel`, selected automatically
via `-dm <dir>`.

## Use

```bash
python examples/chat.py -m <target-dir> -mode chatml \
  -dm <this-dir> -cs 262144 -cq 3
```

Container (click-run, auto-downloads both repos on first run):

```bash
docker run --gpus all -v qwen38-models:/models \
  ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0
```

Metrics and harnesses: https://github.com/r0b0tlab/qwen38-exl3-dflash2
