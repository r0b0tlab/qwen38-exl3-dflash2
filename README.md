# Qwen3.8-27B EXL3 + DFlash2 on the RTX 3090

Quantized [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (EXL3, 4.00 bpw) tuned for a
24 GB RTX 3090, accelerated with [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
block-diffusion speculative decoding through a **DFlash2 pathway added to ExLlamaV3** (this repo's
`exllamav3/` fork), at the model's full advertised 262,144-token context.

## Results

| | autoregressive | MTP head | DFlash2 (this work) |
| --- | --- | --- | --- |
| Acceptance length (GSM8K, greedy, 512 tok) | 1.00 | 4.10 | **5.47** |
| Decode tok/s (RTX 3090, ctx 8192) | 41.8 | 113.9 | **152.4** |

At 150k-token depth (open-ended text): 24.5 tok/s, acceptance 1.79. Full numbers,
methodology and raw JSON: `notes/ACCEPTANCE.md`, `notes/RESULTS.md`.

## Layout

- `exllamav3/` — engine fork (branch `dflash2-pathway`, based on v1.5.0): DFlash2 architecture,
  candidate-selector verify, tests.
- `container/` — optimized runtime image (multi-stage, no toolchain in the final image).
- `scripts/` — VRAM budget gate + acceptance/long-context/sampled benchmarks.
- `notes/` — design doc, spike findings, benchmark logs.
- `patches/` — the engine delta as format-patch files; the same commits are on the fork branch
  `dflash2-pathway` at https://github.com/r0b0tlab/exllamav3.

## Reproduce

```bash
# Engine fork (needed as exllamav3/ for both quickstarts below)
git clone -b dflash2-pathway https://github.com/r0b0tlab/exllamav3 exllamav3
```

## Quickstart (host)

```bash
# 1. Environment (see notes/ENV.md): Python 3.13 venv or system 3.14 with the source build.
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

## Quickstart (container)

Click-run — pulls the prebuilt image and downloads the EXL3 models into a named volume on
first start (no local files needed; ~17 GB on the first run):

```bash
docker run --gpus all -v qwen38-models:/models \
  ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0
```

With local models, or to build the image yourself:

```bash
# requires the engine fork cloned into exllamav3/ first (see Reproduce)
docker build -t qwen38-exl3-dflash2:1.5.0 -f container/Dockerfile .

# Interactive chat at full context (models mounted read-only):
docker run -it --rm --gpus all -v "$PWD/models:/models:ro" qwen38-exl3-dflash2:1.5.0

# One-shot / headless (loads 262k ctx, answers, exits):
docker run --rm --gpus all -v "$PWD/models:/models:ro" \
  -e PROMPT="Explain photosynthesis in one sentence." -e EXTRA_ARGS="-basic -tps" \
  qwen38-exl3-dflash2:1.5.0
```

## What the DFlash2 pathway adds to ExLlamaV3

- `exllamav3/architecture/dflash2.py` — `DFlash2Config`/`DFlash2Model` (arch `DFlash2DraftModel`),
  `DFlash2Block` wrapping each attention/MLP sublayer with a grouped dynamic causal convolution.
- `exllamav3/modules/arch_specific/dflash2.py` — `GroupedDynamicCausalConv` + `CandidateSelector`
  (pure-math ports of the MIT-licensed `z-lab/dflash` reference; parity-tested).
- `exllamav3/generator/generator.py` — selector-driven propose and a q-aware rejection-sampling
  verify (lossless w.r.t. the quantized target for plain temperature/top-k/top-p samplers).

## Validation gates

| Gate | Where | Result |
| --- | --- | --- |
| Conv/selector parity vs reference | `tests/ntest_dflash2_modules.py` | 6/6 |
| Sampling-dist parity vs reference | `tests/ntest_dflash2_verify.py` | 17/17 |
| Acceptance >= 4.0 and > MTP + 0.3 | `scripts/acceptance_check.py` | PASS — 5.474 vs 4.101 |
| Draft quantization neutral (EXL3 vs BF16) | `scripts/acceptance_check.py` | PASS — 5.474 vs 5.463 |
| 262k-context load + 150k prefill + decode | `scripts/long_context_check.py` | PASS — 23.2 GB peak |
| T=1.0 sampled distribution sanity | `scripts/sampled_sanity_check.py` | PASS — notes/LOSSLESS.md |

## Licenses

- Engine: ExLlamaV3 MIT. DFlash2 math ported from `z-lab/dflash` (MIT). This repository's original
  code (scripts, container, docs): MIT, see `LICENSE`.
- Weights: Qwen3.8-27B under the Qwen license; DFlash2 draft — see its HF repo (Apache-2.0 per the
  llama.cpp port).