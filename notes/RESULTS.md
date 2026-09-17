# Results log — Qwen3.8-27B EXL3 + native DFlash2 on RTX 3090

## Current engine (2026-09-17)

- Target `models/qwen38-27b-exl3` — EXL3 4.00 bpw, head 6 bpw, vision 6 bpw, MTP 4 bpw
  (weights unchanged from the 2026-09-15 conversion).
- Draft `models/dflash2-exl3` — EXL3 4.00 bpw (1.2 GB). Selector codebooks stored as
  `*.weight`; community `355c6ee` accepts that key.
- Engine: r0b0tlab/exllamav3 `community` @ `355c6ee` (v1.5.0 + upstream native DFlash2
  CUDA, GDN rewind, TILEBLOCKS_M shapes 5/6, codebook alias). **No dflash2-pathway overlay.**
- Container image `qwen38-exl3-dflash2:1.5.0-native` (CUDA 13 runtime, torch 2.10.0+cu130,
  sm_86 extension).

### Acceptance (GSM8K greedy, max 512, ctx 8192)

| Drafter | n | Mean AL | tok/s | hit cap |
| --- | --- | --- | --- | --- |
| DFlash2 EXL3 | 40 | **5.657** | **162.9** | 5/40 |
| MTP head | 20 | 4.120 | 116.3 | 3/20 |
| Autoregressive | 10 | 1.000 | 42.8 | 2/10 |

Vs overlay (2026-09-15, same weights, `dflash2-pathway` on the v1.5.0 wheel):
DFlash2 5.474 / 152.4 tok/s → **+6.9% decode**. MTP 4.101 / 113.9. AR 41.8.

Raw: `notes/acceptance-native-n40.json`, `notes/acceptance-native-mtp-n20.json`,
`notes/acceptance-native-ar-n10.json`.

### 262k context

- Load at 262,144 ctx with `-cq 3`: no OOM, peak 23.13 GB.
- Prefill 150,000 tokens: 252.4 s = **594 tok/s** (overlay 599; reconstruct path, flat).
- Decode 200 tokens at 150k depth: **25.3 tok/s**, acceptance 1.82 (overlay 24.5 / 1.79).
- Raw: `notes/long-context-native.log`.

### T=1.0 sampled sanity

DFlash2 vs autoregressive, 48 samples each: mean length 456 vs 448 chars, top-20 share
0.309 vs 0.322, distinct-token rate 0.274 vs 0.259. No distributional divergence.
Native verify is accept-while-match (upstream), not the overlay's q-aware RS.
Detail: `notes/LOSSLESS.md`.

### Prefill (short, TILEBLOCKS_M / reconstruct)

- 2048 tokens: 960 tok/s
- 8192 tokens: 700 tok/s
- 150k tokens: 594 tok/s

### Container

Host has no nvidia-container-toolkit, so `--gpus all` cannot attach the 3090.
Image is CPU-smoked (`import torch, exllamav3` + kernel shape count). GPU path is
the host venv numbers above.

## Overlay campaign (2026-09-15), superseded

Engine fork branch `dflash2-pathway` tip `5cdf6bc` (9 commits over v1.5.0), container
`qwen38-exl3-dflash2:1.5.0` (v1.5.0 wheel + Python overlay). Kept as the baseline the
native pin was measured against. See git history and `notes/acceptance-dflash2.json`.
