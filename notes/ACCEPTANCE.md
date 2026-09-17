# Acceptance gate results

Benchmark: GSM8K test split, greedy (argmax), max_new_tokens 512, ctx 8192,
single RTX 3090. Target: `models/qwen38-27b-exl3` (EXL3 4.00 bpw, head 6 bpw, MTP + vision quantized).

## Native community engine (2026-09-17) — current

Engine: r0b0tlab/exllamav3 `community` @ `355c6ee`. No overlay.

| Drafter | Mean acceptance length | tok/s | n | hit cap |
| --- | --- | --- | --- | --- |
| DFlash2 EXL3 (4.00 bpw) | **5.657** | 162.9 | 40 | 5/40 |
| MTP head | 4.120 | 116.3 | 20 | 3/20 |
| Autoregressive baseline | 1.000 | 42.8 | 10 | 2/10 |

Gates:

| Gate | Result |
| --- | --- |
| DFlash2 mean acceptance >= 4.0 | PASS (5.657) |
| DFlash2 > MTP + 0.3 | PASS (5.657 vs 4.420) |
| Decode vs overlay | PASS — 162.9 vs 152.4 tok/s (+6.9%) |

Draft-quantization neutrality vs BF16 was measured on the overlay (5.474 vs 5.463) and
was not re-run: draft weights are unchanged.

Raw: `notes/acceptance-native-n40.json` (and mtp/ar siblings).

## Overlay baseline (2026-09-15)

| Drafter | Mean acceptance length | tok/s | n | hit cap |
| --- | --- | --- | --- | --- |
| DFlash2 EXL3 (4.00 bpw) | 5.474 | 152.4 | 40 | 6/40 |
| DFlash2 BF16 | 5.463 | 139.9 | 40 | 6/40 |
| MTP head | 4.101 | 113.9 | 40 | 6/40 |
| Autoregressive baseline | 1.000 | 41.8 | 10 | 10/10 |

Raw: `notes/acceptance-dflash2.json` and siblings.
