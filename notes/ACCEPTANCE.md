# Acceptance gate results (2026-09-15)

Benchmark: GSM8K test split, greedy (argmax), max_new_tokens 512, ctx 8192,
single RTX 3090. Target: `models/qwen38-27b-exl3` (EXL3 4.00 bpw, head 6 bpw, MTP + vision quantized).

| Drafter | Mean acceptance length | tok/s | n | hit cap |
| --- | --- | --- | --- | --- |
| DFlash2 EXL3 (4.00 bpw) | **5.474** | 152.4 | 40 | 6/40 |
| DFlash2 BF16 (models/dflash2-hf) | 5.463 | 139.9 | 40 | 6/40 |
| MTP head (-mb 4) | 4.101 | 113.9 | 40 | 6/40 |
| Autoregressive baseline | 1.000 | 41.8 | 10 | 10/10 |

Gates (plan Task 2.6/3.3):

| Gate | Result |
| --- | --- |
| DFlash2 mean acceptance >= 4.0 | PASS (5.474) |
| DFlash2 > MTP + 0.3 | PASS (5.474 vs 4.401) |
| EXL3 draft >= BF16 draft - 0.15 (quantization neutrality) | PASS (5.474 vs 5.463) |

Context: the DFlash2 model card quotes 4.80 (H200/BF16) and llama.cpp Q4_K_M 5.03;
5.474 on a 3090 with an EXL3-quantized target is consistent with deeper-context-free
GSM8K shards and the acceptance length growing with the model's thinking-style output.

Raw data: `notes/acceptance-*.json` (summary + per-request), logs `notes/acceptance-*.log`.
Per-request AL range (DFlash2 EXL3): min 3.37 / median 5.61 / max 6.76.

Harness notes (found + fixed during this gate; engine commits on `dflash2-pathway`):
- `b8e1a00` conv output dtype follows input (fp16 EXL3 GEMMs rejected the fp32 conv result)
- `aa6adf8` CandidateSelector is a no-op in the module walk
- `4960047` full draft window kept in verify (no double anchor-crop; candidates/q alignment)
  + selector anchor indices follow the codebook device
- `scripts/*.py` build the target cache with `max_history = draft size` (mirrors
  `model_init.init()`; required for speculative decode on GDN layers)
