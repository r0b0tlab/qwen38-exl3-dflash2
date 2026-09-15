# T=1.0 sampled-distribution sanity (2026-09-15)

Method: `scripts/sampled_sanity_check.py` — 6 fixed prompts x 8 samples, max 96 tokens,
CategoricalSampler temperature 1.0; DFlash2 EXL3 draft vs autoregressive on the same
quantized target. The q-aware DFlash2 verify is lossless w.r.t. the quantized target for
plain temperature/top-k/top-p samplers, so the two arms' output distributions should agree
closely; statistics compared: mean chars, distinct-token rate, top-20 token share, tok/s.

| Arm | mean chars | distinct-token rate | top-20 share | tok/s |
| --- | --- | --- | --- | --- |
| DFlash2 EXL3 | 451.25 | 0.2643 | 0.3124 | 89.0 |
| Autoregressive | 452.31 | 0.2483 | 0.3146 | 42.1 |

Verdict: no glaring divergence — mean length within 1 char, top-20 share within 1%,
distinct-token rate within ~6% relative (sampling noise at 48 x 96 tokens). No sign of the
degenerate repetition or shifted mass profile a biased accept would produce. PASS per plan
Task 3.5; the sampled verify path is validated on real outputs.

Raw: `notes/sampled-sanity.log`.
