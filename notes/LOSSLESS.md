# T=1.0 sampled-distribution sanity

Method: `scripts/sampled_sanity_check.py` — 6 fixed prompts x 8 samples, max 96 tokens,
CategoricalSampler temperature 1.0; DFlash2 EXL3 draft vs autoregressive on the same
quantized target.

Native DFlash2 (community @ 355c6ee) verifies with accept-while-match, not the old
overlay's q-aware rejection sampler. Upstream treats the selector as T-independent;
the two arms should still agree closely on length and token-mass stats.

## Native (2026-09-17)

| Arm | mean chars | distinct-token rate | top-20 share | tok/s |
| --- | --- | --- | --- | --- |
| DFlash2 EXL3 | 456.0 | 0.2741 | 0.3090 | 90.8 |
| Autoregressive | 448.4 | 0.2587 | 0.3221 | 42.3 |

Verdict: no glaring divergence — mean length within 8 chars, top-20 share within 2 pp,
distinct-token rate within ~6% relative. No degenerate repetition. PASS.
Raw: `notes/sampled-sanity-native.log`.

## Overlay baseline (2026-09-15)

| Arm | mean chars | distinct-token rate | top-20 share | tok/s |
| --- | --- | --- | --- | --- |
| DFlash2 EXL3 | 451.25 | 0.2643 | 0.3124 | 89.0 |
| Autoregressive | 452.31 | 0.2483 | 0.3146 | 42.1 |

Raw: `notes/sampled-sanity.log`.
