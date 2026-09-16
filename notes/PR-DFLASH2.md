# Upstream PR draft: DFlash2 draft-model support (final body)

> Target: `turboderp-org/exllamav3:master` from `r0b0tlab/exllamav3:dflash2-pathway`.
> **Submitted: [PR #379](https://github.com/turboderp-org/exllamav3/pull/379) (2026-09-16).**
> Structure follows the repo's de facto PR style (see merged PR #349): Why / Commits /
> Accuracy / Speed / Testing / Notes for review. Plan + prep steps: `notes/PR-DFLASH2-PLAN.md`.

## Why

ExLlamaV3 has a DFlash (v1) draft pathway. DFlash 2 (Inco AI / z-lab, Aug 2026) adds two
checkpoint-carried modules to the drafter — a grouped dynamic depthwise convolution around
each attention and MLP sublayer, and a candidate path selector — and lifts mean acceptance
length over both v1 DFlash and the model's native MTP head. On a quantized Qwen3.8-27B
(EXL3 4.00 bpw) at 262k context on a single 24 GB RTX 3090 we measure 5.474 mean accepted
tokens (DFlash2) vs 4.101 (MTP) vs 1.000 (autoregressive), with decode at 152.4 tok/s vs
113.9 (MTP) vs 41.8 (AR). The pathway is gated entirely by the checkpoint: a
`DFlash2DraftModel` config gets the new modules, a `DFlashDraftModel` config resolves
exactly as before.

## Modifications

```
exllamav3/architecture/dflash2.py           DFlash2Config, DFlash2Model, DFlash2Block
exllamav3/architecture/architectures.py     register DFlash2DraftModel
exllamav3/modules/arch_specific/dflash2.py  GroupedDynamicCausalConv (base kernel +
                                            per-position delta, grouped over channels)
                                            CandidateSelector (top-k per block position from
                                            the target head's logits; adjacent-pair walk
                                            edge(p->c) = <A[p]*project(h), B[c]> + unary[c];
                                            greedy argmax or inverse-CDF walk returning q)
exllamav3/generator/generator.py            propose() wiring for dflash2 drafters; q-aware
                                            rejection-sampling verify for plain
                                            temperature/top-k/top-p samplers (accept d with
                                            p(d)/q(d), else emit one sample from (p-q)+);
                                            token-match acceptance unchanged everywhere else
tests/test_dflash2_config_.py               config + arch registration
tests/test_dflash2_modules_.py              conv/selector parity vs the z-lab reference (MIT)
tests/test_dflash2_load_.py                 real-checkpoint load + propose parity
tests/test_dflash2_verify_.py               sampler introspection + p/q math + rejection
                                            sampler behavior
```

Math is ported from the reference implementation (`z-lab/dflash`, `dflash/model.py`, MIT)
and parity-tested on fixed tensors.

## Accuracy

GSM8K test split, greedy, <=512 new tokens, target Qwen3.8-27B EXL3 4.00 bpw on one RTX 3090:

| drafter | mean acceptance | tok/s |
| --- | --- | --- |
| DFlash2 (EXL3 4 bpw draft) | **5.474** | 152.4 |
| DFlash2 (BF16 draft) | 5.463 | 139.9 |
| MTP head | 4.101 | 113.9 |
| autoregressive | 1.000 | 41.8 |

Draft quantization is acceptance-neutral (5.474 vs 5.463). Multi-needle NIAH passes at
262,080 tokens (needles at 33/66 % and 33/66/90 %, answer = last). Full metrics, harnesses
and raw artifacts: https://github.com/r0b0tlab/qwen38-exl3-dflash2.

## Speed

Single 3090, 8k context, DFlash2 active: 152.4 tok/s decode (3.6x AR, 1.34x MTP).
Concurrency ladder on the same draft path: 109.0 -> 134.6 -> 152.1 tok/s aggregate at
1/2/4 concurrent sequences. (Each extra sequence slot costs ~1.2 GB of fp32 GDN verify
history across 48 linear-attention layers, which caps concurrency on a 24 GB card.)

## Testing

- `pytest tests/ -k dflash2` — config/module/verify tests run anywhere; the load test needs
  the draft checkpoint (`EXL3_TEST_DFLASH2` env var) and a GPU and skips cleanly otherwise,
  following the existing `EXL3_TEST_MODEL` convention.
- End-to-end: `convert.py -b 4.00 -i incoai/Qwen3.8-27B-DFlash2 -o <dir>` then
  `python examples/chat.py -m <target-exl3> -mode chatml -dm <draft-dir> -cs 262144 -cq 3`.

## Notes for review

- Conv `base_kernel` and both selector codebooks stay raw fp16 tensors (uncalibrated);
  `kernel_projection`/`hidden_projection` are ordinary Linears and quantize with everything
  else. The dynamic-convolution projection runs in fp32 and its output is cast back to the
  layer dtype (EXL3 GEMMs require fp16 inputs).
- The sampled verify is exact only for plain temperature/top-k/top-p samplers; anything else
  (penalties, min-p, adaptive) falls back to the existing token-match acceptance, so behavior
  never regresses below v1 for unsupported samplers.
- The q-aware path computes p from the same quantized target lm_head that produced the draft
  logits at propose time, so rejection sampling is lossless w.r.t. the deployed (quantized)
  model. SGLang's DFlash2 path rejects quantized target heads outright; we deliberately
  don't, because the head is shared between p and q.
- The selector sits in the model's module list (so the loader reaches its weights) and is a
  documented no-op in the forward walk; the selection runs via `select()` from `propose()`.
  An alternative would be a `fwd_end_idx`-style walk boundary as in `deepseek_v4_mtp`.
- No new CLI flags: `-dm <dir>` auto-detects `DFlash2DraftModel` from config.json.
- Attaching a draft now asserts the target cache reserves enough recurrent history
  (`max_history >= draft window`) — hand-rolled harnesses otherwise fail at the first
  verify pass with an opaque shape error.
- Open PR #261 ("multiple draft models") also touches `generator.py`; the changes are
  orthogonal (this PR only adds a draft-type branch + verify path), flagging for sequencing.
