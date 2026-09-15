# Upstream PR draft: DFlash2 draft-model support

> Target repo: `turboderp-org/exllamav3`, branch `dflash2-pathway` vs `master` (v1.5.0 base).
> Modeled on the structure of SGLang PR #35371 / vLLM PR #52816.

## Motivation

ExLlamaV3 has a DFlash (v1) draft pathway. DFlash 2 (Inco AI / z-lab, Aug 2026) adds two
checkpoint-carried modules to the drafter — a grouped dynamic depthwise convolution around each
sublayer and a candidate path selector — and lifts mean acceptance length by ~0.5-1.0 tokens over
both v1 DFlash and models' native MTP heads (e.g. Qwen3.8-27B: 4.80 vs 4.28 MTP, published
number, seven draft tokens per step). This PR adds DFlash2 support on top of the existing v1
pathway, gated entirely by the checkpoint: a `DFlash2DraftModel` config gets the new modules; a
`DFlashDraftModel` config resolves exactly as before.

## Modifications

```
exllamav3/architecture/dflash2.py        DFlash2Config, DFlash2Model, DFlash2Block
exllamav3/modules/arch_specific/dflash2.py
    GroupedDynamicCausalConv             dynamic conv (base kernel + per-position delta,
                                         grouped over channels), wrapping attention and MLP
    CandidateSelector                    top-k candidates per block position from the target
                                         head's logits; adjacent-pair walk
                                         edge(p->c) = <A[p] * project(h), B[c]> + unary[c];
                                         greedy argmax walk or inverse-CDF walk returning q
                                         over the candidates
exllamav3/generator/generator.py
    iterate_draftmodel_dflash_gen        runs propose() for dflash2 drafters; the selector walk
                                         uses the batch's consensus temperature when every active
                                         job uses the same plain temperature/top-k/top-p sampler
    iterate_gen / _dflash2_accept        q-aware rejection-sampling verify: accept d with
                                         prob p(d)/q(d), else emit one sample from (p-q)+;
                                         token-match acceptance unchanged for greedy and for
                                         samplers we cannot reproduce (penalties, min-p, ...)
tests/ntest_dflash2_*.py                 config, module parity vs the z-lab reference
                                         (MIT), verify-path unit tests
```

Math is ported from the reference implementation (`z-lab/dflash`, `dflash/model.py`, MIT) and
parity-tested on fixed tensors (greedy walk, sampled q, one-hot behavior, conv prepare/finish).

### Notes for review

- Conv `base_kernel` and both selector codebooks stay raw fp16 tensors (uncalibrated);
  `kernel_projection`/`hidden_projection` are ordinary Linears and quantize with everything else.
- The sampled verify is exact only for plain temperature/top-k/top-p samplers; anything else
  (penalties, min-p, adaptive-p) falls back to the existing token-match acceptance, so behavior
  never regresses below v1 for unsupported samplers.
- The q-aware path computes p from the same quantized target lm_head that produced the draft
  logits at propose time, so rejection sampling is lossless w.r.t. the deployed (quantized)
  model. SGLang's DFlash2 path rejects quantized target heads outright; we deliberately don't,
  because the head is shared between p and q.
- No new CLI flags: `-dm <dir>` auto-detects `DFlash2DraftModel` from config.json.

## Accuracy tests

```
pytest tests/ntest_dflash2_config.py     # 2 passed
pytest tests/ntest_dflash2_modules.py    # 6 passed (reference parity)
pytest tests/ntest_dflash2_verify.py     # 17 passed (sampler introspection, p/q math,
                                           # rejection-sampler behavior incl. reference's
                                           # impossible-event accept and p=0 reject)
```

End-to-end acceptance (GSM8K, greedy, Qwen3.8-27B EXL3 4.00 bpw target, EXL3 4 bpw draft):
see the companion results repo (linked in PR body once posted).

## Speed

Selector + conv cost per draft-verify cycle is dominated by the vocabulary top-k and measures
~0.2 ms at these shapes on Ada/Ampere (vLLM's published component timing for the same arch); the
generator walk itself is a 7-step scalar loop per round.
