## Why

ExLlamaV3 has a DFlash (v1) draft pathway. DFlash 2 (Inco AI / z-lab, Aug 2026) adds two checkpoint-carried modules to the drafter — a grouped dynamic depthwise convolution around each attention and MLP sublayer, and a candidate path selector — and lifts mean acceptance length over both v1 DFlash and a model's native MTP head. On a quantized Qwen3.8-27B (EXL3 4.00 bpw) at 262k context, on a single 24 GB RTX 3090, we measure 5.474 mean accepted tokens (DFlash2) vs 4.101 (MTP) vs 1.000 (autoregressive), with decode at 152.4 tok/s vs 113.9 (MTP) vs 41.8 (AR).

The pathway is gated entirely by the checkpoint: a `DFlash2DraftModel` config gets the new modules, a `DFlashDraftModel` config resolves exactly as before. No new CLI flags — `-dm <dir>` auto-detects the arch from `config.json`.

## Commits

1. **`arch: Add the DFlash2 draft-model pathway`.** `DFlash2Config`/`DFlash2Model` (arch string `DFlash2DraftModel`), `DFlash2Block` wrapping each attention/MLP sublayer with the grouped dynamic causal convolution; registration in `architectures.py`; caps gain `dflash2_draft: True` while v1 caps are inherited unchanged.
2. **`modules: DFlash2 grouped dynamic conv and candidate selector (reference parity)`.** `GroupedDynamicCausalConv` (base kernel + per-position delta, grouped over channels) and `CandidateSelector` (per-position top-k from the target head's logits, adjacent-pair walk `edge(p→c) = <A[p] ⊙ project(h), B[c]> + unary[c]`; greedy argmax walk or inverse-CDF walk returning `q` over the candidates). Pure-math ports of the MIT `z-lab/dflash` reference (`dflash/model.py`), parity-tested on fixed tensors.
3. **`generator: DFlash2 candidate-selector propose and q-aware rejection verify`.** `iterate_draftmodel_dflash_gen` runs `propose()` for dflash2 drafters (selector walk uses the batch's consensus temperature when every active job uses the same plain temperature/top-k/top-p sampler, else the greedy walk); the verify path adds a q-aware rejection sampler for those samplers (accept `d` with probability `p(d)/q(d)`, else emit one sample from the normalized `(p−q)+` residual). Token-match acceptance is unchanged for greedy and for samplers we cannot reproduce (penalties, min-p, …). Also: attaching a draft to a target with recurrent layers now asserts the cache reserves `max_history >= draft tokens`, because the failure mode otherwise is an opaque shape error in the first verify pass.
4. **`tests: DFlash2 config, module, load and verify suites`.** Config fields + registration; conv/selector parity vs the z-lab reference; real-checkpoint load + propose shapes/behavior; sampler introspection, p/q math and rejection-sampler behavior (including the reference's impossible-event accept and p=0 reject).

## Accuracy

GSM8K test split, greedy, ≤512 new tokens, target Qwen3.8-27B EXL3 4.00 bpw, one RTX 3090:

| drafter | mean acceptance | tok/s |
| --- | --- | --- |
| DFlash2 (EXL3 4 bpw draft) | **5.474** | 152.4 |
| DFlash2 (BF16 draft) | 5.463 | 139.9 |
| MTP head | 4.101 | 113.9 |
| autoregressive | 1.000 | 41.8 |

Draft quantization is acceptance-neutral (5.474 vs 5.463). Multi-needle NIAH passes at 262,080 tokens (needles at 33/66% and 33/66/90%, answer = last). Full harnesses, raw artifacts and the served model: https://github.com/r0b0tlab/qwen38-exl3-dflash2

## Speed

Single 3090, 8k context, DFlash2 active: 152.4 tok/s decode (3.6× AR, 1.34× MTP). Concurrency ladder on the same draft path: 109.0 → 134.6 → 152.1 tok/s aggregate at 1/2/4 concurrent sequences (each sequence slot costs ~1.2 GB of fp32 GDN verify history across 48 linear-attention layers, which caps concurrency on a 24 GB card).

## Testing

- `pytest tests/ -k dflash2` — config/module/verify tests run anywhere (the reference-parity and verify tests skip cleanly when the `dflash` reference package is absent); the load suite needs the draft checkpoint via `EXL3_TEST_DFLASH2=<dir>` and a CUDA device, and skips cleanly otherwise (the existing `EXL3_TEST_MODEL` convention).
- End-to-end on any CUDA box: `convert.py -b 4.00 -i incoai/Qwen3.8-27B-DFlash2 -o <draft-exl3> --work-dir <work>` then `python examples/chat.py -m <target-exl3> -mode chatml -dm <draft-exl3> -cs 262144 -cq 3`.

## Notes for review

- Conv `base_kernel` and both selector codebooks stay raw fp16 tensors (uncalibrated); `kernel_projection`/`hidden_projection` are ordinary Linears and quantize with everything else. The dynamic-convolution projection runs in fp32 and its output is cast back to the layer dtype (EXL3 GEMMs require fp16 inputs).
- The q-aware path computes `p` from the same quantized target lm_head that produced the draft logits at propose time, so rejection sampling is lossless with respect to the deployed (quantized) model. SGLang's DFlash2 path rejects quantized target heads outright; we deliberately don't, because the head is shared between p and q. The sampled verify falls back to token-match acceptance for any sampler we cannot reproduce exactly, so behavior never regresses below v1.
- The selector sits in the model's module list so the loader reaches its weights, and is a documented no-op in the forward walk; selection runs via `select()` from `propose()`. An alternative would be a `fwd_end_idx`-style walk boundary as in `deepseek_v4_mtp`.
- Relates to #261 (multiple draft models): both touch `generator.py`, but orthogonally — this PR only adds a draft-type branch and the verify path.

## Related

SGLang PR 35371, vLLM PR 52816, llama.cpp PR 27342, and the MIT `z-lab/dflash` reference used for all parity tests.
