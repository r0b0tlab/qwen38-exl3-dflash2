# DFlash2 pathway design (spike output, 2026-09-15)

## v1 pathway facts (from repo source @ v1.5.0)

- `exllamav3/architecture/dflash.py`: `DFlashConfig` (arch_string `DFlashDraftModel`), `DFlashModel`.
  - `tap_shift` default 1; per-checkpoint override via `dflash_config->tap_shift` (empirical: wrong shift → ~0.3 accepted/round).
  - caps: `uncalibrated_quantize`, `supports_tp: False`, `attach_target`, `dflash_draft`, `default_draft_size = block_size - 1`, `autosplit_load_fwd: False`.
  - `draft_verifier_params = {"export_state_layers": set(target_layer_ids)}` — merged into target prefill/verify params; model forward returns `params["export_states"]` list (hidden AFTER layer j, per tap_shift convention).
  - `update_kv_from_target(target_hidden, cache, params, lengths)`: concats taps → fc proj → hidden_norm → projects k/v per draft layer via each layer's own k_proj/v_proj + rope → writes rows into paged draft cache at `cache_seqlens` (draft KV layout mirrors target layout; SWA window handled by cache layer config).
  - `sample_from_state(state, params)`: runs the ATTACHED target's lm_head over draft state; argmax, or (export_draft_conf) argmax + argmax-logit conf.
  - `prepare_inputs` sets `params["causal"] = False` (bidirectional block attention; SWA layers express causality via window).
- `exllamav3/modules/arch_specific/dflash.py`: `DFlashInputLayer` — appends `block_size-1` mask tokens, looks up embeddings via attached target (`modules[0].forward`), qmap `target_hidden`.
- Generator (`exllamav3/generator/generator.py`):
  - `__init__`: `self.dflash_draft` from caps; `attach_to(model)` at line ~248.
  - Prefill: `job.py:1401-1409` — target prefill exports states; `update_kv_from_target` fills draft cache over prompt.
  - Decode: `iterate_draftmodel_dflash_gen` (line 790): ONE draft forward at fixed block_size → `sample_from_state` → crop first token → optional confidence truncation → returns `draft_ids_pinned[:, :window]`.
  - Verify: `iterate_gen` (line 929) — target forward over [last + draft window] (params get `draft_verifier_params` → exports states), then per-job serial loop (line 1142+): `receive_logits` per position → accept iff `draft_tokens[j,i] == sampled_token` else `reject_remainder` (rewinds recurrent state + pages). Draft stats + calibrator update. Post-loop `update_kv_from_target` for accepted tokens (line 1274).
  - **T>0 today: token-match acceptance only** (no rejection sampler, no q). DFlash2 adds q-aware rejection here.

## DFlash2 delta (reference: z-lab/dflash dflash/model.py — extracted verbatim to REFERENCE notes)

1. `GroupedDynamicCausalConv(hidden_size, kernel_size, group_size)`:
   - `base_kernel` [2, K, hidden]; `kernel_projection` Linear(hidden → 2*K*groups, bias=False).
   - `prepare(x)`: dyn = proj(x).view(B,L,2,K,groups); returns `conv(x, dyn[...,0,:,:], base[0]), dyn[...,1,:,:]`.
   - `finish(x, dyn1)`: `conv(x, dyn1, base[1])`.
   - `conv`: blocks = x.view(B,L,groups,gs); per offset t: values = blocks shifted right t (zero pad at block start → taps zero across block boundary); out += base[t].view(1,1,groups,gs)*values; out = addcmul(out, dyn[:,:,t], values).
   - Wraps attention sublayer (prepare before self_attn input-layernorm output; finish on attn output pre-residual) and MLP sublayer likewise (see `Qwen3DFlashDecoderLayer.forward`).
   - Per-block: operates within the current block tensor only (decode: the [1, block_size] draft pass).
2. `CandidateSelector`: `predecessor_codebook`/`successor_codebook` Embedding(vocab, rank=256), `hidden_projection` Linear(hidden, rank, no bias); `select(hidden, logits, anchor_ids, temperature)`:
   - `unary, candidates = topk(logits, K=16, sorted=False)`; `hidden = proj(hidden)`.
   - Sequential walk over block positions: `scores = unary[:,pos] + einsum("br,bkr->bk", pred_codebook(pred) * hidden[:,pos], succ_codebook(cand[:,pos]))`; greedy → argmax; T>0 → `q = softmax(scores/temp)` over the K candidates, sample index.
   - `pred = candidates[:,pos].gather(-1, index)`; returns (path [B,T], candidates [B,T,K], q_rows [B,T,K] or None).
   - logits come from `compute_logits(draft_hidden, target_lm_head)` — the TARGET's head (× `output_multiplier`, optional softcap).
3. Checkpoint key map (from `models/dflash2-hf` safetensors): `fc.weight`, `hidden_norm.weight`, `layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `layers.{i}.self_attn.{q,k}_norm.weight`, `layers.{i}.mlp.{gate,up,down}_proj.weight`, `layers.{i}.input_layernorm.weight`, `layers.{i}.post_attention_layernorm.weight`, `layers.{i}.attention_conv.base_kernel`, `layers.{i}.attention_conv.kernel_projection.weight`, `layers.{i}.mlp_conv.*`, `candidate_selector.predecessor_codebook.weight`, `candidate_selector.successor_codebook.weight`, `candidate_selector.hidden_projection.weight`, `norm.weight`. (Verify at load time.)
4. DFlash2DraftModel config: arch `DFlash2DraftModel`, `dflash_config`: conv_kernel_size 2, conv_group_size 16, selector_rank 256, selector_top_k 16, block_size 8, mask_token_id 248070, target_layer_ids [5,19,33,47,61].

## Integration design

- `exllamav3/architecture/dflash2.py`: `DFlash2Config(DFlashConfig)` reads the four dflash_config fields; `DFlash2Model(DFlashModel)` adds:
  - conv modules per layer (keys `layers.{i}.attention_conv.*`, `layers.{i}.mlp_conv.*`) — inserted around the block's attn/mlp sublayers. Implementation: subclass/wrap `TransformerBlock` forward. ExLlamaV3 `TransformerBlock` is a Module with attn/mlp; check for a forward hook mechanism; if none, create `DFlash2Block(TransformerBlock)` overriding forward to call conv prepare/finish around sublayers.
  - `candidate_selector` module (fp16, uncalibrated).
  - caps += `dflash2_draft: True`.
  - `propose(state, anchor_ids, params, temperature)` → returns (tokens [B,T], candidates [B,T,K], q [B,T,K] or None).
- Generator:
  - `iterate_draftmodel_dflash_gen`: if dflash2 → run draft forward → compute logits via attached target head → `propose(...)` → stash `self._dflash2_propose = {"tokens","indices","q"}`; greedy tokens path unchanged otherwise.
  - `iterate_gen` verify loop: when `self.dflash2_draft` and job temperature > 0: acceptance per position uses rejection sampler: with u~U(0,1), accept draft token d_i iff `u < p_i/q_i` where `p_i` = target sampling-dist prob of d_i (computed from raw logits with job temp/top-p/top-k; penalties excluded — exact for Qwen defaults penalties=off, documented approximation otherwise), `q_i` = q value at the walked candidate index. On reject: bonus sampled from normalized `(p − q)+` residual over candidates ∪ {sampled}; then reject_remainder. Greedy path untouched.
- Conversion: backbone Linears quantize (uncalibrated_quantize already set); keep `candidate_selector.*` and `*.attention_conv.*`/`*.mlp_conv.*` fp16 via arch-level exclusion list (mechanism TBD in conversion spike — check how vision `-vb 16` "copy unquantized" is implemented; reuse it).
- model_init: `init()` draft path — `Config.from_directory` returns DFlash2Config by arch string; no new CLI flags.

## Risks carried from plan
- Quantized target head: propose + verify share the same EXL3 head → q consistent with p (lossless w.r.t. quantized model). Gate: acceptance + sampled-distribution checks.
- tap_shift: reference `extract_context_feature` uses `hidden_states[layer_id + 1]` (offset 1) — matches v1 default `tap_shift = 1`. Confirm via acceptance.
- Issue #285: check draft cache allocation at 262k.
