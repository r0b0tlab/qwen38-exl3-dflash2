# Results log — Qwen3.8-27B EXL3 + DFlash2 pathway on RTX 3090 (2026-09-15)

## Artifacts

- Target `models/qwen38-27b-exl3` — EXL3 4.00 bpw, head 6 bpw, vision 6 bpw, MTP 4 bpw
  (16 GB, 2 shards), converted from `Qwen/Qwen3.8-27B` with `-b 4.00 -vb 6 -mb 4 -hb 6`
  (single pass, exit 0; MTP + vision quantized).
- Draft `models/dflash2-exl3` — EXL3 4.00 bpw (1.2 GB) from `incoai/Qwen3.8-27B-DFlash2`;
  conv `base_kernel` + selector codebooks fp16 (uncalibrated).
- Engine fork `exllamav3`, branch `dflash2-pathway`, tip `4960047` (8 commits over v1.5.0).
- Container image `qwen38-exl3-dflash2:1.5.0`.

## Gates

### VRAM budget (Tasks 1.2 / 3.1)

Gate 4.00 bpw + cq3 = 22.4 GB -> pass; runtime flags `-cq 3`. Measured peak at 262k
context: 23.14-23.17 GB (fits the 24 GB card).

### Acceptance (Task 3.3) — ALL PASS

GSM8K test split, greedy, max 512 new tokens, ctx 8192:

| Drafter | Mean acceptance length | tok/s |
| --- | --- | --- |
| DFlash2 EXL3 (4.00 bpw) | 5.474 | 152.4 |
| DFlash2 BF16 | 5.463 | 139.9 |
| MTP head | 4.101 | 113.9 |
| Autoregressive | 1.000 | 41.8 |

- DFlash2 >= 4.0: PASS. DFlash2 > MTP + 0.3: PASS (5.474 vs 4.401). EXL3 draft >=
  BF16 draft - 0.15: PASS (5.474 vs 5.463) — draft quantization is acceptance-neutral.
- Full detail + harness notes: `notes/ACCEPTANCE.md`; raw per-request: `notes/acceptance-*.json`.

### 262k context (Tasks 3.2 / 3.4)

- Load at 262,144 ctx with `-cq 3`: no OOM, peak 23.1-23.2 GB. Issue #285 (full-size draft
  KV for all-SWA layers) did not materialize — the budget held.
- Prefill 150,000 tokens: 250.5 s = 599 tok/s.
- Decode 200 tokens at 150k depth: 24.5 tok/s, acceptance 1.79 (window 7). Width-4 retest on
  the identical prompt: acceptance 1.754, 24.7 tok/s — no width sensitivity on this stack
  (llama.cpp's width-4 > width-7 finding at depth did not reproduce here; the decay is
  draft-vs-depth on open-ended text, not a window effect).
- Raw: `notes/long-context-draft7.log`, `notes/long-context-draft4.log`.

### T=1.0 sampled sanity (Task 3.5) — PASS

DFlash2 vs autoregressive, 48 samples each: mean length 451 vs 452 chars, top-20 share
0.312 vs 0.315, distinct-token rate 0.264 vs 0.248. No distributional divergence.
Detail: `notes/LOSSLESS.md`.

### Engine tests

27/27 `ntest_dflash2_*` (config, conv/selector parity vs the z-lab reference, verify,
real-checkpoint load) green on the final tree; existing generator tests pass.

## Integration fixes found while running the gates (all committed)

- `b8e1a00` conv output dtype follows input (fp32 conv result crashed the fp16 EXL3 GEMMs)
- `aa6adf8` CandidateSelector is a no-op in the module walk (selection runs via `select()`)
- `4960047` full draft window kept in verify — the v1 anchor crop was double-applied on the
  DFlash2 path (dropped a draft token + misaligned the selector's candidates/q), and
  selector anchor indices now follow the codebook device
- `scripts/` — target cache built with `max_history = draft window` (GDN verify requirement;
  mirrors `model_init.init()`)
- `container/` — entrypoint fixed (`chat.py` flags, `-prompt` one-shot), uv venv Python
  relocated out of `/root`, build-time smoke as the runtime user

## Pending

- Container GPU validation run (one-shot `-prompt`).
- GitHub publication package.