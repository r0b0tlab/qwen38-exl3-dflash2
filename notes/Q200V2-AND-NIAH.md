# Q200v2 + max-context NIAH — campaign results (2026-09-16)

Local RTX 3090 run of the r0b0bench **Q200v2** frozen kit (text-180 lane) and
max-context **multi-needle NIAH**, against `qwen38-27b-exl3-dflash2`
(Qwen3.8-27B EXL3 4.00 bpw + DFlash2 EXL3 draft, fork branch `dflash2-pathway`).

Serve: `scripts/serve_openai.py` (minimal OpenAI-compatible server; non-streaming;
thinking split into `reasoning_content`). Campaign artifacts under
`/home/am/r0b0bench-q200v2/runs/` (per-run RESULTS.md + raw rows/summary/telemetry).

## Q200v2 text-180

| family | n | transported | graded | correct | accuracy |
| --- | --- | --- | --- | --- | --- |
| gsm8k | 80 | 80 | 80 | 79 | 98.75 % |
| humaneval | 40 | 40 | 40 | 40 | 100 % |
| ifeval | 40 | 39 | 39 | 33 | 84.62 % |
| hard_reasoning | 20 | 20 | 20 | 19 | 95.0 % |

correct 171 / incorrect 8 / ungraded 1; the only ungraded row is the disclosed transport
failure `ifeval-023` (8192 ceiling). hard_reasoning was graded by independent review
(manual-evidence sha256 `af5157fb…`); 19/20 pass, and four frozen-reference defects were
found and documented in the run dir.

- Dataset sha256 `74623ab9...`; run identity `c649677d...`; image `sha256:caf1a95b...`.
- Kit revision (2026-09-16): four defective hard_reasoning entries were corrected upstream
  post-run; the dataset sha256 is now `66a75701...` — this run scored the pre-revision bytes.
- Chat kwargs `{enable_thinking, thinking, reasoning_effort=low}`; max_tokens 8192; 1 worker.
- The published kit was missing two imported modules (`admission_control`, `niah_common`);
  provided as PYTHONPATH shims, frozen bytes untouched.

**E2E throughput** (per PROCEDURES section 4): mean 142.71 / p50 148.96 / aggregate
130.64 tok/s (n=180; 104,725 completion tokens).

**Telemetry** (2 s, 848 s, load-only): power mean 333.4 / max 349.6 W; temp mean 54.4 /
max 59.0 C; util mean 91.9 %; clock mean 1696 MHz; VRAM 21.2 GB; no thermal throttle.

**BFCL-hard20 companion lane: NOT_IMPLEMENTED** — the wrapper drives official `bfcl-eval`
multi-turn tool calling; this serve path has no function-calling surface, so the lane is
explicitly not scored.

## Max-context multi-needle NIAH (262,080 tokens = 262,144 - 64)

| variant | needles | elapsed | result |
| --- | --- | --- | --- |
| 2n | 33 % / 66 % | 611.5 s | **PASS** |
| 3n | 33 % / 66 % / 90 % | 123.3 s | **PASS** |

3n's speed is cross-request prefix/KV reuse (shared ~173k-token prefix), not prefill speed.
Telemetry: 23.29 GB VRAM, 98.6 % util, no thermal throttle.

## Concurrency (DFlash2 raw path, 4-slot ladder)

| batch | aggregate tok/s | per-seq | acceptance | TTFT |
| --- | --- | --- | --- | --- |
| 1 | 109.0 | 109.0 | 3.66 | 94 ms |
| 2 | 134.6 | 67.3 | 3.14 | 163 ms |
| 4 | 152.1 | 38.0 | 3.35 | 350 ms |

Slots cost ~1.2 GB each (fp32 GDN verify states), which caps the card at 4 concurrent
sequences; batch>1 required a contiguous-slice fix in the draft path (commit `5cdf6bc`).
