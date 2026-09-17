# Q200v2 + max-context NIAH — native container campaign (2026-09-17)

Endpoint: `qwen38-exl3-dflash2:1.5.0-native` (r0b0tlab/exllamav3 community @ 355c6ee,
sm_86 in-image build). Serve is `scripts/serve_openai.py` inside that container
(`--cache-tokens 270336`, cq3, DFlash2, one sequence slot). GPU attach is via
`container/run-serve.sh` (device binds; this host has no nvidia-container-toolkit).

Client: host r0b0bench Q200v2 kit, 1 worker, against `http://127.0.0.1:8889`.
Dataset sha256 `66a75701cbeea69f212e1c8be92aab9efaf3fa4d7af3c6911c8f7864a17d8d14`
(post-correction kit). Image id `sha256:982fad27a8b4cc7c3228c4d535e7d526a71a7900dde35f465a030fcd24ba3d71`.
Run identity `7fd44c0cff81aee0df75e06ebdd0ade2d63c5903ce534a4cd4f0a2ffe9b0b10d`.
Chat kwargs `{enable_thinking, thinking, reasoning_effort=low}`; max_tokens 8192.

Raw rows stay under `/home/am/r0b0bench-q200v2/runs/q200v2-native-container-20260917/`.

## Q200v2 text-180

| family | n | transported | graded | correct | accuracy |
| --- | --- | --- | --- | --- | --- |
| gsm8k | 80 | 80 | 80 | 78 | 97.5 % |
| humaneval | 40 | 40 | 40 | 40 | 100 % |
| ifeval | 40 | 39 | 39 | 34 | 87.18 % |
| hard_reasoning | 20 | 20 | 20 | 20 | 100 % |

correct 172 / incorrect 7 / ungraded 1. The ungraded row is the disclosed transport
failure `ifeval-023` (8192 ceiling, `finish_reason=length`). Cap was not raised.
hard_reasoning graded by independent review (`metrics/q200v2/manual-evidence.json`,
sha256 `306ca3cf…`).

Humaneval sandbox timeout is the kit default 8 s (`cpu_seconds` 9 in the sandbox
manifest). A 120 s override makes the sandbox receipt fail closed as ValueError.

**E2E throughput** (PROCEDURES §4, all 180 rows including the length-truncated one):
mean 152.82 / p50 159.29 / aggregate 144.73 tok/s (115,404 completion tokens).

**Telemetry** (2 s, 3277 s span, 1074 load samples with util > 0): power mean 336.4 /
max 351.8 W; temp mean 58.6 / max 66.0 C; util mean 98.5 %; clock mean 1754 MHz;
VRAM mean 22.3 / max 23.4 GiB. Throttle bits: idle (0x1) and SW power cap (0x4).
No HW thermal slowdown.

**BFCL-hard20: NOT_IMPLEMENTED** — this serve has no tools/function-calling surface.

## Max-context multi-needle NIAH (262,080 = 262,144 − 64)

Generation reserve 256 (thinking-enabled serve; kit reference is 64). Disclosed.

| variant | needles | elapsed | prompt_tokens | result |
| --- | --- | --- | --- | --- |
| 2n | 33 % / 66 % | 613.8 s | 262080 | **PASS** (last = R0B0-LYNX-4402) |
| 3n | 33 % / 66 % / 90 % | 122.8 s | 262080 | **PASS** (last = R0B0-RAVEN-9158) |

3n wall time is cross-request prefix/KV reuse on the same serve, not prefill speed.
Client on the serve host. One request per variant.
