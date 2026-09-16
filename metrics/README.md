# Metrics — Q200v2 + NIAH + concurrency (RTX 3090 campaign, 2026-09-16)

Sanitized summaries only; raw traces stay in the local run directories
(`/home/am/r0b0bench-q200v2/runs/`), per r0b0bench rules.

| file | what |
| --- | --- |
| `q200v2/summary.json` | Kit-produced run summary (frozen Q200v2 text-180). Scores, per-family transport/grade counts, identity hashes, response-budget audit. |
| `q200v2/throughput-digest.json` | E2E throughput per PROCEDURES section 4: `completion_tokens / elapsed_seconds` per row, mean / p50 / aggregate over n=180. |
| `q200v2/telemetry.tsv` + `telemetry-digest.json` | Serve-host telemetry at 2 s cadence (power, temp, util, clock, throttle, VRAM, MemAvailable, swap); digest is load-only mean/max plus min MemAvailable and throttle states. |
| `niah/niah-2n.json`, `niah/niah-3n.json` | Max-context multi-needle NIAH at 262,080 tokens (33/66 % and 33/66/90 %, answer = last). |
| `niah/telemetry.tsv` + `telemetry-digest.json` | Telemetry for the NIAH phase. |
| `concurrency/ladder-1-2-4.json` | DFlash2 draft-path concurrency ladder (batch 1/2/4): aggregate and per-sequence tok/s, acceptance, TTFT. |

Method notes:

- Q200v2 identity: dataset sha256 `74623ab9…`, run identity `c649677d…`, image
  `sha256:caf1a95b…`; chat kwargs `{enable_thinking, thinking, reasoning_effort=low}`;
  max_tokens 8192; 1 worker. One row (`ifeval-023`) was length-truncated at the 8192
  ceiling and is disclosed as a transport failure (fail-closed).
- 20 `hard_reasoning` rows are `ungraded` pending independent manual review
  (`--manual-evidence`); status INCOMPLETE is the expected state without it.
- NIAH deviations disclosed: generation reserve 256 (thinking-enabled serve), client on the
  serve host, one request per variant; the 3n prompt shares a ~173k-token prefix with 2n, so
  its shorter wall time reflects cross-request prefix reuse, not raw prefill speed.
- BFCL-hard20 companion lane: **NOT_IMPLEMENTED** — the wrapper drives official `bfcl-eval`
  multi-turn tool-calling; this serve path has no function-calling surface, and scoring it
  would be a fabricated result.
