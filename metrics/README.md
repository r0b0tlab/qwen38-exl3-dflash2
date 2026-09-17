# Metrics — Q200v2 + NIAH (native container, 2026-09-17)

Sanitized summaries. Raw rows stay in `/home/am/r0b0bench-q200v2/runs/q200v2-native-container-20260917/`.

| file | what |
| --- | --- |
| `q200v2/summary.json` | Kit-produced run summary (frozen Q200v2 text-180, dataset `66a75701…`). |
| `q200v2/throughput-digest.json` | E2E throughput per PROCEDURES §4 over n=180. |
| `q200v2/telemetry.tsv` + `telemetry-digest.json` | 2 s host telemetry; digest is load-only (util > 0). |
| `q200v2/manual-evidence.json` | Independent review for 20 hard_reasoning rows. |
| `niah/niah-2n.json`, `niah/niah-3n.json` | Max-context multi-needle NIAH at 262,080 tokens. |
| `concurrency/ladder-1-2-4.json` | Overlay-era concurrency ladder (not re-run this campaign). |

Method notes:

- Q200v2 identity: dataset sha256 `66a75701…`, run identity `7fd44c0c…`, serve image
  `sha256:982fad27…`; chat kwargs `{enable_thinking, thinking, reasoning_effort=low}`;
  max_tokens 8192; 1 worker. `ifeval-023` length-truncated at 8192 (disclosed).
- hard_reasoning 20/20 via `--manual-evidence` (reviewer `hermes-agent`, method
  `independent_manual_review`, evidence sha256 `306ca3cf…`).
- Humaneval sandbox image `sha256:caf1a95b…` with kit default timeout 8 s.
- NIAH generation reserve 256 (thinking). 3n elapsed is prefix/KV reuse after 2n.
- BFCL-hard20: NOT_IMPLEMENTED (no tools surface).
- Serve: container `qwen38-exl3-dflash2:1.5.0-native` via `container/run-serve.sh`.
