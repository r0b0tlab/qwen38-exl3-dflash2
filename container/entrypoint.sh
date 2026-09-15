#!/usr/bin/env bash
# Container entrypoint: launch the validated chat.py path with env-var-driven flags.
# chat.py is an interactive console app (no HTTP server); PROMPT runs a single turn and
# exits, for headless validation and scripting.
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR must point at the EXL3 target model (mounted)}"
: "${DRAFT_DIR:?DRAFT_DIR must point at the EXL3 DFlash2 draft (mounted)}"

ARGS=(
  -m "${MODEL_DIR}"
  -mode "${MODE:-chatml}"
  -dm "${DRAFT_DIR}"
  -cs "${CTX:-262144}"
  -cq "${CQ:-3}"
)

[ -n "${PROMPT:-}" ] && ARGS+=(-prompt "${PROMPT}")

# Extras passed through from the host, e.g. EXTRA_ARGS="-basic -tps -ndt 4"
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

cd /opt/exllamav3
exec python examples/chat.py "${ARGS[@]}"