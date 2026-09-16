#!/usr/bin/env bash
# Container entrypoint: launch the validated chat.py path with env-var-driven flags.
# chat.py is an interactive console app (no HTTP server); PROMPT runs a single turn and
# exits, for headless validation and scripting.
#
# Click-run: if the model dirs are empty (e.g. a fresh named volume mounted at /models),
# the EXL3 models are downloaded from Hugging Face into the volume on first start.
set -euo pipefail

TARGET_REPO="${TARGET_REPO:-r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw}"
DRAFT_REPO="${DRAFT_REPO:-r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw}"

ensure_model () {
    local dir="$1" repo="$2"
    if [ -f "${dir}/config.json" ]; then
        return 0
    fi
    if [ "${AUTO_DOWNLOAD:-1}" != "1" ]; then
        echo "[entrypoint] model missing at ${dir} and AUTO_DOWNLOAD=0" >&2
        exit 1
    fi
    echo "[entrypoint] downloading ${repo} -> ${dir} (first run; this takes a while)"
    mkdir -p "${dir}"
    hf download "${repo}" --local-dir "${dir}"
}

ensure_model "${MODEL_DIR:-/models/qwen38-27b-exl3}" "${TARGET_REPO}"
ensure_model "${DRAFT_DIR:-/models/dflash2-exl3}" "${DRAFT_REPO}"

ARGS=(
    -m "${MODEL_DIR:-/models/qwen38-27b-exl3}"
    -mode "${MODE:-chatml}"
    -dm "${DRAFT_DIR:-/models/dflash2-exl3}"
    -cs "${CTX:-262144}"
    -cq "${CQ:-3}"
)

[ -n "${PROMPT:-}" ] && ARGS+=(-prompt "${PROMPT}")

# Extras passed through from the host, e.g. EXTRA_ARGS="-basic -tps -ndt 4"
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

cd /opt/exllamav3
exec python examples/chat.py "${ARGS[@]}"
