#!/usr/bin/env bash
# Publish the converted EXL3 models to the r0b0tlab Hugging Face org.
# Requires: `hf auth login` with write access to the r0b0tlab org.
# Uploads are resumable; re-running continues where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_REPO="r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw"
DRAFT_REPO="r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw"

hf repo create "$TARGET_REPO" --repo-type model --public --exist-ok 2>/dev/null || true
hf repo create "$DRAFT_REPO" --repo-type model --public --exist-ok 2>/dev/null || true

# Small one first, so a failure surfaces fast; then the 16 GB target.
hf upload "$DRAFT_REPO" models/dflash2-exl3 --repo-type model
hf upload "$DRAFT_REPO" hf-cards/README-dflash2-exl3.md README.md --repo-type model

hf upload "$TARGET_REPO" models/qwen38-27b-exl3 --repo-type model
hf upload "$TARGET_REPO" hf-cards/README-qwen38-27b-exl3.md README.md --repo-type model

echo "published:"
echo "  https://huggingface.co/$DRAFT_REPO"
echo "  https://huggingface.co/$TARGET_REPO"
