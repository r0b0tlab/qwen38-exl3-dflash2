#!/usr/bin/env python3
"""VRAM budget gate for Qwen3.8-27B EXL3 + DFlash2 on 24 GB."""
import sys

PARAMS_B = 27.0
VOCAB = 248320
HIDDEN = 5120
FULL_ATTN_LAYERS = 16   # 64 layers, full_attention_interval 4
KV_HEADS = 4
HEAD_DIM = 256
CTX = 262_144
CARD_GB = 24.0

bpw = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
cq = int(sys.argv[2]) if len(sys.argv) > 2 else 4
head_bpw = 6
draft_bpw = 4.0

weights = PARAMS_B * bpw / 8                      # GB (decimal approx)
head = VOCAB * HIDDEN * head_bpw / 8 / 1e9
vision = 0.35
kv = FULL_ATTN_LAYERS * KV_HEADS * HEAD_DIM * 2 * 2 * CTX / 1e9 * (cq / 16)
staging = 2 * CTX * KV_HEADS * HEAD_DIM * 2 / 1e9  # EXL3_QC_STAGING=1 reservation
recurrent = 0.5
draft = 2.0 * draft_bpw / 8 + 0.3                  # backbone + fp16 codebooks
draft_kv = 0.04
overhead = 1.5

total = weights + head + vision + kv + staging + recurrent + draft + draft_kv + overhead
print(f"bpw={bpw} cq={cq}: weights={weights:.1f} head={head:.2f} kv={kv:.1f} "
      f"staging={staging:.1f} draft={draft:.2f} total={total:.1f} / {CARD_GB} GB")
sys.exit(0 if total < CARD_GB - 0.5 else 1)
