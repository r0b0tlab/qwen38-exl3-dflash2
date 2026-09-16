#!/usr/bin/env python3
"""
PR verification: the draft-attach guard must fire when the target cache lacks
recurrent history, using the real models and the exact failure configuration
(Cache(max_history=0) with a DFlash2 draft attached).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator  # noqa: E402

TARGET = os.path.expanduser("~/exl3-qwen38-dflash2/models/qwen38-27b-exl3")
DRAFT = os.path.expanduser("~/exl3-qwen38-dflash2/models/dflash2-exl3")

tcfg = Config.from_directory(TARGET)
dcfg = Config.from_directory(DRAFT)
draft_model = Model.from_config(dcfg)
model = Model.from_config(tcfg)

cache = Cache(model, max_num_tokens = 8192, max_batch_size = 1)  # no recurrent history
draft_cache = Cache(draft_model, max_num_tokens = 8192, max_batch_size = 1)
model.load(progressbar = False)
draft_model.load(progressbar = False)
tokenizer = Tokenizer.from_config(tcfg)

assert cache.recurrent_layers, "target has no recurrent layers; guard check is not meaningful"
try:
    Generator(model, cache, tokenizer, draft_model = draft_model, draft_cache = draft_cache)
except ValueError as exc:
    message = str(exc)
    assert "max_history" in message, message
    print("GUARD OK:", message[:160])
    sys.exit(0)
print("GUARD FAILED: Generator constructed without recurrent history")
sys.exit(1)
