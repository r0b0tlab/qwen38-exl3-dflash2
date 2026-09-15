#!/usr/bin/env python3
"""
Acceptance-length benchmark for the EXL3 Qwen3.8-27B target with a drafter
(DFlash2 EXL3, DFlash2 HF/BF16, or the built-in MTP head), on a GSM8K sample.

Usage:
  python3 acceptance_check.py --target models/qwen38-27b-exl3 --draft models/dflash2-exl3 --n 40
  python3 acceptance_check.py --target models/qwen38-27b-exl3 --draft mtp --n 40
  python3 acceptance_check.py --target models/qwen38-27b-exl3 --draft none --n 10   (baseline)

Greedy (argmax) decoding. Acceptance length = new_tokens / verification rounds,
with rounds = new_tokens - accepted_draft_tokens (every round emits its accepted
draft tokens plus exactly one target-sampled token).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))

import torch  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator  # noqa: E402


def build(args):
    tcfg = Config.from_directory(args.target)

    draft_model = None
    if args.draft == "mtp":
        draft_model = Model.from_config(tcfg, component = "mtp")
    elif args.draft != "none":
        dcfg = Config.from_directory(args.draft)
        draft_model = Model.from_config(dcfg)

    # Speculative decoding on recurrent (GDN/SWA) models reserves past-state slots for the
    # verify pass: max_history = number of draft tokens (mirrors model_init.init()).
    max_history = draft_model.caps.get("default_draft_size", 4) if draft_model else 0

    model = Model.from_config(tcfg)
    cache = Cache(model, max_num_tokens = args.ctx, max_history = max_history, max_batch_size = 1)
    model.load(progressbar = False)
    tokenizer = Tokenizer.from_config(tcfg)

    draft_cache = None
    if draft_model is not None:
        draft_cache = Cache(draft_model, max_num_tokens = args.ctx, max_batch_size = 1)
        draft_model.load(progressbar = False)

    gen = Generator(
        model, cache, tokenizer,
        draft_model = draft_model,
        draft_cache = draft_cache,
    )
    return gen, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required = True)
    ap.add_argument("--draft", required = True, help = "draft dir, 'mtp', or 'none'")
    ap.add_argument("--n", type = int, default = 40)
    ap.add_argument("--ctx", type = int, default = 8192)
    ap.add_argument("--max-new-tokens", type = int, default = 512)
    ap.add_argument("--json-out", default = None)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split = "test")

    gen, tokenizer = build(args)
    eos_ids = list(getattr(gen.model.config, "eos_token_id_list", None) or [])
    if not eos_ids and tokenizer.eos_token_id is not None:
        eos_ids = [tokenizer.eos_token_id]

    from exllamav3.generator.sampler.presets import ArgmaxSampler
    sampler = ArgmaxSampler()

    per_request = []
    for i in range(args.n):
        q = ds[i]["question"].strip()
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        t0 = time.time()
        _, r = gen.generate(
            prompt = prompt,
            max_new_tokens = args.max_new_tokens,
            sampler = sampler,
            stop_conditions = eos_ids,
            return_last_results = True,
        )
        dt = time.time() - t0

        r = r or {}
        new_tokens = r.get("new_tokens", 0)
        accepted = r.get("accepted_draft_tokens", 0)
        rounds = max(1, new_tokens - accepted)
        per_request.append({
            "new_tokens": new_tokens,
            "accepted_draft_tokens": accepted,
            "rejected_draft_tokens": r.get("rejected", 0),
            "rounds": rounds if new_tokens else 0,
            "acceptance_length": (new_tokens / rounds) if new_tokens else 0.0,
            "tok_per_s": (new_tokens / dt) if dt > 0 and new_tokens else 0.0,
            "time": dt,
        })

    usable = [p for p in per_request if p["new_tokens"] > 0]
    als = [p["acceptance_length"] for p in usable]
    tps = [p["tok_per_s"] for p in usable]
    summary = {
        "target": args.target,
        "draft": args.draft,
        "n": len(per_request),
        "mean_acceptance_length": sum(als) / max(1, len(als)),
        "mean_tok_per_s": sum(tps) / max(1, len(tps)),
        "hit_token_cap": sum(1 for p in per_request if p["new_tokens"] >= args.max_new_tokens),
    }
    print(json.dumps(summary, indent = 2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"summary": summary, "per_request": per_request}, f, indent = 2)


if __name__ == "__main__":
    main()
