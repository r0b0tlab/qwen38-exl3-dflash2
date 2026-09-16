#!/usr/bin/env python3
"""
Concurrency sweep for the DFlash2 pathway: 1/2/4/8 concurrent sequences in one
generator session (single model load), greedy, fixed prompt set. Reports aggregate
and per-sequence throughput plus acceptance length per batch size.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))

import torch  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job  # noqa: E402
from exllamav3.generator.sampler.presets import ArgmaxSampler  # noqa: E402

PROMPTS = [
    "Explain why the sky is blue, briefly.",
    "Write a one-paragraph description of a mechanical keyboard.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What are the main differences between TCP and UDP?",
    "Give me a recipe for pancakes.",
    "Describe how a transformer neural network works.",
    "List five countries in South America and their capitals.",
    "Explain photosynthesis in one sentence.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ctx-per-seq", type = int, default = 1024,
                    help = "Per-sequence token budget; total cache = ctx_per_seq * max(batch-sizes)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    bss = [int(b) for b in args.batch_sizes.split(",")]
    max_bs = max(bss)

    tcfg = Config.from_directory(args.target)
    dcfg = Config.from_directory(args.draft)
    draft_model = Model.from_config(dcfg)
    max_history = draft_model.caps.get("default_draft_size", 4)

    model = Model.from_config(tcfg)
    cache = Cache(model, max_num_tokens = args.ctx_per_seq * max_bs,
                  max_history = max_history, max_batch_size = max_bs)
    model.load(progressbar = False)
    tokenizer = Tokenizer.from_config(tcfg)

    draft_cache = Cache(draft_model, max_num_tokens = args.ctx_per_seq * max_bs,
                        max_batch_size = max_bs)
    draft_model.load(progressbar = False)

    gen = Generator(model, cache, tokenizer, draft_model = draft_model, draft_cache = draft_cache)

    eos_ids = list(getattr(model.config, "eos_token_id_list", None) or [])
    if not eos_ids and tokenizer.eos_token_id is not None:
        eos_ids = [tokenizer.eos_token_id]

    def phase(n, max_new, tag):
        jobs = []
        for i in range(n):
            p = f"<|im_start|>user\n{PROMPTS[i % len(PROMPTS)]}<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer.encode(p)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            jobs.append(Job(input_ids = ids, max_new_tokens = max_new, stop_conditions = eos_ids,
                            sampler = ArgmaxSampler(), identifier = tag))
        t0 = time.time()
        for j in jobs:
            gen.enqueue(j)
        total_new = 0
        total_acc = 0
        done = 0
        first_token_t = None
        sample_keys = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and first_token_t is None:
                    first_token_t = time.time()
                if r.get("eos"):
                    if sample_keys is None:
                        sample_keys = sorted(r.keys())
                    done += 1
                    total_new += int(r.get("new_tokens", 0) or 0)
                    total_acc += int(r.get("accepted_draft_tokens", 0) or 0)
        dt = time.time() - t0
        return {
            "batch": n,
            "wall_s": round(dt, 3),
            "jobs_done": done,
            "total_new_tokens": total_new,
            "tokens_per_seq": round(total_new / max(1, n), 1),
            "aggregate_tps": round(total_new / dt, 2) if dt else None,
            "per_seq_tps": round(total_new / dt / n, 2) if dt else None,
            "acceptance_length": round(total_new / max(1, total_new - total_acc), 3) if total_new else None,
            "ttft_s": round(first_token_t - t0, 3) if first_token_t else None,
            "eos_event_keys": sample_keys,
        }

    results = []
    phase(1, 64, "warmup")  # discard: cudagraph capture + first-touch costs
    for n in bss:
        r = phase(n, args.max_new_tokens, f"bsz{n}")
        r["tag"] = f"bsz{n}"
        print(json.dumps(r), flush = True)
        results.append(r)

    out = {
        "target": args.target,
        "draft": args.draft,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    print(json.dumps(out, indent = 2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent = 2)


if __name__ == "__main__":
    main()
