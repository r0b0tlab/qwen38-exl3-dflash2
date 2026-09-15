#!/usr/bin/env python3
"""
Sampled (T>0) sanity check for the DFlash2 q-aware verify: generate completions
with DFlash2 drafting on vs off and compare distributional statistics. The
q-aware path is lossless w.r.t. the quantized target, so the two distributions
should agree closely (mean length, token n-gram overlap, distinct-token rate).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))

from collections import Counter  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator  # noqa: E402
from exllamav3.generator.sampler.presets import CategoricalSampler  # noqa: E402


def build(args, draft):
    tcfg = Config.from_directory(args.target)
    draft_model = draft_cache = None
    max_history = 0
    if draft:
        dcfg = Config.from_directory(draft)
        draft_model = Model.from_config(dcfg)
        max_history = draft_model.caps.get("default_draft_size", 4)
    model = Model.from_config(tcfg)
    cache = Cache(model, max_num_tokens = 4096, max_history = max_history, max_batch_size = 1)
    model.load(progressbar = False)
    tokenizer = Tokenizer.from_config(tcfg)
    if draft_model is not None:
        draft_cache = Cache(draft_model, max_num_tokens = 4096, max_batch_size = 1)
        draft_model.load(progressbar = False)
    return Generator(model, cache, tokenizer, draft_model = draft_model, draft_cache = draft_cache), tokenizer


def stats(texts):
    lens = [len(t) for t in texts]
    toks = Counter()
    for t in texts:
        toks.update(t.split())
    vocab = len(toks)
    total = sum(toks.values())
    return {
        "mean_chars": sum(lens) / max(1, len(lens)),
        "distinct_token_rate": vocab / max(1, total),
        "top20_share": sum(c for _, c in toks.most_common(20)) / max(1, total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required = True)
    ap.add_argument("--draft", default = None, help = "draft dir; omit for autoregressive")
    ap.add_argument("--prompts", type = int, default = 24)
    ap.add_argument("--samples-per-prompt", type = int, default = 8)
    ap.add_argument("--max-new-tokens", type = int, default = 96)
    ap.add_argument("--temperature", type = float, default = 1.0)
    args = ap.parse_args()

    prompts = [
        "Explain why the sky is blue, briefly.",
        "Write a one-paragraph product description for a mechanical keyboard.",
        "Summarize the plot of Romeo and Juliet in three sentences.",
        "What are the main differences between TCP and UDP?",
        "Give me a recipe for pancakes.",
        "Describe how a transformer neural network works.",
    ][: max(1, min(6, args.prompts))]

    results = {}
    for label, draft in (("dflash2", args.draft), ("autoregressive", None)):
        gen, tokenizer = build(args, draft)
        sampler = CategoricalSampler(temperature = args.temperature)
        texts = []
        t0 = time.time()
        for p in prompts:
            for _ in range(args.samples_per_prompt):
                out = gen.generate(
                    prompt = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
                    max_new_tokens = args.max_new_tokens,
                    sampler = sampler,
                )
                texts.append(out.split("<|im_start|>assistant\n")[-1])
        dt = time.time() - t0
        results[label] = stats(texts)
        results[label]["tok_per_s"] = (
            len(texts) * args.max_new_tokens / dt
        )
        del gen
    print(json.dumps(results, indent = 2))


if __name__ == "__main__":
    main()
