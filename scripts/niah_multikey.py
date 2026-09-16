#!/usr/bin/env python3
"""
Multi-needle NIAH at max context against the local OpenAI-compatible server.

Mirrors the r0b0bench NIAH lane mechanics (render -> splice needles into the rendered
prompt -> raw /v1/completions), with two variants:
  2n: two needles at 33%% / 66%% of the target window; answer = the LAST code
  3n: three needles at 33%% / 66%% / 90%%; answer = the LAST code (repo multi-key protocol)

Disclosed deviations from docs/PROCEDURES.md section 2: generation reserve 256 instead
of 64 (thinking-enabled serve), client on the serve host, single request per depth set.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))

from exllamav3 import Config, Tokenizer  # noqa: E402

CODES = {
    "2n": ["R0B0-ZEBRA-7731", "R0B0-LYNX-4402"],
    "3n": ["R0B0-ZEBRA-7731", "R0B0-LYNX-4402", "R0B0-RAVEN-9158"],
}
FRACTIONS = {"2n": [0.33, 0.66], "3n": [0.33, 0.66, 0.90]}
UNIT = " The archival record contains neutral observations about weather, tools, books, roads, and ordinary daily events."
MARKER = "R0B0BENCH_NIAH_SLOT"


def post(url, payload, timeout):
    req = urllib.request.Request(url, data = json.dumps(payload).encode(),
                                 headers = {"Content-Type": "application/json"}, method = "POST")
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.loads(r.read())


def encode(tokenizer, text):
    ids = tokenizer.encode(text)
    if ids.dim() == 2:
        ids = ids[0]
    return ids.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default = "http://127.0.0.1:8889")
    ap.add_argument("--target", required = True, help = "model dir for the tokenizer")
    ap.add_argument("--variant", choices = ["2n", "3n"], required = True)
    ap.add_argument("--target-tokens", type = int, default = 262080)
    ap.add_argument("--max-tokens", type = int, default = 256)
    ap.add_argument("--timeout", type = int, default = 7200)
    ap.add_argument("--json-out", default = None)
    args = ap.parse_args()

    tcfg = Config.from_directory(args.target)
    tokenizer = Tokenizer.from_config(tcfg)

    instruction = (
        "You are given a long archival record. Secret verification codes appear in order, "
        "each exactly once.\n"
        + MARKER + "\n"
        "Question: What is the LAST secret verification code? Reply with only the code, nothing else."
    )
    rendered = post(args.base_url + "/v1/chat/completions/render", {
        "messages": [{"role": "user", "content": instruction}],
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"},
    }, timeout = 300)["token_ids"]

    marker_ids = encode(tokenizer, MARKER)
    pos = None
    for i in range(len(rendered) - len(marker_ids) + 1):
        if rendered[i:i + len(marker_ids)] == marker_ids:
            pos = i
            break
    if pos is None:
        print(json.dumps({"error": "marker not found in rendered prompt"}))
        return 2

    prefix, suffix = rendered[:pos], rendered[pos + len(marker_ids):]
    codes = CODES[args.variant]
    fractions = FRACTIONS[args.variant]
    unit = encode(tokenizer, UNIT)
    needles = [encode(tokenizer, f"\nIMPORTANT SECRET VERIFICATION CODE: {c}\n") for c in codes]

    body_budget = args.target_tokens - len(prefix) - len(suffix)
    if body_budget < sum(len(n) for n in needles) + 4096:
        print(json.dumps({"error": "target window too small", "body_budget": body_budget}))
        return 2

    # Lay the body out as filler + needle segments at the requested fractions.
    cuts = [int(body_budget * f) for f in fractions]
    body = []
    cursor = 0
    for cut, needle in zip(cuts, needles):
        gap = cut - cursor
        reps = max(1, (gap // len(unit)) + 1)
        seg = (unit * reps)[:max(0, gap)]
        body += seg
        body += needle
        cursor = cut + len(needle)
    tail_gap = body_budget - cursor
    body += (unit * (max(1, (tail_gap // len(unit)) + 1)))[:max(0, tail_gap)]
    if len(body) < body_budget:
        body += (unit * ((body_budget - len(body)) // len(unit) + 2))[: body_budget - len(body)]
    body = body[:body_budget]

    input_ids = prefix + body + suffix
    assert len(input_ids) == args.target_tokens, (len(input_ids), args.target_tokens)

    t0 = time.time()
    resp = post(args.base_url + "/v1/completions", {
        "model": "qwen38-27b-exl3-dflash2",
        "prompt": input_ids,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "skip_special_tokens": True,
    }, timeout = args.timeout)
    elapsed = time.time() - t0

    ch = resp["choices"][0]
    text = (ch.get("text") or "").strip()
    last_code = codes[-1]
    row = {
        "variant": args.variant,
        "fractions": fractions,
        "codes": codes,
        "target_tokens": args.target_tokens,
        "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
        "elapsed_s": round(elapsed, 1),
        "finish_reason": ch.get("finish_reason"),
        "last_code": last_code,
        "passed": last_code in text,
        "all_codes_found": [c for c in codes if c in text],
        "response_text": text[:400],
    }
    print(json.dumps(row, indent = 2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(row, f, indent = 2)
    return 0 if row["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
