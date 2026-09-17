#!/usr/bin/env python3
"""
Minimal OpenAI-compatible server for the EXL3 Qwen3.8-27B + DFlash2 stack.

Implements exactly what the r0b0bench / Q200v2 harnesses need:
  GET  /health
  GET  /v1/models                       (max_model_len = 262144)
  POST /v1/chat/completions             (non-streaming; chat_template_kwargs honored)
  POST /v1/chat/completions/render      (exact chat-template token ids)
  POST /v1/completions                  (raw prompt str or token-id list)

Cache is quantized (cq3), full advertised 262k context, one sequence slot
(max_history = draft window for the DFlash2 verify pass). Requests are served
FIFO through a single worker thread; the generator batches nothing larger than
one job at this context on a 24 GB card.

Usage:
  python3 scripts/serve_openai.py --target models/qwen38-27b-exl3 \
      --draft models/dflash2-exl3 --port 8889
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))

import torch  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402
from exllamav3.generator.sampler.presets import ArgmaxSampler, CategoricalSampler  # noqa: E402

MAX_BODY = 32 * 1024 * 1024


class Server:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.max_model_len = args.max_model_len

        tcfg = Config.from_directory(args.target)
        dcfg = Config.from_directory(args.draft)
        draft_model = Model.from_config(dcfg)
        max_history = draft_model.caps.get("default_draft_size", 4)

        model = Model.from_config(tcfg)
        print(f"[serve] loading target {args.target} (cache {args.cache_tokens} tokens, cq{args.cq}) ...", flush=True)
        t0 = time.time()
        cache = Cache(model, max_num_tokens = args.cache_tokens, layer_type = CacheLayer_quant,
                      k_bits = args.cq, v_bits = args.cq,
                      max_history = max_history, max_batch_size = 1)
        model.load(progressbar = False)
        tokenizer = Tokenizer.from_config(tcfg)

        draft_cache = Cache(draft_model, max_num_tokens = args.cache_tokens, layer_type = CacheLayer_quant,
                            k_bits = args.cq, v_bits = args.cq, max_batch_size = 1)
        draft_model.load(progressbar = False)
        self.gen = Generator(model, cache, tokenizer, draft_model = draft_model, draft_cache = draft_cache)
        self.tokenizer = tokenizer
        free, total = torch.cuda.mem_get_info()
        print(f"[serve] loaded in {time.time()-t0:.0f}s; VRAM in use {(total-free)/1e9:.2f} GB", flush=True)

        eos_ids = list(getattr(model.config, "eos_token_id_list", None) or [])
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(tokenizer.eos_token_id)
        self.stop_ids = eos_ids

        self.queue = queue.Queue()
        self.worker = threading.Thread(target = self._worker, daemon = True)
        self.worker.start()
        print("[serve] READY", flush=True)

    # ---------------------------------------------------------------- worker

    def submit(self, input_ids, max_new_tokens, temperature, skip_special_tokens, want_ids):
        """Queue one job, block until it completes. Returns a result dict."""
        req_id = uuid.uuid4().hex
        holder = {"event": threading.Event(), "result": None, "error": None}
        self.queue.put({
            "id": req_id,
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "decode_special_tokens": not skip_special_tokens,
            "holder": holder,
            "want_ids": want_ids,
        })
        if not holder["event"].wait(timeout = 7200):
            raise TimeoutError("generation timed out (7200 s)")
        if holder["error"]:
            raise RuntimeError(holder["error"])
        return holder["result"]

    def _worker(self):
        while True:
            spec = self.queue.get()
            try:
                sampler = ArgmaxSampler() if not spec["temperature"] else \
                    CategoricalSampler(temperature = spec["temperature"])
                ids = spec["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                job = Job(
                    input_ids = ids,
                    max_new_tokens = spec["max_new_tokens"],
                    sampler = sampler,
                    stop_conditions = self.stop_ids,
                    decode_special_tokens = spec["decode_special_tokens"],
                    identifier = spec["id"],
                )
                t0 = time.time()
                self.gen.enqueue(job)
                final = None
                out_ids = []
                while self.gen.num_remaining_jobs():
                    for r in self.gen.iterate():
                        if r.get("identifier") == spec["id"]:
                            if r.get("stage") == "streaming":
                                tid = r.get("token_ids")
                                if tid is not None:
                                    out_ids.extend(tid.torch().flatten().tolist() if hasattr(tid, "torch") else list(tid))
                            if r.get("eos"):
                                final = r
                r = final or {}
                text = r.get("text")
                if text is None:
                    text = r.get("full_completion") or ""
                spec["holder"]["result"] = {
                    "text": text,
                    "prompt_tokens": int(r.get("prompt_tokens") or ids.shape[-1]),
                    "completion_tokens": int(r.get("new_tokens") or 0),
                    "eos_reason": r.get("eos_reason"),
                    "token_ids": out_ids if spec["want_ids"] else None,
                    "elapsed": time.time() - t0,
                    "accepted_draft_tokens": int(r.get("accepted_draft_tokens") or 0),
                }
            except Exception as exc:  # noqa: BLE001
                spec["holder"]["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                spec["holder"]["event"].set()

    # ---------------------------------------------------------------- helpers

    def render_chat(self, messages, template_kwargs):
        ids = self.tokenizer.hf_chat_template(
            messages, add_generation_prompt = True, **(template_kwargs or {})
        )
        return ids

    def finish_reason(self, r):
        if r.get("eos_reason") == "max_new_tokens":
            return "length"
        return "stop"


SERVE: Server = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):  # noqa: N802
        if self.path in ("/health", "/v1/health"):
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {
                "object": "list",
                "data": [{
                    "id": SERVE.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "max_model_len": SERVE.max_model_len,
                }],
            })
        else:
            self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):  # noqa: N802
        try:
            body = self._read_body()
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": {"message": f"invalid JSON: {exc}"}})
        try:
            if self.path == "/v1/chat/completions/render":
                return self._render(body)
            if self.path == "/v1/chat/completions":
                if body.get("stream"):
                    return self._send(400, {"error": {"message": "streaming not implemented"}})
                return self._chat(body)
            if self.path == "/v1/completions":
                if body.get("stream"):
                    return self._send(400, {"error": {"message": "streaming not implemented"}})
                return self._completions(body)
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
        return self._send(404, {"error": {"message": "not found"}})

    def _render(self, body):
        messages = body.get("messages") or []
        kwargs = body.get("chat_template_kwargs") or {}
        ids = SERVE.render_chat(messages, kwargs)
        self._send(200, {"token_ids": ids[0].tolist() if ids.dim() == 2 else ids.tolist()})

    def _chat(self, body):
        model_name = body.get("model") or SERVE.model_name
        messages = body.get("messages") or []
        template_kwargs = body.get("chat_template_kwargs") or {}
        ids = SERVE.render_chat(messages, template_kwargs)
        max_tokens = int(body.get("max_tokens") or 4096)
        temperature = float(body.get("temperature") or 0)
        r = SERVE.submit(ids, max_tokens, temperature,
                         skip_special_tokens = body.get("skip_special_tokens", True),
                         want_ids = False)
        # Thinking models: keep the reasoning trace out of `content` so quality graders
        # judge the answer, not the reasoning (vLLM reasoning-parser behavior).
        text = r["text"]
        reasoning_content = None
        content = text
        if "</think>" in text:
            reasoning_content, content = text.split("</think>", 1)
            content = content.lstrip("\n ")
            if not content.strip():
                content = text
        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning_content": reasoning_content},
                "finish_reason": SERVE.finish_reason(r),
            }],
            "usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            },
        })

    def _completions(self, body):
        model_name = body.get("model") or SERVE.model_name
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = torch.tensor([prompt], dtype = torch.long)
        elif isinstance(prompt, str):
            ids = SERVE.tokenizer.encode(prompt)
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            ids = torch.tensor([prompt[0]], dtype = torch.long)
        else:
            return self._send(400, {"error": {"message": "prompt must be a string or a token-id list"}})
        max_tokens = int(body.get("max_tokens") or 256)
        temperature = float(body.get("temperature") or 0)
        r = SERVE.submit(ids, max_tokens, temperature,
                         skip_special_tokens = body.get("skip_special_tokens", True),
                         want_ids = False)
        self._send(200, {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": r["text"],
                "finish_reason": SERVE.finish_reason(r),
            }],
            "usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            },
        })


def main():
    global SERVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required = True)
    ap.add_argument("--draft", required = True)
    ap.add_argument("--host", default = "127.0.0.1")
    ap.add_argument("--port", type = int, default = 8889)
    ap.add_argument("--model-name", default = "qwen38-27b-exl3-dflash2")
    ap.add_argument("--max-model-len", type = int, default = 262144)
    ap.add_argument("--cache-tokens", type = int, default = 270336)
    ap.add_argument("--cq", type = int, default = 3)
    args = ap.parse_args()

    SERVE = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
