# Runtime image

`Dockerfile` ships native ExLlamaV3 `community` @ 355c6ee. The CUDA extension is
compiled in-image for sm_86 against Ubuntu 24.04 glibc.

Torch's `nvidia/cu13` libs must precede `/usr/local/cuda` on `LD_LIBRARY_PATH`
(baked in). Triton needs `TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu` and a
host bind of `libcuda.so.1`. Runtime includes gcc + libc6-dev so Triton's driver
stub can compile. `transformers` is required for chat-template render.

```bash
git clone -b community https://github.com/r0b0tlab/exllamav3 exllamav3
docker build -t qwen38-exl3-dflash2:1.5.0-native -f container/Dockerfile .
```

This host has no nvidia-container-toolkit. GPU serve:

```bash
bash container/run-serve.sh
# OpenAI endpoint at http://127.0.0.1:8889
```

Click-run tag: `ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0-native`.
