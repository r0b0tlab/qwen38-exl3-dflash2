# Runtime image

`Dockerfile` ships native ExLlamaV3 `community` (no `dflash2-pathway` overlay).

The CUDA extension is compiled in-image for sm_86 against Ubuntu 24.04 glibc.
Do not copy a host-built `.so` — this host's extension needs GLIBC_2.43.

Build from the repo root after:

```bash
git clone -b community https://github.com/r0b0tlab/exllamav3 exllamav3
# pin 355c6ee
docker build -t qwen38-exl3-dflash2:1.5.0-native -f container/Dockerfile .
```

Click-run tag: `ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0-native`.
