# Runtime image

`Dockerfile` ships native ExLlamaV3 `community` (no `dflash2-pathway` overlay).

Build requires, in the repo root:

1. `git clone -b community https://github.com/r0b0tlab/exllamav3 exllamav3` pinned at `355c6ee`
2. `container/exllamav3_ext.cpython-313-x86_64-linux-gnu.so` — 3090/sm_86 extension
   (CUDA 13, stripped). Gitignored; produced on this host from the community tree.

```bash
docker build -t qwen38-exl3-dflash2:1.5.0-native -f container/Dockerfile .
```

The click-run tag is `ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0-native`.
