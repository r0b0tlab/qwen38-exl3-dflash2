# syntax=docker/dockerfile:1
# Native ExLlamaV3 community runtime for Qwen3.8-27B EXL3 + DFlash2 on a 24 GB
# RTX 3090. This image does NOT overlay the old dflash2-pathway Python tree on
# the official v1.5.0 wheel. It ships:
#   - r0b0tlab/exllamav3 `community` @ 355c6ee (upstream native DFlash2 CUDA,
#     GDN rewind, dense M=32/64 TILEBLOCKS_M, codebook .weight alias)
#   - a CUDA 13 / sm_86 extension (stripped), built on the 3090
#
# Build (from repo root; needs the engine clone and the 3090 .so):
#   git clone -b community https://github.com/r0b0tlab/exllamav3 exllamav3
#   docker build -t qwen38-exl3-dflash2:1.5.0-native -f container/Dockerfile .
#
# Click-run:
#   docker run --gpus all -v qwen38-models:/models \
#     ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0-native

FROM ghcr.io/astral-sh/uv:0.9.15 AS uv

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /opt

# Torch cu130 matches the community extension (libcudart.so.13). The managed
# CPython lives under /opt so the non-root runtime user can traverse it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed
RUN uv venv --python 3.13 /opt/venv \
    && for i in 1 2 3 4; do \
        uv pip install --python /opt/venv/bin/python \
            --index-url https://download.pytorch.org/whl/cu130 \
            "torch==2.10.0" \
        && uv pip install --python /opt/venv/bin/python \
            tokenizers numpy rich typing_extensions safetensors \
            pillow pyyaml marisa-trie pydantic llguidance triton \
            blessed prompt_toolkit pyperclip requests \
        && break || { echo "install attempt $i failed; retrying"; sleep 15; }; \
    done \
    && uv cache clean

RUN uv pip install --python /opt/venv/bin/python huggingface_hub hf_transfer \
    && uv cache clean

# Native package + 3090 extension. No v1.5.0 wheel, no Python overlay.
COPY exllamav3/exllamav3/ /opt/venv/lib/python3.13/site-packages/exllamav3/
COPY container/exllamav3_ext.cpython-313-x86_64-linux-gnu.so \
     /opt/venv/lib/python3.13/site-packages/exllamav3_ext.cpython-313-x86_64-linux-gnu.so
COPY exllamav3/examples/ /opt/exllamav3/examples/
COPY container/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

RUN useradd --create-home --uid 10001 runner \
    && mkdir -p /models && chown runner:runner /models
USER runner

ENV PATH="/opt/venv/bin:${PATH}" \
    MODEL_DIR=/models/qwen38-27b-exl3 \
    DRAFT_DIR=/models/dflash2-exl3 \
    MODE=chatml \
    CTX=262144 \
    CQ=3 \
    HF_HOME=/models/.hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN python -V && python -c "import torch, exllamav3; from exllamav3.ext import exllamav3_ext as e; print('smoke ok:', torch.__version__, 'shapes', e.exl3_gemm_num_kernel_shapes())"

ENTRYPOINT ["/opt/entrypoint.sh"]
