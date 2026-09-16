# syntax=docker/dockerfile:1
# Optimized runtime for Qwen3.8-27B EXL3 + DFlash2 speculative decoding on a single
# 24 GB Ampere-class GPU (developed/validated on RTX 3090).
#
# Build:
#   docker build -t qwen38-exl3-dflash2:1.5.0 -f container/Dockerfile .
#
# Click-run (models auto-download to the named volume on first start):
#   docker run --gpus all -v qwen38-models:/models \
#     ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0
# Run with local models (chat.py is an interactive console app):
#   docker run -it --rm --gpus all -v /path/to/models:/models:ro qwen38-exl3-dflash2:1.5.0
# One-shot:
#   docker run --rm --gpus all -v /path/to/models:/models:ro \
#     -e PROMPT="Explain photosynthesis in one sentence." -e EXTRA_ARGS="-basic -tps" \
#     qwen38-exl3-dflash2:1.5.0
#
# The image ships the dflash2-pathway engine build. It is runtime-only: no compiler
# toolchain, no CUDA toolkit (~2.6 GB smaller than a -devel base).

FROM ghcr.io/astral-sh/uv:0.9.15 AS uv

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Minimal runtime deps: ca-certificates for HTTPS, libgomp for OpenMP kernels.
# No build-essential, no nvcc: the exllamav3 extension comes from the prebuilt wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /opt

# Pinned runtime environment. The explicit cu128 index keeps the torch build matched to
# the base image CUDA 12.8 and to the exllamav3 wheel tag (--torch-backend=auto would
# pick cu130 on hosts with a newer toolkit, which libcudart-mismatches the wheel).
# The managed CPython must live outside /root: uv's default install dir is $HOME/.local
# (i.e. /root/.local as build root, mode 700), and the non-root runtime user cannot
# traverse it, leaving /opt/venv/bin/python a dangling symlink.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed
RUN uv venv --python 3.13 /opt/venv \
    && for i in 1 2 3 4; do \
        uv pip install --python /opt/venv/bin/python \
            --index-url https://download.pytorch.org/whl/cu128 \
            "torch==2.10.0" \
        && uv pip install --python /opt/venv/bin/python \
            "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.0/exllamav3-1.5.0+cu128.torch2.10.0-cp313-cp313-linux_x86_64.whl" \
            safetensors rich blessed prompt_toolkit pyperclip requests pydantic \
        && break || { echo "install attempt $i failed; retrying"; sleep 15; }; \
    done \
    && uv cache clean

# Click-run model download support (own layer, so the large install above stays cached).
RUN uv pip install --python /opt/venv/bin/python huggingface_hub hf_transfer \
    && uv cache clean

# The wheel provides the compiled CUDA extension; the DFlash2 pathway source overlays
# onto it (additive changes only — the ext .so and everything else from the wheel is
# preserved). chat.py ships outside the wheel and runs from /opt/exllamav3/examples.
COPY exllamav3/exllamav3/ /opt/venv/lib/python3.13/site-packages/exllamav3/
COPY exllamav3/examples/ /opt/exllamav3/examples/
COPY container/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Non-root runtime user; /models is where a named volume lands, so downloads persist.
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

# Build-time smoke as the runtime user (PATH now includes the venv): catches venv
# paths that only resolve for root.
RUN python -V && python -c "import torch, exllamav3; print('smoke ok:', torch.__version__)"

ENTRYPOINT ["/opt/entrypoint.sh"]
