# syntax=docker/dockerfile:1
# Native ExLlamaV3 community runtime for Qwen3.8-27B EXL3 + DFlash2 on RTX 3090.
# No dflash2-pathway overlay and no v1.5.0 wheel.
# The extension is compiled in-image for sm_86 against Ubuntu 24.04 glibc
# (the host .so needs GLIBC_2.43 and will not load here).
#
# Engine: r0b0tlab/exllamav3 community @ 355c6ee
#
# Build from repo root:
#   git clone -b community https://github.com/r0b0tlab/exllamav3 exllamav3
#   docker build -t qwen38-exl3-dflash2:1.5.0-native -f container/Dockerfile .
#
# Click-run:
#   docker run --gpus all -v qwen38-models:/models \
#     ghcr.io/r0b0tlab/qwen38-exl3-dflash2:1.5.0-native

FROM ghcr.io/astral-sh/uv:0.9.15 AS uv

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    MAX_JOBS=8 \
    UV_HTTP_TIMEOUT=300 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates g++ ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

RUN uv venv --python 3.13 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.10.0" \
    && uv pip install --python /opt/venv/bin/python \
        setuptools wheel ninja \
        tokenizers numpy rich typing_extensions safetensors \
        pillow pyyaml marisa-trie pydantic llguidance triton \
        blessed prompt_toolkit pyperclip requests \
        huggingface_hub hf_transfer
RUN uv pip install --python /opt/venv/bin/python transformers

COPY exllamav3 /src/exllamav3
WORKDIR /src/exllamav3
RUN uv pip install --python /opt/venv/bin/python --no-build-isolation --no-deps .

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/uv-python /opt/uv-python
COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/exllamav3/examples /opt/exllamav3/examples
COPY container/entrypoint.sh /opt/entrypoint.sh
COPY container/serve_openai.py /opt/serve_openai.py
RUN chmod +x /opt/entrypoint.sh /opt/serve_openai.py

RUN useradd --create-home --uid 10001 runner \
    && mkdir -p /models && chown runner:runner /models
USER runner

ENV PATH="/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}" \
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
    MODEL_DIR=/models/qwen38-27b-exl3 \
    DRAFT_DIR=/models/dflash2-exl3 \
    MODE=chatml \
    CTX=262144 \
    CQ=3 \
    HF_HOME=/models/.hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN python -V && python -c "import torch, exllamav3; from exllamav3.ext import exllamav3_ext as e; print('smoke ok:', torch.__version__, 'shapes', e.exl3_gemm_num_kernel_shapes())"

ENTRYPOINT ["/opt/entrypoint.sh"]
