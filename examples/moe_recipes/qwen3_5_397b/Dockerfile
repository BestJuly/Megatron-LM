# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# IMAGE_NAME: qwen3_5-gb300
#
# Container for the Qwen3.5-397B-A17B and Qwen3.8-2.4T-A95B GB300 MoK recipes
# (qwen3_5_397b/ and qwen3_8_2_4t_a95b/). It pins the component
# revisions recorded in the image that produced the published numbers (see
# README.md in this directory). GB300 / aarch64 only: MoK is built for SM103.

FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

ARG TE_COMMIT="5f6105b900778068144f1b24e25a2e84066983cb"
ARG FLA_COMMIT="6b6f546f9bd10a97be1e8cf2f0120de70cfca482"
ARG DEEPEP_COMMIT="58397eac493b075b0baea5861ac6e9793d8d7a21"
ARG MOK_COMMIT="05d7db7d74f77b6396f98df886f1f9bcda97ec51"
ARG CUDNN_VERSION="9.25.1.1"
ARG CUDNN_FRONTEND_VERSION="1.28.0"
ARG CUTLASS_DSL_VERSION="4.7.0"
ARG FLASH_ATTN_4_VERSION="4.0.0b29"
ARG CAUSAL_CONV1D_VERSION="1.7.0"

RUN bash -euxo pipefail <<"EOF"
    test "$(dpkg --print-architecture)" = "arm64"
    rm -rf /opt/megatron-lm
    apt-get -o Acquire::Retries=5 update
    apt-get install -y --no-install-recommends \
        ca-certificates git curl wget libfabric-dev
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
    einops sentencepiece tokenizers transformers==4.57.1 datasets tqdm pydantic \
    tiktoken nltk wrapt wandb && \
    python -m pip install --no-cache-dir --upgrade --no-deps "nvidia-resiliency-ext==0.6.0"

# cuDNN 9.25.1 with cuDNN frontend 1.28.0 (cudnn.fla, used by --gdn-gdr-backend cudnn),
# FA4 and the CUTLASS DSL they share.
RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir --upgrade \
        "nvidia-cudnn-cu13==${CUDNN_VERSION}" \
        "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" \
        "nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION}" \
        "flash-attn-4[cu13]==${FLASH_ATTN_4_VERSION}"

# TransformerEngine main for SM100a and SM103a. HybridEP handles expert-parallel
# communication, so TE's NCCL-EP backend is disabled.
RUN unset PIP_CONSTRAINT && \
    python -m pip uninstall -y \
        transformer-engine transformer-engine-torch transformer-engine-cu13 && \
    NVTE_CUDA_ARCHS="100a;103a" \
    NVTE_BUILD_THREADS_PER_JOB=8 \
    NVTE_FRAMEWORK=pytorch \
    NVTE_WITH_NCCL_EP=0 \
    python -m pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/NVIDIA/TransformerEngine.git@${TE_COMMIT}"

# Flash Linear Attention (GDN kernels) and causal-conv1d (required by
# --gdn-pre-gated-delta-rule-fusion). --no-deps keeps the base image's torch/triton.
RUN unset PIP_CONSTRAINT && \
    python -m pip uninstall -y fla-core flash-linear-attention && \
    python -m pip install --no-build-isolation --no-cache-dir --no-deps \
        "git+https://github.com/fla-org/flash-linear-attention.git@${FLA_COMMIT}" && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="10.0;10.3" MAX_JOBS=8 \
    python -m pip install --no-build-isolation --no-cache-dir \
        "causal-conv1d==${CAUSAL_CONV1D_VERSION}"

# HybridEP (DeepEP hybrid-ep branch). The EP64 groups of these recipes stay inside
# one NVLink domain, so the multinode RDMA path is not built.
WORKDIR /workspace
RUN bash -euxo pipefail <<"EOF"
    git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git
    cd DeepEP
    git checkout "${DEEPEP_COMMIT}"
    HYBRID_EP_MULTINODE=0 USE_NIXL=0 \
        TORCH_CUDA_ARCH_LIST="10.0;10.3" MAX_JOBS=8 \
        python -m pip install --no-build-isolation .
    rm -rf /root/.cache /tmp/*
EOF

# Mixture-of-Kittens MoE megakernel (--moe-megakernel-backend mok). The pinned
# commit includes the shared-expert output gate. prepare_thunderkittens.sh
# initializes ThunderKittens and applies the MXFP8 MN-major patch shipped in the
# MoK repository. MoK is built in place for SM103 (GB300) and put on sys.path
# with a .pth file, as in the benchmark image.
ENV MOK_HOME=/opt/mixture-of-kittens
RUN bash -euxo pipefail <<"EOF"
    git clone https://github.com/QiZhangNV/mixture-of-kittens.git "${MOK_HOME}"
    cd "${MOK_HOME}"
    git checkout "${MOK_COMMIT}"
    bash scripts/prepare_thunderkittens.sh
    make -B ARCH=SM103
    ls "${MOK_HOME}"/mok/_C*.so
    echo "${MOK_HOME}" > /usr/local/lib/python3.12/dist-packages/mixture_of_kittens.pth
EOF

RUN python - <<'PY'
from importlib import metadata

for package in (
    "torch",
    "transformer-engine",
    "flash-linear-attention",
    "causal-conv1d",
    "nvidia-cudnn-cu13",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "flash-attn-4",
    "deep-ep",
    "transformers",
):
    print(f"{package}=={metadata.version(package)}")

import cudnn
import deep_ep
import fla
import mok
import transformer_engine.pytorch
PY

WORKDIR /workspace
