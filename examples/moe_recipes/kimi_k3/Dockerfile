# syntax=docker/dockerfile:1.7
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Kimi-K3 training dependencies for GB200/GB300 (ARM64).
FROM nvcr.io/nvidia/pytorch:26.04-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Keep runtime and headers on the same NCCL release; preserve NGC CUDA/PyTorch.
ARG NCCL_DEB_VERSION=2.32.3-1+cuda13.4
ENV NCCL_VERSION=2.32.3

ARG TE_COMMIT=0a2ebe942b9f6a66aa0292502966e73114a7b912
ARG CUDNN_FRONTEND_COMMIT=5ba9432e652e0b4df3aaca85cf3f95fe6bb7375a
ARG EMERGING_OPTIMIZERS_COMMIT=a44d1f83a4950b445f2a77ca71177c5f033d45d5
# causal-conv1d 1.6.2.post1. gdn_pre_gated_delta_rule_fusion uses its C++
# depthwise-convolution backward; the KDA layer raises ImportError without it.
ARG CAUSAL_CONV1D_COMMIT=4f6ae4e26ae5fe8af9372f8d312ab25cc4595223
# HybridEP from the DeepEP hybrid-ep branch. Revisions before 17cfb81 (#625)
# read a per-call pinned host token count from the permute/unpermute kernels
# in blocking mode and fail intermittently with an illegal memory access on
# long packed sequences (DeepEP issue 756); 10d4dd7 (2026-09-11) no longer
# does and ran the 512K packed-THD recipe cleanly. Place each EP64 group
# within one NVLink domain; this is not a cross-domain RDMA build.
ARG HYBRID_EP_COMMIT=10d4dd7377d5bce900fbb4b80cce863764892b95
# Mixture-of-Kittens (MoK), the deterministic MoE training megakernel for NVL72
# (https://github.com/QiZhangNV/mixture-of-kittens, Kimi-K3 integration branch
# qizhang/mok-routed-only-and-situ-glu). Pinned to the revision that adds routed-only
# experts and the independent SiTU-GLU activation; the ThunderKittens submodule is
# pinned by the MoK checkout and receives MoK's MXFP8 MN-major patch.
ARG MOK_COMMIT=d59188aab3c9d5f92dd197d9997226ba3a43c701
ARG NVTE_CUDA_ARCHS="100a;103a"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates gettext libfabric-dev python3-venv \
        "libnccl2=${NCCL_DEB_VERSION}" "libnccl-dev=${NCCL_DEB_VERSION}" && \
    ldconfig && \
    sed -i "s/^NCCL_VERSION=.*/NCCL_VERSION=${NCCL_VERSION}/" /etc/environment && \
    rm -rf /var/lib/apt/lists/* /opt/megatron-lm

# Fail the build if the linker still resolves an older NCCL runtime.
RUN python -c 'import ctypes; lib = ctypes.CDLL("libnccl.so.2"); version = ctypes.c_int(); assert lib.ncclGetVersion(ctypes.byref(version)) == 0; assert version.value == 23203, version.value'

# Inherit NGC PyTorch/CUDA while keeping pip additions separate from
# OS-managed packages such as python3-yaml.
RUN python -m venv --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        packaging ninja==1.11.1.1 cmake 'pybind11[global]>=2.13,<3' \
        sentencepiece tiktoken einops wandb tqdm datasets tensorboard \
        pydantic pyyaml omegaconf nvtx psutil flask-restful wrapt \
        tensorstore multi-storage-client opentelemetry-api \
        transformers==5.16.1 flash-linear-attention==0.5.2 flash-attn-4==4.0.0b11 \
        nvidia-resiliency-ext==0.6.0 nvidia-mathdx==25.1.1 \
        'nvidia-cutlass-dsl[cu13]==4.5.2' apache-tvm-ffi==0.1.11

RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        --no-build-isolation \
        "nvidia-cudnn-frontend[cutedsl] @ git+https://github.com/NVIDIA/cudnn-frontend.git@${CUDNN_FRONTEND_COMMIT}"

# These recipes use HybridEP. Keep the unused TE NCCL EP extension disabled
# when upgrading the NCCL runtime.
RUN unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" NVTE_BUILD_THREADS_PER_JOB=8 \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 \
    python -m pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/NVIDIA/TransformerEngine.git@${TE_COMMIT}"

# Muon with per-head QKV splitting. Pin the source revision rather than
# the older release used by some published container tags.
RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        --no-build-isolation \
        "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@${EMERGING_OPTIMIZERS_COMMIT}"

# No prebuilt causal-conv1d wheel matches this base, so build from source.
# nvcc 13 adds the sm_100 and sm_103 cubins to the default architecture list.
RUN unset PIP_CONSTRAINT && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    python -m pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@${CAUSAL_CONV1D_COMMIT}"

# The hybrid-ep build links libibverbs / libmlx5 from the NGC base image's source build of
# rdma-core (60, /workspace/rdma-core/build/lib) and records that directory as the extension's
# RUNPATH. When deep_ep is imported before NCCL creates its first communicator (Megatron imports
# it at module level in megatron/core/transformer/moe/fused_a2a.py), that rdma-core is loaded
# first, NCCL's network plugin then loads the system rdma-core (59.1) as well, its devx HCA query
# fails with "No space left on device" and NCCL silently falls back to NET/Socket: every transfer
# that leaves the NVLink domain (pipeline parallelism across racks, cross-rack all-reduces) runs
# over TCP at about 3 GB/s. Dropping the RUNPATH makes the extension resolve the system libraries
# like NCCL does; the HYBRID_EP_MULTINODE=0 build uses no RDMA path of its own.
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && \
    cd /opt/DeepEP && git checkout "${HYBRID_EP_COMMIT}" && \
    unset PIP_CONSTRAINT && \
    HYBRID_EP_MULTINODE=0 TORCH_CUDA_ARCH_LIST="10.0" \
    python -m pip install --no-build-isolation --no-cache-dir . && \
    patchelf --remove-rpath /usr/local/lib/python3.12/dist-packages/hybrid_ep_cpp*.so && \
    ! readelf -d /usr/local/lib/python3.12/dist-packages/hybrid_ep_cpp*.so | grep -q rdma-core

# MoK compiles csrc/bindings.cu in one nvcc invocation; MOK_ARCH=ALL emits sm_100a and
# sm_103a cubins. setup.py requires Python >= 3.12, torch >= 2.10 built for CUDA >= 13.0 and
# an nvcc whose major.minor equals torch.version.cuda (13.2 here; MoK is validated on 13.0 and
# only warns). prepare_thunderkittens.sh initialises the submodule and applies the MN-major
# MXFP8 B-operand patch MoK needs for native TE column-wise weights (idempotent). The import
# check runs from / so the source tree does not shadow the installed package, and with the
# CUDA driver stub because mok._C links libcuda and no driver exists at build time.
RUN git clone https://github.com/QiZhangNV/mixture-of-kittens.git /opt/mixture-of-kittens && \
    cd /opt/mixture-of-kittens && git checkout "${MOK_COMMIT}" && \
    ./scripts/prepare_thunderkittens.sh && \
    unset PIP_CONSTRAINT && \
    MOK_ARCH=ALL python -m pip install --no-build-isolation --no-deps --no-cache-dir . && \
    mkdir -p /tmp/cuda-stubs && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1 && \
    cd / && LD_LIBRARY_PATH=/tmp/cuda-stubs:${LD_LIBRARY_PATH:-} python -c 'import mok, glob, os, subprocess; assert not mok.__file__.startswith("/opt/mixture-of-kittens"), mok.__file__; so = glob.glob(os.path.join(os.path.dirname(mok.__file__), "_C*.so"))[0]; elf = subprocess.check_output(["cuobjdump", "--list-elf", so], text=True); assert "sm_100a" in elf and "sm_103a" in elf, elf; print("mok", mok.__version__, os.path.basename(so))' && \
    rm -rf /tmp/cuda-stubs

# NGC's triton-kernels declares pytest as a dependency.
# Preserve dependency consistency and the MLA latent CP split-API version pin.
RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir pytest && \
    python -m pip check && \
    python -c 'from importlib.metadata import version; assert version("flash-attn-4") == "4.0.0b11"' && \
    rm -rf /root/.cache/pip
WORKDIR /workspace/Megatron-LM
