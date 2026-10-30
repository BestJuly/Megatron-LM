# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer spec for the Kimi-K3 vision tower (MoonViT).

One MoonViT layer is ``x + wo(attn(rope(wqkv(norm0(x)))))`` followed by
``x + fc1(gelu_tanh(fc0(norm1(x))))``. That is MCore's TE ViT layer: the
pre-attention and pre-MLP RMSNorms are fused into
``TELayerNormColumnParallelLinear`` (``linear_qkv`` / ``linear_fc1``) and the
attention mask is ``no_mask`` over THD sequences.

The self-attention is Qwen3.5-VL's vision attention, reused as-is: it applies
RoPE in fp32 (MoonViT casts q/k to float32 before its complex multiply) and
pins the vision RoPE to a size-1 CP group, because the tower never runs under
context parallelism even when the decoder does. With
``rotary_interleaved=True`` on the vision config, MCore rotates the adjacent
channel pairs MoonViT rotates.
"""

from examples.multimodal_dev.models.qwen35_vl.specs import Qwen35VLVisionSelfAttention
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def get_kimi_k3_vl_vision_spec() -> ModuleSpec:
    """``ModuleSpec`` for one MoonViT layer."""
    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Qwen35VLVisionSelfAttention
    return spec


def get_kimi_k3_vl_vision_reference_spec() -> ModuleSpec:
    """Numerics-reference MoonViT layer: PyTorch linears/norms + TE THD attention.

    Not for training. Transformer Engine runs every pure-fp32 GEMM in TF32
    (``CUBLAS_COMPUTE_32F_FAST_TF32``, independent of
    ``torch.backends.cuda.matmul.allow_tf32``), which bounds an fp32 comparison
    of the TE spec against HF at ~4e-4 relative per layer. This spec computes the
    same layer with ``ColumnParallelLinear`` / ``RowParallelLinear`` and
    ``torch.nn.RMSNorm`` (which honour TF32-off), keeping TE only for the THD
    core attention, whose fp32 path is PyTorch math. Parameter names differ
    from the TE spec only in the norms: ``input_layernorm.weight`` and
    ``pre_mlp_layernorm.weight`` instead of the fused ``layer_norm_weight``.
    """
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=WrappedTorchNorm,
            self_attention=ModuleSpec(
                module=Qwen35VLVisionSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=WrappedTorchNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
