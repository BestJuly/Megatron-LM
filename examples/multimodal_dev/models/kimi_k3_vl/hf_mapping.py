# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HF <-> MCore parameter mapping for the Kimi-K3 vision tower and projector.

HF keys are relative to ``KimiK3ForConditionalGeneration`` (``vision_tower.*``
and ``mm_projector.*``); MCore keys are relative to
:class:`~examples.multimodal_dev.models.kimi_k3_vl.vision_encoder.KimiK3VLVisionEncoder`.

The only non-identity transform is the fused QKV weight. HF's ``wqkv`` output
is ``[3, heads, head_dim]`` (all queries, then all keys, then all values);
MCore's ``linear_qkv`` groups rows per query group, ``[groups, q|k|v, head_dim]``,
and MoonViT is MHA (one head per group).
"""

import re
from typing import Dict

import torch

_LAYER = re.compile(r"^vision_tower\.encoder\.blocks\.(\d+)\.(.+)$")

#: Per-layer HF suffix -> MCore suffix (under ``decoder.layers.<i>.``).
_LAYER_KEYS = {
    "norm0.weight": "self_attention.linear_qkv.layer_norm_weight",
    "wqkv.weight": "self_attention.linear_qkv.weight",
    "wo.weight": "self_attention.linear_proj.weight",
    "norm1.weight": "mlp.linear_fc1.layer_norm_weight",
    "mlp.fc0.weight": "mlp.linear_fc1.weight",
    "mlp.fc1.weight": "mlp.linear_fc2.weight",
}

#: Non-layer HF key -> MCore key.
_GLOBAL_KEYS = {
    "vision_tower.patch_embed.proj.weight": "patch_embed.proj.weight",
    "vision_tower.patch_embed.pos_emb.weight": "pos_emb.weight",
    "vision_tower.encoder.final_layernorm.weight": "decoder.final_layernorm.weight",
    "mm_projector.proj.0.weight": "projector.linear_fc1.weight",
    "mm_projector.proj.2.weight": "projector.linear_fc2.weight",
    "mm_projector.post_norm.weight": "projector.post_norm.weight",
}
_GLOBAL_KEYS_INV = {v: k for k, v in _GLOBAL_KEYS.items()}
_LAYER_KEYS_INV = {v: k for k, v in _LAYER_KEYS.items()}


def qkv_hf_to_mcore(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``[3*H*d, in]`` (q|k|v blocks) -> ``[H*3*d, in]`` (per-head q,k,v)."""
    in_dim = weight.shape[-1]
    head_dim = weight.shape[0] // (3 * num_heads)
    return weight.view(3, num_heads, head_dim, in_dim).transpose(0, 1).reshape(-1, in_dim)


def qkv_mcore_to_hf(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Inverse of :func:`qkv_hf_to_mcore`."""
    in_dim = weight.shape[-1]
    head_dim = weight.shape[0] // (3 * num_heads)
    return weight.view(num_heads, 3, head_dim, in_dim).transpose(0, 1).reshape(-1, in_dim)


def is_hf_vision_key(key: str) -> bool:
    """Whether *key* belongs to the vision tower or the projector."""
    return key.startswith("vision_tower.") or key.startswith("mm_projector.")


#: Norm key names of ``specs.get_kimi_k3_vl_vision_reference_spec`` (separate
#: PyTorch norms) in place of the TE spec's fused ``layer_norm_weight``.
_UNFUSED_NORM_KEYS = {
    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
    "mlp.linear_fc1.layer_norm_weight": "pre_mlp_layernorm.weight",
}


def hf_to_mcore_vision(
    hf_state: Dict[str, torch.Tensor], num_heads: int, fused_norms: bool = True
) -> Dict[str, torch.Tensor]:
    """Map every ``vision_tower.*`` / ``mm_projector.*`` tensor of *hf_state*.

    ``fused_norms=False`` targets the reference spec's separate norm modules.
    Raises ``KeyError`` on a vision key this mapping does not know, so a new HF
    parameter can never be dropped silently.
    """
    out = {}
    for key, value in hf_state.items():
        if not is_hf_vision_key(key):
            continue
        if key in _GLOBAL_KEYS:
            out[_GLOBAL_KEYS[key]] = value
            continue
        match = _LAYER.match(key)
        if match is None or match.group(2) not in _LAYER_KEYS:
            raise KeyError(f"unmapped Kimi-K3 vision key: {key}")
        layer, suffix = match.groups()
        if suffix == "wqkv.weight":
            value = qkv_hf_to_mcore(value, num_heads)
        target = _LAYER_KEYS[suffix]
        if not fused_norms:
            target = _UNFUSED_NORM_KEYS.get(target, target)
        out[f"decoder.layers.{layer}.{target}"] = value
    return out


def mcore_to_hf_vision(
    mcore_state: Dict[str, torch.Tensor], num_heads: int
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`hf_to_mcore_vision` over encoder-relative MCore keys.

    TE's ``_extra_state`` entries carry no weights and are skipped.
    """
    out = {}
    layer_re = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")
    for key, value in mcore_state.items():
        if key.endswith("_extra_state"):
            continue
        if key in _GLOBAL_KEYS_INV:
            out[_GLOBAL_KEYS_INV[key]] = value
            continue
        match = layer_re.match(key)
        if match is None or match.group(2) not in _LAYER_KEYS_INV:
            raise KeyError(f"unmapped Kimi-K3 vision MCore key: {key}")
        layer, suffix = match.groups()
        if suffix == "self_attention.linear_qkv.weight":
            value = qkv_mcore_to_hf(value, num_heads)
        out[f"vision_tower.encoder.blocks.{layer}.{_LAYER_KEYS_INV[suffix]}"] = value
    return out


# ---------------------------------------------------------------------------
# Full model: Kimi-K3 text decoder (KimiLinear) + vision, HF <-> MCore
# ---------------------------------------------------------------------------
#
# HF decoder layer ``i`` is one Kimi block: attention (KDA or MLA) then MLP
# (dense for ``i < first_k_dense_replace``, latent MoE otherwise), each with an
# attention-residual (AttnRes) read. MCore's HybridModel spells the same block
# as two pattern entries: ``2i`` (``K`` / ``+``) and ``2i+1`` (``-`` / ``E``),
# each wrapping its sublayer as ``inner_layer`` next to its ``attn_res``.
#
# HF AttnRes scores a source ``v`` as ``rmsnorm(v) . (norm.weight * proj.weight)``;
# MCore stores the two factors as ``attn_res.key_norm_weight`` and
# ``attn_res.pseudo_query``.
#
# Fused tensors (MCore concatenates rows, HF keeps them separate):
#   KDA  in_proj  = q_proj | k_proj | v_proj        conv1d = q_conv | k_conv | v_conv
#   MLP / experts / shared experts  linear_fc1 = gate (w1) | up (w3)

_TEXT = "language_model.model"
_HF_LAYER = re.compile(rf"^{re.escape(_TEXT)}\.layers\.(\d+)\.(.+)$")

_KDA_IDENTITY = {
    "self_attn.A_log": "A_log",
    "self_attn.dt_bias": "dt_bias",
    "self_attn.f_a_proj.weight": "f_a_proj.weight",
    "self_attn.f_b_proj.weight": "f_b_proj.weight",
    "self_attn.b_proj.weight": "beta_proj.weight",
    "self_attn.g_proj.weight": "g_proj.weight",
    "self_attn.o_norm.weight": "out_norm.weight",
    "self_attn.o_proj.weight": "out_proj.weight",
}
_KDA_FUSED = {
    "in_proj.weight": (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ),
    "conv1d.weight": (
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
    ),
}
_MLA_IDENTITY = {
    "self_attn.q_a_proj.weight": "linear_q_down_proj.weight",
    "self_attn.q_a_layernorm.weight": "linear_q_up_proj.layer_norm_weight",
    "self_attn.q_b_proj.weight": "linear_q_up_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "linear_kv_down_proj.weight",
    "self_attn.kv_a_layernorm.weight": "linear_kv_up_proj.layer_norm_weight",
    "self_attn.kv_b_proj.weight": "linear_kv_up_proj.weight",
    "self_attn.o_proj.weight": "linear_proj.weight",
    "self_attn.g_proj.weight": "linear_gate.weight",
}
_MOE_IDENTITY = {
    "block_sparse_moe.gate.weight": "router.weight",
    "block_sparse_moe.gate.e_score_correction_bias": "router.expert_bias",
    "block_sparse_moe.routed_expert_down_proj.weight": "fc1_latent_proj.weight",
    "block_sparse_moe.routed_expert_up_proj.weight": "fc2_latent_proj.weight",
    "block_sparse_moe.routed_expert_norm.weight": "fc2_latent_proj.layer_norm_weight",
    "block_sparse_moe.shared_experts.down_proj.weight": "shared_experts.linear_fc2.weight",
}
_EXPERT = re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$")


def build_plan(hf_keys, vision_num_heads: int = 12):
    """``[(mcore_key, op, hf_keys)]`` for a full Kimi-K3 HF key set.

    ``op`` is ``"id"``, ``"cat"`` (row concat), ``"row0"`` (``[1, N] -> [N]``)
    or ``"qkv"`` (MoonViT per-head reorder). Layer kinds are read from the keys
    present (KDA vs MLA, dense vs MoE), so no config is needed. Raises
    ``KeyError`` when an HF key is left unplanned.
    """
    hf_keys = set(hf_keys)
    plan = []
    used = set()

    def add(mcore_key, op, *srcs):
        missing = [k for k in srcs if k not in hf_keys]
        if missing:
            raise KeyError(f"Kimi-K3 HF keys missing for {mcore_key}: {missing}")
        used.update(srcs)
        plan.append((mcore_key, op, srcs))

    for key in sorted(k for k in hf_keys if is_hf_vision_key(k)):
        if key in _GLOBAL_KEYS:
            add(f"vision_model.{_GLOBAL_KEYS[key]}", "id", key)
            continue
        match = _LAYER.match(key)
        if match is None or match.group(2) not in _LAYER_KEYS:
            raise KeyError(f"unmapped Kimi-K3 vision key: {key}")
        layer, suffix = match.groups()
        op = "qkv" if suffix == "wqkv.weight" else "id"
        add(f"vision_model.decoder.layers.{layer}.{_LAYER_KEYS[suffix]}", op, key)

    lm = "language_model"
    add(f"{lm}.embedding.word_embeddings.weight", "id", f"{_TEXT}.embed_tokens.weight")
    add(f"{lm}.decoder.final_norm.weight", "id", f"{_TEXT}.norm.weight")
    add(f"{lm}.output_layer.weight", "id", "language_model.lm_head.weight")
    add(
        f"{lm}.decoder.final_attn_res.key_norm_weight", "id", f"{_TEXT}.output_attn_res_norm.weight"
    )
    add(f"{lm}.decoder.final_attn_res.pseudo_query", "row0", f"{_TEXT}.output_attn_res_proj.weight")

    layers = {}
    for key in hf_keys:
        match = _HF_LAYER.match(key)
        if match:
            layers.setdefault(int(match.group(1)), set()).add(match.group(2))
    for i, suffixes in sorted(layers.items()):
        hf = f"{_TEXT}.layers.{i}"
        attn = f"{lm}.decoder.layers.{2 * i}"
        mlp = f"{lm}.decoder.layers.{2 * i + 1}"
        add(f"{attn}.attn_res.key_norm_weight", "id", f"{hf}.self_attention_res_norm.weight")
        add(f"{attn}.attn_res.pseudo_query", "row0", f"{hf}.self_attention_res_proj.weight")
        add(f"{mlp}.attn_res.key_norm_weight", "id", f"{hf}.mlp_res_norm.weight")
        add(f"{mlp}.attn_res.pseudo_query", "row0", f"{hf}.mlp_res_proj.weight")
        add(f"{attn}.inner_layer.input_layernorm.weight", "id", f"{hf}.input_layernorm.weight")
        sa = f"{attn}.inner_layer.self_attention"
        if "self_attn.A_log" in suffixes:  # KDA
            for src, dst in _KDA_IDENTITY.items():
                add(f"{sa}.{dst}", "id", f"{hf}.{src}")
            for dst, srcs in _KDA_FUSED.items():
                add(f"{sa}.{dst}", "cat", *(f"{hf}.{x}" for x in srcs))
        else:  # MLA
            for src, dst in _MLA_IDENTITY.items():
                add(f"{sa}.{dst}", "id", f"{hf}.{src}")
        inner = f"{mlp}.inner_layer"
        if "block_sparse_moe.gate.weight" in suffixes:
            add(f"{inner}.pre_mlp_layernorm.weight", "id", f"{hf}.post_attention_layernorm.weight")
            for src, dst in _MOE_IDENTITY.items():
                if src in suffixes:
                    add(f"{inner}.mlp.{dst}", "id", f"{hf}.{src}")
            add(
                f"{inner}.mlp.shared_experts.linear_fc1.weight",
                "cat",
                f"{hf}.block_sparse_moe.shared_experts.gate_proj.weight",
                f"{hf}.block_sparse_moe.shared_experts.up_proj.weight",
            )
            experts = sorted({int(m.group(1)) for x in suffixes for m in [_EXPERT.match(x)] if m})
            for e in experts:
                pre = f"{hf}.block_sparse_moe.experts.{e}"
                add(
                    f"{inner}.mlp.experts.linear_fc1.weight{e}",
                    "cat",
                    f"{pre}.w1.weight",
                    f"{pre}.w3.weight",
                )
                add(f"{inner}.mlp.experts.linear_fc2.weight{e}", "id", f"{pre}.w2.weight")
        else:  # dense MLP: TE fuses its pre-norm into linear_fc1
            add(
                f"{inner}.mlp.linear_fc1.layer_norm_weight",
                "id",
                f"{hf}.post_attention_layernorm.weight",
            )
            add(
                f"{inner}.mlp.linear_fc1.weight",
                "cat",
                f"{hf}.mlp.gate_proj.weight",
                f"{hf}.mlp.up_proj.weight",
            )
            add(f"{inner}.mlp.linear_fc2.weight", "id", f"{hf}.mlp.down_proj.weight")

    unused = hf_keys - used
    if unused:
        raise KeyError(f"unmapped Kimi-K3 HF keys: {sorted(unused)[:10]}")
    return plan


def hf_to_mcore(
    hf_state: Dict[str, torch.Tensor], vision_num_heads: int = 12
) -> Dict[str, torch.Tensor]:
    """Map a full ``KimiK3ForConditionalGeneration`` state dict to ``KimiK3VLModel``.

    ``e_score_correction_bias`` maps to the router's ``expert_bias``; drop it
    from the result when the MCore model is built without
    ``--moe-router-enable-expert-bias``.
    """
    out = {}
    for mcore_key, op, srcs in build_plan(hf_state, vision_num_heads):
        tensors = [hf_state[k] for k in srcs]
        if op == "id":
            out[mcore_key] = tensors[0]
        elif op == "cat":
            out[mcore_key] = torch.cat(tensors, dim=0)
        elif op == "row0":
            out[mcore_key] = tensors[0][0]
        else:
            out[mcore_key] = qkv_hf_to_mcore(tensors[0], vision_num_heads)
    return out


def mcore_to_hf(
    mcore_state: Dict[str, torch.Tensor], hf_shapes: Dict[str, tuple], vision_num_heads: int = 12
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`hf_to_mcore`; *hf_shapes* gives the HF key set and split sizes."""
    out = {}
    for mcore_key, op, srcs in build_plan(hf_shapes, vision_num_heads):
        if mcore_key not in mcore_state:
            continue  # e.g. router.expert_bias on a model built without it
        value = mcore_state[mcore_key]
        if op == "id":
            out[srcs[0]] = value
        elif op == "cat":
            sizes = [int(hf_shapes[k][0]) for k in srcs]
            for key, part in zip(srcs, torch.split(value, sizes, dim=0)):
                out[key] = part
        elif op == "row0":
            out[srcs[0]] = value.unsqueeze(0)
        else:
            out[srcs[0]] = qkv_mcore_to_hf(value, vision_num_heads)
    return out
