# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical parity of the MCore Kimi-K3 vision tower against HF MoonViT.

Reference: ``MoonViT3dPretrainedModel`` + ``PatchMergerMLPV2`` from the
``moonshotai/Kimi-K3`` remote code at the pinned revision (see
``tests/_kimi_hf.py``), built from a random-init config with the real widths
and a shallow tower. Weights go HF -> MCore through
``models/kimi_k3_vl/hf_mapping.py``, the same mapping the checkpoint converter
uses.

Checks, in fp32 with TF32 off, on one packed batch that mixes a native-size
image (no position-table interpolation), two interpolated images and two
video items (t = 2 and t = 4):

* encoder output (post-projector) relative error <= 1e-5, overall and per item;
* the pre-projector tower output (post final norm) as well, so a projector bug
  and a tower bug are told apart;
* gradients of every vision parameter, for the same output cotangent, within
  1e-4 relative.

These strict bounds apply to the encoder built with
``specs.get_kimi_k3_vl_vision_reference_spec`` (PyTorch linears and norms, TE
THD attention). The production TE spec is checked on the same batch with
TF32-level bounds (2e-3 features, 1e-2 gradients): Transformer Engine runs
pure-fp32 GEMMs in TF32 regardless of ``allow_tf32``, measured at ~4e-4 per
layer. Both specs share every MoonViT-specific module (patch embed, position
table, 2-D RoPE, tpool merge, projector), so the strict run covers their
logic and the TE run covers the production wiring.

HF runs its ``eager`` attention, MCore its THD attention through TE. Run with::

    torchrun --nproc_per_node=1 -m pytest -q \\
        examples/multimodal_dev/tests/test_kimi_k3_vl_vision_parity.py
"""

import os

import pytest
import torch

from examples.multimodal_dev.models.kimi_k3_vl.configuration import (
    get_kimi_k3_vl_vision_config,
    vision_norm_epsilon,
)
from examples.multimodal_dev.models.kimi_k3_vl.hf_mapping import hf_to_mcore_vision
from examples.multimodal_dev.models.kimi_k3_vl.specs import get_kimi_k3_vl_vision_reference_spec
from examples.multimodal_dev.models.kimi_k3_vl.vision_encoder import (
    KimiK3VLVisionEncoder,
    tpool_patch_merge,
)
from examples.multimodal_dev.tests._kimi_hf import import_hf_module
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

NUM_LAYERS = 2
OUT_HIDDEN = 7168
# (t, h, w): native 64x64 table, two interpolated images, two video items.
GRIDS = [(1, 64, 64), (1, 28, 42), (1, 46, 30), (2, 16, 20), (4, 8, 12)]
FEATURE_RTOL = 1e-5
GRAD_RTOL = 1e-4
# Transformer Engine GEMMs in fp32 run as TF32 (see module docstring).
TE_FEATURE_RTOL = 2e-3
TE_GRAD_RTOL = 1e-2


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.fixture(scope="module")
def device():
    """TP=1 model-parallel state, TF32 off for the fp32 comparison."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    Utils.destroy_model_parallel()


def build_hf_vision(device):
    """Random-init HF MoonViT + projector (fp32, eager attention)."""
    cfg_mod = import_hf_module("configuration_kimi_k3")
    mdl = import_hf_module("modeling_kimi_k3")
    vcfg = cfg_mod.KimiK3VisionConfig(
        vt_num_hidden_layers=NUM_LAYERS, text_hidden_size=OUT_HIDDEN, _attn_implementation="eager"
    )
    tower = mdl.MoonViT3dPretrainedModel(mdl.VisionTowerConfig(vcfg))
    # PreTrainedModel resolves the config's attention backend to "sdpa", which
    # MoonViT's own dispatch table lacks; pin its reference ("eager") path.
    for block in tower.encoder.blocks:
        block.attn_implementation = "eager"
    projector = mdl.PatchMergerMLPV2(mdl.ProjectorConfig(vcfg))
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for module in (tower, projector):
            for name, p in module.named_parameters():
                if p.dim() == 1:  # norm gains: around 1, not exactly 1
                    p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
                else:
                    p.copy_(0.02 * torch.randn(p.shape, generator=gen))
    return tower.to(device).float(), projector.to(device).float()


def hf_state_dict(tower, projector):
    """HF state dict with ``KimiK3ForConditionalGeneration`` key prefixes."""
    state = {f"vision_tower.{k}": v for k, v in tower.state_dict().items()}
    state.update({f"mm_projector.{k}": v for k, v in projector.state_dict().items()})
    return state


def build_mcore_vision(device, hf_state, reference: bool = False):
    """MCore encoder in fp32 with HF weights copied in.

    ``reference=True`` builds the TF32-free reference spec (see module docstring).
    """
    config = get_kimi_k3_vl_vision_config(
        num_layers_override=NUM_LAYERS,
        bf16=False,
        params_dtype=torch.float32,
        layernorm_epsilon=vision_norm_epsilon(torch.float32),
    )
    spec = get_kimi_k3_vl_vision_reference_spec() if reference else None
    encoder = KimiK3VLVisionEncoder(
        config, transformer_layer_spec=spec, out_hidden_size=OUT_HIDDEN
    ).to(device)
    mapped = hf_to_mcore_vision(hf_state, config.num_attention_heads, fused_norms=not reference)
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    missing = [k for k in missing if not k.endswith("_extra_state")]
    assert not missing and not unexpected, (missing, unexpected)
    return encoder


def make_pixels(device):
    gen = torch.Generator().manual_seed(11)
    rows = sum(t * h * w for t, h, w in GRIDS)
    pixels = torch.randn(rows, 3 * 14 * 14, generator=gen).to(device)
    return pixels, torch.tensor(GRIDS, dtype=torch.long, device=device)


@pytest.mark.parametrize(
    "reference,feature_rtol,grad_rtol",
    [(True, FEATURE_RTOL, GRAD_RTOL), (False, TE_FEATURE_RTOL, TE_GRAD_RTOL)],
    ids=["reference_spec_strict", "te_spec_tf32"],
)
def test_vision_tower_and_projector_match_hf(device, reference, feature_rtol, grad_rtol):
    tower, projector = build_hf_vision(device)
    hf_state = hf_state_dict(tower, projector)
    encoder = build_mcore_vision(device, hf_state, reference=reference)
    pixels, grid_thw = make_pixels(device)

    # HF: the processor hands the tower [rows, 3, 14, 14] patches.
    hf_merged = tower(pixels.view(-1, 3, 14, 14), grid_thw)
    hf_out = torch.cat(projector(hf_merged))

    tower_out, _ = encoder.encode_tower(pixels, grid_thw)
    mc_out = encoder(pixels, grid_thw)

    assert (
        mc_out.shape == hf_out.shape == (sum((h // 2) * (w // 2) for _, h, w in GRIDS), OUT_HIDDEN)
    )
    hf_tower_merged = torch.cat([x.flatten(1) for x in hf_merged])
    mc_tower_merged = tpool_patch_merge(tower_out, GRIDS, 2)
    tower_rel = _rel(mc_tower_merged, hf_tower_merged)
    out_rel = _rel(mc_out, hf_out)
    print(f"tower rel={tower_rel:.3e} output rel={out_rel:.3e}")
    assert tower_rel <= feature_rtol
    assert out_rel <= feature_rtol

    # Per-item check so an item-ordering bug cannot hide in the global norm.
    offset = 0
    for t, h, w in GRIDS:
        n = (h // 2) * (w // 2)
        item_rel = _rel(mc_out[offset : offset + n], hf_out[offset : offset + n])
        assert item_rel <= feature_rtol, ((t, h, w), item_rel)
        offset += n

    # Gradients for one shared cotangent.
    cot = torch.randn(hf_out.shape, generator=torch.Generator().manual_seed(3)).to(device)
    (hf_out * cot).sum().backward()
    (mc_out * cot).sum().backward()
    hf_grads = {
        k: p.grad
        for k, p in list((f"vision_tower.{n}", p) for n, p in tower.named_parameters())
        + [(f"mm_projector.{n}", p) for n, p in projector.named_parameters()]
    }
    mapped_grads = hf_to_mcore_vision(
        hf_grads, encoder.config.num_attention_heads, fused_norms=not reference
    )
    mc_params = dict(encoder.named_parameters())
    assert set(mapped_grads) == set(mc_params), set(mapped_grads) ^ set(mc_params)
    worst = max(
        ((_rel(mc_params[k].grad, g), k) for k, g in mapped_grads.items()), key=lambda x: x[0]
    )
    print(f"worst grad rel={worst[0]:.3e} ({worst[1]})")
    for key, grad in mapped_grads.items():
        assert mc_params[key].grad is not None, key
        assert _rel(mc_params[key].grad, grad) <= grad_rtol, (key, _rel(mc_params[key].grad, grad))


def test_bf16_forward_close_to_hf(device):
    """bf16 sanity: MCore bf16 output tracks the fp32 HF reference.

    Images only, because HF's own bf16 path promotes video items to fp32
    (its time embedding is an fp32 buffer).
    """
    tower, projector = build_hf_vision(device)
    hf_state = hf_state_dict(tower, projector)
    grids = [g for g in GRIDS if g[0] == 1]
    rows = sum(t * h * w for t, h, w in grids)
    pixels = torch.randn(rows, 588, generator=torch.Generator().manual_seed(5)).to(device)
    grid_thw = torch.tensor(grids, dtype=torch.long, device=device)
    with torch.no_grad():
        hf_out = torch.cat(projector(tower(pixels.view(-1, 3, 14, 14), grid_thw)))

    config = get_kimi_k3_vl_vision_config(
        num_layers_override=NUM_LAYERS,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=vision_norm_epsilon(torch.float32),
    )
    encoder = KimiK3VLVisionEncoder(config, out_hidden_size=OUT_HIDDEN).to(device)
    encoder.load_state_dict(hf_to_mcore_vision(hf_state, 12), strict=False)
    encoder = encoder.bfloat16()
    with torch.no_grad():
        mc_out = encoder(pixels.bfloat16(), grid_thw).float()
    rel = _rel(mc_out, hf_out)
    print(f"bf16 vs fp32-HF rel={rel:.3e}")
    assert rel < 2e-2
