# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Full-model cross-check: one set of tiny Kimi-K3 VL weights in HF and MCore.

The HF ``KimiK3ForConditionalGeneration`` (remote code, pinned revision) is
random-initialized from a scaled-down real config (``kimi_tiny.py``), mapped
to MCore with ``models/kimi_k3_vl/hf_mapping.hf_to_mcore`` -- the same rules
as the toolkit converter -- and loaded into the ``KimiK3VLModel`` built by the
real ``pretrain_multimodal`` path. Stages are then compared on one sample with
two images and text:

1. weights: every HF tensor maps, no MCore parameter is left unloaded, and
   HF -> MCore (loaded into the model) -> HF is bitwise with the same key set;
2. vision features (tower + projector): TE-spec tolerance (TE GEMMs run fp32
   as TF32; the TF32-free 1e-5 check lives in test_kimi_k3_vl_vision_parity);
3. splice: HF ``_merge_input_ids_with_image_features`` vs MCore
   ``_scatter_vision_embeddings`` fed the *same* features -- must match to
   1e-6 relative, with identical image-slot positions;
4. logits: reported (max |delta|, argmax agreement), not asserted. Decoder
   numerics are out of scope for this package (Kimi-K3 decoder HF parity is
   owned by the decoder-only branch); a decoder-only difference is recorded.

Run with (needs the transformers 4.56 overlay, see tests/_kimi_hf.py)::

    torchrun --nproc_per_node=1 -m pytest -q -s \\
        examples/multimodal_dev/tests/test_kimi_k3_vl_crosscheck.py
"""

import pytest
import torch

from examples.multimodal_dev.models.kimi_k3_vl.configuration import KIMI_K3_IMAGE_TOKEN_ID
from examples.multimodal_dev.models.kimi_k3_vl.data import expand_media_pad
from examples.multimodal_dev.models.kimi_k3_vl.hf_mapping import hf_to_mcore, mcore_to_hf
from examples.multimodal_dev.tests import kimi_tiny

GRIDS = [(1, 16, 20), (1, 12, 8)]
VISION_RTOL = 2e-3
SPLICE_RTOL = 1e-6


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.fixture(scope="module")
def models():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hf = kimi_tiny.build_hf_model()
    # kimi_tiny zeroes the router's e_score_correction_bias (torch.empty in HF).
    mc = kimi_tiny.build_mcore_model().float()
    mapped = hf_to_mcore({k: v.detach() for k, v in hf.state_dict().items()})
    mc_state = {k for k in mc.state_dict() if not k.endswith("_extra_state")}
    dropped = [k for k in mapped if k not in mc_state and k.endswith("router.expert_bias")]
    for k in dropped:
        assert mapped[k].abs().max() == 0, k
        del mapped[k]
    unexpected = sorted(set(mapped) - mc_state)
    missing = sorted(mc_state - set(mapped))
    print(f"mapped={len(mapped)} missing={missing[:8]} unexpected={unexpected[:8]}")
    assert not unexpected and not missing
    mc.load_state_dict(mapped, strict=False)
    mc.eval()
    return hf, mc


def _sample(device):
    gen = torch.Generator().manual_seed(3)
    rows = sum(t * h * w for t, h, w in GRIDS)
    pixels = torch.randn(rows, 588, generator=gen).to(device)
    grid = torch.tensor(GRIDS, dtype=torch.long, device=device)
    text = torch.randint(1000, 100000, (40,), generator=gen).tolist()
    hf_ids = (
        text[:10] + [KIMI_K3_IMAGE_TOKEN_ID] + text[10:25] + [KIMI_K3_IMAGE_TOKEN_ID] + text[25:]
    )
    n_tokens = [(h // 2) * (w // 2) for _, h, w in GRIDS]
    mc_ids = expand_media_pad(hf_ids, n_tokens, KIMI_K3_IMAGE_TOKEN_ID)
    return (
        pixels,
        grid,
        torch.tensor([hf_ids], device=device),
        torch.tensor([mc_ids], device=device),
    )


def test_crosscheck_stages(models):
    hf, mc = models
    device = next(hf.parameters()).device
    pixels, grid, hf_ids, mc_ids = _sample(device)

    with torch.no_grad():
        # 2. vision features
        hf_feats = hf.mm_projector(hf.vision_tower(pixels.view(-1, 3, 14, 14), grid))
        hf_flat = torch.cat(list(hf_feats))
        mc_feats = mc.vision_model(pixels, grid)
        vision_rel = _rel(mc_feats, hf_flat)
        print(f"vision rel={vision_rel:.3e}")
        assert vision_rel <= VISION_RTOL

        # 3. splice, with HF's features on both sides
        hf_text = hf.language_model.get_input_embeddings()(hf_ids)
        merged, _, _, _ = hf._merge_input_ids_with_image_features(
            list(hf_feats), hf_text, hf_ids, torch.ones_like(hf_ids), None
        )
        mc_text = mc.language_model.embedding(input_ids=mc_ids, position_ids=None)
        mc_merged = mc._scatter_vision_embeddings(mc_ids, mc_text, hf_flat)
        assert merged.shape[1] == mc_ids.shape[1]
        splice_rel = _rel(mc_merged.transpose(0, 1), merged)
        print(f"splice rel={splice_rel:.3e}")
        assert splice_rel <= SPLICE_RTOL

        # 4. logits (recorded, not asserted)
        hf_logits = hf(
            input_ids=hf_ids,
            attention_mask=torch.ones_like(hf_ids),
            pixel_values=pixels.view(-1, 3, 14, 14),
            grid_thws=grid,
        ).logits
        mc_logits = mc(
            input_ids=mc_ids, position_ids=None, pixel_values=pixels, image_grid_thw=grid
        )
        if mc_logits.dim() == 3 and mc_logits.shape[0] == mc_ids.shape[1]:
            mc_logits = mc_logits.transpose(0, 1)
        hf_logits, mc_logits = hf_logits.float(), mc_logits.float()
        assert torch.isfinite(hf_logits).all() and torch.isfinite(mc_logits).all()
        max_abs = (hf_logits - mc_logits).abs().max().item()
        argmax = (hf_logits.argmax(-1) == mc_logits.argmax(-1)).float().mean().item()
        print(
            f"logits: max|d|={max_abs:.3e} rel={_rel(mc_logits, hf_logits):.3e} "
            f"argmax_agree={argmax:.4f} (decoder: recorded, not asserted)"
        )


def test_hf_mcore_hf_round_trip_is_bitwise(models):
    """HF -> MCore state (as loaded into the model) -> HF reproduces every tensor."""
    hf, mc = models
    hf_state = {k: v.detach().cpu() for k, v in hf.state_dict().items()}
    mc_state = {
        k: v.detach().cpu() for k, v in mc.state_dict().items() if not k.endswith("_extra_state")
    }
    back = mcore_to_hf(mc_state, {k: tuple(v.shape) for k, v in hf_state.items()})
    # The tiny MCore model has no router expert bias; HF's was zeroed above.
    absent = sorted(set(hf_state) - set(back))
    assert all(k.endswith("e_score_correction_bias") for k in absent), absent
    assert all(hf_state[k].abs().max() == 0 for k in absent)
    assert set(back) == set(hf_state) - set(absent)
    mismatched = [k for k in back if not torch.equal(back[k], hf_state[k])]
    print(f"round trip: {len(back)} tensors, {len(mismatched)} mismatched")
    assert not mismatched, mismatched[:5]
