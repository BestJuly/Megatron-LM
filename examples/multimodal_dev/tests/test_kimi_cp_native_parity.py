# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native Kimi-K3 VL (tiny) with decoder CP: contiguous CP2 reproduces CP1.

The tiny Kimi-K3 VL (``kimi_tiny``: KDA + MLA hybrid decoder, MoonViT) is built
through the real ``pretrain_multimodal.model_provider`` path, and one batch of
two image-bearing ``mdp_mock`` samples goes through the real collator
(``forward_step.get_batch`` -> ``pack_or_pad_batch`` with static THD packing),
so the split-before-embed model path runs in Kimi-K3's release CP layout
(``--cp-partition-mode contiguous``: KDA chunkwise CP, MLA converting
internally). The static target is 4096 rows at both sizes (4096 x 1 and
2048 x 2), so both runs see the identical packed batch.

Megatron is initialized once per process, so the comparison spans two
invocations sharing ``KIMI_CP_REF``: CP1 writes the reference (global loss,
token count, every parameter gradient), CP2 compares -- token count exactly,
loss and gradients (summed over the CP group) to bf16 tolerance -- and checks
that a deliberately misrouted variant falls clearly outside that tolerance::

    KIMI_CP_REF=/tmp/kimi_cp_ref.pt torchrun --nproc_per_node=1 -m pytest -q \\
        examples/multimodal_dev/tests/test_kimi_cp_native_parity.py
    KIMI_CP_REF=/tmp/kimi_cp_ref.pt torchrun --nproc_per_node=2 -m pytest -q \\
        examples/multimodal_dev/tests/test_kimi_cp_native_parity.py
"""

import os

import pytest
import torch

REF = os.environ.get("KIMI_CP_REF")
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    REF is None or WORLD not in (1, 2) or not torch.cuda.is_available(),
    reason="needs KIMI_CP_REF and a torchrun world of 1 (reference) or 2 (CP2)",
)

STATIC_ROWS = 4096
LOSS_RTOL = 2e-3
# bf16 throughout (CP attention needs a fused/flash kernel), KDA chunkwise CP
# reduces its recurrent state in a different order than the single-rank kernel,
# and near-tie MoE top-2 routing decisions can flip: on H100 the measured
# gradient difference is a few percent (3.5% decoder, 5.0% vision with the mock
# pixels) at a loss difference of ~2e-5. The gradient bounds are bf16-noise
# bounds; the misrouting control below shows they still separate a routing
# error. The mock dataset's pixels are constant per item, which makes an item's
# rows nearly interchangeable (a misrouted variant then moved the vision error
# only from 5.0% to 6.8%), so the batch's pixels are replaced with seeded noise.
GRAD_GLOBAL_RTOL = 5e-2
VISION_SANITY_RTOL = 1e-1
CONTROL_LOSS_FACTOR = 10.0
CONTROL_VISION_FACTOR = 3.0


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _setup():
    from examples.multimodal_dev.data.mdp_mock import build_scenarios
    from examples.multimodal_dev.forward_step import get_batch
    from examples.multimodal_dev.models.kimi_k3_vl.configuration import KIMI_K3_IMAGE_TOKEN_ID
    from examples.multimodal_dev.models.kimi_k3_vl.data import KimiMdpMockDataset, kimi_scenarios
    from examples.multimodal_dev.tests import kimi_tiny
    from megatron.core import parallel_state

    cp = WORLD
    model = kimi_tiny.build_mcore_model(
        extra_args=[
            "--bf16",
            "--context-parallel-size",
            str(cp),
            "--cp-partition-mode",
            "contiguous",
            "--thd-static-packing",
            "--pad-packed-seq-alignment",
            "max",
            "--max-seqlen-per-dp-cp-rank",
            str(STATIC_ROWS // cp),
            "--thd-max-packed-sequences",
            "4",
            "--seq-length",
            str(STATIC_ROWS),
            "--max-position-embeddings",
            str(STATIC_ROWS),
        ]
    ).bfloat16()
    assert parallel_state.get_context_parallel_world_size() == cp
    assert model.config.cp_partition_mode == "contiguous"

    dataset = KimiMdpMockDataset(
        num_samples=64,
        vocab_size=163840,
        image_token_id=KIMI_K3_IMAGE_TOKEN_ID,
        scenarios=kimi_scenarios(build_scenarios()),
    )
    picks = [i for i, (grids, _) in enumerate(dataset.scenarios) if grids][:2]
    batch = get_batch(iter([[dict(dataset[i]) for i in picks]]))
    # Seeded noise instead of the mock's per-item constant pixels, identical in
    # both invocations, so every vision row is distinct.
    generator = torch.Generator().manual_seed(4321)
    batch["pixel_values"] = torch.randn(tuple(batch["pixel_values"].shape), generator=generator).to(
        batch["pixel_values"].device
    )
    params = batch["packed_seq_params"]
    assert params.cp_partition_mode == "contiguous"
    assert int(params.cu_seqlens_q_padded[-1]) == STATIC_ROWS
    assert int((batch["input_ids"] == KIMI_K3_IMAGE_TOKEN_ID).sum()) > 0
    return model, batch


def _forward_backward(model, batch):
    from examples.multimodal_dev.models.base import MultimodalModel

    model.zero_grad(set_to_none=True)
    params = batch["packed_seq_params"]
    losses = model(
        input_ids=batch["input_ids"],
        position_ids=batch.get("position_ids"),
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        padding_mask=batch.get("padding_mask"),
        pixel_values=batch["pixel_values"].bfloat16(),
        image_grid_thw=batch["image_grid_thw"],
        packed_seq_params=params,
    )
    mask = MultimodalModel.cp_split_loss_mask(batch["loss_mask"], params).float()
    assert losses.shape[-1] == STATIC_ROWS // WORLD
    local = (losses.float() * mask).sum()
    local.backward()
    stats = torch.stack([local.detach(), mask.sum()])
    torch.distributed.all_reduce(stats)  # WORLD == the CP group here (TP=PP=DP=1)
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.float().contiguous()
        torch.distributed.all_reduce(grad)
        grads[name] = grad.cpu()
    return float(stats[0]), int(stats[1]), grads


def _group_rel(grads, ref, prefix=""):
    keys = [k for k in grads if k.startswith(prefix)]
    num = sum(float((grads[k].double() - ref[k].double()).norm() ** 2) for k in keys)
    den = sum(float(ref[k].double().norm() ** 2) for k in keys)
    return (num / max(den, 1e-300)) ** 0.5


def test_contiguous_cp2_native_kimi_matches_cp1(monkeypatch):
    from examples.multimodal_dev.models.base import MultimodalModel

    model, batch = _setup()
    loss, tokens, grads = _forward_backward(model, batch)
    if WORLD == 1:
        torch.save({"loss": loss, "tokens": tokens, "grads": grads}, REF)
        print(f"wrote the CP1 reference to {REF}: loss {loss:.6f}, {tokens} tokens")
        return
    ref = torch.load(REF)
    assert tokens == ref["tokens"]
    assert set(grads) == set(ref["grads"])
    assert any(k.startswith("vision_model.") for k in grads)
    assert any(k.startswith("language_model.") for k in grads)
    loss_rel = abs(loss - ref["loss"]) / abs(ref["loss"])
    errors = {
        "global": _group_rel(grads, ref["grads"]),
        "decoder": _group_rel(grads, ref["grads"], "language_model."),
        "vision": _group_rel(grads, ref["grads"], "vision_model."),
    }
    worst = max(grads, key=lambda k: _rel(grads[k], ref["grads"][k]))

    # Negative control: the same step with this rank's vision rows in reverse
    # order (rows land on other slots and items) -- a routing error of exactly
    # the kind the split must not make. It must be clearly outside the noise.
    select = MultimodalModel._cp_local_vision_rows

    def misrouted(self, *args, **kwargs):
        rows = select(self, *args, **kwargs)
        return rows.flip(0) if rows is not None and rows.shape[0] > 1 else rows

    monkeypatch.setattr(MultimodalModel, "_cp_local_vision_rows", misrouted)
    bad_loss, _, bad_grads = _forward_backward(model, batch)
    control = {
        "loss": abs(bad_loss - ref["loss"]) / abs(ref["loss"]),
        "vision": _group_rel(bad_grads, ref["grads"], "vision_model."),
    }
    print(
        f"CP2 vs CP1: loss rel {loss_rel:.3e}; grad rel {errors} "
        f"(worst tensor {worst}: {_rel(grads[worst], ref['grads'][worst]):.3e}); "
        f"misrouted control: {control}"
    )
    assert control["loss"] >= CONTROL_LOSS_FACTOR * max(loss_rel, 1e-7), (control, loss_rel)
    assert control["vision"] >= CONTROL_VISION_FACTOR * errors["vision"], (control, errors)
    assert loss_rel <= LOSS_RTOL, loss_rel
    assert errors["global"] <= GRAD_GLOBAL_RTOL, errors
    assert errors["vision"] <= VISION_SANITY_RTOL, errors
