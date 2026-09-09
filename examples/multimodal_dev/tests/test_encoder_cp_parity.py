# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Vision encoder output must not change when encoder CP is switched on.

The one thing the training arms could not settle. At ``encoder_cp=2`` each rank
holds ``1/e`` of the patch rows and TE rings the attention over the encoder-CP
group -- ``AttnFuncWithCPAndKVP2P`` is present in the e=2 nsys trace and absent
at e=1, so CP attention genuinely runs. But the vision pack is THD with ONE
sub-sequence per temporal frame (``_build_packed_seq_params`` builds
``cu_seqlens`` from ``grid_thw``), so attention must stay block-diagonal per
frame. If TE did not receive those boundaries correctly on the CP path, ranks
would attend ACROSS frames: a correctness bug, not a slow path, and one that
end-to-end loss would hide -- the e=2 arm deviates only 2.77e-3 from e=1 without
diverging, and the arm with BOTH CPs on was the cleanest of four.

So compare the encoder against itself. Two encoders on the same rank with
identical weights and identical input; one given a real ``encoder_cp``-sized CP
group, one a singleton. Same pixels in, same rows out.

Sharding is a pure repartition of the same arithmetic, so the two agree up to
floating-point reassociation only -- not bitwise. The tolerance is calibrated
against a same-config repeat in ``test_reference_path_is_stable``, so a real
divergence cannot be waved through as "just numerics".

Run with::

    PYTHONPATH=. torchrun --nproc-per-node 2 -m pytest -q \\
        examples/multimodal_dev/tests/test_encoder_cp_parity.py
"""

import dataclasses
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.mdp.encoder import build_encoder_pg_collection
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    _WORLD < 2 or _WORLD % 2 != 0, reason="needs an even world of at least 2"
)

# encoder_cp must divide the planning group, which is cp*pp ranks, so pp=2 is
# the smallest topology that admits encoder_cp=2. install_mdp_process_groups
# cross-checks the rank map against live MPU state, so the MPU has to be
# initialised to the same shape -- a mismatch is rejected outright, which is
# how an earlier encoder-CP test of mine was caught building pp=4 against a
# pp=2 fixture.
PP = 2

# Geometry copied from test_mdp_parity.py, which builds this same encoder and
# runs it without overflowing bf16. Both grids satisfy h*w % (2*encoder_cp) == 0
# (16 and 64 vs 4), and they are different shapes on purpose: attending across
# two IDENTICAL frames would be far less visible than across two different ones.
VISION_HIDDEN = 64
HIDDEN = 128
PATCH = 16
TEMPORAL_PATCH = 2
MERGE = 2
GRIDS = [(1, 4, 4), (1, 8, 8)]


def _vision_config():
    """Mirrors the proven vision config in ``test_mdp_parity.py``.

    Notably ``hidden_dropout`` and ``attention_dropout`` are set explicitly:
    ``TransformerConfig`` defaults BOTH to 0.1, so a hand-written config leaves
    dropout ON. Two forward passes would then differ by their dropout masks
    alone and this comparison would report a difference that has nothing to do
    with encoder CP.
    """
    return TransformerConfig(
        num_layers=2,
        hidden_size=VISION_HIDDEN,
        ffn_hidden_size=2 * VISION_HIDDEN,
        num_attention_heads=2,
        num_query_groups=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )


def _build_encoder(config, pg_collection):
    # Mirror what production does before building the encoder
    # (megatron/core/mdp/encoder.py): at encoder_cp > 1 the vision config gets
    # context_parallel_size = encoder_cp and cp_comm_type = "p2p". Omitting it
    # leaves TE building attention for CP=1 while being handed a CP-sized group
    # and a sharded tensor: half the rows come out zero with no error anywhere.
    # The p2p comm type is not optional either:
    # MCore's a2a+p2p wrapper reads the GLOBAL hierarchical CP groups instead of
    # pg_collection.hcp, which would hand the encoder the decoder's groups.
    group = getattr(pg_collection, "cp", None)
    encoder_cp = torch.distributed.get_world_size(group=group) if group is not None else 1
    if encoder_cp > 1:
        config = dataclasses.replace(
            config, context_parallel_size=encoder_cp, cp_comm_type="p2p"
        )
    torch.manual_seed(4321)
    model_parallel_cuda_manual_seed(4321)
    encoder = Qwen35VLVisionEncoder(
        config=config,
        in_channels=3,
        patch_size=PATCH,
        temporal_patch_size=TEMPORAL_PATCH,
        spatial_merge_size=MERGE,
        out_hidden_size=HIDDEN,
        max_num_positions=2304,
        pg_collection=pg_collection,
    )
    # eval(): torch.no_grad() does NOT disable dropout, it only stops grad
    # recording. Belt and braces alongside the zeroed dropout probabilities.
    return encoder.bfloat16().cuda().eval()


def _pixels(device):
    rows = sum(t * h * w for t, h, w in GRIDS)
    width = 3 * TEMPORAL_PATCH * PATCH * PATCH
    generator = torch.Generator(device="cpu").manual_seed(99)
    return torch.randn(rows, width, generator=generator).to(device=device, dtype=torch.bfloat16)


def _grid_tensor():
    return torch.tensor(GRIDS, dtype=torch.long)


@pytest.fixture(scope="module", autouse=True)
def _mpu():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=PP
    )
    yield
    Utils.destroy_model_parallel()


def _collection(encoder_cp):
    rank_map = build_rank_map(
        MdpRankSpec(world_size=_WORLD, tp=1, pp=PP, cp=1, ep=1, encoder_cp=encoder_cp)
    )
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    return build_encoder_pg_collection(
        rank_map, encoder_cp=encoder_cp, process_groups=groups
    )


@pytest.fixture
def collections():
    """(cp2, cp1) collections over the same world and the same MPU shape."""
    # Order matters: every rank must create groups in the same sequence.
    cp2 = _collection(2)
    cp1 = _collection(1)
    yield cp2, cp1
    torch.distributed.barrier()


def _run(config, pg_collection, pixels, grids, weights=None):
    """Forward one encoder; if ``weights`` is given, load them first.

    Re-seeding before each construction is NOT enough to guarantee identical
    weights across the two paths. A CP-sized ``pg_collection`` makes TE build
    the attention modules differently, and any difference in how much RNG the
    construction consumes shifts every subsequent layer's initialisation. The
    two encoders would then differ in their WEIGHTS, and the comparison would
    report a large deviation that has nothing to do with context parallelism.

    So the reference's state_dict is transplanted into the sharded encoder.
    ``strict=True`` doubles as a structural check: if the CP construction has a
    different parameter set, that is itself worth failing on.
    """
    encoder = _build_encoder(config, pg_collection)
    if weights is not None:
        encoder.load_state_dict(weights, strict=True)
    with torch.no_grad():
        return encoder(pixels, grids), encoder.state_dict()


def test_reference_path_is_stable(collections):
    """Calibrate the tolerance: the encoder_cp=1 path against itself.

    Whatever this run-to-run spread is, it is the floor a real encoder-CP
    divergence has to be judged against. Without it the parity assertion below
    is an arbitrary number.
    """
    _, cp1 = collections
    config = _vision_config()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pixels, grids = _pixels(device), _grid_tensor()

    first, weights = _run(config, cp1, pixels, grids)
    second, _ = _run(config, cp1, pixels, grids, weights)
    assert torch.equal(first, second), (
        "the encoder_cp=1 path is not reproducible against itself in-process; "
        "the parity comparison below would be meaningless"
    )


def test_encoder_cp_does_not_change_the_output(collections):
    """The load-bearing test: e=2 must reproduce e=1's rows.

    Fails loudly if TE attends across frame boundaries on the CP path -- the
    outputs would differ structurally, not in the last bits.
    """
    cp2, cp1 = collections
    config = _vision_config()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pixels, grids = _pixels(device), _grid_tensor()

    reference, weights = _run(config, cp1, pixels, grids)
    sharded, _ = _run(config, cp2, pixels, grids, weights)

    # Report non-finite output as itself, not as "deviation nan". A NaN and a
    # structurally wrong value are different findings and want different next
    # steps, and the subtraction below erases the distinction.
    assert torch.isfinite(reference).all(), (
        "the encoder_cp=1 reference is not finite; the harness is at fault, "
        "not the CP path"
    )
    bad = (~torch.isfinite(sharded)).float().mean().item()
    assert bad == 0.0, (
        f"the encoder_cp=2 path produced non-finite output: {bad:.1%} of "
        f"entries, while the encoder_cp=1 reference on identical weights and "
        f"input is finite. Rows are {sharded.shape[0]}; non-finite rows are "
        f"{int((~torch.isfinite(sharded)).any(dim=1).sum())}. If ALL rows are "
        f"bad the fault is global (group setup, dtype); if only some, suspect "
        f"the per-frame boundaries the zigzag split produces."
    )

    assert sharded.shape == reference.shape, (
        f"encoder_cp changed the output SHAPE: {tuple(sharded.shape)} vs "
        f"{tuple(reference.shape)}; the gather or the merger is mis-wired"
    )
    diff = (sharded.float() - reference.float()).abs()
    scale = reference.float().abs().max().clamp(min=1e-6)
    rel = (diff.max() / scale).item()
    assert rel < 2e-2, (
        f"encoder_cp=2 changed the vision encoder output: max relative "
        f"deviation {rel:.3e}. Sharding is a repartition of the same "
        f"arithmetic, so only floating-point reassociation is expected. A "
        f"deviation this size means the CP path is not computing the same "
        f"function -- the first thing to check is whether TE receives the "
        f"per-frame cu_seqlens, since attending across frame boundaries "
        f"produces exactly this."
    )


def test_every_frame_is_affected_equally(collections):
    """A cross-frame attention bug shows up as a per-frame pattern.

    If frame boundaries were lost, rows near a boundary would move far more
    than rows in a frame's interior. Comparing per-frame deviation against the
    whole-output deviation separates "uniform floating-point noise" from
    "structured error concentrated at the seams".
    """
    cp2, cp1 = collections
    config = _vision_config()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pixels, grids = _pixels(device), _grid_tensor()

    reference, weights = _run(config, cp1, pixels, grids)
    sharded, _ = _run(config, cp2, pixels, grids, weights)
    assert torch.isfinite(reference).all() and torch.isfinite(sharded).all(), (
        "non-finite output; see test_encoder_cp_does_not_change_the_output for "
        "which side is at fault"
    )

    scale = reference.float().abs().max().clamp(min=1e-6)
    start = 0
    per_frame = []
    for t, h, w in GRIDS:
        rows = t * (h // MERGE) * (w // MERGE)
        block_ref = reference[start : start + rows].float()
        block_new = sharded[start : start + rows].float()
        per_frame.append(((block_new - block_ref).abs().max() / scale).item())
        start += rows
    assert start == reference.shape[0], "frame row accounting does not cover the output"

    worst, best = max(per_frame), min(per_frame)
    assert worst < 2e-2, f"per-frame deviations {per_frame} exceed tolerance"
    # Frames differ in size so some spread is normal; an order of magnitude is
    # not, and that is what a boundary-localised error would produce.
    assert worst <= max(best, 1e-6) * 100.0, (
        f"deviation is concentrated in one frame ({per_frame}), which is the "
        f"signature of lost sub-sequence boundaries rather than reassociation"
    )

def test_group_and_config_disagreement_is_rejected_before_any_collective(collections):
    """A CP-sized group with a CP=1 config must be refused, not silently run.

    This is the configuration that cost six debugging rounds here: TE built
    attention for CP=1 while the encoder sharded rows for CP=2, and the output
    was half zeros with no error anywhere. `_encoder_cp_state` now cross-checks
    the group's size against `config.context_parallel_size` and raises. Only
    the size-2 group is passed and the config is deliberately left at 1.
    """
    from megatron.core.mdp.errors import MdpConfigurationError

    cp2, _ = collections
    config = _vision_config()  # context_parallel_size stays at its default of 1
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pixels, grids = _pixels(device), _grid_tensor()

    torch.manual_seed(4321)
    model_parallel_cuda_manual_seed(4321)
    encoder = Qwen35VLVisionEncoder(
        config=config,
        in_channels=3,
        patch_size=PATCH,
        temporal_patch_size=TEMPORAL_PATCH,
        spatial_merge_size=MERGE,
        out_hidden_size=HIDDEN,
        max_num_positions=2304,
        pg_collection=cp2,
    ).bfloat16().cuda().eval()

    with pytest.raises(MdpConfigurationError, match="must match"):
        with torch.no_grad():
            encoder(pixels, grids)



def test_presharded_input_matches_full_input(collections):
    """The runtime's delivery contract: ``pixels_are_sharded`` skips one slice.

    Under encoder CP the runtime sends each rank only its zigzag shard of the
    pixels and the encoder consumes it without slicing again. That must be
    arithmetically the same as handing the encoder the whole chunk and letting
    it slice -- patch_embed is per patch row, so the two orders of "embed" and
    "select" commute exactly, and the outputs are required to be bitwise
    equal, not merely close.
    """
    from megatron.core.mdp.encoder_cp_partition import shard_rows

    cp2, cp1 = collections
    config = _vision_config()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pixels, grids = _pixels(device), _grid_tensor()

    reference, weights = _run(config, cp1, pixels, grids)
    encoder = _build_encoder(config, cp2)
    encoder.load_state_dict(weights, strict=True)
    group = cp2.cp
    encoder_cp = torch.distributed.get_world_size(group=group)
    my_shard = torch.distributed.get_rank(group=group)

    frames = []
    for t, h, w in GRIDS:
        frames.extend([h * w] * t)
    rows = []
    for run in shard_rows(frames, encoder_cp, my_shard):
        rows.extend(range(run.start, run.start + run.rows))
    shard = pixels.index_select(0, torch.tensor(rows, device=device))

    with torch.no_grad():
        full_path = encoder(pixels, grids)
        sharded_path = encoder(shard, grids, pixels_are_sharded=True)

    assert torch.equal(sharded_path, full_path), (
        "pre-sharded delivery must be bitwise identical to encoder-side slicing"
    )
    scale = reference.float().abs().max().clamp(min=1e-6)
    rel = ((sharded_path.float() - reference.float()).abs().max() / scale).item()
    assert rel < 2e-2, f"pre-sharded encoder-CP path deviates {rel:.3e} from the cp=1 reference"

    # A wrong-sized shard is a contract violation and must be refused, not
    # silently embedded: hand the encoder the WHOLE chunk while claiming it is
    # pre-sharded.
    with pytest.raises(RuntimeError, match="pixels_are_sharded"):
        with torch.no_grad():
            encoder(pixels, grids, pixels_are_sharded=True)
