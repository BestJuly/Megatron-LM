# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The NATIVE (non-MDP) vision encoder must keep working under decoder CP > 1.

Regression test for a defect the encoder-CP work introduced and an independent
review caught. The vision RoPE hook in ``specs.py`` used to substitute a
size-1 dummy group unconditionally. The encoder-CP rewrite made it substitute
only when the group it was handed had size 1 -- but in the native build
(``models/qwen35_vl/model.py``) the encoder receives no ``pg_collection``, so
``TransformerBlock`` falls back to the MPU groups and the attention module's
``pg_collection.cp`` is the DECODER's context-parallel group. Under
``--context-parallel-size 2`` with MDP off, that group has size 2 while the
vision tensor is not sharded at all, and ``_is_raw_mrope_freqs_thd`` raised
``freqs sequence length must match local tokens times cp_size`` on the first
vision-bearing microbatch of every native CP>1 run.

The fix branches on ``config.context_parallel_size`` -- which only MDP encoder
CP sets to ``encoder_cp`` -- instead of on the group's size. This test builds
the encoder exactly as the native path does (no ``pg_collection``, config CP
left at 1) inside an MPU whose decoder CP is 2, and requires both that the
forward runs and that it matches the encoder given an explicit singleton group.

Run with::

    PYTHONPATH=. torchrun --nproc-per-node 8 -m pytest -q \\
        examples/multimodal_dev/tests/test_native_vision_rope_under_decoder_cp.py
"""

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

# The decoder's CP degree. This is the group the native encoder inherits through
# the MPU fallback, and the whole point is that it must NOT affect vision RoPE.
DECODER_CP = 2

VISION_HIDDEN = 64
HIDDEN = 128
PATCH = 16
TEMPORAL_PATCH = 2
MERGE = 2
GRIDS = [(1, 4, 4), (1, 8, 8)]
# head_dim = 64 / 2 = 32, so the vision rotary has 16 frequencies; Qwen3.5-VL's
# vision RoPE wants mrope_section = [0, rows, cols] summing to that.
MROPE_SECTION = [0, 8, 8]


def _vision_config(mrope_section):
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
        mrope_section=mrope_section,
    )


def _build_encoder(config, pg_collection):
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
    return encoder.bfloat16().cuda().eval()


def _pixels(device):
    rows = sum(t * h * w for t, h, w in GRIDS)
    width = 3 * TEMPORAL_PATCH * PATCH * PATCH
    generator = torch.Generator(device="cpu").manual_seed(99)
    return torch.randn(rows, width, generator=generator).to(device=device, dtype=torch.bfloat16)


@pytest.fixture(scope="module", autouse=True)
def _mpu():
    # Decoder CP = 2 is the configuration that regressed. The MPU's CP group is
    # what the native encoder's attention inherits.
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=DECODER_CP,
    )
    yield
    Utils.destroy_model_parallel()


@pytest.fixture
def singleton_collection():
    """An explicit encoder_cp=1 collection: `pgs.cp` is a singleton group.

    This is what MDP hands the encoder at encoder_cp=1, and it is the
    behaviour the old unconditional dummy-group substitution produced. It is
    the reference the native build has to match.
    """
    rank_map = build_rank_map(
        MdpRankSpec(world_size=_WORLD, tp=1, pp=1, cp=DECODER_CP, ep=1, encoder_cp=1)
    )
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    yield build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    torch.distributed.barrier()


@pytest.mark.parametrize("mrope_section", [MROPE_SECTION, None], ids=["mrope", "legacy"])
def test_native_encoder_runs_and_matches_singleton_group(singleton_collection, mrope_section):
    """The native build (no pg_collection) must not see the decoder's CP.

    Two encoders, transplanted weights, same pixels: one built the native way,
    one given an explicit singleton CP group. Before the fix the native one
    raised inside RoPE under the mrope config; under the legacy config it would
    have sliced the frequency table per decoder-CP rank and produced wrong
    values silently. Either way it must now equal the singleton-group encoder.
    """
    config = _vision_config(mrope_section)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pixels = _pixels(device)
    grids = torch.tensor(GRIDS, dtype=torch.long)

    reference_encoder = _build_encoder(config, singleton_collection)
    weights = reference_encoder.state_dict()
    with torch.no_grad():
        reference = reference_encoder(pixels, grids)

    native_encoder = _build_encoder(config, pg_collection=None)
    native_encoder.load_state_dict(weights, strict=True)
    with torch.no_grad():
        native = native_encoder(pixels, grids)  # the regression raised here

    assert torch.isfinite(reference).all() and torch.isfinite(native).all()
    assert native.shape == reference.shape
    scale = reference.float().abs().max().clamp(min=1e-6)
    rel = ((native.float() - reference.float()).abs().max() / scale).item()
    assert rel < 2e-2, (
        f"native vision encoder under decoder CP={DECODER_CP} deviates {rel:.3e} from the "
        "singleton-group encoder; the decoder's CP group is leaking into vision RoPE "
        "or attention."
    )
