# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""G1/G2 parity: the MDP phase machine must reproduce the native pixel path.

Both paths run the same weights (the vision encoder is transplanted into the
MDP encoder domain) on the same batch:

* native: ``MultimodalModel`` encodes pixels in-model and scatters;
* MDP: P1-P3 produce the endpoint leaf, the decoder consumes it through
  ``vision_embeddings``, and P5 routes the leaf gradient back through the
  producer handle.

Loss must match bitwise, decoder gradients bitwise, and encoder gradients to
bf16-ulp tolerance after the WORLD-sum/1/T normalization (8 identical ranks,
T=8): once weight gradients materialize in bf16, a one-ulp difference between
the DDP-buffer and plain-autograd accumulation paths is inherent; routing
exactness is proven bitwise by the sentinel tests.
Recompute None/selective/full run a *stochastic* encoder (dropout > 0) and
must agree bitwise with each other, without perturbing the RNG stream visible
after the decoder.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q examples/multimodal_dev/tests/test_mdp_parity.py
"""

import os
import sys
from types import MappingProxyType

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.base import MultimodalModel
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import ModalityBridge
from megatron.core.mdp.config import MdpConfig, apply_vision_config_overrides
from megatron.core.mdp.encoder import (
    EncoderDomain,
    build_encoder_pg_collection,
    finalize_encoder_grads,
)
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1
pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

# Bitwise comparison across two separate invocations of the same ops requires
# deterministic kernels (the Conv3d patch-embed weight gradient is atomics-
# based otherwise).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if _DISTRIBUTED:

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        yield
        Utils.destroy_model_parallel()


HIDDEN = 128
VOCAB = 128
SEQ = 64
IMAGE_TOKEN_ID = 7
GRIDS = ((1, 4, 4), (1, 8, 8))  # 4 + 16 = 20 vision-token slots
PATCH_DIM = 3 * 2 * 16 * 16  # in_channels * temporal_patch * patch^2
DTYPE = torch.bfloat16


def _vision_config(dropout=0.0):
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        bf16=True,
        params_dtype=DTYPE,
        hidden_dropout=dropout,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )


def _language_config():
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        ffn_hidden_size=4 * HIDDEN,
        num_attention_heads=4,
        num_query_groups=4,
        bf16=True,
        params_dtype=DTYPE,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )


def _build_vision_encoder(config):
    torch.manual_seed(4321)
    model_parallel_cuda_manual_seed(4321)
    encoder = Qwen35VLVisionEncoder(
        config=config,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=HIDDEN,
        max_num_positions=2304,
    )
    # Mirror the Float16Module wrapping depth: the Conv3d patch embed and the
    # learned position embedding are plain torch modules and stay fp32
    # otherwise, which TE rejects outside autocast.
    return encoder.bfloat16().cuda()


def _build_language_model(with_encoder, vision_config):
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    encoder = _build_vision_encoder(vision_config) if with_encoder else None
    # Reset both RNG sources so the language weights are identical whether or
    # not an encoder was built first.
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    model = MultimodalModel(
        language_config=_language_config(),
        language_spec=get_gpt_layer_with_transformer_engine_spec(),
        vision_encoder=encoder,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        image_token_id=IMAGE_TOKEN_ID,
        position_embedding_type="rope",
        parallel_output=False,
        pre_process=True,
        post_process=True,
    )
    return model.cuda()


def _make_batch():
    generator = torch.Generator(device="cuda")
    generator.manual_seed(777)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), generator=generator, device="cuda")
    input_ids[input_ids == IMAGE_TOKEN_ID] = (IMAGE_TOKEN_ID + 1) % VOCAB
    slots = []
    cursor = 3
    for grid in GRIDS:
        t, h, w = grid
        rows = t * (h // 2) * (w // 2)
        slots.append(list(range(cursor, cursor + rows)))
        cursor += rows + 4  # interleaved text between items
    for block in slots:
        input_ids[0, block] = IMAGE_TOKEN_ID
    labels = torch.randint(0, VOCAB, (1, SEQ), generator=generator, device="cuda")
    loss_mask = (input_ids != IMAGE_TOKEN_ID).float()
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0)
    payload_rows = sum(t * h * w for t, h, w in GRIDS)
    pixel_values = torch.randn(
        payload_rows, PATCH_DIM, generator=generator, device="cuda", dtype=DTYPE
    )
    image_grid_thw = torch.tensor(GRIDS, device="cuda")
    return input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw, slots


class _ParityAdapter:
    """Real Qwen encoder factory/encode; capture comes from the test batch."""

    payload_width = PATCH_DIM
    spatial_merge_size = 2

    def __init__(self, vision_config, microbatches):
        self._vision_config = vision_config
        self._microbatches = list(microbatches)

    def get_batch(self, iterator):
        next(iterator)
        return self._microbatches.pop(0)

    def estimate_cost(self, item):
        return item.payload_rows

    def build_encoder(self, model_config, *, pg_collection):
        return _build_vision_encoder(model_config)

    def encode(self, encoder, payload, layout):
        grid_thw = torch.tensor(
            [segment.grid_thw for segment in layout.segments],
            dtype=torch.long,
            device=payload.device,
        )
        module = encoder.module if hasattr(encoder, "module") else encoder
        return module(payload, grid_thw)


def _captured_microbatch(pixel_values, slots):
    items = []
    payload_start = 0
    for ordinal, grid in enumerate(GRIDS):
        t, h, w = grid
        rows = t * h * w
        items.append(
            CapturedVisionItem(
                sample_id=0,
                image_ordinal=ordinal,
                grid_thw=grid,
                payload_row_start=payload_start,
                payload_rows=rows,
                decoder_positions=tuple(slots[ordinal]),
            )
        )
        payload_start += rows
    return CapturedMicrobatch(
        decoder_packed_seq_params=None,
        vision_items=tuple(items),
        flat_pixel_payload=pixel_values,
        model_payload=MappingProxyType({}),
    )


def _build_runtime(vision_config, microbatches):
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=1, cp=1, ep=1, encoder_cp=1)
    )
    view = rank_map.view(torch.distributed.get_rank())
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    adapter = _ParityAdapter(vision_config, microbatches)
    encoder = adapter.build_encoder(vision_config, pg_collection=encoder_pgs)
    encoder_ddp = DistributedDataParallel(
        config=vision_config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            # fp32 accumulation, as in production: a bf16 all-reduce rounds
            # differently per NCCL algorithm choice (message-size dependent),
            # which would make the two paths' summed grads differ by one ulp.
            grad_reduce_in_fp32=True,
        ),
        module=encoder,
        pg_collection=encoder_pgs,
    )
    allocator = DirectBufferAllocator()
    config = MdpConfig(enable=True)
    runtime = MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=view,
        process_groups=groups,
        adapter=adapter,
        encoder_domain=EncoderDomain(
            encoder_ddp=encoder_ddp, encoder_optimizer=None, effective_config=vision_config
        ),
        planner=MdpPlanner(
            view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy()
        ),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=HIDDEN,
        params_dtype=DTYPE,
        num_vpp_chunks=1,
    )
    return runtime, encoder_ddp


def _run_native(vision_config, batch):
    input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw, _ = batch
    model = _build_language_model(with_encoder=True, vision_config=vision_config)
    # Stochastic-encoder parity: both paths run encoder forward from the same
    # seeded stream (the MDP path reseeds identically before P2).
    torch.manual_seed(99)
    model_parallel_cuda_manual_seed(99)
    losses = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    total_loss = (losses.float() * loss_mask).sum()
    total_loss.backward()
    return model, total_loss


def _run_mdp(vision_config, batch, *, native_model, corrupt_leaf_grad=False):
    input_ids, labels, loss_mask, position_ids, pixel_values, _grid, slots = batch
    model = _build_language_model(with_encoder=False, vision_config=vision_config)
    # Identical language weights by construction (same seeds); assert one.
    native_emb = native_model.language_model.embedding.word_embeddings.weight
    mdp_emb = model.language_model.embedding.word_embeddings.weight
    assert torch.equal(native_emb, mdp_emb)

    runtime, encoder_ddp = _build_runtime(
        vision_config, [_captured_microbatch(pixel_values, slots)]
    )
    # Transplant the native encoder weights so both paths run the same model.
    encoder_ddp.module.load_state_dict(native_model.vision_model.state_dict())
    encoder_ddp.zero_grad_buffer()

    # Match the native path's RNG consumption for the stochastic encoder:
    # both paths run encoder forward from the same seeded stream.
    torch.manual_seed(99)
    model_parallel_cuda_manual_seed(99)
    replay = runtime.begin_iteration(iter(range(1)), num_microbatches=1, forward_only=False)
    next(replay[0])
    leaf = runtime.storage.get_leaf(0)
    assert leaf is not None and leaf.shape[0] == sum(len(s) for s in slots)

    losses = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        pixel_values=None,
        image_grid_thw=None,
        vision_embeddings=leaf,
    )
    total_loss = (losses.float() * loss_mask).sum()
    total_loss.backward()
    if corrupt_leaf_grad:
        leaf.grad.mul_(-1.0)

    rng_before_p5 = torch.cuda.get_rng_state()
    tracker_before_p5 = {
        name: state.clone()
        for name, state in get_cuda_rng_tracker().get_states().items()
    }
    tokens = torch.tensor(float(torch.distributed.get_world_size()), device="cuda")
    runtime.capture_global_num_tokens(tokens)
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    rng_after_p5 = torch.cuda.get_rng_state()
    tracker_after_p5 = get_cuda_rng_tracker().get_states()

    assert torch.equal(rng_before_p5, rng_after_p5), (
        "P5 backward (checkpoint replay) must not perturb the global RNG stream"
    )
    for name, state in tracker_before_p5.items():
        assert torch.equal(state, tracker_after_p5[name]), name

    return model, total_loss, encoder_ddp


def _seeded(fn, *args, **kwargs):
    torch.manual_seed(99)
    model_parallel_cuda_manual_seed(99)
    return fn(*args, **kwargs)


def _assert_encoder_grads_match(encoder_ddp, native_model, rtol=8e-3, atol=1e-5):
    native_encoder_grads = {
        name: param.grad.float()
        for name, param in native_model.vision_model.named_parameters()
        if param.grad is not None
    }
    for name, param in encoder_ddp.module.named_parameters():
        grad = param.main_grad.float()
        reference = native_encoder_grads[name]
        # bf16 wgrads accumulate in different orders on the two paths, so an
        # element can be off by one ulp of the tensor's LARGEST accumuland;
        # scale the absolute tolerance accordingly.
        atol_eff = max(atol, 2e-3 * float(reference.abs().max()))
        assert torch.allclose(grad, reference, rtol=rtol, atol=atol_eff), (
            name,
            float((grad - reference).abs().max()),
        )


def test_g1_mdp_matches_native_pixel_path():
    batch = _make_batch()
    vision_config = _vision_config(dropout=0.0)
    native_model, native_loss = _seeded(_run_native, vision_config, batch)
    mdp_model, mdp_loss, encoder_ddp = _run_mdp(
        _vision_config(dropout=0.0), batch, native_model=native_model
    )

    # Loss bitwise.
    assert torch.equal(native_loss, mdp_loss), (float(native_loss), float(mdp_loss))

    # Decoder gradients bitwise (pre-clip, pre-optimizer).
    native_grads = {
        name: param.grad
        for name, param in native_model.language_model.named_parameters()
        if param.grad is not None
    }
    for name, param in mdp_model.language_model.named_parameters():
        if param.grad is None:
            assert name not in native_grads, name
            continue
        assert torch.equal(param.grad, native_grads[name]), name

    # Encoder gradients: WORLD sum of 8 identical lanes / T(=8) reproduces the
    # native single-model gradient. Comparison is tight-allclose, not bitwise:
    # under DDP the TE linears accumulate wgrad fused in fp32 (main_grad),
    # while the native non-DDP path accumulates in bf16 param.grad — a
    # last-bit precision difference (~1e-7 abs observed), not a routing error.
    # Routing exactness is proven bitwise by the sentinel tests and the
    # fault-injection twin below.
    _assert_encoder_grads_match(encoder_ddp, native_model)


def test_g1_fault_injection_detects_broken_reverse_routing():
    batch = _make_batch()
    vision_config = _vision_config(dropout=0.0)
    native_model, _ = _seeded(_run_native, vision_config, batch)
    _, _, encoder_ddp = _run_mdp(
        _vision_config(dropout=0.0),
        batch,
        native_model=native_model,
        corrupt_leaf_grad=True,
    )
    native_encoder_grads = {
        name: param.grad.float()
        for name, param in native_model.vision_model.named_parameters()
        if param.grad is not None
    }
    mismatches = sum(
        0
        if torch.allclose(
            param.main_grad.float(),
            native_encoder_grads[name],
            rtol=8e-3,
            atol=max(1e-5, 2e-3 * float(native_encoder_grads[name].abs().max())),
        )
        else 1
        for name, param in encoder_ddp.module.named_parameters()
    )
    assert mismatches > 0, "corrupting the leaf gradient must change encoder grads"


@pytest.mark.parametrize(
    "overrides",
    [
        (("recompute_granularity", "selective"),),
        (
            ("recompute_granularity", "full"),
            ("recompute_method", "uniform"),
            ("recompute_num_layers", 1),
        ),
    ],
    ids=["selective", "full"],
)
def test_g1_recompute_modes_match_reference(overrides):
    """Stochastic encoder (dropout>0): every recompute mode must reproduce the
    no-recompute reference **bitwise** through the one shared P2/P5 path.

    The reference is the MDP run without recompute (identical seeding and op
    order, so bitwise equality is the correct bar); native-vs-MDP numerical
    parity is covered by the deterministic test above. Recompute replay draws
    its dropout masks through MCore's RNG fork, so any custom RNG handling
    would break the bitwise match here.
    """
    batch = _make_batch()
    # Weight source: a native model provides the transplant weights for both
    # MDP runs, keeping them identical.
    native_model, _ = _seeded(_run_native, _vision_config(dropout=0.0), batch)

    reference_config = _vision_config(dropout=0.1)
    _, reference_loss, reference_ddp = _run_mdp(
        reference_config, batch, native_model=native_model
    )
    reference_grads = {
        name: param.main_grad.float().clone()
        for name, param in reference_ddp.module.named_parameters()
    }

    mode_config = apply_vision_config_overrides(_vision_config(dropout=0.1), overrides)
    _, mode_loss, mode_ddp = _run_mdp(mode_config, batch, native_model=native_model)
    assert torch.equal(reference_loss, mode_loss), (
        float(reference_loss),
        float(mode_loss),
    )
    for name, param in mode_ddp.module.named_parameters():
        assert torch.equal(param.main_grad.float(), reference_grads[name]), name


def test_g1_one_step_update_parity():
    """G1's third leg: one optimizer step from identical state must produce the
    same updated weights on both paths.

    Both decoders are DDP-wrapped identically (all-reduce over the default DP
    group) and both paths use the same OptimizerConfig: native is one Megatron
    optimizer over vision+language; MDP is the composite
    [decoder, encoder]. The captured token count is 1 so the encoder's
    WORLD-sum matches the native full-model DDP reduction exactly. Clipping is
    forced (clip_grad=1.0 with grad norm >> 1), so a wrong shared clipping
    factor, scaler binding, or non-atomic member step changes the update.
    Post-step comparison uses tight allclose rather than bitwise: the native
    path computes one total norm while the composite computes
    sqrt(sum(member_norm^2)) — mathematically equal partitions whose fp32
    reduction order differs at the last bit.
    """
    from megatron.core.mdp.optimizer import build_mdp_composite_optimizer
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    batch = _make_batch()
    input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw, slots = batch
    vision_config = _vision_config(dropout=0.0)

    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        # fp32 accumulation (production default): the two paths' DDP buffers
        # differ in size, NCCL may pick different reduction algorithms, and a
        # bf16 collective would round their sums differently by one ulp.
        grad_reduce_in_fp32=True,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam", lr=1e-2, bf16=True, clip_grad=1.0, weight_decay=0.0
    )

    # ---- native path: one DDP over vision+language, one optimizer ----
    native = _build_language_model(with_encoder=True, vision_config=vision_config)
    native_ddp = DistributedDataParallel(
        config=_language_config(), ddp_config=ddp_config, module=native
    )
    native_opt = get_megatron_optimizer(
        optimizer_config, [native_ddp], use_gloo_process_groups=False
    )
    native_ddp.zero_grad_buffer()
    losses = native_ddp(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    native_loss = (losses.float() * loss_mask).sum()
    native_loss.backward()
    native_ddp.finish_grad_sync()

    # ---- MDP path: DDP decoder + runtime encoder + composite optimizer ----
    mdp_model = _build_language_model(with_encoder=False, vision_config=vision_config)
    decoder_ddp = DistributedDataParallel(
        config=_language_config(), ddp_config=ddp_config, module=mdp_model
    )
    runtime, encoder_ddp = _build_runtime(
        vision_config, [_captured_microbatch(pixel_values, slots)]
    )
    encoder_ddp.module.load_state_dict(native.vision_model.state_dict())
    decoder_opt = get_megatron_optimizer(
        optimizer_config, [decoder_ddp], use_gloo_process_groups=False
    )
    encoder_opt = get_megatron_optimizer(
        optimizer_config, [encoder_ddp], use_gloo_process_groups=False
    )
    composite = build_mdp_composite_optimizer(decoder_opt, encoder_opt)

    decoder_ddp.zero_grad_buffer()
    replay = runtime.begin_iteration(iter(range(1)), num_microbatches=1, forward_only=False)
    next(replay[0])
    leaf = runtime.storage.get_leaf(0)
    losses = decoder_ddp(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        pixel_values=None,
        image_grid_thw=None,
        vision_embeddings=leaf,
    )
    mdp_loss = (losses.float() * loss_mask).sum()
    mdp_loss.backward()
    decoder_ddp.finish_grad_sync()
    # T=1: finalize divides the WORLD sum by 1, matching the native full-model
    # DDP reduction (8 identical lanes summed on both paths).
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    assert torch.equal(native_loss, mdp_loss)

    # ---- one atomic step on each path ----
    native_success, native_norm, _ = native_opt.step()
    mdp_success, mdp_norm, _ = composite.step()
    assert native_success and mdp_success
    # Clipping genuinely triggered, and both paths agree on the total norm.
    assert native_norm > optimizer_config.clip_grad
    assert abs(native_norm - mdp_norm) / native_norm < 1e-4, (native_norm, mdp_norm)

    # ---- updated weights match across paths ----
    # Parameters are STORED in bf16: a last-bit fp32 difference in the clip
    # coefficient can flip one bf16 rounding boundary (~4e-3 relative), so the
    # comparison tolerance is bf16-ulp scale. A wrong shared clipping factor,
    # scaler binding, or skipped member is orders of magnitude larger and is
    # additionally excluded by the 1e-4 grad-norm agreement above.
    native_language = dict(native.language_model.named_parameters())
    for name, param in mdp_model.language_model.named_parameters():
        assert torch.allclose(
            param.float(), native_language[name].float(), rtol=8e-3, atol=1e-6
        ), name
    native_vision = dict(native.vision_model.named_parameters())
    for name, param in encoder_ddp.module.named_parameters():
        assert torch.allclose(
            param.float(), native_vision[name].float(), rtol=8e-3, atol=1e-5
        ), name
