# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

The TP2 variants (with and without sequence parallelism) run both paths with a
TP=2 decoder: four decoder replicas, eight encoder workers. The endpoint
broadcasts its leaf to its TP peer and alone returns the gradient, so the
WORLD-summed encoder gradient over T=4 replicas must again equal the native
single-model gradient; a twin injects the factor-of-TP error that peers
returning their (identical) gradients would cause and must be caught.

The CP variants (CP2, TP2xCP2, TP2xCP2 with SP; zigzag THD) run both paths with
a context-parallel decoder: every first-stage rank consumes only its CP shard of
the vision rows. The planner cuts every item into per-CP-rank slices that the
producers send straight to their owners, and owners return their slices'
gradients straight to the producers; per-rank losses and decoder
gradients must match bitwise and the encoder gradient to the same tolerance.
Twins drop one shard's gradient or duplicate every shard cp times and must be
caught. A native-only test checks the split-before-embed model path at CP2
against CP1.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q examples/multimodal_dev/tests/test_mdp_parity.py
"""

import math
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
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import ModalityBridge
from megatron.core.mdp.config import MdpConfig, apply_encoder_recompute_config
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
from megatron.core.mdp.window import pixel_capture_suppressed
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
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

    @pytest.fixture
    def _tp2_parallel():
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(1234)
        yield
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)

    @pytest.fixture
    def _tp_cp_parallel(request):
        tp, cp = request.param
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, context_parallel_size=cp
        )
        model_parallel_cuda_manual_seed(1234)
        yield tp, cp
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)


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


def _language_config(tp=1, sp=False, cp=1):
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
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        context_parallel_size=cp,
    )


_ENCODER_PGS = {}


def _encoder_pg_collection(tp, cp=1):
    """MDP's rank-local encoder groups, so both paths build the same replicated
    encoder: under TP (CP) the native encoder would otherwise shard over the
    decoder's TP group (run CP attention over its CP group) and its weights
    could not be transplanted."""
    if (tp, cp) not in _ENCODER_PGS:
        rank_map = build_rank_map(
            MdpRankSpec(
                world_size=torch.distributed.get_world_size(),
                tp=tp,
                pp=1,
                cp=cp,
                ep=1,
                encoder_cp=1,
            )
        )
        groups = _install_groups(rank_map, tp, cp)
        _ENCODER_PGS[(tp, cp)] = build_encoder_pg_collection(
            rank_map, encoder_cp=1, process_groups=groups
        )
    return _ENCODER_PGS[(tp, cp)]


def _install_groups(rank_map, tp, cp):
    return install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        decoder_tp_group=parallel_state.get_tensor_model_parallel_group() if tp > 1 else None,
        decoder_cp_group=parallel_state.get_context_parallel_group() if cp > 1 else None,
    )


def _thd_params():
    """One packed sample of SEQ rows (a multiple of 2 * cp * tp for every layout
    tested), so decoder CP runs on THD the way MDP requires."""
    cu = torch.tensor([0, SEQ], dtype=torch.int32, device="cuda")
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=SEQ,
        max_seqlen_kv=SEQ,
        total_tokens=SEQ,
    )


def _build_vision_encoder(config, pg_collection=None):
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
        pg_collection=pg_collection,
    )
    # Mirror the Float16Module wrapping depth: the Conv3d patch embed and the
    # learned position embedding are plain torch modules and stay fp32
    # otherwise, which TE rejects outside autocast.
    encoder = encoder.bfloat16().cuda()
    # A replicated encoder starts identical everywhere, as build_encoder_domain
    # guarantees in production: under TP the model-parallel RNG tracker that
    # initializes its linears differs per TP rank.
    for param in encoder.parameters():
        torch.distributed.broadcast(param.data, src=0)
    return encoder


def _build_language_model(with_encoder, vision_config, tp=1, sp=False, cp=1):
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    encoder = (
        _build_vision_encoder(
            vision_config,
            pg_collection=_encoder_pg_collection(tp, cp) if tp > 1 or cp > 1 else None,
        )
        if with_encoder
        else None
    )
    # Reset both RNG sources so the language weights are identical whether or
    # not an encoder was built first.
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    # Any decoder MCore builds can sit behind MultimodalModel; a plain GPTModel
    # keeps this test about MDP routing rather than a particular stack.
    language_config = _language_config(tp, sp, cp)
    language_model = GPTModel(
        config=language_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        position_embedding_type="rope",
        parallel_output=False,
        pre_process=True,
        post_process=True,
    )
    model = MultimodalModel(
        language_config=language_config,
        language_model=language_model,
        vision_encoder=encoder,
        image_token_id=IMAGE_TOKEN_ID,
        pre_process=True,
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
    shape_key_width = 3
    output_hidden_size = HIDDEN

    def __init__(self, vision_config, microbatches):
        self._vision_config = vision_config
        self._microbatches = list(microbatches)

    def get_batch(self, iterator):
        next(iterator)
        captured = self._microbatches.pop(0)
        if pixel_capture_suppressed():
            # Honor owner-sharded capture like the real collate path: under
            # TP only the TP source owns pixels.
            captured = CapturedMicrobatch(
                decoder_packed_seq_params=captured.decoder_packed_seq_params,
                vision_items=captured.vision_items,
                flat_pixel_payload=None,
                model_payload=captured.model_payload,
            )
        return captured

    def estimate_cost(self, item):
        return item.payload_rows

    def validate_item(self, item):
        t, h, w = item.shape_key
        assert item.payload_rows == t * h * w
        assert item.output_rows == t * (h // 2) * (w // 2)

    def build_encoder(self, model_config, *, pg_collection):
        return _build_vision_encoder(model_config, pg_collection=pg_collection)

    def encode(self, encoder, payload, layout):
        grid_thw = torch.tensor(
            [segment.shape_key for segment in layout.segments],
            dtype=torch.long,
            device=payload.device,
        )
        module = encoder.module if hasattr(encoder, "module") else encoder
        return module(payload, grid_thw)


def _captured_microbatch(pixel_values, slots, packed_seq_params=None):
    items = []
    payload_start = 0
    for ordinal, grid in enumerate(GRIDS):
        t, h, w = grid
        rows = t * h * w
        items.append(
            CapturedVisionItem(
                sample_id=0,
                image_ordinal=ordinal,
                shape_key=grid,
                payload_row_start=payload_start,
                payload_rows=rows,
                output_rows=t * (h // 2) * (w // 2),
                decoder_positions=tuple(slots[ordinal]),
            )
        )
        payload_start += rows
    return CapturedMicrobatch(
        decoder_packed_seq_params=packed_seq_params,
        vision_items=tuple(items),
        flat_pixel_payload=pixel_values,
        model_payload=MappingProxyType({}),
    )


def _build_runtime(vision_config, microbatches, *, mdp_config=None, tp=1, cp=1):
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=tp, pp=1, cp=cp, ep=1, encoder_cp=1)
    )
    view = rank_map.view(torch.distributed.get_rank())
    groups = _install_groups(rank_map, tp, cp)
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
    config = mdp_config or MdpConfig(enable=True)
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
            view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy(),
            shape_key_width=3, cp_size=cp, cp_partition_mode="zigzag",
        ),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        output_hidden_size=HIDDEN,
        params_dtype=DTYPE,
        num_vpp_chunks=1,
    )
    return runtime, encoder_ddp


def _run_native(vision_config, batch, tp=1, sp=False, cp=1):
    input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw, _ = batch
    model = _build_language_model(
        with_encoder=True, vision_config=vision_config, tp=tp, sp=sp, cp=cp
    )
    packed_seq_params = _thd_params() if cp > 1 else None
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
        packed_seq_params=packed_seq_params,
    )
    loss_mask = MultimodalModel.cp_split_loss_mask(loss_mask, packed_seq_params)
    total_loss = (losses.float() * loss_mask).sum()
    total_loss.backward()
    return model, total_loss


def _run_mdp(
    vision_config,
    batch,
    *,
    native_model,
    corrupt_leaf_grad=False,
    mdp_config=None,
    num_iterations=1,
    tp=1,
    sp=False,
    cp=1,
    leaf_grad_scale=None,
    drop_cp_shard_grad=False,
):
    input_ids, labels, loss_mask, position_ids, pixel_values, _grid, slots = batch
    model = _build_language_model(
        with_encoder=False, vision_config=vision_config, tp=tp, sp=sp, cp=cp
    )
    packed_seq_params = _thd_params() if cp > 1 else None
    # Identical language weights by construction (same seeds); assert one.
    native_emb = native_model.language_model.embedding.word_embeddings.weight
    mdp_emb = model.language_model.embedding.word_embeddings.weight
    assert torch.equal(native_emb, mdp_emb)

    config = mdp_config or MdpConfig(enable=True)
    num_windows = num_iterations + int(config.overlap_window_capture)
    runtime, encoder_ddp = _build_runtime(
        vision_config,
        [
            _captured_microbatch(pixel_values, slots, packed_seq_params)
            for _ in range(num_windows)
        ],
        mdp_config=config,
        tp=tp,
        cp=cp,
    )
    # Transplant the native encoder weights so both paths run the same model.
    encoder_ddp.module.load_state_dict(native_model.vision_model.state_dict())
    encoder_ddp.zero_grad_buffer()

    # Match the native path's RNG consumption for the stochastic encoder:
    # both paths run encoder forward from the same seeded stream.
    torch.manual_seed(99)
    model_parallel_cuda_manual_seed(99)
    data_iterator = iter(range(num_windows))
    total_loss = None
    for _ in range(num_iterations):
        replay = runtime.begin_iteration(
            data_iterator, num_microbatches=1, forward_only=False
        )
        next(replay[0])
        leaf = runtime.storage.get_leaf(0)
        view = runtime.rank_view
        if cp == 1:
            assert leaf is not None and leaf.shape[0] == sum(len(s) for s in slots)
        else:
            # This rank's CP shard: exactly the image slots of its CP slice.
            local_ids = model._cp_split_sequence(input_ids, 1, packed_seq_params)
            local_rows = int((local_ids == IMAGE_TOKEN_ID).sum())
            assert runtime.expects_leaf(0) == (local_rows > 0)
            assert (0 if leaf is None else leaf.shape[0]) == local_rows
        # Only a shard owner's copy is trainable; a TP peer holds a bitwise copy.
        assert leaf is None or leaf.requires_grad == view.is_shard_owner
        if tp > 1 and leaf is not None:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            copies = [torch.empty_like(leaf) for _ in range(tp)]
            torch.distributed.all_gather(copies, leaf.detach(), group=tp_group)
            assert all(torch.equal(copies[0], copy) for copy in copies[1:])

        losses = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            pixel_values=None,
            image_grid_thw=None,
            vision_embeddings=leaf,
            packed_seq_params=packed_seq_params,
        )
        local_loss_mask = MultimodalModel.cp_split_loss_mask(loss_mask, packed_seq_params)
        total_loss = (losses.float() * local_loss_mask).sum()
        total_loss.backward()
        if corrupt_leaf_grad:
            leaf.grad.mul_(-1.0)
        if leaf_grad_scale is not None and leaf is not None and leaf.grad is not None:
            leaf.grad.mul_(leaf_grad_scale)
        if drop_cp_shard_grad and leaf is not None and leaf.grad is not None:
            # A non-endpoint shard owner whose gradient never reaches the encoder.
            if view.is_shard_owner and not view.is_endpoint:
                leaf.grad.zero_()

        rng_before_p5 = torch.cuda.get_rng_state()
        tracker_before_p5 = {
            name: state.clone()
            for name, state in get_cuda_rng_tracker().get_states().items()
        }
        # One decoder replica per TP x CP group: each replica's encoder gradient is
        # the sum of its disjoint CP shards, so T counts DP replicas only.
        tokens = torch.tensor(
            float(torch.distributed.get_world_size() // (tp * cp)), device="cuda"
        )
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


def _native_encoder_grads(native_model):
    """The native single-model encoder gradient. Under decoder CP every CP rank
    encodes all pixels but backpropagates only its own rows, so the model's
    gradient is the sum over the CP group."""
    grads = {
        name: param.grad.float()
        for name, param in native_model.vision_model.named_parameters()
        if param.grad is not None
    }
    if parallel_state.get_context_parallel_world_size() > 1:
        for grad in grads.values():
            torch.distributed.all_reduce(grad, group=parallel_state.get_context_parallel_group())
    return grads


def _encoder_atol(reference, atol=1e-5):
    """Absolute tolerance for one encoder gradient tensor (see the caller)."""
    largest = float(reference.abs().max())
    tol = max(atol, 2e-3 * largest)
    cp = parallel_state.get_context_parallel_world_size()
    if cp > 1 and largest > 0:
        # Half a bf16 ulp of the largest magnitude per CP rank's rounded partial.
        ulp = torch.finfo(torch.bfloat16).eps * 2.0 ** math.floor(math.log2(largest))
        tol = max(tol, 0.5 * cp * ulp)
    return tol


def _assert_encoder_grads_match(encoder_ddp, native_model, rtol=8e-3, atol=1e-5):
    native_encoder_grads = _native_encoder_grads(native_model)
    for name, param in encoder_ddp.module.named_parameters():
        grad = param.main_grad.float()
        reference = native_encoder_grads[name]
        # bf16 wgrads accumulate in different orders on the two paths, so an
        # element can be off by one ulp of the tensor's LARGEST accumuland;
        # scale the absolute tolerance accordingly. Under decoder CP the native
        # reference is itself a sum of cp bf16-rounded per-rank partials (the
        # MDP side accumulates in fp32), each off by up to half an ulp. A dropped
        # or duplicated shard is off by a whole shard's contribution and fails.
        atol_eff = _encoder_atol(reference, atol)
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


def _decoder_grads_bitwise(native_model, mdp_model):
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


def _encoder_grad_mismatches(encoder_ddp, native_model):
    native_encoder_grads = _native_encoder_grads(native_model)
    return sum(
        0
        if torch.allclose(
            param.main_grad.float(),
            native_encoder_grads[name],
            rtol=8e-3,
            atol=_encoder_atol(native_encoder_grads[name]),
        )
        else 1
        for name, param in encoder_ddp.module.named_parameters()
    )


@pytest.mark.parametrize("sequence_parallel", [False, True], ids=["tp2", "tp2-sp"])
def test_g1_tp2_mdp_matches_native_pixel_path(_tp2_parallel, sequence_parallel):
    batch = _make_batch()
    native_model, native_loss = _seeded(
        _run_native, _vision_config(dropout=0.0), batch, tp=2, sp=sequence_parallel
    )
    mdp_model, mdp_loss, encoder_ddp = _run_mdp(
        _vision_config(dropout=0.0),
        batch,
        native_model=native_model,
        tp=2,
        sp=sequence_parallel,
    )
    assert torch.equal(native_loss, mdp_loss), (float(native_loss), float(mdp_loss))
    _decoder_grads_bitwise(native_model, mdp_model)
    # WORLD sum over 8 producers of the 4 endpoints' gradients / T(=4) is the
    # native single-model gradient; the peers contributed nothing.
    _assert_encoder_grads_match(encoder_ddp, native_model)


def test_g1_tp2_detects_encoder_grads_scaled_by_tp(_tp2_parallel):
    # If both TP ranks of a replica returned their (identical) leaf gradient,
    # the WORLD sum would see it TP times. Inject exactly that factor on the
    # endpoint: the tolerance used above must reject it.
    batch = _make_batch()
    native_model, _ = _seeded(_run_native, _vision_config(dropout=0.0), batch, tp=2)
    _, _, encoder_ddp = _run_mdp(
        _vision_config(dropout=0.0),
        batch,
        native_model=native_model,
        tp=2,
        leaf_grad_scale=2.0,
    )
    assert _encoder_grad_mismatches(encoder_ddp, native_model) > 0, (
        "a factor-of-TP encoder gradient must fail the parity tolerance"
    )


_CP_LAYOUTS = [(1, 2, False), (2, 2, False), (2, 2, True)]
_CP_IDS = ["cp2", "tp2-cp2", "tp2-cp2-sp"]


@pytest.mark.parametrize(
    "_tp_cp_parallel,sequence_parallel",
    [((tp, cp), sp) for tp, cp, sp in _CP_LAYOUTS],
    ids=_CP_IDS,
    indirect=["_tp_cp_parallel"],
)
def test_g1_cp_mdp_matches_native_pixel_path(_tp_cp_parallel, sequence_parallel):
    # Decoder CP (zigzag THD): every PP0 rank consumes its own CP shard of the
    # vision rows. Producers send each slice straight to its owner and owners
    # return their gradients directly; both paths run the identical rank-local
    # scatter, so the per-rank loss and decoder gradients match bitwise.
    tp, cp = _tp_cp_parallel
    batch = _make_batch()
    native_model, native_loss = _seeded(
        _run_native, _vision_config(dropout=0.0), batch, tp=tp, sp=sequence_parallel, cp=cp
    )
    mdp_model, mdp_loss, encoder_ddp = _run_mdp(
        _vision_config(dropout=0.0),
        batch,
        native_model=native_model,
        tp=tp,
        sp=sequence_parallel,
        cp=cp,
    )
    assert torch.equal(native_loss, mdp_loss), (float(native_loss), float(mdp_loss))
    _decoder_grads_bitwise(native_model, mdp_model)
    # WORLD sum of the gathered shard gradients / T(= DP replicas) is the native
    # single-model gradient (itself the CP-sum of the native ranks' gradients).
    _assert_encoder_grads_match(encoder_ddp, native_model)


@pytest.mark.parametrize("_tp_cp_parallel", [(1, 2)], ids=["cp2"], indirect=True)
@pytest.mark.parametrize("fault", ["dropped-shard", "duplicated-x-cp"])
def test_g1_cp_detects_misrouted_shard_gradients(_tp_cp_parallel, fault):
    # Two ways a CP shard gradient can go wrong on its way back to the encoder:
    # a non-endpoint shard is lost, or every shard arrives cp times (what CP
    # ranks returning full, reduced gradients would produce). Both must fail.
    tp, cp = _tp_cp_parallel
    batch = _make_batch()
    native_model, _ = _seeded(_run_native, _vision_config(dropout=0.0), batch, tp=tp, cp=cp)
    _, _, encoder_ddp = _run_mdp(
        _vision_config(dropout=0.0),
        batch,
        native_model=native_model,
        tp=tp,
        cp=cp,
        drop_cp_shard_grad=fault == "dropped-shard",
        leaf_grad_scale=float(cp) if fault == "duplicated-x-cp" else None,
    )
    mismatches = torch.tensor(
        _encoder_grad_mismatches(encoder_ddp, native_model), device="cuda"
    )
    torch.distributed.all_reduce(mismatches)
    assert int(mismatches) > 0, f"{fault}: a misrouted CP shard gradient must fail parity"


def _global_cp_loss(loss):
    loss = loss.detach().clone()
    if parallel_state.get_context_parallel_world_size() > 1:
        torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
    return loss


def _cp_summed_decoder_grads(model):
    grads = {}
    for name, param in model.language_model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.float().clone()
        if parallel_state.get_context_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                grad, group=parallel_state.get_context_parallel_group()
            )
        grads[name] = grad
    return grads


def test_g1_native_cp2_matches_cp1():
    # The split-before-embed model change on its own (no MDP): the native pixel
    # path at CP2 reproduces CP1 -- global loss, CP-summed decoder gradients and
    # CP-summed encoder gradients -- to bf16 reduction-order tolerance.
    batch = _make_batch()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    try:
        ref_model, ref_loss = _seeded(
            _run_native, _vision_config(dropout=0.0), batch, cp=1
        )
        ref_decoder = _cp_summed_decoder_grads(ref_model)
        ref_encoder = _native_encoder_grads(ref_model)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
        model_parallel_cuda_manual_seed(1234)
        cp_model, cp_loss = _seeded(_run_native, _vision_config(dropout=0.0), batch, cp=2)
        loss = _global_cp_loss(cp_loss)
        assert torch.allclose(loss, ref_loss, rtol=2e-3), (float(loss), float(ref_loss))
        for name, grad in _cp_summed_decoder_grads(cp_model).items():
            ref = ref_decoder[name]
            atol = max(1e-5, 2e-2 * float(ref.abs().max()))
            assert torch.allclose(grad, ref, rtol=2e-2, atol=atol), (
                name,
                float((grad - ref).abs().max()),
            )
        for name, grad in _native_encoder_grads(cp_model).items():
            ref = ref_encoder[name]
            atol = max(1e-5, 2e-2 * float(ref.abs().max()))
            assert torch.allclose(grad, ref, rtol=2e-2, atol=atol), (
                name,
                float((grad - ref).abs().max()),
            )
    finally:
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)


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
    "mdp_config",
    [
        MdpConfig(enable=True, encoder_recompute_granularity="selective"),
        MdpConfig(
            enable=True,
            encoder_recompute_granularity="full",
            encoder_recompute_method="uniform",
            encoder_recompute_num_layers=1,
        ),
    ],
    ids=["selective", "full"],
)
def test_g1_recompute_modes_match_reference(mdp_config):
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

    mode_config = apply_encoder_recompute_config(
        _vision_config(dropout=0.1), mdp_config
    )
    _, mode_loss, mode_ddp = _run_mdp(
        mode_config, batch, native_model=native_model, mdp_config=mdp_config
    )
    assert torch.equal(reference_loss, mode_loss), (
        float(reference_loss),
        float(mode_loss),
    )
    for name, param in mode_ddp.module.named_parameters():
        assert torch.equal(param.main_grad.float(), reference_grads[name]), name


@pytest.mark.parametrize(
    "overlap_window_capture,num_iterations",
    [(False, 1), (True, 2)],
    ids=["sync-window-capture", "overlap-window-capture"],
)
def test_g1_whole_encoder_recompute_matches_reference(
    overlap_window_capture, num_iterations
):
    """The Design-Doc P2-no-grad/P5-full-replay path must preserve stochastic
    encoder outputs and the ambient RNG stream bitwise.

    ``_run_mdp`` already asserts the P5 RNG fork. The overlap case runs two
    iterations so the second one consumes the window captured concurrently
    with the first decoder schedule. Using dropout and forcing two encoder
    chunks ensures that neither a missing RNG restore nor a
    single-recipe-for-all-chunks implementation can pass. Encoder gradients
    use the same BF16-ULP tolerance as native-vs-MDP parity: the retained-graph
    reference performs one multi-tensor backward, while complete replay must
    backward and release one chunk at a time, so weight-gradient accumulation
    order is intentionally different.
    """
    batch = _make_batch()
    native_model, _ = _seeded(_run_native, _vision_config(dropout=0.0), batch)

    reference_config = _vision_config(dropout=0.1)
    reference_mdp_config = MdpConfig(
        enable=True,
        encoder_max_payload_rows=64,
        overlap_window_capture=overlap_window_capture,
    )
    _, reference_loss, reference_ddp = _run_mdp(
        reference_config,
        batch,
        native_model=native_model,
        mdp_config=reference_mdp_config,
        num_iterations=num_iterations,
    )
    reference_grads = {
        name: param.main_grad.float().clone()
        for name, param in reference_ddp.module.named_parameters()
    }

    _, mode_loss, mode_ddp = _run_mdp(
        _vision_config(dropout=0.1),
        batch,
        native_model=native_model,
        mdp_config=MdpConfig(
            enable=True,
            encoder_max_payload_rows=64,
            encoder_recompute_granularity="whole",
            overlap_window_capture=overlap_window_capture,
        ),
        num_iterations=num_iterations,
    )
    assert torch.equal(reference_loss, mode_loss), (
        float(reference_loss),
        float(mode_loss),
    )
    for name, param in mode_ddp.module.named_parameters():
        grad = param.main_grad.float()
        reference = reference_grads[name]
        atol = max(1e-5, 2e-3 * float(reference.abs().max()))
        assert torch.allclose(grad, reference, rtol=8e-3, atol=atol), (
            name,
            float((grad - reference).abs().max()),
        )


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
