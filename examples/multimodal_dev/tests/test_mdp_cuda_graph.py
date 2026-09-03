# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-layer (partial) CUDA graph support for the MDP decoder.

MDP's P4 runs the *unmodified* decoder schedule, so a per-layer CUDA graph over a
decoder submodule never interacts with the bridge: MDP only supplies the decoder's
embedding leaves (P3) and consumes their gradients (P5). What did block the feature
is that ``TECudaGraphHelper`` resolves graphable layers with
``get_attr_wrapped_model(chunk, 'decoder')``, which only unwraps through ``.module``
and therefore stopped at :class:`MultimodalModel` and captured nothing.

These tests run the SBHD path deliberately: graph replay needs a fixed static input
shape, and the THD side of that contract now comes from ``--thd-static-packing``,
which is an end-to-end property of the data path rather than something a unit test
can construct. What is covered here is everything that is *not* the packing
contract -- layer discovery through the multimodal wrapper, the loud failure when
discovery finds nothing, and per-layer capture/replay numerics on the decoder. The
packing contract itself is only exercised end-to-end on GPU, by a real
``--thd-static-packing`` run with ``--cuda-graph-impl``.

These tests need exactly 1 rank::

    torchrun --nproc-per-node 1 -m pytest -q \\
        examples/multimodal_dev/tests/test_mdp_cuda_graph.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.base import MultimodalModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import HAVE_TE_GRAPHS, TECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

if not HAVE_TE_GRAPHS:
    pytest.skip(
        "Per-layer CUDA graphs require a TransformerEngine build with "
        "make_graphed_callables().",
        allow_module_level=True,
    )

NUM_LAYERS = 2
HIDDEN = 128
HEADS = 4
VOCAB = 128
SEQ = 32
BATCH = 2
IMAGE_TOKEN_ID = 7
IMAGE_POSITIONS = (0, 1, 2, 3)
NUM_VISUAL_TOKENS = BATCH * len(IMAGE_POSITIONS)
DTYPE = torch.bfloat16


class _StubVisionEncoder(MegatronModule):
    """Minimal trainable stand-in for the real vision encoder."""

    def __init__(self, config, hidden_size, dtype=DTYPE):
        super().__init__(config=config)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, pixel_values, image_grid_thw):
        """Project pixel features to decoder-width embeddings."""
        return self.proj(pixel_values)


def _make_config(cuda_graph_impl="none", cuda_graph_modules=None):
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=4 * HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=HEADS,
        bf16=True,
        params_dtype=DTYPE,
        pipeline_dtype=DTYPE,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=(
            [] if cuda_graph_modules is None else list(cuda_graph_modules)
        ),
        cuda_graph_warmup_steps=0,
    )


class _ModelChunk(torch.nn.Module):
    """Stand-in for the DDP wrapper the training loop hands to the helper.

    ``TECudaGraphHelper`` unwraps ``.module`` to reach the decoder and calls
    ``zero_grad_buffer()`` on the chunk once capture finishes; DDP provides both.
    A real DDP is deliberately not used here: its bucket gradient sync lands
    inside the captured backward and invalidates the capture on a 1-rank DP
    group, which would test the wrapper rather than the graph.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """Delegate to the wrapped model."""
        return self.module(*args, **kwargs)

    def zero_grad_buffer(self):
        """DDP's gradient-buffer reset; plain ``.grad`` tensors here."""
        self.zero_grad(set_to_none=True)


def _build_model(config):
    """One model chunk, shaped the way the training loop hands it to the helper."""
    torch.manual_seed(1234)
    vision = _StubVisionEncoder(config, HIDDEN)
    model = MultimodalModel(
        language_config=config,
        language_spec=get_gpt_layer_with_transformer_engine_spec(),
        vision_encoder=vision,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        image_token_id=IMAGE_TOKEN_ID,
        position_embedding_type="rope",
        parallel_output=False,
        pre_process=True,
        post_process=True,
    )
    return _ModelChunk(model.cuda())


def _grad_norm(model):
    """L2 norm of every populated parameter gradient."""
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.float().norm(2).item() ** 2
    return total**0.5


def _make_batch(seed=1234):
    """Deterministic batch, identical for every model built in this module."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=g, device="cuda")
    input_ids[input_ids == IMAGE_TOKEN_ID] = (IMAGE_TOKEN_ID + 1) % VOCAB
    for pos in IMAGE_POSITIONS:
        input_ids[:, pos] = IMAGE_TOKEN_ID
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=g, device="cuda")
    loss_mask = torch.ones(BATCH, SEQ, device="cuda")
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(BATCH, -1).contiguous()
    pixel_values = torch.randn(
        NUM_VISUAL_TOKENS, HIDDEN, generator=g, device="cuda", dtype=DTYPE
    )
    image_grid_thw = torch.tensor([[1, 2, 2]] * BATCH, device="cuda")
    return input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw


def _forward_backward(model):
    """One forward + backward on the shared batch; returns (loss, grad norm)."""
    input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw = _make_batch()
    model.zero_grad_buffer()
    per_token_loss = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    flat = per_token_loss.float().view(-1)
    mask = loss_mask.float().view(-1)
    loss = (flat * mask).sum() / mask.sum().clamp(min=1)
    loss.backward()
    return loss.item(), _grad_norm(model)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_parallel_state():
    """TP=1/PP=1 model-parallel groups plus the microbatch calculator."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        pytest.skip("needs exactly 1 rank")
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    model_parallel_cuda_manual_seed(1234)
    init_num_microbatches_calculator(
        rank=0,
        global_batch_size=BATCH,
        micro_batch_size=BATCH,
        data_parallel_size=1,
        decrease_batch_size_if_needed=False,
    )
    yield
    destroy_num_microbatches_calculator()
    Utils.destroy_model_parallel()


def test_multimodal_model_surfaces_graph_discovery_attributes():
    """``TECudaGraphHelper`` reads these off the object it unwraps to."""
    model = _build_model(_make_config())

    model = model.module

    assert model.decoder is model.language_model.decoder
    assert model.rotary_pos_emb is model.language_model.rotary_pos_emb
    assert model.position_embedding_type == model.language_model.position_embedding_type
    # MTP is off here, so `mtp` must stay invisible exactly as on a bare GPTModel.
    assert not hasattr(model, "mtp")
    assert not hasattr(model.language_model, "mtp")


def test_te_helper_discovers_decoder_layers():
    """Regression: discovery used to stop at MultimodalModel and capture nothing."""
    config = _make_config("transformer_engine", [CudaGraphModule.mlp])
    model = _build_model(config)

    helper = TECudaGraphHelper(
        model=[model], config=config, seq_length=SEQ, micro_batch_size=BATCH
    )

    assert len(helper.flattened_callables) == NUM_LAYERS
    assert helper.flattened_callables == list(model.module.language_model.decoder.layers)
    assert helper.flattened_callables_is_mtp == [False] * NUM_LAYERS


def test_per_layer_cuda_graph_forward_backward_parity():
    """Loss and grad norm match between graphed and eager decoder MLPs."""
    eager_model = _build_model(_make_config())
    eager_loss, eager_grad_norm = _forward_backward(eager_model)

    config = _make_config("transformer_engine", [CudaGraphModule.mlp])
    graph_model = _build_model(config)
    # `_build_model` reseeds, so the two models start from identical weights; assert
    # it rather than trusting it, otherwise a numeric mismatch would be ambiguous.
    graph_model.load_state_dict(eager_model.state_dict())

    helper = TECudaGraphHelper(
        model=[graph_model], config=config, seq_length=SEQ, micro_batch_size=BATCH
    )
    helper.create_cudagraphs()
    assert helper.graphs_created(), "no per-layer graph was captured"
    assert all(
        layer.cuda_graphs for layer in graph_model.module.language_model.decoder.layers
    )

    graph_loss, graph_grad_norm = _forward_backward(graph_model)

    assert graph_loss == pytest.approx(eager_loss, rel=2e-2, abs=2e-2)
    assert graph_grad_norm == pytest.approx(eager_grad_norm, rel=5e-2)


def test_unreachable_decoder_fails_loudly():
    """A wrapper whose decoder the helper cannot reach must not capture zero layers.

    ``get_attr_wrapped_model`` unwraps only through ``.module``, so before the
    forwarding properties above existed the lookup raised, the helper swallowed
    its own RuntimeError at DEBUG level, and a run with CUDA graphs explicitly
    enabled silently trained in eager mode.
    """

    class _NoDecoder(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("not called")

    config = _make_config("transformer_engine", [CudaGraphModule.mlp])
    with pytest.raises(RuntimeError, match="forwarding property"):
        TECudaGraphHelper(
            model=[_ModelChunk(_NoDecoder())],
            config=config,
            seq_length=SEQ,
            micro_batch_size=BATCH,
        )
