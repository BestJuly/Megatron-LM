# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Encoder DDP domain tests: WORLD reduction with prescale 1, ZeRO-1 optimizer,
parameter disjointness, 1/T_global finalization, and the Approach B zero-padded
vision FFN.

The zero-padding construction tests are plain arithmetic on a bare MLP and run
anywhere. The rest need a torchrun world::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_encoder_domain.py
"""

import os

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import (
    assert_parameter_disjointness,
    build_encoder_domain,
    build_encoder_pg_collection,
    finalize_encoder_grads,
    zero_pad_vision_mlp_channels,
)
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

# Gate per test rather than per module. Constructing a bare MLP needs no world:
# ``get_tensor_model_parallel_group_if_none`` hands it ``None`` and
# ``get_pg_size(None)`` is 1. So the zero-padding construction tests below stay in
# the CPU suite instead of skipping wherever a torchrun world is absent. (Running
# one is a different matter -- see the numerical-equivalence section.)
requires_distributed = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


def _tiny_config(**overrides):
    """The 1-layer, hidden=8 encoder config every test in this file builds on;
    the zero-padding tests override ``ffn_hidden_size`` and friends."""
    kwargs = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class _TinyEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        torch.manual_seed(42)  # identical replica weights on every rank
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


class _TinyAdapter:
    """The MdpModelAdapter surface build_encoder_domain consumes. The encoder
    class varies: a bare Linear for the domain tests, a real MLP (below) for the
    zero-padding ones."""

    payload_width = 8
    spatial_merge_size = 2

    def __init__(self, encoder_class=_TinyEncoder):
        self.encoder_class = encoder_class

    def build_encoder(self, model_config, *, pg_collection):
        return self.encoder_class(model_config)


def _build_encoder_pgs():
    """The encoder process groups for this file's pp=2 topology."""
    rank_map = build_rank_map(
        MdpRankSpec(
            world_size=torch.distributed.get_world_size(), tp=1, pp=2, cp=1, ep=1, encoder_cp=1
        )
    )
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    return build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)


def _build_domain(*, adapter=None, model_config=None, mdp_config=None, decoder_overlap=False):
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=decoder_overlap,
        overlap_param_gather=decoder_overlap,
        align_param_gather=decoder_overlap,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0
    )
    return build_encoder_domain(
        adapter=adapter or _TinyAdapter(),
        model_config=model_config or _tiny_config(),
        mdp_config=mdp_config or MdpConfig(enable=True),
        ddp_config=ddp_config,
        optimizer_config=optimizer_config,
        encoder_pgs=_build_encoder_pgs(),
        wrap_mixed_precision=False,
    )


@requires_distributed
def test_decoder_overlap_is_isolated_from_encoder_domain():
    encoder_config = _build_domain(decoder_overlap=True).encoder_ddp.ddp_config
    assert not encoder_config.overlap_grad_reduce
    assert not encoder_config.overlap_param_gather
    assert not encoder_config.align_param_gather


def _assert_identical_on_every_rank(tensor):
    """Every rank holds a full encoder replica, so anything the P5/P6 domain
    produces -- the reduced gradient, the stepped weight -- must agree bitwise."""
    gathered = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    for other in gathered[1:]:
        assert torch.equal(other, gathered[0])


def _build_allreduce_ddp():
    """A plain all-reduce DDP over the encoder groups, so the full summed
    gradient is observable on every rank (the ZeRO-1 path reduce-scatters and
    leaves each rank only its shard; its semantics are covered by the
    optimizer-step test)."""
    from megatron.core.distributed import DistributedDataParallel

    model_config = _tiny_config()
    return DistributedDataParallel(
        config=model_config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False, overlap_grad_reduce=False, overlap_param_gather=False
        ),
        module=_TinyEncoder(model_config).cuda(),
        pg_collection=_build_encoder_pgs(),
    )


@requires_distributed
@pytest.mark.parametrize(
    "num_tokens, divisor",
    [
        pytest.param(40.0, 40.0, id="scaled_by_token_count"),
        # clamp(T_global, min=1): a zero token count must leave the sum alone
        # rather than divide by zero.
        pytest.param(0.0, 1.0, id="zero_token_count_means_no_scaling"),
    ],
)
def test_world_sum_reduction_with_prescale_one_and_token_scaling(num_tokens, divisor):
    ddp = _build_allreduce_ddp()
    ddp.zero_grad_buffer()

    # Rank-distinct work: gradients must be summed over WORLD (prescale 1),
    # then scaled by 1/clamp(T_global, 1) exactly once.
    rank = torch.distributed.get_rank()
    ddp(torch.full((4, 8), float(rank + 1), device="cuda")).sum().backward()

    param = next(ddp.module.parameters())
    expected = param.main_grad.clone()
    torch.distributed.all_reduce(expected)  # WORLD sum, no pre-division
    expected /= divisor

    finalize_encoder_grads(ddp, globally_reduced_num_tokens=torch.tensor(num_tokens, device="cuda"))
    assert torch.allclose(param.main_grad, expected, rtol=1e-6, atol=1e-6)
    _assert_identical_on_every_rank(param.main_grad)


@requires_distributed
def test_optimizer_steps_identically_on_all_ranks():
    domain = _build_domain()
    ddp = domain.encoder_ddp
    ddp.zero_grad_buffer()
    ddp(torch.ones(2, 8, device="cuda")).sum().backward()
    finalize_encoder_grads(ddp, globally_reduced_num_tokens=torch.tensor(16.0, device="cuda"))
    success, _, _ = domain.encoder_optimizer.step()
    assert success
    _assert_identical_on_every_rank(next(ddp.module.parameters()).data)


@requires_distributed
def test_disjointness_assertion_catches_leaked_parameter():
    domain = _build_domain()

    leaky_chunk = torch.nn.Sequential(torch.nn.Linear(4, 4), domain.encoder_ddp.module)
    with pytest.raises(MdpConfigurationError, match="contains encoder parameters"):
        assert_parameter_disjointness(domain.encoder_ddp, [leaky_chunk])
    # And a clean chunk passes.
    assert_parameter_disjointness(domain.encoder_ddp, [torch.nn.Linear(4, 4).cuda()])


# ---------------------- zero_pad_vision_ffn (Approach B) ----------------------


class _TinyMLPEncoder(torch.nn.Module):
    """Wraps a real MLP so zero_pad_vision_mlp_channels' isinstance(module, MLP)
    walk finds a genuine linear_fc1/linear_fc2 pair, same as the vision
    encoder's per-layer MLP submodules."""

    def __init__(self, config, use_te=False):
        super().__init__()
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_submodules,
            get_mlp_module_spec,
        )
        from megatron.core.transformer.mlp import MLP
        from megatron.core.transformer.spec_utils import get_submodules

        self.config = config
        spec = get_mlp_module_spec(use_te=True) if use_te else get_gpt_layer_local_submodules().mlp
        self.mlp = MLP(config, submodules=get_submodules(spec))

    def forward(self, x):
        return self.mlp(x)


def _perturb_padding_channels(mlp, real_ffn_hidden_size):
    """Make the padding channels non-zero, so that whatever is expected to zero
    them next -- zero_pad_vision_mlp_channels at construction, or a checkpoint
    load -- has observable work to do (MCore initializes biases to zero)."""
    with torch.no_grad():
        mlp.linear_fc1.weight.data[real_ffn_hidden_size:, :].normal_()
        mlp.linear_fc1.bias.data[real_ffn_hidden_size:].normal_()
        mlp.linear_fc2.weight.data[:, real_ffn_hidden_size:].normal_()


def _assert_padding_is_zero(mlp, real_ffn_hidden_size, *, attribute="data"):
    """The three tensors zero_pad_vision_mlp_channels owns -- linear_fc1's padding
    rows and bias entries, linear_fc2's padding columns -- read from ``data`` (the
    invariant) or ``grad`` (what makes it self-stabilizing rather than merely true
    at construction time)."""
    fc1, fc2 = mlp.linear_fc1, mlp.linear_fc2
    for tensor in (
        getattr(fc1.weight, attribute)[real_ffn_hidden_size:, :],
        getattr(fc1.bias, attribute)[real_ffn_hidden_size:],
        getattr(fc2.weight, attribute)[:, real_ffn_hidden_size:],
    ):
        assert torch.equal(tensor, torch.zeros_like(tensor))


def _build_equivalent_pair(real_ffn_hidden_size, padded_ffn_hidden_size, **config_overrides):
    """An unpadded MLP and a zero-padded one that must behave identically.

    The padded encoder is assembled the way ``build_encoder_domain`` assembles the
    real one -- construct at the padded width, then call
    ``zero_pad_vision_mlp_channels`` once -- and is handed the unpadded MLP's real
    channels while keeping its own perturbed padding, so zeroing that padding is
    what has to make the two agree."""
    real = _TinyMLPEncoder(_tiny_config(ffn_hidden_size=real_ffn_hidden_size, **config_overrides))
    padded = _TinyMLPEncoder(
        _tiny_config(ffn_hidden_size=padded_ffn_hidden_size, **config_overrides)
    )
    _perturb_padding_channels(padded.mlp, real_ffn_hidden_size)
    with torch.no_grad():
        rows = slice(None, real_ffn_hidden_size)
        padded.mlp.linear_fc1.weight.data[rows, :].copy_(real.mlp.linear_fc1.weight.data)
        padded.mlp.linear_fc1.bias.data[rows].copy_(real.mlp.linear_fc1.bias.data)
        padded.mlp.linear_fc2.weight.data[:, rows].copy_(real.mlp.linear_fc2.weight.data)
        padded.mlp.linear_fc2.bias.data.copy_(real.mlp.linear_fc2.bias.data)
    zero_pad_vision_mlp_channels(padded, real_ffn_hidden_size=real_ffn_hidden_size)
    return real, padded


def test_zero_pad_vision_ffn_zeros_padding_channels_only():
    real_ffn, padded_ffn = 6, 8
    encoder = _TinyMLPEncoder(_tiny_config(ffn_hidden_size=padded_ffn))
    _perturb_padding_channels(encoder.mlp, real_ffn)
    real_fc1_before = encoder.mlp.linear_fc1.weight.data[:real_ffn, :].clone()

    zero_pad_vision_mlp_channels(encoder, real_ffn_hidden_size=real_ffn)

    assert torch.equal(encoder.mlp.linear_fc1.weight.data[:real_ffn, :], real_fc1_before)
    _assert_padding_is_zero(encoder.mlp, real_ffn)


@pytest.mark.parametrize(
    "built_ffn, real_ffn, config_overrides, expected_message",
    [
        # A no-op (no override in effect) must raise rather than silently do
        # nothing -- callers only invoke this when they mean to pad.
        pytest.param(6, 6, {}, "no vision MLP layer", id="equal_size"),
        pytest.param(6, 8, {}, "real_ffn_hidden_size", id="target_smaller_than_real"),
        # Under GLU linear_fc1 emits a concatenated gate/up pair, so the padding
        # slice would straddle both halves instead of the alignment channels.
        pytest.param(8, 6, {"gated_linear_unit": True}, "gated_linear_unit", id="a_gated_ffn"),
        # The padding stays inert only if the activation passes through the
        # origin; sigmoid(0)=0.5 would let it contribute to linear_fc2.
        pytest.param(
            8, 6, {"activation_func": torch.sigmoid}, r"activation\(0\) == 0", id="moves_zero"
        ),
    ],
)
def test_zero_pad_vision_ffn_rejects(built_ffn, real_ffn, config_overrides, expected_message):
    encoder = _TinyMLPEncoder(_tiny_config(ffn_hidden_size=built_ffn, **config_overrides))
    with pytest.raises(MdpConfigurationError, match=expected_message):
        zero_pad_vision_mlp_channels(encoder, real_ffn_hidden_size=real_ffn)


@requires_distributed
def test_build_encoder_domain_applies_zero_pad_vision_ffn():
    real_ffn, padded_ffn = 6, 8
    domain = _build_domain(
        adapter=_TinyAdapter(_TinyMLPEncoder),
        model_config=_tiny_config(ffn_hidden_size=real_ffn),
        mdp_config=MdpConfig(
            enable=True,
            zero_pad_vision_ffn=True,
            encoder_ffn_hidden_size=padded_ffn,
        ),
    )

    mlp = domain.encoder_ddp.module.mlp
    assert mlp.linear_fc1.weight.shape[0] == padded_ffn
    _assert_padding_is_zero(mlp, real_ffn)


# ---------------- zero_pad_vision_ffn: numerical equivalence ----------------
#
# Unlike the construction tests above, these need a world. Building an MLP
# without one works, but running it does not: ``RowParallelLinear.forward``
# all-reduces unconditionally, and ``_reduce`` asserts ``group is not None``
# before it short-circuits on size 1, so linear_fc2 cannot execute until a
# tensor-model-parallel group exists.


@requires_distributed
def test_zero_padded_vision_ffn_matches_unpadded_through_optimizer_steps():
    """The padded MLP must reproduce the unpadded one -- that is what makes
    --mdp-zero-pad-vision-ffn an alignment change rather than an architecture
    change -- and it must keep doing so under training, because the padding is
    inert rather than frozen: ``dL/da_j`` is zero for a padding channel because
    linear_fc2's column is zero, and ``dL/dW2[:, j]`` is zero because
    ``a_j = GELU(0) = 0``. Nothing masks these gradients, so the chain rule is
    the whole mechanism."""
    real_ffn = 8
    torch.manual_seed(0)
    real, padded = _build_equivalent_pair(real_ffn, 16)
    _assert_padding_is_zero(padded.mlp, real_ffn)

    x = torch.randn(5, 2, real.config.hidden_size)
    real_output, real_bias = real(x)
    padded_output, padded_bias = padded(x)

    # Not bitwise: the padding channels contribute exact zeros, but linear_fc2
    # reduces over K=8 in one model and K=16 in the other, and BLAS is free to
    # pick a different micro-kernel -- and so a different summation order for the
    # real channels -- per K. The agreement is exact in real arithmetic, so the
    # residual has to sit at fp32 rounding noise, orders of magnitude below the
    # difference any leaked padding channel would make.
    assert torch.allclose(real_output, padded_output, rtol=1e-6, atol=1e-8)
    # The bias is copied, not reduced, so this half stays exact.
    assert torch.equal(real_bias, padded_bias)

    real_output.float().pow(2).sum().backward()
    padded_output.float().pow(2).sum().backward()
    # Exact: these are products of an exactly-zero factor, whatever the GEMM does.
    _assert_padding_is_zero(padded.mlp, real_ffn, attribute="grad")
    # The real channels see the gradients the unpadded model produces, up to the
    # same fp32 rounding noise the forward carries.
    for padded_grad, real_grad in (
        (padded.mlp.linear_fc1.weight.grad[:real_ffn, :], real.mlp.linear_fc1.weight.grad),
        (padded.mlp.linear_fc2.weight.grad[:, :real_ffn], real.mlp.linear_fc2.weight.grad),
    ):
        assert torch.allclose(padded_grad, real_grad, rtol=1e-6, atol=1e-8)

    real_optimizer = torch.optim.Adam(real.parameters(), lr=0.1)
    padded_optimizer = torch.optim.Adam(padded.parameters(), lr=0.1)
    real_channels_before = padded.mlp.linear_fc1.weight.data[:real_ffn, :].clone()
    for _ in range(5):
        x = torch.randn(4, 2, real.config.hidden_size)
        for model, optimizer in ((real, real_optimizer), (padded, padded_optimizer)):
            optimizer.zero_grad()
            model(x)[0].float().pow(2).mean().backward()
            optimizer.step()

    _assert_padding_is_zero(padded.mlp, real_ffn)
    # The real channels did move, so the assertion above is not vacuous, and the
    # two models still agree after training -- to within the same fp32 rounding
    # noise, which five Adam steps propagate but do not amplify, since the update
    # is normalized by sqrt(v).
    assert not torch.equal(padded.mlp.linear_fc1.weight.data[:real_ffn, :], real_channels_before)
    x = torch.randn(4, 2, real.config.hidden_size)
    assert torch.allclose(real(x)[0], padded(x)[0], rtol=1e-5, atol=1e-7)


# MXFP8 block scaling needs Blackwell, and GB200 unit-test selection is marker-
# driven -- without the marker find_test_cases.py --ignores the whole file and
# run_ci_test.sh's `-m launch_on_gb200` deselects the test, so it would run in no
# CI configuration at all. Marked per test rather than per module (the convention
# in tests/unit_tests/transformer/moe/test_paged_stashing.py) because this is the
# only Blackwell-specific test in the file.
@pytest.mark.launch_on_gb200
@requires_cuda
def test_zero_padded_vision_ffn_holds_under_mxfp8():
    """MXFP8 is the recipe the padding exists for, so the invariant has to
    survive real hardware block quantization, not only the bf16 reference path
    above. Skipped where MXFP8 block scaling is unavailable."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 block scaling requires Blackwell (compute capability 10.x)")
    te = pytest.importorskip("transformer_engine.pytorch")
    te_recipe = pytest.importorskip("transformer_engine.common.recipe")

    # 48 % 32 == 16 is exactly the shape MXFP8's 32-element block quantizer
    # rejects ("MXFP8 requires tensor dims that are divisible by 32"); 64 is the
    # next multiple, the same relationship Qwen3.5-VL's 4304 -> 4320 has.
    real_ffn, padded_ffn = 48, 64
    torch.manual_seed(0)
    padded = _TinyMLPEncoder(
        _tiny_config(
            ffn_hidden_size=padded_ffn,
            hidden_size=32,
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        ),
        use_te=True,
    ).cuda()
    _perturb_padding_channels(padded.mlp, real_ffn)
    zero_pad_vision_mlp_channels(padded, real_ffn_hidden_size=real_ffn)
    _assert_padding_is_zero(padded.mlp, real_ffn)

    # The token count is the quantized GEMM's other block-aligned dimension.
    x = torch.randn(32, 1, 32, device="cuda", dtype=torch.bfloat16)
    with te.fp8_autocast(
        enabled=True,
        fp8_recipe=te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.HYBRID),
    ):
        intermediate, _ = padded.mlp.linear_fc1(x)
        output, _ = padded(x)

    # There is no unpadded MXFP8 forward to compare against: real_ffn is exactly the
    # width the block quantizer rejects, which is why Approach B exists. So pin the
    # mechanism the equivalence rests on instead. linear_fc1's padding rows are
    # exactly zero, so the quantized GEMM must emit exactly zero on those channels;
    # a degenerate all-zero block scale surfaces here as NaN rather than as the
    # plausible finite number a weaker `isfinite(output).all()` would accept. That
    # exact zero is also what keeps the padding out of MXFP8's *shared* block scale,
    # the failure mode bf16 cannot exhibit at all: linear_fc2 reduces over
    # ffn_hidden_size in 32-element blocks, so its second block spans channels
    # [32, 64) -- 16 real, 16 padding -- and only exactly-zero padding leaves the
    # real channels' amax, and therefore their quantization, untouched.
    padding_activation = intermediate[..., real_ffn:]
    assert torch.equal(padding_activation, torch.zeros_like(padding_activation))

    output.float().pow(2).sum().backward()
    _assert_padding_is_zero(padded.mlp, real_ffn, attribute="grad")
