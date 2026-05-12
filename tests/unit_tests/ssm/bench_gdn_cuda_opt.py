# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Direct GatedDeltaNet CUDA optimization correctness and performance runner.

This runner intentionally uses installed packages and normal project imports.
Install `mcore_gdn_opt` and FLA in editable mode before running it.
"""

import argparse
import os
import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


FLAGS = (
    "FLA_CUTE_FWD_H",
    "CHUNK_DELTA_FWD_USE_BWD_PORT",
    "FLA_CUTE_WY_BWD",
    "FLA_CUTE_BWD_DHU",
    "FLA_CUTE_BWD_DQKWG",
    "FLA_CUTE_BWD_DHU_DQKWG",
    "FLA_CUTE_BWD_DHU_DQKWG_KERNEL",
    "FLA_CUTE_BWD_DHU_DQKWG_DIRECT",
)


SCENARIOS = {
    "baseline": ("Triton baseline", {}),
    "wy": ("CUDA wy_bwd", {"FLA_CUTE_WY_BWD": "1"}),
    "dhu": ("CUDA delta_h", {"FLA_CUTE_BWD_DHU": "1"}),
    "dqkwg": ("CUDA dqkwg", {"FLA_CUTE_BWD_DQKWG": "1"}),
    "fused": (
        "CUDA wy+dhu+dqkwg fused",
        {"FLA_CUTE_WY_BWD": "1", "FLA_CUTE_BWD_DHU_DQKWG": "1"},
    ),
    "separate": (
        "CUDA all three separate",
        {"FLA_CUTE_WY_BWD": "1", "FLA_CUTE_BWD_DHU": "1", "FLA_CUTE_BWD_DQKWG": "1"},
    ),
    "all_four": (
        "CUDA all four",
        {
            "FLA_CUTE_FWD_H": "1",
            "CHUNK_DELTA_FWD_USE_BWD_PORT": "1",
            "FLA_CUTE_WY_BWD": "1",
            "FLA_CUTE_BWD_DHU": "1",
            "FLA_CUTE_BWD_DQKWG": "1",
        },
    ),
}


@dataclass
class AccuracyRow:
    name: str
    status: str
    output_max_abs: float
    input_grad_max_abs: float
    worst_param: str
    worst_param_max_abs: float


@dataclass
class PerfRow:
    name: str
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    speedup: float


def set_env(overrides):
    for flag in FLAGS:
        os.environ.pop(flag, None)
    os.environ.update(overrides)


def make_model(dtype):
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    cfg = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=64,
        linear_num_value_heads=64,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=64,
        activation_func=F.silu,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    submodules = get_experimental_attention_variant_module_spec(config=cfg).submodules
    return GatedDeltaNet(
        cfg,
        submodules=submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    ).cuda().to(dtype)


def zero_grads(model):
    model.zero_grad(set_to_none=True)


def compute_loss(output, loss):
    if loss == "sum":
        return output.float().sum()
    if loss == "square_mean":
        return output.float().square().mean()
    raise ValueError(f"unknown loss: {loss}")


def run_once(model, x, env, loss):
    set_env(env)
    zero_grads(model)
    inp = x.detach().clone().requires_grad_(True)
    out, _ = model(inp, attention_mask=None)
    compute_loss(out, loss).backward()
    torch.cuda.synchronize()
    grads = {
        name: param.grad.detach().float().clone().cpu()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return out.detach().float().clone().cpu(), inp.grad.detach().float().clone().cpu(), grads


def diff_max_abs(actual, expected):
    return float((actual - expected).abs().max().item())


def allclose(actual, expected, atol, rtol):
    return bool(torch.isfinite(actual).all().item()) and bool(
        torch.allclose(actual, expected, atol=atol, rtol=rtol)
    )


def check_accuracy(model, x, scenario_items, loss, atol, rtol):
    base_name, base_env = SCENARIOS["baseline"]
    base_out, base_grad, base_params = run_once(model, x, base_env, loss)
    rows = []
    for _, (name, env) in scenario_items:
        out, grad, params = run_once(model, x, env, loss)
        output_ok = allclose(out, base_out, atol, rtol)
        grad_ok = allclose(grad, base_grad, atol, rtol)
        worst_param = ""
        worst_param_abs = 0.0
        params_ok = True
        for param_name, expected in base_params.items():
            actual = params[param_name]
            params_ok = params_ok and allclose(actual, expected, atol, rtol)
            param_abs = diff_max_abs(actual, expected)
            if param_abs > worst_param_abs:
                worst_param = param_name
                worst_param_abs = param_abs
        rows.append(
            AccuracyRow(
                name=name,
                status="PASS" if output_ok and grad_ok and params_ok else "FAIL",
                output_max_abs=diff_max_abs(out, base_out),
                input_grad_max_abs=diff_max_abs(grad, base_grad),
                worst_param=worst_param,
                worst_param_max_abs=worst_param_abs,
            )
        )
    return rows


def fwd_bwd(model, x, env, loss):
    set_env(env)
    zero_grads(model)
    inp = x.detach().requires_grad_(True)
    out, _ = model(inp, attention_mask=None)
    compute_loss(out, loss).backward()


def benchmark(model, x, scenario_items, loss, warmup, repeats, rounds):
    rows = []
    baseline_us = None
    for _, (name, env) in scenario_items:
        for _ in range(warmup):
            fwd_bwd(model, x, env, loss)
        torch.cuda.synchronize()
        samples = []
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fwd_bwd(model, x, env, loss)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / repeats)
        mean_us = statistics.mean(samples)
        if baseline_us is None:
            baseline_us = mean_us
        rows.append(
            PerfRow(
                name=name,
                mean_us=mean_us,
                median_us=statistics.median(samples),
                min_us=min(samples),
                max_us=max(samples),
                speedup=baseline_us / mean_us,
            )
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--loss", choices=("sum", "square_mean"), default="sum")
    parser.add_argument("--scenarios", default="baseline,fused,separate,all_four")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--fail-on-accuracy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    keys = [key.strip() for key in args.scenarios.split(",") if key.strip()]
    if "baseline" not in keys:
        keys.insert(0, "baseline")
    unknown = [key for key in keys if key not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}; choices={sorted(SCENARIOS)}")
    scenario_items = [(key, SCENARIOS[key]) for key in keys]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    torch.manual_seed(123)
    set_env({})
    print(
        f"DEVICE {torch.cuda.get_device_name(0)} SHAPE B=2 T=8192 H=64 D=128 "
        f"dtype={args.dtype} loss={args.loss}"
    )
    try:
        model = make_model(dtype).eval()
        x = torch.randn(8192, 2, 128, device="cuda", dtype=dtype)
        accuracy_rows = check_accuracy(model, x, scenario_items, args.loss, args.atol, args.rtol)
        for row in accuracy_rows:
            print(
                f"ACCURACY name={row.name!r} status={row.status} "
                f"output_max_abs={row.output_max_abs:.9f} "
                f"input_grad_max_abs={row.input_grad_max_abs:.9f} "
                f"worst_param={row.worst_param} "
                f"worst_param_max_abs={row.worst_param_max_abs:.9f}"
            )
        perf_rows = benchmark(model, x, scenario_items, args.loss, args.warmup, args.repeats, args.rounds)
        for row in perf_rows:
            print(
                f"PERF name={row.name!r} mean_us={row.mean_us:.3f} "
                f"median_us={row.median_us:.3f} min_us={row.min_us:.3f} "
                f"max_us={row.max_us:.3f} speedup_vs_baseline={row.speedup:.3f}"
            )
        if args.fail_on_accuracy and any(row.status != "PASS" for row in accuracy_rows):
            raise SystemExit(1)
    finally:
        set_env({})
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
