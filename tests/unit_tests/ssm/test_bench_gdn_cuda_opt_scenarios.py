import ast
from pathlib import Path

BENCH = Path(__file__).with_name("bench_gdn_cuda_opt.py")


def _literal_assignment(name):
    tree = ast.parse(BENCH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


def test_optimized_scenarios_route_through_mcore_wrapper():
    scenarios = _literal_assignment("SCENARIOS")
    optimized = [
        "wy",
        "dv_dhu",
        "dhu",
        "dqkwg",
        "separate",
        "dv_dhu_dqkwg",
        "all_four",
        "fwd_h_wy_dv_dhu_dqkwg",
    ]

    for key in optimized:
        env = scenarios[key][1]
        assert env["MCORE_GDN_USE_OPT_WRAPPER"] == "1", key
        assert env["MCORE_GDN_OPT_BACKEND"] == "cuda", key
        assert not any(flag.startswith("FLA_CUTE_") for flag in env), key


def test_benchmark_does_not_require_patched_fla_sources():
    text = BENCH.read_text()

    assert "patched flash-linear-attention" not in text
    assert "FLA_DISPATCH_SOURCE" not in text


def test_benchmark_does_not_expose_dhu_dqkwg_wrapper_path():
    scenarios = _literal_assignment("SCENARIOS")
    flags = _literal_assignment("FLAGS")
    forbidden = {
        "MCORE_GDN_OPT_ENABLE_DHU_DQKWG",
        "FLA_CUTE_BWD_DHU_DQKWG",
        "FLA_CUTE_BWD_DHU_DQKWG_KERNEL",
        "FLA_CUTE_BWD_DHU_DQKWG_DIRECT",
    }

    assert forbidden.isdisjoint(flags)
    for key, (_label, env) in scenarios.items():
        assert forbidden.isdisjoint(env), key


def test_triton_baseline_pins_prefill_backend():
    scenarios = _literal_assignment("SCENARIOS")

    label, env = scenarios["baseline"]

    assert label == "Triton baseline"
    assert env == {"MCORE_GDN_PREFILL_BACKEND": "triton"}


def test_flashinfer_prefill_scenario_can_run_fwd_bwd():
    scenarios = _literal_assignment("SCENARIOS")
    flags = _literal_assignment("FLAGS")

    assert "MCORE_GDN_PREFILL_BACKEND" in flags
    assert "flashinfer_prefill" in scenarios
    label, env = scenarios["flashinfer_prefill"]
    assert label == "FlashInfer GDN prefill fwd + existing bwd"
    assert env == {"MCORE_GDN_PREFILL_BACKEND": "flashinfer"}

    label, env = scenarios["flashinfer_prefill_cuda_bwd"]
    assert label == "FlashInfer GDN prefill fwd + CUDA bwd"
    assert env["MCORE_GDN_PREFILL_BACKEND"] == "flashinfer"
    assert env["MCORE_GDN_USE_OPT_WRAPPER"] == "1"
    assert env["MCORE_GDN_OPT_BACKEND"] == "cuda"
    assert "flashinfer_prefill is forward-only" not in BENCH.read_text()


def test_benchmark_supports_square_sum_loss():
    text = BENCH.read_text()

    assert 'if loss == "square_sum"' in text
    assert 'choices=("sum", "square_mean", "square_sum")' in text
    assert 'default="square_sum"' in text


def test_cuda_opt_unit_defaults_to_square_sum_allopt():
    test_file = BENCH.with_name("test_gated_delta_net_cuda_opt.py")
    text = test_file.read_text()

    assert 'MCORE_GDN_UNIT_TEST_LOSS", "square_sum"' in text
    assert 'baseline,flashinfer_prefill_cuda_bwd' in text

