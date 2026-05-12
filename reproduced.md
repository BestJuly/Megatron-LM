# GDN CUDA Optimization Reproduction

This note covers the current GatedDeltaNet CUDA optimization test flow for
Megatron-LM on B200/H100. The optimized kernels are provided by
`third_party/mcore_gdn_opt`; FLA routes its gated-delta-rule calls through that
package.

## Install

Run these from the Megatron-LM repository root inside the GPU container.

```bash
git submodule update --init --recursive third_party/mcore_gdn_opt
pip install -e . --user --no-build-isolation

cd third_party/mcore_gdn_opt
./install_gdn_opt.sh
cd ../..

# FLA must contain the mcore_gdn_opt routing patch.
cd third_party/flash-linear-attention
pip install -e . --user --no-build-isolation
cd ../..
```

Do not use `PYTHONPATH` or ad-hoc `sys.modules` injection for these tests. The
submodules should be installed in editable mode.

## Runtime Flags

| Case | Flags |
|---|---|
| Triton baseline | unset all `FLA_CUTE_*` flags |
| `wy_bwd` only | `FLA_CUTE_WY_BWD=1` |
| `dhu` only | `FLA_CUTE_BWD_DHU=1` |
| `dqkwg` only | `FLA_CUTE_BWD_DQKWG=1` |
| fused backward | `FLA_CUTE_WY_BWD=1 FLA_CUTE_BWD_DHU_DQKWG=1` |
| all three separate | `FLA_CUTE_WY_BWD=1 FLA_CUTE_BWD_DHU=1 FLA_CUTE_BWD_DQKWG=1` |
| all four | `FLA_CUTE_FWD_H=1 CHUNK_DELTA_FWD_USE_BWD_PORT=1 FLA_CUTE_WY_BWD=1 FLA_CUTE_BWD_DHU=1 FLA_CUTE_BWD_DQKWG=1` |

## GDN-Only Direct Test

This bypasses the full GPT layer and measures a direct `GatedDeltaNet`
forward/backward. It checks output, input grad, and parameter grads against the
Triton baseline.

```bash
python -m tests.unit_tests.ssm.bench_gdn_cuda_opt \
  --dtype bf16 \
  --loss sum \
  --scenarios baseline,fused,separate,all_four \
  --warmup 5 --repeats 20 --rounds 3
```

Use `--loss square_mean` to reproduce the earlier loss used during debugging,
and add `--fail-on-accuracy` when the command should return non-zero on any
accuracy mismatch.

Latest B200 observation for `B=2,T=8192,H=64,D=128,bf16,loss=sum`:

| Scenario | Accuracy vs Triton | Mean us | Speedup |
|---|---:|---:|---:|
| Triton baseline | PASS | 16740.830 | 1.000x |
| CUDA `wy+dhu+dqkwg fused` | FAIL | 13706.724 | 1.221x |
| CUDA all three separate | FAIL | 13642.510 | 1.227x |
| CUDA all four | FAIL | 28586.833 | 0.586x |

With `--loss square_mean`, the fused case matched the Triton baseline in the
sanity run, but that does not prove correctness for arbitrary upstream
gradients. Treat `loss=sum` as the stricter correctness signal.

## E2E Pytest

This runs the Megatron GPT layer path and then prints the GDN benchmark table
from `tests/unit_tests/ssm/test_gated_delta_net.py`.

```bash
FLA_CUTE_WY_BWD=1 \
FLA_CUTE_BWD_DHU_DQKWG=1 \
MCORE_GDN_BENCH_ONLY=wy+dhu+dqkwg \
MCORE_GDN_BENCH_WARMUP=5 \
MCORE_GDN_BENCH_REPEATS=20 \
pytest -s tests/unit_tests/ssm/test_gated_delta_net.py::test_parallel_gated_delta_net_correctness -k bf16
```

To generate the five-scenario E2E table with NVTX labels:

```bash
MCORE_GDN_BENCH_FIVE_SCENARIOS=1 \
MCORE_GDN_BENCH_NVTX_MEASURE_ONLY=1 \
MCORE_GDN_BENCH_WARMUP=5 \
MCORE_GDN_BENCH_REPEATS=10 \
pytest -s tests/unit_tests/ssm/test_gated_delta_net.py::test_parallel_gated_delta_net_correctness -k bf16
```

Current B200 status: the fused E2E pytest fails the `input_grad` close check for
`loss=sum` by a small number of elements. The latest observed failure was
`2 / 2,097,152` mismatched elements with max absolute difference `0.00732421875`
for tolerance `0.005`.

## Nsight Systems

Use the E2E pytest command above under `nsys profile`. The benchmark emits NVTX
labels in this format:

```text
scenario/<index>_<scenario_name>/T=<sequence_length>/<dtype>/iter
```

Example:

```bash
MCORE_GDN_BENCH_FIVE_SCENARIOS=1 \
MCORE_GDN_BENCH_NVTX_MEASURE_ONLY=1 \
MCORE_GDN_BENCH_WARMUP=5 \
MCORE_GDN_BENCH_REPEATS=10 \
nsys profile -f true -o gdn_e2e_b200 \
  pytest -s tests/unit_tests/ssm/test_gated_delta_net.py::test_parallel_gated_delta_net_correctness -k bf16
```

Profiler outputs (`*.nsys-rep`, `*.sqlite`, `*.qdrep`) and local run directories
are ignored by `.gitignore`.
