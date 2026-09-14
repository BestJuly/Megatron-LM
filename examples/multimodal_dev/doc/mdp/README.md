# MDP 性能测试 — Qwen3.5-VL rebench milestone

本分支从 Li 的 [`lit/qwen35-397b-rebench`](https://github.com/BestJuly/Megatron-LM/tree/lit/qwen35-397b-rebench)
checkout 出来，再 cherry-pick [`lit/mdp_fast_pass`](https://github.com/BestJuly/Megatron-LM/tree/lit/mdp_fast_pass)
的 MDP 相关代码形成。

本目录沿用 fast-pass 的 **测试页 + 完整 recipe** 组织方式，记录集成后的已验证结果。
测试完成于 2026-09-11，文档更新于 2026-09-14。

## 当前 milestone

| 场景 | 实测结果 | 说明与复现 |
|---|---|---|
| 35B-A3B，16K 变长 THD，8× GB300，BF16 | **约 474 TFLOP/s/GPU** | [35B rebench 更新](benchmarks/qwen35vl_35b_a3b_16k.md#2026-09-13-rebench-更新) |
| 397B-A17B，单条 4096，128× GB300，decoder MXFP8 + encoder BF16 | **625.35 TFLOP/s/GPU**；保留约 **96.2%** 的 Decoder-only 吞吐 | [单条 4096 对照](benchmarks/qwen35vl_397b_a17b_4k.md#单条-4096加上-vl-的整体开销) |
| 397B-A17B，4K 变长 THD，128× GB300 | Native **310.55** → MDP **501.05 TFLOP/s/GPU**，约 **+61.3%** | [变长 4K 对照](benchmarks/qwen35vl_397b_a17b_4k.md#变长-4kmdp-与-native-vl) |

单条 4096 对照的 step time 增加约 **4.5%**，说明在该视觉负载下使用 MDP
加入 encoder 的整体开销有限，**不是完全零开销，也不是 MDP 调度本身的独立开销**。
变长 4K 只比较 Native VL 与 MDP；不拿它与单条 4096 的 Decoder-only 横比。

## 从这里开始

1. 阅读 [rebench 环境与运行方式](benchmarks/rebench_environment.md)，准备同版本镜像和正确 CPU/NUMA 绑定。
2. 在对应测试页选择 cell；相邻 `.sh` 包含完整环境变量与训练参数，`.yaml` 是完整 recipe。
   无需原作者的 toolkit 目录，也无需重新拼接外部 model/data YAML。
3. 数据 JSON 已随仓库提供；在已分配的 GPU 容器里运行，不在登录节点执行训练。
4. 记录实际 `git rev-parse HEAD`。表中的实测 SHA 属于当前分支历史的不同验证点，
   不应把历史结果改写为在新的文档 HEAD 上重新测得。

```text
doc/mdp/
├── README.md
├── commit.md                             fast-pass 历史 + 本分支集成说明
├── benchmarks/
│   ├── qwen35vl_35b_a3b_16k.md           历史 35B 结果 + 474 milestone
│   ├── qwen35vl_397b_a17b_4k.md          单条 4096 与变长 4K，分开比较
│   ├── rebench_environment.md            统一环境、运行与测量约定
│   └── mdp_feature_ablations.md          历史 fast-pass 消融
└── recipes/
    ├── data/                             定长 / 变长 mock 配置
    ├── qwen35_vl_35b_a3b/                历史 recipe + rebench YAML / SH
    ├── qwen35_vl_397b_a17b/               4 个完整 rebench cell
    └── proxy/                            历史 fast-pass proxy recipe
```

[分支来源与新增修改](commit.md#2026-09-13rebench-分支说明)。
以下历史快照来自 `lit/mdp_fast_pass@5885d65a9`；其旧镜像、旧 SHA、测量窗口和
支持边界只属于各自记录，不覆盖上面的 rebench milestone。
旧页的近似计算和命令日志分隔符见 [35B 页的历史勘误](benchmarks/qwen35vl_35b_a3b_16k.md)；
复现当前 milestone 请使用上方新 recipe，而不是直接执行历史命令快照。

<details>
<summary>原 fast-pass README（历史快照，原文保留）</summary>

# MDP 性能测试 — Qwen3.5-VL 35B-A3B on GB300

本目录记录 **MDP（Modality Decoupled Parallelism，模态解耦并行）** 在
`multimodal_dev` 训练入口上的实测结果，以及产出这些结果的完整配置。

```
doc/mdp/
├── README.md                             本文件 —— 测量约定与注意事项
├── commit.md                             提交历史 —— 每个 commit 做了什么
├── benchmarks/
│   ├── qwen35vl_35b_a3b_16k.md           全量模型，3 种拓扑 × {native, MDP}
│   └── mdp_feature_ablations.md          CUDA graph 与 window overlap 各值多少
└── recipes/
    ├── qwen35_vl_35b_a3b/                全量模型的 7 个 recipe
    └── proxy/                            20 层代理模型的 4 个 recipe
```

每个测试页都附有对应 cell 的完整 `pretrain_multimodal.py` 命令行，因此无论是否
使用 launcher，配置都是可读、可复现的。

---

## 测试对象

| | |
|---|---|
| 模型 | Qwen3.5-VL 35B-A3B —— 40 层 decoder、256 专家、top-8、`moe_ffn_hidden_size` 512、27 层 vision、hidden 2048，`experimental_attention_variant=gated_delta_net` 且 `linear_attention_freq 4` |
| 代理模型 | 同一模型，`--num-layers 20 --num-experts 128 --vision-num-layers 13`（`--model-variant 35b_a3b_light`） |
| 硬件 | 全量模型 8× GB300（2 个 tray，同一 NVL72 机架）；代理模型 4× GB300（1 个 tray） |
| 精度 | BF16，precision-aware 分布式优化器，Adam 一阶/二阶矩为 bf16 |
| 序列 | THD packing，16384 token 预算，greedy + static packing，`--thd-max-packed-sequences 32` |
| 数据 | `--dataset-provider mdp_mock`，lognormal 分布 `min 512 / max 4096 / mean 2048 / sigma 1.1` |
| 迭代数 | 20 次（CUDA graph 消融为 40 次）；取第 6 次及以后的中位数 |

代码版本：**`ab9bcd537`**。该 commit 包含 MDP 主体、greedy/static THD packing、
GDN 的 per-layer CUDA graph、完整 vision encoder 重算，以及 vision
`head_dim` 72 → 80 的 padding 修复。

## 数据口径

- **TFLOP/s** 取 Megatron 日志中的 `throughput per GPU (TFLOP/s/GPU)`，第 6 次
  迭代及以后的中位数。第 1–3 次迭代是编译与 CUDA graph capture，比稳态慢
  10–40 倍，一律排除。
- **devUsed** 取 `total device memory used`（NVML 口径），**在各 pipeline stage
  中取最大值** —— 这才是真正需要装进显存的数字。`maxAlloc` 是 PyTorch 的
  `max allocated`，总是更小。
- 各组对比均在**完全相同的 instance 数量**下进行（相同的 micro-batch 数、层数、
  集合通信次数），因此是"等工作量"对比，而不是"等时间"对比。

---

## 三个会静默污染测量结果的陷阱

以下三点在本轮工作中都真实产出过错误数据，之后才被定位。

### 1. dataset provider 必须是 `mdp_mock`，不能是 `mock`

`examples/multimodal_dev/data/mock.py` 生成的是**长度恰好等于
`--total-seq-length` 的定长样本**，每条带一张图，并且**完全不读取
`--mdp-mock-dataset-config-json`**。只有 `data/mdp_mock.py` 会消费这个参数。

如果在 `--dataset-provider mock` 下传入 lognormal 分布配置，程序会静默接受但
毫无效果：此时每个 greedy bin 里恰好只有一条满长序列，greedy packing 退化为
空操作，core attention 的 `sum(L^2)` 会是真实 packing 的约 **100 倍**。

请检查解析后的参数 dump，而不是 recipe 本身：`dataset_provider`、
`total_seq_length`、`image_seq_length`、`image_size`。

### 2. 必须显式给 SLURM step 分配 CPU

「每节点一个 task、内部用 torchrun fork 出 `nproc_per_node` 个进程」这种写法，
会继承 srun 默认的 `--cpus-per-task=1`。于是所有 rank、dataloader worker、
NCCL progress 线程**共享一个 CPU 核**。实测比同一 cell 拿到完整 CPU 集合
**慢 3.9 倍**，而日志里没有任何提示。

`--exclusive` 解决不了这个问题，`--cpu-bind=none` 也不行——前者只给 job 分配
整节点，后者只是去掉绑定 mask。**step 必须自己把 CPU 要过来**：
`--cpus-per-task=<每节点核数>`。

任何一次运行都可以用一行 grep 自查 —— PyTorch 会打印自己的 CPU mask：

```
This DataLoader will create 2 worker processes in total. Our suggested max
number of worker in current system is <N>
```

`<N>` 就是 `len(os.sched_getaffinity(0))`。**`N = 1` 说明你测的是启动器，不是
模型。** 正常的运行根本不会打印这条 warning。

### 3. window overlap 与 CUDA graph 的组合默认是被拒绝的

`megatron/core/mdp/config.py` 会拒绝 `--mdp-overlap-window-capture` 与
per-layer CUDA graph 同时开启。同时使用两者的 recipe 设置了
`MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1` 来解除该限制。

**这是实验开关，不是受支持的配置。** 具体结论与边界见
`benchmarks/mdp_feature_ablations.md`。

---

## 如何运行

目录下的 YAML 是 mcore-devtoolkit 的 launch recipe。与站点相关的取值已替换为
`<占位符>`，并在每个文件开头列明。如果不使用该工具链，直接取各测试页中给出的
`pretrain_multimodal.py` 命令行，自行配 `torchrun` / `srun` 即可，两者等价。

所有 cell 都需要的环境变量：

```bash
export NVTE_FUSED_ATTN=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=8
export PYTHONPATH=<FLA_PREFIX>:<NVRX_PREFIX>          # fla >= 0.6.0，nvidia-resiliency-ext 0.6.0
export NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=128       # HybridEP 128-token 分块
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=128
```

开启 CUDA graph 时还需要：

```bash
export FLA_DISABLE_TENSOR_CACHE=1   # fla 按 tensor 身份对 prepare_chunk_offsets 做记忆化，
                                    # 而 static THD 路径在 warmup/capture/replay 之间复用同一个
                                    # cu_seqlens 对象，capture 时若命中缓存会把 chunk_offsets 冻住
export NCCL_GRAPH_REGISTER=0        # 与 expandable_segments 共存时 Megatron 会断言此项
```

---

## 2026-09-10 更新

35B-A3B @ 16k 的 MDP PP2/EP4 配置经四项优化后，当前最佳实测为
**424.0 TFLOP/s/GPU，6,597.7 ms/iter**（8× GB300，BF16）。
成功优化过程与最新配置见测试页末尾的
[增量更新](benchmarks/qwen35vl_35b_a3b_16k.md#2026-09-10-更新)；原有拓扑测试与测量记录保留不变。

</details>
