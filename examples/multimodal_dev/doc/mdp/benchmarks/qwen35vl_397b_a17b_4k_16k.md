# Qwen3.5-VL 397B-A17B @ 4K / 16K — rebench milestone

以下先保留 4K 的配置与结果，16K 的配置与结果见末尾增量更新。

128× GB300（32 节点），TP1 / PP2 / VPP2 / EP64 / CP1 / ETP1，
GBS2048 / MBS1（每个 dense-DP rank 32 个 packed microbatch）。
Decoder MXFP8，vision encoder BF16，cuDNN GDN，
partial CUDA graph `[attn, moe_router, moe_preprocess]`。
20 步无 profiler，取第 **9–20** 步中位数。

保留 PP layout `Et*16|t*16|t*16|t*12,m,L`（物理 32/28）。
MDP encoder whole recompute，单次执行/重算块上限 65,536 patch rows；
decoder recompute 关闭。环境与运行方式见 [统一复现说明](rebench_environment.md)。

本页将两种数据分开：**单条 4096 用于评估加上 VL 的整体开销；
变长 4K 只比较 Native VL 与 MDP。** 两组数字不能连成一条优化提升曲线。

## 单条 4096：加上 VL 的整体开销

每个 pack 恰好一条真实 4096-token 序列，无非零 padding tail。
VL 的视觉 token **包含在** 4096 内，而不是在 4096 个文本 token 后额外追加。
两侧 intended decoder token 数均为 8,388,608/step，序列长度平方和相同。


| Cell         | Job        | 实测 SHA       | TFLOP/s/GPU | ms/step       |
| ------------ | ---------- | ------------ | ----------- | ------------- |
| Decoder-only | 711422     | `447689542`  | 650.15      | 11,010.75     |
| **VL + MDP** | **717519** | `6a975c1aab` | **625.35**  | **11,505.95** |


**约 625 TFLOP/s/GPU，保留 Decoder-only 约 96.2% 的吞吐。**
观测到 step time 增加 **495.20 ms（+4.50%）**，TFLOP/s 下降 **3.81%**：
在这个视觉负载下，使用 MDP 加入 encoder 的整体开销有限，并非完全零开销。

比较使用历史不同节点，保留了 VL 的 mRoPE/输入准备、MDP 以及 stream 设置差异
（`CUDA_DEVICE_MAX_CONNECTIONS`：Decoder-only 1、VL 8）。
因此不是 encoder 单独耗时或 MDP 调度自身的测量，也不能直接外推到更重的视觉负载。

VL mock 的实际 pool：64 个场景，12 个 text-only，122 个视觉 item（25 个多帧），
平均每样本 **201.56** 个 merged vision token，单 frame 最大 1024 patch rows。
每步 2048 个样本、1,651,200 个 vision patch rows。两侧均完成 20/20，
没有 NaN、skipped iteration 或 OOM；VL 最终 LM/MTP loss 为 12.47872/12.53059。

### 显存


| 相同日志范围：rank 0/64 峰值 | Decoder-only | VL + MDP   | 差值        |
| ------------------- | ------------ | ---------- | --------- |
| max allocated       | 156.31 GiB   | 159.05 GiB | +2.74 GiB |
| max reserved        | 167.95 GiB   | 169.87 GiB | +1.91 GiB |
| device used         | 181.75 GiB   | 183.42 GiB | +1.67 GiB |


VL 全 128 rank 的最大 allocated/reserved 为 **159.33/171.35 GiB**。
Decoder-only 未收集全 rank 的 memory exit audit，不能拿上表的局部峰值与此直接相减。

### 完整配置与命令

- Decoder-only：[YAML](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_decoder_single4096.yaml) · [完整命令](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_decoder_single4096.sh)
- VL + MDP：[YAML](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_mdp_single4096.yaml) · [完整命令](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_mdp_single4096.sh)
- 数据：[mock_fixed_4096.json](../recipes/data/mock_fixed_4096.json)，`min=max=mean=4096`；
JSON 仍通过 `mdp_mock` 消费，不切换成另一套固定单图 `mock` provider。

按统一复现页提供 launcher，在每个节点执行一次，例如：

```bash
bash "$MCORE_ROOT/examples/multimodal_dev/doc/mdp/recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_mdp_single4096.sh" \
  torchrun --nnodes=32 --nproc_per_node=4 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
```



## 变长 4K：MDP 与 Native VL

这是实际多序列 THD 形状的 **mock** 数据：样本长度 lognormal
`min512 / max4096 / mean2048 / sigma1.1`，greedy packing 到 4096，
static capacity 8（为 dummy tail 预留一槽）。模型、并行和 decoder 优化不变。


| Cell       | Job        | 实测 SHA       | TFLOP/s/GPU | ms/step       | 全 rank max allocated / reserved |
| ---------- | ---------- | ------------ | ----------- | ------------- | ------------------------------- |
| Native VL  | 711528     | `6a975c1aab` | 310.55      | 19,352.60     | 156.37 / 166.18 GiB             |
| **MDP VL** | **711618** | `6a975c1aab` | **501.05**  | **11,993.75** | 156.60 / 217.04 GiB             |


**MDP 比 Native VL 吞吐高 61.34%，step time 短 38.03%。**
两者使用同一数据/packing 规则；Native 在首个 PP stage 内运行 encoder，
使用 vision 层级 recompute，MDP 则进行 whole-encoder replay。
这是两套 recipe 的 E2E 收益，不能拆成单个开关的独立收益。

实际每步为 7,033,088 个真实 decoder token，平均 pack 填充率约 **83.84%**；
MDP 每步消费 7,648 个原始样本。Native 的 greedy 样本计数仍报告 nominal GBS，
不要用该计数差异判断它少算了数据。Pool64 与 DP64 的周期耦合仍保留，
没有把本次实验描述为已经解决了数据分布问题。

两者都完成 20/20，无 NaN/skipped/OOM。Native 最终 LM/MTP loss
12.50398/12.66927，MDP 为 12.46608/12.55821。不同模型初始化和重算路径
下的 mock loss 不作为完整数值一致性或收敛证据。

### 完整配置与命令

- Native VL：[YAML](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_native_varlen.yaml) · [完整命令](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_native_varlen.sh)
- MDP VL：[YAML](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_mdp_varlen.yaml) · [完整命令](../recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_mdp_varlen.sh)
- 数据：[mock_lognormal.json](../recipes/data/mock_lognormal.json)。

```bash
# Native and MDP use the same launcher/resources; select the corresponding .sh.
bash "$MCORE_ROOT/examples/multimodal_dev/doc/mdp/recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_mdp_varlen.sh" \
  torchrun --nnodes=32 --nproc_per_node=4 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
```



## 当前保留的配置

变长 4K 下补测了 PP31/29 和 PP30/30，分别为 487.2、482.0 TFLOP/s/GPU；
原 PP32/28 复测为 500.2（31/29 与复测使用相同节点）。因此仍保留 32/28，
不把 PP 重划记为成功优化。上述历史测试未扩大到更长序列，也不作相应支持承诺。
更早的长序列试跑出现过 OOM，不能从本页 4K 结果推导其可运行。

## 2026-10-26 更新：16K 配置与结果

397B-A17B 在单条 16384-token 输入下，VL + MDP 达到 **841.60 TFLOP/s/GPU**，
Decoder-only 为 **845.75 TFLOP/s/GPU**，保留约 **99.5%** 的吞吐。
模型代码基线为 `3a20252464deab35e47e5a3c6305f35fedb626ae`

上方 4K 历史结果保留，不将不同配置的数字连成优化提升曲线。

### 与 4K 的配置对比

128× GB300（32 节点），TP1 / PP2 / EP64 / CP1 / ETP1，
GBS2048 / MBS1，每个 dense-DP rank 32 个 packed microbatch。
16K 关闭 VPP，使用 `Et*32|t*28,m,L`。


| 配置项                         | 原有 4K 对照               | 16K 对照                  |
| --------------------------- | ---------------------- | ----------------------- |
| 单条序列配置长度                    | 4096                   | 16384                   |
| VPP 与层划分                    | VPP2，16/16/16/12       | 关闭，32/28                |
| Decoder recompute           | 关闭                     | Selective：`moe_act gdn` |
| THD 序列槽上限                   | 8                      | 16                      |
| CUDA_DEVICE_MAX_CONNECTIONS | Decoder-only 1 / MDP 8 | 双方均为 8                  |


镜像、cuDNN 9.25.1.1 库路径及 CPU/NUMA 绑定见
[统一复现说明](rebench_environment.md#2026-10-26-更新397b-16k)。
本次未新增模型实现修改；数值对齐逻辑仅用于下方 benchmark 入口。

双方保留 decoder MXFP8、FP8 parameter gather 与 buffer reuse、
precision-aware optimizer、gradient/parameter overlap、HybridEP、
grouped GEMM、router/独立 permutation fusion、cuDNN GDN、MTP，
以及 TE partial CUDA graph `[attn, moe_router, moe_preprocess]`。
MoK 仍关闭。非交错调度会关闭 PP P2P overlap 和 `align_param_gather`，
但 `overlap_param_gather` 仍开启。

MDP 侧增加 27 层 BF16 vision encoder，保持 whole recompute，
单次执行/重算块上限为 65,536 patch rows。原有 vision attention
D72→D128 padding 和图像/视频生成规则不变。视觉 token **包含在** 16384 内，
不是在 16384 个文本 token 后额外追加。

实际 mock pool 为 64 个场景，12 个 text-only，121 个视觉 item（24 个多帧），
每条总长度均为 16384，平均每条 **199.45** 个 merged vision token（约 1.22%），
单 frame 最大 1024 patch rows。这是轻视觉负载，不外推到更重的图像/视频配置。

### 实测结果

各运行 20 步，不开 profiler；以下性能取第 **9–20** 步中位数，保留慢步与 GC 步。


| Cell         | TFLOP/s/GPU | ms/step       | 全 rank max allocated / reserved | 最终 LM loss   |
| ------------ | ----------- | ------------- | ------------------------------- | ------------ |
| Decoder-only | 845.75      | 36,672.45     | 226.43 / 253.11 GiB             | 12.45068     |
| **VL + MDP** | **841.60**  | **36,895.50** | **229.45 / 255.95 GiB**         | **12.46745** |


观测到中位数 step time **+0.61%**、TFLOP/s **-0.49%**，整体性能基本持平。
两组平均 step time 分别为 **37,429.15 / 37,292.95 ms**，均值受慢步影响、方向略有不同；
不把中位数的微小差值解释为精确的 encoder 或 MDP 独立开销。

两组均完成全部 20 步，无 OOM、NaN 或 skipped iteration；
encoder 的全 128 rank 参数审计均为 BF16，未启用 encoder FP8。
Allocated/reserved 为全 128 rank 的全程峰值，MDP 增量约 **3.02 / 2.84 GiB**。
周期采样 rank 0/64 的 device-used 峰值为 **272.56 / 276.54 GiB**，
不代表全 rank 连续高水位。显存余量较小，20 步通过不保证长训练或更重视觉负载可用。

### 输入数值规则

两侧使用可重复的 uniform-random 文本规则：以 `1234 + sample_id` 为训练 seed，
在 `[1, 248320)` 内采样，排除 image/vision-start 特殊 token，并对齐 next-token label
及末尾 label/loss mask。Decoder-only 输出 16384 个真实 token，不再以一行 padding 补齐。

MDP 的视觉占位、图像/视频特征和视觉 loss mask 保持不变；不是两边加载同一份 tensor，
也不强制模型权重、activation 或 gradient 逐位一致。强制 MoE 负载均衡保持开启，
mock loss 仅作运行诊断，不作为精度或收敛验证。

### 完整配置与命令

每份 YAML 都包含完整参数，不依赖外部 BASE/model 文件；相邻 shell 脚本
包含相同参数，在已分配的 GPU 容器内接收调用方提供的 launcher 前缀。
这些脚本本身不申请 GPU。两侧均通过
[aligned_entry.py](../recipes/qwen35_vl_397b_a17b/aligned_entry.py) 安装本次验证的
benchmark-only 数值适配，再调用原有训练入口；
[dataset_alignment.py](../recipes/qwen35_vl_397b_a17b/dataset_alignment.py) 提供对应规则与审计。
运行时使用包含这些 recipe 和辅助文件的 checkout，不只复制单个 shell 文件。


| 对照组          | YAML                                                                               | 完整命令                                                                           |
| ------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Decoder-only | [YAML](../recipes/qwen35_vl_397b_a17b/s16k_pp2ep64_decoder_single16384_novpp.yaml) | [SH](../recipes/qwen35_vl_397b_a17b/s16k_pp2ep64_decoder_single16384_novpp.sh) |
| VL + MDP     | [YAML](../recipes/qwen35_vl_397b_a17b/s16k_pp2ep64_mdp_single16384_novpp.yaml)     | [SH](../recipes/qwen35_vl_397b_a17b/s16k_pp2ep64_mdp_single16384_novpp.sh)     |


数据：[mock_fixed_16384.json](../recipes/data/mock_fixed_16384.json)，`min=max=mean=16384`。

通过 `MCORE_BENCH_OUTPUT` 指定输出目录。例如，在每个已分配的 GPU 容器内，
使用正确配置的 per-rank launcher：

```bash
export MCORE_BENCH_OUTPUT=/path/to/run-specific-output
bash examples/multimodal_dev/doc/mdp/recipes/qwen35_vl_397b_a17b/s16k_pp2ep64_decoder_single16384_novpp.sh \
  /path/to/bindpcie python
```

MDP 对照替换为对应脚本；容器、挂载和分布式环境沿用统一复现说明。
