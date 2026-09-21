# Qwen3.5-VL 397B-A17B @ 4K — rebench milestone

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

| Cell | Job | 实测 SHA | TFLOP/s/GPU | ms/step |
|---|---:|---|---:|---:|
| Decoder-only | 711422 | `447689542` | 650.15 | 11,010.75 |
| **VL + MDP** | **717519** | `6a975c1aab` | **625.35** | **11,505.95** |

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

| 相同日志范围：rank 0/64 峰值 | Decoder-only | VL + MDP | 差值 |
|---|---:|---:|---:|
| max allocated | 156.31 GiB | 159.05 GiB | +2.74 GiB |
| max reserved | 167.95 GiB | 169.87 GiB | +1.91 GiB |
| device used | 181.75 GiB | 183.42 GiB | +1.67 GiB |

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

| Cell | Job | 实测 SHA | TFLOP/s/GPU | ms/step | 全 rank max allocated / reserved |
|---|---:|---|---:|---:|---:|
| Native VL | 711528 | `6a975c1aab` | 310.55 | 19,352.60 | 156.37 / 166.18 GiB |
| **MDP VL** | **711618** | `6a975c1aab` | **501.05** | **11,993.75** | 156.60 / 217.04 GiB |

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
不把 PP 重划记为成功优化。本轮没有扩大到 8K/16K，也不作相应支持承诺。
更早的长序列试跑出现过 OOM，不能从本页 4K 结果推导其可运行。
