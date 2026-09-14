# MDP 提交历史

按时间顺序记录本分支上与 MDP 相关的每一次提交。面向了解 Megatron-Core 与多模态
训练的读者，每条给出 high-level 的意图与影响，不展开实现细节。

本分支（远端 `lit/mdp_fast_pass`），HEAD `ab9bcd537`，由三段构成：

1. **前置**：`multimodal_dev` 与 Qwen3.5-VL 的基础能力（MDP 之前）
2. **MDP 主体**：从 `243a0d7f1` 引入 `megatron/core/mdp/` 到 `638bba205`
3. **fast_pass 0901 新增**：在 `638bba205` 之上集成的若干 PR、GDN CUDA graph 修复与 benchmark 工具

---

## 一、前置 —— multimodal_dev 与 Qwen3.5-VL 基础

### `88894e3ee` [Dev] Skip no-op BSHD padding masks in Qwen3.5-VL (#5964)

BSHD 布局下若 padding mask 实际上不屏蔽任何位置，构造并传递它是纯开销。这个提交
让这类无效 mask 直接跳过。

### `9304a2564` [Dev] Add Triton fused mRoPE for Qwen3.5-VL (#5962)

为 Qwen3.5-VL 的 3D mRoPE 加入 Triton 融合 kernel，替代逐段拼接的实现。mRoPE 在
vision encoder 与 decoder 上都要跑，是热点之一。

### `3f29946dc` Add pipeline-parallel support to multimodal_dev and Qwen3.5-VL

**MDP 的前提条件。** 在此之前 `multimodal_dev` 在每个 rank 上构建完整模型，并直接
拒绝 `--pipeline-model-parallel-size > 1`。这个提交把标准的 Megatron pipeline-stage
契约接进多模态路径：`MultimodalModel` 接受 `pre_process` / `post_process` /
`vp_stage` 并转发给内部的 `GPTModel`，**vision encoder 只在第一个 PP stage 上构建**；
同时在外层模块暴露 `share_embeddings_and_output_weights` 等接口，供
`finalize_model_grads` 的 word-embedding 梯度 all-reduce 使用。

顺带一提：「vision encoder 只在 stage 0」正是后来 MDP 要解决的负载失衡的来源。

### `816ccf36d` / `62e10cadc` 修复 mRoPE parity 测试的过时 config stub

融合 mRoPE（#5962）让 `apply_rotary_pos_emb` 开始直接读取 `config.mrope_section`
和 `config.apply_rope_fusion`，而测试里的 `SimpleNamespace` stub 写于这两个字段
出现之前，于是抛 `AttributeError`。两个提交把字段补进 stub，把测试钉在它原本要验证
的 legacy THD 路径上。

### `01941906a` test(examples/multimodal_dev): make the standalone harnesses pytest tests

`examples/multimodal_dev/tests` 下有四个文件是 argparse + `main()` 的脚本，没有任何
`test_*` 函数——pytest 导入它们但什么都不跑，于是 BSHD-vs-THD、CP-vs-CP=1、
patch-merger 这三组等价性检查实际上只能靠手工执行。这个提交把每组比较抽成可复用
函数，再在其上加 pytest 入口，`main()` 调用同一份代码，保证 CLI 与测试不会漂移。

### `e0df73690` ci: run the examples/multimodal_dev suite as a unit-test bucket

`multimodal_dev` 紧跟 `megatron.core` 的 API（rope_utils、GPTModel、
TransformerConfig、GPT layer spec）却位于 `tests/unit_tests` 之外，所以 core 的改动
可以在无人察觉的情况下把示例弄坏——上面那个 mRoPE stub 就是一例。这个提交把该目录
加为独立的 H100 单测 bucket，让它和会破坏它的改动跑在同一条流水线里。

---

## 二、MDP 主体

### `243a0d7f1` feat(mdp): add topology-aware configuration and deterministic planning

**MDP 的起点**，一次性引入 `megatron/core/mdp/` 的骨架：`config.py`（`MdpConfig`
与支持矩阵校验）、`rank_mapping.py`（从 `RankGenerator` 坐标推出 outer-DP planning
group 与逻辑 encoder worker）、`plan.py` / `planner.py`（plan 数据模型、blake2b
摘要、确定性整数 LPT）、`groups.py`（进程组与定宽 descriptor 广播）、
`allocator.py` / `storage.py`（MDP 缓冲区的唯一分配点、endpoint leaf 存储），以及
`protocols.py`、`errors.py`。同时在 `examples/multimodal_dev/arguments.py` 加入
`--mdp-*` 参数。

这一层是纯计算与配置，还没有接进训练循环。

### `4a1733c45` feat(mdp): add owner-sharded pixel capture and all-to-all routing

加入数据通路：`bridge.py`（pixel / embedding / gradient 三个阶段共用的 ledger 与
`all_to_all_single` 传输）、`window.py`（迭代窗口，owner 分片的 pixel 捕获）、
`packing.py`，以及 `examples` 侧的 `mdp_adapter.py`、`data/mdp_mock.py`。

「owner 分片」是关键设计：每个 microbatch 的 pixel 只由 `microbatch_id %
num_workers` 对应的那个 worker 做 materialize 和 H2D，其余 worker 跳过，因此图像
搬运的成本天然被摊开且零通信。

### `eae318a58` feat(mdp): integrate the phased encoder runtime with multimodal training

把 MDP 接进训练循环：`runtime.py`（P0–P6 相位机）、`schedule.py`（调度与
finalizer 包装）、`encoder.py`（WORLD 上的 encoder DDP + ZeRO-1）、
`activation.py`（encoder THD 分块前向）、`integration.py`（训练循环接缝），并修改
`megatron/training/training.py` 与 `pretrain_multimodal.py`。同时加入
`megatron/core/mdp/README.md` 与 `test_mdp_parity.py`。

至此 MDP 可以端到端跑通：像素不进 decoder，encoder 不进 decoder 的模型列表。

### `bff4392a3` feat(mdp): coordinate optimizer and checkpoint state across model domains

`optimizer.py`（组合优化器：WORLD MAX 溢出并集、合并范数裁剪、
`[decoder_dense, decoder_expert?, encoder]` 一次原子 step）与 `checkpoint.py`
（`torch_dist` 门面，按 `vision_model.*` 保存/加载并带 WORLD 副本元数据）。

MDP 把参数拆成了两个域（decoder 分片 + 复制的 encoder），优化器和 checkpoint 必须
显式协调，否则溢出判定和梯度裁剪会在两个域上各算各的。

### `bc24db139` perf(mdp): streamline window capture, collectives, and synchronization

对 window 捕获、集合通信与同步点做的一轮性能整理，覆盖 `runtime.py`、`window.py`、
`groups.py`、`encoder.py`、`optimizer.py`。

### `bf02ef3f2` perf(multimodal): cache reusable vision position metadata

缓存可复用的 vision 位置元数据（`mdp_adapter.py`）。同一个网格形状的位置信息在多个
样本、多次迭代之间是重复的，缓存后避免每次重算。

### `0eaf12403` fix(multimodal): correct packed-sequence and vision FLOP accounting

修正 THD packing 与 vision 的 FLOPs 统计。packing 之后按 `--seq-length` 的闭式公式
不再成立，需要按真实 `cu_seqlens` 计 token-linear 项与 `sum(L^2)` 的 attention 项；
vision encoder 的 FLOPs 此前完全没有计入。

### `28ce97da1` test(multimodal): add reproducible MDP benchmark tooling

加入可复现的 MDP benchmark 工具（`scripts/run_mdp_experiments.sh` 等）。

### `77e9a8084` test(multimodal): make the mock scenario pool representative by default (#43)

`MdpThdMockDataset` 原本只有五个手写场景、每个 32–80 token。它们能让数据集契约跑
起来，但**什么都测不出来**：这个尺寸下 decoder 是空闲的、THD packing 无物可 pack、
MDP planner 没有负载可平衡、vision encoder 的网格缓存只见到八个不同网格。于是每次
benchmark 都得把 `ENTRY` 指向一个 out-of-tree 的 wrapper 去 monkey-patch
`_SCENARIOS`，仓库内的 launcher 自己产不出有意义的数字。这个提交把有代表性的场景池
做成默认值。

### `e5e06c51e` fix(multimodal): avoid VPP FLOP double counting (#44)

开启 VPP 时，同一物理 rank 上的每个 virtual chunk 都会为同一个 microbatch 调一次
forward step，FLOPs 统计因此按 VPP 倍数重复计入。修正为只由 canonical chunk 上报。

### `b1c75e437` feat(mdp): support native decoder dp overlap (#52)

支持原生的 decoder 数据并行 overlap（`overlap_grad_reduce` /
`overlap_param_gather`）。decoder 侧的 overlap 仍由原生 PP/VPP 调度拥有，独立的
encoder DDP 域在 P5/P6 保持同步。

### `ee3b6937a` feat(mdp): persist composite optimizer state in the torch_dist checkpoint (#46)

此前 MDP 的 checkpoint 是「仅权重」契约：encoder 权重写得出去但读不回 encoder DDP，
组合优化器状态被 `--no-save-optim` / `--no-load-optim` / `--no-save-rng` /
`--no-load-rng` 直接排除，启动时还有断言强制要求这几个开关。这个提交改成**精确
resume**：组合优化器状态、LR scheduler 与 RNG 状态都进入 `torch_dist` checkpoint，
同世界规模下可原样恢复。

### `22ae2a10b` perf(mrope): tile rotary kernels and fuse THD position IDs (#45)

THD mRoPE kernel 原本每个 `(token, head)` 对启动一个 program，每个 program 还要线性
重扫整个 `cu_seqlens`，且子序列数量作为运行时标量传入使 Triton 无法展开。结果是
sin/cos 在 Qwen3.5-VL 的 vision encoder 上被重复计算十六遍，且每个 program 只搬一行
`head_dim`，大部分 warp lane 空闲。改为每个 program 处理 `[BLOCK_T, BLOCK_H]` 的
token/head 瓦片，并融合 THD position id 的生成。

### `638bba205` feat(mdp): enable decoder EP communication overlap (#47)

支持 decoder 侧的 MoE 专家并行通信 overlap
（`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`），并顺带重构了
decoder 输入准备的共享逻辑、修复 MTP 在 EP overlap 下对 packed sequence 的支持。
vision encoder 仍在该调度之外。

**这是 `dev_mdp` 的基线提交，下面是 fast_pass 0901 分支在其之上的新增。**

---

## 三、fast_pass 0901 分支新增

### `192860ef0` feat(mdp): greedy token-budget packing and fixed-shape THD batches (PR #48)

**benchmark 口径的关键改动。** 引入 greedy token-budget 装箱（`packing.py` 的
`GreedySampleStream`）与 `--thd-static-packing`：前者把样本按 token 预算装满一个
bin，使 `global_batch_size` 的含义从「样本条数」变为「bin 个数」；后者把每个
microbatch 的 THD 形状固定下来（`cu_seqlens` 补到定长、`max_seqlen` 钉在静态目标），
这是 per-layer CUDA graph 能够 capture 的前提。

同时给出跨迭代的 sample buffer——底层 dataloader 按 `--micro-batch-size` 成批吐出，
而装箱是逐条进行的，剩余样本必须跨迭代保留，否则会静默丢数据。该 buffer 不进
checkpoint，因此 greedy packing 与 `--save`/`--load` 互斥。

### `ff7a147f1` feat(mdp): run per-layer CUDA graphs on the MDP decoder (PR #49)

让 per-layer（partial）CUDA graph 在 MDP 的 decoder 上工作。graph 完全活在 P4 的
原生调度内部，与 MDP 的相位机不交互——MDP 的 bridge 只接触 decoder 的 embedding
叶子节点及其梯度，从不碰 transformer 层。

### `76a7ec387` feat(mdp): add complete vision encoder recomputation (PR #53)

加入完整的 vision encoder 重算：P2 在 `no_grad` 下前向并保留 pixel/layout/RNG 配方，
P5 恢复 RNG 后逐块重放整个 encoder chunk 再做反向。

同时把此前自由格式的 vision config override 通道（`--mdp-vision-config-override`）
替换为**类型化参数** `--encoder-recompute-{granularity,method,num-layers,modules}`。
旧通道在本分支上已不存在。

### `d486d6ca2` fix(training): honor --record-memory-history and dump on the recording ranks

`start_memory_history_recording()` 此前只能从 `_build_model_wrapper()` 到达，而它的
函数体被 `cfg_container is not None and cfg_container.model is not None` 门控住；
自带 `model_provider` 的入口（`pretrain_multimodal.py` 就是其一）永远进不去，于是
`--record-memory-history` 是个静默的 no-op，导出的快照既没有 `device_traces` 也没有
frame。改为同时从 `pretrain()` 调用，并在真正做记录的那些 rank 上导出。

### `50c2f1cad` bench(mdp): align run_mdp_experiments.sh with the EP8 reference and add observability toggles

把 `run_mdp_experiments.sh` 的默认值对齐到 GB200 8×EP8 参考命令，每一项都带 env
覆盖：`DISPATCHER=flex` + `FLEX_BACKEND=hybridep` + `HYBRIDEP_NUM_SMS=32`（取代原先
硬编码的 `alltoall`）、`VISION_RECOMPUTE=1`、`PRECISION_AWARE_OPT=1`、
`FORCE_LOAD_BALANCING=1`、`GDN_FUSION=1`，并加入若干可观测性开关。

### `d37969c86` HACK(bench): greedy token-budget packing on the non-MDP path

**仅供 benchmark，明确不上游。** `GreedySampleStream` 只在 `--mdp-enable` 下由
`MdpRuntime` 安装；MDP 关闭时 `pack_or_pad_batch` 每个 bin 只拼 `micro_batch_size`
条原始样本。这会让 MDP-off 的基线与它要对比的 MDP 运行**装箱方式不同**，两者的吞吐
差异就同时混进了「greedy packing」与「模态解耦」两个因素。这个提交在非 MDP 路径上
也包一层同样的 greedy stream，使 A/B 只测量模态解耦本身。

已知限制并被刻意保留：`consumed_train_samples` 的统计仍是 MDP-gated，因此这条路径
会**少报** consumed samples。对吞吐测量无碍，对任何读取样本计数的用途都是错的。

### `e4c15d835` perf(mdp): recycle MDP-owned buffers through a pooled allocator

MDP 的 reserved 显存远高于其 live set：4×GB300 上（PP2/EP2，16K THD，20 次迭代）
rank 0 reserve 了 162.3 GiB 而分配峰值只有 111.3 GiB，51 GiB 被占住不用；同形状下
MDP-off 的基线只比峰值高 3.7 GiB。根因不是数据量而是**请求尺寸**：
`RowCapacityPolicy` 的 `alignment_rows` 在生产配置下为 1，于是每个 MDP 缓冲区的请求
大小都随该迭代的 vision 条目数逐次变化，caching allocator 无法复用。这个提交让 MDP
自有缓冲区走池化分配器、按迭代回收。

**后续实测该提交对吞吐与显存均无可观测影响**（`mem-profile` 归因显示相关分配只占
峰值的 0.5%），改动本身正确且有测试覆盖，但针对的不是主要矛盾。

### `e03e2cd00` fix(qwen35_vl): pad vision attention head_dim 72 -> 80 around the attention call

Qwen3.5-VL 的 vision encoder 是 `hidden 1152 / 16 heads / kv_channels 72`。在
`head_dim = 72` 上，cuDNN 的 THD 融合 attention **反向**会申请约 73 GiB 的
workspace；换成 64 / 80 / 96 / 128 都正常（实测分别为 0.86 / 1.05 / 1.26 / 1.68 GiB）。

修法是在 attention 调用前把 Q/K/V 沿 head_dim 补零到 80、调用后裁回 72，并把
`softmax_scale` 钉在 `1/sqrt(72)` 以保持数值等价（独立验证输出逐位相同）。代价是
约 11% 的 vision attention FLOPs 与两个 elementwise kernel。

端到端效果（rank 0）：`max alloc` 120.5 → 71.0 GB，`max reserved` 144.1 → 72.4 GB，
`device used` 160.4 → 88.7 GB，降幅 44.7%。

### `1243df8b3` fix(ssm): skip GDN cu_seqlens D2H sync during CUDA graph capture

`gated_delta_net.py` 的 `_resolve_cu_seqlens` 无条件执行
`cu_seqlens[-1].cpu().item()`，这是一次 device-to-host 同步，TE 的 CUDA graph
capture 会直接拒绝。这一条阻断了**所有 hybrid GDN 模型**（含 Qwen3.5-VL）的
per-layer `attn` CUDA graph。静态 THD packing 已经保证了该不变量，因此 capture 期间
可以安全跳过。

### `82c7b95ac` fix(ssm): guard remaining GDN sync points during CUDA graph capture

上一条的后续：`(seq_lengths % cp_size != 0).any()` 与
`torch.equal(cu_seqlens_q, cu_seqlens_kv)` 都会通过隐式 `__bool__` 触发 D2H 同步，
属于同一类失败。同样在 `torch.cuda.is_current_stream_capturing()` 下跳过——不变量由
静态 packing 保证，且 warmup 迭代已在 capture 之前 eager 地验证过。

### `4312e8dff` fix(ssm): thread cu_seqlens_cpu into fla GDN calls

fla 的 `chunk_gated_delta_rule` 与 `causal_conv1d` 内部会调用
`prepare_chunk_indices`，若不提供 CPU 侧镜像就会对 CUDA 上的 `cu_seqlens` 做
`.tolist()`——capture 期间禁止。这个提交把 `cu_seqlens_cpu` 一路传进去。

### `01d7c3c32` fix(ssm): build fla chunk_indices on device at a static shape for CUDA graphs

fla 的变长分块 kernel 需要一张 `[NT, 2]` 的表，把每个 chunk 槽位映射到
「第几条序列、序列内第几块」。fla 在主机侧**依据 `cu_seqlens` 的内容**构建它，这对
CUDA graph 是双重违规：构建本身是主机↔设备往返；而 fla 0.5.2 的 `cu_seqlens_cpu`
参数只消除了 D2H 那一半。改为在设备上以**固定形状**构建——capture 时只冻结形状，
数值在每次 replay 时按静态 `cu_seqlens` 缓冲区的当前内容重新计算。

### `ae90f06e5` test(ssm): cover the fixed-shape GDN chunk_indices table

为上一条加测试。这个改动很容易出微妙的错误——只有表的**形状**可以在 capture 时
冻结，**数值**必须每次 replay 重算。测试在六种 packing 布局上覆盖了 fla kernel 真正
依赖的四条性质，包括与 fla 自己的 `prepare_chunk_indices` 逐元素一致。

### `3a966f833` fix(fusions): build packed seq_idx without a data-dependent shape

pre-GDR fusion 无法被 capture：开 `--cuda-graph-modules attn` 后 capture 启动随即在
`fused_streamed_pre_gated_delta_rule` 内部死于
`operation not permitted when stream is capturing`。根因是
`_resolve_packed_seq_idx` 用 `repeat_interleave` 构建 token→序列 id 的映射，其输出
**长度**取自设备数据。改用 `searchsorted`，在由 `total_tokens` 决定的固定形状上得到
同一张映射表。

### `ab6284178` test(fusions): cover CUDA-graph-safe packed seq_idx construction

为上一条加测试，钉住两件事：映射表在六种 packing 布局下（含带 padding 的 THD 批次
产生的尾部零长度序列）与 `repeat_interleave` 写法逐元素相同；以及它确实可被
capture——测试还会用不同的 packing 做 replay，确认数值跟随 `cu_seqlens` 缓冲区而不是
在 capture 时被冻结。

### `ab9bcd537` EXPERIMENT(mdp): env-gated escape hatch for overlap_window_capture + CUDA graphs

`config.py` 基于一个**预测中的**竞态，拒绝 `--mdp-overlap-window-capture` 与
per-layer CUDA graph 同时开启；而拒绝信息自己写明该冲突「时序相关、无法稳定复现」，
即它是推理出来的、从未被测量。这个提交加入
`MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1` 以便对该预测做验证。

**默认行为完全不变**：不设该变量时抛出的仍是同样的 `MdpConfigurationError`；设置后
改为打印一条醒目的 warning。这不是移除该 guard 的第一步——被拦的是竞态，干净的运行
无法为其定界。相关结论见 `benchmarks/mdp_feature_ablations.md`。
