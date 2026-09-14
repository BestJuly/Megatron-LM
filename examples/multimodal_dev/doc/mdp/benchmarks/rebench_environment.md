# Rebench milestone — 环境、运行与测量约定

本页适用于 35B 的 474 milestone 和 397B 的两组 4K 对照。旧 fast-pass 页面的
`<FLA_PREFIX>/<NVRX_PREFIX>` 侧安装方式不是本轮环境，不要混用两套库。

## 代码与镜像

- Benchmark 分支：`qizhang/mdp-qwen35vl-35b/397b-benchmark`。
- 来源：从 Li 的 [`lit/qwen35-397b-rebench`](https://github.com/BestJuly/Megatron-LM/tree/lit/qwen35-397b-rebench)
  checkout，再 cherry-pick [`lit/mdp_fast_pass`](https://github.com/BestJuly/Megatron-LM/tree/lit/mdp_fast_pass)
  的 MDP 相关代码。
- 基础代码 SHA：`18bb5b1cd6174785904f0f4f3ab35c2c5d05add2`。
- 当前已验证的 397B 代码：`6a975c1aab457ae88b16247582674c3dc81a1300`。
  35B 的实测快照见其测试页；这些快照都保留在本分支历史中。
- 镜像文件名：
  `gb200-gb300-2607-cudnn9.25.1.1-cudnnfe1.28.0-temain-fa4b29-flamain-20260908.sqsh`。
- 镜像 SHA256：
  `c1c3cd3bad118db4c6c54ae72970ea134c5c2e0591cbd40f556634b275205e53`。
- 镜像内版本：TE `2.20.0.dev0+5f6105b9`、cuDNN **9.25.1.1**、
  cuDNN frontend **1.28.0**、FLA **0.6.0**、FA4 beta29。

镜像不是本仓库的一部分：从有权限的镜像来源取得同一文件，放到集群可访问的位置，
校验 SHA256，替换 YAML 的 `<CONTAINER_SQSH>`。本次没有导出镜像的新 Docker 构建配方，
也没有在运行时升级包。

同一镜像还带有另一套系统 cuDNN。复现必须同时设置：

```bash
export CUDNN_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
```

新的 `.sh` 已包含这两项及对应 cell 的全部性能环境变量。397B 的
`HF_HUB_OFFLINE=1` 也保持原设置：请事先缓存所需 tokenizer / processor 配置，
并将缓存目录挂入容器；无需下载模型权重。35B 使用记录中的
`Qwen/Qwen3.5-397B-A17B` tokenizer 和 `Qwen/Qwen3.5-35B-A3B` processor；
397B 两者均为 `Qwen/Qwen3.5-397B-A17B`。

## 运行：完整命令随分支提供

新增 `.sh` 是从成功作业的完整命令整理出来的，不调用 toolkit，也不申请 GPU。
脚本定位当前 checkout 根目录、读取仓库内 JSON，然后将训练入口和全部参数交给
你传入的 launcher。它不自动 checkout、下载数据或提交作业。

例如，在每个已分配且已配置好容器的节点上执行一次（不是在登录节点运行）：

```bash
export MCORE_ROOT=/path/to/Megatron-LM
export HF_HOME=/path/to/cached/huggingface
export MASTER_ADDR=first-node-hostname  # replace with the first allocated node
export MASTER_PORT=29500
export NODE_RANK=0  # set this node's index: 0..31 for 397B, 0..1 for 35B

# 397B: 32 nodes x 4 GB300. Replace the recipe filename for another 397B cell.
bash "$MCORE_ROOT/examples/multimodal_dev/doc/mdp/recipes/qwen35_vl_397b_a17b/s4k_pp2ep64_mdp_single4096.sh" \
  torchrun --nnodes=32 --nproc_per_node=4 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
```

35B 使用相同模式，但设 `--nnodes=2 --nproc_per_node=4`，选择 35B 的脚本。
完整参数直接查看对应 `.sh`；输出目录可用 `MCORE_BENCH_OUTPUT` 指定。
启动参数先于训练入口传给 `.sh`，不要把额外训练开关混在 launcher 参数中。

**CPU/NUMA 与资源布局不能省略。** 实测使用 Slurm 每 GPU 一个 task，再以
`bindpcie` 将该 rank 绑定到 GPU 邻近的 CPU/NUMA；上面的通用 torchrun 示例
不会自动复制这一绑定。若已有等效 per-rank Slurm/容器配置，可将前缀写为：

```bash
# Run once PER RANK via the site's configured srun/container wrapper.
bash /path/to/the/selected_recipe.sh /path/to/bindpcie python
```

每个 rank 必须获得足够 CPU，不能让同节点所有 rank 共用一个 CPU 核。
397B 的两个 EP64 group 分别应落在 16 节点的 NVLink 域内；OCI-AGA 实测使用
`--segment=16`，35B 使用 `--segment=2`。其它集群按其拓扑配置等效放置。
站点的 account、partition、容器挂载、HF cache 和 affinity helper 由部署侧提供。
本轮曾排除 `nvl72d142-T[01-16],nvl72d159-T[01-16]`，这是低性能观察后的预防措施，
不是已确认的硬件故障，也不是通用跨集群参数。

## Recipe 的用途与边界

- YAML 包含完整 enabled 参数，不依赖原作者机器上的 `BASE` 或 model/data YAML。
  用 toolkit 导入时还需提供 `MCORE_ROOT`、`MCORE_BENCH_OUTPUT` 和集群配置。
- YAML 的 `MEGATRON.commit` 标记**实测代码**。直接执行 `.sh` 则使用当前 checkout；
  精确重放旧快照时先保存复现材料，再自行选择该 SHA，不要误把文档 HEAD 当成新实测 SHA。
- 只移除了站点路径、W&B run ID 和退出时的本地审计钩子；模型/优化/训练窗口均保留。
  `.sh` 与 YAML 同源，并与保存的训练命令逐项核对；本次文档整理未重新运行 GPU benchmark。
- 现有 MDP benchmark 仍设置 `MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1`。
  **这是已测的实验组合，不是通用生产支持承诺**；没有移除相关检查。
- 397B 的 MXFP8 仅用于 decoder，encoder 为 BF16；无保存/加载 checkpoint。
  不将本轮结果作为 MXFP8 checkpoint/resume、收敛或其它未测功能的验证。
- 保留 standalone permutation fusion，不开启 HybridEP 内置 permutation fusion；
  不使用 MoK，也不新增 decoder recompute。

## 测量口径

- 运行 20 步，不开 profiler。397B 取第 **9–20** 步中位数；35B 的初验与回归窗口
  在测试页分别标注，保留周期性 GC 慢步。
- TFLOP/s 是日志的有效**模型 FLOPs**除以 E2E 时间及 GPU 数；不包含所有 padding、
  recompute、通信或 CPU 开销。比较前先对齐真实 token 数和 `sum(L²)`。
- `max allocated`、`max reserved`、`device used` 不能混用；显存范围按每张表说明。
  日志标签 MB 实际以 MiB 为单位，页面统一换算为 GiB。
- 变长数据由 `mdp_mock` 生成，是真实多序列 THD 形状，但**不是实际训练语料或解码视频**。
  强制 MoE 负载均衡保持开启，loss 仅作有限步数的运行诊断。
- Native VL 使用 vision 层级 recompute；MDP 使用 whole-encoder replay。
  Native 与 MDP 的收益是完整 recipe 的 E2E 对比，不是单个开关的独立因果测量。
