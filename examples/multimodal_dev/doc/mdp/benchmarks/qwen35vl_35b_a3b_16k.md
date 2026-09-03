# Qwen3.5-VL 35B-A3B @ 16k —— native 与 MDP 对比

8× GB300（2 个 tray，同一 NVL72 机架），TP1 / CP1，MBS 1，**梯度累积 16**，
序列长度 16384，THD greedy + static packing，BF16，decoder 重算关闭，
partial CUDA graph `[attn, moe_router, moe_preprocess]`，pre-GDR fusion 与
HybridEP 128-token 分块均开启，`--dataset-provider mdp_mock`（lognormal 分布）。
取第 6 次迭代及以后的中位数。代码版本 `ab9bcd537`。

所有 cell 都开启了 vision 重算：native 路径用 `--recompute-vision`，MDP 路径用
`--encoder-recompute-granularity whole`（`--recompute-vision` 在 `--mdp-enable`
下会被直接拒绝，所以两条路径必然写法不同）。

## 主要结果

| 拓扑 | DP | GBS | native | MDP（最佳） | MDP 增益 | native 显存 | MDP 显存 |
|---|---|---|---|---|---|---|---|
| PP1 / EP8 | 8 | 128 | **358.9** | 未测试 | — | 240.9 GB | — |
| PP2 / EP4 | 4 | 64 | 305.1 | 314.2 | **+3.0%** | 187.3 GB | 211.0 GB |
| PP4 / EP2 | 2 | 32 | 190.8 | **211.2** | **+10.7%** | 187.1 GB | 186.0 GB |

单位为每 GPU 的 TFLOP/s。**MDP 的优势随 pipeline 深度增加而增大**，并且在 PP4
下不再额外消耗显存。

两个现象出自同一个机制。在 native 路径上，vision encoder 运行在 decoder 前向
内部、且**只在 pipeline stage 0 上**；其余所有 stage 都要在它后面等待。pipeline
越深，被这一个 stage 卡住的机器份额越大，MDP 把 encoder 移出关键路径所能回收的
就越多。显存同理：MDP 按 `workers = CP × PP` 对 encoder 状态分片，PP2 → PP4 使
每 rank 的缓冲区减半，PP2 下那 +23.7 GB 的额外开销随之消失。

**PP1/EP8 的 native 仍然是绝对性能最高的配置**，358.9 TFLOP/s，代价是 240.9 GB
（约占 277 GiB 的 87%）。MDP 在 PP1/EP8 下尚未测试；按上述趋势推断，那里应当是
MDP 收益最小的场景，因为不存在需要等待 stage 0 的下游 stage。

## 全部测试 cell

| cell | PP/EP | DP | GBS | MDP | CUDA graph | window overlap | ms/iter | TFLOP/s | maxAlloc | devUsed | 末次 lm loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `s16k_ep8_base` | 1/8 | 8 | 128 | 关 | 开 | – | 15,113.7 | **358.9** | 210.7 GB | 240.9 GB | 12.55561 |
| `s16k_pp2ep4_base` | 2/4 | 4 | 64 | 关 | 开 | – | 9,180.1 | 305.1 | 163.7 GB | 187.3 GB | 12.55553 |
| `s16k_pp2ep4_mdp_noovlp` | 2/4 | 4 | 64 | 开 | **关** | 关 | 10,882.5 | 257.0 | — | 179.3 GB | — |
| `s16k_pp2ep4_mdp` | 2/4 | 4 | 64 | 开 | 开 | 关 | 9,217.2 | 303.6 | 162.9 GB | 210.9 GB | 12.60779 |
| `s16k_pp2ep4_mdp_cg_ovlp` | 2/4 | 4 | 64 | 开 | 开 | **开** | 8,900.3 | **314.2** | 163.1 GB | 211.0 GB | 12.59087 |
| `s16k_pp4ep2_base` | 4/2 | 2 | 32 | 关 | 开 | – | 7,175.2 | 190.8 | 163.6 GB | 187.1 GB | 12.60054 |
| `s16k_pp4ep2_mdp_cg_ovlp` | 4/2 | 2 | 32 | 开 | 开 | **开** | 6,451.8 | **211.2** | 161.0 GB | 186.0 GB | 12.62413 |

所有 cell 均完整跑完 20/20，NaN 次数为 0，skipped 次数为 0。没有任何 cell 发生
OOM，全程未开启 decoder 重算。

标注 *window overlap: 开* 的 cell 需要 `MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1`
—— 在把它当作可用配置之前，请先阅读 `mdp_feature_ablations.md`。

## 关于这些数字的说明

**packing 确实生效了。** `s16k_pp2ep4_mdp` 在 GBS 64 下 20 次迭代累计
consumed samples 为 12,802，即**每次迭代约 640 条样本、每个 16384 token 的 bin
里约有 10 条真实序列**，与 lognormal 分布的均值预期一致。native cell 报告的是
64/迭代，那是因为 `consumed_train_samples` 的统计在 benchmark packing 路径上仅
对 MDP 生效；两者消费的 bin 是相同的。

**PP4 不是为吞吐而选的。** GA 16 下 PP4 的流水线气泡为 `(PP-1)/GA` = 18.75%，
而 PP2 只有 6.25%，这正是两个 PP4 cell 远低于 PP1 的原因。它的价值在显存：
186 GB（占 67%）对比 PP1 的 240.9 GB（占 87%）。

**层切分为均分**（10/10 与 10/10/10/10）。不等分切分尚未探索，而且两条路径想要
的方向是相反的 —— native 路径的 stage 0 因为 vision 而过载，MDP 路径的 stage 0
反而比最后一个 stage 轻（后者还额外承担 output projection、loss 与 MTP head）。
按路径分别调优是待办事项。

---

## 附：更早一组 16k 数据（数据构造不同，不可与上表混读）

2026-08 在同一模型、同样 16384 seq length 下测过一组，但**数据构造完全不同**。
那组用 `--dataset-provider mock`：**每个 microbatch 恰好一条 16384 token 的样本，
带一张 224×224 的图（256 个 image token）**，一个 bin 里只有一条满长序列，
greedy packing 退化为空操作；上表则是 lognormal 分布，一个 bin 约 10 条真实序列。
core attention 按 `sum(L^2)` 计算，`1 x 16384^2` 对比 `~10 x 1657^2`，相差约
**100 倍**。因此两组数字不可比，只能各自组内比较。

代码版本 `57274486`，GBS 64。

| cell | PP/EP | DP | GA | MDP | pre-GDR fusion | HybridEP 128-token 分块 | CUDA graph | ms/iter | TFLOP/s | device used |
|---|---|---|---|---|---|---|---|---|---|---|
| `EP8-base-opt` | 1/8 | 8 | 8 | 关 | 开 | 开 | **关** | 8,449.7 | **395.6** | 226.7 GB |
| `B-base-opt` | 2/4 | 4 | 16 | 关 | 开 | 开 | **关** | 10,445.9 | 320.0 | 178.5 GB |
| `B-mdp-opt` | 2/4 | 4 | 16 | **开** | 开 | 开 | **关** | 9,245.7 | 361.6 | 179.5 GB |
| `B-mdp-opt-cg` | 2/4 | 4 | 16 | **开** | 开 | 开 | **开** | 8,522.2 | **392.3** | 183.9 GB |

同拓扑下（PP2/EP4，均未开 CUDA graph）MDP 的增益是 **+11.9%**（320.0 → 361.6）；
再叠加 CUDA graph 到 392.3。

**注意 `EP8-base-opt` 的 395.6 与 `B-mdp-opt-cg` 的 392.3 不是对等条件**：前者是
PP1/EP8、MDP 关闭、**未开 CUDA graph**，后者是 PP2/EP4、MDP 开启、**开了 CUDA
graph**。这组测试中没有 PP1/EP8 + CUDA graph 的 cell，也没有 PP1/EP8 + MDP 的
cell。

完整 recipe 不在本目录中。

---

## 命令行

各 cell 之间只有各自变化的那一个维度不同。请自行补充 `torchrun` / `srun` 与
`../README.md` 中列出的环境变量。

### `s16k_ep8_base` —— PP1/EP8，native —— 绝对性能最高的 cell

Recipe：[`../recipes/qwen35_vl_35b_a3b/s16k_ep8_base.yaml`](../recipes/qwen35_vl_35b_a3b/s16k_ep8_base.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --use-packed-sequence \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}' \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-warmup-steps 2 \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --recompute-vision \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 ============================================================
```

### `s16k_pp2ep4_base` —— PP2/EP4，native

Recipe：[`../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_base.yaml`](../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_base.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --use-packed-sequence \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}' \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-warmup-steps 2 \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --recompute-vision \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 ============================================================
```

### `s16k_pp2ep4_mdp_noovlp` —— PP2/EP4，MDP，CUDA graph 关闭

Recipe：[`../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_mdp_noovlp.yaml`](../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_mdp_noovlp.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --use-packed-sequence \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}' \
    --cuda-graph-impl none \
    --mdp-enable \
    --encoder-recompute-granularity whole \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 ============================================================
```

### `s16k_pp2ep4_mdp` —— PP2/EP4，MDP，CUDA graph

Recipe：[`../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_mdp.yaml`](../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_mdp.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --use-packed-sequence \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}' \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-warmup-steps 2 \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --mdp-enable \
    --encoder-recompute-granularity whole \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 ============================================================
```

### `s16k_pp2ep4_mdp_cg_ovlp` —— PP2/EP4，MDP，CUDA graph + window overlap

Recipe：[`../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_mdp_cg_ovlp.yaml`](../recipes/qwen35_vl_35b_a3b/s16k_pp2ep4_mdp_cg_ovlp.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --use-packed-sequence \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}' \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-warmup-steps 2 \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --mdp-enable \
    --mdp-overlap-window-capture \
    --encoder-recompute-granularity whole \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 ============================================================
```

### `s16k_pp4ep2_base` —— PP4/EP2，native

Recipe：[`../recipes/qwen35_vl_35b_a3b/s16k_pp4ep2_base.yaml`](../recipes/qwen35_vl_35b_a3b/s16k_pp4ep2_base.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --use-packed-sequence \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}' \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-warmup-steps 2 \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --recompute-vision \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 ============================================================
```

### `s16k_pp4ep2_mdp_cg_ovlp` —— PP4/EP2，MDP，CUDA graph + window overlap

Recipe：[`../recipes/qwen35_vl_35b_a3b/s16k_pp4ep2_mdp_cg_ovlp.yaml`](../recipes/qwen35_vl_35b_a3b/s16k_pp4ep2_mdp_cg_ovlp.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --use-packed-sequence \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":512,"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}' \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-warmup-steps 2 \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --mdp-enable \
    --mdp-overlap-window-capture \
    --encoder-recompute-granularity whole \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 ============================================================
```

