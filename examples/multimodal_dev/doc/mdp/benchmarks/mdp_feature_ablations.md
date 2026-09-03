# CUDA graph 与 window overlap 在 MDP 下各值多少

对两个 MDP 侧开关做了单独与组合测量：

- **`--cuda-graph-impl transformer_engine`** 配合
  `--cuda-graph-modules attn moe_router moe_preprocess` —— per-layer CUDA graph。
  在本模型上，`attn` 的 capture 覆盖 gated-delta-net 层，即 40 层 decoder 中的
  30 层（`--linear-attention-freq 4`）。
- **`--mdp-overlap-window-capture`** —— MDP 在当前迭代执行期间，用后台线程预取
  下一次迭代的数据窗口。

## 代理模型 —— 4× GB300，PP1/EP4，GBS 32（GA 8），四种组合

20 层 / 128 专家代理模型，`--dataset-provider mdp_mock`，MDP 开启，
`--encoder-recompute-granularity whole`。开启 CUDA graph 的两组跑了 40 次迭代
而非 20 次（原因见下文竞态说明）；取第 10 次及以后的中位数。

| cell | CUDA graph | window overlap | ms/iter | TFLOP/s | 相对基准 | devUsed |
|---|---|---|---|---|---|---|
| `proxy_mdp_noovlp` | 关 | 关 | 4,602.3 | 390.3 | — | 155.0 GB |
| `proxy_mdp_ovlp` | 关 | **开** | 4,538.4 | 393.3 | +0.8% | 155.4 GB |
| `proxy_mdp_cg` | **开** | 关 | 4,125.9 | 434.7 | **+11.4%** | 165.9 GB |
| `proxy_mdp_ovlp_cg` | **开** | **开** | 4,024.5 | **446.6** | **+14.4%** | 160.1 GB |

作为参照，同样形状下 **MDP 关闭**且不开 CUDA graph 是 376.4 TFLOP/s、154.5 GB，
因此仅 MDP 本身在这里值 +3.7%。

**overlap 单独开启几乎没有收益**（+0.8%，落在约 3% 的 run-to-run 波动内），但
**叠加在 CUDA graph 之上值 +2.7%**，而且还省回 5.8 GB。

## 全量模型 —— 8× GB300，PP2/EP4，GBS 64（GA 16）

| cell | CUDA graph | window overlap | ms/iter | TFLOP/s | 相对无 graph | devUsed |
|---|---|---|---|---|---|---|
| `s16k_pp2ep4_mdp_noovlp` | 关 | 关 | 10,882.5 | 257.0 | — | 179.3 GB |
| `s16k_pp2ep4_mdp` | **开** | 关 | 9,217.2 | 303.6 | **+18.1%** | 210.9 GB |
| `s16k_pp2ep4_mdp_cg_ovlp` | **开** | **开** | 8,900.3 | **314.2** | **+22.3%** | 211.0 GB |

CUDA graph 在全量模型上值 **+18.1%**，高于代理模型上的 +11.4%，代价是
**+31.6 GB** 显存。overlap 在此基础上再加 **+3.5%**，且不增加显存。

**推荐配置**：开启 CUDA graph。window overlap

---

## 命令行

### `proxy_mdp_noovlp` —— 代理模型，MDP，无 graph、无 overlap

Recipe：[`../recipes/proxy/proxy_mdp_noovlp.yaml`](../recipes/proxy/proxy_mdp_noovlp.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b_light \
    --image-token-id 248056 \
    --vision-num-layers 13 \
    --num-layers 20 \
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
    --num-experts 128 \
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
    --expert-model-parallel-size 4 \
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

### `proxy_mdp_ovlp` —— 代理模型，MDP，无 graph、overlap 开启

Recipe：[`../recipes/proxy/proxy_mdp_ovlp.yaml`](../recipes/proxy/proxy_mdp_ovlp.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b_light \
    --image-token-id 248056 \
    --vision-num-layers 13 \
    --num-layers 20 \
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
    --num-experts 128 \
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
    --expert-model-parallel-size 4 \
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
    --cuda-graph-impl none \
    --mdp-enable \
    --encoder-recompute-granularity whole \
    --mdp-overlap-window-capture \
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

### `proxy_mdp_cg` —— 代理模型，MDP，CUDA graph

Recipe：[`../recipes/proxy/proxy_mdp_cg.yaml`](../recipes/proxy/proxy_mdp_cg.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b_light \
    --image-token-id 248056 \
    --vision-num-layers 13 \
    --num-layers 20 \
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
    --num-experts 128 \
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
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --total-seq-length 16384 \
    --train-iters 40 \
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

### `proxy_mdp_ovlp_cg` —— 代理模型，MDP，CUDA graph + overlap（已解除 guard）

Recipe：[`../recipes/proxy/proxy_mdp_ovlp_cg.yaml`](../recipes/proxy/proxy_mdp_ovlp_cg.yaml)

```bash
python pretrain_multimodal.py \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b_light \
    --image-token-id 248056 \
    --vision-num-layers 13 \
    --num-layers 20 \
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
    --num-experts 128 \
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
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --total-seq-length 16384 \
    --train-iters 40 \
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
    --mdp-overlap-window-capture \
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

本页涉及的全量模型 cell 见
[`qwen35vl_35b_a3b_16k.md`](qwen35vl_35b_a3b_16k.md)。
