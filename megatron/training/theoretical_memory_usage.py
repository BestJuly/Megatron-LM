# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""


import math
from .utils import print_rank_0

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


# ────────────────────────────────────────────────────────────────────────────
# Attention-variant parameter helpers
# ────────────────────────────────────────────────────────────────────────────

def _compute_standard_attention_params(args, H, norm_multiplier):
    """Standard GQA / MHA attention: QKV + output projection, optional output gate, QK layernorm."""
    if args.multi_latent_attention:
        assert not args.group_query_attention
        if args.q_lora_rank is None:
            q_term = H * args.num_attention_heads * (
                args.qk_head_dim + args.qk_pos_emb_head_dim
            )
        else:
            q_term = args.q_lora_rank * (
                H + args.num_attention_heads * (
                    args.qk_head_dim + args.qk_pos_emb_head_dim
                ) + 1
            )
        return (
            q_term
            + args.kv_lora_rank * (
                H + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) + 1
            )
            + H * args.qk_pos_emb_head_dim
            + (args.num_attention_heads * args.v_head_dim) * H
        )

    query_projection_size = args.kv_channels * args.num_attention_heads
    kv_projection_size = args.kv_channels * args.num_query_groups

    qkv_out_dim = query_projection_size + 2 * kv_projection_size
    if getattr(args, 'attention_output_gate', False):
        qkv_out_dim += query_projection_size  # gate has same dim as Q

    params = H * qkv_out_dim                  # QKV (+ gate) projection
    params += query_projection_size * H       # output projection

    if getattr(args, 'qk_layernorm', False):
        params += 2 * args.kv_channels * norm_multiplier

    return params


def _compute_gdn_attention_params(args, H, norm_multiplier):
    """Gated Delta Net (GDN) linear attention: in_proj, conv1d, SSM, output norm + projection."""
    qk_dim = args.linear_key_head_dim * args.linear_num_key_heads
    v_dim = args.linear_value_head_dim * args.linear_num_value_heads

    # in_proj: H -> (q, k, v, gate, beta, alpha)
    in_proj_out_dim = qk_dim * 2 + v_dim * 2 + args.linear_num_value_heads * 2
    params = H * in_proj_out_dim

    # Depthwise Conv1d on (q, k, v)
    conv_dim = qk_dim * 2 + v_dim
    params += conv_dim * args.linear_conv_kernel_dim

    # SSM scalars: dt_bias + A_log
    params += 2 * args.linear_num_value_heads

    # Output norm (RMSNorm with hidden_size=value_head_dim)
    params += args.linear_value_head_dim * norm_multiplier

    # Output projection: v_dim -> H
    params += v_dim * H

    return params


def _get_experimental_attention_pattern(args):
    """Per-layer pattern for experimental attention variants (1 = variant, 0 = standard).

    Returns (has_variant, pattern) where *pattern* has the same length as args.num_layers.
    To add a new variant, extend the dispatch in this function and add a corresponding
    ``_compute_<variant>_attention_params`` helper above.
    """
    variant = getattr(args, 'experimental_attention_variant', None)
    freq = getattr(args, 'linear_attention_freq', None)

    if variant == 'gated_delta_net' and freq is not None:
        if isinstance(freq, int):
            pattern = [0 if ((i + 1) % freq == 0) else 1 for i in range(args.num_layers)]
        elif isinstance(freq, list):
            pattern = freq
        else:
            pattern = [0] * args.num_layers
        return True, pattern

    # Future variants can be added here with elif branches.

    return False, [0] * args.num_layers


def _compute_experimental_attention_params(args, H, norm_multiplier):
    """Dispatch to the correct experimental attention variant parameter counter.

    To add a new variant, implement ``_compute_<variant>_attention_params(args, H, norm_multiplier)``
    and add a branch here.
    """
    variant = getattr(args, 'experimental_attention_variant', None)
    if variant == 'gated_delta_net':
        return _compute_gdn_attention_params(args, H, norm_multiplier)
    # Future variants can be added here with elif branches.
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Main parameter-counting entry point
# ────────────────────────────────────────────────────────────────────────────

def compute_weight_and_optimizer_memory(args, verbose=False):
    H = args.hidden_size

    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads

    # Normalization: RMSNorm has weight only; LayerNorm has weight + bias.
    normalization = getattr(args, 'normalization', 'LayerNorm')
    norm_multiplier = 1 if normalization == 'RMSNorm' else 2
    layernorm_params_per_layer = 2 * H * norm_multiplier  # input_layernorm + pre_mlp_layernorm

    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )

    # --- MoE layer pattern ---
    if args.num_experts is not None:
        if isinstance(args.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
            ]
        elif isinstance(args.moe_layer_freq, list):
            moe_layer_pattern = args.moe_layer_freq
            assert len(moe_layer_pattern) == args.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {args.num_layers}, "
                f"current moe layer pattern: {args.moe_layer_freq}"
            )
        moe_ffn_hidden_size = args.moe_ffn_hidden_size
    else:
        moe_layer_pattern = [0] * args.num_layers
        moe_ffn_hidden_size = 0

    # --- Attention variant pattern ---
    has_variant, variant_pattern = _get_experimental_attention_pattern(args)

    # --- Attention terms ---
    full_attn_term = _compute_standard_attention_params(args, H, norm_multiplier)
    variant_attn_term = (
        _compute_experimental_attention_params(args, H, norm_multiplier) if has_variant else 0
    )

    # --- MLP terms ---
    dense_mlp_term = 2 * H * args.ffn_hidden_size * gated_linear_multiplier

    moe_mlp_term = (
        2 * H * moe_ffn_hidden_size * num_experts * gated_linear_multiplier  # routed experts
        + 2 * H * shared_expert_ffn_hidden_size * gated_linear_multiplier    # shared expert
        + num_experts * H                                                     # router weights
    )
    if getattr(args, 'moe_shared_expert_gate', False) and shared_expert_ffn_hidden_size > 0:
        moe_mlp_term += H  # shared expert gate: Linear(H, 1, bias=False)

    # --- Per-layer parameter counts for (is_variant_attn, is_moe) combos ---
    layer_params = {
        (False, False): full_attn_term    + dense_mlp_term + layernorm_params_per_layer,
        (False, True):  full_attn_term    + moe_mlp_term   + layernorm_params_per_layer,
        (True,  False): variant_attn_term + dense_mlp_term + layernorm_params_per_layer,
        (True,  True):  variant_attn_term + moe_mlp_term   + layernorm_params_per_layer,
    }

    layer_type_counts = {k: 0 for k in layer_params}
    for i in range(args.num_layers):
        key = (variant_pattern[i] == 1, moe_layer_pattern[i] == 1)
        layer_type_counts[key] += 1

    # --- Transformer block ---
    final_layernorm = H * norm_multiplier
    num_parameters_in_transformer_block = final_layernorm
    for key, count in layer_type_counts.items():
        num_parameters_in_transformer_block += layer_params[key] * count

    # --- MTP block ---
    if args.mtp_num_layers is not None and args.mtp_num_layers > 0:
        last_layer_key = (variant_pattern[-1] == 1, moe_layer_pattern[-1] == 1)
        mtp_transformer_layer_params = layer_params[last_layer_key]
        mtp_overhead = (
            2 * H * norm_multiplier  # enorm + hnorm
            + 2 * H * H             # eh_proj: Linear(2*H -> H, bias=False)
            + H * norm_multiplier    # final_layernorm
        )
        num_parameters_in_mtp_block = (
            (mtp_transformer_layer_params + mtp_overhead) * args.mtp_num_layers
        )
    else:
        num_parameters_in_mtp_block = 0

    # --- Embeddings ---
    embedding_size = H * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    num_total_parameters = (
        num_parameters_in_transformer_block
        + num_parameters_in_mtp_block
        + num_parameters_in_embedding_layers
    )
    if verbose:
        print(
            f"Number of parameters in transformer block in billions: "
            f"{num_parameters_in_transformer_block / 10**9: .2f}"
        )
        if has_variant:
            variant_name = args.experimental_attention_variant
            n_var = layer_type_counts[(True, False)] + layer_type_counts[(True, True)]
            n_full = layer_type_counts[(False, False)] + layer_type_counts[(False, True)]
            print(f"  ({n_var} {variant_name} layers, {n_full} full attention layers)")
        if args.mtp_num_layers is not None:
            print(
                f"Number of parameters in mtp block in billions: "
                f"{num_parameters_in_mtp_block / 10**9: .2f}"
            )
        print(
            f"Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 mtp block + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_block / args.pipeline_model_parallel_size)
        + num_parameters_in_mtp_block
        + embedding_size
    ) / args.tensor_model_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_block / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
        18 + (4 * (args.ffn_hidden_size / args.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / args.tensor_model_parallel_size


def compute_activation_memory_without_sp(args, num_microbatches, verbose=False):
    """Compute activation memory without sequence parallelism"""

    # 4. Compute per-layer memory
    per_layer_memory = args.seq_length * args.micro_batch_size * args.hidden_size * (10 + (24 / args.tensor_model_parallel_size))

    if verbose:
        print(
            f"Activation memory footprint per transformer layer (precise, without SP): "
            f"{per_layer_memory / NUM_BYTES_IN_MEGABYTE:.1f} MB"
        )

    # 5. Multiply by number of layers
    total_activation_memory = per_layer_memory * args.num_layers

    # 6. Add embedding activations
    # Input to embedding (pp_size microbatches in flight)
    total_activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight)
    total_activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # 7. Handle pipeline parallelism schedules
    # Multiply by interleaved PP memory factor
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        total_activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            total_activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    # 8. Add output layer memory if needed
    if args.pipeline_model_parallel_size == 1:
        # Logits calculation
        logits_size = args.seq_length * args.micro_batch_size * args.padded_vocab_size
        # The output projection is partitioned across TP
        logits_size /= args.tensor_model_parallel_size

        # Outputs from final layer norm
        final_ln_output = args.seq_length * args.micro_batch_size * args.hidden_size

        total_activation_memory += (logits_size + final_ln_output) * 2  # multiply by 2 for bytes

    # 9. Add buffer for optimizer and miscellaneous temporaries (5% overhead)
    overhead_factor = 1.05
    total_activation_memory *= overhead_factor

    return total_activation_memory


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    if args.is_hybrid_model:
        print("Theoretical memory footprints not yet supported for hybrid Mamba-Transformer models.")
        return

    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )

    # Choose the appropriate activation memory calculation based on parallelism strategy
    if args.sequence_parallel and args.recompute_granularity == 'selective':
        print_rank_0("compute_activation_memory with SP")
        activation_memory = (
            compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
            / NUM_BYTES_IN_MEGABYTE
        )
    else:
        print_rank_0("compute_activation_memory_without_sp")
        activation_memory = (
            compute_activation_memory_without_sp(args, num_microbatches=num_microbatches, verbose=verbose)
            / NUM_BYTES_IN_MEGABYTE
        )

    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )

    return weight_and_optimizer_memory, activation_memory, total_memory
