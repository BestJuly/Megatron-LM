# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Base multimodal model for Megatron VLM training.

Composes a vision encoder and a ``GPTModel`` language decoder. Pipeline
stages split language preprocessing and postprocessing; PP x CP MDP can
replicate the vision encoder on every stage.

Subclasses override ``compute_position_ids()`` for model-specific
position encoding (e.g. MRoPE for Qwen3.5-VL).
"""

import contextlib
from collections import deque
from typing import Optional

import torch
from torch import Tensor

from examples.multimodal_dev.modality_bridge import (
    cp_local_image_positions_and_row_ids_from_cpu_metadata,
    gather_to_inner_dp_zero,
    get_mdp_images_to_language_group,
    reorder_gathered_embeddings,
    scatter_vision_rows_at_positions,
    select_vision_rows_for_cp_rank,
)
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.training import get_args
from megatron.training.utils import get_nvtx_range


def _cp_split_tensor(tensor, seq_dim, cp_size, cp_rank):
    """Zigzag-split *tensor* along *seq_dim* for context parallelism (BSHD).

    Splits the sequence into ``2 * cp_size`` equal chunks, then selects
    chunks ``[cp_rank, 2*cp_size - cp_rank - 1]`` and concatenates them.
    This mirrors ``megatron.core.utils.get_batch_on_this_cp_rank``.
    """
    S = tensor.shape[seq_dim]
    assert S % (2 * cp_size) == 0, (
        f"seq_len {S} not divisible by 2*cp_size={2 * cp_size}"
    )
    tensor = tensor.view(
        *tensor.shape[:seq_dim],
        2 * cp_size,
        S // (2 * cp_size),
        *tensor.shape[seq_dim + 1 :],
    )
    index = torch.zeros(2, dtype=torch.int64, device=tensor.device)
    index[0] = cp_rank
    index[1] = 2 * cp_size - cp_rank - 1
    tensor = tensor.index_select(seq_dim, index)
    tensor = tensor.view(
        *tensor.shape[:seq_dim],
        -1,
        *tensor.shape[seq_dim + 2 :],
    )
    return tensor


class _NoCPGroup:
    """Dummy size-1 process group used to bypass MRoPE's BSHD-style
    zigzag of pre-computed THD freqs (Megatron-Core gap:
    ``MultimodalRotaryEmbedding.forward`` lacks the ``not packed_seq``
    skip that plain ``RotaryEmbedding`` has).
    """

    def size(self):
        return 1

    def rank(self):
        return 0


_NO_CP_GROUP = _NoCPGroup()

# Note: reported ``mtp_1 loss`` drifts ~1.3% from the CP=1 baseline under
# THD+CP. Megatron-Core's logging averages per-rank pre-divided ratios
# with op=AVG, and per-rank num_tokens are unequal after MTP rolling.
# Gradients are correct; only the *logged* value drifts.


def _thd_cp_partition_index(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
    """Per-rank token index for THD + CP via TE's
    ``thd_get_partitioned_indices``.  Cast to int64 so the result can be
    used directly with ``index_select`` regardless of TE's return dtype.
    """
    from transformer_engine.pytorch import cpp_extensions as tex

    idx = tex.thd_get_partitioned_indices(
        cu_seqlens_padded, total_tokens, cp_size, cp_rank,
    )
    return idx.long()


def _zero_dep_on_trainable_params(module):
    """Return scalar zero while keeping trainable vision params in autograd."""
    zero = None
    for param in module.parameters():
        if not param.requires_grad or param.numel() == 0:
            continue
        try:
            if param.untyped_storage().nbytes() < param.numel() * param.element_size():
                continue
        except RuntimeError:
            # Meta/fake tensors (FSDP sharding) raise on untyped_storage().
            continue
        term = param.reshape(-1)[:1].sum() * 0.0
        zero = term if zero is None else zero + term
    return zero


def _replace_pp_replica_id(replica_id, pp_rank: int):
    """Replace only the pipeline coordinate of checkpoint replica metadata."""
    if isinstance(replica_id, int):
        return int(pp_rank)
    if not replica_id:
        return replica_id
    return (int(pp_rank), *tuple(replica_id)[1:])


def _mark_replicated_pp_vision_shards(sharded_state_dict, pp_rank: int) -> None:
    """Make PP0 the main checkpoint replica without changing TP/DP identity."""
    for shard in sharded_state_dict.values():
        if hasattr(shard, "replica_id"):
            shard.replica_id = _replace_pp_replica_id(shard.replica_id, pp_rank)


def _zero_dep_on_tensor(tensor):
    if tensor.numel() == 0:
        return tensor.sum() * 0.0
    return tensor.reshape(-1)[:1].sum() * 0.0


def _module_has_partial_storage(module) -> bool:
    if module is None:
        return False
    for param in module.parameters():
        try:
            if param.untyped_storage().nbytes() < param.numel() * param.element_size():
                return True
        except RuntimeError:
            # Meta/fake tensors (FSDP sharding) raise on untyped_storage().
            return True
    return False


def _empty_vision_rank_must_call_module(module) -> bool:
    return bool(getattr(get_args(), "use_megatron_fsdp", False)) or _module_has_partial_storage(
        module
    )


def _dummy_vision_inputs(vision_model, device, dtype):
    patch_embed = vision_model.patch_embed
    patch_size = int(patch_embed.patch_size)
    temporal_patch_size = int(patch_embed.temporal_patch_size)
    in_channels = int(patch_embed.in_channels)
    h = w = int(getattr(vision_model, "spatial_merge_size", 2))
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    pixel_values = torch.zeros((h * w, patch_dim), dtype=dtype, device=device)
    grid_thw = torch.tensor([[1, h, w]], dtype=torch.int64, device=device)
    return pixel_values, grid_thw


class MultimodalModel(MegatronModule):
    """Base class for multimodal vision-language models.

    Composes a pre-constructed vision encoder and a ``GPTModel`` language
    decoder with Megatron pipeline stage ownership.

    Args:
        language_config: ``TransformerConfig`` for the language decoder.
        language_spec: ``ModuleSpec`` for decoder transformer layers.
        vision_encoder: Pre-constructed vision encoder module.
        vocab_size: Language model vocabulary size.
        max_sequence_length: Maximum sequence length.
        image_token_id: Token ID for image placeholder tokens.
        position_embedding_type: Position embedding type for the decoder.
        rotary_percent: Fraction of hidden dim for RoPE.
        rotary_base: Base frequency for RoPE.
        mrope_section: MRoPE channel sections.
        mtp_block_spec: Optional MTP block spec.
        parallel_output: Keep outputs split across TP ranks.
        share_embeddings_and_output_weights: Tie input/output embeddings.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_spec: ModuleSpec,
        vision_encoder: Optional[MegatronModule],
        vocab_size: int,
        max_sequence_length: int,
        image_token_id: int,
        position_embedding_type: str = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        mrope_section: list = None,
        mtp_block_spec: ModuleSpec = None,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=language_config)

        self.image_token_id = image_token_id
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        self.vision_model = vision_encoder
        self.language_model = GPTModel(
            config=language_config,
            transformer_layer_spec=language_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                share_embeddings_and_output_weights
            ),
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )
        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    @property
    def embedding_activation_buffer(self):
        return self.language_model.embedding_activation_buffer

    @property
    def grad_output_buffer(self):
        return self.language_model.grad_output_buffer

    @property
    def output_layer(self):
        return self.language_model.output_layer

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Build checkpoint state with PP-replicated vision saved once."""
        sharded_state = {}
        if self.vision_model is not None:
            vision_state = sharded_state_dict_default(
                self.vision_model,
                f'{prefix}vision_model.',
                sharded_offsets,
                metadata,
            )
            if bool(getattr(self, '_mdp_pp_cp_inner', False)):
                _mark_replicated_pp_vision_shards(
                    vision_state,
                    parallel_state.get_pipeline_model_parallel_rank(),
                )
            sharded_state.update(vision_state)
        sharded_state.update(
            sharded_state_dict_default(
                self.language_model,
                f'{prefix}language_model.',
                sharded_offsets,
                metadata,
            )
        )
        return sharded_state

    def mdp_pp_cp_sidecar_compute_vision(
        self,
        *,
        pixel_values: Optional[Tensor],
        image_grid_thw: Optional[Tensor],
        mdp_cp_local_plan=None,
    ):
        """Run the CP-local bridge supplied by the preceding MDP layer."""
        del mdp_cp_local_plan
        if not (
            bool(getattr(self, '_mdp_pp_cp_inner', False))
            or bool(getattr(self, '_mdp_cp_fused_sidecar', False))
        ):
            raise RuntimeError("vision sidecar requires PP x CP or CP-fused MDP mode")
        return self._run_mdp_vision_bridge(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def pipeline_sidecar_pre_forward(
        self,
        *,
        data_iterator=None,
        current_microbatch=None,
        num_microbatches=None,
        forward_only=False,
    ) -> None:
        """Precompute one ordered MDP vision window before pipeline P2P."""
        if not bool(getattr(self, '_pipeline_sidecar_enabled', False)):
            return
        from examples.multimodal_dev.forward_step import (
            build_mdp_pp_cp_sidecar_cache,
            build_mdp_pp_cp_sidecar_cache_window,
        )
        from examples.multimodal_dev.sidecar_prefetch import (
            sidecar_prefetch_window_count,
            validate_fused_vision_window,
        )

        args = get_args()
        max_sequence_length = int(
            getattr(args, "mdp_vision_encoder_max_sequence_length", 0) or 0
        )
        fused_window = validate_fused_vision_window(
            getattr(args, "mdp_fused_vision_window", False), max_sequence_length
        )
        queue = getattr(self, '_mdp_pp_cp_sidecar_cache', None)
        if queue is None:
            queue = deque()
            object.__setattr__(self, '_mdp_pp_cp_sidecar_cache', queue)

        count = sidecar_prefetch_window_count(
            fused_window,
            current_microbatch=current_microbatch,
            num_microbatches=num_microbatches,
        )
        if count <= 0:
            return

        if fused_window:
            queue.extend(
                build_mdp_pp_cp_sidecar_cache_window(
                    data_iterator=data_iterator,
                    model=self,
                    vp_stage=self.vp_stage,
                    count=count,
                    max_sequence_length=max_sequence_length,
                    forward_only=forward_only,
                )
            )
            return

        for _ in range(count):
            queue.append(
                build_mdp_pp_cp_sidecar_cache(
                    data_iterator=data_iterator,
                    model=self,
                    vp_stage=self.vp_stage,
                    forward_only=forward_only,
                )
            )

    def mdp_pp_cp_sidecar_pop_cache(self):
        queue = getattr(self, '_mdp_pp_cp_sidecar_cache', None)
        if not queue:
            return None
        return queue.popleft()

    def mdp_pp_cp_sidecar_activate_cache(self, cache) -> None:
        vision_embeddings = None if cache is None else cache.get('vision_embeddings')
        object.__setattr__(
            self,
            '_mdp_pp_cp_active_vision_embeddings',
            vision_embeddings,
        )
        if (
            not torch.is_grad_enabled()
            or bool(cache.get('forward_only', False))
        ):
            return
        fused_entries = cache.get('fused_backward_entries')
        if fused_entries is not None:
            queue = getattr(self, '_mdp_pp_cp_sidecar_backward_cache', None)
            if queue is None:
                queue = deque()
                object.__setattr__(
                    self,
                    '_mdp_pp_cp_sidecar_backward_cache',
                    queue,
                )
            queue.append({"kind": "fused_backward", "entries": fused_entries})
            return
        if (
            not self.pre_process
            and torch.is_tensor(vision_embeddings)
        ):
            queue = getattr(self, '_mdp_pp_cp_sidecar_backward_cache', None)
            if queue is None:
                queue = deque()
                object.__setattr__(
                    self,
                    '_mdp_pp_cp_sidecar_backward_cache',
                    queue,
                )
            queue.append(_zero_dep_on_tensor(vision_embeddings))

    def pipeline_sidecar_post_backward(self) -> None:
        """Backpropagate the matching generic or fused vision dependency."""
        if not bool(getattr(self, '_pipeline_sidecar_enabled', False)):
            return
        if bool(getattr(self, '_pp_cp_batch_sidecar', False)):
            return
        queue = getattr(self, '_mdp_pp_cp_sidecar_backward_cache', None)
        if not queue:
            if self.pre_process:
                return
            raise RuntimeError("PP x CP vision sidecar backward cache is empty")
        dependency = queue.popleft()
        if (
            isinstance(dependency, dict)
            and dependency.get("kind") == "fused_backward"
        ):
            from examples.multimodal_dev.fused_vision_window import (
                fused_vision_post_backward,
            )

            fused_vision_post_backward(self, dependency.get("entries") or [])
            return
        if self.pre_process:
            return
        if dependency.requires_grad:
            dependency.backward()

    def set_input_tensor(self, input_tensor):
        """Route pipeline input tensors to the language decoder."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        self.language_model.set_input_tensor(input_tensor[0])

    def _run_mdp_vision_bridge(
        self,
        *,
        pixel_values: Optional[Tensor],
        image_grid_thw: Optional[Tensor],
    ) -> Tensor:
        """Encode this rank's owner images and gather canonical vision rows."""
        device = (
            pixel_values.device
            if pixel_values is not None
            else next(self.language_model.parameters()).device
        )
        gather_group = get_mdp_images_to_language_group(self)
        rank_assignment = getattr(self, "_mdp_rank_assignment", None)
        if gather_group is None:
            raise RuntimeError("MDP CP-local mode requires an InnerDP process group")
        if rank_assignment is None:
            raise RuntimeError(
                "MDP requires _mdp_rank_assignment before the vision bridge. "
                "The loader must publish its LPT prepartition metadata."
            )

        has_local_imgs = pixel_values is not None and pixel_values.numel() > 0
        zero_dep = None
        if has_local_imgs:
            with get_nvtx_range()("MultimodalModel/vision_encoder"):
                local_embeddings = self.vision_model(pixel_values, image_grid_thw)
            if (
                bool(getattr(self, "_mdp_pp_cp_inner", False))
                and torch.is_grad_enabled()
                and not local_embeddings.requires_grad
            ):
                # Frozen vision replicas still need the bridge autograd node
                # so every PP x CP rank enters the gather backward collective.
                local_embeddings = local_embeddings.detach().requires_grad_(True)
        else:
            lm_param = next(self.language_model.parameters())
            hidden_size = int(getattr(self.config, "hidden_size", 0))
            if hidden_size <= 0:
                raise RuntimeError(
                    "MDP could not infer language hidden size for an empty "
                    "local image shard"
                )
            local_embeddings = torch.empty(
                (0, hidden_size), dtype=lm_param.dtype, device=device
            ).requires_grad_(True)
            if _empty_vision_rank_must_call_module(self.vision_model):
                dummy_pixels, dummy_grid = _dummy_vision_inputs(
                    self.vision_model, device, lm_param.dtype
                )
                zero_dep = _zero_dep_on_tensor(
                    self.vision_model(dummy_pixels, dummy_grid)
                )
            else:
                zero_dep = _zero_dep_on_trainable_params(self.vision_model)

        global_row_counts = getattr(
            self, "_mdp_rank_assignment_row_counts", None
        )
        vision_embeddings = gather_to_inner_dp_zero(
            local_embeddings=local_embeddings,
            rank_assignment=rank_assignment,
            encoder_dp_group=gather_group,
            global_per_image_row_counts=global_row_counts,
            local_zero_dep=zero_dep if not has_local_imgs else None,
            return_zero_dependency_only=(
                not self.pre_process
                and bool(getattr(self, "_mdp_pp_cp_inner", False))
            ),
        )
        if not self.pre_process and bool(
            getattr(self, "_mdp_pp_cp_inner", False)
        ):
            return vision_embeddings
        if global_row_counts is not None:
            local_per_image_row_counts = None
        elif image_grid_thw is not None and image_grid_thw.numel() > 0:
            merge_size = int(getattr(get_args(), "vision_spatial_merge_size", 2))
            merge_sq = max(merge_size * merge_size, 1)
            grid_i64 = image_grid_thw.to(dtype=torch.int64)
            local_per_image_row_counts = (
                grid_i64[:, 0] * grid_i64[:, 1] * grid_i64[:, 2]
            ) // merge_sq
        else:
            local_per_image_row_counts = torch.zeros(
                (0,), dtype=torch.int64, device=local_embeddings.device
            )
        return reorder_gathered_embeddings(
            gathered_embeddings=vision_embeddings,
            local_per_image_row_counts=local_per_image_row_counts,
            rank_assignment=rank_assignment,
            group=gather_group,
            global_per_image_row_counts=global_row_counts,
        )

    def _scatter_vision_embeddings(
        self,
        input_ids: Tensor,
        text_embeddings: Tensor,
        vision_embeddings: Tensor,
        packed_seq_params=None,
    ) -> Tensor:
        """Replace image-token positions with vision embeddings.

        Handles sequence parallelism and applies CP before the final SP shard.

        Args:
            input_ids: ``[B, S]`` token IDs.
            text_embeddings: ``[S, B, D]`` (or ``[S/TP, B, D]`` with SP).
            vision_embeddings: ``[num_visual_tokens, D]``.

        Returns:
            Combined embeddings, same shape as *text_embeddings*.
        """
        with get_nvtx_range()("MultimodalModel/scatter_vision_embeddings"):
            sp = (
                self.config.sequence_parallel
                and parallel_state.get_tensor_model_parallel_world_size() > 1
            )

            if sp:
                text_embeddings = (
                    tensor_parallel.gather_from_sequence_parallel_region(
                        text_embeddings, tensor_parallel_output_grad=False,
                    )
                )

            combined = text_embeddings.transpose(0, 1).contiguous()
            image_mask = input_ids == self.image_token_id
            mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
            combined = combined.masked_scatter(mask_expanded, vision_embeddings)

            cp_size = parallel_state.get_context_parallel_world_size()
            if sp and cp_size > 1:
                index = self._cp_local_thd_index_for_length(
                    combined.size(1), packed_seq_params
                )
                if index is not None:
                    combined = combined.index_select(1, index)
                else:
                    combined = _cp_split_tensor(
                        combined,
                        seq_dim=1,
                        cp_size=cp_size,
                        cp_rank=parallel_state.get_context_parallel_rank(),
                    )
                object.__setattr__(
                    self, "_decoder_input_already_cp_partitioned", True
                )
            combined = combined.transpose(0, 1).contiguous()

            if sp:
                combined = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined
                )

            return combined

    def _cp_local_thd_index_for_length(
        self, total_tokens: int, packed_seq_params=None
    ):
        if packed_seq_params is None:
            return None
        cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
        if cu_seqlens_padded is None:
            cu_seqlens_padded = packed_seq_params.cu_seqlens_q
        return _thd_cp_partition_index(
            cu_seqlens_padded,
            int(total_tokens),
            parallel_state.get_context_parallel_world_size(),
            parallel_state.get_context_parallel_rank(),
        )

    def _partition_cp_local_input_ids(
        self, input_ids: Tensor, packed_seq_params=None
    ) -> Tensor:
        index = self._cp_local_thd_index_for_length(
            input_ids.size(1), packed_seq_params
        )
        if index is not None:
            return input_ids.index_select(1, index)
        return _cp_split_tensor(
            input_ids,
            seq_dim=1,
            cp_size=parallel_state.get_context_parallel_world_size(),
            cp_rank=parallel_state.get_context_parallel_rank(),
        )

    def _cp_local_merge_decoder_input(
        self,
        input_ids: Tensor,
        text_embeddings: Tensor,
        vision_embeddings: Tensor,
        packed_seq_params=None,
        mdp_cp_local_plan=None,
    ) -> Tensor:
        """Merge canonical vision rows directly into this rank's CP shard."""
        args = get_args()
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        if cp_size <= 1:
            raise RuntimeError(
                "MDP CP-local merge requires context_parallel_size > 1"
            )
        if getattr(args, "dynamic_context_parallel", False) or (
            packed_seq_params is not None
            and getattr(packed_seq_params, "local_cp_size", None) is not None
        ):
            raise RuntimeError(
                "MDP CP-local merge requires static context parallelism"
            )
        if not isinstance(mdp_cp_local_plan, dict):
            raise RuntimeError(
                "MDP CP-local merge requires _mdp_cp_local_plan from the loader"
            )

        is_thd = (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
        )
        (
            image_positions_cpu,
            cp_local_row_ids_cpu,
            global_img_tokens,
        ) = cp_local_image_positions_and_row_ids_from_cpu_metadata(
            image_positions=mdp_cp_local_plan.get("image_positions"),
            input_shape=mdp_cp_local_plan.get("input_shape"),
            cp_size=cp_size,
            cp_rank=cp_rank,
            cu_seqlens_padded=(
                mdp_cp_local_plan.get("cu_seqlens_padded") if is_thd else None
            ),
        )
        image_positions_cp = image_positions_cpu.to(
            device=input_ids.device, dtype=torch.int64
        )
        cp_local_row_ids = cp_local_row_ids_cpu.to(
            device=input_ids.device, dtype=torch.int64
        )
        if int(global_img_tokens) != int(vision_embeddings.shape[0]):
            raise RuntimeError(
                "MDP CP-local merge CPU row plan mismatch: expected "
                f"{global_img_tokens} global vision rows but got "
                f"{int(vision_embeddings.shape[0])}"
            )

        vision_embeddings_cp = select_vision_rows_for_cp_rank(
            vision_embeddings, cp_local_row_ids, validate_indices=False
        )
        sp = (
            self.config.sequence_parallel
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        )
        if sp:
            text_embeddings = tensor_parallel.gather_from_sequence_parallel_region(
                text_embeddings, tensor_parallel_output_grad=False
            )
        decoder_input = scatter_vision_rows_at_positions(
            text_embeddings, vision_embeddings_cp, image_positions_cp
        )
        if sp:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                decoder_input
            )
        if vision_embeddings.requires_grad and vision_embeddings_cp.numel() == 0:
            decoder_input = decoder_input + _zero_dep_on_tensor(vision_embeddings)
        return decoder_input

    def compute_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor] = None,
        packed_seq_params=None,
    ) -> Tensor:
        """Compute position IDs.  Override for MRoPE etc.

        Default: simple sequential positions.  ``packed_seq_params`` is
        accepted for subclass compatibility (e.g. MRoPE in THD mode).
        """
        B, S = input_ids.shape
        return (
            torch.arange(S, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

    def _cp_split_for_forward(
        self,
        *,
        decoder_input,
        input_ids,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        packed_seq_params,
        decoder_input_already_cp_partitioned=False,
    ):
        """Apply CP split to model-forward inputs.

        BSHD path zigzag-splits each tensor along its seq dim.  THD path
        partitions per-sample via ``tex.thd_get_partitioned_indices`` so
        chunks line up with ``cu_seqlens_q_padded`` boundaries.
        ``position_ids`` and ``attention_mask`` are NOT split in THD —
        MRoPE returns full freqs and TE attention's
        ``_apply_rotary_pos_emb_thd`` does the per-sample CP zigzag
        itself via ``_get_thd_freqs_on_this_cp_rank``.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1:
            return (
                decoder_input, input_ids, labels, loss_mask,
                attention_mask, position_ids,
            )
        cp_rank = parallel_state.get_context_parallel_rank()

        if packed_seq_params is not None:
            total_tokens = (
                input_ids.shape[1]
                if decoder_input_already_cp_partitioned
                else (
                    decoder_input.shape[0]
                    if decoder_input is not None
                    else input_ids.shape[1]
                )
            )
            idx = _thd_cp_partition_index(
                packed_seq_params.cu_seqlens_q_padded,
                total_tokens, cp_size, cp_rank,
            )
            if decoder_input is not None and not decoder_input_already_cp_partitioned:
                decoder_input = decoder_input.index_select(0, idx)
            if input_ids is not None:
                input_ids = input_ids.index_select(1, idx)
            if labels is not None:
                labels = labels.index_select(1, idx)
            if loss_mask is not None:
                loss_mask = loss_mask.index_select(1, idx)
        else:
            def _split(t, seq_dim):
                return None if t is None else _cp_split_tensor(
                    t, seq_dim=seq_dim, cp_size=cp_size, cp_rank=cp_rank,
                )
            if not decoder_input_already_cp_partitioned:
                decoder_input = _split(decoder_input, 0)
            input_ids = _split(input_ids, 1)
            labels = _split(labels, 1)
            loss_mask = _split(loss_mask, 1)
            attention_mask = _split(attention_mask, 1)

        return (
            decoder_input, input_ids, labels, loss_mask,
            attention_mask, position_ids,
        )

    @contextlib.contextmanager
    def _thd_mrope_no_cp_override(self, packed_seq_params):
        """Force ``rotary_pos_emb.cp_group`` to size 1 for the wrapped
        forward call so MRoPE returns full-length freqs in THD mode.
        Attention then applies per-sample CP zigzag itself via
        ``_apply_rotary_pos_emb_thd``.  Done by direct mutation rather
        than via ``packed_seq_params.cp_group`` so MTP's CP-aware roll
        (which reads that field) still sees the real CP group.
        """
        mrope = (
            getattr(self.language_model, "rotary_pos_emb", None)
            if packed_seq_params is not None
            and parallel_state.get_context_parallel_world_size() > 1
            else None
        )
        saved = getattr(mrope, "cp_group", None) if mrope is not None else None
        if mrope is not None:
            mrope.cp_group = _NO_CP_GROUP
        try:
            yield
        finally:
            if mrope is not None:
                mrope.cp_group = saved

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_ids: ``[B, S]`` token IDs (or ``[1, T]`` in THD mode).
            position_ids: ``[3, B, S]`` for MRoPE or ``[B, S]``
                (``[3, 1, T]`` / ``[1, T]`` in THD mode).
            attention_mask: ``[B, S]`` attention mask (None in THD).
            labels: ``[B, S]`` target token IDs (``[1, T]`` in THD).
            loss_mask: ``[B, S]`` mask for loss (``[1, T]`` in THD).
            pixel_values: Preprocessed image pixels.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.
            decoder_input: Pre-computed decoder input (skip embed).
            packed_seq_params: ``PackedSeqParams`` for THD attention.

        Returns:
            Loss tensor (post_process=True) or hidden states.
        """
        if position_ids is None and input_ids is not None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                packed_seq_params=packed_seq_params,
            )

        object.__setattr__(self, "_decoder_input_already_cp_partitioned", False)

        mdp_enabled = bool(getattr(self, "_mdp_enabled", False))
        cp_local_mdp = (
            mdp_enabled and parallel_state.get_context_parallel_world_size() > 1
        )
        sidecar_active = (
            bool(getattr(self, "_mdp_pp_cp_inner", False))
            or bool(getattr(self, "_mdp_cp_fused_sidecar", False))
        )
        vision_embeddings = None
        if sidecar_active:
            active_vision_embeddings = getattr(
                self, "_mdp_pp_cp_active_vision_embeddings", None
            )
            if active_vision_embeddings is None:
                raise RuntimeError(
                    "MDP vision sidecar cache was not activated before model forward"
                )
            object.__setattr__(
                self, "_mdp_pp_cp_active_vision_embeddings", None
            )
            if self.pre_process:
                vision_embeddings = active_vision_embeddings
        elif mdp_enabled:
            vision_embeddings = self._run_mdp_vision_bridge(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        elif (
            self.pre_process
            and self.vision_model is not None
            and pixel_values is not None
            and pixel_values.numel() > 0
        ):
            vision_embeddings = self.vision_model(
                pixel_values, image_grid_thw,
            )

        decoder_input_already_cp_partitioned = False
        if decoder_input is None and self.language_model is not None and self.pre_process:
            input_ids_for_embedding = (
                self._partition_cp_local_input_ids(input_ids, packed_seq_params)
                if cp_local_mdp
                else input_ids
            )
            text_embeddings = self.language_model.embedding(
                input_ids=input_ids_for_embedding, position_ids=None,
            )

            if vision_embeddings is not None:
                if cp_local_mdp:
                    decoder_input = self._cp_local_merge_decoder_input(
                        input_ids=input_ids_for_embedding,
                        text_embeddings=text_embeddings,
                        vision_embeddings=vision_embeddings,
                        packed_seq_params=packed_seq_params,
                        mdp_cp_local_plan=kwargs.get("mdp_cp_local_plan"),
                    )
                    decoder_input_already_cp_partitioned = True
                else:
                    decoder_input = self._scatter_vision_embeddings(
                        input_ids,
                        text_embeddings,
                        vision_embeddings,
                        packed_seq_params=packed_seq_params,
                    )
                    decoder_input_already_cp_partitioned = bool(
                        getattr(
                            self, "_decoder_input_already_cp_partitioned", False
                        )
                    )
            else:
                decoder_input = text_embeddings

        (
            decoder_input, input_ids, labels, loss_mask,
            attention_mask, position_ids,
        ) = self._cp_split_for_forward(
            decoder_input=decoder_input,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            decoder_input_already_cp_partitioned=(
                decoder_input_already_cp_partitioned
            ),
        )

        with self._thd_mrope_no_cp_override(packed_seq_params):
            return self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
            )

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Build the inner GPT schedule plan after multimodal preprocessing."""
        del kwargs
        if position_ids is None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                packed_seq_params=packed_seq_params,
            )

        pre_process = getattr(
            self,
            "pre_process",
            getattr(self.language_model, "pre_process", True),
        )
        vision_embeddings = None
        if (
            pre_process
            and self.vision_model is not None
            and pixel_values is not None
        ):
            vision_embeddings = self.vision_model(
                pixel_values, image_grid_thw,
            )

        if (
            pre_process
            and decoder_input is None
            and self.language_model is not None
        ):
            text_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None,
            )
            if vision_embeddings is not None:
                decoder_input = self._scatter_vision_embeddings(
                    input_ids, text_embeddings, vision_embeddings,
                )
            else:
                decoder_input = text_embeddings

        (
            decoder_input, input_ids, labels, loss_mask,
            attention_mask, position_ids,
        ) = self._cp_split_for_forward(
            decoder_input=decoder_input,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )

        return self.language_model.build_schedule_plan(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )
