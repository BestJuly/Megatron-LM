# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Base multimodal model for FSDP + EP and PP training.

Composes a vision encoder and a ``HybridModel`` language decoder.  The
vision encoder is built only on the first PP stage; the language
decoder spans all PP stages following standard Megatron PP layout
flags.

Subclasses override ``compute_position_ids()`` for model-specific
position encoding (e.g. MRoPE for Qwen3.5-VL).
"""

import contextlib
from typing import Optional

import torch
from torch import Tensor

from examples.multimodal_dev.observability import nvtx_phase
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def _cp_split_tensor(tensor, seq_dim, cp_size, cp_rank):
    """Zigzag-split *tensor* along *seq_dim* for context parallelism (BSHD).

    Splits the sequence into ``2 * cp_size`` equal chunks, then selects
    chunks ``[cp_rank, 2*cp_size - cp_rank - 1]`` and concatenates them.
    This mirrors ``megatron.core.utils.get_batch_on_this_cp_rank``.
    """
    S = tensor.shape[seq_dim]
    assert S % (2 * cp_size) == 0, f"seq_len {S} not divisible by 2*cp_size={2 * cp_size}"
    tensor = tensor.view(
        *tensor.shape[:seq_dim], 2 * cp_size, S // (2 * cp_size), *tensor.shape[seq_dim + 1 :]
    )
    index = torch.zeros(2, dtype=torch.int64, device=tensor.device)
    index[0] = cp_rank
    index[1] = 2 * cp_size - cp_rank - 1
    tensor = tensor.index_select(seq_dim, index)
    tensor = tensor.view(*tensor.shape[:seq_dim], -1, *tensor.shape[seq_dim + 2 :])
    return tensor


class _NoCPGroup:
    """Dummy size-1 process group used to bypass MRoPE's BSHD-style
    zigzag of pre-computed THD freqs (Megatron-Core gap:
    ``MultimodalRotaryEmbedding.forward`` lacks the ``not packed_seq``
    skip that plain ``RotaryEmbedding`` has).
    """

    def size(self):
        """Pretend this group has exactly one rank."""
        return 1

    def rank(self):
        """This rank's id within the fake group is always 0."""
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

    idx = tex.thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank)
    return idx.long()


def cp_partition_index(packed_seq_params, total_tokens, cp_size, cp_rank, device=None):
    """Global packed positions this CP rank owns, in its local order (THD).

    Follows ``packed_seq_params.cp_partition_mode`` -- the decoder input layout:
    ``zigzag`` is TE's per-sequence ``thd_get_partitioned_indices``;
    ``contiguous`` is rows ``[r * T / cp, (r + 1) * T / cp)`` of the whole packed
    buffer, as ``get_cp_slice_for_thd`` slices decoder-only batches. MDP cuts
    vision rows into CP shards with the same rule
    (``megatron.core.mdp.cp_partition``).
    """
    mode = getattr(packed_seq_params, "cp_partition_mode", "zigzag")
    if mode == "contiguous":
        if total_tokens % cp_size != 0:
            raise RuntimeError(
                f"contiguous CP needs packed rows {total_tokens} divisible by cp_size={cp_size}"
            )
        local = total_tokens // cp_size
        if device is None:
            device = packed_seq_params.cu_seqlens_q_padded.device
        return torch.arange(cp_rank * local, (cp_rank + 1) * local, device=device)
    if mode != "zigzag":
        raise ValueError(f"unsupported cp_partition_mode {mode!r}")
    return _thd_cp_partition_index(
        packed_seq_params.cu_seqlens_q_padded, total_tokens, cp_size, cp_rank
    )


class MultimodalModel(MegatronModule):
    """Base class for multimodal vision-language models.

    Composes a pre-constructed vision encoder and a ``HybridModel`` language
    decoder.  Supports pipeline parallelism: the vision encoder is built
    only on the first PP stage, while the language decoder spans all PP
    stages following standard Megatron PP layout flags.

    Args:
        language_config: ``TransformerConfig`` for the language decoder.
        hybrid_stack_spec: ``ModuleSpec`` defining HybridModel layer families.
        hybrid_layer_pattern: Ordered HybridModel decoder and MTP layer pattern.
        vision_encoder: Pre-constructed vision encoder module (or ``None``
            on non-first PP stages).
        vocab_size: Language model vocabulary size.
        max_sequence_length: Maximum sequence length.
        image_token_id: Token ID for image placeholder tokens.
        position_embedding_type: Position embedding type for the decoder.
        rotary_percent: Fraction of hidden dim for RoPE.
        rotary_base: Base frequency for RoPE.
        parallel_output: Keep outputs split across TP ranks.
        share_embeddings_and_output_weights: Tie input/output embeddings.
        pre_process: First PP stage flag — when True, build embedding +
            run vision encoder + scatter image embeddings.
        post_process: Last PP stage flag — when True, build output layer +
            compute loss inside ``HybridModel``.
        vp_stage: Virtual pipeline stage (forwarded to ``HybridModel``).
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        hybrid_stack_spec: ModuleSpec,
        hybrid_layer_pattern: str,
        vision_encoder: Optional[MegatronModule],
        vocab_size: int,
        max_sequence_length: int,
        image_token_id: int,
        position_embedding_type: str = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=language_config)

        self.image_token_id = image_token_id
        self.pre_process = pre_process
        self.vp_stage = vp_stage
        # ``post_process`` is deliberately not stored: nothing reads it off this
        # wrapper, and ``get_attr_wrapped_model(model, 'post_process', ...)``
        # (schedules.clear_embedding_activation_buffer) must keep resolving to
        # the HybridModel that owns ``embedding_activation_buffer``.
        # Surfaced for ``finalize_model_grads._allreduce_word_embedding_grads``
        # which inspects the outer module (not the wrapped HybridModel) when
        # PP > 1 and either tied embeddings or MTP layers are in use.
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        # Vision encoder lives only on the first PP stage.
        self.vision_model = vision_encoder if pre_process else None
        self.language_model = HybridModel(
            config=language_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=hybrid_layer_pattern,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(share_embeddings_and_output_weights),
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
        )

    def shared_embedding_or_output_weight(self):
        """Surface the wrapped language model's shared embedding / output
        weight to the PP grad-finalize step (mirrors the LLaVA wrapper).
        Needed when PP>1 and embeddings are tied or MTP is enabled.
        """
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        """Forward the previous PP stage's activation into ``HybridModel``,
        which routes it according to its own ``pre_process`` flag.
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        self.language_model.set_input_tensor(input_tensor[0])

    def _prepare_decoder_inputs(
        self,
        *,
        input_ids: Tensor,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        padding_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        packed_seq_params=None,
        vision_embeddings: Tensor = None,
    ):
        """Prepare the rank-local inputs shared by eager and scheduled decoder execution.

        The native path runs the vision encoder here, while MDP supplies its
        detached endpoint leaf through ``vision_embeddings`` after P2/P3.
        Vision work and embedding scatter intentionally stay outside the
        decoder's fine-grained schedule.
        """
        if position_ids is None:
            with nvtx_phase("compute_position_ids"):
                position_ids = self.compute_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    packed_seq_params=packed_seq_params,
                )

        # Context parallelism splits the sequence BEFORE the embedding: each CP
        # rank embeds only its own tokens and scatters only the vision rows whose
        # image-token slots it owns. MDP delivers exactly those rows (its leaf is
        # this rank's CP shard); the native path selects them from the full
        # encoder output, so both paths run the same rank-local scatter.
        # ``position_ids`` are NOT split: MRoPE returns full freqs and TE
        # attention's ``_apply_rotary_pos_emb_thd`` does the per-sample CP split
        # itself via ``_get_thd_freqs_on_this_cp_rank``.
        full_input_ids = input_ids
        cp_index = self._cp_index(packed_seq_params, input_ids.shape[1], input_ids.device)
        split = self._cp_split_sequence
        # A caller-supplied full-length decoder input is split like the rest.
        decoder_input = split(decoder_input, 0, cp_index)
        input_ids = split(input_ids, 1, cp_index)
        labels = split(labels, 1, cp_index)
        loss_mask = split(loss_mask, 1, cp_index)
        padding_mask = split(padding_mask, 1, cp_index)
        if packed_seq_params is None:  # THD carries no attention mask
            attention_mask = split(attention_mask, 1, cp_index)

        if self.pre_process:
            # An MDP endpoint supplies the pre-encoded leaf. The native path
            # encodes pixels here; both feed the same scatter below.
            if vision_embeddings is None:
                if self.vision_model is not None and pixel_values is not None:
                    with nvtx_phase("native_vision_encoder_forward"):
                        vision_embeddings = self.vision_model(pixel_values, image_grid_thw)
                    vision_embeddings = self._cp_local_vision_rows(
                        full_input_ids, vision_embeddings, cp_index
                    )

            if decoder_input is None:
                with nvtx_phase("text_embedding"):
                    text_embeddings = self.language_model.embedding(
                        input_ids=input_ids, position_ids=None
                    )

                if vision_embeddings is not None:
                    with nvtx_phase("scatter_vision_embeddings"):
                        decoder_input = self._scatter_vision_embeddings(
                            input_ids, text_embeddings, vision_embeddings
                        )
                else:
                    if bool((input_ids == self.image_token_id).any()):
                        raise RuntimeError(
                            "input_ids contain image-token slots but no vision "
                            "source was provided (neither pixel_values nor "
                            "vision_embeddings); the text path would silently "
                            "train on placeholder embeddings"
                        )
                    decoder_input = text_embeddings
        else:
            # Intermediate PP/VPP chunks receive their activation through
            # GPTModel.set_input_tensor during scheduled or eager execution.
            decoder_input = None

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
        )

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        padding_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        packed_seq_params=None,
        vision_embeddings: Tensor = None,
        **kwargs,
    ):
        """Build the native fine-grained schedule for the decoder only."""
        decoder_inputs = self._prepare_decoder_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            decoder_input=decoder_input,
            packed_seq_params=packed_seq_params,
            vision_embeddings=vision_embeddings,
        )
        return self.language_model.build_schedule_plan(**decoder_inputs)

    def _scatter_vision_embeddings(
        self, input_ids: Tensor, text_embeddings: Tensor, vision_embeddings: Tensor
    ) -> Tensor:
        """Replace image-token positions with vision embeddings.

        Handles sequence parallelism (gather → scatter → re-scatter).

        Args:
            input_ids: ``[B, S]`` token IDs.
            text_embeddings: ``[S, B, D]`` (or ``[S/TP, B, D]`` with SP).
            vision_embeddings: ``[num_visual_tokens, D]``.

        Returns:
            Combined embeddings, same shape as *text_embeddings*.
        """
        sp = (
            self.config.sequence_parallel
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        )

        if sp:
            text_embeddings = tensor_parallel.gather_from_sequence_parallel_region(
                text_embeddings, tensor_parallel_output_grad=False
            )

        combined = text_embeddings.transpose(0, 1).contiguous()
        image_mask = input_ids == self.image_token_id
        num_slots = int(image_mask.sum())
        if num_slots != vision_embeddings.shape[0]:
            raise RuntimeError(
                f"image-token slot count {num_slots} != vision embedding rows "
                f"{vision_embeddings.shape[0]}; masked_scatter would silently "
                "mis-fill the sequence"
            )
        mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
        combined = combined.masked_scatter(
            mask_expanded, vision_embeddings.to(combined.dtype)
        )
        combined = combined.transpose(0, 1).contiguous()

        if sp:
            combined = tensor_parallel.scatter_to_sequence_parallel_region(combined)

        return combined

    def compute_position_ids(
        self, input_ids: Tensor, image_grid_thw: Optional[Tensor] = None, packed_seq_params=None
    ) -> Tensor:
        """Compute position IDs.  Override for MRoPE etc.

        Default: simple sequential positions.  ``packed_seq_params`` is
        accepted for subclass compatibility (e.g. MRoPE in THD mode).
        """
        B, S = input_ids.shape
        return torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

    def _cp_index(self, packed_seq_params, total_tokens, device):
        """This CP rank's THD token index (:func:`cp_partition_index`), or ``None``.

        ``None`` without context parallelism and for BSHD, whose tensors are
        zigzag-split directly. One index per forward serves every token-indexed
        input, so they all land on this rank by the decoder's own input layout.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1 or packed_seq_params is None:
            return None
        # The collator's THD layout must be the one the decoder was built for.
        expected = getattr(self.config, "cp_partition_mode", "zigzag")
        actual = getattr(packed_seq_params, "cp_partition_mode", "zigzag")
        if actual != expected:
            raise RuntimeError(
                f"packed_seq_params.cp_partition_mode={actual!r} but the decoder is built "
                f"for cp_partition_mode={expected!r}; tokens would land on the wrong CP rank"
            )
        cp_rank = parallel_state.get_context_parallel_rank()
        return cp_partition_index(packed_seq_params, total_tokens, cp_size, cp_rank, device=device)

    @staticmethod
    def _cp_split_sequence(tensor, seq_dim, cp_index):
        """This CP rank's slice of ``tensor`` along ``seq_dim`` (None passes through).

        THD selects ``cp_index`` (:meth:`_cp_index`); BSHD (``cp_index=None``) is
        zigzag only, like ``get_batch_on_this_cp_rank``.
        """
        if tensor is None:
            return None
        if cp_index is not None:
            return tensor.index_select(seq_dim, cp_index)
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1:
            return tensor
        cp_rank = parallel_state.get_context_parallel_rank()
        return _cp_split_tensor(tensor, seq_dim=seq_dim, cp_size=cp_size, cp_rank=cp_rank)

    def _cp_local_vision_rows(self, full_input_ids, vision_embeddings, cp_index):
        """Rows of the full encoder output whose image-token slots this CP rank owns.

        ``vision_embeddings`` is ordered like ``masked_scatter`` fills the full
        sequence (row-major over ``full_input_ids``). Splitting a position index
        with the same rule as the tokens gives the owned slots in local order,
        which is exactly the order the rank-local scatter consumes them in.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1 or vision_embeddings is None:
            return vision_embeddings
        image_mask = (full_input_ids == self.image_token_id).reshape(-1)
        row_of_slot = torch.cumsum(image_mask.to(torch.long), dim=0) - 1
        positions = torch.arange(
            image_mask.numel(), device=full_input_ids.device
        ).view_as(full_input_ids)
        local = self._cp_split_sequence(positions, 1, cp_index).reshape(-1)
        local = local[image_mask[local]]
        return vision_embeddings.index_select(0, row_of_slot[local])

    @staticmethod
    def cp_split_loss_mask(loss_mask, packed_seq_params):
        """Slice ``loss_mask`` the same way the model slices its inputs.

        Mirrors the slicing done inside :meth:`_prepare_decoder_inputs` so
        the loss computation outside the model can index a mask aligned
        with the model's CP-shard output. Returns ``loss_mask`` unchanged
        when ``CP <= 1``.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1 or loss_mask is None:
            return loss_mask
        cp_rank = parallel_state.get_context_parallel_rank()
        if packed_seq_params is not None:
            idx = cp_partition_index(
                packed_seq_params, loss_mask.shape[1], cp_size, cp_rank, device=loss_mask.device
            )
            return loss_mask.index_select(1, idx)
        return _cp_split_tensor(loss_mask, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)

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
        padding_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        packed_seq_params=None,
        vision_embeddings: Tensor = None,
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
            padding_mask: ``[B, S]`` bool mask, True at collate-padded
                positions (``[1, T]`` in THD). Forwarded to the language
                decoder so MoE routing excludes padded tokens from aux
                loss / z-loss / expert-bias accumulation. Distinct from
                ``loss_mask``: only true padding, not SFT prompt tokens.
            pixel_values: Preprocessed image pixels.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.
            decoder_input: Pre-computed decoder input (skip embed). Only
                honored on the first pipeline stage; later stages take
                their activation from ``set_input_tensor``.
            packed_seq_params: ``PackedSeqParams`` for THD attention.

        Returns:
            Loss tensor (post_process=True) or hidden states.
        """
        decoder_inputs = self._prepare_decoder_inputs(
            decoder_input=decoder_input,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_embeddings=vision_embeddings,
        )

        with self._thd_mrope_no_cp_override(packed_seq_params), nvtx_phase(
            "decoder_forward"
        ):
            return self.language_model(**decoder_inputs)
