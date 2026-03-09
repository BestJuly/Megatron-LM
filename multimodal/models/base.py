# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Base multimodal model with pipeline parallelism support.

Provides the shared logic for composing a vision encoder + language decoder,
with PP-aware conditional module building.
"""

from typing import Dict, Optional

import torch
from torch import Tensor

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from multimodal.models.vision import VisionEncoder


class MultimodalModel(MegatronModule):
    """Base class for multimodal (vision-language) models.

    Composes a vision encoder and a GPTModel-based language decoder with
    pipeline parallelism support.

    PP layout:
        - Stage 0 (add_encoder=True): vision encoder + first decoder layers
        - Stage 1+ (add_decoder=True): remaining decoder layers
        - Last stage (post_process=True): output layer + loss computation

    Subclasses should override:
        - `compute_position_ids()` for model-specific position encoding
        - `preprocess_images()` if custom image preprocessing is needed

    Args:
        language_config: TransformerConfig for the language decoder.
        language_spec: ModuleSpec for decoder transformer layers.
        vision_config: TransformerConfig for the vision encoder.
        vision_spec: ModuleSpec for vision transformer layers.
        vocab_size: Language model vocabulary size.
        max_sequence_length: Maximum sequence length.
        image_token_id: Token ID used as placeholder for image embeddings.
        position_embedding_type: Position embedding type for the decoder.
        rotary_percent: Fraction of hidden dim for RoPE.
        rotary_base: Base frequency for RoPE.
        mrope_section: MRoPE channel sections (for position_embedding_type='mrope').
        vision_kwargs: Extra kwargs passed to VisionEncoder.
        pre_process: First PP stage (includes embeddings).
        post_process: Last PP stage (includes output layer).
        add_encoder: Build vision encoder on this stage.
        add_decoder: Build language decoder on this stage.
        parallel_output: Keep outputs split across TP ranks.
        share_embeddings_and_output_weights: Tie input/output embeddings.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_spec: ModuleSpec,
        vision_config: TransformerConfig,
        vision_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        image_token_id: int = -200,
        position_embedding_type: str = "mrope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        mrope_section: list = None,
        vision_kwargs: dict = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        super().__init__(config=language_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.image_token_id = image_token_id

        # Vision encoder — only on first PP stage
        self.vision_encoder = None
        if self.add_encoder:
            vkw = vision_kwargs or {}
            self.vision_encoder = VisionEncoder(
                config=vision_config,
                transformer_layer_spec=vision_spec,
                **vkw,
            )

        # Language decoder — on all stages that have decoder
        self.language_model = None
        if self.add_decoder:
            self.language_model = GPTModel(
                config=language_config,
                transformer_layer_spec=language_spec,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                pre_process=self.pre_process,
                post_process=self.post_process,
                parallel_output=parallel_output,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                position_embedding_type=position_embedding_type,
                rotary_percent=rotary_percent,
                rotary_base=rotary_base,
            )

        # For receiving encoder hidden state across PP stages
        self.encoder_hidden_state = None

    def set_input_tensor(self, input_tensor):
        """Route input tensors for pipeline parallelism.

        Stage 0 (encoder+decoder): vision encoder receives input
        Stage 0 (encoder only): vision encoder receives input
        Stage 1 (first decoder stage after encoder): receives encoder output
        Stage 2+: language decoder receives hidden states
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            # Stage 0: both encoder and decoder on same stage
            # No cross-stage input needed for vision encoder
            assert len(input_tensor) == 1
            self.language_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            # Encoder-only stage (not typical for this model)
            pass
        elif self.pre_process:
            # First decoder-only stage: receives vision output
            self.encoder_hidden_state = input_tensor[0]
        else:
            # Intermediate/last decoder stages
            assert len(input_tensor) == 1
            self.language_model.set_input_tensor(input_tensor[0])

    def _scatter_vision_embeddings(
        self,
        input_ids: Tensor,
        text_embeddings: Tensor,
        vision_embeddings: Tensor,
    ) -> Tensor:
        """Replace image token positions in text embeddings with vision embeddings.

        Args:
            input_ids: [B, S] token IDs.
            text_embeddings: [S, B, D] text embeddings from the decoder embedding layer.
            vision_embeddings: [num_visual_tokens, D] visual embeddings.

        Returns:
            Combined embeddings [S, B, D].
        """
        # Create mask for image token positions: [B, S]
        image_mask = (input_ids == self.image_token_id)

        # text_embeddings is [S, B, D], transpose to [B, S, D] for scatter
        combined = text_embeddings.transpose(0, 1).contiguous()
        # image_mask: [B, S] -> [B, S, 1] for broadcasting
        mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
        combined = combined.masked_scatter(mask_expanded, vision_embeddings)
        return combined.transpose(0, 1).contiguous()

    def compute_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute position IDs. Override in subclasses for MRoPE etc.

        Default: simple sequential position IDs.

        Args:
            input_ids: [B, S] token IDs.
            image_grid_thw: [num_images, 3] grid dimensions.

        Returns:
            Position IDs tensor.
        """
        B, S = input_ids.shape
        return torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

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
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_ids: [B, S] token IDs.
            position_ids: [3, B, S] for MRoPE or [B, S] for standard.
            attention_mask: [B, S] attention mask.
            labels: [B, S] target token IDs for loss.
            loss_mask: [B, S] mask for loss computation.
            pixel_values: Preprocessed image pixels for vision encoder.
            image_grid_thw: [num_images, 3] grid dimensions.
            decoder_input: Pre-computed decoder input (for intermediate PP stages).

        Returns:
            If post_process: loss tensor.
            If not post_process: hidden states.
        """
        # --- Vision encoding (stage 0 only) ---
        vision_embeddings = None
        if self.add_encoder and self.vision_encoder is not None and pixel_values is not None:
            vision_embeddings = self.vision_encoder(pixel_values, image_grid_thw)

        # --- Prepare decoder input ---
        if decoder_input is None and self.pre_process and self.language_model is not None:
            # Get text embeddings from language model's embedding layer
            text_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None
            )  # [S, B, D]

            # Scatter vision embeddings into text embedding positions
            if vision_embeddings is not None:
                decoder_input = self._scatter_vision_embeddings(
                    input_ids, text_embeddings, vision_embeddings
                )
            else:
                decoder_input = text_embeddings

        # --- Language model forward ---
        if self.language_model is not None:
            output = self.language_model(
                input_ids=None,  # Already embedded
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                loss_mask=loss_mask,
            )
            return output

        # Encoder-only stage: return vision embeddings
        return vision_embeddings
