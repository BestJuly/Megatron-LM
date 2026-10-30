# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kimi-K3 vision-language model package (``--model-arch kimi_k3_vl``).

MoonViT 3D vision tower + ``patchmergerv2`` projector in MCore modules, and the
Kimi-K3 hybrid decoder (KDA / MLA / latent MoE / attention residuals / MTP)
built by ``HybridModelBuilder`` from the recipe.

Modules:
    configuration  token ids, vision config, dtype-dependent norm epsilon
    vision_encoder MoonViT tower, tpool merge, projector
    specs          ViT layer spec
    model          ``KimiK3VLModel`` (tower + decoder)
    factory        registry hooks (model, FLOPs metadata)
    mdp_adapter    collate spec + MDP adapter
    preprocessing  native port of the HF image processor
    data           mock / mdp_mock / cord_v2 providers
    hf_mapping     HF <-> MCore vision parameter mapping
"""
