# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pytest configuration for multimodal unit tests.

Tests must be run from the Megatron-LM repo root so that both
`multimodal` and `tests.unit_tests.test_utilities` are importable.

Single-process (TP=1 tests only):
    python -m pytest multimodal/tests/ -v

Two-process (enables TP=2 tests):
    torchrun --nproc_per_node 2 -m pytest multimodal/tests/ -v
"""
