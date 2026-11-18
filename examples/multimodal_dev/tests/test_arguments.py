# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression coverage for the real multimodal training argument parser."""

import argparse

import pytest

from examples.multimodal_dev.arguments import add_multimodal_args
from megatron.training.arguments import add_megatron_arguments


@pytest.mark.parametrize('layout', ['zigzag', 'contiguous'])
def test_packed_sequence_option_composes_with_megatron_arguments(layout):
    """The generated THD option must coexist with multimodal extra arguments."""
    parser = add_multimodal_args(add_megatron_arguments(argparse.ArgumentParser()))
    assert parser.parse_args([]).use_packed_sequence is False
    args = parser.parse_args(['--use-packed-sequence', '--cp-partition-mode', layout])
    assert args.use_packed_sequence is True
    assert args.cp_partition_mode == layout
