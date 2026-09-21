# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the real CLI-to-MDP packing contract without distributed state."""

import argparse

import pytest

from examples.multimodal_dev.arguments import add_multimodal_args
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.integration import (
    compatibility_options_from_args,
    mdp_config_from_args,
    validate_from_args,
)
from megatron.training.arguments import add_megatron_arguments


@pytest.fixture(scope="module")
def packing_parser():
    """Use both native core flags and the multimodal entry point's flags."""
    return add_multimodal_args(add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False)))


def _parse(parser, *flags):
    args = parser.parse_args(
        [
            "--mdp-enable",
            "--bf16",
            "--use-distributed-optimizer",
            "--calculate-per-token-loss",
            "--max-seqlen-per-dp-cp-rank",
            "1024",
            "--thd-max-packed-sequences",
            "8",
            *flags,
        ]
    )
    # Normally supplied by parse_args() from the launcher's WORLD_SIZE.
    args.world_size = 4
    return args


@pytest.mark.parametrize("policy", [None, "greedy", "ffd"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_grouping_static_and_overlap_are_independent(packing_parser, policy, static, overlap):
    flags = [] if policy is None else [f"--mdp-{policy}-packing"]
    if static:
        flags += ["--thd-static-packing", "--pad-packed-seq-alignment", "max"]
    if overlap:
        flags += ["--mdp-overlap-window-capture"]
    args = _parse(packing_parser, *flags)
    validate_from_args(args)
    config = mdp_config_from_args(args)
    options = compatibility_options_from_args(args)
    assert config.greedy_packing == (policy == "greedy")
    assert config.ffd_packing == (policy == "ffd")
    assert config.packing_enabled == (policy is not None)
    assert config.overlap_window_capture == overlap
    assert options.thd_static_packing == static
    assert config.ffd_packing_buffer_size == 128


def test_cli_rejects_two_grouping_policies(packing_parser):
    with pytest.raises(SystemExit) as error:
        _parse(packing_parser, "--mdp-greedy-packing", "--mdp-ffd-packing")
    assert error.value.code == 2


def test_ffd_buffer_reaches_the_config(packing_parser):
    args = _parse(packing_parser, "--mdp-ffd-packing", "--mdp-ffd-packing-buffer-size", "17")
    validate_from_args(args)
    assert mdp_config_from_args(args).ffd_packing_buffer_size == 17


@pytest.mark.parametrize("policy", ["greedy", "ffd"])
@pytest.mark.parametrize(
    "resume_flag", ["--mdp-greedy-packing-approximate-resume", "--mdp-packing-approximate-resume"]
)
def test_resume_aliases_reach_the_shared_guard(packing_parser, policy, resume_flag):
    flags = [f"--mdp-{policy}-packing", "--save", "unused-checkpoint-path"]
    with pytest.raises(MdpConfigurationError, match="not checkpointed"):
        validate_from_args(_parse(packing_parser, *flags))
    args = _parse(packing_parser, *flags, resume_flag)
    validate_from_args(args)
    assert mdp_config_from_args(args).greedy_packing_approximate_resume
