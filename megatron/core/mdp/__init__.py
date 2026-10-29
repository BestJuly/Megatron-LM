# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP: co-located, phase-serialized multimodal training.

MDP rebalances vision-encoder work across each decoder replica's ``CP x PP``
worker pool while leaving the native decoder schedule, data ownership, and
training accounting untouched. See ``megatron/core/mdp/README.md``.
"""

from megatron.core.mdp.config import (
    MdpCompatibilityOptions,
    MdpConfig,
    apply_encoder_recompute_config,
    validate_mdp_config,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpCheckpointError,
    MdpConfigurationError,
    MdpError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
