# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP: co-located, phase-serialized multimodal training.

MDP rebalances vision-encoder work across each decoder replica's ``CP x PP``
worker pool while leaving the native decoder schedule, data ownership, and
training accounting untouched. See ``megatron/core/mdp/README.md``.
"""

from megatron.core.mdp.config import (
    VISION_CONFIG_OVERRIDE_ALLOWLIST,
    MdpCompatibilityOptions,
    MdpConfig,
    apply_vision_config_overrides,
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
