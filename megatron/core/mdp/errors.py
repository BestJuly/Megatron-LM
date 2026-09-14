# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP error types.

Each class maps to a distinct handling strategy (API design section 19):

- ``MdpConfigurationError``: startup rejection, including invalid rank mappings.
- ``MdpPlanError``: coordinated plan/digest mismatch raised before any P2P post.
- ``MdpBridgeError``: bridge ledger or exchange contract violation.
- ``MdpStateError``: invalid runtime state transition.
- ``MdpCheckpointError``: unsupported or inconsistent checkpoint operation.
- ``MdpTaskFatalError``: error after a P2P/collective post; recovery is from
  checkpoint only, in-process recovery is not promised.
"""


class MdpError(RuntimeError):
    """Base class for all MDP errors."""


class MdpConfigurationError(MdpError):
    """Invalid configuration, unsupported combination, or invalid rank mapping."""


class MdpPlanError(MdpError):
    """Plan construction, validation, or cross-rank consistency failure."""


class MdpBridgeError(MdpError):
    """Bridge ledger or exchange contract violation."""


class MdpStateError(MdpError):
    """Invalid MDP runtime state transition."""


class MdpCheckpointError(MdpError):
    """Unsupported or inconsistent checkpoint operation."""


class MdpTaskFatalError(MdpError):
    """Error raised after a P2P or collective post; the task must restart from checkpoint."""
