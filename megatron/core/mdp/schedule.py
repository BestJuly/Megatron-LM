# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP schedule and finalizer wiring (API design section 13).

Two wrappers, neither of which modifies MCore:

* :func:`wrap_forward_backward` runs P1-P3 before the native schedule and
  P5/cleanup after it, substituting the replay iterators for the real data
  iterator. Training and evaluation call sites are wrapped separately; a
  callable must not be wrapped twice.
* :func:`wrap_finalize_model_grads` captures the global token count by
  wrapping ``config.finalize_model_grads_func`` — the one injectable exit all
  three schedule variants share. The native implementation broadcasts and
  all-reduces ``num_tokens`` **in place**, so holding the reference yields the
  global count after the inner call returns; a clone taken beforehand would
  hold this lane's partial count.
"""

from inspect import signature
from typing import Callable

from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.runtime import MdpRuntime

_WRAPPED_MARKER = "_mdp_wrapped"


def wrap_forward_backward(forward_backward_func: Callable, runtime: MdpRuntime) -> Callable:
    """Wrap one schedule callable with the MDP phase machine."""
    if getattr(forward_backward_func, _WRAPPED_MARKER, False):
        raise MdpConfigurationError(
            "MDP: forward_backward_func is already wrapped; a callable must not be "
            "wrapped twice."
        )
    schedule_signature = signature(forward_backward_func)

    def wrapped(*args, **kwargs):
        bound = schedule_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data_iterator = bound.arguments["data_iterator"]
        replay_iterators = runtime.begin_iteration(
            data_iterator,
            num_microbatches=bound.arguments["num_microbatches"],
            forward_only=bound.arguments["forward_only"],
        )
        if isinstance(data_iterator, (list, tuple)):
            bound.arguments["data_iterator"] = list(replay_iterators)
        else:
            bound.arguments["data_iterator"] = replay_iterators[0]

        result = forward_backward_func(*bound.args, **bound.kwargs)

        runtime.mark_decoder_complete()
        runtime.end_iteration()
        return result

    setattr(wrapped, _WRAPPED_MARKER, True)
    wrapped._mdp_inner = forward_backward_func
    return wrapped


def wrap_finalize_model_grads(config, runtime: MdpRuntime) -> None:
    """Install the token-count capture on ``config.finalize_model_grads_func``.

    Idempotent per config object; requires the native finalizer to already be
    installed. The wrapper preserves positional/keyword flexibility and the
    extra ``pg_collection``/``force_all_reduce`` arguments the dev schedules
    pass through.
    """
    if getattr(config, "_mdp_finalize_wrapped", False):
        return
    native = config.finalize_model_grads_func
    if native is None:
        raise MdpConfigurationError(
            "MDP: config.finalize_model_grads_func is None; the native finalizer must "
            "be installed before MDP wraps it to source the global token count."
        )

    def wrapped(model, num_tokens=None, *args, **kwargs):
        result = native(model, num_tokens, *args, **kwargs)
        # Post-call: the native finalizer reduced num_tokens in place, so the
        # same tensor object now holds the global count on every rank.
        runtime.capture_global_num_tokens(num_tokens)
        return result

    wrapped._mdp_native = native
    config.finalize_model_grads_func = wrapped
    config._mdp_finalize_wrapped = True


def unwrap_finalize_model_grads(config) -> None:
    """Remove the capture wrapper (teardown/tests)."""
    if getattr(config, "_mdp_finalize_wrapped", False):
        config.finalize_model_grads_func = config.finalize_model_grads_func._mdp_native
        config._mdp_finalize_wrapped = False
