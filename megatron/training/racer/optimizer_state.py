"""Distributed optimizer state helpers for RACER memory checkpoints."""

from __future__ import annotations

from typing import Any

DISTRIB_OPTIM_STATE_KEY = "__racer_distributed_optimizer_parameter_state__"
OPTIM_FORMAT_KEY = "__racer_distributed_optimizer_format__"
OPTIM_STATE_KEY = "state"
OPTIM_FORMAT_DP_RESHARDABLE = "dp_reshardable"
OPTIM_FORMAT_DP_ZERO = "dp_zero_gather_scatter"


def capture_distributed_optimizer_state(racer_enabled: bool, optimizer: Any) -> Any | None:
    if not racer_enabled or optimizer is None or optimizer.is_stub_optimizer:
        return None
    return get_distributed_optimizer_parameter_state(optimizer)


def pop_distributed_optimizer_state(state_dict: dict[str, Any]) -> Any | None:
    return state_dict.pop(DISTRIB_OPTIM_STATE_KEY, None)


def load_distributed_optimizer_state(
    optimizer: Any,
    parameter_state: Any,
    *,
    update_legacy_format: bool = False,
) -> None:
    if parameter_state is None or optimizer is None or optimizer.is_stub_optimizer:
        return
    if load_one_distributed_optimizer_state(
        optimizer, parameter_state, update_legacy_format=update_legacy_format
    ):
        return
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        states = parameter_state if isinstance(parameter_state, list) else [parameter_state]
        for idx, opt in enumerate(chained):
            state = states[idx] if idx < len(states) else None
            load_one_distributed_optimizer_state(
                opt, state, update_legacy_format=update_legacy_format
            )
        return
    raise TypeError("optimizer does not support in-memory distributed parameter state loading")


def get_distributed_optimizer_parameter_state(optimizer: Any) -> Any | None:
    state = get_one_distributed_optimizer_state(optimizer)
    if state is not None:
        return state
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        states = []
        found = False
        for opt in chained:
            state = get_one_distributed_optimizer_state(opt)
            states.append(state)
            found = found or state is not None
        return states if found else None
    return None


def get_one_distributed_optimizer_state(optimizer: Any) -> Any | None:
    if hasattr(optimizer, "get_parameter_state_dp_reshardable"):
        state = optimizer.get_parameter_state_dp_reshardable()
        mark_dp_reshardable_padding_flags(state)
        return {OPTIM_FORMAT_KEY: OPTIM_FORMAT_DP_RESHARDABLE, OPTIM_STATE_KEY: state}
    if hasattr(optimizer, "get_parameter_state_dp_zero"):
        return {
            OPTIM_FORMAT_KEY: OPTIM_FORMAT_DP_ZERO,
            OPTIM_STATE_KEY: optimizer.get_parameter_state_dp_zero(),
        }
    return None


def load_one_distributed_optimizer_state(
    optimizer: Any,
    parameter_state: Any,
    *,
    update_legacy_format: bool,
) -> bool:
    if optimizer is None or parameter_state is None:
        return False
    if isinstance(parameter_state, dict) and OPTIM_FORMAT_KEY in parameter_state:
        state_format = parameter_state.get(OPTIM_FORMAT_KEY)
        state = parameter_state.get(OPTIM_STATE_KEY)
        if state_format == OPTIM_FORMAT_DP_RESHARDABLE:
            if not hasattr(optimizer, "load_parameter_state_from_dp_reshardable"):
                raise TypeError("optimizer does not support RACER dp_reshardable state loading")
            optimizer.load_parameter_state_from_dp_reshardable(state)
            return True
        if state_format == OPTIM_FORMAT_DP_ZERO:
            if not hasattr(optimizer, "load_parameter_state_from_dp_zero"):
                raise TypeError("optimizer does not support RACER dp_zero state loading")
            optimizer.load_parameter_state_from_dp_zero(
                state, update_legacy_format=update_legacy_format
            )
            return True
        raise ValueError(f"unknown RACER distributed optimizer state format: {state_format!r}")
    if hasattr(optimizer, "load_parameter_state_from_dp_zero"):
        optimizer.load_parameter_state_from_dp_zero(
            parameter_state, update_legacy_format=update_legacy_format
        )
        return True
    return False


def mark_dp_reshardable_padding_flags(state: Any) -> None:
    if not isinstance(state, dict):
        return
    for gbuf_idx, dtype_state in state.items():
        if not isinstance(gbuf_idx, int) or not isinstance(dtype_state, dict):
            continue
        for buckets_state in dtype_state.values():
            if not isinstance(buckets_state, list):
                continue
            for bucket_state in buckets_state:
                if not isinstance(bucket_state, list):
                    continue
                for tensors in bucket_state:
                    if isinstance(tensors, dict) and "padding" not in tensors:
                        tensors["padding"] = False
