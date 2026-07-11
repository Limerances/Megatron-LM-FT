"""Distributed optimizer state helpers for RACER memory checkpoints."""

from __future__ import annotations

from typing import Any

DISTRIB_OPTIM_STATE_KEY = "__racer_distributed_optimizer_parameter_state__"
OPTIM_FORMAT_KEY = "__racer_distributed_optimizer_format__"
OPTIM_FORMAT_VERSION_KEY = "__racer_distributed_optimizer_format_version__"
OPTIM_METADATA_KEY = "__racer_distributed_optimizer_metadata__"
OPTIM_STATE_KEY = "state"
OPTIM_FORMAT_DP_RESHARDABLE = "dp_reshardable"
OPTIM_FORMAT_DP_RESHARDABLE_NATIVE = "dp_reshardable_precision_native"
OPTIM_FORMAT_DP_RESHARDABLE_NATIVE_VERSION = 1
OPTIM_FORMAT_DP_ZERO = "dp_zero_gather_scatter"


def capture_distributed_optimizer_state(racer_enabled: bool, optimizer: Any) -> Any | None:
    if not racer_enabled or optimizer is None or optimizer.is_stub_optimizer:
        return None
    return get_distributed_optimizer_parameter_state(optimizer)


def pop_distributed_optimizer_state(state_dict: dict[str, Any]) -> Any | None:
    return state_dict.pop(DISTRIB_OPTIM_STATE_KEY, None)


def clear_native_optimizer_state_preflight(optimizer: Any) -> None:
    """Clear one-shot native-load intent recursively."""
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for child in chained:
            clear_native_optimizer_state_preflight(child)
        return
    clear_fn = getattr(optimizer, "clear_parameter_state_dp_reshardable_native_load", None)
    if callable(clear_fn):
        clear_fn()


def prepare_native_optimizer_state_load(optimizer: Any, parameter_state: Any) -> dict[str, Any]:
    """Validate a complete native envelope and arm one-shot zero-copy loading."""
    import time

    started = time.perf_counter()
    prepared = []

    def prepare_one(current_optimizer: Any, current_state: Any, path: str) -> None:
        chained = getattr(current_optimizer, "chained_optimizers", None)
        if chained is not None:
            if not isinstance(current_state, list):
                raise TypeError(f"{path}: chained optimizer requires a list state")
            if len(current_state) != len(chained):
                raise ValueError(
                    f"{path}: native state count does not match chained optimizers: "
                    f"state_count={len(current_state)}, optimizer_count={len(chained)}"
                )
            for idx, (child, child_state) in enumerate(zip(chained, current_state)):
                prepare_one(child, child_state, f"{path}[{idx}]")
            return

        if not isinstance(current_state, dict):
            raise TypeError(f"{path}: native optimizer state envelope must be a dict")
        expected_keys = {
            OPTIM_FORMAT_KEY,
            OPTIM_FORMAT_VERSION_KEY,
            OPTIM_METADATA_KEY,
            OPTIM_STATE_KEY,
        }
        if set(current_state) != expected_keys:
            raise ValueError(
                f"{path}: native optimizer envelope keys mismatch: "
                f"expected={sorted(expected_keys)}, actual={sorted(current_state)}"
            )
        if current_state[OPTIM_FORMAT_KEY] != OPTIM_FORMAT_DP_RESHARDABLE_NATIVE:
            raise ValueError(
                f"{path}: expected native optimizer format, got "
                f"{current_state[OPTIM_FORMAT_KEY]!r}"
            )
        if current_state[OPTIM_FORMAT_VERSION_KEY] != OPTIM_FORMAT_DP_RESHARDABLE_NATIVE_VERSION:
            raise ValueError(
                f"{path}: unsupported native optimizer version "
                f"{current_state[OPTIM_FORMAT_VERSION_KEY]!r}"
            )
        metadata = current_state[OPTIM_METADATA_KEY]
        if not isinstance(metadata, dict):
            raise TypeError(f"{path}: native optimizer metadata must be a dict")
        if not isinstance(current_state[OPTIM_STATE_KEY], dict):
            raise TypeError(f"{path}: native optimizer payload must be a dict")
        prepare_fn = getattr(
            current_optimizer, "prepare_parameter_state_dp_reshardable_native_load", None
        )
        if not callable(prepare_fn):
            raise TypeError(f"{path}: optimizer does not support native zero-copy loading")
        prepare_fn(metadata)
        prepared.append(current_optimizer)

    try:
        prepare_one(optimizer, parameter_state, "optimizer")
    except Exception:
        for prepared_optimizer in prepared:
            prepared_optimizer.clear_parameter_state_dp_reshardable_native_load()
        raise

    return {
        "metadata_ms": (time.perf_counter() - started) * 1000.0,
        "optimizers": len(prepared),
    }


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
        if len(states) != len(chained):
            raise ValueError(
                "RACER distributed optimizer state count does not match chained optimizers: "
                f"state_count={len(states)}, optimizer_count={len(chained)}"
            )
        for idx, opt in enumerate(chained):
            load_one_distributed_optimizer_state(opt, states[idx], update_legacy_format=update_legacy_format)
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
    native_metadata_fn = getattr(
        optimizer, "get_parameter_state_dp_reshardable_native_metadata", None
    )
    native_state_fn = getattr(optimizer, "get_parameter_state_dp_reshardable_native", None)
    if callable(native_metadata_fn) and callable(native_state_fn):
        metadata = native_metadata_fn()
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("native dp_reshardable optimizer metadata must be a dict")
            state = native_state_fn()
            mark_dp_reshardable_padding_flags(state)
            return {
                OPTIM_FORMAT_KEY: OPTIM_FORMAT_DP_RESHARDABLE_NATIVE,
                OPTIM_FORMAT_VERSION_KEY: OPTIM_FORMAT_DP_RESHARDABLE_NATIVE_VERSION,
                OPTIM_METADATA_KEY: dict(metadata),
                OPTIM_STATE_KEY: state,
            }
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
        if state_format == OPTIM_FORMAT_DP_RESHARDABLE_NATIVE:
            version = parameter_state.get(OPTIM_FORMAT_VERSION_KEY)
            if version != OPTIM_FORMAT_DP_RESHARDABLE_NATIVE_VERSION:
                raise ValueError(
                    "unsupported RACER native dp_reshardable optimizer format version: "
                    f"checkpoint={version!r}, supported={OPTIM_FORMAT_DP_RESHARDABLE_NATIVE_VERSION}"
                )
            checkpoint_metadata = parameter_state.get(OPTIM_METADATA_KEY)
            if not isinstance(checkpoint_metadata, dict):
                raise ValueError("RACER native dp_reshardable checkpoint metadata is missing")
            metadata_fn = getattr(
                optimizer, "get_parameter_state_dp_reshardable_native_metadata", None
            )
            load_fn = getattr(
                optimizer, "load_parameter_state_from_dp_reshardable_native", None
            )
            if not callable(metadata_fn) or not callable(load_fn):
                raise TypeError(
                    "optimizer does not support RACER native dp_reshardable state loading"
                )
            current_metadata = metadata_fn()
            if current_metadata is None:
                raise RuntimeError(
                    "current optimizer configuration is not eligible for the RACER native "
                    "dp_reshardable format"
                )
            if checkpoint_metadata != current_metadata:
                raise ValueError(
                    "RACER native dp_reshardable optimizer schema/config mismatch: "
                    f"checkpoint={checkpoint_metadata!r}, current={current_metadata!r}"
                )
            load_fn(state)
            return True
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
