# Copyright (c) 2026, RACER integration contributors.

"""RACER checkpoint adapter for Megatron state_dicts.

The adapter stores tensor leaves as CUDA byte views and routes durable checkpoint
chunks only through a restart-aware RACER Checkpoint Storage Daemon.
"""

from __future__ import annotations

import atexit
import base64
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import pickle
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

from megatron.training.racer import optimizer_state, rank_map, session_state, tensor_tree


DISTRIB_OPTIM_STATE_KEY = optimizer_state.DISTRIB_OPTIM_STATE_KEY
_META_KEY = "__racer_memory_checkpoint_meta__"
_MANIFEST_DIR = "racer_manifests"
_TREE_MANIFEST_CSD_PREFIX = "__racer_megatron_tree_manifest__:"
_CHECKPOINT_COMMIT_MARKER_CSD_PREFIX = "__racer_megatron_checkpoint_commit_marker__:"
_CHECKPOINT_COMMIT_MARKER_CSD_KIND = "megatron_checkpoint_commit_marker"
_SESSION = session_state.RacerSessionState()


@dataclass
class _PreparedDistributedSave:
    args: Any
    tag: str
    iteration: int
    release: bool
    ckpt_type: str
    device: torch.device
    tree: dict[str, Any]
    payload_chunks: Any
    packet_sizes_by_tag: dict[str, dict[int, int]]
    chunk_tags: list[str]
    runtime: Any
    chunk_storage: Any
    metadata_ms_local: float
    payload_pack_ms_local: float
    payload_checksum_ms_local: float
    payload_checksums: list[str]
    local_leaf_count: int
    max_leaf_count: int
    local_bytes: int
    payload_bytes_total: int
    local_chunk_count: int
    max_chunk_count: int
    chunk_size: int
    store_command_ms_local: float
    distributed_store_handle: Any
    spare_completion_key: str | None
    tensor_view_ms_local: float
    racer_prepare_ms_local: float
    store_started_at: float


@dataclass
class _ExecutedDistributedSave:
    prepared: _PreparedDistributedSave
    tensor_view_ms_local: float
    racer_calls_ms_local: float
    racer_profile_local: dict[str, float]
    chunk_store_ms_values: list[float]
    store_wall_ms_local: float


@dataclass
class _PendingAsyncSave:
    prepared: _PreparedDistributedSave
    future: Future
    scheduled_at: float


def _payload_buffer_pool() -> Any:
    if _SESSION.payload_buffer_pool is None:
        _SESSION.payload_buffer_pool = tensor_tree.PinnedPayloadBufferPool()
    return _SESSION.payload_buffer_pool


def _racer_launch_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    stream = _SESSION.launch_streams.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SESSION.launch_streams[device_index] = stream
    return stream


def _prewarm_payload_buffer_pool_from_tree(tree: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(tree, dict):
        return {}
    chunk_size = int(tree.get("chunk_size", 0) or 0)
    chunk_count = len(list(tree.get("chunks", []) or []))
    if chunk_size <= 0 or chunk_count <= 0:
        return {}
    return dict(
        _payload_buffer_pool().prewarm(
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            device=_current_cuda_device(),
        )
    )


def racer_checkpoint_enabled(args: Any) -> bool:
    return bool(getattr(args, "racer_checkpoint", False))


def racer_distributed_store_enabled(args: Any) -> bool:
    return racer_checkpoint_enabled(args) and bool(getattr(args, "racer_distributed_store", False))


def racer_async_offload_enabled(args: Any) -> bool:
    return (
        racer_distributed_store_enabled(args)
        and bool(getattr(args, "racer_async_offload", False))
    )


_CSD_STORAGE_BACKENDS = {
    "csd_native_pinned",
    "daemon_native_pinned",
    "csd_egm",
    "daemon_egm",
}


def _racer_storage_backend(args: Any) -> str:
    backend = str(getattr(args, "racer_storage_backend", "csd_native_pinned")).lower().replace("-", "_")
    if backend in _CSD_STORAGE_BACKENDS:
        return backend
    raise ValueError(
        f"unsupported --racer-storage-backend={backend!r}; "
        "only csd_native_pinned and csd_egm are allowed. "
        "In-process CUDA, fd/mmap, and other fallback storage paths are disabled."
    )


def _racer_csd_storage_enabled(args: Any) -> bool:
    return _racer_storage_backend(args) in _CSD_STORAGE_BACKENDS


def _racer_csd_delete_coordinator_rank() -> int:
    if os.environ.get("RACER_CSD_PER_NODE", "0").lower() not in {"1", "true", "yes", "on"}:
        return 0
    value = os.environ.get("RACER_CSD_LOCAL_COORDINATOR_RANK")
    if value not in (None, ""):
        return int(value)
    ranks = []
    for item in str(os.environ.get("RACER_CSD_LOCAL_RANKS", "")).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            step = 1 if end >= start else -1
            ranks.extend(range(start, end + step, step))
        else:
            ranks.append(int(item))
    return min(ranks) if ranks else 0


def _is_racer_csd_delete_coordinator() -> bool:
    return _rank() == _racer_csd_delete_coordinator_rank()


def _racer_csd_storage_options(args: Any) -> dict[str, Any]:
    socket_path = getattr(args, "racer_csd_socket_path", None)
    if socket_path:
        return {
            "address": str(socket_path),
            "authkey": getattr(args, "racer_csd_authkey", "racer-csd"),
            "cuda_register_fd_mappings": bool(getattr(args, "racer_csd_cuda_register_fd_mappings", False)),
        }
    port = getattr(args, "racer_csd_port", None)
    if port is None:
        raise RuntimeError(
            "--racer-csd-socket-path or --racer-csd-port is required when "
            "--racer-storage-backend is csd_native_pinned or csd_egm"
        )
    return {
        "address": (str(getattr(args, "racer_csd_host", "127.0.0.1")), int(port)),
        "authkey": getattr(args, "racer_csd_authkey", "racer-csd"),
        "cuda_register_fd_mappings": bool(getattr(args, "racer_csd_cuda_register_fd_mappings", False)),
    }


def _racer_chunk_storage_from_options(storage_backend: str, storage_options: dict[str, Any] | None) -> Any | None:
    normalized = str(storage_backend).lower().replace("-", "_")
    if normalized not in _CSD_STORAGE_BACKENDS:
        return None
    options = dict(storage_options or {})
    cache_key = _racer_chunk_storage_cache_key(normalized, options)
    cached = _SESSION.chunk_storage_clients.get(cache_key)
    if cached is not None:
        return cached
    from racer.csd import CheckpointStorageDaemonClient  # type: ignore

    client = CheckpointStorageDaemonClient(
        options["address"],
        authkey=options.get("authkey"),
        cuda_register_fd_mappings=bool(options.get("cuda_register_fd_mappings", False)),
    )
    _validate_csd_backend_matches_cli(normalized, client)
    _SESSION.chunk_storage_clients[cache_key] = client
    return client


def _racer_chunk_storage_cache_key(storage_backend: str, storage_options: dict[str, Any]) -> tuple[Any, ...]:
    address = storage_options.get("address")
    if isinstance(address, (list, tuple)):
        address_key: Any = tuple(address)
    else:
        address_key = str(address)
    authkey = storage_options.get("authkey")
    if isinstance(authkey, (bytes, bytearray)):
        authkey_key: Any = bytes(authkey)
    else:
        authkey_key = str(authkey)
    return (
        str(storage_backend).lower().replace("-", "_"),
        address_key,
        authkey_key,
        bool(storage_options.get("cuda_register_fd_mappings", False)),
    )


def _validate_csd_backend_matches_cli(storage_backend: str, client: Any) -> None:
    caps = dict(client.capabilities())
    actual_backend = str(caps.get("backend", caps.get("storage_backend", ""))).lower()
    if storage_backend in {"csd_native_pinned", "daemon_native_pinned"}:
        if actual_backend != "native_pinned" or not bool(caps.get("cuda_native_pinned", False)):
            raise RuntimeError(
                "--racer-storage-backend=csd_native_pinned requires CSD backend=native_pinned; "
                f"got backend={actual_backend or '<unknown>'}, capabilities={caps}"
            )
        if not (bool(caps.get("supports_cuda_ipc", False)) and bool(caps.get("supports_async_copy", False))):
            raise RuntimeError(
                "--racer-storage-backend=csd_native_pinned requires CUDA IPC async copy support; "
                f"capabilities={caps}"
            )
    elif storage_backend in {"csd_egm", "daemon_egm"}:
        if actual_backend != "egm" or not bool(caps.get("supports_egm_native_transport", False)):
            raise RuntimeError(
                "--racer-storage-backend=csd_egm requires CSD backend=egm with native handle transport; "
                f"got backend={actual_backend or '<unknown>'}, capabilities={caps}"
            )


def _racer_chunk_storage(args: Any) -> Any | None:
    return _racer_chunk_storage_from_options(
        _racer_storage_backend(args),
        _racer_csd_storage_options(args) if _racer_csd_storage_enabled(args) else None,
    )


def _profile_root(args: Any) -> Path | None:
    root = getattr(args, "racer_profile_dir", None)
    if not root:
        return None
    return Path(root)


def _write_profile_event(args: Any, event: str, report: dict[str, Any]) -> None:
    root = _profile_root(args)
    if root is None:
        return
    rank = _rank()
    payload = {
        "event": str(event),
        "rank": int(rank),
        "world_size": _world_size(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp": time.time(),
        "storage_backend": _racer_storage_backend(args),
        "distributed_store": racer_distributed_store_enabled(args),
    }
    payload.update(dict(report))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rank_{rank:05d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _debug_payload_checksum_mode() -> str:
    value = os.environ.get("RACER_DEBUG_PAYLOAD_CHECKSUM", "").strip().lower().replace("-", "_")
    if value in {"", "0", "false", "no", "off", "disabled", "none"}:
        return ""
    if value in {"1", "true", "yes", "on", "fast", "sampled"}:
        return "sample64"
    if value in {"sample64", "sample64_v1", "sum64"}:
        return "sample64"
    if value in {"sha256", "strict", "strict_sha256"}:
        return "sha256"
    raise ValueError(f"unsupported RACER_DEBUG_PAYLOAD_CHECKSUM={value!r}")


def _debug_payload_checksum(tensor: torch.Tensor, *, mode: str) -> str:
    from racer.checksum import tensor_checksum  # type: ignore

    return tensor_checksum(
        tensor.detach().contiguous().view(-1),
        buffer_size=16 * 1024 * 1024,
        sync_device=lambda device: torch.cuda.synchronize(device) if device.type == "cuda" else None,
        mode=mode,
    )


def _debug_payload_checksums(payloads: Any, *, mode: str) -> tuple[list[str], float]:
    start = time.perf_counter()
    checksums: list[str] = []
    for index, buffer in enumerate(payloads.buffers):
        valid_nbytes = int(payloads.valid_nbytes[index])
        checksums.append(_debug_payload_checksum(buffer.narrow(0, 0, valid_nbytes), mode=mode))
    return checksums, (time.perf_counter() - start) * 1000.0


def capture_distributed_optimizer_state(args: Any, optimizer: Any) -> Any | None:
    """Return Megatron distributed optimizer parameter state without touching disk."""

    return optimizer_state.capture_distributed_optimizer_state(
        racer_checkpoint_enabled(args),
        optimizer,
    )


def _publish_completed_save(args: Any, prepared: _PreparedDistributedSave, report: dict[str, Any]) -> None:
    tag = prepared.tag
    iteration = prepared.iteration
    _SESSION.latest_tag = tag
    _SESSION.tags_by_iteration[int(iteration)] = tag
    _SESSION.reports[tag] = report
    _write_profile_event(args, "save", report)
    _prune_old_checkpoints(args)


def _async_save_executor() -> ThreadPoolExecutor:
    executor = _SESSION.async_save_executor
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="racer-checkpoint-save")
        _SESSION.async_save_executor = executor
    return executor


def _submit_prepared_distributed_save(prepared: _PreparedDistributedSave) -> dict[str, Any]:
    if _SESSION.pending_async_save is not None:
        raise RuntimeError("RACER async save queue already has an in-flight checkpoint")
    scheduled_at = time.perf_counter()
    future = _async_save_executor().submit(_execute_prepared_distributed_store, prepared)
    _SESSION.pending_async_save = _PendingAsyncSave(
        prepared=prepared,
        future=future,
        scheduled_at=scheduled_at,
    )
    report = {
        "tag": prepared.tag,
        "iteration": int(prepared.iteration),
        "release": bool(prepared.release),
        "ckpt_type": str(prepared.ckpt_type),
        "async_scheduled": True,
        "payload_bytes_total": int(prepared.payload_bytes_total),
        "payload_bytes_local": int(prepared.local_bytes),
        "racer_chunk_count_local": int(prepared.local_chunk_count),
        "racer_chunk_count_max": int(prepared.max_chunk_count),
    }
    _print_rank0(
        "RACER async checkpoint scheduled: "
        f"iteration={prepared.iteration}, tag={prepared.tag}, "
        f"bytes={prepared.payload_bytes_total}, chunks={prepared.max_chunk_count}"
    )
    return report


def _async_future_status(future: Future) -> tuple[bool, BaseException | None, bool]:
    """Return all-rank completion/error state with one packed reduction.

    Error is checked even when another rank is still running.  The previous
    done-MIN-then-error-SUM sequence could hide an early worker failure forever
    when a peer subsequently stalled in runtime P2P or CSD wait.
    """

    local_done = bool(future.done())
    local_error: BaseException | None = None
    if local_done:
        try:
            local_error = future.exception()
        except BaseException as exc:
            local_error = exc
    if not _distributed_initialized():
        return local_done, local_error, local_error is not None

    status = torch.tensor(
        [1 if local_error is not None else 0, 0 if local_done else 1],
        dtype=torch.int32,
        device=_current_cuda_device(),
    )
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
    any_error, any_incomplete = [int(value) for value in status.cpu().tolist()]
    return not bool(any_incomplete), local_error, bool(any_error)


def maybe_finalize_async_saves(blocking: bool = False, terminate: bool = False) -> bool:
    del terminate  # The executor is reusable because Megatron may schedule a final save after train().
    pending = _SESSION.pending_async_save
    if pending is None:
        return False

    while True:
        all_done, local_error, any_error = _async_future_status(pending.future)
        if any_error:
            break
        if all_done:
            break
        if not blocking:
            return False
        time.sleep(0.01)

    if any_error:
        local_detail = (
            f"{type(local_error).__name__}: {local_error}"
            if local_error is not None
            else "the local worker completed, but another rank's worker failed"
        )
        raise RuntimeError(
            "RACER async checkpoint failed before default-process-group finalization on "
            f"one or more ranks; {local_detail}"
        ) from local_error

    executed = pending.future.result()

    report = _finalize_executed_distributed_store(executed)
    commit_wall_ms_local = (time.perf_counter() - pending.scheduled_at) * 1000.0
    report["racer_async_commit_wall_ms_max"] = _max_float_across_ranks(
        commit_wall_ms_local,
        pending.prepared.device,
    )
    commit_wall_ms = float(report["racer_async_commit_wall_ms_max"])
    if commit_wall_ms > 0.0:
        report["racer_async_logical_commit_bandwidth_gbps"] = (
            float(report.get("payload_bytes_total", 0)) / 1_000_000_000.0
        ) / (commit_wall_ms / 1000.0)
        report["racer_async_encoded_commit_bandwidth_gbps"] = (
            float(report.get("racer_encoded_storage_bytes_total", 0)) / 1_000_000_000.0
        ) / (commit_wall_ms / 1000.0)
    else:
        report["racer_async_logical_commit_bandwidth_gbps"] = 0.0
        report["racer_async_encoded_commit_bandwidth_gbps"] = 0.0
    _SESSION.pending_async_save = None
    _publish_completed_save(pending.prepared.args, pending.prepared, report)
    _print_rank0(
        "RACER async checkpoint committed: "
        f"iteration={pending.prepared.iteration}, tag={pending.prepared.tag}, "
        f"commit_wall={float(report['racer_async_commit_wall_ms_max']):.2f} ms"
    )
    return True


def save_memory_checkpoint(
    args: Any,
    *,
    iteration: int,
    release: bool,
    ckpt_type: str,
    state_dict: dict[str, Any],
    distributed_optimizer_state: Any | None = None,
    content_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Store the local Megatron state_dict in RACER CUDA memory."""

    if not racer_checkpoint_enabled(args):
        return None

    async_offload = racer_async_offload_enabled(args)
    if async_offload:
        if bool(getattr(args, "racer_verify_on_save", False)):
            raise RuntimeError("--racer-verify-on-save is not supported with --racer-async-offload")
        # The current pinned arena is single-generation. Backpressure here is normally
        # a no-op at save_interval>1, and prevents overwriting an in-flight snapshot.
        maybe_finalize_async_saves(blocking=True)

    rank = _rank()
    tag = _tag_for_iteration(iteration, release)
    state_for_save = dict(state_dict)
    if content_metadata is not None:
        state_for_save["content_metadata"] = content_metadata
    if distributed_optimizer_state is not None:
        state_for_save[DISTRIB_OPTIM_STATE_KEY] = distributed_optimizer_state
    state_for_save[_META_KEY] = {
        "iteration": int(iteration),
        "release": bool(release),
        "tag": tag,
        "rank": int(rank),
        "ckpt_type": str(ckpt_type),
    }
    _prepare_state_for_save(state_for_save, ckpt_type=ckpt_type)

    device = _current_cuda_device()

    if racer_distributed_store_enabled(args):
        if async_offload:
            prepared = _prepare_distributed_store_tensor_tree(
                args,
                tag=tag,
                iteration=iteration,
                release=release,
                ckpt_type=ckpt_type,
                state_for_save=state_for_save,
                device=device,
            )
            prepared.tree.pop("source_tensors", None)
            return _submit_prepared_distributed_save(prepared)
        else:
            report = _distributed_store_tensor_tree(
                args,
                tag=tag,
                iteration=iteration,
                release=release,
                ckpt_type=ckpt_type,
                state_for_save=state_for_save,
                device=device,
            )
    else:
        report = _local_store_tensor_tree(
            args,
            tag=tag,
            iteration=iteration,
            release=release,
            ckpt_type=ckpt_type,
            state_for_save=state_for_save,
            device=device,
        )

    _SESSION.latest_tag = tag
    _SESSION.tags_by_iteration[int(iteration)] = tag
    _SESSION.reports[tag] = report
    _write_profile_event(args, "save", report)
    _prune_old_checkpoints(args)
    _barrier()
    if bool(getattr(args, "racer_verify_on_save", False)):
        verify_memory_checkpoint(args, iteration=iteration, release=release)
    return report


def load_memory_checkpoint(
    args: Any,
    *,
    iteration: int | None = None,
    release: bool = False,
    failed_train_ranks: list[int] | None = None,
) -> tuple[dict[str, Any], str, int, bool, dict[str, Any]] | None:
    """Load this rank's state_dict from a daemon-backed RACER checkpoint."""

    if not racer_checkpoint_enabled(args):
        return None
    if racer_async_offload_enabled(args):
        maybe_finalize_async_saves(blocking=True)
    failed_train_ranks = _resolve_failed_train_ranks(args, failed_train_ranks)
    replacement_mapping = _parse_rank_mapping(getattr(args, "racer_replacement_mapping", None))
    rank = _rank()
    selection_error: BaseException | None = None
    selection_error_text: str | None = None
    if rank == 0:
        try:
            tag = _select_tag(iteration)
            if tag is not None and not _manifest_chunks_committed(args, str(tag)):
                raise RuntimeError(
                    f"RACER checkpoint {tag!r} is not fully committed and resident in the rank-0 CSD"
                )
            if tag is None:
                tag = _select_manifest_tag(args, iteration, release)
        except BaseException as exc:
            tag = None
            selection_error = exc
            selection_error_text = f"{type(exc).__name__}: {exc}"
    else:
        tag = None

    if _distributed_initialized():
        control_box = [tag, failed_train_ranks, replacement_mapping, selection_error_text]
        torch.distributed.broadcast_object_list(control_box, src=0)
        tag = control_box[0]
        failed_train_ranks = [int(rank) for rank in control_box[1]]
        replacement_mapping = {int(k): int(v) for k, v in dict(control_box[2]).items()}
        selection_error_text = control_box[3]
    if selection_error_text is not None:
        raise RuntimeError(
            f"RACER checkpoint selection failed consistently across ranks: {selection_error_text}"
        ) from selection_error
    found = tag is not None
    if found and racer_distributed_store_enabled(args) and _racer_csd_storage_enabled(args):
        unavailable_csd_nodes = _unavailable_checkpoint_csd_nodes(args, str(tag))
        if unavailable_csd_nodes:
            raise RuntimeError(
                f"RACER checkpoint {tag!r} is not committed and data-resident on "
                f"{unavailable_csd_nodes} CSD node(s)"
            )
    loaded_iteration = _iteration_from_tag(str(tag)) if tag is not None else -1
    if not found:
        if release or (iteration is not None and int(iteration) >= 0):
            raise RuntimeError(
                "requested RACER checkpoint is not available or not fully committed: "
                f"iteration={iteration}, release={release}"
            )
        if racer_distributed_store_enabled(args) and _distributed_initialized():
            prewarm = _prewarm_distributed_runtime_and_staging(args)
            _print_rank0(
                "RACER distributed memory runtime prewarmed: "
                f"total={float(prewarm.get('total_ms', 0.0)):.2f} ms, "
                f"runtime={float(prewarm.get('runtime_ms', 0.0)):.2f} ms, "
                f"staging={float(prewarm.get('staging_ms', 0.0)):.2f} ms, "
                f"staging_slabs={int(prewarm.get('staging_slabs', 0.0))}, "
                f"payload_pool={float(prewarm.get('payload_pool_ms', 0.0)):.2f} ms, "
                f"payload_chunks={int(prewarm.get('payload_pool_chunks', 0.0))}, "
                f"payload_cuda_slots={int(prewarm.get('payload_cuda_slots', 0.0))}, "
                f"store_path={float(prewarm.get('store_path_ms', 0.0)):.2f} ms"
            )
        return None

    load_tag = str(tag)
    _SESSION.latest_tag = load_tag
    _SESSION.tags_by_iteration[int(loaded_iteration)] = load_tag
    _ensure_tree_checkpoint(args, load_tag, rank=_replacement_source_rank_for_current_process(replacement_mapping))
    if load_tag in _SESSION.tree_checkpoints:
        state_dict, report = _distributed_load_tensor_tree(
            args,
            tag=load_tag,
            failed_train_ranks=failed_train_ranks,
            replacement_mapping=replacement_mapping,
            requested_train_ranks=_train_ranks(args),
        ) if racer_distributed_store_enabled(args) else _local_load_tensor_tree(
            args,
            tag=load_tag,
            failed_train_ranks=failed_train_ranks,
            replacement_mapping=replacement_mapping,
            requested_train_ranks=_train_ranks(args),
        )
        loaded_release = bool(state_dict.get(_META_KEY, {}).get("release", release))
        loaded_ckpt_type = str(state_dict.get(_META_KEY, {}).get("ckpt_type", "LEGACY"))
        state_dict.pop(_META_KEY, None)
        checkpoint_name = f"racer-memory://{_tag_for_iteration(loaded_iteration, loaded_release)}"
        report.update(
            {
                "iteration": int(loaded_iteration),
                "release": loaded_release,
                "ckpt_type": loaded_ckpt_type,
                "checkpoint_name": checkpoint_name,
            }
        )
        if os.environ.get("RACER_PAYLOAD_POOL_PREWARM_AFTER_LOAD", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            payload_pool_prewarm = _prewarm_payload_buffer_pool_from_tree(
                _SESSION.tree_checkpoints.get(load_tag)
            )
        else:
            payload_pool_prewarm = {}
            report["racer_payload_pool_tree_prewarm_skipped_after_load"] = 1
        if payload_pool_prewarm:
            device = _current_cuda_device()
            report["racer_payload_pool_prewarm_after_load_ms"] = _max_float_across_ranks(
                float(payload_pool_prewarm.get("payload_pool_ensure_ms", 0.0)), device
            )
            report["racer_payload_pool_allocated_after_load"] = int(
                _max_float_across_ranks(float(payload_pool_prewarm.get("payload_pool_allocated_buffers", 0.0)), device)
            )
            report["racer_payload_pool_reused_after_load"] = int(
                _max_float_across_ranks(float(payload_pool_prewarm.get("payload_pool_reused_buffers", 0.0)), device)
            )
            report["racer_payload_pool_reserved_after_load"] = int(
                _max_float_across_ranks(float(payload_pool_prewarm.get("payload_pool_reserved_nbytes", 0.0)), device)
            )
        if racer_distributed_store_enabled(args) and _distributed_initialized():
            prewarm = _prewarm_distributed_runtime_and_staging(args)
            report["racer_runtime_prewarm_after_load_ms"] = float(prewarm.get("total_ms", 0.0))
            report["racer_runtime_init_after_load_ms"] = float(prewarm.get("runtime_ms", 0.0))
            report["racer_staging_prewarm_after_load_ms"] = float(prewarm.get("staging_ms", 0.0))
            report["racer_staging_slabs_after_load"] = int(prewarm.get("staging_slabs", 0.0))
            report["racer_payload_pool_config_prewarm_after_load_ms"] = float(prewarm.get("payload_pool_ms", 0.0))
            report["racer_payload_pool_config_chunks_after_load"] = int(prewarm.get("payload_pool_chunks", 0.0))
        _write_profile_event(args, "load", report)
    else:
        raise KeyError(f"RACER tensor-tree checkpoint tag is not available in this process: {load_tag}")
    if rank == 0:
        store_mode = "distributed" if racer_distributed_store_enabled(args) else "local"
        _print_rank0(
            f"RACER {store_mode} memory checkpoint loaded: "
            f"tag={_tag_for_iteration(loaded_iteration, loaded_release)}, "
            f"total={report['total_ms_local']:.2f} ms, "
            f"racer_fetch={float(report.get('racer_fetch_ms', report.get('load_scatter_ms', 0.0))):.2f} ms, "
            f"load_read_wait={float(report.get('racer_load_read_wait_ms_max', 0.0)):.2f} ms, "
            f"load_route={float(report.get('racer_load_route_total_ms_max', 0.0)):.2f} ms, "
            f"load_barrier={float(report.get('racer_load_final_barrier_ms_max', 0.0)):.2f} ms, "
            f"tensor_materialize={float(report.get('tensor_materialize_ms', 0.0)):.2f} ms, "
            f"tensor_scatter_sync={float(report.get('tensor_scatter_sync_ms', 0.0)):.2f} ms, "
            f"tree_decode={float(report.get('tree_decode_ms', 0.0)):.2f} ms, "
            f"unpack={float(report.get('unpack_ms', 0.0)):.2f} ms, "
            f"tensor_rebuild={float(report.get('tensor_rebuild_ms', 0.0)):.2f} ms, "
            f"payload_pool_prewarm_after_load={float(report.get('racer_payload_pool_prewarm_after_load_ms', 0.0)):.2f} ms, "
            f"payload_pool_config_prewarm_after_load={float(report.get('racer_payload_pool_config_prewarm_after_load_ms', 0.0)):.2f} ms, "
            f"runtime_prewarm_after_load={float(report.get('racer_runtime_prewarm_after_load_ms', 0.0)):.2f} ms, "
            f"runtime_init_after_load={float(report.get('racer_runtime_init_after_load_ms', 0.0)):.2f} ms, "
            f"staging_prewarm_after_load={float(report.get('racer_staging_prewarm_after_load_ms', 0.0)):.2f} ms"
        )
    _barrier()
    return state_dict, checkpoint_name, loaded_iteration, loaded_release, report


def verify_memory_checkpoint(
    args: Any,
    *,
    iteration: int,
    release: bool,
    local_payload: torch.Tensor | None = None,
) -> None:
    """Read this rank from RACER memory and compare the raw CUDA payload."""

    if racer_async_offload_enabled(args):
        maybe_finalize_async_saves(blocking=True)
    rank = _rank()
    tag = _tag_for_iteration(iteration, release)
    _ensure_tree_checkpoint(args, tag)
    if tag in _SESSION.tree_checkpoints:
        _barrier()
        if racer_distributed_store_enabled(args):
            ok, metrics = _distributed_verify_tensor_tree(args, tag=tag)
        else:
            ok, metrics = _local_verify_tensor_tree(args, tag=tag)
        if rank == 0:
            if not ok:
                raise RuntimeError("RACER tensor-tree memory checkpoint verify-on-save failed")
            _print_rank0(
                "RACER tensor-tree memory checkpoint verify-on-save passed: "
                f"tag={tag}, "
                f"load_scatter={float(metrics.get('load_scatter_ms', 0.0)):.2f} ms, "
                f"compare={float(metrics.get('compare_ms', 0.0)):.2f} ms"
            )
        _barrier()
        return

    raise KeyError(f"RACER tensor-tree checkpoint tag is not available in this process: {tag}")


def pop_distributed_optimizer_state(state_dict: dict[str, Any]) -> Any | None:
    return optimizer_state.pop_distributed_optimizer_state(state_dict)


def load_distributed_optimizer_state(
    optimizer: Any,
    parameter_state: Any,
    *,
    update_legacy_format: bool = False,
) -> None:
    """Load distributed optimizer parameter state captured in the memory payload."""

    optimizer_state.load_distributed_optimizer_state(
        optimizer,
        parameter_state,
        update_legacy_format=update_legacy_format,
    )


def latest_report() -> dict[str, Any] | None:
    if _SESSION.latest_tag is None:
        return None
    return _SESSION.reports.get(_SESSION.latest_tag)


def reset_session_state() -> None:
    _SESSION.reset()


def _prepare_state_for_save(state_for_save: dict[str, Any], *, ckpt_type: str) -> None:
    if str(ckpt_type) == "LEGACY":
        return
    try:
        from megatron.core.dist_checkpointing.mapping import apply_factories

        apply_factories(state_for_save)
    except Exception:
        return


def _parse_rank_mapping(value: Any) -> dict[int, int]:
    return rank_map.parse_rank_mapping(value)


def _resolve_failed_train_ranks(args: Any, failed_train_ranks: list[int] | None) -> list[int]:
    return rank_map.resolve_failed_train_ranks(
        args,
        failed_train_ranks,
        current_rank=_rank(),
        train=_train_ranks(args),
    )


def _manifest_root(args: Any) -> Path:
    root = getattr(args, "racer_manifest_dir", None)
    if root:
        return Path(root)
    base = getattr(args, "save", None) or getattr(args, "load", None) or os.getcwd()
    return Path(base) / _MANIFEST_DIR


def _manifest_filename(tag: str) -> str:
    return tag.replace(":", "__").replace("/", "_")


def _tree_manifest_csd_tag(tag: str, rank: int) -> str:
    return f"{_TREE_MANIFEST_CSD_PREFIX}rank_{int(rank):05d}:{str(tag)}"


def _checkpoint_tag_from_tree_manifest_csd_tag(tag: str, rank: int = 0) -> str | None:
    prefix = f"{_TREE_MANIFEST_CSD_PREFIX}rank_{int(rank):05d}:"
    if not str(tag).startswith(prefix):
        return None
    return str(tag)[len(prefix) :]


def _manifest_path(args: Any, tag: str, rank: int | None = None) -> Path:
    manifest_rank = _rank() if rank is None else int(rank)
    return _manifest_root(args) / f"{_manifest_filename(tag)}.rank_{manifest_rank:05d}.pt"


def _checkpoint_commit_marker_path(args: Any, tag: str) -> Path:
    return _manifest_root(args) / f"{_manifest_filename(tag)}.committed.json"


def _checkpoint_commit_marker_csd_tag(tag: str) -> str:
    return f"{_CHECKPOINT_COMMIT_MARKER_CSD_PREFIX}{str(tag)}"


def _checkpoint_commit_marker_payload(
    tag: str,
    iteration: int,
    generation: str,
) -> dict[str, Any]:
    return {
        "version": 1,
        "tag": str(tag),
        "iteration": int(iteration),
        "generation": str(generation),
    }


_CSD_DERIVED_METADATA_FIELDS = frozenset(
    {
        "backend_capabilities",
        "checkpoint_state",
        "committed",
        "daemon_owned",
        "data_resident",
        "expected_chunks",
        "storage_backend",
        "total_valid_bytes",
    }
)


def _csd_metadata_payload_matches(record: Any, desired: dict[str, Any]) -> bool:
    """Compare the original metadata payload, excluding CSD-derived fields."""

    if not isinstance(record, dict):
        return False
    payload = {
        key: value
        for key, value in record.items()
        if key not in _CSD_DERIVED_METADATA_FIELDS
    }
    if "chunks" not in desired and payload.get("chunks") == []:
        payload.pop("chunks")
    return payload == dict(desired)


def _tree_manifest(tree: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        key: value
        for key, value in tree.items()
        if key not in {"source_tensors"}
    }
    if bool(manifest.get("requires_commit_marker", False)):
        manifest["manifest_version"] = 3
        manifest["commit_metadata_backend"] = "csd_v1"
    else:
        manifest["manifest_version"] = 1
    return manifest


def _write_checkpoint_commit_marker(
    args: Any,
    tag: str,
    iteration: int,
    generation: str,
) -> None:
    path = _checkpoint_commit_marker_path(args, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = _checkpoint_commit_marker_payload(tag, iteration, generation)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_checkpoint_commit_marker_to_csd(
    args: Any,
    tag: str,
    iteration: int,
    generation: str,
) -> bool:
    storage = _racer_chunk_storage(args)
    if storage is None:
        raise RuntimeError("RACER CSD storage is unavailable for checkpoint commit marker")
    marker = _checkpoint_commit_marker_payload(tag, iteration, generation)
    marker.update(
        {
            "manifest_version": 1,
            "racer_manifest_kind": _CHECKPOINT_COMMIT_MARKER_CSD_KIND,
            "checkpoint_tag": str(tag),
        }
    )
    csd_tag = _checkpoint_commit_marker_csd_tag(tag)
    atomic_commit = getattr(storage, "commit_metadata", None)
    if callable(atomic_commit):
        try:
            result = atomic_commit(csd_tag, marker)
            return bool(result.get("created", True)) if isinstance(result, dict) else True
        except Exception as exc:
            confirmed_absent = False
            try:
                committed = dict(storage.get_manifest(csd_tag))
            except KeyError:
                committed = None
                confirmed_absent = True
            except Exception:
                committed = None
            else:
                if _csd_metadata_payload_matches(committed, marker):
                    # The server committed, but the create response was lost.
                    # Treat ownership as uncertain so a failure on another CSD
                    # cannot trigger a destructive rollback here.
                    return False
            if confirmed_absent:
                try:
                    storage.delete(csd_tag)
                except Exception:
                    pass
            raise RuntimeError(
                f"failed to atomically store RACER checkpoint commit marker in CSD for tag {tag!r}"
            ) from exc

    # Compatibility path for storage implementations that predate the
    # metadata-only single-RPC protocol.
    try:
        existing = dict(storage.get_manifest(csd_tag))
    except KeyError:
        existing = None
    if existing is not None:
        if _csd_metadata_payload_matches(existing, marker):
            return False
        raise RuntimeError(
            f"CSD checkpoint commit marker {csd_tag!r} already exists with different content"
        )
    try:
        storage.begin(csd_tag, marker, expected_chunks=0)
        storage.put_manifest(csd_tag, marker)
        storage.commit(csd_tag)
        return True
    except Exception as exc:
        confirmed_absent = False
        try:
            committed = dict(storage.get_manifest(csd_tag))
        except KeyError:
            committed = None
            confirmed_absent = True
        except Exception:
            committed = None
        else:
            if _csd_metadata_payload_matches(committed, marker):
                return False
        if confirmed_absent:
            # A confirmed non-committed/absent record is safe to remove. If the
            # status RPC itself was unavailable, the branch above deliberately
            # avoids issuing a destructive delete under uncertainty.
            try:
                storage.delete(csd_tag)
            except Exception:
                pass
        raise RuntimeError(
            f"failed to store RACER checkpoint commit marker in CSD for tag {tag!r}"
        ) from exc


def _delete_checkpoint_commit_marker_from_csd(args: Any, tag: str) -> None:
    storage = _racer_chunk_storage(args)
    if storage is not None:
        storage.delete(_checkpoint_commit_marker_csd_tag(tag))


def _publish_tree_manifest_with_commit_marker(prepared: _PreparedDistributedSave) -> dict[str, float]:
    publish_started_at = time.perf_counter()
    tree_manifest_started_at = time.perf_counter()
    local_manifest_error: BaseException | None = None
    try:
        _write_tree_manifest(prepared.args, prepared.tag, prepared.tree)
    except BaseException as exc:
        local_manifest_error = exc
    failed_manifest_ranks = _sum_int_across_ranks(
        1 if local_manifest_error is not None else 0,
        prepared.device,
    )
    if failed_manifest_ranks:
        detail = (
            f"{type(local_manifest_error).__name__}: {local_manifest_error}"
            if local_manifest_error is not None
            else "another rank failed to publish its tree manifest"
        )
        raise RuntimeError(
            "RACER checkpoint tree-manifest publication failed on "
            f"{failed_manifest_ranks} rank(s); {detail}"
        ) from local_manifest_error
    tree_manifest_publish_ms_local = (time.perf_counter() - tree_manifest_started_at) * 1000.0

    generation = str(prepared.tree["checkpoint_generation"])
    csd_marker_started_at = time.perf_counter()
    local_csd_marker_error: BaseException | None = None
    local_csd_marker_created = False
    if _is_racer_csd_delete_coordinator():
        try:
            local_csd_marker_created = _write_checkpoint_commit_marker_to_csd(
                prepared.args,
                prepared.tag,
                prepared.iteration,
                generation,
            )
        except BaseException as exc:
            local_csd_marker_error = exc
    failed_csd_marker_ranks = _sum_int_across_ranks(
        1 if local_csd_marker_error is not None else 0,
        prepared.device,
    )
    if failed_csd_marker_ranks:
        if _is_racer_csd_delete_coordinator() and local_csd_marker_created:
            try:
                _delete_checkpoint_commit_marker_from_csd(prepared.args, prepared.tag)
            except Exception:
                pass
        detail = (
            f"{type(local_csd_marker_error).__name__}: {local_csd_marker_error}"
            if local_csd_marker_error is not None
            else "another CSD node failed to publish the checkpoint commit marker"
        )
        raise RuntimeError(
            "RACER checkpoint CSD commit-marker publication failed; "
            f"{detail}"
        ) from local_csd_marker_error
    csd_marker_publish_ms_local = (time.perf_counter() - csd_marker_started_at) * 1000.0

    shared_marker_started_at = time.perf_counter()
    local_file_marker_error: BaseException | None = None
    if _rank() == 0:
        try:
            _write_checkpoint_commit_marker(
                prepared.args,
                prepared.tag,
                prepared.iteration,
                generation,
            )
        except BaseException as exc:
            local_file_marker_error = exc
    failed_file_marker_ranks = _sum_int_across_ranks(
        1 if local_file_marker_error is not None else 0,
        prepared.device,
    )
    if failed_file_marker_ranks:
        detail = (
            f"{type(local_file_marker_error).__name__}: {local_file_marker_error}"
            if local_file_marker_error is not None
            else "rank 0 failed to publish the shared checkpoint commit-marker cache"
        )
        _print_rank0(
            "WARNING: RACER checkpoint is committed in every storage-bearing training CSD, but the "
            f"shared commit-marker cache was not published; {detail}"
        )
    shared_marker_publish_ms_local = (time.perf_counter() - shared_marker_started_at) * 1000.0
    return {
        "tree_manifest_publish_ms": tree_manifest_publish_ms_local,
        "csd_commit_marker_publish_ms": csd_marker_publish_ms_local,
        "shared_commit_marker_cache_ms": shared_marker_publish_ms_local,
        "checkpoint_metadata_publish_ms": (time.perf_counter() - publish_started_at) * 1000.0,
    }


def _write_tree_manifest(args: Any, tag: str, tree: dict[str, Any]) -> None:
    tree["source_rank"] = int(tree.get("source_rank", _rank()))
    manifest = _tree_manifest(tree)
    csd_authoritative = _racer_csd_storage_enabled(args)
    if csd_authoritative:
        storage = _racer_chunk_storage(args)
        if storage is None:
            raise RuntimeError("RACER CSD storage is unavailable for tensor-tree manifest")
        csd_manifest = {
            "manifest_version": 1,
            "racer_manifest_kind": "megatron_tensor_tree",
            "checkpoint_tag": str(tag),
            "source_rank": int(tree["source_rank"]),
            "tree_pickle_b64": base64.b64encode(pickle.dumps(manifest, protocol=4)).decode("ascii"),
        }
        csd_tag = _tree_manifest_csd_tag(tag, int(tree["source_rank"]))
        generation = str(manifest.get("checkpoint_generation", ""))
        atomic_commit = getattr(storage, "commit_metadata", None)
        if callable(atomic_commit):
            try:
                atomic_commit(csd_tag, csd_manifest)
            except Exception as exc:
                try:
                    committed = dict(storage.get_manifest(csd_tag))
                except Exception:
                    committed = None
                if not _csd_metadata_payload_matches(committed, csd_manifest):
                    raise RuntimeError(
                        f"failed to atomically store RACER tensor-tree manifest in CSD for tag {tag!r}"
                    ) from exc
        else:
            # Compatibility path for storage implementations that predate the
            # metadata-only single-RPC protocol.
            try:
                existing = dict(storage.get_manifest(csd_tag))
            except KeyError:
                existing = None
            if existing is not None and not _csd_metadata_payload_matches(
                existing, csd_manifest
            ):
                raise RuntimeError(
                    f"CSD tensor-tree manifest {csd_tag!r} already exists with different content"
                )
            try:
                if existing is None:
                    storage.begin(csd_tag, csd_manifest, expected_chunks=0)
                    storage.put_manifest(csd_tag, csd_manifest)
                    storage.commit(csd_tag)
            except Exception as exc:
                try:
                    committed = dict(storage.get_manifest(csd_tag))
                except Exception:
                    committed = None
                if not _csd_metadata_payload_matches(committed, csd_manifest):
                    raise RuntimeError(
                        f"failed to store RACER tensor-tree manifest in CSD for tag {tag!r}"
                    ) from exc

    path = _manifest_path(args, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(manifest, temporary)
        os.replace(temporary, path)
    except BaseException as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if not csd_authoritative:
            raise
        _print_rank0(
            "WARNING: RACER tensor-tree manifest is committed in CSD, but its "
            f"shared-file cache was not published for tag {tag!r}: "
            f"{type(exc).__name__}: {exc}"
        )


def _load_tree_manifest(args: Any, tag: str, rank: int | None = None) -> dict[str, Any] | None:
    manifest_rank = _rank() if rank is None else int(rank)
    path = _manifest_path(args, tag, rank=manifest_rank)
    manifest = None
    should_try_csd = bool(_SESSION.chunk_storage_clients) or bool(
        getattr(args, "racer_csd_socket_path", None)
        or getattr(args, "racer_csd_port", None) is not None
    )
    if should_try_csd:
        # Only a confirmed missing CSD tag may fall through to an old shared
        # manifest.  A transport/control-plane error must fail closed; otherwise
        # a stale v2 file could bypass v3 CSD generation authority.
        manifest = _load_tree_manifest_from_csd(args, tag, manifest_rank)
    if manifest is None and path.exists():
        manifest = torch.load(path, map_location="cpu", weights_only=False)
    if manifest is None:
        return None
    if not isinstance(manifest, dict):
        raise TypeError(f"RACER manifest at {path} is not a dict")
    manifest.setdefault("source_rank", int(manifest_rank))
    manifest.setdefault("source_tensors", [])
    return manifest


def _load_tree_manifest_from_csd(args: Any, tag: str, rank: int) -> dict[str, Any] | None:
    if not _racer_csd_storage_enabled(args):
        return None
    storage = _racer_chunk_storage(args)
    if storage is None:
        return None
    try:
        raw = storage.get_manifest(_tree_manifest_csd_tag(tag, rank))
    except KeyError:
        return None
    manifest = dict(raw)
    if manifest.get("racer_manifest_kind") == "megatron_tensor_tree" and isinstance(
        manifest.get("tree_pickle_b64"), str
    ):
        tree_manifest = pickle.loads(base64.b64decode(str(manifest["tree_pickle_b64"]).encode("ascii")))
        if not isinstance(tree_manifest, dict):
            raise TypeError(f"RACER CSD tensor-tree manifest for tag {tag!r} rank {rank} is not a dict")
    elif manifest.get("racer_manifest_kind") == "megatron_tensor_tree" and isinstance(manifest.get("tree"), dict):
        tree_manifest = dict(manifest["tree"])
    else:
        tree_manifest = manifest
    tree_manifest.setdefault("source_rank", int(rank))
    return tree_manifest


def _tree_manifest_available(args: Any, tag: str, rank: int = 0) -> bool:
    return _manifest_path(args, tag, rank=rank).exists() or _load_tree_manifest_from_csd(args, tag, rank) is not None


def _load_checkpoint_commit_marker_from_csd(
    args: Any,
    tag: str,
    *,
    strict: bool = False,
) -> dict[str, Any] | None:
    if not _racer_csd_storage_enabled(args):
        return None
    try:
        storage = _racer_chunk_storage(args)
        if storage is None:
            return None
        marker = storage.get_manifest(_checkpoint_commit_marker_csd_tag(tag))
    except Exception:
        if strict:
            raise
        # Old checkpoints have only the shared-file marker. Treat an absent or
        # unreachable CSD marker as a cache miss so they remain loadable.
        return None
    return dict(marker)


def _checkpoint_commit_marker_matches(
    marker: Any,
    tag: str,
    expected_generation: str,
) -> bool:
    return (
        isinstance(marker, dict)
        and str(marker.get("tag", "")) == str(tag)
        and str(marker.get("generation", "")) == str(expected_generation)
    )


def _checkpoint_commit_marker_available(
    args: Any,
    tag: str,
    expected_generation: str,
) -> bool:
    csd_marker = _load_checkpoint_commit_marker_from_csd(args, tag)
    if csd_marker is not None:
        return (
            csd_marker.get("racer_manifest_kind") == _CHECKPOINT_COMMIT_MARKER_CSD_KIND
            and str(csd_marker.get("checkpoint_tag", "")) == str(tag)
            and _checkpoint_commit_marker_matches(csd_marker, tag, expected_generation)
        )

    path = _checkpoint_commit_marker_path(args, tag)
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return _checkpoint_commit_marker_matches(marker, tag, expected_generation)


def _replacement_sources_by_target(replacement_mapping: dict[int, int] | None) -> dict[int, int]:
    """Invert source-to-target routing for source-keyed recovered payloads."""

    sources_by_target: dict[int, int] = {}
    for source_rank, target_rank in dict(replacement_mapping or {}).items():
        source_rank = int(source_rank)
        target_rank = int(target_rank)
        if target_rank in sources_by_target and sources_by_target[target_rank] != source_rank:
            raise ValueError(
                "replacement_mapping is ambiguous: source ranks "
                f"{sources_by_target[target_rank]} and {source_rank} both map "
                f"to target rank {target_rank}"
            )
        sources_by_target[target_rank] = source_rank
    return sources_by_target


def _replacement_source_rank_for_current_process(replacement_mapping: dict[int, int] | None) -> int:
    rank = _rank()
    return _replacement_sources_by_target(replacement_mapping).get(rank, rank)


def _ensure_tree_checkpoint(args: Any, tag: str, rank: int | None = None) -> None:
    manifest_rank = _rank() if rank is None else int(rank)
    existing = _SESSION.tree_checkpoints.get(tag)
    if existing is not None and int(existing.get("source_rank", manifest_rank)) == manifest_rank:
        return
    manifest = _load_tree_manifest(args, tag, rank=manifest_rank)
    if manifest is not None:
        _SESSION.tree_checkpoints[tag] = manifest
        iteration = _iteration_from_tag(tag)
        if iteration >= 0:
            _SESSION.tags_by_iteration[int(iteration)] = tag


def _manifest_chunks_committed(args: Any, tag: str, *, manifest_rank: int = 0) -> bool:
    if not _racer_csd_storage_enabled(args):
        return True
    manifest = _load_tree_manifest(args, tag, rank=int(manifest_rank))
    if manifest is None:
        return False
    if str(manifest.get("commit_metadata_backend", "")) == "csd_v1":
        generation = str(manifest.get("checkpoint_generation", ""))
        if not generation:
            return False
        try:
            csd_tree = _load_tree_manifest_from_csd(args, tag, int(manifest_rank))
            csd_marker = _load_checkpoint_commit_marker_from_csd(args, tag, strict=True)
        except Exception:
            return False
        if not (
            isinstance(csd_tree, dict)
            and str(csd_tree.get("commit_metadata_backend", "")) == "csd_v1"
            and str(csd_tree.get("checkpoint_generation", "")) == generation
            and isinstance(csd_marker, dict)
            and csd_marker.get("racer_manifest_kind") == _CHECKPOINT_COMMIT_MARKER_CSD_KIND
            and str(csd_marker.get("checkpoint_tag", "")) == str(tag)
            and _checkpoint_commit_marker_matches(csd_marker, tag, generation)
        ):
            return False
        manifest = csd_tree
    storage = _racer_chunk_storage(args)
    if storage is None:
        return False
    max_chunk_count = int(manifest.get("max_chunk_count", manifest.get("max_leaf_count", 0)))
    for chunk_index in range(max_chunk_count):
        try:
            child_manifest = dict(storage.get_manifest(_chunk_tag(tag, chunk_index)))
        except Exception:
            return False
        if not (
            bool(child_manifest.get("committed", False))
            and bool(child_manifest.get("daemon_owned", False))
            and bool(child_manifest.get("data_resident", False))
        ):
            return False
    return True


def _unavailable_checkpoint_csd_nodes(args: Any, tag: str) -> int:
    """Collectively count CSD coordinators that cannot serve the generation."""

    local_available = True
    if _is_racer_csd_delete_coordinator():
        try:
            local_available = _manifest_chunks_committed(
                args,
                str(tag),
                manifest_rank=_rank(),
            )
        except Exception:
            local_available = False
    return _sum_int_across_ranks(
        0 if local_available else 1,
        _current_cuda_device(),
    )


def _manifest_tag_available(args: Any, tag: str) -> bool:
    manifest = _load_tree_manifest(args, tag, rank=0)
    if manifest is None:
        return False
    if bool(manifest.get("requires_commit_marker", False)):
        generation = str(manifest.get("checkpoint_generation", ""))
        if not generation or not _checkpoint_commit_marker_available(args, tag, generation):
            return False
    return _manifest_chunks_committed(args, tag)


def _select_manifest_tag(args: Any, iteration: int | None, release: bool) -> str | None:
    root = _manifest_root(args)
    if not root.exists():
        return _select_manifest_tag_from_csd(args, iteration, release)
    if release:
        tag = _tag_for_iteration(0, True)
        if _manifest_tag_available(args, tag):
            return tag
        csd_tag = _select_manifest_tag_from_csd(args, iteration, release)
        if csd_tag is not None:
            return csd_tag
        if _manifest_path(args, tag, rank=0).exists():
            raise RuntimeError(
                f"RACER manifest for {tag!r} exists, but the checkpoint is not committed and data-resident"
            )
        return None
    if iteration is not None and int(iteration) >= 0:
        tag = _tag_for_iteration(int(iteration), False)
        if _manifest_tag_available(args, tag):
            return tag
        csd_tag = _select_manifest_tag_from_csd(args, iteration, release)
        if csd_tag is not None:
            return csd_tag
        if _manifest_path(args, tag, rank=0).exists():
            raise RuntimeError(
                f"RACER manifest for {tag!r} exists, but the checkpoint is not committed and data-resident"
            )
        return None
    candidates: list[tuple[int, str]] = []
    for path in root.glob("*.rank_00000.pt"):
        name = path.name
        if not name.startswith("megatron__iter_"):
            continue
        tag = name.split(".rank_", 1)[0].replace("__", ":", 1)
        iter_value = _iteration_from_tag(tag)
        if iter_value >= 0:
            candidates.append((iter_value, tag))
    if not candidates:
        release_tag = _tag_for_iteration(0, True)
        if _manifest_tag_available(args, release_tag):
            return release_tag
        return _select_manifest_tag_from_csd(args, iteration, release)
    for _, tag in sorted(candidates, reverse=True):
        if _manifest_tag_available(args, tag):
            return tag
    csd_tag = _select_manifest_tag_from_csd(args, iteration, release)
    if csd_tag is not None:
        return csd_tag
    if candidates:
        raise RuntimeError(
            "RACER checkpoint manifests exist, but no generation is fully committed and data-resident"
        )
    return None


def _select_manifest_tag_from_csd(args: Any, iteration: int | None, release: bool) -> str | None:
    if not _racer_csd_storage_enabled(args):
        return None
    storage = _racer_chunk_storage(args)
    if storage is None:
        raise RuntimeError("RACER CSD storage is unavailable while selecting a checkpoint")
    try:
        raw_tags = list(storage.list_tags())
    except Exception as exc:
        raise RuntimeError("failed to list RACER checkpoint metadata from CSD") from exc
    candidates: list[tuple[int, str]] = []
    requested_candidate_seen = False
    for raw_tag in raw_tags:
        tag = _checkpoint_tag_from_tree_manifest_csd_tag(str(raw_tag), rank=0)
        if tag is None:
            continue
        if release:
            if tag == _tag_for_iteration(0, True):
                requested_candidate_seen = True
                if _manifest_tag_available(args, tag):
                    return tag
            continue
        iter_value = _iteration_from_tag(tag)
        if iteration is not None and int(iteration) >= 0:
            if iter_value == int(iteration):
                requested_candidate_seen = True
                if _manifest_tag_available(args, tag):
                    return tag
            continue
        if iter_value >= 0:
            candidates.append((iter_value, tag))
    for _, tag in sorted(candidates, reverse=True):
        if _manifest_tag_available(args, tag):
            return tag
    if requested_candidate_seen or candidates:
        raise RuntimeError(
            "RACER CSD contains checkpoint metadata, but the requested generation is not fully "
            "committed and data-resident"
        )
    return None


class _RacerDistributedRuntime:
    def __init__(
        self,
        *,
        store: Any,
        process_group: Any,
        workers: list[Any],
        rank: int,
        generation_id: str,
    ) -> None:
        self.store = store
        self.process_group = process_group
        self.workers = list(workers)
        self.rank = int(rank)
        self.generation_id = str(generation_id)
        self.checkpoint_generation_seq = 0
        self.seq = 0
        self.states: dict[str, Any] = {}
        self.warmed = False
        self.store_path_warmed = False
        self.init_profile: dict[str, float] = {}


def _free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _racer_store_host() -> str:
    return os.environ.get("MASTER_ADDR", "127.0.0.1")


def _racer_spare_launch_mode(args: Any) -> str:
    mode = str(getattr(args, "racer_spare_launch_mode", "local")).lower().replace("-", "_")
    if mode not in {"local", "remote"}:
        raise ValueError(f"unsupported --racer-spare-launch-mode={mode!r}")
    return mode


def _racer_runtime_port(args: Any) -> int | None:
    value = getattr(args, "racer_runtime_port", None)
    if value is None:
        env_value = os.environ.get("RACER_RUNTIME_PORT")
        if not env_value:
            return None
        value = env_value
    port = int(value)
    if port <= 0 or port > 65535:
        raise ValueError(f"invalid RACER runtime TCPStore port: {port}")
    return port


def _racer_pg_train_ranks(args: Any) -> list[int]:
    return rank_map.racer_pg_train_ranks(_train_ranks(args))


def _racer_pg_spare_ranks(args: Any) -> list[int]:
    return rank_map.racer_pg_spare_ranks(_train_ranks(args), _spare_ranks(args))


def _racer_pg_rank_map(args: Any) -> dict[int, int]:
    return rank_map.racer_pg_rank_map(_train_ranks(args))


def _to_racer_pg_train_ranks(args: Any, ranks: list[int] | None) -> list[int]:
    return rank_map.to_racer_pg_train_ranks(ranks, _train_ranks(args))


def _to_racer_pg_replacement_mapping(args: Any, mapping: dict[int, int] | None) -> dict[int, int]:
    return rank_map.to_racer_pg_replacement_mapping(
        mapping,
        train=_train_ranks(args),
        spare=_spare_ranks(args),
    )


def _distributed_config_dict(args: Any) -> dict[str, Any]:
    return {
        "k": int(getattr(args, "racer_k", 3)),
        "m": int(getattr(args, "racer_m", 1)),
        "train_ranks": _racer_pg_train_ranks(args),
        "spare_ranks": _racer_pg_spare_ranks(args),
        "buffer_size": int(getattr(args, "racer_buffer_size", 64 * 1024 * 1024)),
        "optimize_cauchy": bool(getattr(args, "racer_optimize_cauchy", False)),
        "max_inflight_ec_groups": int(os.environ.get("RACER_MAX_INFLIGHT_EC_GROUPS", "0")),
    }


def _distributed_config(args: Any):
    racer_path = getattr(args, "racer_path", None)
    if racer_path:
        root = str(Path(racer_path).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    from racer.config import RacerConfig  # type: ignore

    return RacerConfig(**_distributed_config_dict(args))


def _spare_worker_main(
    *,
    port: int,
    store_host: str,
    world_size: int,
    spare_rank: int,
    spare_cuda_device: int,
    config_data: dict[str, Any],
    racer_path: str,
    storage_backend: str,
    storage_options: dict[str, Any] | None,
) -> None:
    if racer_path:
        root = str(Path(racer_path).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    import torch.distributed as dist
    from racer.config import RacerConfig  # type: ignore
    from racer.distributed import (  # type: ignore
        distributed_load,
        distributed_load_from_storage,
        finalize_distributed_store_many,
        prepare_distributed_store_many,
        distributed_store,
    )

    torch.cuda.set_device(spare_cuda_device)
    store = dist.TCPStore(
        str(store_host),
        int(port),
        int(world_size),
        False,
        timeout=timedelta(seconds=300),
        wait_for_workers=True,
        use_libuv=True,
    )
    pg = dist.ProcessGroupNCCL(store, int(spare_rank), int(world_size), dist.ProcessGroupNCCL.Options())
    _eager_connect_single_device(pg, torch.device("cuda", int(spare_cuda_device)))
    config = RacerConfig(**config_data)
    chunk_storage = _racer_chunk_storage_from_options(storage_backend, storage_options)
    states: dict[str, Any] = {}
    seq = 0
    try:
        while True:
            key = f"racer_cmd_{seq}"
            try:
                store.wait([key])
                raw = store.get(key)
            except Exception as exc:
                if exc.__class__.__name__ in {"DistNetworkError", "DistStoreError"}:
                    print(f"[RACER] spare worker exiting after TCPStore closed: {exc}", file=sys.stderr)
                    break
                raise
            command = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            op = command.get("op")
            if op == "shutdown":
                break
            tag = str(command["tag"])
            if op == "warmup":
                _warmup_process_group_p2p(
                    pg,
                    train_ranks=[int(v) for v in config_data.get("train_ranks", [])],
                    spare_ranks=[int(v) for v in config_data.get("spare_ranks", [])],
                    k=int(config_data.get("k", 1)),
                )
            elif op in {"store", "store_many"}:
                store_tags = [str(value) for value in command.get("tags", [tag])]
                packet_sizes_by_tag = {
                    str(tag_key): {int(rank): int(nbytes) for rank, nbytes in dict(sizes).items()}
                    for tag_key, sizes in dict(command.get("packet_sizes_by_tag", {}) or {}).items()
                }
                if op == "store_many":
                    completion_key = str(command.get("completion_key", "") or "")
                    try:
                        batch_handle = prepare_distributed_store_many(
                            config=config,
                            tags=store_tags,
                            local_packets=[None] * len(store_tags),
                            process_group=pg,
                            chunk_storage=chunk_storage,
                            packet_sizes_by_tag=packet_sizes_by_tag,
                        )
                        batch_states = finalize_distributed_store_many(batch_handle)
                        states.update({state.tag: state for state in batch_states})
                        if completion_key:
                            store.set(
                                completion_key,
                                json.dumps({"ok": True, "tags": store_tags}).encode("utf-8"),
                            )
                    except BaseException as exc:
                        if completion_key:
                            store.set(
                                completion_key,
                                json.dumps(
                                    {
                                        "ok": False,
                                        "error_type": type(exc).__name__,
                                        "message": str(exc),
                                    }
                                ).encode("utf-8"),
                            )
                        raise
                else:
                    for store_tag in store_tags:
                        states[store_tag] = distributed_store(
                            config=config,
                            local_packet=None,
                            tag=store_tag,
                            process_group=pg,
                            chunk_storage=chunk_storage,
                            packet_sizes_by_rank=packet_sizes_by_tag.get(store_tag),
                        )
            elif op in {"load", "load_many"}:
                failed = [int(v) for v in command.get("failed_train_ranks", [])]
                requested = [int(v) for v in command.get("requested_train_ranks", [])]
                replacement = {
                    int(k): int(v)
                    for k, v in dict(command.get("replacement_mapping", {})).items()
                }
                load_tags = [str(value) for value in command.get("tags", [tag])]
                for load_tag in load_tags:
                    if load_tag in states:
                        distributed_load(
                            state=states[load_tag],
                            failed_train_ranks=failed,
                            requested_train_ranks=requested,
                            replacement_mapping=replacement,
                            process_group=pg,
                        )
                    elif chunk_storage is not None:
                        distributed_load_from_storage(
                            config=config,
                            tag=load_tag,
                            chunk_storage=chunk_storage,
                            failed_train_ranks=failed,
                            requested_train_ranks=requested,
                            replacement_mapping=replacement,
                            process_group=pg,
                        )
                    else:
                        raise KeyError(f"RACER distributed state {load_tag!r} is not resident")
            elif op == "delete_many":
                for delete_tag in command.get("tags", []):
                    delete_tag = str(delete_tag)
                    states.pop(delete_tag, None)
                    if chunk_storage is not None:
                        chunk_storage.delete(delete_tag)
            else:
                raise ValueError(f"unknown RACER spare worker command: {op!r}")
            seq += 1
    finally:
        try:
            pg.shutdown()
        except Exception:
            pass


def _shutdown_distributed_runtime() -> None:
    runtime = _SESSION.distributed_runtime
    if runtime is None:
        return
    _SESSION.distributed_runtime = None
    try:
        if runtime.rank == 0:
            key = f"racer_cmd_{runtime.seq}"
            runtime.store.set(key, json.dumps({"op": "shutdown"}).encode("utf-8"))
            runtime.seq += 1
            for worker in runtime.workers:
                worker.join(timeout=2.0)
    except Exception:
        pass
    shutdown_pg = str(os.environ.get("RACER_DISTRIBUTED_RUNTIME_SHUTDOWN_PG", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if shutdown_pg:
        try:
            runtime.process_group.shutdown()
        except Exception:
            pass
    runtime.workers.clear()
    runtime.process_group = None
    runtime.store = None


def _distributed_runtime(args: Any) -> _RacerDistributedRuntime:
    if _SESSION.distributed_runtime is not None:
        return _SESSION.distributed_runtime
    if not _distributed_initialized():
        raise RuntimeError("RACER distributed memory checkpointing requires Megatron distributed initialization")

    runtime_start = time.perf_counter()
    init_profile: dict[str, float] = {}
    rank = _rank()
    device = _current_cuda_device()
    train_ranks = _train_ranks(args)
    spare_ranks = _spare_ranks(args)
    if rank not in train_ranks:
        raise RuntimeError(f"Megatron rank {rank} is not listed in --racer-train-ranks={train_ranks}")
    if not spare_ranks:
        raise RuntimeError("--racer-spare-ranks must contain at least one visible CUDA device")

    rank_map = _racer_pg_rank_map(args)
    racer_rank = int(rank_map[int(rank)])
    racer_spare_ranks = _racer_pg_spare_ranks(args)
    world_size = len(train_ranks) + len(spare_ranks)
    coordinator_rank = int(train_ranks[0])
    spare_launch_mode = _racer_spare_launch_mode(args)
    runtime_port = _racer_runtime_port(args)
    if spare_launch_mode == "remote" and runtime_port is None:
        raise RuntimeError("--racer-runtime-port is required when --racer-spare-launch-mode=remote")
    if spare_launch_mode == "local":
        for spare_cuda_device in spare_ranks:
            if torch.cuda.device_count() <= int(spare_cuda_device):
                raise RuntimeError(
                    f"RACER distributed spare worker needs visible CUDA device {int(spare_cuda_device)}, "
                    f"but torch sees {torch.cuda.device_count()} devices"
                )

    port_tensor = torch.zeros(1, dtype=torch.long, device=device)
    store_host = _racer_store_host()
    workers: list[Any] = []
    store = None
    store_start = time.perf_counter()
    if rank == coordinator_rank:
        last_bind_error: Exception | None = None
        for _ in range(20):
            port = int(runtime_port) if runtime_port is not None else _free_tcp_port()
            try:
                store = torch.distributed.TCPStore(
                    store_host,
                    port,
                    world_size,
                    True,
                    timeout=timedelta(seconds=300),
                    wait_for_workers=False,
                    use_libuv=True,
                )
                port_tensor[0] = int(port)
                break
            except Exception as exc:
                message = str(exc)
                if runtime_port is not None or (
                    "EADDRINUSE" not in message and "address already in use" not in message
                ):
                    raise
                last_bind_error = exc
        if store is None:
            raise RuntimeError("failed to bind RACER distributed TCPStore on a free local port") from last_bind_error
        init_profile["racer_runtime_tcpstore_create_ms"] = (time.perf_counter() - store_start) * 1000.0
        spawn_start = time.perf_counter()
        if spare_launch_mode == "local":
            ctx = mp.get_context("spawn")
            for spare_rank, spare_cuda_device in zip(racer_spare_ranks, spare_ranks):
                worker = ctx.Process(
                    target=_spare_worker_main,
                    kwargs={
                        "port": int(port_tensor.item()),
                        "store_host": store_host,
                        "world_size": int(world_size),
                        "spare_rank": int(spare_rank),
                        "spare_cuda_device": int(spare_cuda_device),
                        "config_data": _distributed_config_dict(args),
                        "racer_path": str(getattr(args, "racer_path", "")),
                        "storage_backend": _racer_storage_backend(args),
                        "storage_options": (
                            _racer_csd_storage_options(args) if _racer_csd_storage_enabled(args) else None
                        ),
                    },
                    daemon=True,
                )
                worker.start()
                workers.append(worker)
        init_profile["racer_runtime_worker_spawn_ms"] = (time.perf_counter() - spawn_start) * 1000.0
        init_profile["racer_runtime_remote_spare"] = 1.0 if spare_launch_mode == "remote" else 0.0
    torch.distributed.broadcast(port_tensor, src=coordinator_rank)
    port = int(port_tensor.item())
    if rank != coordinator_rank:
        client_store_start = time.perf_counter()
        store = torch.distributed.TCPStore(
            store_host,
            port,
            world_size,
            False,
            timeout=timedelta(seconds=300),
            wait_for_workers=True,
            use_libuv=True,
        )
        init_profile["racer_runtime_tcpstore_connect_ms"] = (time.perf_counter() - client_store_start) * 1000.0
    assert store is not None
    generation_key = "racer_checkpoint_generation_id"
    if rank == coordinator_rank:
        store.set(
            generation_key,
            f"{time.time_ns():x}-{os.getpid():x}".encode("utf-8"),
        )
    raw_generation = store.get(generation_key)
    generation_id = (
        raw_generation.decode("utf-8")
        if isinstance(raw_generation, bytes)
        else str(raw_generation)
    )
    pg_start = time.perf_counter()
    pg = torch.distributed.ProcessGroupNCCL(store, racer_rank, world_size, torch.distributed.ProcessGroupNCCL.Options())
    init_profile["racer_runtime_pg_create_ms"] = (time.perf_counter() - pg_start) * 1000.0
    eager_start = time.perf_counter()
    _eager_connect_single_device(pg, device)
    init_profile["racer_runtime_eager_connect_ms"] = (time.perf_counter() - eager_start) * 1000.0
    _SESSION.distributed_runtime = _RacerDistributedRuntime(
        store=store,
        process_group=pg,
        workers=workers,
        rank=racer_rank,
        generation_id=generation_id,
    )
    atexit.register(_shutdown_distributed_runtime)
    warmup_enabled = os.environ.get("RACER_DISTRIBUTED_RUNTIME_WARMUP", "1").lower() in {"1", "true", "yes", "on"}
    if warmup_enabled:
        warmup_start = time.perf_counter()
        _warmup_distributed_runtime(args)
        init_profile["racer_runtime_warmup_ms"] = (time.perf_counter() - warmup_start) * 1000.0
    else:
        _SESSION.distributed_runtime.warmed = True
        init_profile["racer_runtime_warmup_ms"] = 0.0
    final_barrier_start = time.perf_counter()
    _barrier()
    init_profile["racer_runtime_final_barrier_ms"] = (time.perf_counter() - final_barrier_start) * 1000.0
    init_profile["racer_runtime_init_ms"] = (time.perf_counter() - runtime_start) * 1000.0
    _SESSION.distributed_runtime.init_profile = init_profile
    return _SESSION.distributed_runtime


def _eager_connect_single_device(process_group: Any, device: torch.device) -> None:
    try:
        process_group.eager_connect_single_device(device)
    except AttributeError:
        return
    except RuntimeError:
        return


def _warmup_p2p_pairs(*, train_ranks: list[int], spare_ranks: list[int], k: int) -> list[tuple[int, int]]:
    if not spare_ranks:
        return []
    pairs: set[tuple[int, int]] = set()
    for spare_rank in [int(rank) for rank in spare_ranks]:
        for train_rank in train_ranks:
            train_rank = int(train_rank)
            pairs.add((train_rank, spare_rank))
            pairs.add((spare_rank, train_rank))
    k = max(1, int(k))
    for idx, train_rank in enumerate([int(rank) for rank in train_ranks]):
        owner = int(train_ranks[idx % k])
        if owner != train_rank:
            pairs.add((train_rank, owner))
            pairs.add((owner, train_rank))
    return sorted(pairs)


def _warmup_process_group_p2p(
    process_group: Any,
    *,
    train_ranks: list[int],
    spare_ranks: list[int],
    k: int,
) -> None:
    rank = int(process_group.rank())
    device = _current_cuda_device()
    send_buf = torch.zeros(1, dtype=torch.uint8, device=device)
    recv_buf = torch.empty(1, dtype=torch.uint8, device=device)
    for src, dst in _warmup_p2p_pairs(train_ranks=train_ranks, spare_ranks=spare_ranks, k=k):
        if rank == src:
            process_group.send([send_buf], int(dst), 0).wait()
        elif rank == dst:
            process_group.recv([recv_buf], int(src), 0).wait()
        process_group.barrier().wait()
    torch.cuda.synchronize(device)


def _warmup_distributed_runtime(args: Any) -> None:
    runtime = _SESSION.distributed_runtime
    if runtime is None or runtime.warmed:
        return
    _send_spare_command(args, op="warmup", tag="__warmup__")
    _warmup_process_group_p2p(
        runtime.process_group,
        train_ranks=_racer_pg_train_ranks(args),
        spare_ranks=_racer_pg_spare_ranks(args),
        k=int(getattr(args, "racer_k", 1)),
    )
    runtime.warmed = True


def _warmup_distributed_store_path(args: Any, device: torch.device) -> dict[str, float]:
    """Exercise the real batched runtime/CSD path before training starts."""

    runtime = _distributed_runtime(args)
    if runtime.store_path_warmed:
        return {"store_path_warmup_ms": 0.0}

    from racer.distributed import (  # type: ignore
        finalize_distributed_store_many,
        prepare_distributed_store_many,
    )

    started_at = time.perf_counter()
    warmup_nbytes = max(
        1,
        int(os.environ.get("RACER_DISTRIBUTED_STORE_WARMUP_BYTES", "1")),
    )
    tag = f"__racer_store_path_warmup__:{runtime.generation_id}"
    packet_sizes = {
        int(rank): warmup_nbytes for rank in _racer_pg_train_ranks(args)
    }
    runtime, completion_key = _send_spare_store_many(
        args,
        tags=[tag],
        packet_sizes_by_tag={tag: packet_sizes},
    )
    chunk_storage = _racer_chunk_storage(args)
    packet = torch.zeros(warmup_nbytes, dtype=torch.uint8, device=device)
    launch_stream = _racer_launch_stream(device)
    with torch.cuda.stream(launch_stream):
        handle = prepare_distributed_store_many(
            config=_distributed_config(args),
            tags=[tag],
            local_packets=[packet],
            process_group=runtime.process_group,
            chunk_storage=chunk_storage,
            packet_sizes_by_tag={tag: packet_sizes},
        )
    states = finalize_distributed_store_many(handle)
    if runtime.rank == 0:
        runtime.store.wait([completion_key])
        raw_status = runtime.store.get(completion_key)
        status = json.loads(
            raw_status.decode("utf-8")
            if isinstance(raw_status, bytes)
            else str(raw_status)
        )
        if not bool(status.get("ok", False)):
            raise RuntimeError(
                "RACER remote spare failed during distributed store warmup: "
                f"{status.get('error_type', 'RuntimeError')}: "
                f"{status.get('message', 'unknown error')}"
            )
    _barrier()
    for state in states:
        runtime.states.pop(str(state.tag), None)
    _send_spare_delete_many(args, [tag])
    if _is_racer_csd_delete_coordinator():
        chunk_storage.delete(tag)
    _barrier()
    runtime.store_path_warmed = True
    return {
        "store_path_warmup_ms": (time.perf_counter() - started_at) * 1000.0,
        "store_path_warmup_bytes": float(warmup_nbytes),
    }


def _prewarm_distributed_runtime_and_staging(args: Any) -> dict[str, float]:
    if not (racer_distributed_store_enabled(args) and _distributed_initialized()):
        return {}
    device = _current_cuda_device()
    runtime_start = time.perf_counter()
    _distributed_runtime(args)
    runtime_ms_local = (time.perf_counter() - runtime_start) * 1000.0
    staging_ms_local = 0.0
    staging_slabs = 0.0
    payload_pool_ms_local = 0.0
    payload_pool_chunks = 0.0
    payload_pool_reserved = 0.0
    prewarm_payload_chunks = int(
        os.environ.get(
            "RACER_PAYLOAD_POOL_PREWARM_CHUNKS",
            str(int(getattr(args, "racer_payload_pool_prewarm_chunks", 0) or 0)),
        )
    )
    if prewarm_payload_chunks > 0:
        payload_start = time.perf_counter()
        stats = _payload_buffer_pool().prewarm(
            chunk_size=int(getattr(args, "racer_buffer_size", 64 * 1024 * 1024)),
            chunk_count=prewarm_payload_chunks,
            device=device,
        )
        payload_pool_ms_local = (time.perf_counter() - payload_start) * 1000.0
        payload_pool_chunks = float(stats.get("payload_pool_buffer_count", prewarm_payload_chunks))
        payload_pool_reserved = float(stats.get("payload_pool_reserved_nbytes", 0.0))
        payload_cuda_slots = float(stats.get("payload_cuda_slot_count", 0.0))
    else:
        payload_cuda_slots = 0.0
    if _racer_csd_storage_enabled(args):
        from racer.csd import prewarm_cuda_ipc_staging  # type: ignore

        default_slabs = max(2, int(getattr(args, "racer_m", 1)) + 1)
        slab_count = int(os.environ.get("RACER_CSD_STAGING_PREWARM_SLABS", str(default_slabs)))
        slab_bytes = int(os.environ.get("RACER_CSD_STAGING_PREWARM_SLAB_BYTES", str(int(getattr(args, "racer_buffer_size", 64 * 1024 * 1024)))))
        staging_start = time.perf_counter()
        stats = prewarm_cuda_ipc_staging(
            device=int(device.index or 0),
            slab_bytes=slab_bytes,
            count=slab_count,
        )
        staging_ms_local = (time.perf_counter() - staging_start) * 1000.0
        staging_slabs = float(stats.get("slabs_in_pool", slab_count))
    store_path_profile = _warmup_distributed_store_path(args, device)
    store_path_ms_local = float(store_path_profile.get("store_path_warmup_ms", 0.0))
    runtime_ms_max = _max_float_across_ranks(runtime_ms_local, device)
    staging_ms_max = _max_float_across_ranks(staging_ms_local, device)
    staging_slabs_max = _max_float_across_ranks(staging_slabs, device)
    payload_pool_ms_max = _max_float_across_ranks(payload_pool_ms_local, device)
    payload_pool_chunks_max = _max_float_across_ranks(payload_pool_chunks, device)
    payload_pool_reserved_max = _max_float_across_ranks(payload_pool_reserved, device)
    payload_cuda_slots_max = _max_float_across_ranks(payload_cuda_slots, device)
    store_path_ms_max = _max_float_across_ranks(store_path_ms_local, device)
    return {
        "runtime_ms": runtime_ms_max,
        "staging_ms": staging_ms_max,
        "payload_pool_ms": payload_pool_ms_max,
        "total_ms": (
            runtime_ms_max + staging_ms_max + payload_pool_ms_max + store_path_ms_max
        ),
        "staging_slabs": staging_slabs_max,
        "payload_pool_chunks": payload_pool_chunks_max,
        "payload_pool_reserved_nbytes": payload_pool_reserved_max,
        "payload_cuda_slots": payload_cuda_slots_max,
        "store_path_ms": store_path_ms_max,
    }


def _gather_packet_sizes_by_chunk(
    args: Any,
    *,
    chunk_tags: list[str],
    local_valid_nbytes: list[int],
    device: torch.device,
) -> dict[str, dict[int, int]]:
    if not chunk_tags:
        return {}
    local = torch.zeros(len(chunk_tags), dtype=torch.long, device=device)
    for index, nbytes in enumerate(local_valid_nbytes[: len(chunk_tags)]):
        local[index] = int(nbytes)
    if _distributed_initialized():
        gathered = [torch.zeros_like(local) for _ in range(_world_size())]
        torch.distributed.all_gather(gathered, local)
    else:
        gathered = [local]
    physical_to_racer = _racer_pg_rank_map(args)
    missing_physical = [rank for rank in _train_ranks(args) if int(rank) >= len(gathered)]
    if missing_physical:
        raise RuntimeError(f"cannot gather RACER packet sizes for missing train ranks {missing_physical}")
    gathered_cpu = [tensor.cpu().tolist() for tensor in gathered]
    packet_sizes_by_tag: dict[str, dict[int, int]] = {}
    for chunk_index, chunk_tag in enumerate(chunk_tags):
        sizes: dict[int, int] = {}
        for physical_rank, values in enumerate(gathered_cpu):
            racer_rank = physical_to_racer.get(int(physical_rank))
            if racer_rank is None:
                continue
            sizes[int(racer_rank)] = int(values[chunk_index])
        missing_racer = [rank for rank in _racer_pg_train_ranks(args) if int(rank) not in sizes]
        if missing_racer:
            raise RuntimeError(f"packet size table for {chunk_tag} is missing RACER ranks {missing_racer}")
        packet_sizes_by_tag[str(chunk_tag)] = sizes
    return packet_sizes_by_tag


def _send_spare_command(
    args: Any,
    *,
    op: str,
    tag: str,
    failed_train_ranks: list[int] | None = None,
    requested_train_ranks: list[int] | None = None,
    replacement_mapping: dict[int, int] | None = None,
) -> _RacerDistributedRuntime:
    runtime = _distributed_runtime(args)
    seq = runtime.seq
    if runtime.rank == 0:
        command = {
            "op": op,
            "tag": tag,
            "failed_train_ranks": _to_racer_pg_train_ranks(args, failed_train_ranks),
            "requested_train_ranks": _to_racer_pg_train_ranks(args, requested_train_ranks),
            "replacement_mapping": _to_racer_pg_replacement_mapping(args, replacement_mapping),
        }
        runtime.store.set(f"racer_cmd_{seq}", json.dumps(command).encode("utf-8"))
    _barrier()
    runtime.seq += 1
    return runtime


def _send_spare_store_many(
    args: Any,
    *,
    tags: list[str],
    packet_sizes_by_tag: dict[str, dict[int, int]] | None = None,
) -> tuple[_RacerDistributedRuntime, str]:
    runtime = _distributed_runtime(args)
    seq = runtime.seq
    completion_key = f"racer_store_many_complete_{seq:08d}"
    if runtime.rank == 0:
        command = {
            "op": "store_many",
            "tag": "__store_many__",
            "tags": [str(tag) for tag in tags],
            "packet_sizes_by_tag": {
                str(tag): {str(rank): int(nbytes) for rank, nbytes in dict(sizes).items()}
                for tag, sizes in dict(packet_sizes_by_tag or {}).items()
            },
            "completion_key": completion_key,
            "failed_train_ranks": [],
            "requested_train_ranks": [],
            "replacement_mapping": {},
        }
        runtime.store.set(f"racer_cmd_{seq}", json.dumps(command).encode("utf-8"))
    runtime.seq += 1
    return runtime, completion_key


def _send_spare_load_many(
    args: Any,
    *,
    tags: list[str],
    failed_train_ranks: list[int] | None = None,
    requested_train_ranks: list[int] | None = None,
    replacement_mapping: dict[int, int] | None = None,
) -> _RacerDistributedRuntime:
    runtime = _distributed_runtime(args)
    seq = runtime.seq
    if runtime.rank == 0:
        command = {
            "op": "load_many",
            "tag": "__load_many__",
            "tags": [str(tag) for tag in tags],
            "failed_train_ranks": _to_racer_pg_train_ranks(args, failed_train_ranks),
            "requested_train_ranks": _to_racer_pg_train_ranks(args, requested_train_ranks),
            "replacement_mapping": _to_racer_pg_replacement_mapping(args, replacement_mapping),
        }
        runtime.store.set(f"racer_cmd_{seq}", json.dumps(command).encode("utf-8"))
    runtime.seq += 1
    return runtime


def _send_spare_delete_many(args: Any, tags: list[str]) -> None:
    if not tags:
        return
    runtime = _distributed_runtime(args)
    seq = runtime.seq
    if runtime.rank == 0:
        command = {"op": "delete_many", "tag": "__delete_many__", "tags": list(tags)}
        runtime.store.set(f"racer_cmd_{seq}", json.dumps(command).encode("utf-8"))
    runtime.seq += 1


def _prune_old_checkpoints(args: Any) -> None:
    retain = max(1, int(getattr(args, "racer_retain_checkpoints", 1)))
    if len(_SESSION.tags_by_iteration) <= retain:
        return
    keep_iterations = sorted(_SESSION.tags_by_iteration)[-retain:]
    keep_tags = {_SESSION.tags_by_iteration[item] for item in keep_iterations}
    stale_tags = [tag for tag in list(_SESSION.tree_checkpoints) if tag not in keep_tags]
    if not stale_tags:
        return

    chunk_tags: list[str] = []
    tree_manifest_csd_tags: list[str] = []
    for stale_tag in stale_tags:
        tree = _SESSION.tree_checkpoints.pop(stale_tag, None)
        max_chunk_count = (
            int(tree.get("max_chunk_count", tree.get("max_leaf_count", 0)))
            if isinstance(tree, dict)
            else 0
        )
        chunk_tags.extend(_chunk_tag(stale_tag, chunk_index) for chunk_index in range(max_chunk_count))
        manifest_rank = int(tree.get("source_rank", _rank())) if isinstance(tree, dict) else _rank()
        tree_manifest_csd_tags.append(_tree_manifest_csd_tag(stale_tag, manifest_rank))
        _SESSION.reports.pop(stale_tag, None)
        try:
            _manifest_path(args, stale_tag).unlink(missing_ok=True)
        except OSError:
            pass
        if _rank() == 0:
            try:
                _checkpoint_commit_marker_path(args, stale_tag).unlink(missing_ok=True)
            except OSError:
                pass
    for stale_iteration, stale_tag in list(_SESSION.tags_by_iteration.items()):
        if stale_tag not in keep_tags:
            _SESSION.tags_by_iteration.pop(stale_iteration, None)

    runtime = _SESSION.distributed_runtime
    if runtime is not None:
        for chunk_tag in chunk_tags:
            runtime.states.pop(chunk_tag, None)
    csd_storage = _racer_chunk_storage(args) if _racer_csd_storage_enabled(args) else None
    if racer_distributed_store_enabled(args):
        if chunk_tags:
            _send_spare_delete_many(args, chunk_tags)
        if csd_storage is not None and _is_racer_csd_delete_coordinator():
            for chunk_tag in chunk_tags:
                try:
                    csd_storage.delete(chunk_tag)
                except Exception as exc:
                    print(
                        f"WARNING: rank {_rank()} could not prune RACER CSD chunk tag "
                        f"{chunk_tag!r}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
            if os.environ.get("RACER_EGM_TRIM_FREE_SEGMENTS", "0").lower() in {
                "1", "true", "yes", "on",
            }:
                trim_report = dict(csd_storage.trim())
                print(
                    f"RACER EGM post-prune trim: rank={_rank()}, "
                    f"released_segments={int(trim_report.get('released_segments', 0))}, "
                    f"released_bytes={int(trim_report.get('released_bytes', 0))}, "
                    f"replenished_segments={int(trim_report.get('replenished_segments', 0))}",
                    flush=True,
                )
    else:
        ctx = _SESSION.local_racer_context
        if ctx is not None:
            for chunk_tag in chunk_tags:
                try:
                    ctx.delete(chunk_tag)
                except AttributeError:
                    try:
                        ctx.storage.delete(chunk_tag)
                    except Exception:
                        pass
                except Exception:
                    pass
    if csd_storage is not None:
        # Every rank owns one tree-manifest tag. This removes the correct local
        # tag with both a shared CSD and the per-node CSD deployment.
        for tree_manifest_csd_tag in tree_manifest_csd_tags:
            try:
                csd_storage.delete(tree_manifest_csd_tag)
            except Exception as exc:
                print(
                    f"WARNING: rank {_rank()} could not prune RACER CSD tree manifest "
                    f"{tree_manifest_csd_tag!r}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if _is_racer_csd_delete_coordinator():
            for stale_tag in stale_tags:
                marker_tag = _checkpoint_commit_marker_csd_tag(stale_tag)
                try:
                    csd_storage.delete(marker_tag)
                except Exception as exc:
                    print(
                        f"WARNING: rank {_rank()} could not prune RACER CSD commit marker "
                        f"{marker_tag!r}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
    if os.environ.get("RACER_AGGRESSIVE_CUDA_CLEANUP", "0") == "1" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    _print_rank0(
        "RACER checkpoint retention pruned: "
        f"removed_checkpoints={len(stale_tags)}, removed_chunk_states={len(chunk_tags)}, retain={retain}"
    )



def _chunk_tag(tag: str, chunk_index: int) -> str:
    return f"{tag}::chunk::{int(chunk_index):08d}"


def _accumulate_numeric_profile(target: dict[str, float], profile: dict[str, Any] | None) -> None:
    for key, value in dict(profile or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            target[str(key)] = target.get(str(key), 0.0) + float(value)


_STORE_PROFILE_KEYS = (
    "setup_ms",
    "sizing_ms",
    "data_rows_ms",
    "parity_ms",
    "storage_ms",
    "final_barrier_ms",
    "group_transfer_barrier_ms",
    "group_storage_barrier_ms",
    "manifest_ms",
    "total_ms",
    "store_batch_prepare_ms",
    "store_batch_finalize_ms",
    "store_batch_total_ms",
    "store_batch_tag_count",
    "data_rows_bytes_sent",
    "parity_bytes_sent",
    "local_storage_nbytes",
    "local_storage_chunk_count",
    "local_chunks_released_nbytes",
    "local_chunks_released_count",
    "csd_daemon_allocate_ms",
    "csd_daemon_memcpy_enqueue_wall_ms",
    "csd_daemon_memcpy_ms_cuda_event",
    "csd_daemon_memcpy_ms_wall",
    "csd_daemon_event_wait_ms",
    "csd_daemon_ipc_open_us",
    "csd_daemon_ipc_event_open_us",
    "csd_daemon_set_device_ms",
    "csd_daemon_event_create_ms",
    "csd_daemon_enqueue_api_ms",
    "csd_checksum_ms",
    "csd_sqlite_ms",
    "csd_daemon_allocate_source_pool_bump_count",
    "csd_daemon_allocate_source_pool_reuse_count",
    "storage_begin_ms",
    "storage_begin_barrier_ms",
    "storage_enqueue_ms",
    "storage_enqueue_barrier_ms",
    "storage_wait_ms",
    "storage_manifest_put_ms",
    "storage_commit_pre_barrier_ms",
    "storage_commit_ms",
    "storage_commit_post_barrier_ms",
    "storage_commit_retry_count",
    "storage_commit_wait_ms",
    "csd_client_put_contiguous_ms",
    "csd_client_put_export_ms",
    "csd_client_put_rpc_ms",
    "csd_client_put_total_ms",
    "csd_staging_alloc_us",
    "csd_staging_copy_enqueue_us",
    "csd_daemon_put_submit_ms",
    "csd_daemon_put_dispatch_ms",
    "csd_daemon_put_backend_write_ms",
    "csd_daemon_put_total_ms",
)


_LOAD_PROFILE_KEYS = (
    "load_manifest_ms",
    "load_setup_ms",
    "load_read_enqueue_ms",
    "load_read_wait_ms",
    "load_final_barrier_ms",
    "load_total_ms",
    "load_read_nbytes",
    "load_read_chunk_count",
    "load_route_pre_barrier_ms",
    "load_route_requested_ms",
    "load_route_decode_ms",
    "load_route_final_barrier_ms",
    "load_route_total_ms",
    "load_decode_request_count",
    "csd_get_daemon_event_wait_ms",
    "csd_get_daemon_ipc_open_us",
    "csd_get_daemon_memcpy_enqueue_wall_ms",
    "csd_get_daemon_memcpy_ms_cuda_event",
    "csd_get_daemon_memcpy_ms_wall",
    "csd_get_sqlite_ms",
    "racer_runtime_tcpstore_create_ms",
    "racer_runtime_tcpstore_connect_ms",
    "racer_runtime_worker_spawn_ms",
    "racer_runtime_pg_create_ms",
    "racer_runtime_eager_connect_ms",
    "racer_runtime_warmup_ms",
    "racer_runtime_final_barrier_ms",
    "racer_runtime_init_ms",
)


_extract_tensor_tree = tensor_tree.extract_tensor_tree
_tensor_to_cuda_bytes = tensor_tree.tensor_to_cuda_bytes
_pack_tensor_chunks = tensor_tree.pack_tensor_chunks
_pack_tensor_chunks_to_pinned = tensor_tree.pack_tensor_chunks_to_pinned
_tensor_from_cuda_bytes = tensor_tree.tensor_from_cuda_bytes
_decode_tensor_tree = tensor_tree.decode_tensor_tree


def _prepare_distributed_store_tensor_tree(
    args: Any,
    *,
    tag: str,
    iteration: int,
    release: bool,
    ckpt_type: str,
    state_for_save: dict[str, Any],
    device: torch.device,
) -> _PreparedDistributedSave:
    if os.environ.get("RACER_PRUNE_BEFORE_SAVE", "0").lower() in {"1", "true", "yes", "on"}:
        if not racer_distributed_store_enabled(args) or _racer_storage_backend(args) != "csd_egm":
            raise RuntimeError("RACER_PRUNE_BEFORE_SAVE is restricted to distributed csd_egm")
        previous_tag = _SESSION.latest_tag
        if previous_tag is not None and str(previous_tag) != str(tag):
            payload_pool = _payload_buffer_pool()
            released_host_bytes = sum(
                int(buffer.numel()) * int(buffer.element_size())
                for buffer in payload_pool.buffers
            )
            payload_pool.buffers.clear()
            payload_pool.chunk_size = 0
            payload_pool.release_cuda_load_slots(device=device)
            import gc
            gc.collect()
            host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
            if callable(host_empty_cache):
                host_empty_cache()
            _print_rank0(
                "RACER pre-save payload pool release: "
                f"bytes={released_host_bytes}, upcoming={tag}"
            )
            _SESSION.tags_by_iteration[int(iteration)] = str(tag)
            try:
                _prune_old_checkpoints(args)
            finally:
                if _SESSION.tags_by_iteration.get(int(iteration)) == str(tag):
                    _SESSION.tags_by_iteration.pop(int(iteration), None)
            _SESSION.latest_tag = None
            _barrier()
            _print_rank0(
                "RACER EGM pre-save rolling prune complete: "
                f"removed={previous_tag}, upcoming={tag}"
            )
    metadata_start = time.perf_counter()
    skeleton, metas, tensors = _extract_tensor_tree(state_for_save)
    metadata_ms = (time.perf_counter() - metadata_start) * 1000.0
    local_leaf_count = len(tensors)
    max_leaf_count = _max_int_across_ranks(local_leaf_count, device)
    local_bytes = sum(int(meta.get("nbytes", 0)) for meta in metas)
    payload_bytes_total = _sum_int_across_ranks(local_bytes, device)
    chunk_size = int(getattr(args, "racer_buffer_size", 64 * 1024 * 1024))
    expected_local_chunk_count = (int(local_bytes) + chunk_size - 1) // chunk_size if int(local_bytes) > 0 else 0
    payload_pack_start = time.perf_counter()
    payload_chunks = _pack_tensor_chunks_to_pinned(
        tensors,
        device,
        chunk_size,
        host_buffer_pool=_payload_buffer_pool(),
        expected_chunk_count=expected_local_chunk_count,
    )
    payload_pack_ms_local = (time.perf_counter() - payload_pack_start) * 1000.0
    rank = _rank()
    chunks = payload_chunks.chunks
    local_chunk_count = len(payload_chunks)
    max_chunk_count = _max_int_across_ranks(local_chunk_count, device)
    payload_checksum_mode = _debug_payload_checksum_mode()
    payload_checksums: list[str] = []
    payload_checksum_ms_local = 0.0
    if payload_checksum_mode:
        payload_checksums, payload_checksum_ms_local = _debug_payload_checksums(payload_chunks, mode=payload_checksum_mode)
        print(
            "RACER payload checksum recorded: "
            f"rank={rank}, tag={tag}, mode={payload_checksum_mode}, "
            f"chunks={len(payload_checksums)}, ms={payload_checksum_ms_local:.2f}",
            flush=True,
        )
    tree = {
        "skeleton": skeleton,
        "metas": metas,
        "source_tensors": tensors,
        "chunks": chunks,
        "max_leaf_count": int(max_leaf_count),
        "max_chunk_count": int(max_chunk_count),
        "chunk_size": int(chunk_size),
        "iteration": int(iteration),
        "release": bool(release),
        "ckpt_type": str(ckpt_type),
        "requires_commit_marker": True,
    }
    if payload_checksum_mode:
        tree["payload_checksum_mode"] = payload_checksum_mode
        tree["payload_checksums"] = list(payload_checksums)
    chunk_tags = [_chunk_tag(tag, chunk_index) for chunk_index in range(int(max_chunk_count))]

    _racer_module(args)
    packet_sizes_by_tag = _gather_packet_sizes_by_chunk(
        args,
        chunk_tags=chunk_tags,
        local_valid_nbytes=list(payload_chunks.valid_nbytes),
        device=device,
    )
    command_start = time.perf_counter()
    if chunk_tags:
        runtime, spare_completion_key = _send_spare_store_many(
            args,
            tags=chunk_tags,
            packet_sizes_by_tag=packet_sizes_by_tag,
        )
    else:
        runtime = _distributed_runtime(args)
        spare_completion_key = None
    tree["checkpoint_generation"] = (
        f"{runtime.generation_id}:{runtime.checkpoint_generation_seq}:{tag}"
    )
    runtime.checkpoint_generation_seq += 1
    store_command_ms_local = (time.perf_counter() - command_start) * 1000.0
    chunk_storage = _racer_chunk_storage(args)
    store_started_at = time.perf_counter()
    from racer.distributed import prepare_distributed_store_many  # type: ignore

    launch_stream = _racer_launch_stream(device)
    with torch.cuda.stream(launch_stream):
        tensor_view_start = time.perf_counter()
        local_packets: list[torch.Tensor] = []
        for chunk_index, _chunk_tag_value in enumerate(chunk_tags):
            if chunk_index < local_chunk_count:
                local_packet = payload_chunks.cuda_payload(
                    chunk_index,
                    device,
                    slot_id=chunk_index,
                )
            else:
                local_packet = _tensor_to_cuda_bytes(None, device)
            local_packets.append(local_packet)
        tensor_view_ms_local = (time.perf_counter() - tensor_view_start) * 1000.0
        # cuda_payload() above may leave fragmented cached blocks after all
        # strongly referenced payload slots have been materialized. Release
        # only those unreferenced blocks immediately before EC allocates its
        # group buffer.
        if os.environ.get("RACER_AGGRESSIVE_CUDA_CLEANUP", "0") == "1":
            launch_stream.synchronize()
            if os.environ.get("RACER_RELEASE_PINNED_AFTER_MATERIALIZE", "0").lower() in {
                "1", "true", "yes", "on",
            }:
                released_pinned_bytes = int(payload_chunks.total_reserved_nbytes)
                payload_chunks.profile["payload_pinned_reserved_nbytes_before_release"] = float(
                    released_pinned_bytes
                )
                payload_chunks.buffers.clear()
                payload_pool = _payload_buffer_pool()
                payload_pool.buffers.clear()
                payload_pool.chunk_size = 0
                import gc
                gc.collect()
                host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
                if callable(host_empty_cache):
                    host_empty_cache()
                if runtime.rank == 0:
                    print(
                        "RACER pre-store pinned payload release: "
                        f"bytes={released_pinned_bytes}",
                        flush=True,
                    )
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                if runtime.rank == 0:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    print(
                        "RACER pre-EC CUDA cleanup: "
                        f"allocated={torch.cuda.memory_allocated(device)} "
                        f"reserved={torch.cuda.memory_reserved(device)} "
                        f"free={free_bytes} total={total_bytes}",
                        flush=True,
                    )
        racer_prepare_start = time.perf_counter()
        distributed_config = _distributed_config(args)
        distributed_store_handle = prepare_distributed_store_many(
            config=distributed_config,
            tags=chunk_tags,
            local_packets=local_packets,
            process_group=runtime.process_group,
            chunk_storage=chunk_storage,
            packet_sizes_by_tag=packet_sizes_by_tag,
        )
        if int(distributed_config.max_inflight_ec_groups) > 0:
            local_packets.clear()
            chunks_release = payload_chunks.release_cuda_load_slots()
            pool_release = _payload_buffer_pool().release_cuda_load_slots(device=device)
            payload_chunks.profile.update(
                {
                    "payload_cuda_slot_released_count": max(
                        float(chunks_release["released_cuda_load_slot_count"]),
                        float(pool_release["released_cuda_load_slot_count"]),
                    ),
                    "payload_cuda_slot_released_nbytes": max(
                        float(chunks_release["released_cuda_load_slot_nbytes"]),
                        float(pool_release["released_cuda_load_slot_nbytes"]),
                    ),
                }
            )
    racer_prepare_ms_local = (time.perf_counter() - racer_prepare_start) * 1000.0
    return _PreparedDistributedSave(
        args=args,
        tag=tag,
        iteration=int(iteration),
        release=bool(release),
        ckpt_type=str(ckpt_type),
        device=device,
        tree=tree,
        payload_chunks=payload_chunks,
        packet_sizes_by_tag=packet_sizes_by_tag,
        chunk_tags=chunk_tags,
        runtime=runtime,
        chunk_storage=chunk_storage,
        metadata_ms_local=metadata_ms,
        payload_pack_ms_local=payload_pack_ms_local,
        payload_checksum_ms_local=payload_checksum_ms_local,
        payload_checksums=list(payload_checksums),
        local_leaf_count=int(local_leaf_count),
        max_leaf_count=int(max_leaf_count),
        local_bytes=int(local_bytes),
        payload_bytes_total=int(payload_bytes_total),
        local_chunk_count=int(local_chunk_count),
        max_chunk_count=int(max_chunk_count),
        chunk_size=int(chunk_size),
        store_command_ms_local=store_command_ms_local,
        distributed_store_handle=distributed_store_handle,
        spare_completion_key=spare_completion_key,
        tensor_view_ms_local=tensor_view_ms_local,
        racer_prepare_ms_local=racer_prepare_ms_local,
        store_started_at=store_started_at,
    )


def _execute_prepared_distributed_store(prepared: _PreparedDistributedSave) -> _ExecutedDistributedSave:
    args = prepared.args
    tag = prepared.tag
    device = prepared.device
    runtime = prepared.runtime
    torch.cuda.set_device(device)
    racer_profile_local: dict[str, float] = {}
    _accumulate_numeric_profile(racer_profile_local, getattr(runtime, "init_profile", None))
    from racer.distributed import finalize_distributed_store_many  # type: ignore

    finalize_start = time.perf_counter()
    batch_states = finalize_distributed_store_many(prepared.distributed_store_handle)
    if runtime.rank == 0 and prepared.spare_completion_key:
        runtime.store.wait([prepared.spare_completion_key])
        raw_status = runtime.store.get(prepared.spare_completion_key)
        status = json.loads(
            raw_status.decode("utf-8")
            if isinstance(raw_status, bytes)
            else str(raw_status)
        )
        if not bool(status.get("ok", False)):
            raise RuntimeError(
                "RACER remote spare failed while finalizing async checkpoint: "
                f"{status.get('error_type', 'RuntimeError')}: "
                f"{status.get('message', 'unknown error')}"
            )
    racer_finalize_ms_local = (time.perf_counter() - finalize_start) * 1000.0
    racer_calls_ms_local = (
        float(prepared.racer_prepare_ms_local) + racer_finalize_ms_local
    )
    chunk_store_ms_values: list[float] = []
    for state in batch_states:
        chunk_tag = str(state.tag)
        runtime.states[chunk_tag] = state
        state_profile = dict(getattr(state, "profile", None) or {})
        chunk_store_ms_values.append(float(state_profile.get("total_ms", 0.0)))
        _accumulate_numeric_profile(racer_profile_local, state_profile)
    store_wall_ms_local = (time.perf_counter() - prepared.store_started_at) * 1000.0
    return _ExecutedDistributedSave(
        prepared=prepared,
        tensor_view_ms_local=prepared.tensor_view_ms_local,
        racer_calls_ms_local=racer_calls_ms_local,
        racer_profile_local=racer_profile_local,
        chunk_store_ms_values=chunk_store_ms_values,
        store_wall_ms_local=store_wall_ms_local,
    )


def _finalize_executed_distributed_store(executed: _ExecutedDistributedSave) -> dict[str, Any]:
    prepared = executed.prepared
    args = prepared.args
    tag = prepared.tag
    iteration = prepared.iteration
    release = prepared.release
    ckpt_type = prepared.ckpt_type
    device = prepared.device
    payload_chunks = prepared.payload_chunks
    metadata_ms = prepared.metadata_ms_local
    payload_pack_ms_local = prepared.payload_pack_ms_local
    payload_checksum_ms_local = prepared.payload_checksum_ms_local
    payload_checksums = prepared.payload_checksums
    local_leaf_count = prepared.local_leaf_count
    max_leaf_count = prepared.max_leaf_count
    local_bytes = prepared.local_bytes
    payload_bytes_total = prepared.payload_bytes_total
    local_chunk_count = prepared.local_chunk_count
    max_chunk_count = prepared.max_chunk_count
    chunk_size = prepared.chunk_size
    store_command_ms_local = prepared.store_command_ms_local
    tensor_view_ms_local = executed.tensor_view_ms_local
    racer_calls_ms_local = executed.racer_calls_ms_local
    racer_profile_local = executed.racer_profile_local
    chunk_store_ms_values = executed.chunk_store_ms_values
    store_wall_ms_local = executed.store_wall_ms_local
    publish_profile_local = _publish_tree_manifest_with_commit_marker(prepared)
    checkpoint_commit_wall_ms_local = (time.perf_counter() - prepared.store_started_at) * 1000.0
    chunk_store_ms_sum_local = sum(chunk_store_ms_values)
    chunk_store_ms_max_local = max(chunk_store_ms_values, default=0.0)
    wrapper_gap_ms_local = max(
        0.0,
        store_wall_ms_local - tensor_view_ms_local - racer_calls_ms_local,
    )
    payload_profile = payload_chunks.profile
    max_values_local = {
        "metadata_ms": metadata_ms,
        "store_wall_ms": store_wall_ms_local,
        "checkpoint_commit_wall_ms": checkpoint_commit_wall_ms_local,
        **publish_profile_local,
        "store_command_ms": store_command_ms_local,
        "tensor_view_ms": tensor_view_ms_local,
        "racer_calls_ms": racer_calls_ms_local,
        "chunk_store_ms_sum": chunk_store_ms_sum_local,
        "chunk_store_ms": chunk_store_ms_max_local,
        "wrapper_gap_ms": wrapper_gap_ms_local,
        "payload_pack_ms": payload_pack_ms_local,
        "payload_checksum_save_ms": payload_checksum_ms_local,
        "payload_pack_slot_alloc_ms": float(payload_profile.get("payload_pack_slot_alloc_ms", 0.0)),
        "payload_pack_host_alloc_ms": float(payload_profile.get("payload_pack_host_alloc_ms", 0.0)),
        "payload_pack_tensor_view_ms": float(payload_profile.get("payload_pack_tensor_view_ms", 0.0)),
        "payload_pack_device_gather_enqueue_ms": float(
            payload_profile.get("payload_pack_device_gather_enqueue_ms", 0.0)
        ),
        "payload_pack_host_copy_enqueue_ms": float(
            payload_profile.get("payload_pack_host_copy_enqueue_ms", 0.0)
        ),
        "payload_pack_slot_reuse_wait_enqueue_ms": float(
            payload_profile.get("payload_pack_slot_reuse_wait_enqueue_ms", 0.0)
        ),
        "payload_pack_final_sync_ms": float(payload_profile.get("payload_pack_final_sync_ms", 0.0)),
        "payload_pack_segment_count": float(payload_profile.get("payload_pack_segment_count", 0.0)),
        "payload_pack_chunk_count": float(payload_profile.get("payload_pack_chunk_count", 0.0)),
        "payload_pool_ensure_ms": float(payload_profile.get("payload_pool_ensure_ms", 0.0)),
        "payload_pool_allocated_buffers": float(payload_profile.get("payload_pool_allocated_buffers", 0.0)),
        "payload_pool_reused_buffers": float(payload_profile.get("payload_pool_reused_buffers", 0.0)),
        "payload_pool_buffer_count": float(payload_profile.get("payload_pool_buffer_count", 0.0)),
        "payload_pool_reserved_nbytes": float(payload_profile.get("payload_pool_reserved_nbytes", 0.0)),
        "payload_cuda_slot_released_count": float(
            payload_profile.get("payload_cuda_slot_released_count", 0.0)
        ),
        "payload_cuda_slot_released_nbytes": float(
            payload_profile.get("payload_cuda_slot_released_nbytes", 0.0)
        ),
    }
    max_values_local.update(
        {f"racer_profile:{key}": racer_profile_local.get(key, 0.0) for key in _STORE_PROFILE_KEYS}
    )
    max_values = _max_floats_across_ranks(max_values_local, device)
    metadata_ms_max = max_values["metadata_ms"]
    store_wall_ms_max = max_values["store_wall_ms"]
    checkpoint_commit_wall_ms_max = max_values["checkpoint_commit_wall_ms"]
    store_command_ms_max = max_values["store_command_ms"]
    tensor_view_ms_max = max_values["tensor_view_ms"]
    racer_calls_ms_max = max_values["racer_calls_ms"]
    chunk_store_ms_sum_max = max_values["chunk_store_ms_sum"]
    chunk_store_ms_max = max_values["chunk_store_ms"]
    wrapper_gap_ms_max = max_values["wrapper_gap_ms"]
    racer_profile_max = {
        key: max_values[f"racer_profile:{key}"] for key in _STORE_PROFILE_KEYS
    }
    encoded_storage_bytes_total = _sum_int_across_ranks(
        int(racer_profile_local.get("local_storage_nbytes", 0.0)),
        device,
    )
    store_wall_seconds = store_wall_ms_max / 1000.0
    logical_store_bandwidth_gbps = (
        (float(payload_bytes_total) / 1_000_000_000.0) / store_wall_seconds
        if store_wall_seconds > 0.0
        else 0.0
    )
    encoded_store_bandwidth_gbps = (
        (float(encoded_storage_bytes_total) / 1_000_000_000.0) / store_wall_seconds
        if store_wall_seconds > 0.0
        else 0.0
    )
    checkpoint_commit_wall_seconds = checkpoint_commit_wall_ms_max / 1000.0
    logical_commit_bandwidth_gbps = (
        (float(payload_bytes_total) / 1_000_000_000.0) / checkpoint_commit_wall_seconds
        if checkpoint_commit_wall_seconds > 0.0
        else 0.0
    )
    encoded_commit_bandwidth_gbps = (
        (float(encoded_storage_bytes_total) / 1_000_000_000.0) / checkpoint_commit_wall_seconds
        if checkpoint_commit_wall_seconds > 0.0
        else 0.0
    )
    copy_event_ms_max = float(racer_profile_max.get("csd_daemon_memcpy_ms_cuda_event", 0.0))
    egm_copy_bandwidth_gbps = (
        (float(encoded_storage_bytes_total) / 1_000_000_000.0) / (copy_event_ms_max / 1000.0)
        if copy_event_ms_max > 0.0
        else 0.0
    )
    report = {
        "tag": tag,
        "iteration": int(iteration),
        "release": bool(release),
        "ckpt_type": str(ckpt_type),
        "tensor_tree_metadata_ms_max": metadata_ms_max,
        "payload_pack_ms_max": max_values["payload_pack_ms"],
        "payload_checksum_save_ms_max": max_values["payload_checksum_save_ms"],
        "payload_checksum_chunks_local": int(len(payload_checksums)),
        "payload_pack_slot_alloc_ms_max": max_values["payload_pack_slot_alloc_ms"],
        "payload_pack_host_alloc_ms_max": max_values["payload_pack_host_alloc_ms"],
        "payload_pack_tensor_view_ms_max": max_values["payload_pack_tensor_view_ms"],
        "payload_pack_device_gather_enqueue_ms_max": max_values["payload_pack_device_gather_enqueue_ms"],
        "payload_pack_host_copy_enqueue_ms_max": max_values["payload_pack_host_copy_enqueue_ms"],
        "payload_pack_slot_reuse_wait_enqueue_ms_max": max_values[
            "payload_pack_slot_reuse_wait_enqueue_ms"
        ],
        "payload_pack_final_sync_ms_max": max_values["payload_pack_final_sync_ms"],
        "payload_pack_segment_count_max": max_values["payload_pack_segment_count"],
        "payload_pack_chunk_count_max": max_values["payload_pack_chunk_count"],
        "payload_pool_ensure_ms_max": max_values["payload_pool_ensure_ms"],
        "payload_pool_allocated_buffers_max": max_values["payload_pool_allocated_buffers"],
        "payload_pool_reused_buffers_max": max_values["payload_pool_reused_buffers"],
        "payload_pool_buffer_count_max": max_values["payload_pool_buffer_count"],
        "payload_pool_reserved_nbytes_max": int(max_values["payload_pool_reserved_nbytes"]),
        "payload_pinned_reserved_bytes_local": int(payload_chunks.total_reserved_nbytes),
        "payload_pinned_valid_bytes_local": int(payload_chunks.total_valid_nbytes),
        "racer_store_command_ms_max": store_command_ms_max,
        "tensor_view_ms_max": tensor_view_ms_max,
        "racer_distributed_calls_ms_max": racer_calls_ms_max,
        "racer_distributed_total_ms_max": racer_profile_max.get("total_ms", 0.0),
        "racer_store_batch_prepare_ms_max": racer_profile_max.get("store_batch_prepare_ms", 0.0),
        "racer_store_batch_finalize_ms_max": racer_profile_max.get("store_batch_finalize_ms", 0.0),
        "racer_store_batch_total_ms_max": racer_profile_max.get("store_batch_total_ms", 0.0),
        "racer_store_batch_tag_count_max": int(racer_profile_max.get("store_batch_tag_count", 0.0)),
        "racer_chunk_store_ms_sum_max": chunk_store_ms_sum_max,
        "racer_chunk_store_ms_max": chunk_store_ms_max,
        "racer_store_wrapper_gap_ms_max": wrapper_gap_ms_max,
        "racer_distributed_setup_ms_max": racer_profile_max.get("setup_ms", 0.0),
        "racer_distributed_sizing_ms_max": racer_profile_max.get("sizing_ms", 0.0),
        "racer_distributed_data_rows_ms_max": racer_profile_max.get("data_rows_ms", 0.0),
        "racer_distributed_parity_ms_max": racer_profile_max.get("parity_ms", 0.0),
        "racer_distributed_storage_ms_max": racer_profile_max.get("storage_ms", 0.0),
        "racer_storage_begin_ms_max": racer_profile_max.get("storage_begin_ms", 0.0),
        "racer_storage_begin_barrier_ms_max": racer_profile_max.get("storage_begin_barrier_ms", 0.0),
        "racer_storage_enqueue_ms_max": racer_profile_max.get("storage_enqueue_ms", 0.0),
        "racer_storage_enqueue_barrier_ms_max": racer_profile_max.get("storage_enqueue_barrier_ms", 0.0),
        "racer_storage_wait_ms_max": racer_profile_max.get("storage_wait_ms", 0.0),
        "racer_storage_manifest_put_ms_max": racer_profile_max.get("storage_manifest_put_ms", 0.0),
        "racer_storage_commit_pre_barrier_ms_max": racer_profile_max.get("storage_commit_pre_barrier_ms", 0.0),
        "racer_storage_commit_ms_max": racer_profile_max.get("storage_commit_ms", 0.0),
        "racer_storage_commit_retry_count_max": racer_profile_max.get("storage_commit_retry_count", 0.0),
        "racer_storage_commit_wait_ms_max": racer_profile_max.get("storage_commit_wait_ms", 0.0),
        "racer_storage_commit_post_barrier_ms_max": racer_profile_max.get("storage_commit_post_barrier_ms", 0.0),
        "racer_csd_client_put_contiguous_ms_max": racer_profile_max.get("csd_client_put_contiguous_ms", 0.0),
        "racer_csd_client_put_export_ms_max": racer_profile_max.get("csd_client_put_export_ms", 0.0),
        "racer_csd_client_put_rpc_ms_max": racer_profile_max.get("csd_client_put_rpc_ms", 0.0),
        "racer_csd_client_put_total_ms_max": racer_profile_max.get("csd_client_put_total_ms", 0.0),
        "racer_csd_staging_alloc_us_max": racer_profile_max.get("csd_staging_alloc_us", 0.0),
        "racer_csd_staging_copy_enqueue_us_max": racer_profile_max.get("csd_staging_copy_enqueue_us", 0.0),
        "racer_csd_daemon_put_backend_write_ms_max": racer_profile_max.get("csd_daemon_put_backend_write_ms", 0.0),
        "racer_csd_daemon_put_total_ms_max": racer_profile_max.get("csd_daemon_put_total_ms", 0.0),
        "racer_distributed_final_barrier_ms_max": racer_profile_max.get("final_barrier_ms", 0.0),
        "racer_distributed_manifest_ms_max": racer_profile_max.get("manifest_ms", 0.0),
        "racer_distributed_data_rows_bytes_sent_max": int(racer_profile_max.get("data_rows_bytes_sent", 0.0)),
        "racer_distributed_parity_bytes_sent_max": int(racer_profile_max.get("parity_bytes_sent", 0.0)),
        "racer_distributed_local_storage_nbytes_max": int(racer_profile_max.get("local_storage_nbytes", 0.0)),
        "racer_distributed_local_storage_chunk_count_max": int(racer_profile_max.get("local_storage_chunk_count", 0.0)),
        "racer_distributed_local_chunks_released_nbytes_max": int(racer_profile_max.get("local_chunks_released_nbytes", 0.0)),
        "racer_distributed_local_chunks_released_count_max": int(racer_profile_max.get("local_chunks_released_count", 0.0)),
        "racer_distributed_store_wall_ms_max": store_wall_ms_max,
        "racer_distributed_store_wall_ms_local": store_wall_ms_local,
        "racer_checkpoint_commit_wall_ms_max": checkpoint_commit_wall_ms_max,
        "racer_checkpoint_commit_wall_ms_local": checkpoint_commit_wall_ms_local,
        "racer_tree_manifest_publish_ms_max": max_values["tree_manifest_publish_ms"],
        "racer_csd_commit_marker_publish_ms_max": max_values["csd_commit_marker_publish_ms"],
        "racer_shared_commit_marker_cache_ms_max": max_values["shared_commit_marker_cache_ms"],
        "racer_checkpoint_metadata_publish_ms_max": max_values["checkpoint_metadata_publish_ms"],
        "payload_bytes_total": int(payload_bytes_total),
        "payload_cuda_slot_released_count_max": int(
            max_values["payload_cuda_slot_released_count"]
        ),
        "payload_cuda_slot_released_nbytes_max": int(
            max_values["payload_cuda_slot_released_nbytes"]
        ),
        "racer_encoded_storage_bytes_total": int(encoded_storage_bytes_total),
        "racer_storage_amplification": (
            float(encoded_storage_bytes_total) / float(payload_bytes_total)
            if payload_bytes_total > 0
            else 0.0
        ),
        "racer_logical_store_bandwidth_gbps": logical_store_bandwidth_gbps,
        "racer_encoded_store_bandwidth_gbps": encoded_store_bandwidth_gbps,
        "racer_logical_commit_bandwidth_gbps": logical_commit_bandwidth_gbps,
        "racer_encoded_commit_bandwidth_gbps": encoded_commit_bandwidth_gbps,
        "racer_egm_copy_bandwidth_gbps": egm_copy_bandwidth_gbps,
        "payload_bytes_local": int(local_bytes),
        "tensor_leaf_count_local": int(local_leaf_count),
        "tensor_leaf_count_max": int(max_leaf_count),
        "racer_chunk_count_local": int(local_chunk_count),
        "racer_chunk_count_max": int(max_chunk_count),
        "racer_chunk_size": int(chunk_size),
    }
    _SESSION.tree_checkpoints[tag] = prepared.tree
    _print_rank0(
        "RACER distributed tensor-tree checkpoint stored: "
        f"tag={tag}, store={store_wall_ms_max:.2f} ms, "
        f"commit_wall={checkpoint_commit_wall_ms_max:.2f} ms, "
        f"metadata_publish={max_values['checkpoint_metadata_publish_ms']:.2f} ms, "
        f"metadata={metadata_ms_max:.2f} ms, "
        f"command={store_command_ms_max:.2f} ms, "
        f"tensor_view={tensor_view_ms_max:.2f} ms, "
        f"racer_calls={racer_calls_ms_max:.2f} ms, "
        f"racer_inner_total={racer_profile_max.get('total_ms', 0.0):.2f} ms, "
        f"batch_prepare={racer_profile_max.get('store_batch_prepare_ms', 0.0):.2f} ms, "
        f"batch_finalize={racer_profile_max.get('store_batch_finalize_ms', 0.0):.2f} ms, "
        f"batch_total={racer_profile_max.get('store_batch_total_ms', 0.0):.2f} ms, "
        f"batch_tags={int(racer_profile_max.get('store_batch_tag_count', 0.0))}, "
        f"chunk_store_max={chunk_store_ms_max:.2f} ms, "
        f"setup={racer_profile_max.get('setup_ms', 0.0):.2f} ms, "
        f"sizing={racer_profile_max.get('sizing_ms', 0.0):.2f} ms, "
        f"data_rows={racer_profile_max.get('data_rows_ms', 0.0):.2f} ms, "
        f"parity={racer_profile_max.get('parity_ms', 0.0):.2f} ms, "
        f"storage={racer_profile_max.get('storage_ms', 0.0):.2f} ms, "
        f"storage_begin={racer_profile_max.get('storage_begin_ms', 0.0):.2f} ms, "
        f"storage_enqueue={racer_profile_max.get('storage_enqueue_ms', 0.0):.2f} ms, "
        f"storage_enqueue_barrier={racer_profile_max.get('storage_enqueue_barrier_ms', 0.0):.2f} ms, "
        f"storage_wait={racer_profile_max.get('storage_wait_ms', 0.0):.2f} ms, "
        f"storage_commit_wait={racer_profile_max.get('storage_commit_wait_ms', 0.0):.2f} ms, "
        f"storage_commit={racer_profile_max.get('storage_commit_ms', 0.0):.2f} ms, "
        f"client_put={racer_profile_max.get('csd_client_put_total_ms', 0.0):.2f} ms, "
        f"client_export={racer_profile_max.get('csd_client_put_export_ms', 0.0):.2f} ms, "
        f"client_rpc={racer_profile_max.get('csd_client_put_rpc_ms', 0.0):.2f} ms, "
        f"daemon_put={racer_profile_max.get('csd_daemon_put_total_ms', 0.0):.2f} ms, "
        f"daemon_dispatch={racer_profile_max.get('csd_daemon_put_dispatch_ms', 0.0):.2f} ms, "
        f"daemon_submit={racer_profile_max.get('csd_daemon_put_submit_ms', 0.0):.2f} ms, "
        f"daemon_backend_write={racer_profile_max.get('csd_daemon_put_backend_write_ms', 0.0):.2f} ms, "
        f"daemon_event_open_us={racer_profile_max.get('csd_daemon_ipc_event_open_us', 0.0):.2f}, "
        f"daemon_set_device={racer_profile_max.get('csd_daemon_set_device_ms', 0.0):.2f} ms, "
        f"daemon_event_create={racer_profile_max.get('csd_daemon_event_create_ms', 0.0):.2f} ms, "
        f"daemon_enqueue_api={racer_profile_max.get('csd_daemon_enqueue_api_ms', 0.0):.2f} ms, "
        f"staging_copy_enqueue_us={racer_profile_max.get('csd_staging_copy_enqueue_us', 0.0):.2f}, "
        f"wrapper_gap={wrapper_gap_ms_max:.2f} ms, "
        f"encoded_bytes={int(encoded_storage_bytes_total)}, "
        f"amplification={float(report['racer_storage_amplification']):.3f}x, "
        f"logical_bw={logical_store_bandwidth_gbps:.2f} GB/s, "
        f"encoded_bw={encoded_store_bandwidth_gbps:.2f} GB/s, "
        f"egm_copy_bw={egm_copy_bandwidth_gbps:.2f} GB/s, "
        f"bytes={int(payload_bytes_total)}, "
        f"local_bytes={int(local_bytes)}, "
        f"leaves={int(max_leaf_count)}, "
        f"chunks={int(max_chunk_count)}"
    )
    return report


def _distributed_store_tensor_tree(
    args: Any,
    *,
    tag: str,
    iteration: int,
    release: bool,
    ckpt_type: str,
    state_for_save: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    prepared = _prepare_distributed_store_tensor_tree(
        args,
        tag=tag,
        iteration=iteration,
        release=release,
        ckpt_type=ckpt_type,
        state_for_save=state_for_save,
        device=device,
    )
    executed = _execute_prepared_distributed_store(prepared)
    return _finalize_executed_distributed_store(executed)


def _distributed_load_tensor_tree(
    args: Any,
    *,
    tag: str,
    failed_train_ranks: list[int] | None,
    replacement_mapping: dict[int, int] | None,
    requested_train_ranks: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        tree = _SESSION.tree_checkpoints[tag]
    except KeyError as exc:
        raise KeyError(f"RACER tensor-tree checkpoint tag is not available in this process: {tag}") from exc
    metas = list(tree["metas"])
    chunks = list(tree.get("chunks", []))
    max_chunk_count = int(tree.get("max_chunk_count", tree.get("max_leaf_count", 0)))
    device = _current_cuda_device()
    load_start = time.perf_counter()
    load_fetch_ms_local = 0.0
    tensor_materialize_ms_local = 0.0
    tensor_scatter_enqueue_ms_local = 0.0
    tensor_scatter_sync_ms_local = 0.0
    spare_command_ms_local = 0.0
    racer_load_profile_local: dict[str, float] = {}
    direct_storage_load = (
        _racer_csd_storage_enabled(args)
        and not failed_train_ranks
        and not dict(replacement_mapping or {})
        and os.environ.get("RACER_DISTRIBUTED_DIRECT_STORAGE_LOAD", "1").lower() in {"1", "true", "yes", "on"}
    )
    tensor_payloads: dict[int, torch.Tensor] = {
        idx: torch.empty(int(meta.get("nbytes", 0)), dtype=torch.uint8, device=device)
        for idx, meta in enumerate(metas)
    }
    materialize_stream = torch.cuda.Stream(device=device)
    pending_payloads: list[torch.Tensor] = []
    load_tags = [_chunk_tag(tag, chunk_index) for chunk_index in range(max_chunk_count)]
    expected_payload_checksums = [str(value) for value in list(tree.get("payload_checksums", []) or [])]
    payload_checksum_mode = str(tree.get("payload_checksum_mode", "") or "")
    if expected_payload_checksums and not payload_checksum_mode:
        payload_checksum_mode = _debug_payload_checksum_mode() or "sample64"
    payload_checksum_ms_local = 0.0
    payload_checksum_chunks_local = 0
    payload_checksum_mismatch_messages: list[str] = []
    if load_tags and not direct_storage_load:
        command_start = time.perf_counter()
        runtime = _send_spare_load_many(
            args,
            tags=load_tags,
            failed_train_ranks=failed_train_ranks or [],
            replacement_mapping=replacement_mapping,
            requested_train_ranks=requested_train_ranks,
        )
        _accumulate_numeric_profile(racer_load_profile_local, getattr(runtime, "init_profile", None))
        spare_command_ms_local = (time.perf_counter() - command_start) * 1000.0
    for chunk_index in range(max_chunk_count):
        fetch_start = time.perf_counter()
        payload, payload_profile = _distributed_load_payload(
            args,
            tag=_chunk_tag(tag, chunk_index),
            failed_train_ranks=failed_train_ranks or [],
            replacement_mapping=replacement_mapping,
            requested_train_ranks=requested_train_ranks,
            return_profile=True,
            send_command=False,
            direct_storage=direct_storage_load,
        )
        load_fetch_ms_local += (time.perf_counter() - fetch_start) * 1000.0
        _accumulate_numeric_profile(racer_load_profile_local, payload_profile)
        if payload_checksum_mode and chunk_index < len(expected_payload_checksums):
            checksum_start = time.perf_counter()
            actual_checksum = _debug_payload_checksum(payload, mode=payload_checksum_mode)
            payload_checksum_ms_local += (time.perf_counter() - checksum_start) * 1000.0
            payload_checksum_chunks_local += 1
            expected_checksum = str(expected_payload_checksums[chunk_index])
            if actual_checksum != expected_checksum:
                payload_checksum_mismatch_messages.append(
                    "chunk="
                    f"{chunk_index}, expected={expected_checksum}, got={actual_checksum}, "
                    f"payload_nbytes={int(payload.numel())}"
                )
        if chunk_index < len(chunks):
            materialize_start = time.perf_counter()
            materialize_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(materialize_stream):
                for segment in chunks[chunk_index].get("segments", []):
                    tensor_id = int(segment["tensor_id"])
                    nbytes = int(segment["nbytes"])
                    if nbytes <= 0:
                        continue
                    if int(metas[tensor_id].get("nbytes", 0)) <= 0:
                        continue
                    tensor_offset = int(segment["tensor_offset"])
                    chunk_offset = int(segment["chunk_offset"])
                    if tensor_offset + nbytes > int(tensor_payloads[tensor_id].numel()):
                        raise RuntimeError(
                            "RACER tensor-tree chunk segment exceeds tensor payload: "
                            f"tensor_id={tensor_id}, tensor_offset={tensor_offset}, nbytes={nbytes}, "
                            f"tensor_payload_nbytes={int(tensor_payloads[tensor_id].numel())}"
                        )
                    if chunk_offset + nbytes > int(payload.numel()):
                        raise RuntimeError(
                            "RACER tensor-tree chunk segment exceeds fetched chunk payload: "
                            f"chunk_index={chunk_index}, chunk_offset={chunk_offset}, nbytes={nbytes}, "
                            f"chunk_payload_nbytes={int(payload.numel())}"
                        )
                    tensor_payloads[tensor_id].narrow(0, tensor_offset, nbytes).copy_(
                        payload.narrow(0, chunk_offset, nbytes),
                        non_blocking=True,
                    )
            pending_payloads.append(payload)
            enqueue_ms = (time.perf_counter() - materialize_start) * 1000.0
            tensor_scatter_enqueue_ms_local += enqueue_ms
            tensor_materialize_ms_local += enqueue_ms
    scatter_sync_start = time.perf_counter()
    materialize_stream.synchronize()
    tensor_scatter_sync_ms_local = (time.perf_counter() - scatter_sync_start) * 1000.0
    tensor_materialize_ms_local += tensor_scatter_sync_ms_local
    pending_payloads.clear()
    if payload_checksum_mode and expected_payload_checksums:
        any_checksum_mismatch = bool(payload_checksum_mismatch_messages)
        if _distributed_initialized():
            mismatch_tensor = torch.tensor([1 if any_checksum_mismatch else 0], dtype=torch.int32, device=device)
            torch.distributed.all_reduce(mismatch_tensor, op=torch.distributed.ReduceOp.MAX)
            any_checksum_mismatch = bool(int(mismatch_tensor.item()))
        if payload_checksum_mismatch_messages:
            raise RuntimeError(
                "RACER payload checksum mismatch after load: "
                f"rank={_rank()}, tag={tag}, mode={payload_checksum_mode}; "
                + "; ".join(payload_checksum_mismatch_messages[:4])
            )
        if any_checksum_mismatch:
            raise RuntimeError(
                "RACER payload checksum mismatch after load on another rank: "
                f"rank={_rank()}, tag={tag}, mode={payload_checksum_mode}"
            )
        print(
            "RACER payload checksum verified: "
            f"rank={_rank()}, tag={tag}, mode={payload_checksum_mode}, "
            f"chunks={payload_checksum_chunks_local}, ms={payload_checksum_ms_local:.2f}",
            flush=True,
        )
    tensor_by_id: dict[int, torch.Tensor] = {}
    for tensor_id, meta in enumerate(metas):
        materialize_start = time.perf_counter()
        tensor_by_id[tensor_id] = _tensor_from_cuda_bytes(tensor_payloads[tensor_id], meta)
        tensor_materialize_ms_local += (time.perf_counter() - materialize_start) * 1000.0
    decode_start = time.perf_counter()
    state_dict = _decode_tensor_tree(tree["skeleton"], tensor_by_id)
    tree_decode_ms_local = (time.perf_counter() - decode_start) * 1000.0
    # The decoded leaves own independent CUDA storage. Drop the fetch-slot
    # cache after scatter/decode so it does not overlap optimizer restore and
    # the first resumed training step. The loop-local payload otherwise keeps
    # the final slot alive even after the pool releases its references.
    if "payload" in locals():
        del payload
    load_slot_allocated_before = int(torch.cuda.memory_allocated(device))
    load_slot_reserved_before = int(torch.cuda.memory_reserved(device))
    load_slot_release = _payload_buffer_pool().release_cuda_load_slots(device=device)
    if os.environ.get("RACER_AGGRESSIVE_CUDA_CLEANUP", "0") == "1":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    load_slot_allocated_after = int(torch.cuda.memory_allocated(device))
    load_slot_reserved_after = int(torch.cuda.memory_reserved(device))
    if _rank() == 0:
        print(
            "RACER load fetch-slot release: "
            f"count={int(load_slot_release.get('released_cuda_load_slot_count', 0.0))}, "
            f"bytes={int(load_slot_release.get('released_cuda_load_slot_nbytes', 0.0))}, "
            f"allocated={load_slot_allocated_before}->{load_slot_allocated_after}, "
            f"reserved={load_slot_reserved_before}->{load_slot_reserved_after}",
            flush=True,
        )
    load_rebuild_ms = (time.perf_counter() - load_start) * 1000.0
    load_rebuild_ms_max = _max_float_across_ranks(load_rebuild_ms, device)
    load_fetch_ms_max = _max_float_across_ranks(load_fetch_ms_local, device)
    spare_command_ms_max = _max_float_across_ranks(spare_command_ms_local, device)
    tensor_materialize_ms_max = _max_float_across_ranks(tensor_materialize_ms_local, device)
    tensor_scatter_enqueue_ms_max = _max_float_across_ranks(tensor_scatter_enqueue_ms_local, device)
    tensor_scatter_sync_ms_max = _max_float_across_ranks(tensor_scatter_sync_ms_local, device)
    tree_decode_ms_max = _max_float_across_ranks(tree_decode_ms_local, device)
    racer_load_profile_max = {
        f"racer_{key}_max": _max_float_across_ranks(racer_load_profile_local.get(key, 0.0), device)
        for key in _LOAD_PROFILE_KEYS
    }
    report = {
        "load_scatter_ms": load_fetch_ms_max,
        "racer_fetch_ms": load_fetch_ms_max,
        "racer_spare_command_ms": spare_command_ms_max,
        "unpack_ms": 0.0,
        "tensor_materialize_ms": tensor_materialize_ms_max,
        "tensor_scatter_enqueue_ms": tensor_scatter_enqueue_ms_max,
        "tensor_scatter_sync_ms": tensor_scatter_sync_ms_max,
        "tree_decode_ms": tree_decode_ms_max,
        "tensor_rebuild_ms": tensor_materialize_ms_max + tree_decode_ms_max,
        "payload_cuda_load_slots_released_count_max": int(
            _max_float_across_ranks(
                load_slot_release.get("released_cuda_load_slot_count", 0.0), device
            )
        ),
        "payload_cuda_load_slots_released_nbytes_max": int(
            _max_float_across_ranks(
                load_slot_release.get("released_cuda_load_slot_nbytes", 0.0), device
            )
        ),
        "payload_checksum_verify_ms_max": _max_float_across_ranks(payload_checksum_ms_local, device),
        "payload_checksum_chunks_local": int(payload_checksum_chunks_local),
        "total_ms_local": load_rebuild_ms_max,
        "tensor_leaf_count_local": len(metas),
        "tensor_leaf_count_max": int(tree.get("max_leaf_count", len(metas))),
        "racer_chunk_count_local": len(chunks),
        "racer_chunk_count_max": max_chunk_count,
        "racer_direct_storage_load": bool(direct_storage_load),
        "racer_load_profile": racer_load_profile_max,
    }
    report.update(racer_load_profile_max)
    return state_dict, report


def _distributed_verify_tensor_tree(args: Any, *, tag: str) -> tuple[bool, dict[str, float]]:
    try:
        tree = _SESSION.tree_checkpoints[tag]
    except KeyError as exc:
        raise KeyError(f"RACER tensor-tree checkpoint tag is not available in this process: {tag}") from exc
    tensors = list(tree.get("source_tensors", []))
    if not tensors:
        raise RuntimeError(
            "RACER verify-on-save requires source tensors in this live process; "
            "only metadata is available from the manifest"
        )
    device = _current_cuda_device()
    chunk_size = int(tree.get("chunk_size", getattr(args, "racer_buffer_size", 64 * 1024 * 1024)))
    _, chunk_payloads = _pack_tensor_chunks(tensors, device, chunk_size)
    max_chunk_count = int(tree.get("max_chunk_count", len(chunk_payloads)))
    load_ms_total = 0.0
    compare_ms_total = 0.0
    ok = True
    for chunk_index in range(max_chunk_count):
        load_start = time.perf_counter()
        payload = _distributed_load_payload(
            args,
            tag=_chunk_tag(tag, chunk_index),
            failed_train_ranks=[],
            replacement_mapping=None,
            requested_train_ranks=_train_ranks(args),
        )
        load_ms_total += (time.perf_counter() - load_start) * 1000.0
        if chunk_index < len(chunk_payloads):
            expected = chunk_payloads[chunk_index]
        else:
            expected = _tensor_to_cuda_bytes(None, device)
        compare_start = time.perf_counter()
        ok = ok and bool(torch.equal(payload[: expected.numel()], expected))
        torch.cuda.synchronize(device)
        compare_ms_total += (time.perf_counter() - compare_start) * 1000.0
    load_ms_total = _max_float_across_ranks(load_ms_total, device)
    compare_ms_total = _max_float_across_ranks(compare_ms_total, device)
    if _distributed_initialized():
        ok_tensor = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(ok_tensor, op=torch.distributed.ReduceOp.MIN)
        ok = bool(int(ok_tensor.item()))
    return ok, {"load_scatter_ms": load_ms_total, "compare_ms": compare_ms_total}


def _local_config_dict(args: Any) -> dict[str, Any]:
    return {
        "k": int(getattr(args, "racer_k", 3)),
        "m": int(getattr(args, "racer_m", 1)),
        "train_ranks": _train_ranks(args),
        "spare_ranks": _spare_ranks(args),
        "buffer_size": int(getattr(args, "racer_buffer_size", 64 * 1024 * 1024)),
        "optimize_cauchy": bool(getattr(args, "racer_optimize_cauchy", False)),
    }


def _local_context(args: Any) -> Any:
    if _SESSION.local_racer_context is not None:
        return _SESSION.local_racer_context
    _racer_module(args)
    import racer  # type: ignore

    storage_backend = _racer_storage_backend(args)
    storage_options = _racer_csd_storage_options(args)
    _SESSION.local_racer_context = racer.init(
        **_local_config_dict(args),
        storage_backend=storage_backend,
        storage_options=storage_options,
    )
    return _SESSION.local_racer_context


def _all_gather_variable_payload(
    args: Any,
    local_payload: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    if local_payload.dtype != torch.uint8:
        raise TypeError("RACER local payload gather expects uint8 tensors")
    local_payload = local_payload.contiguous()
    if not _distributed_initialized():
        return {_rank(): local_payload}

    world = _world_size()
    size = torch.tensor([int(local_payload.numel())], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(size) for _ in range(world)]
    torch.distributed.all_gather(sizes, size)
    max_size = max(int(item.item()) for item in sizes)
    max_size = max(1, max_size)
    padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
    if int(local_payload.numel()) > 0:
        padded.narrow(0, 0, int(local_payload.numel())).copy_(local_payload, non_blocking=True)
    gathered = [torch.empty(max_size, dtype=torch.uint8, device=device) for _ in range(world)]
    torch.distributed.all_gather(gathered, padded)
    packets: dict[int, torch.Tensor] = {}
    for rank in _train_ranks(args):
        rank = int(rank)
        if rank >= world:
            raise ValueError(f"--racer-train-ranks contains rank {rank}, but world size is {world}")
        nbytes = int(sizes[rank].item())
        packets[rank] = gathered[rank].narrow(0, 0, nbytes).contiguous()
    return packets


def _local_store_tensor_tree(
    args: Any,
    *,
    tag: str,
    iteration: int,
    release: bool,
    ckpt_type: str,
    state_for_save: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    metadata_start = time.perf_counter()
    skeleton, metas, tensors = _extract_tensor_tree(state_for_save)
    metadata_ms = (time.perf_counter() - metadata_start) * 1000.0
    metadata_ms_max = _max_float_across_ranks(metadata_ms, device)
    local_leaf_count = len(tensors)
    max_leaf_count = _max_int_across_ranks(local_leaf_count, device)
    local_bytes = sum(int(meta.get("nbytes", 0)) for meta in metas)
    payload_bytes_total = _sum_int_across_ranks(local_bytes, device)
    chunk_size = int(getattr(args, "racer_buffer_size", 64 * 1024 * 1024))
    chunks, chunk_payloads = _pack_tensor_chunks(tensors, device, chunk_size)
    local_chunk_count = len(chunk_payloads)
    max_chunk_count = _max_int_across_ranks(local_chunk_count, device)
    _SESSION.tree_checkpoints[tag] = {
        "skeleton": skeleton,
        "metas": metas,
        "source_tensors": tensors,
        "chunks": chunks,
        "max_leaf_count": int(max_leaf_count),
        "max_chunk_count": int(max_chunk_count),
        "chunk_size": int(chunk_size),
        "iteration": int(iteration),
        "release": bool(release),
        "ckpt_type": str(ckpt_type),
    }
    _write_tree_manifest(args, tag, _SESSION.tree_checkpoints[tag])

    _barrier()
    store_start = time.perf_counter()
    tensor_view_ms_local = 0.0
    racer_calls_ms_local = 0.0
    for chunk_index in range(int(max_chunk_count)):
        tensor_view_start = time.perf_counter()
        if chunk_index < local_chunk_count:
            local_packet = chunk_payloads[chunk_index]
        else:
            local_packet = _tensor_to_cuda_bytes(None, device)
        tensor_view_ms_local += (time.perf_counter() - tensor_view_start) * 1000.0
        racer_call_start = time.perf_counter()
        _local_store_payload(args, tag=_chunk_tag(tag, chunk_index), local_payload=local_packet)
        racer_calls_ms_local += (time.perf_counter() - racer_call_start) * 1000.0
    store_wall_ms_local = (time.perf_counter() - store_start) * 1000.0
    store_wall_ms_max = _max_float_across_ranks(store_wall_ms_local, device)
    tensor_view_ms_max = _max_float_across_ranks(tensor_view_ms_local, device)
    racer_calls_ms_max = _max_float_across_ranks(racer_calls_ms_local, device)
    report = {
        "tag": tag,
        "iteration": int(iteration),
        "release": bool(release),
        "ckpt_type": str(ckpt_type),
        "tensor_tree_metadata_ms_max": metadata_ms_max,
        "tensor_view_ms_max": tensor_view_ms_max,
        "racer_local_calls_ms_max": racer_calls_ms_max,
        "racer_local_store_wall_ms_max": store_wall_ms_max,
        "racer_local_store_wall_ms_local": store_wall_ms_local,
        "payload_bytes_total": int(payload_bytes_total),
        "payload_bytes_local": int(local_bytes),
        "tensor_leaf_count_local": int(local_leaf_count),
        "tensor_leaf_count_max": int(max_leaf_count),
        "racer_chunk_count_local": int(local_chunk_count),
        "racer_chunk_count_max": int(max_chunk_count),
        "racer_chunk_size": int(chunk_size),
    }
    _print_rank0(
        "RACER local tensor-tree checkpoint stored: "
        f"tag={tag}, store={store_wall_ms_max:.2f} ms, "
        f"metadata={metadata_ms_max:.2f} ms, "
        f"tensor_view={tensor_view_ms_max:.2f} ms, "
        f"racer_calls={racer_calls_ms_max:.2f} ms, "
        f"bytes={int(payload_bytes_total)}, "
        f"local_bytes={int(local_bytes)}, "
        f"leaves={int(max_leaf_count)}, "
        f"chunks={int(max_chunk_count)}"
    )
    return report


def _local_load_tensor_tree(
    args: Any,
    *,
    tag: str,
    failed_train_ranks: list[int] | None,
    replacement_mapping: dict[int, int] | None,
    requested_train_ranks: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        tree = _SESSION.tree_checkpoints[tag]
    except KeyError as exc:
        raise KeyError(f"RACER tensor-tree checkpoint tag is not available in this process: {tag}") from exc
    metas = list(tree["metas"])
    chunks = list(tree.get("chunks", []))
    max_chunk_count = int(tree.get("max_chunk_count", tree.get("max_leaf_count", 0)))
    device = _current_cuda_device()
    load_start = time.perf_counter()
    load_fetch_ms_local = 0.0
    tensor_materialize_ms_local = 0.0
    tensor_scatter_enqueue_ms_local = 0.0
    tensor_scatter_sync_ms_local = 0.0
    tensor_payloads: dict[int, torch.Tensor] = {
        idx: torch.empty(int(meta.get("nbytes", 0)), dtype=torch.uint8, device=device)
        for idx, meta in enumerate(metas)
    }
    materialize_stream = torch.cuda.Stream(device=device)
    pending_payloads: list[torch.Tensor] = []
    for chunk_index in range(max_chunk_count):
        fetch_start = time.perf_counter()
        payload = _local_load_payload(
            args,
            tag=_chunk_tag(tag, chunk_index),
            failed_train_ranks=failed_train_ranks or [],
            replacement_mapping=replacement_mapping,
            requested_train_ranks=requested_train_ranks,
        )
        load_fetch_ms_local += (time.perf_counter() - fetch_start) * 1000.0
        if chunk_index < len(chunks):
            materialize_start = time.perf_counter()
            materialize_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(materialize_stream):
                for segment in chunks[chunk_index].get("segments", []):
                    tensor_id = int(segment["tensor_id"])
                    nbytes = int(segment["nbytes"])
                    if nbytes <= 0:
                        continue
                    if int(metas[tensor_id].get("nbytes", 0)) <= 0:
                        continue
                    tensor_offset = int(segment["tensor_offset"])
                    chunk_offset = int(segment["chunk_offset"])
                    if tensor_offset + nbytes > int(tensor_payloads[tensor_id].numel()):
                        raise RuntimeError(
                            "RACER tensor-tree chunk segment exceeds tensor payload: "
                            f"tensor_id={tensor_id}, tensor_offset={tensor_offset}, nbytes={nbytes}, "
                            f"tensor_payload_nbytes={int(tensor_payloads[tensor_id].numel())}"
                        )
                    if chunk_offset + nbytes > int(payload.numel()):
                        raise RuntimeError(
                            "RACER tensor-tree chunk segment exceeds fetched chunk payload: "
                            f"chunk_index={chunk_index}, chunk_offset={chunk_offset}, nbytes={nbytes}, "
                            f"chunk_payload_nbytes={int(payload.numel())}"
                        )
                    tensor_payloads[tensor_id].narrow(0, tensor_offset, nbytes).copy_(
                        payload.narrow(0, chunk_offset, nbytes),
                        non_blocking=True,
                    )
            pending_payloads.append(payload)
            enqueue_ms = (time.perf_counter() - materialize_start) * 1000.0
            tensor_scatter_enqueue_ms_local += enqueue_ms
            tensor_materialize_ms_local += enqueue_ms
    scatter_sync_start = time.perf_counter()
    materialize_stream.synchronize()
    tensor_scatter_sync_ms_local = (time.perf_counter() - scatter_sync_start) * 1000.0
    tensor_materialize_ms_local += tensor_scatter_sync_ms_local
    pending_payloads.clear()
    tensor_by_id: dict[int, torch.Tensor] = {}
    for tensor_id, meta in enumerate(metas):
        materialize_start = time.perf_counter()
        tensor_by_id[tensor_id] = _tensor_from_cuda_bytes(tensor_payloads[tensor_id], meta)
        tensor_materialize_ms_local += (time.perf_counter() - materialize_start) * 1000.0
    decode_start = time.perf_counter()
    state_dict = _decode_tensor_tree(tree["skeleton"], tensor_by_id)
    tree_decode_ms_local = (time.perf_counter() - decode_start) * 1000.0
    load_rebuild_ms = (time.perf_counter() - load_start) * 1000.0
    load_rebuild_ms_max = _max_float_across_ranks(load_rebuild_ms, device)
    load_fetch_ms_max = _max_float_across_ranks(load_fetch_ms_local, device)
    tensor_materialize_ms_max = _max_float_across_ranks(tensor_materialize_ms_local, device)
    tensor_scatter_enqueue_ms_max = _max_float_across_ranks(tensor_scatter_enqueue_ms_local, device)
    tensor_scatter_sync_ms_max = _max_float_across_ranks(tensor_scatter_sync_ms_local, device)
    tree_decode_ms_max = _max_float_across_ranks(tree_decode_ms_local, device)
    report = {
        "load_scatter_ms": load_fetch_ms_max,
        "racer_fetch_ms": load_fetch_ms_max,
        "unpack_ms": 0.0,
        "tensor_materialize_ms": tensor_materialize_ms_max,
        "tensor_scatter_enqueue_ms": tensor_scatter_enqueue_ms_max,
        "tensor_scatter_sync_ms": tensor_scatter_sync_ms_max,
        "tree_decode_ms": tree_decode_ms_max,
        "tensor_rebuild_ms": tensor_materialize_ms_max + tree_decode_ms_max,
        "total_ms_local": load_rebuild_ms_max,
        "tensor_leaf_count_local": len(metas),
        "tensor_leaf_count_max": int(tree.get("max_leaf_count", len(metas))),
        "racer_chunk_count_local": len(chunks),
        "racer_chunk_count_max": max_chunk_count,
    }
    return state_dict, report


def _local_verify_tensor_tree(args: Any, *, tag: str) -> tuple[bool, dict[str, float]]:
    try:
        tree = _SESSION.tree_checkpoints[tag]
    except KeyError as exc:
        raise KeyError(f"RACER tensor-tree checkpoint tag is not available in this process: {tag}") from exc
    tensors = list(tree.get("source_tensors", []))
    if not tensors:
        raise RuntimeError(
            "RACER verify-on-save requires source tensors in this live process; "
            "only metadata is available from the manifest"
        )
    device = _current_cuda_device()
    chunk_size = int(tree.get("chunk_size", getattr(args, "racer_buffer_size", 64 * 1024 * 1024)))
    _, chunk_payloads = _pack_tensor_chunks(tensors, device, chunk_size)
    max_chunk_count = int(tree.get("max_chunk_count", len(chunk_payloads)))
    load_ms_total = 0.0
    compare_ms_total = 0.0
    ok = True
    for chunk_index in range(max_chunk_count):
        load_start = time.perf_counter()
        payload = _local_load_payload(
            args,
            tag=_chunk_tag(tag, chunk_index),
            failed_train_ranks=[],
            replacement_mapping=None,
            requested_train_ranks=_train_ranks(args),
        )
        load_ms_total += (time.perf_counter() - load_start) * 1000.0
        if chunk_index < len(chunk_payloads):
            expected = chunk_payloads[chunk_index]
        else:
            expected = _tensor_to_cuda_bytes(None, device)
        compare_start = time.perf_counter()
        ok = ok and bool(torch.equal(payload[: expected.numel()], expected))
        torch.cuda.synchronize(device)
        compare_ms_total += (time.perf_counter() - compare_start) * 1000.0
    load_ms_total = _max_float_across_ranks(load_ms_total, device)
    compare_ms_total = _max_float_across_ranks(compare_ms_total, device)
    if _distributed_initialized():
        ok_tensor = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(ok_tensor, op=torch.distributed.ReduceOp.MIN)
        ok = bool(int(ok_tensor.item()))
    return ok, {"load_scatter_ms": load_ms_total, "compare_ms": compare_ms_total}


def _local_store_payload(args: Any, *, tag: str, local_payload: torch.Tensor) -> Any:
    ctx = _local_context(args)
    device = _current_cuda_device()
    packets = _all_gather_variable_payload(args, local_payload, device)
    handle = ctx.store(packets, tag=tag, async_op=False)
    return handle


def _local_load_payload(
    args: Any,
    *,
    tag: str,
    failed_train_ranks: list[int] | None,
    replacement_mapping: dict[int, int] | None,
    requested_train_ranks: list[int],
) -> torch.Tensor:
    rank = _rank()
    if rank not in _train_ranks(args):
        raise RuntimeError(f"Megatron rank {rank} is not listed in --racer-train-ranks={_train_ranks(args)}")
    requested = [rank] if rank in [int(item) for item in requested_train_ranks] else []
    if not requested:
        raise RuntimeError(f"RACER local load did not request current rank {rank}")
    result = _local_context(args).load(
        tag=tag,
        failed_train_ranks=failed_train_ranks or [],
        requested_train_ranks=requested,
        replacement_mapping=replacement_mapping,
    )
    try:
        return result[rank].detach().contiguous()
    except KeyError as exc:
        raise KeyError(f"RACER local load did not return payload for train rank {rank}") from exc


def _distributed_store_payload(args: Any, *, tag: str, local_payload: torch.Tensor) -> Any:
    _racer_module(args)
    from racer.distributed import distributed_store  # type: ignore

    runtime = _send_spare_command(args, op="store", tag=tag)
    chunk_storage = _racer_chunk_storage(args)
    state = distributed_store(
        config=_distributed_config(args),
        local_packet=local_payload,
        tag=tag,
        process_group=runtime.process_group,
        chunk_storage=chunk_storage,
    )
    runtime.states[tag] = state
    return state


def _distributed_load_payload(
    args: Any,
    *,
    tag: str,
    failed_train_ranks: list[int] | None,
    replacement_mapping: dict[int, int] | None,
    requested_train_ranks: list[int],
    return_profile: bool = False,
    send_command: bool = True,
    direct_storage: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    _racer_module(args)
    from racer.distributed import (  # type: ignore
        distributed_load,
        distributed_load_from_storage,
        distributed_load_local_payload_from_storage,
    )

    failed = _to_racer_pg_train_ranks(args, failed_train_ranks)
    requested = _to_racer_pg_train_ranks(args, requested_train_ranks)
    replacement = _to_racer_pg_replacement_mapping(args, replacement_mapping)
    replacement_sources = _replacement_sources_by_target(replacement)
    if direct_storage:
        if failed or replacement:
            raise RuntimeError("direct RACER storage load is only valid when no failed ranks need recovery")
        rank_map_data = _racer_pg_rank_map(args)
        racer_rank = int(rank_map_data[int(_rank())])
        if racer_rank not in requested:
            raise KeyError(f"RACER direct storage load did not request local rank {racer_rank}")
        chunk_storage = _racer_chunk_storage(args)
        if chunk_storage is None:
            raise KeyError(f"RACER distributed storage tag is not available: {tag}")
        result = distributed_load_local_payload_from_storage(
            config=_distributed_config(args),
            tag=tag,
            chunk_storage=chunk_storage,
            requested_train_rank=racer_rank,
            device=_current_cuda_device(),
        )
        payload = result.recovered[racer_rank].detach().contiguous()
        if return_profile:
            return payload, dict(getattr(result, "profile", None) or {})
        return payload

    if send_command:
        runtime = _send_spare_command(
            args,
            op="load",
            tag=tag,
            failed_train_ranks=failed_train_ranks or [],
            replacement_mapping=replacement_mapping,
            requested_train_ranks=requested_train_ranks,
        )
    else:
        runtime = _distributed_runtime(args)
    cached_state = runtime.states.get(tag)
    if cached_state is not None and getattr(cached_state, "local_chunks", None):
        result = distributed_load(
            state=cached_state,
            failed_train_ranks=failed,
            requested_train_ranks=requested,
            replacement_mapping=replacement,
            process_group=runtime.process_group,
        )
    else:
        chunk_storage = _racer_chunk_storage(args)
        if chunk_storage is None:
            raise KeyError(f"RACER distributed memory checkpoint tag is not available in this process: {tag}")
        result = distributed_load_from_storage(
            config=_distributed_config(args),
            tag=tag,
            chunk_storage=chunk_storage,
            failed_train_ranks=failed,
            requested_train_ranks=requested,
            replacement_mapping=replacement,
            process_group=runtime.process_group,
        )
    runtime_rank = int(runtime.rank)
    source_rank = replacement_sources.get(runtime_rank, runtime_rank)
    try:
        payload = result.recovered[source_rank].detach().contiguous()
    except KeyError as exc:
        raise KeyError(
            "RACER distributed load did not return payload for "
            f"source RACER rank {source_rank} routed to runtime RACER rank {runtime_rank}"
        ) from exc
    if return_profile:
        return payload, dict(getattr(result, "profile", None) or {})
    return payload


def _distributed_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return int(torch.distributed.get_rank()) if _distributed_initialized() else 0


def _world_size() -> int:
    return int(torch.distributed.get_world_size()) if _distributed_initialized() else 1


def _barrier() -> None:
    if _distributed_initialized():
        torch.distributed.barrier()


def _print_rank0(message: str) -> None:
    if _rank() == 0:
        print(message, flush=True)


def _parse_int_list(value: Any, default: list[int]) -> list[int]:
    return rank_map.parse_int_list(value, default)


def _train_ranks(args: Any) -> list[int]:
    return rank_map.train_ranks(args, _world_size())


def _spare_ranks(args: Any) -> list[int]:
    return rank_map.spare_ranks(args, _train_ranks(args))


def _racer_module(args: Any):
    racer_path = getattr(args, "racer_path", None)
    if racer_path:
        root = str(Path(racer_path).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    import racer  # type: ignore

    return racer


def _current_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("RACER memory checkpointing requires CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _max_float_across_ranks(value: float, device: torch.device) -> float:
    if not _distributed_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def _max_floats_across_ranks(values: dict[str, float], device: torch.device) -> dict[str, float]:
    items = [(key, float(value)) for key, value in values.items()]
    if not _distributed_initialized():
        return dict(items)
    tensor = torch.tensor([value for _, value in items], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return {key: float(value) for (key, _), value in zip(items, tensor.tolist())}


def _max_int_across_ranks(value: int, device: torch.device) -> int:
    if not _distributed_initialized():
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def _sum_int_across_ranks(value: int, device: torch.device) -> int:
    if not _distributed_initialized():
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def _tag_for_iteration(iteration: int, release: bool) -> str:
    suffix = "release" if release else f"iter_{int(iteration):07d}"
    return f"megatron:{suffix}"


def _iteration_from_tag(tag: str | None) -> int:
    if not tag:
        return -1
    if tag.endswith("release"):
        return 0
    try:
        return int(tag.rsplit("_", 1)[1])
    except Exception:
        return -1


def _select_tag(iteration: int | None) -> str | None:
    if iteration is not None and int(iteration) >= 0:
        return _SESSION.tags_by_iteration.get(int(iteration))
    return _SESSION.latest_tag
