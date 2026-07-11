"""Tensor-tree extraction and materialization for Megatron RACER checkpoints."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch

TENSOR_MARKER = "__racer_tensor_ref__"
TENSOR_ATTR_MARKER = "__racer_tensor_attr_ref__"


def is_tensor_data_object(value: Any) -> bool:
    return (
        hasattr(value, "data")
        and isinstance(getattr(value, "data"), torch.Tensor)
        and hasattr(value, "without_data")
    )


def restore_tensor_data_as_tensor(value: Any) -> bool:
    return value.__class__.__name__ == "ShardedTensor"


def extract_tensor_tree(obj: Any) -> tuple[Any, list[dict[str, Any]], list[torch.Tensor]]:
    tensors: list[torch.Tensor] = []
    metas: list[dict[str, Any]] = []

    def encode_tensor(value: torch.Tensor) -> dict[str, int]:
        idx = len(tensors)
        detached = value.detach()
        if not detached.is_contiguous():
            detached = detached.contiguous()
        nbytes = int(detached.numel() * detached.element_size())
        metas.append(
            {
                "id": idx,
                "dtype": str(detached.dtype),
                "shape": tuple(int(dim) for dim in detached.shape),
                "requires_grad": bool(value.requires_grad),
                "device": str(detached.device),
                "nbytes": nbytes,
            }
        )
        tensors.append(detached)
        return {TENSOR_MARKER: idx}

    def encode(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return encode_tensor(value)
        if is_tensor_data_object(value):
            idx = len(tensors)
            tensor_marker = encode_tensor(getattr(value, "data"))
            base = value.without_data()
            return {
                TENSOR_ATTR_MARKER: {
                    "object": base,
                    "attr": "data",
                    "tensor": tensor_marker,
                    "restore": "tensor" if restore_tensor_data_as_tensor(value) else "object",
                    "tensor_id": idx,
                }
            }
        if isinstance(value, dict):
            return {copy.deepcopy(key): encode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [encode(child) for child in value]
        if isinstance(value, tuple):
            return tuple(encode(child) for child in value)
        return copy.deepcopy(value)

    return encode(obj), metas, tensors


def tensor_to_cuda_bytes(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if tensor is None:
        return torch.empty(0, dtype=torch.uint8, device=device)
    detached = tensor.detach()
    if not detached.is_contiguous():
        detached = detached.contiguous()
    if detached.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=device)
    byte_view = detached.view(torch.uint8).reshape(-1)
    if byte_view.device != device:
        byte_view = byte_view.to(device=device, non_blocking=True)
    return byte_view.contiguous()


def pack_tensor_chunks(
    tensors: list[torch.Tensor],
    device: torch.device,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    chunk_size = max(1, int(chunk_size))
    chunks: list[dict[str, Any]] = []
    payloads: list[torch.Tensor] = []
    parts: list[torch.Tensor] = []
    segments: list[dict[str, int]] = []
    chunk_nbytes = 0

    def flush() -> None:
        nonlocal parts, segments, chunk_nbytes
        if not parts:
            return
        payload = torch.cat(parts).contiguous()
        chunks.append(
            {
                "id": len(chunks),
                "nbytes": int(chunk_nbytes),
                "segments": [dict(segment) for segment in segments],
            }
        )
        payloads.append(payload)
        parts = []
        segments = []
        chunk_nbytes = 0

    for tensor_id, tensor in enumerate(tensors):
        payload = tensor_to_cuda_bytes(tensor, device)
        tensor_nbytes = int(payload.numel())
        tensor_offset = 0
        while tensor_offset < tensor_nbytes:
            remaining_in_chunk = chunk_size - chunk_nbytes
            if remaining_in_chunk <= 0:
                flush()
                remaining_in_chunk = chunk_size
            take = min(tensor_nbytes - tensor_offset, remaining_in_chunk)
            parts.append(payload.narrow(0, tensor_offset, take))
            segments.append(
                {
                    "tensor_id": int(tensor_id),
                    "tensor_offset": int(tensor_offset),
                    "chunk_offset": int(chunk_nbytes),
                    "nbytes": int(take),
                }
            )
            tensor_offset += take
            chunk_nbytes += take
            if chunk_nbytes >= chunk_size:
                flush()
    flush()
    return chunks, payloads


class PinnedPayloadBufferPool:
    """Reusable training-process-owned pinned host buffers for payload chunks."""

    def __init__(self) -> None:
        self.chunk_size = 0
        self.buffers: list[torch.Tensor] = []
        self.cuda_load_slots: dict[str, list[torch.Tensor]] = {}
        self.cuda_load_slot_sizes: dict[str, int] = {}
        self.last_ensure_profile: dict[str, float] = {}

    def ensure(self, *, chunk_size: int, chunk_count: int) -> list[torch.Tensor]:
        import time

        chunk_size = max(1, int(chunk_size))
        chunk_count = max(0, int(chunk_count))
        start = time.perf_counter()
        reused = 0
        allocated = 0
        if self.chunk_size != chunk_size:
            self.buffers = []
            self.chunk_size = chunk_size
        reused = min(len(self.buffers), chunk_count)
        while len(self.buffers) < chunk_count:
            self.buffers.append(torch.empty(chunk_size, dtype=torch.uint8, pin_memory=True))
            allocated += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.last_ensure_profile = {
            "payload_pool_ensure_ms": float(elapsed_ms),
            "payload_pool_allocated_buffers": float(allocated),
            "payload_pool_reused_buffers": float(reused),
            "payload_pool_buffer_count": float(len(self.buffers)),
            "payload_pool_reserved_nbytes": float(len(self.buffers) * chunk_size),
        }
        return self.buffers[:chunk_count]

    def ensure_cuda_load_slots(
        self,
        *,
        device: torch.device,
        chunk_size: int,
        slot_count: int,
    ) -> dict[int, torch.Tensor]:
        """Keep the batched pinned-to-device slots alive across checkpoints."""

        import time

        chunk_size = max(1, int(chunk_size))
        slot_count = max(0, int(slot_count))
        device = torch.device(device)
        device_key = str(device)
        start = time.perf_counter()
        if self.cuda_load_slot_sizes.get(device_key) != chunk_size:
            self.cuda_load_slots[device_key] = []
            self.cuda_load_slot_sizes[device_key] = chunk_size
        slots = self.cuda_load_slots.setdefault(device_key, [])
        reused = min(len(slots), slot_count)
        allocated = 0
        while len(slots) < slot_count:
            slots.append(torch.empty(chunk_size, dtype=torch.uint8, device=device))
            allocated += 1
        self.last_ensure_profile.update(
            {
                "payload_cuda_slot_ensure_ms": float(
                    (time.perf_counter() - start) * 1000.0
                ),
                "payload_cuda_slot_allocated": float(allocated),
                "payload_cuda_slot_reused": float(reused),
                "payload_cuda_slot_count": float(len(slots)),
                "payload_cuda_slot_reserved_nbytes": float(len(slots) * chunk_size),
            }
        )
        return {index: tensor for index, tensor in enumerate(slots[:slot_count])}

    def release_cuda_load_slots(self, *, device: torch.device) -> dict[str, float]:
        """Drop pool-owned CUDA load-slot references for one device."""

        device_key = str(torch.device(device))
        slots = self.cuda_load_slots.pop(device_key, [])
        self.cuda_load_slot_sizes.pop(device_key, None)
        released_nbytes = sum(
            int(slot.numel()) * int(slot.element_size()) for slot in slots
        )
        return {
            "released_cuda_load_slot_count": float(len(slots)),
            "released_cuda_load_slot_nbytes": float(released_nbytes),
        }

    def prewarm(
        self,
        *,
        chunk_size: int,
        chunk_count: int,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        buffers = self.ensure(chunk_size=chunk_size, chunk_count=chunk_count)
        if device is not None:
            slots = self.ensure_cuda_load_slots(
                device=device,
                chunk_size=chunk_size,
                slot_count=chunk_count,
            )
            if torch.device(device).type == "cuda" and slots:
                import time

                warmup_start = time.perf_counter()
                for index, host in enumerate(buffers):
                    slots[index].copy_(host, non_blocking=True)
                torch.cuda.synchronize(device)
                self.last_ensure_profile["payload_cuda_slot_warmup_ms"] = float(
                    (time.perf_counter() - warmup_start) * 1000.0
                )
        return dict(self.last_ensure_profile)


@dataclass
class PinnedPayloadChunks:
    """Training-process-owned pinned host payload chunks.

    The storage here is intentionally not CSD-owned final checkpoint storage.
    It is a transient staging area that keeps the full per-rank payload out of
    GPU memory while RACER streams chunks through a bounded number of CUDA
    slots.
    """

    chunks: list[dict[str, Any]]
    buffers: list[torch.Tensor]
    valid_nbytes: list[int]
    chunk_size: int
    profile: dict[str, float] = field(default_factory=dict)
    _ready_events: list[torch.cuda.Event | None] = field(default_factory=list)
    _load_slots: dict[int, torch.Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.buffers)

    @property
    def total_valid_nbytes(self) -> int:
        return sum(int(value) for value in self.valid_nbytes)

    @property
    def total_reserved_nbytes(self) -> int:
        return int(self.chunk_size) * len(self.buffers)

    def cuda_payload(self, index: int, device: torch.device, *, slot_id: int = 0) -> torch.Tensor:
        index = int(index)
        slot_id = int(slot_id)
        if index < 0 or index >= len(self.buffers):
            raise IndexError(index)
        nbytes = int(self.valid_nbytes[index])
        if nbytes <= 0:
            return torch.empty(0, dtype=torch.uint8, device=device)
        ready = self._ready_events[index] if index < len(self._ready_events) else None
        if ready is not None:
            torch.cuda.current_stream(device).wait_event(ready)
        load_slot = self._load_slots.get(slot_id)
        if load_slot is None or load_slot.device != device or int(load_slot.numel()) < nbytes:
            load_slot = torch.empty(int(self.chunk_size), dtype=torch.uint8, device=device)
            self._load_slots[slot_id] = load_slot
        dst = load_slot.narrow(0, 0, nbytes)
        dst.copy_(self.buffers[index].narrow(0, 0, nbytes), non_blocking=True)
        return dst

    def release_cuda_load_slots(self) -> dict[str, float]:
        """Drop chunk-container CUDA slot references while retaining pinned payloads."""

        slots = list(self._load_slots.values())
        released_nbytes = sum(
            int(slot.numel()) * int(slot.element_size()) for slot in slots
        )
        self._load_slots.clear()
        return {
            "released_cuda_load_slot_count": float(len(slots)),
            "released_cuda_load_slot_nbytes": float(released_nbytes),
        }


def pack_tensor_chunks_to_pinned(
    tensors: list[torch.Tensor],
    device: torch.device,
    chunk_size: int,
    *,
    device_slot_count: int = 2,
    host_buffer_pool: PinnedPayloadBufferPool | None = None,
    expected_chunk_count: int | None = None,
) -> PinnedPayloadChunks:
    chunk_size = max(1, int(chunk_size))
    slot_count = max(1, int(device_slot_count))
    import time

    wall_start = time.perf_counter()
    chunks: list[dict[str, Any]] = []
    if expected_chunk_count is None:
        expected_chunk_count = 0
        for tensor in tensors:
            expected_chunk_count += int(tensor.detach().numel() * tensor.detach().element_size())
        expected_chunk_count = (int(expected_chunk_count) + chunk_size - 1) // chunk_size
    pooled_buffers = (
        host_buffer_pool.ensure(chunk_size=chunk_size, chunk_count=int(expected_chunk_count))
        if host_buffer_pool is not None
        else []
    )
    shared_load_slots = (
        host_buffer_pool.ensure_cuda_load_slots(
            device=device,
            chunk_size=chunk_size,
            slot_count=int(expected_chunk_count),
        )
        if host_buffer_pool is not None
        else {}
    )
    pool_profile = dict(getattr(host_buffer_pool, "last_ensure_profile", {}) or {})
    buffers: list[torch.Tensor] = []
    valid_nbytes: list[int] = []
    ready_events: list[torch.cuda.Event | None] = []
    segments: list[dict[str, int]] = []
    chunk_nbytes = 0
    chunk_index = 0
    slot_alloc_start = time.perf_counter()
    device_slots = [torch.empty(chunk_size, dtype=torch.uint8, device=device) for _ in range(slot_count)]
    copy_streams = [torch.cuda.Stream(device=device) for _ in range(slot_count)]
    slot_done: list[torch.cuda.Event | None] = [None for _ in range(slot_count)]
    slot_alloc_ms = (time.perf_counter() - slot_alloc_start) * 1000.0
    host_alloc_ms = 0.0
    tensor_view_ms = 0.0
    device_gather_enqueue_ms = 0.0
    host_copy_enqueue_ms = 0.0
    current_slot_wait_enqueue_ms = 0.0
    segment_count = 0

    def current_slot() -> torch.Tensor:
        nonlocal current_slot_wait_enqueue_ms
        slot_id = chunk_index % slot_count
        done = slot_done[slot_id]
        if done is not None:
            wait_start = time.perf_counter()
            torch.cuda.current_stream(device).wait_event(done)
            current_slot_wait_enqueue_ms += (time.perf_counter() - wait_start) * 1000.0
        return device_slots[slot_id]

    def flush() -> None:
        nonlocal segments, chunk_nbytes, chunk_index, host_alloc_ms, host_copy_enqueue_ms
        if chunk_nbytes <= 0:
            return
        slot_id = chunk_index % slot_count
        slot = device_slots[slot_id]
        host_alloc_start = time.perf_counter()
        if chunk_index < len(pooled_buffers):
            host = pooled_buffers[chunk_index]
        else:
            host = torch.empty(chunk_size, dtype=torch.uint8, pin_memory=True)
        host_alloc_ms += (time.perf_counter() - host_alloc_start) * 1000.0
        valid = int(chunk_nbytes)
        fill_ready = torch.cuda.Event()
        fill_ready.record(torch.cuda.current_stream(device))
        copy_stream = copy_streams[slot_id]
        copy_enqueue_start = time.perf_counter()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(fill_ready)
            host.narrow(0, 0, valid).copy_(slot.narrow(0, 0, valid), non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(copy_stream)
        host_copy_enqueue_ms += (time.perf_counter() - copy_enqueue_start) * 1000.0
        slot_done[slot_id] = copy_done
        chunks.append(
            {
                "id": len(chunks),
                "nbytes": valid,
                "segments": [dict(segment) for segment in segments],
            }
        )
        buffers.append(host)
        valid_nbytes.append(valid)
        ready_events.append(copy_done)
        segments = []
        chunk_nbytes = 0
        chunk_index += 1

    for tensor_id, tensor in enumerate(tensors):
        tensor_view_start = time.perf_counter()
        payload = tensor_to_cuda_bytes(tensor, device)
        tensor_view_ms += (time.perf_counter() - tensor_view_start) * 1000.0
        tensor_nbytes = int(payload.numel())
        tensor_offset = 0
        while tensor_offset < tensor_nbytes:
            remaining_in_chunk = chunk_size - chunk_nbytes
            if remaining_in_chunk <= 0:
                flush()
                remaining_in_chunk = chunk_size
            take = min(tensor_nbytes - tensor_offset, remaining_in_chunk)
            slot = current_slot()
            gather_start = time.perf_counter()
            slot.narrow(0, chunk_nbytes, take).copy_(
                payload.narrow(0, tensor_offset, take),
                non_blocking=True,
            )
            device_gather_enqueue_ms += (time.perf_counter() - gather_start) * 1000.0
            segment_count += 1
            segments.append(
                {
                    "tensor_id": int(tensor_id),
                    "tensor_offset": int(tensor_offset),
                    "chunk_offset": int(chunk_nbytes),
                    "nbytes": int(take),
                }
            )
            tensor_offset += take
            chunk_nbytes += take
            if chunk_nbytes >= chunk_size:
                flush()
    flush()
    final_sync_start = time.perf_counter()
    for copy_stream in copy_streams:
        copy_stream.synchronize()
    final_sync_ms = (time.perf_counter() - final_sync_start) * 1000.0
    total_ms = (time.perf_counter() - wall_start) * 1000.0
    return PinnedPayloadChunks(
        chunks=chunks,
        buffers=buffers,
        valid_nbytes=valid_nbytes,
        chunk_size=chunk_size,
        profile={
            "payload_pack_total_ms": float(total_ms),
            "payload_pack_slot_alloc_ms": float(slot_alloc_ms),
            "payload_pack_host_alloc_ms": float(host_alloc_ms),
            "payload_pack_tensor_view_ms": float(tensor_view_ms),
            "payload_pack_device_gather_enqueue_ms": float(device_gather_enqueue_ms),
            "payload_pack_host_copy_enqueue_ms": float(host_copy_enqueue_ms),
            "payload_pack_slot_reuse_wait_enqueue_ms": float(current_slot_wait_enqueue_ms),
            "payload_pack_final_sync_ms": float(final_sync_ms),
            "payload_pack_segment_count": float(segment_count),
            "payload_pack_chunk_count": float(len(buffers)),
            "payload_pack_valid_nbytes": float(sum(int(v) for v in valid_nbytes)),
            "payload_pack_reserved_nbytes": float(chunk_size * len(buffers)),
            **pool_profile,
        },
        _ready_events=ready_events,
        _load_slots=shared_load_slots,
    )


def dtype_from_name(name: str) -> torch.dtype:
    if not name.startswith("torch."):
        raise TypeError(f"unsupported tensor dtype metadata: {name}")
    attr = name.split(".", 1)[1]
    try:
        dtype = getattr(torch, attr)
    except AttributeError as exc:
        raise TypeError(f"unsupported tensor dtype metadata: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported tensor dtype metadata: {name}")
    return dtype


def tensor_from_cuda_bytes(payload: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    dtype = dtype_from_name(str(meta["dtype"]))
    shape = tuple(int(dim) for dim in meta["shape"])
    nbytes = int(meta.get("nbytes", 0))
    target_device = str(meta.get("device", "cpu"))
    payload = payload[:nbytes].detach().contiguous()
    if target_device.startswith("cuda") and torch.cuda.is_available():
        target = torch.device(target_device)
        if nbytes == 0:
            tensor = torch.empty(shape, dtype=dtype, device=target)
        else:
            if payload.device != target:
                payload = payload.to(device=target, non_blocking=False).contiguous()
            tensor = payload.view(dtype).reshape(shape)
    else:
        tensor = torch.empty(shape, dtype=dtype, device="cpu")
        if nbytes:
            tensor.view(torch.uint8).reshape(-1).copy_(payload.cpu().reshape(-1))
    if bool(meta.get("requires_grad", False)) and tensor.is_floating_point():
        if not tensor.is_leaf:
            tensor = tensor.clone()
        tensor.requires_grad_(True)
    return tensor


def decode_tensor_tree(skeleton: Any, tensor_by_id: dict[int, torch.Tensor]) -> Any:
    def decode(value: Any) -> Any:
        if isinstance(value, dict) and set(value.keys()) == {TENSOR_MARKER}:
            return tensor_by_id[int(value[TENSOR_MARKER])]
        if isinstance(value, dict) and set(value.keys()) == {TENSOR_ATTR_MARKER}:
            spec = dict(value[TENSOR_ATTR_MARKER])
            tensor = decode(spec["tensor"])
            if spec.get("restore") == "tensor":
                return tensor
            obj = copy.copy(spec["object"])
            setattr(obj, str(spec.get("attr", "data")), tensor)
            return obj
        if isinstance(value, dict):
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        if isinstance(value, tuple):
            return tuple(decode(child) for child in value)
        return value

    return decode(skeleton)
