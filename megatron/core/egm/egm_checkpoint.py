# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import base64
import ctypes
import hashlib
import io
import json
import os
import struct
import threading
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional

import torch
import torch.distributed

try:
    import cuda.bindings.driver as cuda_driver

    _cuda_driver_available = True
except ImportError:
    try:
        import cuda.cuda as cuda_driver

        _cuda_driver_available = True
    except ImportError:
        cuda_driver = None
        _cuda_driver_available = False

logger = getLogger(__name__)

_HEADER_SIZE = 64
_HEADER_STRUCT_FORMAT = "<IIQQQ"
_HEADER_STRUCT_SIZE = struct.calcsize(_HEADER_STRUCT_FORMAT)  # 32
_CHECKSUM_SIZE = 32
_HEADER_MAGIC = 0x45474D43
_HEADER_VERSION = 2
_EGM_ALIGNMENT = 2 * 1024 * 1024

_SLOT_STATUS_FREE = 0
_SLOT_STATUS_WRITING = 1
_SLOT_STATUS_VALID = 2


@dataclass
class _SlotMetadata:
    status: int = _SLOT_STATUS_FREE
    iteration: int = -1
    data_size: int = 0
    timestamp: float = 0.0
    source_rank: int = -1


@dataclass
class _ImportedSlotMapping:
    imported_handle: Any
    va_addr: int
    size_bytes: int


@dataclass
class EGMCheckpointConfig:
    enabled: bool = False
    pool_size_gb: float = 16.0
    num_slots: int = 2
    save_interval: int = 10
    enable_hierarchical_backup: bool = False
    rack_id: int = 0
    backup_rack_id: int = -1
    use_daemon: bool = False
    daemon_socket_path: str = "/tmp/megatron_egm_manager.sock"
    numa_node_id: int = 0


class EGMCheckpointManager:

    def __init__(
        self,
        config: EGMCheckpointConfig,
        device_id: int = 0,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.device_id = device_id
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        self._egm_manager = None
        self._egm_client = None
        self._slot_buffers: List[Optional[torch.Tensor]] = []
        self._slot_metadata: List[_SlotMetadata] = []
        self._imported_slot_mappings: List[Optional[_ImportedSlotMapping]] = []
        self._backend_slot_ids: List[int] = list(range(config.num_slots))
        self._current_write_slot: int = 0
        self._slot_capacity_bytes: int = 0
        self._explicit_restore_source_rank: Optional[int] = None
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        pool_size_bytes = int(self.config.pool_size_gb * 1024 ** 3)
        slot_size = pool_size_bytes // self.config.num_slots
        self._slot_capacity_bytes = slot_size

        if self.config.use_daemon:
            self._init_daemon_mode(slot_size)
        else:
            self._init_inprocess_mode(pool_size_bytes, slot_size)

        self._initialized = True

    def _init_inprocess_mode(self, pool_size_bytes: int, slot_size: int) -> None:
        from megatron.core.egm.egm_manager import EGMConfig, EGMManager

        mgr_config = EGMConfig(
            pool_size_bytes=pool_size_bytes,
            numa_node_id=self.config.numa_node_id,
            num_slots=self.config.num_slots,
            device_id=self.device_id,
        )
        self._egm_manager = EGMManager(mgr_config)
        self._egm_manager.initialize()

        for i in range(self.config.num_slots):
            self._slot_buffers.append(None)
            self._slot_metadata.append(_SlotMetadata())
            self._imported_slot_mappings.append(None)

        self._refresh_slot_metadata_from_backend()

        logger.info(
            f"[MCORE][EGM] Initialized in-process mode: "
            f"{self.config.num_slots} slots x {slot_size} bytes, "
            f"EGMManager initialized={self._egm_manager.initialized}"
        )

    def _init_daemon_mode(self, slot_size: int) -> None:
        from megatron.core.egm.egm_manager import EGMClient

        self._egm_client = EGMClient(
            socket_path=self.config.daemon_socket_path,
            timeout=30.0,
        )
        self._egm_client.connect()

        if not self._egm_client.ping():
            raise RuntimeError(
                f"Cannot connect to EGM daemon at {self.config.daemon_socket_path}"
            )

        all_slots = self._egm_client.get_all_slots()
        total_slots = len(all_slots.get("slots", [])) if all_slots.get("status") == "ok" else 0
        required_slots = (self.local_rank + 1) * self.config.num_slots
        if total_slots < required_slots:
            raise RuntimeError(
                "EGM daemon does not have enough slots for per-rank isolation: "
                f"need at least {required_slots}, got {total_slots}. "
                "Start the daemon with num_slots >= local_world_size * egm_num_slots."
            )
        slot_base = self.local_rank * self.config.num_slots
        self._backend_slot_ids = list(range(slot_base, slot_base + self.config.num_slots))

        for i in range(self.config.num_slots):
            self._slot_buffers.append(None)
            self._slot_metadata.append(_SlotMetadata())
            self._imported_slot_mappings.append(None)

        self._refresh_slot_metadata_from_backend()

        logger.info(
            f"[MCORE][EGM] Initialized daemon mode: connected to "
            f"{self.config.daemon_socket_path}, "
            f"{self.config.num_slots} local staging buffers, "
            f"global_rank={self.rank}, local_rank={self.local_rank}, "
            f"backend_slots={self._backend_slot_ids}"
        )

    def save_state_dict(self, state_dict: Dict[str, Any], iteration: int) -> bool:
        if not self._initialized:
            return False

        with self._lock:
            try:
                data = self._serialize_to_buffer(state_dict)
                if len(data) + _HEADER_SIZE > self._slot_capacity_bytes:
                    logger.error(
                        f"Data size ({len(data) + _HEADER_SIZE}) exceeds "
                        f"slot capacity ({self._slot_capacity_bytes})"
                    )
                    return False

                checksum = _compute_checksum(data)
                header = _build_header(iteration, len(data), checksum, self.rank)
                raw_bytes = header + data
                slot = self._write_raw_slot_data_internal(
                    self._current_write_slot, raw_bytes, iteration
                )
                if slot is None:
                    return False

                logger.info(
                    f"[MCORE][EGM] Saved checkpoint to slot {slot}: "
                    f"iteration={iteration}, data_size={len(data)}"
                )
                return True

            except Exception as e:
                logger.error(f"[MCORE][EGM] save_state_dict failed: {e}")
                return False

    def load_state_dict(self) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            return None

        with self._lock:
            self._refresh_slot_metadata_from_backend()
            restore_source_rank = self._resolve_restore_source_rank()
            best_slot = self._find_best_valid_slot(
                target_source_rank=restore_source_rank,
            )
            if best_slot is None:
                logger.warning(
                    f"[MCORE][EGM] No valid slot found for restore_source_rank="
                    f"{restore_source_rank}"
                )
                return None

            try:
                raw_data = self._read_raw_slot_data_internal(best_slot)
                if raw_data is None or len(raw_data) < _HEADER_SIZE:
                    return None

                header_bytes = raw_data[:_HEADER_SIZE]
                magic, version, iteration, data_size, reserved, stored_checksum = _parse_header(header_bytes)

                if magic != _HEADER_MAGIC:
                    logger.error(f"[MCORE][EGM] Invalid magic: 0x{magic:x}")
                    return None

                payload = raw_data[_HEADER_SIZE:_HEADER_SIZE + data_size]
                actual_checksum = _compute_checksum(payload)

                if actual_checksum != stored_checksum:
                    logger.error("[MCORE][EGM] Checksum mismatch")
                    return None

                state_dict = self._deserialize_from_buffer(payload)
                logger.info(
                    f"[MCORE][EGM] Loaded checkpoint from slot {best_slot}: "
                    f"iteration={iteration}, source_rank={restore_source_rank}"
                )
                return state_dict

            except Exception as e:
                logger.error(f"[MCORE][EGM] load_state_dict failed: {e}")
                return None

    def has_valid_checkpoint(
        self,
        source_rank: Optional[int] = None,
        include_remote: bool = False,
    ) -> bool:
        self._refresh_slot_metadata_from_backend()
        if source_rank is None and not include_remote:
            source_rank = self._resolve_restore_source_rank()
        return (
            self._find_best_valid_slot(
                target_source_rank=source_rank,
                allow_any_source=include_remote,
            )
            is not None
        )

    def get_latest_iteration(
        self,
        source_rank: Optional[int] = None,
        include_remote: bool = False,
    ) -> int:
        self._refresh_slot_metadata_from_backend()
        if source_rank is None and not include_remote:
            source_rank = self._resolve_restore_source_rank()
        best = self._find_best_valid_slot(
            target_source_rank=source_rank,
            allow_any_source=include_remote,
        )
        if best is not None:
            return self._slot_metadata[best].iteration
        return -1

    def set_restore_source_rank(self, source_rank: Optional[int]) -> None:
        self._explicit_restore_source_rank = source_rank

    def get_restore_source_rank(self) -> int:
        return self._resolve_restore_source_rank()

    def get_available_source_ranks(self) -> List[int]:
        self._refresh_slot_metadata_from_backend()
        available = set()
        for meta in self._slot_metadata:
            if meta.status != _SLOT_STATUS_VALID:
                continue
            if meta.source_rank >= 0:
                available.add(meta.source_rank)
            else:
                available.add(self.rank)
        return sorted(available)

    def get_raw_slot_data(self, slot: int) -> Optional[bytes]:
        self._refresh_slot_metadata_from_backend()
        return self._read_raw_slot_data_internal(slot)

    def write_raw_slot_data(self, slot: int, data: bytes, iteration: int = -1) -> bool:
        """Write raw checkpoint data (with header) to a slot buffer.

        Metadata (iteration, data_size) is extracted from the embedded header,
        so the iteration parameter is only used as a fallback if the header
        cannot be parsed.
        """
        with self._lock:
            return self._write_raw_slot_data_internal(slot, data, iteration) is not None

    def shutdown(self) -> None:
        if not self._initialized:
            return

        if self._egm_manager is not None:
            self._egm_manager.shutdown()
            self._egm_manager = None

        if self._egm_client is not None:
            self._egm_client.close()
            self._egm_client = None

        if _cuda_driver_available:
            for mapping in self._imported_slot_mappings:
                if mapping is None:
                    continue
                try:
                    cuda_driver.cuMemUnmap(mapping.va_addr, mapping.size_bytes)
                except Exception:
                    pass
                try:
                    cuda_driver.cuMemAddressFree(mapping.va_addr, mapping.size_bytes)
                except Exception:
                    pass
                try:
                    cuda_driver.cuMemRelease(mapping.imported_handle)
                except Exception:
                    pass

        self._slot_buffers.clear()
        self._slot_metadata.clear()
        self._imported_slot_mappings.clear()
        self._slot_capacity_bytes = 0
        self._initialized = False
        logger.info("[MCORE][EGM] EGMCheckpointManager shut down")

    def _serialize_to_buffer(self, state_dict: Dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    def _deserialize_from_buffer(self, data: bytes) -> Dict[str, Any]:
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location='cpu', weights_only=False)

    def _resolve_restore_source_rank(self) -> int:
        if self._explicit_restore_source_rank is not None:
            return self._explicit_restore_source_rank

        env_value = os.environ.get("FT_EGM_RESTORE_SOURCE_RANK")
        if env_value is not None:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(
                    f"[MCORE][EGM] Ignoring invalid FT_EGM_RESTORE_SOURCE_RANK="
                    f"{env_value!r}"
                )

        restart_path = os.environ.get(
            "FT_RESTART_INFO_PATH",
            "/tmp/megatron_ft_restart_info.json",
        )
        if not os.path.exists(restart_path):
            return self.rank

        try:
            with open(restart_path, "r") as f:
                restart_info = json.load(f)
        except Exception as e:
            logger.warning(
                f"[MCORE][EGM] Failed to read restart plan {restart_path}: {e}"
            )
            return self.rank

        if restart_info.get("strategy") != "egm_restore":
            return self.rank

        rank_mapping = restart_info.get("egm_restore_rank_by_new_rank")
        if isinstance(rank_mapping, dict):
            mapped_rank = rank_mapping.get(str(self.rank), rank_mapping.get(self.rank))
            if mapped_rank is not None:
                try:
                    return int(mapped_rank)
                except (TypeError, ValueError):
                    logger.warning(
                        f"[MCORE][EGM] Invalid restore rank mapping for rank "
                        f"{self.rank}: {mapped_rank!r}"
                    )

        mapped_rank = restart_info.get("egm_restore_source_rank")
        if mapped_rank is not None:
            try:
                return int(mapped_rank)
            except (TypeError, ValueError):
                logger.warning(
                    f"[MCORE][EGM] Invalid egm_restore_source_rank={mapped_rank!r} "
                    f"in restart plan {restart_path}"
                )

        return self.rank

    def _refresh_slot_metadata_from_backend(self) -> None:
        if self._egm_manager is None and self._egm_client is None:
            return

        for slot_id in range(self.config.num_slots):
            slot_info = None
            backend_slot_id = self._backend_slot_ids[slot_id]
            if self._egm_manager is not None:
                slot_info = self._egm_manager.get_slot_info(backend_slot_id)
            elif self._egm_client is not None:
                response = self._egm_client.get_slot_info(backend_slot_id)
                if response.get("status") == "ok":
                    slot_info = response.get("slot")

            if slot_info is not None:
                self._slot_metadata[slot_id] = self._slot_info_to_metadata(slot_info)
                self._ensure_daemon_mapping(slot_id, slot_info)

    def _slot_info_to_metadata(self, slot_info: Dict[str, Any]) -> _SlotMetadata:
        state_name = slot_info.get("state", "FREE")
        if state_name == "COMMITTED":
            status = _SLOT_STATUS_VALID
        elif state_name in ("ALLOCATED", "WRITING"):
            status = _SLOT_STATUS_WRITING
        else:
            status = _SLOT_STATUS_FREE

        return _SlotMetadata(
            status=status,
            iteration=int(slot_info.get("iteration", -1)),
            data_size=int(slot_info.get("data_size", 0)),
            timestamp=time.time(),
            source_rank=int(slot_info.get("source_rank", -1)),
        )

    def _ensure_daemon_mapping(self, local_slot: int, slot_info: Dict[str, Any]) -> None:
        if self._egm_client is None or not _cuda_driver_available:
            return
        if self._imported_slot_mappings[local_slot] is not None:
            return

        handle_b64 = slot_info.get("shareable_handle_b64")
        if not handle_b64:
            return

        try:
            from megatron.core.egm.egm_manager import (
                _coerce_cuda_ptr,
                _normalize_cuda_error_code,
                _set_host_device_rw_access,
            )

            raw_handle = base64.b64decode(handle_b64)
            err, imported_handle = cuda_driver.cuMemImportFromShareableHandle(
                raw_handle,
                cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            )
            if _normalize_cuda_error_code(err) != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemImportFromShareableHandle failed: {err}")

            size_bytes = int(slot_info.get("size_bytes", self._slot_capacity_bytes))
            err, va_addr = cuda_driver.cuMemAddressReserve(size_bytes, 0, 0, 0)
            if _normalize_cuda_error_code(err) != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemAddressReserve failed: {err}")
            va_addr = _coerce_cuda_ptr(va_addr)

            err = cuda_driver.cuMemMap(va_addr, size_bytes, 0, imported_handle, 0)
            if _normalize_cuda_error_code(err) != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemMap failed: {err}")

            _set_host_device_rw_access(
                va_addr,
                size_bytes,
                device_id=self.device_id,
                numa_node_id=self.config.numa_node_id,
            )

            self._imported_slot_mappings[local_slot] = _ImportedSlotMapping(
                imported_handle=imported_handle,
                va_addr=va_addr,
                size_bytes=size_bytes,
            )
            logger.info(
                f"[MCORE][EGM] Imported daemon EGM mapping for local_slot={local_slot} "
                f"backend_slot={self._backend_slot_ids[local_slot]}"
            )
        except Exception as e:
            logger.warning(
                f"[MCORE][EGM] Failed to import daemon EGM mapping for local slot "
                f"{local_slot}, falling back to socket payload path: {e}"
            )
            if 'va_addr' in locals() and va_addr:
                try:
                    cuda_driver.cuMemUnmap(va_addr, size_bytes)
                except Exception:
                    pass
                try:
                    cuda_driver.cuMemAddressFree(va_addr, size_bytes)
                except Exception:
                    pass
            if 'imported_handle' in locals() and imported_handle is not None:
                try:
                    cuda_driver.cuMemRelease(imported_handle)
                except Exception:
                    pass

    def _write_raw_slot_data_internal(
        self, local_slot: Optional[int], data: bytes, fallback_iteration: int = -1
    ) -> Optional[int]:
        if len(data) > self._slot_capacity_bytes:
            return None

        parsed_iteration = fallback_iteration
        parsed_data_size = max(len(data) - _HEADER_SIZE, 0)
        source_rank = self.rank
        checksum_hex = ""

        if len(data) >= _HEADER_SIZE:
            magic, _, iter_val, data_size, reserved, checksum = _parse_header(
                data[:_HEADER_SIZE]
            )
            if magic == _HEADER_MAGIC:
                payload = data[_HEADER_SIZE:_HEADER_SIZE + data_size]
                actual_checksum = _compute_checksum(payload)
                if actual_checksum != checksum:
                    logger.error("[MCORE][EGM] checksum mismatch while writing raw slot data")
                    return None
                parsed_iteration = int(iter_val)
                parsed_data_size = int(data_size)
                source_rank = int(reserved)
                checksum_hex = checksum.hex()

        pid = self.local_rank
        if local_slot is None or local_slot < 0 or local_slot >= self.config.num_slots:
            local_slot = self._current_write_slot
        backend_slot_id = self._backend_slot_ids[local_slot]

        if self._egm_manager is not None:
            manager_slot = self._egm_manager.slots[backend_slot_id]
            manager_slot.force_release()
            if not manager_slot.acquire(pid):
                return None
            if not manager_slot.begin_write(pid):
                return None
            if not self._egm_manager.write_raw_slot_data(
                backend_slot_id,
                pid,
                data,
                iteration=parsed_iteration,
                data_size=parsed_data_size,
                source_rank=source_rank,
                checksum_hex=checksum_hex,
            ):
                return None
            if not manager_slot.commit(pid):
                return None
        elif self._egm_client is not None:
            self._egm_client.release_slot(backend_slot_id, pid)
            acquire_resp = self._egm_client.acquire_specific_slot(backend_slot_id, pid)
            if acquire_resp.get("status") != "ok":
                return None
            begin_resp = self._egm_client.begin_write(backend_slot_id, pid)
            if begin_resp.get("status") != "ok":
                return None
            mapping = self._imported_slot_mappings[local_slot]
            if mapping is not None:
                ctypes.memmove(mapping.va_addr, data, len(data))
                meta_resp = self._egm_client.update_slot_metadata(
                    backend_slot_id,
                    iteration=parsed_iteration,
                    data_size=parsed_data_size,
                    total_bytes=len(data),
                    source_rank=source_rank,
                    checksum_hex=checksum_hex,
                    pid=pid,
                )
                if meta_resp.get("status") != "ok":
                    return None
            else:
                write_resp = self._egm_client.write_raw_slot_data(
                    backend_slot_id,
                    data,
                    iteration=parsed_iteration,
                    data_size=parsed_data_size,
                    source_rank=source_rank,
                    checksum_hex=checksum_hex,
                    pid=pid,
                )
                if write_resp.get("status") != "ok":
                    return None
            commit_resp = self._egm_client.commit_slot(backend_slot_id, pid)
            if commit_resp.get("status") != "ok":
                return None
        else:
            buf = self._slot_buffers[local_slot]
            if buf is None:
                buf = torch.zeros(self._slot_capacity_bytes, dtype=torch.uint8, device='cpu')
                self._slot_buffers[local_slot] = buf
            buf[:len(data)].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))

        self._slot_metadata[local_slot] = _SlotMetadata(
            status=_SLOT_STATUS_VALID,
            iteration=parsed_iteration,
            data_size=parsed_data_size,
            timestamp=time.time(),
            source_rank=source_rank,
        )
        self._current_write_slot = (local_slot + 1) % self.config.num_slots
        return local_slot

    def _read_raw_slot_data_internal(self, slot: int) -> Optional[bytes]:
        if slot < 0 or slot >= len(self._slot_metadata):
            return None
        meta = self._slot_metadata[slot]
        if meta.status != _SLOT_STATUS_VALID:
            return None

        if self._egm_manager is not None:
            return self._egm_manager.read_raw_slot_data(self._backend_slot_ids[slot])
        if self._egm_client is not None:
            mapping = self._imported_slot_mappings[slot]
            if mapping is not None:
                total_size = _HEADER_SIZE + meta.data_size
                return ctypes.string_at(mapping.va_addr, total_size)
            return self._egm_client.read_raw_slot_data(self._backend_slot_ids[slot])

        buf = self._slot_buffers[slot]
        if buf is None:
            return None
        total_size = _HEADER_SIZE + meta.data_size
        return bytes(buf[:total_size].numpy())

    def _find_best_valid_slot(
        self,
        target_source_rank: Optional[int] = None,
        allow_any_source: bool = False,
    ) -> Optional[int]:
        preferred_slot = None
        preferred_iter = -1
        fallback_slot = None
        fallback_iter = -1

        if target_source_rank is None and not allow_any_source:
            target_source_rank = self.rank

        for i, meta in enumerate(self._slot_metadata):
            if meta.status != _SLOT_STATUS_VALID:
                continue

            is_preferred = False
            if target_source_rank is None:
                is_preferred = True
            elif meta.source_rank == target_source_rank:
                is_preferred = True
            elif meta.source_rank == -1 and target_source_rank == self.rank:
                # Backward compatibility for older slots written before source_rank existed.
                is_preferred = True

            if is_preferred and meta.iteration > preferred_iter:
                preferred_iter = meta.iteration
                preferred_slot = i
            elif allow_any_source and meta.iteration > fallback_iter:
                fallback_iter = meta.iteration
                fallback_slot = i

        return preferred_slot if preferred_slot is not None else fallback_slot

    def _get_backup_slot(self) -> int:
        best = self._find_best_valid_slot()
        for i in range(len(self._slot_metadata)):
            if i != best:
                return i
        return 0


def _compute_checksum(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _build_header(
    iteration: int, data_size: int, checksum: bytes, source_rank: int = 0
) -> bytes:
    struct_part = struct.pack(
        _HEADER_STRUCT_FORMAT,
        _HEADER_MAGIC,
        _HEADER_VERSION,
        iteration,
        data_size,
        source_rank,
    )
    return struct_part + checksum


def _parse_header(header_bytes: bytes):
    struct_part = header_bytes[:_HEADER_STRUCT_SIZE]
    checksum = header_bytes[_HEADER_STRUCT_SIZE:_HEADER_STRUCT_SIZE + _CHECKSUM_SIZE]
    magic, version, iteration, data_size, reserved = struct.unpack(
        _HEADER_STRUCT_FORMAT, struct_part
    )
    return magic, version, iteration, data_size, reserved, checksum


def intra_rack_ring_backup(manager: EGMCheckpointManager, raw_data: bytes) -> bool:
    if not torch.distributed.is_initialized():
        return False

    try:
        rank = torch.distributed.get_rank()
        local_size = torch.cuda.device_count()
        if local_size <= 1:
            return True

        local_rank = rank % local_size
        base_rank = rank - local_rank
        next_rank = base_rank + (local_rank + 1) % local_size
        prev_rank = base_rank + (local_rank - 1) % local_size

        data_tensor = torch.frombuffer(bytearray(raw_data), dtype=torch.uint8)
        if torch.cuda.is_available():
            data_tensor = data_tensor.cuda()

        recv_tensor = torch.zeros_like(data_tensor)

        send_op = torch.distributed.isend(data_tensor, dst=next_rank)
        recv_op = torch.distributed.irecv(recv_tensor, src=prev_rank)

        send_op.wait()
        recv_op.wait()

        backup_slot = manager._get_backup_slot()
        received_bytes = bytes(recv_tensor.cpu().numpy())
        manager.write_raw_slot_data(backup_slot, received_bytes, -1)

        logger.info(
            f"[MCORE][EGM] Ring backup: rank {rank} sent to {next_rank}, "
            f"received from {prev_rank}, size={len(raw_data)}"
        )
        return True

    except Exception as e:
        logger.error(f"[MCORE][EGM] intra_rack_ring_backup failed: {e}")
        return False


def inter_rack_pair_backup(
    manager: EGMCheckpointManager, raw_data: bytes
) -> bool:
    if not torch.distributed.is_initialized():
        return False

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_size = torch.cuda.device_count()
        if local_size == 0:
            local_size = 1

        local_rank = rank % local_size
        rack_id = manager.config.rack_id
        backup_rack_id = manager.config.backup_rack_id

        if backup_rack_id < 0:
            return False

        peer_rank = backup_rack_id * local_size + local_rank
        if peer_rank >= world_size or peer_rank == rank:
            return False

        data_tensor = torch.frombuffer(bytearray(raw_data), dtype=torch.uint8)
        if torch.cuda.is_available():
            data_tensor = data_tensor.cuda()

        recv_tensor = torch.zeros_like(data_tensor)

        torch.distributed.barrier()

        send_op = torch.distributed.isend(data_tensor, dst=peer_rank)
        recv_op = torch.distributed.irecv(recv_tensor, src=peer_rank)

        send_op.wait()
        recv_op.wait()

        torch.distributed.barrier()

        backup_slot = manager._get_backup_slot()
        received_bytes = bytes(recv_tensor.cpu().numpy())
        manager.write_raw_slot_data(backup_slot, received_bytes, -1)

        logger.info(
            f"[MCORE][EGM] Inter-rack backup: rank {rank} exchanged with "
            f"peer rank {peer_rank} (rack {rack_id} <-> {backup_rack_id}), "
            f"size={len(raw_data)}"
        )
        return True

    except Exception as e:
        logger.error(f"[MCORE][EGM] inter_rack_pair_backup failed: {e}")
        return False
