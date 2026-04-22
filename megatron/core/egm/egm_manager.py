# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import ctypes
import base64
import json
import logging
import os
import signal
import socket
import struct
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)

_EGM_ALIGNMENT = 2 * 1024 * 1024
_DEFAULT_SOCKET_PATH = "/tmp/megatron_egm_manager.sock"
_IPC_HEADER_SIZE = 8
_IPC_MAX_MSG_SIZE = 1 << 40


class EGMError(Exception):
    pass


class EGMAllocationError(EGMError):
    pass


class EGMHandleExportError(EGMError):
    pass


class EGMSlotState(Enum):
    FREE = auto()
    ALLOCATED = auto()
    WRITING = auto()
    COMMITTED = auto()


def _serialize_shareable_handle(shareable_handle: Any) -> Optional[str]:
    if shareable_handle is None:
        return None

    raw: Optional[bytes] = None
    data = getattr(shareable_handle, "data", None)
    if data is not None:
        try:
            raw = bytes(data)
        except Exception:
            raw = None

    if raw is None:
        try:
            raw = bytes(shareable_handle)
        except Exception:
            raw = None

    if raw is None:
        return None

    return base64.b64encode(raw).decode("ascii")


def _normalize_cuda_error_code(err: Any) -> Any:
    # cuda.bindings.driver and cuda.cuda do not always return the CUresult
    # in the same shape. For example, cuInit() may return CUDA_SUCCESS or
    # a single-element tuple like (CUDA_SUCCESS,).
    if isinstance(err, tuple):
        if len(err) == 0:
            return None
        return err[0]
    return err


def _coerce_cuda_ptr(value: Any) -> int:
    if isinstance(value, int):
        return value
    raw = getattr(value, "value", None)
    if isinstance(raw, int):
        return raw
    return int(value)


def _check_cuda_error(err, msg="CUDA driver API call failed"):
    if _cuda_driver_available:
        normalized = _normalize_cuda_error_code(err)
        if normalized != cuda_driver.CUresult.CUDA_SUCCESS:
            raise EGMError(f"{msg}: {err}")


def _align_size(size: int, alignment: int = _EGM_ALIGNMENT) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def _build_host_device_access_descs(device_id: int, numa_node_id: int) -> List[Any]:
    host_access = cuda_driver.CUmemAccessDesc()
    host_access.location.type = (
        cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    )
    host_access.location.id = numa_node_id
    host_access.flags = cuda_driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    device_access = cuda_driver.CUmemAccessDesc()
    device_access.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    device_access.location.id = device_id
    device_access.flags = cuda_driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    return [host_access, device_access]


def _set_host_device_rw_access(
    va_addr: int,
    size: int,
    *,
    device_id: int,
    numa_node_id: int,
) -> None:
    access_descs = _build_host_device_access_descs(
        device_id=device_id,
        numa_node_id=numa_node_id,
    )
    err = cuda_driver.cuMemSetAccess(va_addr, size, access_descs, len(access_descs))
    _check_cuda_error(err, "cuMemSetAccess failed")


@dataclass
class EGMConfig:
    """Configuration for EGM memory pool."""

    pool_size_bytes: int = 64 * 1024 * 1024 * 1024
    numa_node_id: int = 0
    num_slots: int = 2
    socket_path: str = _DEFAULT_SOCKET_PATH
    device_id: int = 0
    use_fabric_handle: bool = True
    allocation_granularity: int = _EGM_ALIGNMENT
    server_timeout_s: float = 300.0
    max_clients: int = 64


@dataclass
class EGMSlot:
    """Represents a single checkpoint slot in EGM memory."""

    slot_id: int
    size_bytes: int
    state: EGMSlotState = EGMSlotState.FREE
    mem_handle: Any = None
    shareable_handle: Any = None
    va_addr: int = 0
    generation: int = 0
    owner_pid: Optional[int] = None
    iteration: int = -1
    data_size: int = 0
    total_bytes: int = 0
    source_rank: int = -1
    checksum_hex: str = ""
    host_payload: Optional[bytearray] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, pid: int) -> bool:
        with self._lock:
            if self.state != EGMSlotState.FREE:
                return False
            self.state = EGMSlotState.ALLOCATED
            self.owner_pid = pid
            return True

    def begin_write(self, pid: int) -> bool:
        with self._lock:
            if self.state != EGMSlotState.ALLOCATED or self.owner_pid != pid:
                return False
            self.state = EGMSlotState.WRITING
            return True

    def commit(self, pid: int) -> bool:
        with self._lock:
            if self.state != EGMSlotState.WRITING or self.owner_pid != pid:
                return False
            self.state = EGMSlotState.COMMITTED
            self.generation += 1
            return True

    def release(self, pid: int) -> bool:
        with self._lock:
            if self.owner_pid is not None and self.owner_pid != pid:
                return False
            self.state = EGMSlotState.FREE
            self.owner_pid = None
            self.iteration = -1
            self.data_size = 0
            self.total_bytes = 0
            self.source_rank = -1
            self.checksum_hex = ""
            return True

    def force_release(self):
        with self._lock:
            self.state = EGMSlotState.FREE
            self.owner_pid = None
            self.iteration = -1
            self.data_size = 0
            self.total_bytes = 0
            self.source_rank = -1
            self.checksum_hex = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "size_bytes": self.size_bytes,
            "state": self.state.name,
            "va_addr": self.va_addr,
            "generation": self.generation,
            "owner_pid": self.owner_pid,
            "iteration": self.iteration,
            "data_size": self.data_size,
            "total_bytes": self.total_bytes,
            "source_rank": self.source_rank,
            "checksum_hex": self.checksum_hex,
            "shareable_handle_b64": _serialize_shareable_handle(self.shareable_handle),
        }


class EGMManager:
    """Manages EGM (Extended GPU Memory) pools on GB200 NVL72 architecture."""

    def __init__(self, config: EGMConfig):
        self._config = config
        self._slots: List[EGMSlot] = []
        self._initialized = False
        self._lock = threading.Lock()
        self._ctx = None
        self._device = None
        self._alloc_handles: List[Any] = []
        self._va_ranges: List[Any] = []

    @property
    def config(self) -> EGMConfig:
        return self._config

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def slots(self) -> List[EGMSlot]:
        return list(self._slots)

    def initialize(self):
        """Initialize EGM memory pools and allocate slots."""
        with self._lock:
            if self._initialized:
                return

            if not _cuda_driver_available:
                logger.warning(
                    "[MCORE][EGM] cuda.bindings.driver not available, "
                    "running in stub mode without actual EGM allocation"
                )
                self._initialize_stub()
                return

            try:
                self._init_cuda_context()
                self._allocate_slots()
                self._initialized = True
                logger.info(
                    f"[MCORE][EGM] Initialized EGM manager: "
                    f"pool_size={self._config.pool_size_bytes}, "
                    f"num_slots={self._config.num_slots}, "
                    f"numa_node={self._config.numa_node_id}"
                )
            except Exception as e:
                logger.error(f"[MCORE][EGM] Failed to initialize EGM manager: {e}")
                self._cleanup_cuda_resources()
                raise EGMAllocationError(f"EGM initialization failed: {e}") from e

    def _initialize_stub(self):
        slot_size = _align_size(
            self._config.pool_size_bytes // self._config.num_slots,
            self._config.allocation_granularity,
        )
        for i in range(self._config.num_slots):
            slot = EGMSlot(
                slot_id=i,
                size_bytes=slot_size,
                state=EGMSlotState.FREE,
                host_payload=bytearray(slot_size),
            )
            self._slots.append(slot)
        self._initialized = True

    def _init_cuda_context(self):
        err = cuda_driver.cuInit(0)
        _check_cuda_error(err, "cuInit failed")

        err, self._device = cuda_driver.cuDeviceGet(self._config.device_id)
        _check_cuda_error(err, "cuDeviceGet failed")

        err, self._ctx = cuda_driver.cuDevicePrimaryCtxRetain(self._device)
        _check_cuda_error(err, "cuDevicePrimaryCtxRetain failed")

    def _allocate_slots(self):
        slot_size = _align_size(
            self._config.pool_size_bytes // self._config.num_slots,
            self._config.allocation_granularity,
        )

        for i in range(self._config.num_slots):
            mem_handle, shareable_handle, va_addr = self._allocate_egm_region(slot_size)
            slot = EGMSlot(
                slot_id=i,
                size_bytes=slot_size,
                state=EGMSlotState.FREE,
                mem_handle=mem_handle,
                shareable_handle=shareable_handle,
                va_addr=va_addr,
                # Avoid allocating a huge Python bytearray for each slot in the daemon.
                # The authoritative bytes live in the mapped EGM VA range (slot.va_addr).
                host_payload=None,
            )
            self._slots.append(slot)
            logger.info(
                f"[MCORE][EGM] Allocated slot {i}: "
                f"size={slot_size}, va_addr=0x{va_addr:x}"
            )

    def _allocate_egm_region(self, size: int):
        prop = cuda_driver.CUmemAllocationProp()
        prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = self._config.numa_node_id

        if self._config.use_fabric_handle:
            prop.requestedHandleTypes = (
                cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )

        err, granularity = cuda_driver.cuMemGetAllocationGranularity(
            prop, cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        _check_cuda_error(err, "cuMemGetAllocationGranularity failed")

        aligned_size = _align_size(size, max(granularity, self._config.allocation_granularity))

        err, mem_handle = cuda_driver.cuMemCreate(aligned_size, prop, 0)
        _check_cuda_error(err, "cuMemCreate failed")
        self._alloc_handles.append(mem_handle)

        shareable_handle = None
        if self._config.use_fabric_handle:
            try:
                err, shareable_handle = cuda_driver.cuMemExportToShareableHandle(
                    mem_handle,
                    cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    0,
                )
                _check_cuda_error(err, "cuMemExportToShareableHandle failed")
            except Exception as e:
                logger.warning(
                    f"[MCORE][EGM] Failed to export fabric handle, "
                    f"multi-node sharing may be unavailable: {e}"
                )

        err, va_addr = cuda_driver.cuMemAddressReserve(aligned_size, 0, 0, 0)
        _check_cuda_error(err, "cuMemAddressReserve failed")
        va_addr = _coerce_cuda_ptr(va_addr)
        self._va_ranges.append((va_addr, aligned_size))

        err = cuda_driver.cuMemMap(va_addr, aligned_size, 0, mem_handle, 0)
        _check_cuda_error(err, "cuMemMap failed")

        _set_host_device_rw_access(
            va_addr,
            aligned_size,
            device_id=self._config.device_id,
            numa_node_id=self._config.numa_node_id,
        )

        return mem_handle, shareable_handle, va_addr

    def acquire_slot(self, pid: int) -> Optional[EGMSlot]:
        """Acquire a free slot for a training process."""
        with self._lock:
            for slot in self._slots:
                if slot.acquire(pid):
                    logger.info(
                        f"[MCORE][EGM] Slot {slot.slot_id} acquired by pid {pid}"
                    )
                    return slot
        return None

    def acquire_specific_slot(self, slot_id: int, pid: int) -> Optional[EGMSlot]:
        """Acquire a specific slot if it is free."""
        with self._lock:
            if 0 <= slot_id < len(self._slots):
                slot = self._slots[slot_id]
                if slot.acquire(pid):
                    logger.info(
                        f"[MCORE][EGM] Specific slot {slot_id} acquired by pid {pid}"
                    )
                    return slot
        return None

    def release_slot(self, slot_id: int, pid: int) -> bool:
        """Release a slot back to the free pool."""
        with self._lock:
            if 0 <= slot_id < len(self._slots):
                result = self._slots[slot_id].release(pid)
                if result:
                    logger.info(
                        f"[MCORE][EGM] Slot {slot_id} released by pid {pid}"
                    )
                return result
        return False

    def begin_write(self, slot_id: int, pid: int) -> bool:
        """Mark a slot as being written to."""
        if 0 <= slot_id < len(self._slots):
            return self._slots[slot_id].begin_write(pid)
        return False

    def commit_slot(self, slot_id: int, pid: int) -> bool:
        """Commit a slot after writing is complete."""
        if 0 <= slot_id < len(self._slots):
            return self._slots[slot_id].commit(pid)
        return False

    def get_slot_info(self, slot_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific slot."""
        if 0 <= slot_id < len(self._slots):
            return self._slots[slot_id].to_dict()
        return None

    def get_all_slots_info(self) -> List[Dict[str, Any]]:
        """Get information about all slots."""
        return [slot.to_dict() for slot in self._slots]

    def write_raw_slot_data(
        self,
        slot_id: int,
        pid: int,
        data: bytes,
        *,
        iteration: int,
        data_size: int,
        source_rank: int,
        checksum_hex: str,
    ) -> bool:
        if not (0 <= slot_id < len(self._slots)):
            return False
        slot = self._slots[slot_id]
        with slot._lock:
            if slot.owner_pid != pid or slot.state != EGMSlotState.WRITING:
                return False
            if len(data) > slot.size_bytes:
                return False
            try:
                self._write_slot_bytes(slot, data)
                slot.iteration = iteration
                slot.data_size = data_size
                slot.total_bytes = len(data)
                slot.source_rank = source_rank
                slot.checksum_hex = checksum_hex
                return True
            except Exception as e:
                logger.error(
                    f"[MCORE][EGM] Failed to write raw slot data for slot {slot_id}: {e}"
                )
                return False

    def read_raw_slot_data(self, slot_id: int) -> Optional[bytes]:
        if not (0 <= slot_id < len(self._slots)):
            return None
        slot = self._slots[slot_id]
        with slot._lock:
            if slot.state != EGMSlotState.COMMITTED or slot.total_bytes <= 0:
                return None
            try:
                return self._read_slot_bytes(slot, slot.total_bytes)
            except Exception as e:
                logger.error(
                    f"[MCORE][EGM] Failed to read raw slot data for slot {slot_id}: {e}"
                )
                return None

    def _write_slot_bytes(self, slot: EGMSlot, data: bytes) -> None:
        # IMPORTANT: never memset/zero-fill the rest of the slot here.
        # The reader always uses slot.total_bytes, so writing extra gigabytes of zeros
        # would dominate checkpoint time and make bandwidth look terrible.
        if _cuda_driver_available and slot.va_addr:
            ctypes.memmove(slot.va_addr, data, len(data))
            return

        # Stub / fallback path: keep a Python-side copy for readback.
        # Allocate lazily to avoid huge memory overhead when CUDA driver path is available.
        if slot.host_payload is None or len(slot.host_payload) < len(data):
            slot.host_payload = bytearray(len(data))
        slot.host_payload[:len(data)] = data

    def _read_slot_bytes(self, slot: EGMSlot, total_bytes: int) -> bytes:
        if _cuda_driver_available and slot.va_addr:
            return ctypes.string_at(slot.va_addr, total_bytes)
        if slot.host_payload is None:
            return b""
        return bytes(slot.host_payload[:total_bytes])

    def update_slot_metadata(
        self,
        slot_id: int,
        pid: int,
        *,
        iteration: int,
        data_size: int,
        total_bytes: int,
        source_rank: int,
        checksum_hex: str,
    ) -> bool:
        if not (0 <= slot_id < len(self._slots)):
            return False
        slot = self._slots[slot_id]
        with slot._lock:
            if slot.owner_pid != pid or slot.state != EGMSlotState.WRITING:
                return False
            slot.iteration = iteration
            slot.data_size = data_size
            slot.total_bytes = total_bytes
            slot.source_rank = source_rank
            slot.checksum_hex = checksum_hex
            return True

    def get_active_slot(self) -> Optional[EGMSlot]:
        """Get the most recently committed slot (for ping-pong reads)."""
        best = None
        for slot in self._slots:
            if slot.state == EGMSlotState.COMMITTED:
                if best is None or slot.generation > best.generation:
                    best = slot
        return best

    def get_next_write_slot(self, pid: int) -> Optional[EGMSlot]:
        """Get the next slot for writing in ping-pong pattern."""
        active = self.get_active_slot()
        for slot in self._slots:
            if slot.state == EGMSlotState.FREE:
                if slot.acquire(pid):
                    return slot
        if active is not None:
            for slot in self._slots:
                if slot.slot_id != active.slot_id and slot.state == EGMSlotState.COMMITTED:
                    slot.force_release()
                    if slot.acquire(pid):
                        return slot
        return None

    def shutdown(self):
        """Shutdown the EGM manager and release all resources."""
        with self._lock:
            if not self._initialized:
                return
            logger.info("[MCORE][EGM] Shutting down EGM manager")
            for slot in self._slots:
                slot.force_release()
            self._cleanup_cuda_resources()
            self._slots.clear()
            self._initialized = False
            logger.info("[MCORE][EGM] EGM manager shutdown complete")

    def _cleanup_cuda_resources(self):
        if not _cuda_driver_available:
            return

        for va_addr, size in self._va_ranges:
            try:
                cuda_driver.cuMemUnmap(va_addr, size)
                cuda_driver.cuMemAddressFree(va_addr, size)
            except Exception:
                pass
        self._va_ranges.clear()

        for handle in self._alloc_handles:
            try:
                cuda_driver.cuMemRelease(handle)
            except Exception:
                pass
        self._alloc_handles.clear()

        if self._device is not None and self._ctx is not None:
            try:
                cuda_driver.cuDevicePrimaryCtxRelease(self._device)
            except Exception:
                pass
            self._ctx = None


class _IPCRequestType(Enum):
    ACQUIRE_SLOT = "acquire_slot"
    ACQUIRE_SPECIFIC_SLOT = "acquire_specific_slot"
    RELEASE_SLOT = "release_slot"
    BEGIN_WRITE = "begin_write"
    COMMIT_SLOT = "commit_slot"
    GET_SLOT_INFO = "get_slot_info"
    GET_ALL_SLOTS = "get_all_slots"
    GET_ACTIVE_SLOT = "get_active_slot"
    GET_NEXT_WRITE_SLOT = "get_next_write_slot"
    WRITE_RAW_SLOT_DATA = "write_raw_slot_data"
    READ_RAW_SLOT_DATA = "read_raw_slot_data"
    UPDATE_SLOT_METADATA = "update_slot_metadata"
    PING = "ping"
    SHUTDOWN = "shutdown"


def _send_msg(sock: socket.socket, data: bytes):
    header = struct.pack("!Q", len(data))
    sock.sendall(header + data)


def _recv_msg(sock: socket.socket) -> Optional[bytes]:
    header = b""
    while len(header) < _IPC_HEADER_SIZE:
        chunk = sock.recv(_IPC_HEADER_SIZE - len(header))
        if not chunk:
            return None
        header += chunk
    msg_len = struct.unpack("!Q", header)[0]
    if msg_len > _IPC_MAX_MSG_SIZE:
        return None
    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if not chunk:
            return None
        data += chunk
    return data


class EGMManagerServer:
    """Daemon process wrapper for EGM Manager with Unix domain socket IPC."""

    def __init__(self, config: EGMConfig):
        self._config = config
        self._manager = EGMManager(config)
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()

    @property
    def manager(self) -> EGMManager:
        return self._manager

    def start(self):
        """Start the EGM manager daemon."""
        self._manager.initialize()

        socket_path = self._config.socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(socket_path)
        self._server_socket.listen(self._config.max_clients)
        self._server_socket.settimeout(1.0)

        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="egm-accept"
        )
        self._accept_thread.start()

        logger.info(
            f"[MCORE][EGM] Server started on {socket_path}"
        )

    def _accept_loop(self):
        while self._running and not self._shutdown_event.is_set():
            try:
                client_sock, _ = self._server_socket.accept()
                t = threading.Thread(
                    target=self._handle_client,
                    args=(client_sock,),
                    daemon=True,
                    name="egm-client",
                )
                t.start()
                self._client_threads.append(t)
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.error("[MCORE][EGM] Server socket error in accept loop")
                break

    def _handle_client(self, client_sock: socket.socket):
        try:
            client_sock.settimeout(self._config.server_timeout_s)
            while self._running:
                raw = _recv_msg(client_sock)
                if raw is None:
                    break
                try:
                    request = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response = {"status": "error", "message": "invalid request format"}
                    _send_msg(client_sock, json.dumps(response).encode("utf-8"))
                    continue

                req_type = request.get("type")
                if req_type == _IPCRequestType.WRITE_RAW_SLOT_DATA.value:
                    payload = _recv_msg(client_sock)
                    if payload is None:
                        response = {"status": "error", "message": "missing payload"}
                    else:
                        response = self._process_write_raw_slot_request(request, payload)
                    _send_msg(client_sock, json.dumps(response).encode("utf-8"))
                    continue

                if req_type == _IPCRequestType.READ_RAW_SLOT_DATA.value:
                    response, payload = self._process_read_raw_slot_request(request)
                    _send_msg(client_sock, json.dumps(response).encode("utf-8"))
                    if response.get("status") == "ok" and payload is not None:
                        _send_msg(client_sock, payload)
                    continue

                if req_type == _IPCRequestType.UPDATE_SLOT_METADATA.value:
                    response = self._process_update_slot_metadata_request(request)
                    _send_msg(client_sock, json.dumps(response).encode("utf-8"))
                    continue

                response = self._process_request(request)
                _send_msg(client_sock, json.dumps(response).encode("utf-8"))

                if req_type == _IPCRequestType.SHUTDOWN.value:
                    break
        except socket.timeout:
            logger.warning("[MCORE][EGM] Client connection timed out")
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.error(f"[MCORE][EGM] Error handling client: {e}")
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_type = request.get("type")
        pid = request.get("pid", 0)

        if req_type == _IPCRequestType.PING.value:
            return {"status": "ok", "message": "pong"}

        if req_type == _IPCRequestType.ACQUIRE_SLOT.value:
            slot = self._manager.acquire_slot(pid)
            if slot is not None:
                return {"status": "ok", "slot": slot.to_dict()}
            return {"status": "error", "message": "no free slot available"}

        if req_type == _IPCRequestType.ACQUIRE_SPECIFIC_SLOT.value:
            slot_id = request.get("slot_id", -1)
            slot = self._manager.acquire_specific_slot(slot_id, pid)
            if slot is not None:
                return {"status": "ok", "slot": slot.to_dict()}
            return {"status": "error", "message": "requested slot unavailable"}

        if req_type == _IPCRequestType.RELEASE_SLOT.value:
            slot_id = request.get("slot_id", -1)
            result = self._manager.release_slot(slot_id, pid)
            return {"status": "ok" if result else "error", "released": result}

        if req_type == _IPCRequestType.BEGIN_WRITE.value:
            slot_id = request.get("slot_id", -1)
            result = self._manager.begin_write(slot_id, pid)
            return {"status": "ok" if result else "error", "writing": result}

        if req_type == _IPCRequestType.COMMIT_SLOT.value:
            slot_id = request.get("slot_id", -1)
            result = self._manager.commit_slot(slot_id, pid)
            return {"status": "ok" if result else "error", "committed": result}

        if req_type == _IPCRequestType.GET_SLOT_INFO.value:
            slot_id = request.get("slot_id", -1)
            info = self._manager.get_slot_info(slot_id)
            if info is not None:
                return {"status": "ok", "slot": info}
            return {"status": "error", "message": "invalid slot_id"}

        if req_type == _IPCRequestType.GET_ALL_SLOTS.value:
            return {"status": "ok", "slots": self._manager.get_all_slots_info()}

        if req_type == _IPCRequestType.GET_ACTIVE_SLOT.value:
            slot = self._manager.get_active_slot()
            if slot is not None:
                return {"status": "ok", "slot": slot.to_dict()}
            return {"status": "ok", "slot": None}

        if req_type == _IPCRequestType.GET_NEXT_WRITE_SLOT.value:
            slot = self._manager.get_next_write_slot(pid)
            if slot is not None:
                return {"status": "ok", "slot": slot.to_dict()}
            return {"status": "error", "message": "no write slot available"}

        if req_type == _IPCRequestType.SHUTDOWN.value:
            self._running = False
            self._shutdown_event.set()
            return {"status": "ok", "message": "shutting down"}

        return {"status": "error", "message": f"unknown request type: {req_type}"}

    def _process_write_raw_slot_request(
        self, request: Dict[str, Any], payload: bytes
    ) -> Dict[str, Any]:
        slot_id = request.get("slot_id", -1)
        pid = request.get("pid", 0)
        ok = self._manager.write_raw_slot_data(
            slot_id,
            pid,
            payload,
            iteration=int(request.get("iteration", -1)),
            data_size=int(request.get("data_size", max(len(payload) - 64, 0))),
            source_rank=int(request.get("source_rank", -1)),
            checksum_hex=str(request.get("checksum_hex", "")),
        )
        return {"status": "ok" if ok else "error", "written": ok}

    def _process_read_raw_slot_request(
        self, request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[bytes]]:
        slot_id = request.get("slot_id", -1)
        payload = self._manager.read_raw_slot_data(slot_id)
        if payload is None:
            return (
                {"status": "error", "message": "slot data unavailable"},
                None,
            )
        return (
            {"status": "ok", "data_size": len(payload)},
            payload,
        )

    def _process_update_slot_metadata_request(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        ok = self._manager.update_slot_metadata(
            int(request.get("slot_id", -1)),
            int(request.get("pid", 0)),
            iteration=int(request.get("iteration", -1)),
            data_size=int(request.get("data_size", 0)),
            total_bytes=int(request.get("total_bytes", 0)),
            source_rank=int(request.get("source_rank", -1)),
            checksum_hex=str(request.get("checksum_hex", "")),
        )
        return {"status": "ok" if ok else "error", "updated": ok}

    def stop(self):
        """Stop the EGM manager daemon."""
        self._running = False
        self._shutdown_event.set()

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=5.0)

        for t in self._client_threads:
            t.join(timeout=2.0)
        self._client_threads.clear()

        self._manager.shutdown()

        socket_path = self._config.socket_path
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass

        logger.info("[MCORE][EGM] Server stopped")

    def run_forever(self):
        """Run the server until a shutdown signal is received."""
        self.start()

        def _signal_handler(signum, _frame):
            logger.info(f"[MCORE][EGM] Received signal {signum}, shutting down")
            self._shutdown_event.set()

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        self._shutdown_event.wait()
        self.stop()


class EGMClient:
    """Client for communicating with the EGM Manager daemon via Unix domain socket."""

    def __init__(self, socket_path: str = _DEFAULT_SOCKET_PATH, timeout: float = 30.0):
        self._socket_path = socket_path
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self):
        """Connect to the EGM Manager daemon."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect(self._socket_path)

    def close(self):
        """Close the connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if self._sock is None:
            raise EGMError("Not connected to EGM manager")
        _send_msg(self._sock, json.dumps(request).encode("utf-8"))
        raw = _recv_msg(self._sock)
        if raw is None:
            raise EGMError("Connection lost to EGM manager")
        return json.loads(raw.decode("utf-8"))

    def ping(self) -> bool:
        """Ping the EGM manager."""
        resp = self._request({"type": _IPCRequestType.PING.value})
        return resp.get("status") == "ok"

    def acquire_slot(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Request a free slot from the manager."""
        if pid is None:
            pid = os.getpid()
        return self._request({
            "type": _IPCRequestType.ACQUIRE_SLOT.value,
            "pid": pid,
        })

    def acquire_specific_slot(
        self, slot_id: int, pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Request a specific slot from the manager."""
        if pid is None:
            pid = os.getpid()
        return self._request({
            "type": _IPCRequestType.ACQUIRE_SPECIFIC_SLOT.value,
            "slot_id": slot_id,
            "pid": pid,
        })

    def release_slot(self, slot_id: int, pid: Optional[int] = None) -> Dict[str, Any]:
        """Release a slot back to the manager."""
        if pid is None:
            pid = os.getpid()
        return self._request({
            "type": _IPCRequestType.RELEASE_SLOT.value,
            "slot_id": slot_id,
            "pid": pid,
        })

    def begin_write(self, slot_id: int, pid: Optional[int] = None) -> Dict[str, Any]:
        """Mark a slot as being written to."""
        if pid is None:
            pid = os.getpid()
        return self._request({
            "type": _IPCRequestType.BEGIN_WRITE.value,
            "slot_id": slot_id,
            "pid": pid,
        })

    def commit_slot(self, slot_id: int, pid: Optional[int] = None) -> Dict[str, Any]:
        """Commit a slot after writing is complete."""
        if pid is None:
            pid = os.getpid()
        return self._request({
            "type": _IPCRequestType.COMMIT_SLOT.value,
            "slot_id": slot_id,
            "pid": pid,
        })

    def get_slot_info(self, slot_id: int) -> Dict[str, Any]:
        """Get information about a specific slot."""
        return self._request({
            "type": _IPCRequestType.GET_SLOT_INFO.value,
            "slot_id": slot_id,
        })

    def get_all_slots(self) -> Dict[str, Any]:
        """Get information about all slots."""
        return self._request({"type": _IPCRequestType.GET_ALL_SLOTS.value})

    def get_active_slot(self) -> Dict[str, Any]:
        """Get the most recently committed slot."""
        return self._request({"type": _IPCRequestType.GET_ACTIVE_SLOT.value})

    def get_next_write_slot(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Get the next slot for writing in ping-pong pattern."""
        if pid is None:
            pid = os.getpid()
        return self._request({
            "type": _IPCRequestType.GET_NEXT_WRITE_SLOT.value,
            "pid": pid,
        })

    def write_raw_slot_data(
        self,
        slot_id: int,
        data: bytes,
        *,
        iteration: int,
        data_size: int,
        source_rank: int,
        checksum_hex: str,
        pid: Optional[int] = None,
    ) -> Dict[str, Any]:
        if pid is None:
            pid = os.getpid()
        request = {
            "type": _IPCRequestType.WRITE_RAW_SLOT_DATA.value,
            "slot_id": slot_id,
            "pid": pid,
            "iteration": iteration,
            "data_size": data_size,
            "source_rank": source_rank,
            "checksum_hex": checksum_hex,
        }
        if self._sock is None:
            raise EGMError("Not connected to EGM manager")
        _send_msg(self._sock, json.dumps(request).encode("utf-8"))
        _send_msg(self._sock, data)
        raw = _recv_msg(self._sock)
        if raw is None:
            raise EGMError("Connection lost to EGM manager")
        return json.loads(raw.decode("utf-8"))

    def read_raw_slot_data(self, slot_id: int) -> Optional[bytes]:
        if self._sock is None:
            raise EGMError("Not connected to EGM manager")
        request = {
            "type": _IPCRequestType.READ_RAW_SLOT_DATA.value,
            "slot_id": slot_id,
        }
        _send_msg(self._sock, json.dumps(request).encode("utf-8"))
        raw = _recv_msg(self._sock)
        if raw is None:
            raise EGMError("Connection lost to EGM manager")
        response = json.loads(raw.decode("utf-8"))
        if response.get("status") != "ok":
            return None
        return _recv_msg(self._sock)

    def update_slot_metadata(
        self,
        slot_id: int,
        *,
        iteration: int,
        data_size: int,
        total_bytes: int,
        source_rank: int,
        checksum_hex: str,
        pid: Optional[int] = None,
    ) -> Dict[str, Any]:
        if pid is None:
            pid = os.getpid()
        return self._request(
            {
                "type": _IPCRequestType.UPDATE_SLOT_METADATA.value,
                "slot_id": slot_id,
                "pid": pid,
                "iteration": iteration,
                "data_size": data_size,
                "total_bytes": total_bytes,
                "source_rank": source_rank,
                "checksum_hex": checksum_hex,
            }
        )

    def request_shutdown(self) -> Dict[str, Any]:
        """Request the manager to shut down."""
        return self._request({"type": _IPCRequestType.SHUTDOWN.value})

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


def run_egm_daemon():
    """Entry point for running the EGM Manager as a standalone daemon process.

    Usage:
        python -m megatron.core.egm.egm_manager \\
            --pool-size-gb 16.0 \\
            --numa-node-id 0 \\
            --num-slots 2 \\
            --device-id 0 \\
            --socket-path /tmp/megatron_egm_manager.sock

    This daemon must be started BEFORE the training processes.
    It holds the EGM memory via CUDA VMM API, ensuring checkpoint
    data survives training process crashes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="EGM Manager Daemon — holds EGM checkpoint memory "
        "independent of training processes"
    )
    parser.add_argument(
        "--pool-size-gb", type=float, default=16.0,
        help="EGM memory pool size per GPU in GB (default: 16.0)"
    )
    parser.add_argument(
        "--numa-node-id", type=int, default=0,
        help="NUMA node ID for EGM allocation (Grace CPU on GB200)"
    )
    parser.add_argument(
        "--num-slots", type=int, default=2,
        help="Number of checkpoint slots for double-buffering (default: 2)"
    )
    parser.add_argument(
        "--device-id", type=int, default=0,
        help="CUDA device ID for memory access (default: 0)"
    )
    parser.add_argument(
        "--socket-path", type=str, default=_DEFAULT_SOCKET_PATH,
        help=f"Unix domain socket path (default: {_DEFAULT_SOCKET_PATH})"
    )
    parser.add_argument(
        "--no-fabric-handle", action="store_true",
        help="Disable fabric handle export (for single-node testing)"
    )

    args = parser.parse_args()

    pool_size_bytes = int(args.pool_size_gb * 1024 ** 3)
    config = EGMConfig(
        pool_size_bytes=pool_size_bytes,
        numa_node_id=args.numa_node_id,
        num_slots=args.num_slots,
        device_id=args.device_id,
        socket_path=args.socket_path,
        use_fabric_handle=not args.no_fabric_handle,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [EGM-DAEMON] %(levelname)s %(message)s",
    )
    logger.info(
        f"Starting EGM daemon: pool={args.pool_size_gb}GB, "
        f"numa={args.numa_node_id}, slots={args.num_slots}, "
        f"device={args.device_id}, socket={args.socket_path}"
    )

    server = EGMManagerServer(config)
    server.run_forever()


if __name__ == "__main__":
    run_egm_daemon()
