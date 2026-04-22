#!/usr/bin/env python3
# Copyright (c) 2026.
#
# Standalone bandwidth microbenchmark for GB200/NVL72 memory paths.
#
# This script does NOT depend on Megatron-LM code.
#
# It measures three common paths:
# - EGM (HOST_NUMA pinned via cuMemCreate) GPU<->EGM using cuMemcpyDtoDAsync.
# - Host pinned (cuMemAllocHost) GPU<->pinned using cuMemcpyHtoDAsync / cuMemcpyDtoHAsync.
# - Host pageable (malloc/bytearray) GPU<->pageable using cuMemcpyHtoDAsync / cuMemcpyDtoHAsync.
#
# Notes:
# - "write" means GPU -> host-like target (EGM VA or host memory).
# - "read"  means host-like source -> GPU.
# - Pageable copies may involve an internal staging step and can be much slower.
#
# Run on an NVL72 / GB200 environment (NOT on a laptop):
#   python scripts/egm_bandwidth_bench.py --device 0 --numa-node 0 --size-gb 1 --iters 20 --direction both
#
# Notes:
# - "write" means GPU -> EGM (device buffer copied into EGM VA).
# - "read"  means EGM -> GPU.
# - The measured bandwidth is the copy engine throughput for these operations.

from __future__ import annotations

import argparse
import math
import sys
from typing import Any, Tuple, Optional


def _import_cuda_driver():
    try:
        import cuda.bindings.driver as cuda_driver  # type: ignore

        return cuda_driver, True
    except Exception:
        try:
            import cuda.cuda as cuda_driver  # type: ignore

            return cuda_driver, True
        except Exception:
            return None, False


def _normalize_err(err: Any) -> Any:
    if isinstance(err, tuple):
        if len(err) == 0:
            return None
        return err[0]
    return err


def _check(err: Any, cuda_driver: Any, msg: str) -> None:
    if _normalize_err(err) != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{msg}: {err}")


def _coerce_ptr(value: Any) -> int:
    if isinstance(value, int):
        return value
    raw = getattr(value, "value", None)
    if isinstance(raw, int):
        return raw
    return int(value)


def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _create_stream(cuda_driver: Any) -> Any:
    err, stream = cuda_driver.cuStreamCreate(0)
    _check(err, cuda_driver, "cuStreamCreate failed")
    return stream


def _create_event(cuda_driver: Any) -> Any:
    err, ev = cuda_driver.cuEventCreate(0)
    _check(err, cuda_driver, "cuEventCreate failed")
    return ev


def _event_elapsed_ms(cuda_driver: Any, start: Any, end: Any) -> float:
    err, ms = cuda_driver.cuEventElapsedTime(start, end)
    _check(err, cuda_driver, "cuEventElapsedTime failed")
    return float(ms)


def _alloc_device_buffer(cuda_driver: Any, nbytes: int) -> int:
    err, dptr = cuda_driver.cuMemAlloc(nbytes)
    _check(err, cuda_driver, "cuMemAlloc failed")
    return _coerce_ptr(dptr)


def _free_device_buffer(cuda_driver: Any, dptr: int) -> None:
    err = cuda_driver.cuMemFree(dptr)
    _check(err, cuda_driver, "cuMemFree failed")


def _alloc_egm_region(
    cuda_driver: Any, *, device_id: int, numa_node_id: int, size_bytes: int
) -> Tuple[Any, int, int]:
    """
    Returns (mem_handle, va_addr, aligned_size_bytes).
    The returned va_addr is a CUdeviceptr-sized VA that is also CPU-accessible in this process
    after cuMemSetAccess adds HOST_NUMA RW access.
    """
    prop = cuda_driver.CUmemAllocationProp()
    prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = int(numa_node_id)
    # Prefer FABRIC handles on NVL72 (multi-node). Not required for this benchmark.
    prop.requestedHandleTypes = cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC

    err, gran = cuda_driver.cuMemGetAllocationGranularity(
        prop,
        cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    )
    _check(err, cuda_driver, "cuMemGetAllocationGranularity failed")
    gran = int(gran)
    aligned_size = _align_up(int(size_bytes), max(gran, 2 * 1024 * 1024))

    err, mem_handle = cuda_driver.cuMemCreate(aligned_size, prop, 0)
    _check(err, cuda_driver, "cuMemCreate failed")

    err, va_addr = cuda_driver.cuMemAddressReserve(aligned_size, 0, 0, 0)
    _check(err, cuda_driver, "cuMemAddressReserve failed")
    va_addr_i = _coerce_ptr(va_addr)

    err = cuda_driver.cuMemMap(va_addr_i, aligned_size, 0, mem_handle, 0)
    _check(err, cuda_driver, "cuMemMap failed")

    # Grant RW access from both the GPU device and the Grace NUMA "host" location.
    host_access = cuda_driver.CUmemAccessDesc()
    host_access.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    host_access.location.id = int(numa_node_id)
    host_access.flags = cuda_driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    dev_access = cuda_driver.CUmemAccessDesc()
    dev_access.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    dev_access.location.id = int(device_id)
    dev_access.flags = cuda_driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    access_descs = [host_access, dev_access]
    err = cuda_driver.cuMemSetAccess(va_addr_i, aligned_size, access_descs, len(access_descs))
    _check(err, cuda_driver, "cuMemSetAccess failed")

    return mem_handle, va_addr_i, aligned_size


def _free_egm_region(cuda_driver: Any, mem_handle: Any, va_addr: int, size_bytes: int) -> None:
    try:
        err = cuda_driver.cuMemUnmap(va_addr, size_bytes)
        _check(err, cuda_driver, "cuMemUnmap failed")
    finally:
        err = cuda_driver.cuMemAddressFree(va_addr, size_bytes)
        _check(err, cuda_driver, "cuMemAddressFree failed")
        err = cuda_driver.cuMemRelease(mem_handle)
        _check(err, cuda_driver, "cuMemRelease failed")


def _bench_copy(
    cuda_driver: Any,
    *,
    stream: Any,
    src: int,
    dst: int,
    nbytes: int,
    iters: int,
    warmup: int,
) -> float:
    # Warmup
    for _ in range(max(0, warmup)):
        err = cuda_driver.cuMemcpyDtoDAsync(dst, src, nbytes, stream)
        _check(err, cuda_driver, "cuMemcpyDtoDAsync failed (warmup)")
    _check(cuda_driver.cuStreamSynchronize(stream), cuda_driver, "cuStreamSynchronize failed (warmup)")

    start = _create_event(cuda_driver)
    end = _create_event(cuda_driver)
    _check(cuda_driver.cuEventRecord(start, stream), cuda_driver, "cuEventRecord(start) failed")
    for _ in range(iters):
        err = cuda_driver.cuMemcpyDtoDAsync(dst, src, nbytes, stream)
        _check(err, cuda_driver, "cuMemcpyDtoDAsync failed")
    _check(cuda_driver.cuEventRecord(end, stream), cuda_driver, "cuEventRecord(end) failed")
    _check(cuda_driver.cuEventSynchronize(end), cuda_driver, "cuEventSynchronize failed")
    ms = _event_elapsed_ms(cuda_driver, start, end)
    return ms / 1000.0


def _alloc_host_pinned(cuda_driver: Any, nbytes: int) -> int:
    err, hptr = cuda_driver.cuMemAllocHost(nbytes)
    _check(err, cuda_driver, "cuMemAllocHost failed")
    return _coerce_ptr(hptr)


def _free_host_pinned(cuda_driver: Any, hptr: int) -> None:
    err = cuda_driver.cuMemFreeHost(hptr)
    _check(err, cuda_driver, "cuMemFreeHost failed")


def _host_memcpy_htod(cuda_driver: Any, *, stream: Any, src_host_ptr: int, dst_dev_ptr: int, nbytes: int) -> None:
    err = cuda_driver.cuMemcpyHtoDAsync(dst_dev_ptr, src_host_ptr, nbytes, stream)
    _check(err, cuda_driver, "cuMemcpyHtoDAsync failed")


def _host_memcpy_dtoh(cuda_driver: Any, *, stream: Any, src_dev_ptr: int, dst_host_ptr: int, nbytes: int) -> None:
    err = cuda_driver.cuMemcpyDtoHAsync(dst_host_ptr, src_dev_ptr, nbytes, stream)
    _check(err, cuda_driver, "cuMemcpyDtoHAsync failed")


def _bench_host_copy(
    cuda_driver: Any,
    *,
    stream: Any,
    direction: str,
    host_ptr: int,
    dev_ptr: int,
    nbytes: int,
    iters: int,
    warmup: int,
) -> float:
    # Warmup
    for _ in range(max(0, warmup)):
        if direction == "h2d":
            _host_memcpy_htod(cuda_driver, stream=stream, src_host_ptr=host_ptr, dst_dev_ptr=dev_ptr, nbytes=nbytes)
        else:
            _host_memcpy_dtoh(cuda_driver, stream=stream, src_dev_ptr=dev_ptr, dst_host_ptr=host_ptr, nbytes=nbytes)
    _check(cuda_driver.cuStreamSynchronize(stream), cuda_driver, "cuStreamSynchronize failed (warmup)")

    start = _create_event(cuda_driver)
    end = _create_event(cuda_driver)
    _check(cuda_driver.cuEventRecord(start, stream), cuda_driver, "cuEventRecord(start) failed")
    for _ in range(iters):
        if direction == "h2d":
            _host_memcpy_htod(cuda_driver, stream=stream, src_host_ptr=host_ptr, dst_dev_ptr=dev_ptr, nbytes=nbytes)
        else:
            _host_memcpy_dtoh(cuda_driver, stream=stream, src_dev_ptr=dev_ptr, dst_host_ptr=host_ptr, nbytes=nbytes)
    _check(cuda_driver.cuEventRecord(end, stream), cuda_driver, "cuEventRecord(end) failed")
    _check(cuda_driver.cuEventSynchronize(end), cuda_driver, "cuEventSynchronize failed")
    ms = _event_elapsed_ms(cuda_driver, start, end)
    return ms / 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="EGM bandwidth benchmark (GB200/NVL72).")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (default: 0)")
    parser.add_argument("--numa-node", type=int, default=0, help="Grace NUMA node id (default: 0)")
    parser.add_argument("--size-gb", type=float, default=1.0, help="Copy size in GiB (default: 1.0)")
    parser.add_argument("--iters", type=int, default=20, help="Iterations (default: 20)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument(
        "--direction",
        choices=("write", "read", "both"),
        default="both",
        help="Direction: write=GPU->EGM, read=EGM->GPU, both=both (default: both)",
    )
    parser.add_argument(
        "--modes",
        default="egm,pinned,pageable",
        help="Comma-separated modes to run: egm,pinned,pageable (default: egm,pinned,pageable)",
    )
    args = parser.parse_args()

    cuda_driver, ok = _import_cuda_driver()
    if not ok or cuda_driver is None:
        print("ERROR: cuda-python not available. Install 'cuda-python' / 'nvidia-cuda-python'.", file=sys.stderr)
        return 2

    _check(cuda_driver.cuInit(0), cuda_driver, "cuInit failed")
    err, dev = cuda_driver.cuDeviceGet(int(args.device))
    _check(err, cuda_driver, "cuDeviceGet failed")
    err, ctx = cuda_driver.cuDevicePrimaryCtxRetain(dev)
    _check(err, cuda_driver, "cuDevicePrimaryCtxRetain failed")
    # cuDevicePrimaryCtxRetain does NOT make the context current. Many driver calls
    # (e.g., cuStreamCreate) require a current context, otherwise they fail with
    # CUDA_ERROR_INVALID_CONTEXT (often reported as error code 201).
    err = cuda_driver.cuCtxSetCurrent(ctx)
    _check(err, cuda_driver, "cuCtxSetCurrent failed")

    size_bytes = int(args.size_gb * (1024 ** 3))
    if size_bytes <= 0:
        raise ValueError("--size-gb must be > 0")

    stream = _create_stream(cuda_driver)

    mem_handle: Optional[Any] = None
    egm_va = 0
    egm_size = 0
    d_a = 0
    d_b = 0
    pinned_h = 0
    pageable_buf: Optional[bytearray] = None

    try:
        d_a = _alloc_device_buffer(cuda_driver, size_bytes)
        d_b = _alloc_device_buffer(cuda_driver, size_bytes)

        # Touch buffers (optional). Keep it simple and only measure memcpy.
        _check(cuda_driver.cuStreamSynchronize(stream), cuda_driver, "cuStreamSynchronize failed")

        def _report(label: str, seconds: float) -> None:
            total = float(size_bytes) * float(args.iters)
            gib = total / (1024 ** 3)
            bw = gib / seconds if seconds > 0 else math.inf
            print(f"{label}: size={args.size_gb:.3f} GiB, iters={args.iters}, time={seconds:.6f} s, bw={bw:.3f} GiB/s")

        modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]

        if "egm" in modes:
            mem_handle, egm_va, egm_size = _alloc_egm_region(
                cuda_driver, device_id=int(args.device), numa_node_id=int(args.numa_node), size_bytes=size_bytes
            )
            if args.direction in ("write", "both"):
                t = _bench_copy(
                    cuda_driver,
                    stream=stream,
                    src=d_a,
                    dst=egm_va,
                    nbytes=size_bytes,
                    iters=int(args.iters),
                    warmup=int(args.warmup),
                )
                _report("GPU->EGM (write)", t)

            if args.direction in ("read", "both"):
                t = _bench_copy(
                    cuda_driver,
                    stream=stream,
                    src=egm_va,
                    dst=d_b,
                    nbytes=size_bytes,
                    iters=int(args.iters),
                    warmup=int(args.warmup),
                )
                _report("EGM->GPU (read)", t)

        if "pinned" in modes:
            pinned_h = _alloc_host_pinned(cuda_driver, size_bytes)
            if args.direction in ("write", "both"):
                # GPU -> pinned host (DtoH)
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="d2h",
                    host_ptr=pinned_h,
                    dev_ptr=d_a,
                    nbytes=size_bytes,
                    iters=int(args.iters),
                    warmup=int(args.warmup),
                )
                _report("GPU->HostPinned (DtoH/write)", t)
            if args.direction in ("read", "both"):
                # pinned host -> GPU (HtoD)
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="h2d",
                    host_ptr=pinned_h,
                    dev_ptr=d_b,
                    nbytes=size_bytes,
                    iters=int(args.iters),
                    warmup=int(args.warmup),
                )
                _report("HostPinned->GPU (HtoD/read)", t)

        if "pageable" in modes:
            # Pageable memory: use a Python bytearray as the backing store and pass its pointer.
            # This is expected to be slower due to staging/pinning overhead.
            pageable_buf = bytearray(size_bytes)
            import ctypes as _ctypes

            pageable_ptr = _ctypes.addressof(_ctypes.c_char.from_buffer(pageable_buf))
            if args.direction in ("write", "both"):
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="d2h",
                    host_ptr=pageable_ptr,
                    dev_ptr=d_a,
                    nbytes=size_bytes,
                    iters=int(args.iters),
                    warmup=max(1, int(args.warmup // 2)),
                )
                _report("GPU->HostPageable (DtoH/write)", t)
            if args.direction in ("read", "both"):
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="h2d",
                    host_ptr=pageable_ptr,
                    dev_ptr=d_b,
                    nbytes=size_bytes,
                    iters=int(args.iters),
                    warmup=max(1, int(args.warmup // 2)),
                )
                _report("HostPageable->GPU (HtoD/read)", t)

        return 0
    finally:
        # Best-effort cleanup
        try:
            if d_a:
                _free_device_buffer(cuda_driver, d_a)
        except Exception:
            pass
        try:
            if d_b:
                _free_device_buffer(cuda_driver, d_b)
        except Exception:
            pass
        try:
            if pinned_h:
                _free_host_pinned(cuda_driver, pinned_h)
        except Exception:
            pass
        try:
            if mem_handle is not None and egm_va and egm_size:
                _free_egm_region(cuda_driver, mem_handle, egm_va, egm_size)
        except Exception:
            pass
        try:
            cuda_driver.cuDevicePrimaryCtxRelease(dev)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
