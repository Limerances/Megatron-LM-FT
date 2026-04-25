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
# Usage examples (run on GB200 / NVL72, NOT on a laptop):
#
# 1. List visible GPU -> HOST_NUMA topology:
#    python scripts/egm_bandwidth_bench.py --list-topology
#
# 2. Single-GPU baseline:
#    python scripts/egm_bandwidth_bench.py \
#      --preset single --device 0 --modes egm --direction both --size-gb 30 --iters 20
#
# 3. One GB200 aggregate benchmark (typically 2 B200 under the first HOST_NUMA group):
#    python scripts/egm_bandwidth_bench.py \
#      --preset gb200 --modes egm --direction both --size-gb 30 --iters 20
#
# 4. Two GB200 aggregate benchmark (typically 4 B200 across the first two HOST_NUMA groups):
#    python scripts/egm_bandwidth_bench.py \
#      --preset two-gb200 --modes egm --direction both --size-gb 30 --iters 20
#
# 5. Explicit device selection:
#    python scripts/egm_bandwidth_bench.py \
#      --devices 0,1,2,3 --modes egm --direction both --size-gb 30 --iters 20
#
# 6. Compare EGM / pinned / pageable paths on a single GPU:
#    python scripts/egm_bandwidth_bench.py \
#      --preset single --device 0 --modes egm,pinned,pageable --direction both --size-gb 30 --iters 20
#
# Notes:
# - "write" means GPU -> EGM (device buffer copied into EGM VA).
# - "read"  means EGM -> GPU.
# - The measured bandwidth is the copy engine throughput for these operations.

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import traceback
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
    # Use CU_STREAM_NON_BLOCKING (0x1) so streams do NOT implicitly synchronize
    # with the legacy default (NULL) stream. This is required for the
    # multi-stream DtoD fan-out path to actually let concurrent launches map to
    # different copy engines; the default flag=0 produces a "blocking" stream
    # that would serialize on the NULL stream.
    err, stream = cuda_driver.cuStreamCreate(0x1)
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


def _bench_copy_multi_stream(
    cuda_driver: Any,
    *,
    streams: list,
    timing_stream: Any,
    src: int,
    dst: int,
    nbytes: int,
    iters: int,
    warmup: int,
) -> float:
    """Benchmark a DtoD copy that is fanned across multiple streams.

    The buffer of ``nbytes`` is split into ``len(streams)`` equal shards,
    each shard is issued on its own stream so the driver can use separate
    copy engines concurrently. We time across a common stream by recording
    an event on each worker stream and making the timing stream wait on all
    of them, so the elapsed time reflects the makespan of the slowest
    engine, i.e. the achievable aggregate per-GPU bandwidth.
    """
    assert len(streams) >= 1
    n = len(streams)
    # 128-byte aligned shards; tail shard absorbs the remainder.
    shard = max(128, (nbytes // n) & ~127)

    def _issue_once() -> None:
        offset = 0
        for i, s in enumerate(streams):
            this_nbytes = shard if i < n - 1 else (nbytes - offset)
            if this_nbytes <= 0:
                continue
            err = cuda_driver.cuMemcpyDtoDAsync(dst + offset, src + offset, this_nbytes, s)
            _check(err, cuda_driver, "cuMemcpyDtoDAsync failed (multi-stream)")
            offset += this_nbytes

    # Warmup
    for _ in range(max(0, warmup)):
        _issue_once()
    for s in streams:
        _check(cuda_driver.cuStreamSynchronize(s), cuda_driver, "cuStreamSynchronize failed (warmup)")

    start = _create_event(cuda_driver)
    end = _create_event(cuda_driver)
    _check(cuda_driver.cuEventRecord(start, timing_stream), cuda_driver, "cuEventRecord(start) failed")
    # Fan-out: make worker streams wait on the timing stream's start.
    for s in streams:
        _check(cuda_driver.cuStreamWaitEvent(s, start, 0), cuda_driver, "cuStreamWaitEvent(start) failed")
    for _ in range(iters):
        _issue_once()
    # Fan-in: per-worker events joined into the timing stream.
    per_stream_events = []
    for s in streams:
        ev = _create_event(cuda_driver)
        _check(cuda_driver.cuEventRecord(ev, s), cuda_driver, "cuEventRecord(worker) failed")
        per_stream_events.append(ev)
    for ev in per_stream_events:
        _check(cuda_driver.cuStreamWaitEvent(timing_stream, ev, 0), cuda_driver, "cuStreamWaitEvent(join) failed")
    _check(cuda_driver.cuEventRecord(end, timing_stream), cuda_driver, "cuEventRecord(end) failed")
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


def _detect_host_numa_id(cuda_driver: Any, device_id: int) -> int:
    err, dev = cuda_driver.cuDeviceGet(int(device_id))
    _check(err, cuda_driver, f"cuDeviceGet failed for device {device_id}")
    err, numa = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
        dev,
    )
    _check(err, cuda_driver, f"cuDeviceGetAttribute(HOST_NUMA_ID) failed for device {device_id}")
    return int(numa)


def _detect_topology(cuda_driver: Any) -> list[dict[str, int]]:
    _check(cuda_driver.cuInit(0), cuda_driver, "cuInit failed")
    err, count = cuda_driver.cuDeviceGetCount()
    _check(err, cuda_driver, "cuDeviceGetCount failed")
    topology: list[dict[str, int]] = []
    for device_id in range(int(count)):
        topology.append(
            {
                "device": device_id,
                "numa": _detect_host_numa_id(cuda_driver, device_id),
            }
        )
    return topology


def _group_topology_by_numa(topology: list[dict[str, int]]) -> list[list[dict[str, int]]]:
    groups: dict[int, list[dict[str, int]]] = {}
    for entry in topology:
        groups.setdefault(int(entry["numa"]), []).append(entry)
    return [sorted(groups[numa], key=lambda x: int(x["device"])) for numa in sorted(groups.keys())]


def _parse_devices_arg(devices_arg: str) -> list[int]:
    if devices_arg.strip().lower() == "all":
        return []
    return [int(part.strip()) for part in devices_arg.split(",") if part.strip()]


def _resolve_selected_devices(args, topology: list[dict[str, int]]) -> list[dict[str, int]]:
    by_device = {int(entry["device"]): entry for entry in topology}
    if args.devices:
        explicit = _parse_devices_arg(args.devices)
        if len(explicit) == 0:
            return topology
        return [by_device[d] for d in explicit]

    if args.preset == "single":
        if int(args.device) not in by_device:
            raise ValueError(f"--device {args.device} is not visible in current topology")
        entry = dict(by_device[int(args.device)])
        if int(args.numa_node) >= 0:
            entry["numa"] = int(args.numa_node)
        return [entry]

    numa_groups = _group_topology_by_numa(topology)
    if args.preset == "gb200":
        if len(numa_groups) < 1:
            raise ValueError("No NUMA groups found")
        return list(numa_groups[0])
    if args.preset == "two-gb200":
        if len(numa_groups) < 2:
            raise ValueError("Less than two NUMA groups found; cannot benchmark two GB200 presets")
        return list(numa_groups[0]) + list(numa_groups[1])
    if args.preset == "all":
        return topology
    raise ValueError(f"Unsupported preset: {args.preset}")


def _format_result_line(result: dict[str, Any], prefix: str = "") -> str:
    return (
        f"{prefix}{result['label']}: device={result['device']}, numa={result['numa']}, "
        f"size={result['size_gib']:.3f} GiB, iters={result['iters']}, "
        f"time={result['seconds']:.6f} s, bw={result['bw_gib_s']:.3f} GiB/s"
    )


def _run_single_device_bench(
    *,
    device_id: int,
    numa_node_id: int,
    size_bytes: int,
    size_gib: float,
    iters: int,
    warmup: int,
    direction: str,
    modes: list[str],
    streams: int = 1,
    start_event: Optional[Any] = None,
    ready_queue: Optional[Any] = None,
) -> list[dict[str, Any]]:
    cuda_driver, ok = _import_cuda_driver()
    if not ok or cuda_driver is None:
        raise RuntimeError("cuda-python not available. Install 'cuda-python' / 'nvidia-cuda-python'.")

    _check(cuda_driver.cuInit(0), cuda_driver, "cuInit failed")
    err, dev = cuda_driver.cuDeviceGet(int(device_id))
    _check(err, cuda_driver, "cuDeviceGet failed")
    err, ctx = cuda_driver.cuDevicePrimaryCtxRetain(dev)
    _check(err, cuda_driver, "cuDevicePrimaryCtxRetain failed")
    err = cuda_driver.cuCtxSetCurrent(ctx)
    _check(err, cuda_driver, "cuCtxSetCurrent failed")

    stream = _create_stream(cuda_driver)
    extra_streams: list[Any] = []
    n_multi_streams = max(1, int(streams))
    if n_multi_streams > 1:
        # Allocate extra streams for the multi-stream fan-out path. Note these
        # are non-blocking streams (see _create_stream) so they do not
        # synchronize with the NULL stream.
        for _ in range(n_multi_streams):
            extra_streams.append(_create_stream(cuda_driver))
    mem_handle: Optional[Any] = None
    egm_va = 0
    egm_size = 0
    d_a = 0
    d_b = 0
    pinned_h = 0
    pageable_buf: Optional[bytearray] = None
    results: list[dict[str, Any]] = []

    try:
        d_a = _alloc_device_buffer(cuda_driver, size_bytes)
        d_b = _alloc_device_buffer(cuda_driver, size_bytes)

        if "egm" in modes:
            mem_handle, egm_va, egm_size = _alloc_egm_region(
                cuda_driver, device_id=int(device_id), numa_node_id=int(numa_node_id), size_bytes=size_bytes
            )
        if "pinned" in modes:
            pinned_h = _alloc_host_pinned(cuda_driver, size_bytes)
        if "pageable" in modes:
            pageable_buf = bytearray(size_bytes)

        _check(cuda_driver.cuStreamSynchronize(stream), cuda_driver, "cuStreamSynchronize failed")
        if ready_queue is not None:
            ready_queue.put({"device": int(device_id), "numa": int(numa_node_id), "status": "ready"})
        if start_event is not None:
            start_event.wait()

        def _append_result(label: str, seconds: float) -> None:
            total_gib = (float(size_bytes) * float(iters)) / (1024 ** 3)
            bw = total_gib / seconds if seconds > 0 else math.inf
            results.append(
                {
                    "label": label,
                    "device": int(device_id),
                    "numa": int(numa_node_id),
                    "size_gib": float(size_gib),
                    "iters": int(iters),
                    "seconds": float(seconds),
                    "total_gib": total_gib,
                    "bw_gib_s": bw,
                }
            )

        if "egm" in modes:
            if direction in ("write", "both"):
                t = _bench_copy(
                    cuda_driver,
                    stream=stream,
                    src=d_a,
                    dst=egm_va,
                    nbytes=size_bytes,
                    iters=int(iters),
                    warmup=int(warmup),
                )
                _append_result("GPU->EGM (write)", t)
                if n_multi_streams > 1:
                    t_ms = _bench_copy_multi_stream(
                        cuda_driver,
                        streams=extra_streams,
                        timing_stream=stream,
                        src=d_a,
                        dst=egm_va,
                        nbytes=size_bytes,
                        iters=int(iters),
                        warmup=int(warmup),
                    )
                    _append_result(f"GPU->EGM (write, x{n_multi_streams} streams)", t_ms)

            if direction in ("read", "both"):
                t = _bench_copy(
                    cuda_driver,
                    stream=stream,
                    src=egm_va,
                    dst=d_b,
                    nbytes=size_bytes,
                    iters=int(iters),
                    warmup=int(warmup),
                )
                _append_result("EGM->GPU (read)", t)
                if n_multi_streams > 1:
                    t_ms = _bench_copy_multi_stream(
                        cuda_driver,
                        streams=extra_streams,
                        timing_stream=stream,
                        src=egm_va,
                        dst=d_b,
                        nbytes=size_bytes,
                        iters=int(iters),
                        warmup=int(warmup),
                    )
                    _append_result(f"EGM->GPU (read, x{n_multi_streams} streams)", t_ms)

        if "pinned" in modes:
            if direction in ("write", "both"):
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="d2h",
                    host_ptr=pinned_h,
                    dev_ptr=d_a,
                    nbytes=size_bytes,
                    iters=int(iters),
                    warmup=int(warmup),
                )
                _append_result("GPU->HostPinned (DtoH/write)", t)
            if direction in ("read", "both"):
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="h2d",
                    host_ptr=pinned_h,
                    dev_ptr=d_b,
                    nbytes=size_bytes,
                    iters=int(iters),
                    warmup=int(warmup),
                )
                _append_result("HostPinned->GPU (HtoD/read)", t)

        if "pageable" in modes:
            import ctypes as _ctypes

            assert pageable_buf is not None
            pageable_ptr = _ctypes.addressof(_ctypes.c_char.from_buffer(pageable_buf))
            if direction in ("write", "both"):
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="d2h",
                    host_ptr=pageable_ptr,
                    dev_ptr=d_a,
                    nbytes=size_bytes,
                    iters=int(iters),
                    warmup=max(1, int(warmup // 2)),
                )
                _append_result("GPU->HostPageable (DtoH/write)", t)
            if direction in ("read", "both"):
                t = _bench_host_copy(
                    cuda_driver,
                    stream=stream,
                    direction="h2d",
                    host_ptr=pageable_ptr,
                    dev_ptr=d_b,
                    nbytes=size_bytes,
                    iters=int(iters),
                    warmup=max(1, int(warmup // 2)),
                )
                _append_result("HostPageable->GPU (HtoD/read)", t)

        return results
    finally:
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
            for s in extra_streams:
                cuda_driver.cuStreamDestroy(s)
        except Exception:
            pass
        try:
            cuda_driver.cuDevicePrimaryCtxRelease(dev)
        except Exception:
            pass


def _worker_entry(
    device_id: int,
    numa_node_id: int,
    size_bytes: int,
    size_gib: float,
    iters: int,
    warmup: int,
    direction: str,
    modes: list[str],
    streams: int,
    start_event: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    try:
        results = _run_single_device_bench(
            device_id=device_id,
            numa_node_id=numa_node_id,
            size_bytes=size_bytes,
            size_gib=size_gib,
            iters=iters,
            warmup=warmup,
            direction=direction,
            modes=modes,
            streams=streams,
            start_event=start_event,
            ready_queue=ready_queue,
        )
        result_queue.put(
            {
                "device": int(device_id),
                "numa": int(numa_node_id),
                "results": results,
            }
        )
    except Exception:
        result_queue.put(
            {
                "device": int(device_id),
                "numa": int(numa_node_id),
                "error": traceback.format_exc(),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="EGM bandwidth benchmark (GB200/NVL72).")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (default: 0)")
    parser.add_argument("--numa-node", type=int, default=-1, help="Grace NUMA node id for single-device mode; -1 means auto-detect")
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
    parser.add_argument(
        "--preset",
        choices=("single", "gb200", "two-gb200", "all"),
        default="single",
        help="Device selection preset: single=one GPU, gb200=first HOST_NUMA group, two-gb200=first two HOST_NUMA groups, all=all visible GPUs",
    )
    parser.add_argument(
        "--devices",
        default="",
        help="Explicit device list like '0,1' or '0,1,2,3'; overrides --preset. Use 'all' for all visible GPUs.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run selected devices sequentially instead of concurrently. Default is concurrent for multi-device selections.",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=1,
        help=(
            "Number of concurrent CUDA streams per GPU to fan DtoD copies across. "
            "A single cuMemcpyDtoDAsync on NULL/one stream uses one copy engine "
            "and plateaus around ~180 GiB/s on GB200; using >=4 streams lets the "
            "driver schedule multiple copy engines and approach the ~225 GB/s "
            "C2C link limit per B200. Default 1 (legacy behaviour)."
        ),
    )
    parser.add_argument(
        "--list-topology",
        action="store_true",
        help="Print device -> HOST_NUMA topology and exit.",
    )
    args = parser.parse_args()

    cuda_driver, ok = _import_cuda_driver()
    if not ok or cuda_driver is None:
        print("ERROR: cuda-python not available. Install 'cuda-python' / 'nvidia-cuda-python'.", file=sys.stderr)
        return 2

    topology = _detect_topology(cuda_driver)
    if args.list_topology:
        print("Detected GPU -> HOST_NUMA topology:")
        for entry in topology:
            print(f"  device={entry['device']}, host_numa={entry['numa']}")
        return 0

    size_bytes = int(args.size_gb * (1024 ** 3))
    if size_bytes <= 0:
        raise ValueError("--size-gb must be > 0")
    modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    selected = _resolve_selected_devices(args, topology)
    if len(selected) == 0:
        raise ValueError("No devices selected")

    print("Detected GPU -> HOST_NUMA topology:")
    for entry in topology:
        print(f"  device={entry['device']}, host_numa={entry['numa']}")
    print("Selected devices:")
    for entry in selected:
        print(f"  device={entry['device']}, host_numa={entry['numa']}")

    if len(selected) == 1 or args.sequential:
        all_results: list[dict[str, Any]] = []
        for entry in selected:
            device_id = int(entry["device"])
            numa_node_id = int(entry["numa"])
            results = _run_single_device_bench(
                device_id=device_id,
                numa_node_id=numa_node_id,
                size_bytes=size_bytes,
                size_gib=float(args.size_gb),
                iters=int(args.iters),
                warmup=int(args.warmup),
                direction=args.direction,
                modes=modes,
                streams=int(args.streams),
            )
            for result in results:
                print(_format_result_line(result))
            all_results.extend(results)
        if len(selected) > 1:
            print("Sequential mode does not report aggregate concurrency bandwidth.")
        return 0

    ctx = mp.get_context("spawn")
    start_event = ctx.Event()
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = []
    try:
        for entry in selected:
            device_id = int(entry["device"])
            numa_node_id = int(entry["numa"])
            p = ctx.Process(
                target=_worker_entry,
                args=(
                    device_id,
                    numa_node_id,
                    size_bytes,
                    float(args.size_gb),
                    int(args.iters),
                    int(args.warmup),
                    args.direction,
                    modes,
                    int(args.streams),
                    start_event,
                    ready_queue,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        for _ in processes:
            ready_msg = ready_queue.get()
            if ready_msg.get("status") != "ready":
                raise RuntimeError(f"Worker failed before benchmark start: {ready_msg}")

        start_event.set()

        device_results: list[dict[str, Any]] = []
        for _ in processes:
            msg = result_queue.get()
            if "error" in msg:
                raise RuntimeError(
                    f"Worker failed for device={msg.get('device')} numa={msg.get('numa')}:\n{msg['error']}"
                )
            device_results.extend(msg["results"])

        for result in sorted(device_results, key=lambda r: (r["label"], r["device"])):
            print(_format_result_line(result))

        labels = sorted({result["label"] for result in device_results})
        print("Aggregate concurrent bandwidth:")
        for label in labels:
            label_results = [result for result in device_results if result["label"] == label]
            total_gib = sum(float(result["total_gib"]) for result in label_results)
            wall_s = max(float(result["seconds"]) for result in label_results)
            agg_bw = total_gib / wall_s if wall_s > 0 else math.inf
            devices_str = ",".join(str(result["device"]) for result in sorted(label_results, key=lambda r: r["device"]))
            print(
                f"AGG {label}: devices={devices_str}, total={total_gib:.3f} GiB, "
                f"wall={wall_s:.6f} s, bw={agg_bw:.3f} GiB/s"
            )

        per_numa: dict[int, list[dict[str, Any]]] = {}
        for result in device_results:
            per_numa.setdefault(int(result["numa"]), []).append(result)
        if len(per_numa) > 1:
            print("Per-NUMA aggregate concurrent bandwidth:")
            for numa in sorted(per_numa.keys()):
                for label in labels:
                    label_results = [result for result in per_numa[numa] if result["label"] == label]
                    if not label_results:
                        continue
                    total_gib = sum(float(result["total_gib"]) for result in label_results)
                    wall_s = max(float(result["seconds"]) for result in label_results)
                    agg_bw = total_gib / wall_s if wall_s > 0 else math.inf
                    devices_str = ",".join(str(result["device"]) for result in sorted(label_results, key=lambda r: r["device"]))
                    print(
                        f"NUMA {numa} | {label}: devices={devices_str}, total={total_gib:.3f} GiB, "
                        f"wall={wall_s:.6f} s, bw={agg_bw:.3f} GiB/s"
                    )
        return 0
    finally:
        for p in processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    sys.exit(main())
