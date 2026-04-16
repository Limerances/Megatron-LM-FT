# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.cuda.memory import CUDAPluggableAllocator
from torch.utils.cpp_extension import CUDA_HOME, load_inline

from megatron.core.utils import is_torch_min_version, log_single_rank

try:
    if is_torch_min_version("2.8.0"):
        from torch.cuda.memory import MemPool
    else:
        from torch.cuda import MemPool
    _has_mem_pool = True
except ImportError:
    _has_mem_pool = False

logger = logging.getLogger(__name__)

_allocator = None
_mod = None
_so_path = None


def _build_egm_allocator():
    global _allocator, _mod, _so_path

    if _allocator is not None:
        return

    egm_allocator_source = r"""
    #include <cuda.h>
    #include <cuda_runtime_api.h>
    #include <cstddef>
    #include <cstdio>
    #include <cstring>
    #include <unordered_map>
    #include <mutex>

    #define EXPORT extern "C"

    #define CU_CHECK(cmd) do { \
        CUresult r = cmd; \
        if (r != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(r, &errStr); \
            fprintf(stderr, "EGM Allocator CUDA Driver error %s:%d '%s'\n", \
                __FILE__, __LINE__, errStr ? errStr : "unknown"); \
            return nullptr; \
        } \
    } while(0)

    #define CU_CHECK_VOID(cmd) do { \
        CUresult r = cmd; \
        if (r != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(r, &errStr); \
            fprintf(stderr, "EGM Allocator CUDA Driver error (free) %s:%d '%s'\n", \
                __FILE__, __LINE__, errStr ? errStr : "unknown"); \
        } \
    } while(0)

    struct EGMAllocation {
        CUmemGenericAllocationHandle handle;
        size_t aligned_size;
    };

    static std::unordered_map<void*, EGMAllocation> g_alloc_map;
    static std::mutex g_alloc_mutex;
    static int g_numa_node_id = 0;
    static bool g_multinode = false;

    EXPORT void egm_set_numa_node(int numa_node_id) {
        g_numa_node_id = numa_node_id;
    }

    EXPORT void egm_set_multinode(int multinode) {
        g_multinode = (multinode != 0);
    }

    EXPORT void* egm_malloc(size_t size, int device, void* stream) {
        (void)stream;

        if (size == 0) return nullptr;

        CUdevice cu_device;
        CU_CHECK(cuDeviceGet(&cu_device, device));

        size_t granularity = 0;
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = g_numa_node_id;

        if (g_multinode) {
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        }

        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                 CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, aligned_size, &prop, 0));

        CUdeviceptr dptr = 0;
        CU_CHECK(cuMemAddressReserve(&dptr, aligned_size, granularity, 0, 0));
        CU_CHECK(cuMemMap(dptr, aligned_size, 0, handle, 0));

        CUmemAccessDesc access_desc;
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = (int)cu_device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(dptr, aligned_size, &access_desc, 1));

        void* ptr = (void*)dptr;

        {
            std::lock_guard<std::mutex> lock(g_alloc_mutex);
            g_alloc_map[ptr] = {handle, aligned_size};
        }

        return ptr;
    }

    EXPORT void egm_free(void* ptr, size_t size, int device, void* stream) {
        (void)size;
        (void)device;
        (void)stream;

        if (!ptr) return;

        EGMAllocation alloc;
        {
            std::lock_guard<std::mutex> lock(g_alloc_mutex);
            auto it = g_alloc_map.find(ptr);
            if (it == g_alloc_map.end()) {
                fprintf(stderr, "EGM Allocator: attempted to free unknown pointer %p\n", ptr);
                return;
            }
            alloc = it->second;
            g_alloc_map.erase(it);
        }

        CUdeviceptr dptr = (CUdeviceptr)ptr;
        CU_CHECK_VOID(cuMemUnmap(dptr, alloc.aligned_size));
        CU_CHECK_VOID(cuMemAddressFree(dptr, alloc.aligned_size));
        CU_CHECK_VOID(cuMemRelease(alloc.handle));
    }

    EXPORT CUmemGenericAllocationHandle egm_get_handle(void* ptr) {
        std::lock_guard<std::mutex> lock(g_alloc_mutex);
        auto it = g_alloc_map.find(ptr);
        if (it == g_alloc_map.end()) {
            return 0;
        }
        return it->second.handle;
    }
    """

    module_dir = os.path.dirname(__file__)
    source_dir = os.path.join(module_dir, "build")
    os.makedirs(source_dir, exist_ok=True)

    _extra_ldflags = ["-lcuda", "-lcudart"]
    if CUDA_HOME:
        _cuda_lib = os.path.join(CUDA_HOME, "lib64")
        _cuda_stubs = os.path.join(CUDA_HOME, "lib64", "stubs")
        if os.path.isdir(_cuda_lib):
            _extra_ldflags = [f"-L{_cuda_lib}", f"-L{_cuda_stubs}", "-lcuda", "-lcudart"]

    _mod = load_inline(
        name="egm_allocator",
        cpp_sources=[egm_allocator_source],
        functions=[],
        with_cuda=True,
        extra_ldflags=_extra_ldflags,
        verbose=True,
        build_directory=source_dir,
    )

    _so_path = Path(_mod.__file__).as_posix()
    _cpa = CUDAPluggableAllocator(_so_path, "egm_malloc", "egm_free")
    _allocator = _cpa.allocator()


def create_egm_mem_pool(numa_node_id: int) -> "MemPool":
    _build_egm_allocator()

    import ctypes

    lib = ctypes.CDLL(_so_path)
    lib.egm_set_numa_node.argtypes = [ctypes.c_int]
    lib.egm_set_numa_node.restype = None
    lib.egm_set_numa_node(numa_node_id)

    assert _allocator is not None, "EGM allocator is not initialized"
    _pool = MemPool(allocator=_allocator)
    return _pool


class EGMMemPoolAllocator:

    def __init__(self, pool, enabled=True, multinode=False, device=None):
        self.pool = pool
        self.enabled = enabled
        self.multinode = multinode
        self.device = None
        self.mem_context = None

        if enabled:
            if device is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, int):
                self.device = torch.device("cuda", device)
            elif isinstance(device, str):
                assert "cuda" in device, "only cuda devices are supported"
                self.device = torch.device(device)

            self.mem_context = torch.cuda.use_mem_pool(self.pool)
        else:
            from contextlib import nullcontext

            self.mem_context = nullcontext()

    def __enter__(self):
        if self.multinode and _so_path is not None:
            import ctypes

            lib = ctypes.CDLL(_so_path)
            lib.egm_set_multinode.argtypes = [ctypes.c_int]
            lib.egm_set_multinode.restype = None
            lib.egm_set_multinode(1)

        self.mem_context.__enter__()
        return self

    def __exit__(self, *args):
        self.mem_context.__exit__(*args)

        if self.multinode and _so_path is not None:
            import ctypes

            lib = ctypes.CDLL(_so_path)
            lib.egm_set_multinode.argtypes = [ctypes.c_int]
            lib.egm_set_multinode.restype = None
            lib.egm_set_multinode(0)


@contextmanager
def egm_checkpoint_context(numa_node_id: int, multinode: bool = False, device=None):
    _build_egm_allocator()
    pool = create_egm_mem_pool(numa_node_id)
    allocator = EGMMemPoolAllocator(
        pool=pool, enabled=True, multinode=multinode, device=device
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[MCORE][EGM_ALLOCATOR] Entering EGM checkpoint context "
        f"(numa_node={numa_node_id}, multinode={multinode})",
    )
    with allocator:
        yield pool
    log_single_rank(
        logger,
        logging.INFO,
        "[MCORE][EGM_ALLOCATOR] Exiting EGM checkpoint context",
    )


def init(numa_node_id: int = 0) -> None:
    _build_egm_allocator()
    log_single_rank(
        logger,
        logging.INFO,
        f"[MCORE][EGM_ALLOCATOR] Initialized EGM Allocator (numa_node={numa_node_id})",
    )
