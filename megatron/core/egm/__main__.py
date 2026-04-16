# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Entry point for running the EGM daemon as a module.

Usage:
    python -m megatron.core.egm --pool-size-gb 16 --numa-node-id 0 --num-slots 2
"""

from megatron.core.egm.egm_manager import run_egm_daemon

if __name__ == "__main__":
    run_egm_daemon()
