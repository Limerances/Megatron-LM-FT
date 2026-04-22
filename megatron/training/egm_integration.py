# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
EGM (Extended GPU Memory) checkpoint integration for Megatron-LM.

Provides fast in-memory checkpointing using EGM on NVL72/GB200 architecture.
Supports two modes:
  - daemon mode (--egm-use-daemon): checkpoint memory is held by an independent
    EGMManagerServer daemon process, surviving training process crashes.
  - in-process mode (default): checkpoint memory is in the training process.

Fault recovery integration: when an exception is caught in the training loop,
on_fault_detected() is called to trigger emergency save, node exclusion, and
recovery coordination.
"""

import os
import signal
import sys
import time
from logging import getLogger
from typing import Any, Dict, Optional, Set

import torch

from . import global_vars
from .global_vars import get_args, get_timers
from .utils import print_rank_0

logger = getLogger(__name__)

_GLOBAL_EGM_CHECKPOINT_MANAGER = None
_GLOBAL_NODE_EXCLUSION_MANAGER = None
_GLOBAL_RECOVERY_COORDINATOR = None
_ORIGINAL_EXCEPTHOOK = None
_EGM_ITERATION_TRACKER = 0
_ORIGINAL_SIGNAL_HANDLERS = {}
_EGM_TIMER_WARNING_KEYS = set()


def _log_timer_issue_once(action: str, name: str, error: Exception) -> None:
    message = str(error)
    if "dummy timer should not be used" in message:
        return

    key = (action, name, message)
    if key in _EGM_TIMER_WARNING_KEYS:
        return
    _EGM_TIMER_WARNING_KEYS.add(key)
    logger.warning(f"EGM: failed to {action} timer {name}: {error}")


def _safe_timer_start(name: str, log_level: int = 1) -> bool:
    try:
        timers = get_timers()
        if timers is None:
            return False
        timers(name, log_level=log_level).start(barrier=False)
        return True
    except Exception as e:
        _log_timer_issue_once("start", name, e)
        return False


def _safe_timer_stop_elapsed(name: str) -> Optional[float]:
    try:
        timers = get_timers()
        if timers is None:
            return None
        timers(name).stop()
        return timers(name).elapsed(reset=True)
    except Exception as e:
        _log_timer_issue_once("collect", name, e)
        return None


def get_egm_checkpoint_manager():
    return _GLOBAL_EGM_CHECKPOINT_MANAGER


def get_recovery_coordinator():
    return _GLOBAL_RECOVERY_COORDINATOR


def get_excluded_nodes() -> Set[str]:
    if _GLOBAL_NODE_EXCLUSION_MANAGER is not None:
        return _GLOBAL_NODE_EXCLUSION_MANAGER.get_excluded_nodes()
    return set()


def classify_fault_type(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    hang_markers = (
        "timeout",
        "timed out",
        "watchdog",
        "stuck",
        "hang",
        "heartbeat",
        "connection closed by peer",
    )
    if any(marker in text for marker in hang_markers):
        return "hang"
    return "crash"


def setup(args) -> None:
    if not getattr(args, 'enable_egm_checkpoint', False):
        return

    from megatron.core.egm.egm_checkpoint import EGMCheckpointConfig, EGMCheckpointManager
    from .fault_recovery import NodeExclusionManager, RecoveryCoordinator

    print_rank_0("EGM: initializing...")

    use_daemon = getattr(args, 'egm_use_daemon', False)
    daemon_socket_path = getattr(args, 'egm_daemon_socket_path',
                                 '/tmp/megatron_egm_manager.sock')
    numa_node_id = getattr(args, 'egm_numa_node_id', 0)

    config = EGMCheckpointConfig(
        enabled=True,
        pool_size_gb=getattr(args, 'egm_pool_size_gb', 16.0),
        num_slots=getattr(args, 'egm_num_slots', 2),
        save_interval=getattr(args, 'egm_save_interval', 10),
        enable_hierarchical_backup=getattr(args, 'egm_hierarchical_backup', False),
        rack_id=getattr(args, 'egm_rack_id', 0),
        backup_rack_id=getattr(args, 'egm_backup_rack_id', -1),
        use_daemon=use_daemon,
        daemon_socket_path=daemon_socket_path,
        numa_node_id=numa_node_id,
    )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    local_rank = getattr(args, 'local_rank', None)
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    manager = EGMCheckpointManager(
        config=config,
        device_id=device_id,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    manager.initialize()

    if use_daemon:
        print_rank_0(
            f"EGM: initialized in DAEMON mode, connected to {daemon_socket_path}. "
            f"Checkpoint memory is held by external daemon — survives training crashes."
        )
    else:
        print_rank_0(
            "EGM: initialized in IN-PROCESS mode. "
            "Checkpoint memory will NOT survive training process crashes. "
            "Use --egm-use-daemon for production fault tolerance."
        )

    max_failures = getattr(args, 'egm_fault_max_failures', 3)
    exclusion_window = getattr(args, 'egm_fault_exclusion_window', 3600)
    node_exclusion_mgr = NodeExclusionManager(
        max_failures_per_node=max_failures,
        exclusion_window_secs=float(exclusion_window),
    )
    recovery_coord = RecoveryCoordinator(
        node_exclusion_manager=node_exclusion_mgr,
        egm_manager=manager,
    )

    global _GLOBAL_EGM_CHECKPOINT_MANAGER
    global _GLOBAL_NODE_EXCLUSION_MANAGER
    global _GLOBAL_RECOVERY_COORDINATOR
    global_vars._ensure_var_is_not_initialized(
        _GLOBAL_EGM_CHECKPOINT_MANAGER, 'egm checkpoint manager'
    )
    _GLOBAL_EGM_CHECKPOINT_MANAGER = manager
    _GLOBAL_NODE_EXCLUSION_MANAGER = node_exclusion_mgr
    _GLOBAL_RECOVERY_COORDINATOR = recovery_coord

    # Install sys.excepthook to catch unhandled exceptions and trigger fault recovery
    _install_fault_excepthook()
    _install_fault_signal_handlers()

    print_rank_0(
        f"EGM: initialized. pool_size_gb={config.pool_size_gb}, "
        f"num_slots={config.num_slots}, save_interval={config.save_interval}, "
        f"daemon={use_daemon}, numa_node={numa_node_id}, local_rank={local_rank}, "
        f"fault_max_failures={max_failures}, exclusion_window={exclusion_window}s"
    )


def _install_fault_excepthook():
    """Install a sys.excepthook that triggers fault recovery on unhandled exceptions."""
    global _ORIGINAL_EXCEPTHOOK
    _ORIGINAL_EXCEPTHOOK = sys.excepthook

    def _egm_excepthook(exc_type, exc_value, exc_tb):
        # Avoid recursion and skip KeyboardInterrupt / SystemExit
        if exc_type in (KeyboardInterrupt, SystemExit):
            if _ORIGINAL_EXCEPTHOOK is not None:
                _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_tb)
            return

        try:
            iteration = _EGM_ITERATION_TRACKER
            fault_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.error(
                f"EGM: unhandled exception caught by excepthook: "
                f"{exc_type.__name__}: {exc_value}"
            )
            on_fault_detected(
                iteration=iteration,
                fault_type=classify_fault_type(exc_value),
                fault_rank=fault_rank,
            )
        except Exception as hook_err:
            logger.error(f"EGM: error in excepthook handler: {hook_err}")
        finally:
            if _ORIGINAL_EXCEPTHOOK is not None:
                _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_tb)

    sys.excepthook = _egm_excepthook


def _install_fault_signal_handlers():
    handled_signals = (signal.SIGTERM, signal.SIGINT)

    for sig in handled_signals:
        _ORIGINAL_SIGNAL_HANDLERS[sig] = signal.getsignal(sig)

        def _egm_signal_handler(signum, frame, _sig=sig):
            fault_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            try:
                on_fault_detected(
                    iteration=_EGM_ITERATION_TRACKER,
                    fault_type="hang",
                    fault_rank=fault_rank,
                )
            except Exception as hook_err:
                logger.error(f"EGM: error in signal handler: {hook_err}")

            previous = _ORIGINAL_SIGNAL_HANDLERS.get(_sig)
            if callable(previous):
                previous(signum, frame)
            else:
                raise SystemExit(128 + signum)

        signal.signal(sig, _egm_signal_handler)


def on_training_step_end(iteration, model, optimizer, opt_param_scheduler) -> None:
    manager = get_egm_checkpoint_manager()
    if manager is None:
        return

    global _EGM_ITERATION_TRACKER
    _EGM_ITERATION_TRACKER = iteration

    args = get_args()
    save_interval = getattr(args, 'egm_save_interval', 10)

    if iteration > 0 and iteration % save_interval == 0:
        _save_egm_checkpoint(iteration, model, optimizer, opt_param_scheduler)


def on_fault_detected(
    iteration: int,
    model=None,
    optimizer=None,
    opt_param_scheduler=None,
    fault_type: str = "unknown",
    fault_rank: int = -1,
) -> None:
    """Called when a fault is detected — either from the training loop exception
    handler or from sys.excepthook.

    Performs:
    1. Emergency EGM checkpoint save (if model/optimizer available)
    2. Fault recording in NodeExclusionManager
    3. Recovery coordination (strategy selection, exclusion file writing)
    """
    import os
    import time
    from .fault_recovery import FaultRecord, get_node_id_for_rank, write_exclusion_file

    manager = get_egm_checkpoint_manager()
    coordinator = get_recovery_coordinator()

    print_rank_0(
        f"EGM: fault detected — type={fault_type}, rank={fault_rank}, "
        f"iteration={iteration}"
    )

    # 1. Emergency save if we have model/optimizer
    if (
        fault_type != "hang"
        and model is not None
        and optimizer is not None
        and manager is not None
    ):
        logger.warning("EGM: attempting emergency checkpoint save before recovery")
        try:
            _save_egm_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        except Exception as e:
            logger.error(f"EGM: emergency save failed: {e}")
    elif fault_type == "hang":
        logger.warning(
            "EGM: skipping emergency checkpoint save for hang/timeout fault; "
            "will rely on the most recent committed EGM checkpoint"
        )

    if coordinator is None:
        logger.warning(
            "EGM: no recovery coordinator available, skipping recovery orchestration"
        )
        return

    # 2. Determine fault rank
    if fault_rank < 0:
        fault_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )

    # 3. Record fault and run recovery coordination
    node_id = get_node_id_for_rank(fault_rank)
    fault_record = FaultRecord(
        rank=fault_rank,
        node_id=node_id,
        fault_type=fault_type,
        timestamp=time.time(),
        iteration=iteration,
    )

    result = coordinator.initiate_recovery([fault_record])

    # 4. Write exclusion file for ft_launcher
    excluded = get_excluded_nodes()
    if excluded:
        print_rank_0(f"EGM: excluded nodes after recovery: {excluded}")
        exclusion_path = os.environ.get(
            "FT_EXCLUSION_FILE_PATH",
            "/tmp/megatron_excluded_nodes.json",
        )
        write_exclusion_file(excluded, exclusion_path)
        print_rank_0(f"EGM: exclusion file written to {exclusion_path}")

    print_rank_0(f"EGM: recovery result: strategy={result.get('strategy')}")


def try_load_from_egm() -> Optional[Dict[str, Any]]:
    manager = get_egm_checkpoint_manager()
    if manager is None:
        return None

    restore_source_rank = manager.get_restore_source_rank()
    if not manager.has_valid_checkpoint():
        print_rank_0(
            f"EGM: no valid checkpoint found in EGM for "
            f"restore_source_rank={restore_source_rank}"
        )
        return None

    print_rank_0(
        f"EGM: found valid checkpoint at iteration "
        f"{manager.get_latest_iteration()} for restore_source_rank="
        f"{restore_source_rank}, loading..."
    )

    timed = _safe_timer_start('egm-load', log_level=1)

    state_dict = manager.load_state_dict()

    load_time = _safe_timer_stop_elapsed('egm-load') if timed else None

    if state_dict is not None:
        if load_time is not None:
            print_rank_0(
                f"EGM: loaded checkpoint from EGM in {load_time:.3f} seconds"
            )
        else:
            print_rank_0("EGM: loaded checkpoint from EGM")
    else:
        print_rank_0("EGM: failed to load checkpoint from EGM")

    return state_dict


def shutdown() -> None:
    global _GLOBAL_EGM_CHECKPOINT_MANAGER
    global _GLOBAL_NODE_EXCLUSION_MANAGER
    global _GLOBAL_RECOVERY_COORDINATOR

    # Restore original excepthook
    global _ORIGINAL_EXCEPTHOOK
    if _ORIGINAL_EXCEPTHOOK is not None:
        sys.excepthook = _ORIGINAL_EXCEPTHOOK
        _ORIGINAL_EXCEPTHOOK = None

    manager = get_egm_checkpoint_manager()
    if manager is not None:
        print_rank_0("EGM: shutting down...")
        manager.shutdown()
        print_rank_0("EGM: shut down.")

    _GLOBAL_EGM_CHECKPOINT_MANAGER = None
    _GLOBAL_NODE_EXCLUSION_MANAGER = None
    _GLOBAL_RECOVERY_COORDINATOR = None


def _save_egm_checkpoint(iteration, model, optimizer, opt_param_scheduler) -> None:
    manager = get_egm_checkpoint_manager()
    if manager is None:
        return

    args = get_args()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = (
        int(os.environ.get("LOCAL_RANK", rank))
        if torch.distributed.is_initialized() or torch.cuda.is_available()
        else 0
    )
    print_rank_0(f"EGM: saving checkpoint at iteration {iteration}")
    timed = _safe_timer_start('egm-save', log_level=1)

    from .checkpointing import generate_state_dict, get_rng_state
    from .utils import unwrap_model
    from megatron.core import parallel_state as mpu
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    model_unwrapped = unwrap_model(model)

    # Collect RNG state
    try:
        tp_group = mpu.get_tensor_model_parallel_group()
        pp_group = mpu.get_pipeline_model_parallel_group()
        rng_state = get_rng_state("torch", tp_group, pp_group)
    except Exception:
        rng_state = None

    # Collect rerun state machine state
    rerun_state = None
    try:
        rerun_sm = get_rerun_state_machine()
        rerun_state = rerun_sm.state_dict(
            data_iterator=None, ckpt_format="torch", force=True,
        )
    except Exception:
        pass

    original_ckpt_format = args.ckpt_format
    args.ckpt_format = "torch"
    try:
        state_dict = generate_state_dict(
            args,
            model_unwrapped,
            optimizer,
            opt_param_scheduler,
            rng_state=rng_state,
            iteration=iteration,
            rerun_state=rerun_state,
        )
    finally:
        args.ckpt_format = original_ckpt_format

    state_dict['num_floating_point_operations_so_far'] = getattr(
        args, 'num_floating_point_operations_so_far', 0
    )

    wall_start = time.perf_counter()
    success = manager.save_state_dict(state_dict, iteration)
    wall_end = time.perf_counter()

    save_time = _safe_timer_stop_elapsed('egm-save') if timed else None
    save_stats = manager.get_last_save_stats() if success else None

    if success:
        if save_stats is not None:
            payload_bytes = float(save_stats["payload_bytes"])
            total_bytes = float(save_stats["total_bytes"])
            serialize_s = float(save_stats["serialize_seconds"])
            write_s = float(save_stats["write_seconds"])
            total_s = float(save_stats["total_seconds"])
            wall_s = wall_end - wall_start
            payload_gib = payload_bytes / (1024 ** 3)
            total_gib = total_bytes / (1024 ** 3)
            write_bw_gib_s = (
                total_gib / write_s if write_s > 0 else 0.0
            )
            end_to_end_bw_gib_s = (
                total_gib / total_s if total_s > 0 else 0.0
            )
            tensor_count = save_stats.get("tensor_count")
            tensor_bytes = save_stats.get("tensor_bytes")
            gpu_copy_bw = save_stats.get("gpu_copy_bw_gib_s")
            gpu_tensor_bytes = save_stats.get("gpu_tensor_bytes")
            cpu_copy_bw = save_stats.get("cpu_copy_bw_gib_s")
            cpu_tensor_bytes = save_stats.get("cpu_tensor_bytes")
            save_format = save_stats.get("save_format")
            direct_copy_s = save_stats.get("direct_copy_seconds")
            direct_copy_bw = save_stats.get("direct_copy_bw_gib_s")
            direct_copy_path = save_stats.get("direct_copy_path")
            direct_copy_str = ""
            if direct_copy_s is not None and direct_copy_bw is not None:
                direct_copy_str = (
                    f" | direct_copy={float(direct_copy_s):.6f} s"
                    f" | direct_copy_bw={float(direct_copy_bw):.3f} GiB/s"
                    f" | direct_path={direct_copy_path}"
                )
            structured_str = ""
            if tensor_count is not None:
                structured_str = (
                    f" | format={save_format}"
                    f" | tensor_count={int(tensor_count)}"
                    f" | tensor_bytes={(float(tensor_bytes) / (1024 ** 3)) if tensor_bytes is not None else 0.0:.3f} GiB"
                    f" | gpu_tensor_bytes={(float(gpu_tensor_bytes) / (1024 ** 3)) if gpu_tensor_bytes is not None else 0.0:.3f} GiB"
                    f" | gpu_copy_bw={float(gpu_copy_bw) if gpu_copy_bw is not None else 0.0:.3f} GiB/s"
                    f" | cpu_tensor_bytes={(float(cpu_tensor_bytes) / (1024 ** 3)) if cpu_tensor_bytes is not None else 0.0:.3f} GiB"
                    f" | cpu_copy_bw={float(cpu_copy_bw) if cpu_copy_bw is not None else 0.0:.3f} GiB/s"
                )
            print(
                "EGM_SAVE_STATS"
                f" | rank={rank} (local_rank={local_rank})"
                f" | iter={iteration}"
                f" | slot=local:{save_stats['local_slot']}/backend:{save_stats['backend_slot']}"
                f" | size={payload_gib:.3f} GiB"
                f" | serialize={serialize_s:.3f} s"
                f" | write={write_s:.3f} s"
                f" | total={total_s:.3f} s"
                f" | wall={wall_s:.3f} s"
                f" | write_bw={write_bw_gib_s:.3f} GiB/s"
                f" | end_to_end_bw={end_to_end_bw_gib_s:.3f} GiB/s"
                f"{structured_str}"
                f"{direct_copy_str}",
                flush=True,
            )
        elif save_time is not None:
            print(
                "EGM_SAVE_STATS"
                f" | rank={rank} (local_rank={local_rank})"
                f" | iter={iteration}"
                f" | total={save_time:.3f} s",
                flush=True,
            )
        else:
            print(
                "EGM_SAVE_STATS"
                f" | rank={rank} (local_rank={local_rank})"
                f" | iter={iteration}"
                " | status=ok",
                flush=True,
            )
    else:
        if save_time is not None:
            print_rank_0(
                f"EGM: failed to save checkpoint at iteration {iteration} "
                f"(elapsed {save_time:.3f} seconds)"
            )
        else:
            print_rank_0(f"EGM: failed to save checkpoint at iteration {iteration}")

    if success and getattr(args, 'egm_hierarchical_backup', False):
        _trigger_hierarchical_backup(manager)


def _trigger_hierarchical_backup(manager) -> None:
    from megatron.core.egm.egm_checkpoint import (
        inter_rack_pair_backup,
        intra_rack_ring_backup,
    )

    args = get_args()
    print_rank_0("EGM: triggering hierarchical backup")
    timed = _safe_timer_start('egm-backup', log_level=1)

    best_slot = manager._find_best_valid_slot()
    if best_slot is None:
        print_rank_0("EGM: no valid slot for hierarchical backup")
        if timed:
            _safe_timer_stop_elapsed('egm-backup')
        return

    raw_data = manager.get_raw_slot_data(best_slot)
    if raw_data is None:
        print_rank_0("EGM: failed to read slot data for backup")
        if timed:
            _safe_timer_stop_elapsed('egm-backup')
        return

    intra_success = intra_rack_ring_backup(manager, raw_data)

    inter_success = False
    if getattr(args, 'egm_backup_rack_id', -1) >= 0:
        inter_success = inter_rack_pair_backup(manager, raw_data)

    backup_time = _safe_timer_stop_elapsed('egm-backup') if timed else None

    if backup_time is not None:
        print_rank_0(
            f"EGM: hierarchical backup completed in {backup_time:.3f} seconds "
            f"(intra_rack={intra_success}, inter_rack={inter_success})"
        )
    else:
        print_rank_0(
            f"EGM: hierarchical backup completed "
            f"(intra_rack={intra_success}, inter_rack={inter_success})"
        )
