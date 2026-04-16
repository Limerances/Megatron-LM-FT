# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from enum import Enum, auto
from logging import getLogger
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributed

from megatron.core.egm.egm_checkpoint import EGMCheckpointManager

logger = getLogger(__name__)

_RECOVERY_STRATEGY_EGM = "egm_restore"
_RECOVERY_STRATEGY_DISK = "disk_checkpoint"
_RECOVERY_STRATEGY_FULL_RESTART = "full_restart"

_DEFAULT_EXCLUSION_FILE = "excluded_nodes.json"


def _atomic_write_json(filepath: str, payload: Dict[str, Any]) -> None:
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-egm-",
        suffix=".json",
        dir=dirpath or None,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _merge_exclusion_payload(
    filepath: str, excluded_nodes: Set[str]
) -> Dict[str, Any]:
    merged_nodes = set(excluded_nodes)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                existing = json.load(f)
            merged_nodes.update(existing.get("excluded_nodes", []))
        except Exception:
            pass

    return {
        "excluded_nodes": sorted(merged_nodes),
        "num_excluded": len(merged_nodes),
        "timestamp": time.time(),
    }


class NodeHealthStatus(Enum):
    HEALTHY = auto()
    SUSPECTED = auto()
    EXCLUDED = auto()
    RECOVERING = auto()


@dataclass
class FaultRecord:
    rank: int
    node_id: str
    fault_type: str
    timestamp: float
    iteration: int


class NodeExclusionManager:

    def __init__(
        self,
        max_failures_per_node: int = 3,
        exclusion_window_secs: float = 3600.0,
    ):
        self._max_failures_per_node = max_failures_per_node
        self._exclusion_window_secs = exclusion_window_secs
        self._fault_history: List[FaultRecord] = []
        self._excluded_nodes: Set[str] = set()
        self._node_status: Dict[str, NodeHealthStatus] = {}

    def record_fault(self, fault_record: FaultRecord) -> None:
        self._fault_history.append(fault_record)
        node_id = fault_record.node_id

        if node_id not in self._node_status:
            self._node_status[node_id] = NodeHealthStatus.HEALTHY

        if self._node_status[node_id] != NodeHealthStatus.EXCLUDED:
            self._node_status[node_id] = NodeHealthStatus.SUSPECTED

        logger.warning(
            f"Fault recorded: rank={fault_record.rank}, "
            f"node={node_id}, type={fault_record.fault_type}, "
            f"iteration={fault_record.iteration}"
        )

        if self.should_exclude_node(node_id):
            self._excluded_nodes.add(node_id)
            self._node_status[node_id] = NodeHealthStatus.EXCLUDED
            logger.error(
                f"Node {node_id} has been excluded due to exceeding "
                f"failure threshold ({self._max_failures_per_node})"
            )

    def get_excluded_nodes(self) -> Set[str]:
        return set(self._excluded_nodes)

    def get_node_status(self, node_id: str) -> NodeHealthStatus:
        return self._node_status.get(node_id, NodeHealthStatus.HEALTHY)

    def should_exclude_node(self, node_id: str) -> bool:
        if node_id in self._excluded_nodes:
            return True

        cutoff = time.time() - self._exclusion_window_secs
        recent_faults = [
            f for f in self._fault_history
            if f.node_id == node_id and f.timestamp >= cutoff
        ]
        return len(recent_faults) >= self._max_failures_per_node

    def clear_exclusion(self, node_id: str) -> None:
        self._excluded_nodes.discard(node_id)
        if node_id in self._node_status:
            self._node_status[node_id] = NodeHealthStatus.RECOVERING
        logger.info(f"Node {node_id} exclusion cleared, status set to RECOVERING")

    def get_fault_history(self) -> List[FaultRecord]:
        return list(self._fault_history)


class RecoveryCoordinator:

    def __init__(
        self,
        node_exclusion_manager: NodeExclusionManager,
        egm_manager: Optional[EGMCheckpointManager] = None,
    ):
        self._node_exclusion_manager = node_exclusion_manager
        self._egm_manager = egm_manager

    def initiate_recovery(self, fault_records: List[FaultRecord]) -> Dict[str, Any]:
        logger.info(f"Initiating recovery for {len(fault_records)} fault(s)")

        for record in fault_records:
            self._node_exclusion_manager.record_fault(record)

        self._exclude_bad_nodes(fault_records)

        strategy = self._determine_recovery_strategy(fault_records)
        excluded_nodes = self._node_exclusion_manager.get_excluded_nodes()

        logger.info(
            f"Recovery strategy: {strategy}, "
            f"excluded nodes: {excluded_nodes}"
        )

        result: Dict[str, Any] = {
            "strategy": strategy,
            "excluded_nodes": list(excluded_nodes),
            "fault_records": [
                {
                    "rank": r.rank,
                    "node_id": r.node_id,
                    "fault_type": r.fault_type,
                    "iteration": r.iteration,
                }
                for r in fault_records
            ],
        }

        if strategy == _RECOVERY_STRATEGY_EGM:
            restore_success = self._restore_from_egm()
            result["egm_restore_success"] = restore_success
            if not restore_success:
                logger.warning(
                    "EGM restore failed, falling back to disk checkpoint strategy"
                )
                result["strategy"] = _RECOVERY_STRATEGY_DISK

        rank_mapping = self._reassign_ranks(excluded_nodes)
        result["rank_mapping"] = rank_mapping

        if result["strategy"] == _RECOVERY_STRATEGY_EGM:
            result["egm_restore_rank_by_new_rank"] = self._build_egm_restore_rank_plan(
                rank_mapping
            )
            if self._egm_manager is not None:
                result["egm_available_source_ranks"] = (
                    self._egm_manager.get_available_source_ranks()
                )

        self._notify_launcher_for_restart(
            excluded_nodes,
            result["strategy"],
            rank_mapping,
        )

        logger.info(f"Recovery plan finalized: strategy={result['strategy']}")
        return result

    def _determine_recovery_strategy(
        self, fault_records: List[FaultRecord]
    ) -> str:
        if self._egm_manager is not None and self._egm_manager.has_valid_checkpoint(
            include_remote=True
        ):
            logger.info(
                "EGM checkpoint available at iteration "
                f"{self._egm_manager.get_latest_iteration(include_remote=True)}, "
                "selecting egm_restore strategy"
            )
            return _RECOVERY_STRATEGY_EGM

        has_crash = any(r.fault_type == "crash" for r in fault_records)
        has_hang = any(r.fault_type == "hang" for r in fault_records)

        if has_crash and has_hang:
            logger.info(
                "Multiple fault types detected (crash + hang), "
                "selecting full_restart strategy"
            )
            return _RECOVERY_STRATEGY_FULL_RESTART

        logger.info("Falling back to disk_checkpoint strategy")
        return _RECOVERY_STRATEGY_DISK

    def _exclude_bad_nodes(self, fault_records: List[FaultRecord]) -> None:
        affected_nodes: Set[str] = set()
        for record in fault_records:
            affected_nodes.add(record.node_id)

        for node_id in affected_nodes:
            if self._node_exclusion_manager.should_exclude_node(node_id):
                logger.warning(f"Excluding node {node_id} from future runs")

    def _reassign_ranks(
        self, excluded_nodes: Set[str]
    ) -> Dict[int, Optional[str]]:
        if not torch.distributed.is_initialized():
            logger.warning(
                "Distributed not initialized, cannot reassign ranks"
            )
            return {}

        world_size = torch.distributed.get_world_size()
        rank_mapping: Dict[int, Optional[str]] = {}
        new_rank = 0

        for rank in range(world_size):
            node_id = get_node_id_for_rank(rank)
            if node_id in excluded_nodes:
                rank_mapping[rank] = None
                logger.info(f"Rank {rank} on excluded node {node_id} removed")
            else:
                rank_mapping[rank] = node_id
                new_rank += 1

        logger.info(
            f"Rank reassignment complete: "
            f"{new_rank}/{world_size} ranks active"
        )
        return rank_mapping

    def _build_egm_restore_rank_plan(
        self,
        rank_mapping: Dict[int, Optional[str]],
    ) -> Dict[int, int]:
        active_old_ranks = [
            old_rank
            for old_rank, node_id in sorted(rank_mapping.items())
            if node_id is not None
        ]
        return {
            new_rank: old_rank for new_rank, old_rank in enumerate(active_old_ranks)
        }

    def _restore_from_egm(self) -> bool:
        if self._egm_manager is None:
            logger.error("EGM manager not available for restore")
            return False

        try:
            if not self._egm_manager.has_valid_checkpoint(include_remote=True):
                logger.error("No valid EGM checkpoint available for restore")
                return False

            iteration = self._egm_manager.get_latest_iteration(include_remote=True)
            available_source_ranks = self._egm_manager.get_available_source_ranks()
            logger.info(
                f"EGM checkpoint available at iteration {iteration}, "
                f"recovery will proceed via load_checkpoint with EGM priority; "
                f"available_source_ranks={available_source_ranks}"
            )
            return True

        except Exception as e:
            logger.error(f"EGM restore check failed: {e}")
            return False

    def _notify_launcher_for_restart(
        self,
        excluded_nodes: Set[str],
        strategy: str,
        rank_mapping: Dict[int, Optional[str]],
    ) -> None:
        egm_restore_rank_by_new_rank = {}
        egm_available_source_ranks: List[int] = []
        if strategy == _RECOVERY_STRATEGY_EGM:
            egm_restore_rank_by_new_rank = self._build_egm_restore_rank_plan(
                rank_mapping
            )
            if self._egm_manager is not None:
                egm_available_source_ranks = (
                    self._egm_manager.get_available_source_ranks()
                )

        restart_info = {
            "action": "restart",
            "strategy": strategy,
            "excluded_nodes": list(excluded_nodes),
            "rank_mapping": rank_mapping,
            "egm_restore_rank_by_new_rank": egm_restore_rank_by_new_rank,
            "egm_available_source_ranks": egm_available_source_ranks,
            "timestamp": time.time(),
        }

        restart_file = os.environ.get(
            "FT_RESTART_INFO_PATH",
            "/tmp/megatron_ft_restart_info.json",
        )

        try:
            _atomic_write_json(restart_file, restart_info)
            logger.info(
                f"Restart notification written to {restart_file} "
                f"for ft_launcher (strategy={strategy}, "
                f"excluded={len(excluded_nodes)} nodes)"
            )
        except OSError as e:
            logger.error(f"Failed to write restart info to {restart_file}: {e}")


_RANK_TO_NODE_CACHE: Dict[int, str] = {}


def get_node_id_for_rank(rank: int) -> str:
    if rank in _RANK_TO_NODE_CACHE:
        return _RANK_TO_NODE_CACHE[rank]

    if not torch.distributed.is_initialized():
        hostname = socket.gethostname()
        _RANK_TO_NODE_CACHE[rank] = hostname
        return hostname

    hostname = socket.gethostname()
    hostname_bytes = hostname.encode("utf-8")
    max_len = 256
    padded = hostname_bytes[:max_len].ljust(max_len, b"\0")
    local_tensor = torch.ByteTensor(list(padded))

    if torch.cuda.is_available():
        local_tensor = local_tensor.cuda()

    world_size = torch.distributed.get_world_size()
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local_tensor)

    for r in range(world_size):
        raw = bytes(gathered[r].cpu().tolist())
        _RANK_TO_NODE_CACHE[r] = raw.rstrip(b"\0").decode("utf-8")

    return _RANK_TO_NODE_CACHE.get(rank, f"unknown-rank-{rank}")


def create_recovery_plan(
    fault_records: List[FaultRecord],
    egm_available: bool,
) -> Dict[str, Any]:
    affected_nodes: Set[str] = set()
    affected_ranks: Set[int] = set()
    fault_types: Set[str] = set()

    for record in fault_records:
        affected_nodes.add(record.node_id)
        affected_ranks.add(record.rank)
        fault_types.add(record.fault_type)

    if egm_available:
        strategy = _RECOVERY_STRATEGY_EGM
    elif len(fault_types) > 1 or len(affected_nodes) > 1:
        strategy = _RECOVERY_STRATEGY_FULL_RESTART
    else:
        strategy = _RECOVERY_STRATEGY_DISK

    latest_iteration = max(
        (r.iteration for r in fault_records), default=-1
    )

    plan: Dict[str, Any] = {
        "strategy": strategy,
        "affected_nodes": list(affected_nodes),
        "affected_ranks": list(affected_ranks),
        "fault_types": list(fault_types),
        "num_faults": len(fault_records),
        "latest_fault_iteration": latest_iteration,
        "egm_available": egm_available,
        "timestamp": time.time(),
    }

    logger.info(
        f"Recovery plan created: strategy={strategy}, "
        f"affected_nodes={len(affected_nodes)}, "
        f"affected_ranks={len(affected_ranks)}"
    )

    return plan


def write_exclusion_file(
    excluded_nodes: Set[str],
    filepath: str = _DEFAULT_EXCLUSION_FILE,
) -> None:
    try:
        lock_path = f"{filepath}.lock"
        with open(lock_path, "w") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass

            exclusion_data = _merge_exclusion_payload(filepath, excluded_nodes)
            _atomic_write_json(filepath, exclusion_data)

        logger.info(
            f"Exclusion file written to {filepath} "
            f"with {len(excluded_nodes)} excluded node(s)"
        )
    except OSError as e:
        logger.error(f"Failed to write exclusion file to {filepath}: {e}")
