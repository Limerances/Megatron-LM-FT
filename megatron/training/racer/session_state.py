"""Mutable session state for Megatron's RACER adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RacerSessionState:
    latest_tag: str | None = None
    tags_by_iteration: dict[int, str] = field(default_factory=dict)
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    tree_checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    chunk_storage_clients: dict[Any, Any] = field(default_factory=dict)
    distributed_runtime: Any | None = None
    local_racer_context: Any | None = None
    launch_streams: dict[int, Any] = field(default_factory=dict)
    payload_buffer_pool: Any | None = None
    async_save_executor: Any | None = None
    pending_async_save: Any | None = None

    def reset(self) -> None:
        if self.pending_async_save is not None:
            future = getattr(self.pending_async_save, "future", None)
            if future is not None and not future.done():
                raise RuntimeError("cannot reset RACER session while an async checkpoint save is in flight")
        executor = self.async_save_executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        self.async_save_executor = None
        self.pending_async_save = None
        self.latest_tag = None
        self.tags_by_iteration.clear()
        self.reports.clear()
        self.tree_checkpoints.clear()
        for client in list(self.chunk_storage_clients.values()):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.chunk_storage_clients.clear()
        self.distributed_runtime = None
        self.local_racer_context = None
        self.launch_streams.clear()
        self.payload_buffer_pool = None
