from concurrent.futures import Future
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.timers import Timer
from megatron.training.racer import rank_map, session_state, tensor_tree
from megatron.training import checkpointing
from megatron.training import racer_checkpointing as rc


@pytest.fixture
def isolated_async_session(monkeypatch):
    state = session_state.RacerSessionState()
    monkeypatch.setattr(rc, "_SESSION", state)
    return state


def _pending_async_save(future: Future):
    prepared = SimpleNamespace(
        args=SimpleNamespace(),
        tag="megatron:iter_0000007",
        iteration=7,
        device=torch.device("cpu"),
    )
    return rc._PendingAsyncSave(
        prepared=prepared, future=future, scheduled_at=time.perf_counter()
    )


def test_extract_tensor_tree_decodes_sharded_tensor_data_as_tensor():
    data = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    sharded = ShardedTensor.from_rank_offsets("model.weight", data, (0, 0, 1))

    skeleton, metas, tensors = rc._extract_tensor_tree({"model": {"weight": sharded}, "meta": {"step": 7}})
    restored = rc._decode_tensor_tree(skeleton, {0: tensors[0].clone()})

    assert len(metas) == 1
    assert torch.equal(restored["model"]["weight"], data)
    assert restored["meta"] == {"step": 7}


def test_tensor_tree_pack_skips_empty_tensor_payload_segments():
    empty = torch.empty(0, dtype=torch.uint8)
    payload = torch.arange(4, dtype=torch.uint8)

    assert rc._tensor_to_cuda_bytes(empty, torch.device("cpu")).numel() == 0
    chunks, payloads = rc._pack_tensor_chunks([empty, payload], torch.device("cpu"), chunk_size=2)

    assert [int(item.numel()) for item in payloads] == [2, 2]
    assert [segment["tensor_id"] for chunk in chunks for segment in chunk["segments"]] == [1, 1]


def test_payload_pool_reuses_device_load_slots_across_checkpoints():
    pool = tensor_tree.PinnedPayloadBufferPool()

    first = pool.ensure_cuda_load_slots(
        device=torch.device("cpu"), chunk_size=8, slot_count=2
    )
    second = pool.ensure_cuda_load_slots(
        device=torch.device("cpu"), chunk_size=8, slot_count=2
    )

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert pool.last_ensure_profile["payload_cuda_slot_allocated"] == 0.0
    assert pool.last_ensure_profile["payload_cuda_slot_reused"] == 2.0


def test_replacement_mapping_converts_physical_ranks_to_racer_pg_ranks():
    args = SimpleNamespace(racer_train_ranks="2,4", racer_spare_ranks="7,8")

    assert rc._to_racer_pg_train_ranks(args, [4]) == [1]
    assert rc._to_racer_pg_replacement_mapping(args, {4: 7, 2: 4}) == {1: 2, 0: 1}


def test_rank_map_module_converts_physical_failed_and_replacement_ranks():
    train = [2, 4, 6]
    spare = [9, 10]

    assert rank_map.to_racer_pg_train_ranks([6, 2], train) == [2, 0]
    assert rank_map.to_racer_pg_replacement_mapping(
        {6: 9, 2: 4},
        train=train,
        spare=spare,
    ) == {2: 3, 0: 1}


def test_session_state_reset_clears_mutable_adapter_state():
    state = session_state.RacerSessionState(
        latest_tag="tag",
        distributed_runtime=object(),
        local_racer_context=object(),
    )
    state.tags_by_iteration[1] = "tag"
    state.reports["tag"] = {"ok": True}
    state.tree_checkpoints["tag"] = {"rank": 0}

    state.reset()

    assert state.latest_tag is None
    assert state.tags_by_iteration == {}
    assert state.reports == {}
    assert state.tree_checkpoints == {}
    assert state.distributed_runtime is None
    assert state.local_racer_context is None


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (SimpleNamespace(), False),
        (
            SimpleNamespace(
                racer_checkpoint=True,
                racer_distributed_store=True,
                racer_async_offload=True,
            ),
            True,
        ),
        (
            SimpleNamespace(
                racer_checkpoint=False,
                racer_distributed_store=True,
                racer_async_offload=True,
            ),
            False,
        ),
        (
            SimpleNamespace(
                racer_checkpoint=True,
                racer_distributed_store=False,
                racer_async_offload=True,
            ),
            False,
        ),
        (
            SimpleNamespace(
                racer_checkpoint=True,
                racer_distributed_store=True,
                racer_async_offload=False,
            ),
            False,
        ),
    ],
)
def test_racer_async_offload_requires_enabled_distributed_checkpoint(args, expected):
    assert rc.racer_async_offload_enabled(args) is expected


def test_async_save_nonblocking_poll_preserves_incomplete_future(
    isolated_async_session, monkeypatch
):
    future = Future()
    pending = _pending_async_save(future)
    isolated_async_session.pending_async_save = pending
    monkeypatch.setattr(rc, "_distributed_initialized", lambda: False)
    monkeypatch.setattr(
        rc,
        "_finalize_executed_distributed_store",
        lambda executed: pytest.fail("an incomplete Future must not be finalized"),
    )
    monkeypatch.setattr(
        rc,
        "_publish_completed_save",
        lambda args, prepared, report: pytest.fail(
            "an incomplete Future must not be published"
        ),
    )

    assert rc.maybe_finalize_async_saves(blocking=False) is False
    assert isolated_async_session.pending_async_save is pending

    future.cancel()


def test_async_save_nonblocking_poll_waits_for_all_ranks(
    isolated_async_session, monkeypatch
):
    future = Future()
    future.set_result(object())
    pending = _pending_async_save(future)
    isolated_async_session.pending_async_save = pending
    monkeypatch.setattr(rc, "_distributed_initialized", lambda: True)
    monkeypatch.setattr(rc, "_current_cuda_device", lambda: torch.device("cpu"))

    def mark_peer_incomplete(status, *, op):
        assert op is torch.distributed.ReduceOp.MAX
        assert status.tolist() == [0, 0]
        status[1] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", mark_peer_incomplete)
    monkeypatch.setattr(
        rc,
        "_finalize_executed_distributed_store",
        lambda executed: pytest.fail("local completion is insufficient"),
    )

    assert rc.maybe_finalize_async_saves(blocking=False) is False
    assert isolated_async_session.pending_async_save is pending


def test_async_save_local_worker_error_preserves_pending_without_finalizing(
    isolated_async_session, monkeypatch
):
    worker_error = ValueError("EGM write failed")
    future = Future()
    future.set_exception(worker_error)
    pending = _pending_async_save(future)
    isolated_async_session.pending_async_save = pending
    monkeypatch.setattr(rc, "_distributed_initialized", lambda: False)
    monkeypatch.setattr(
        rc,
        "_finalize_executed_distributed_store",
        lambda executed: pytest.fail("a failed Future must not be finalized"),
    )
    monkeypatch.setattr(
        rc,
        "_publish_completed_save",
        lambda args, prepared, report: pytest.fail("a failed Future must not be published"),
    )

    with pytest.raises(
        RuntimeError,
        match="failed.*one or more ranks.*ValueError: EGM write failed",
    ) as exc_info:
        rc.maybe_finalize_async_saves(blocking=False)

    assert exc_info.value.__cause__ is worker_error
    assert isolated_async_session.pending_async_save is pending


def test_async_save_remote_worker_error_is_raised_before_local_finalize(
    isolated_async_session, monkeypatch
):
    future = Future()
    future.set_result(object())
    pending = _pending_async_save(future)
    isolated_async_session.pending_async_save = pending
    reductions = []
    monkeypatch.setattr(rc, "_distributed_initialized", lambda: True)
    monkeypatch.setattr(rc, "_current_cuda_device", lambda: torch.device("cpu"))

    def report_remote_failure(value, *, op):
        reductions.append(op)
        assert op is torch.distributed.ReduceOp.MAX
        assert value.tolist() == [0, 0]
        value[0] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", report_remote_failure)
    monkeypatch.setattr(
        rc,
        "_finalize_executed_distributed_store",
        lambda executed: pytest.fail("remote failure must prevent local finalization"),
    )
    monkeypatch.setattr(
        rc,
        "_publish_completed_save",
        lambda args, prepared, report: pytest.fail("remote failure must prevent publication"),
    )

    with pytest.raises(
        RuntimeError,
        match="failed.*one or more ranks.*local worker completed, but another rank's worker failed",
    ):
        rc.maybe_finalize_async_saves(blocking=False)

    assert reductions == [torch.distributed.ReduceOp.MAX]
    assert isolated_async_session.pending_async_save is pending


def test_async_save_completed_future_finalizes_publishes_and_clears_pending(
    isolated_async_session, monkeypatch
):
    executed = object()
    future = Future()
    future.set_result(executed)
    pending = _pending_async_save(future)
    isolated_async_session.pending_async_save = pending
    events = []

    def finalize(value):
        events.append(("finalize", value))
        return {"finalized": True}

    def publish(args, prepared, report):
        assert isolated_async_session.pending_async_save is None
        events.append(("publish", prepared, dict(report)))

    monkeypatch.setattr(rc, "_distributed_initialized", lambda: False)
    monkeypatch.setattr(rc, "_finalize_executed_distributed_store", finalize)
    monkeypatch.setattr(rc, "_publish_completed_save", publish)
    monkeypatch.setattr(rc, "_print_rank0", lambda message: None)

    assert rc.maybe_finalize_async_saves(blocking=False) is True
    assert isolated_async_session.pending_async_save is None
    assert events[0] == ("finalize", executed)
    assert events[1][0:2] == ("publish", pending.prepared)
    assert events[1][2]["finalized"] is True
    assert events[1][2]["racer_async_commit_wall_ms_max"] >= 0.0


def test_async_save_blocking_drain_waits_for_future(
    isolated_async_session, monkeypatch
):
    future = Future()
    pending = _pending_async_save(future)
    isolated_async_session.pending_async_save = pending
    completed = threading.Event()
    results = []
    errors = []

    monkeypatch.setattr(rc, "_distributed_initialized", lambda: False)
    monkeypatch.setattr(
        rc, "_finalize_executed_distributed_store", lambda executed: {"value": executed}
    )
    monkeypatch.setattr(rc, "_publish_completed_save", lambda args, prepared, report: None)
    monkeypatch.setattr(rc, "_print_rank0", lambda message: None)

    def drain():
        try:
            results.append(rc.maybe_finalize_async_saves(blocking=True))
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=drain)
    thread.start()
    assert not completed.wait(timeout=0.05)

    future.set_result("stored")
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert not errors
    assert results == [True]
    assert isolated_async_session.pending_async_save is None


def test_async_worker_only_finalizes_prepared_runtime_handle(monkeypatch):
    monkeypatch.syspath_prepend("/mnt/data/luohaonan/workspace/racer")
    from racer import distributed

    events = []
    handle = object()
    states = [SimpleNamespace(tag="chunk:0", profile={"total_ms": 3.0})]

    def finalize(value):
        events.append(("finalize", value))
        return states

    monkeypatch.setattr(
        distributed,
        "prepare_distributed_store_many",
        lambda **kwargs: pytest.fail("runtime-PG prepare must finish before training resumes"),
    )
    monkeypatch.setattr(distributed, "finalize_distributed_store_many", finalize)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(("set_device", device)))

    runtime = SimpleNamespace(
        rank=1,
        process_group=object(),
        init_profile={},
        states={},
    )
    prepared = SimpleNamespace(
        args=SimpleNamespace(),
        tag="checkpoint",
        device=torch.device("cpu"),
        runtime=runtime,
        distributed_store_handle=handle,
        racer_prepare_ms_local=2.0,
        tensor_view_ms_local=1.0,
        store_started_at=time.perf_counter(),
        spare_completion_key=None,
    )

    executed = rc._execute_prepared_distributed_store(prepared)

    assert events[0] == ("set_device", torch.device("cpu"))
    assert events[1] == ("finalize", handle)
    assert executed.prepared is prepared
    assert executed.tensor_view_ms_local == 1.0
    assert executed.racer_calls_ms_local >= 2.0
    assert runtime.states == {"chunk:0": states[0]}


def test_racer_blocking_profile_uses_one_packed_max_reduction(monkeypatch):
    profile = {
        "pre_state_ms": 1.0,
        "optimizer_capture_ms": 2.0,
        "state_dict_ms": 3.0,
        "racer_adapter_save_ms": 4.0,
        "finalize_ms": 5.0,
        "save_checkpoint_fn_total_ms": 6.0,
    }
    calls = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("RACER_BLOCKING_PROFILE_GLOBAL_MAX", "1")

    def reduce(values, *, op):
        calls.append((values.numel(), op))
        values.add_(10.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    checkpointing._racer_reduce_blocking_profile(profile)

    assert calls == [(6, torch.distributed.ReduceOp.MAX)]
    assert list(profile.values()) == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]


def test_racer_blocking_profile_is_rank_local_by_default(monkeypatch):
    profile = {"racer_adapter_save_ms": 4.0}

    monkeypatch.delenv("RACER_BLOCKING_PROFILE_GLOBAL_MAX", raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("rank-local profiling must not synchronize"),
    )

    checkpointing._racer_reduce_blocking_profile(profile)

    assert profile["racer_adapter_save_ms"] == 4.0
    assert profile["pre_state_ms"] == 0.0


def test_timer_can_measure_async_caller_without_cuda_synchronization(monkeypatch):
    timer = Timer("racer-async-caller")
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: pytest.fail("async caller timing must not wait for RACER streams"),
    )

    timer.start(sync_cuda=False)
    timer.stop(sync_cuda=False)

    assert timer.elapsed() >= 0.0


def test_session_state_reset_rejects_in_flight_async_save():
    state = session_state.RacerSessionState()
    future = Future()
    pending = SimpleNamespace(future=future)
    state.pending_async_save = pending

    with pytest.raises(RuntimeError, match="async checkpoint save is in flight"):
        state.reset()

    assert state.pending_async_save is pending
    future.cancel()
    state.reset()
    assert state.pending_async_save is None


def test_session_state_reset_clears_launch_stream_cache():
    state = session_state.RacerSessionState()
    state.launch_streams[0] = object()

    state.reset()

    assert state.launch_streams == {}


def test_racer_storage_backend_defaults_to_restart_aware_csd():
    args = SimpleNamespace()

    assert rc._racer_storage_backend(args) == "csd_native_pinned"
    assert rc._racer_csd_storage_enabled(args)
    with pytest.raises(RuntimeError, match="--racer-csd-port"):
        rc._racer_csd_storage_options(args)


def test_racer_storage_backend_accepts_explicit_native_csd_and_rejects_legacy_cuda():
    csd_args = SimpleNamespace(
        racer_storage_backend="csd_native_pinned",
        racer_csd_host="127.0.0.1",
        racer_csd_port=7007,
        racer_csd_authkey="key",
    )
    legacy_args = SimpleNamespace(racer_storage_backend="cuda_legacy")

    assert rc._racer_csd_storage_options(csd_args) == {
        "address": ("127.0.0.1", 7007),
        "authkey": "key",
        "cuda_register_fd_mappings": False,
    }
    with pytest.raises(ValueError, match="unsupported --racer-storage-backend"):
        rc._racer_storage_backend(legacy_args)


def test_racer_csd_backend_validation_rejects_mismatched_daemon():
    class FakeClient:
        def capabilities(self):
            return {"backend": "egm", "cuda_native_pinned": False}

    with pytest.raises(RuntimeError, match="requires CSD backend=native_pinned"):
        rc._validate_csd_backend_matches_cli("csd_native_pinned", FakeClient())

    with pytest.raises(RuntimeError, match="requires CSD backend=egm"):
        rc._validate_csd_backend_matches_cli("csd_egm", FakeClient())


def test_racer_csd_socket_path_takes_precedence_over_tcp_port():
    args = SimpleNamespace(
        racer_csd_socket_path="/tmp/racer-csd.sock",
        racer_csd_host="127.0.0.1",
        racer_csd_port=7007,
        racer_csd_authkey="key",
    )

    assert rc._racer_csd_storage_options(args) == {
        "address": "/tmp/racer-csd.sock",
        "authkey": "key",
        "cuda_register_fd_mappings": False,
    }


def test_manifest_path_is_per_rank(tmp_path, monkeypatch):
    args = SimpleNamespace(racer_manifest_dir=str(tmp_path))
    monkeypatch.setattr(rc, "_rank", lambda: 3)

    path = rc._manifest_path(args, "megatron:iter_0000005")

    assert path.parent == tmp_path
    assert path.name == "megatron__iter_0000005.rank_00003.pt"


def test_manifest_tag_requires_commit_marker_for_version_two(tmp_path, monkeypatch):
    args = SimpleNamespace(racer_manifest_dir=str(tmp_path))
    tag = "megatron:iter_0000005"
    monkeypatch.setattr(
        rc,
        "_load_tree_manifest",
        lambda args, tag, rank=0: {
            "requires_commit_marker": True,
            "checkpoint_generation": "gen-1",
        },
    )
    monkeypatch.setattr(rc, "_manifest_chunks_committed", lambda args, tag: True)

    assert not rc._manifest_tag_available(args, tag)

    rc._write_checkpoint_commit_marker(args, tag, 5, "stale-generation")

    assert not rc._manifest_tag_available(args, tag)

    rc._write_checkpoint_commit_marker(args, tag, 5, "gen-1")

    assert rc._manifest_tag_available(args, tag)


def test_publish_tree_manifest_writes_commit_marker_last(tmp_path, monkeypatch):
    args = SimpleNamespace(racer_manifest_dir=str(tmp_path))
    prepared = SimpleNamespace(
        args=args,
        tag="megatron:iter_0000005",
        iteration=5,
        tree={
            "requires_commit_marker": True,
            "checkpoint_generation": "gen-1",
        },
        device=torch.device("cpu"),
    )
    events = []
    monkeypatch.setattr(
        rc,
        "_write_tree_manifest",
        lambda args, tag, tree: events.append(("manifest", tag)),
    )
    monkeypatch.setattr(
        rc,
        "_write_checkpoint_commit_marker",
        lambda args, tag, iteration, generation: events.append(
            ("marker", tag, iteration, generation)
        ),
    )
    monkeypatch.setattr(rc, "_sum_int_across_ranks", lambda value, device: int(value))
    monkeypatch.setattr(rc, "_rank", lambda: 0)

    rc._publish_tree_manifest_with_commit_marker(prepared)

    assert events == [
        ("manifest", "megatron:iter_0000005"),
        ("marker", "megatron:iter_0000005", 5, "gen-1"),
    ]


def test_publish_tree_manifest_failure_never_writes_commit_marker(tmp_path, monkeypatch):
    prepared = SimpleNamespace(
        args=SimpleNamespace(racer_manifest_dir=str(tmp_path)),
        tag="megatron:iter_0000005",
        iteration=5,
        tree={
            "requires_commit_marker": True,
            "checkpoint_generation": "gen-1",
        },
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        rc,
        "_write_tree_manifest",
        lambda args, tag, tree: (_ for _ in ()).throw(OSError("manifest write failed")),
    )
    monkeypatch.setattr(
        rc,
        "_write_checkpoint_commit_marker",
        lambda args, tag, iteration, generation: pytest.fail(
            "marker must not be written"
        ),
    )
    monkeypatch.setattr(rc, "_sum_int_across_ranks", lambda value, device: int(value))

    with pytest.raises(RuntimeError, match="tree-manifest publication failed"):
        rc._publish_tree_manifest_with_commit_marker(prepared)
