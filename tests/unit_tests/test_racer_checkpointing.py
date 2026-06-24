from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.training.racer import rank_map, session_state
from megatron.training import racer_checkpointing as rc


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
