"""Rank parsing and physical-to-RACER-process-group mapping helpers."""

from __future__ import annotations

from typing import Any


def parse_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(item) for item in str(value).split(",") if item.strip()]


def parse_rank_mapping(value: Any) -> dict[int, int]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    mapping: dict[int, int] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            left, right = item.split(":", 1)
        elif "=" in item:
            left, right = item.split("=", 1)
        else:
            raise ValueError(
                "--racer-replacement-mapping entries must be failed:replacement or failed=replacement"
            )
        mapping[int(left.strip())] = int(right.strip())
    return mapping


def train_ranks(args: Any, world_size: int) -> list[int]:
    return parse_int_list(getattr(args, "racer_train_ranks", None), list(range(int(world_size))))


def spare_ranks(args: Any, train: list[int]) -> list[int]:
    default = [len(train)]
    return parse_int_list(getattr(args, "racer_spare_ranks", None), default)


def racer_pg_train_ranks(train: list[int]) -> list[int]:
    return list(range(len(train)))


def racer_pg_spare_ranks(train: list[int], spare: list[int]) -> list[int]:
    base = len(train)
    return [base + idx for idx, _ in enumerate(spare)]


def racer_pg_rank_map(train: list[int]) -> dict[int, int]:
    return {int(rank): idx for idx, rank in enumerate(train)}


def to_racer_pg_train_ranks(ranks: list[int] | None, train: list[int]) -> list[int]:
    if not ranks:
        return []
    rank_map = racer_pg_rank_map(train)
    converted: list[int] = []
    for rank in ranks:
        physical_rank = int(rank)
        try:
            converted.append(rank_map[physical_rank])
        except KeyError as exc:
            raise ValueError(
                f"RACER rank {physical_rank} is not listed in --racer-train-ranks={train}"
            ) from exc
    return converted


def to_racer_pg_replacement_mapping(
    mapping: dict[int, int] | None,
    *,
    train: list[int],
    spare: list[int],
) -> dict[int, int]:
    if not mapping:
        return {}
    train_map = racer_pg_rank_map(train)
    spare_map = {
        int(physical): int(racer)
        for physical, racer in zip(spare, racer_pg_spare_ranks(train, spare))
    }
    converted: dict[int, int] = {}
    for failed, replacement in mapping.items():
        failed_rank = int(failed)
        replacement_rank = int(replacement)
        if failed_rank not in train_map:
            raise ValueError(f"replacement_mapping failed rank {failed_rank} is not in --racer-train-ranks")
        if replacement_rank in train_map:
            converted[train_map[failed_rank]] = train_map[replacement_rank]
        elif replacement_rank in spare_map:
            converted[train_map[failed_rank]] = spare_map[replacement_rank]
        else:
            raise ValueError(
                f"replacement_mapping target {replacement_rank} is neither a train rank nor spare CUDA device"
            )
    return converted


def resolve_failed_train_ranks(
    args: Any,
    failed_train_ranks: list[int] | None,
    *,
    current_rank: int,
    train: list[int],
) -> list[int]:
    failed = (
        [int(rank) for rank in failed_train_ranks]
        if failed_train_ranks is not None
        else parse_int_list(getattr(args, "racer_recover_ranks", None), [])
    )
    if bool(getattr(args, "racer_force_recover", False)):
        if current_rank in train and current_rank not in failed:
            failed.append(current_rank)
    return failed
