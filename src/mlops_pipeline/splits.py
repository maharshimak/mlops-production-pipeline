"""Leakage-safe, deterministic train/test splitting.

Rows are assigned to train or test by *group* (entity) so that every row of an
entity lands on the same side. Each distinct group is ranked by
``sha256(canonical_json({"seed": seed, "group": [type, value]}))`` and groups
are moved to the test side in rank order until the target row count is reached.

Consequences that the tests pin down:

* train and test never share a group, and every row is assigned exactly once;
* the result depends only on the seed and the multiset of groups, not on row
  order, Python hash randomisation or the platform;
* without a group column each row is its own group (keyed by row index).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from mlops_pipeline.canonical import canonical_sha256

MIN_TEST_FRACTION = 0.1
MAX_TEST_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class GroupSplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_group_count: int
    test_group_count: int
    seed: int
    test_fraction: float


def _group_key(value: str | int) -> list[object]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("group values must be strings or integers")
    return ["str" if isinstance(value, str) else "int", value]


def group_rank(group: str | int, seed: int) -> str:
    """Seeded, platform-independent ordering key for one group value."""
    return canonical_sha256({"seed": seed, "group": _group_key(group)})


def group_train_test_split(
    groups: Sequence[str | int],
    *,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> GroupSplit:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, (int, float))
        or not MIN_TEST_FRACTION <= test_fraction <= MAX_TEST_FRACTION
    ):
        raise ValueError(
            f"test_fraction must be between {MIN_TEST_FRACTION} and {MAX_TEST_FRACTION}"
        )
    members: dict[tuple[str, str | int], list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        tag, value = _group_key(group)
        members[(tag, value)].append(index)  # type: ignore[index]
    if len(members) < 2:
        raise ValueError("at least two distinct groups are required for a train/test split")

    ranked = sorted(members, key=lambda key: (group_rank(key[1], seed), key))
    target = ceil(len(groups) * test_fraction)
    test_keys: list[tuple[str, str | int]] = []
    test_rows = 0
    for key in ranked[:-1]:  # always keep at least one group for training
        if test_rows >= target:
            break
        test_keys.append(key)
        test_rows += len(members[key])

    test_key_set = set(test_keys)
    test_indices = sorted(i for key in test_keys for i in members[key])
    train_indices = sorted(
        i for key, rows in members.items() if key not in test_key_set for i in rows
    )
    return GroupSplit(
        train_indices=tuple(train_indices),
        test_indices=tuple(test_indices),
        train_group_count=len(members) - len(test_keys),
        test_group_count=len(test_keys),
        seed=seed,
        test_fraction=float(test_fraction),
    )


def row_train_test_split(
    row_count: int,
    *,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> GroupSplit:
    """Split independent rows (each row is its own group)."""
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 2:
        raise ValueError("row_count must be an integer >= 2")
    return group_train_test_split(
        list(range(row_count)), test_fraction=test_fraction, seed=seed
    )
