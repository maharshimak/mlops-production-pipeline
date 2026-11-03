import random
from math import ceil

import pytest

from mlops_pipeline.splits import group_rank, group_train_test_split, row_train_test_split


def assignment(groups, split) -> dict[object, str]:
    sides: dict[object, set[str]] = {}
    for index in split.train_indices:
        sides.setdefault(groups[index], set()).add("train")
    for index in split.test_indices:
        sides.setdefault(groups[index], set()).add("test")
    assert all(len(side) == 1 for side in sides.values()), "group spans both sides"
    return {group: next(iter(side)) for group, side in sides.items()}


@pytest.mark.parametrize("seed", range(25))
def test_groups_never_span_train_and_test(seed: int) -> None:
    rng = random.Random(seed)
    groups = [rng.choice([f"user-{i}" for i in range(12)] + list(range(5))) for _ in range(80)]
    if len(set(groups)) < 2:
        groups[0], groups[1] = "a", "b"
    fraction = rng.choice([0.1, 0.2, 0.25, 0.3, 0.5])
    split = group_train_test_split(groups, test_fraction=fraction, seed=seed)

    assert set(split.train_indices).isdisjoint(split.test_indices)
    assert sorted(split.train_indices + split.test_indices) == list(range(len(groups)))
    assert split.train_indices and split.test_indices
    train_groups = {groups[i] for i in split.train_indices}
    test_groups = {groups[i] for i in split.test_indices}
    assert train_groups.isdisjoint(test_groups)
    assert split.train_group_count == len(train_groups)
    assert split.test_group_count == len(test_groups)
    assignment(groups, split)


def test_split_is_deterministic_and_seed_sensitive() -> None:
    groups = [f"e{i // 3}" for i in range(60)]
    first = group_train_test_split(groups, seed=7)
    second = group_train_test_split(groups, seed=7)
    assert first == second
    others = {group_train_test_split(groups, seed=s).test_indices for s in range(8, 16)}
    assert any(test != first.test_indices for test in others)


def test_assignment_is_independent_of_row_order() -> None:
    groups = [f"e{i % 17}" for i in range(85)]
    baseline = assignment(groups, group_train_test_split(groups, seed=3))
    for shuffle_seed in range(5):
        shuffled = list(groups)
        random.Random(shuffle_seed).shuffle(shuffled)
        assert assignment(shuffled, group_train_test_split(shuffled, seed=3)) == baseline


def test_string_and_integer_groups_are_distinct_entities() -> None:
    assert group_rank("1", 0) != group_rank(1, 0)


def test_rank_matches_documented_hash() -> None:
    from mlops_pipeline.canonical import canonical_sha256

    assert group_rank("abc", 9) == canonical_sha256({"seed": 9, "group": ["str", "abc"]})


def test_test_side_reaches_target_and_train_keeps_a_group() -> None:
    groups = [f"e{i}" for i in range(40)]
    split = group_train_test_split(groups, test_fraction=0.25, seed=1)
    assert len(split.test_indices) == ceil(40 * 0.25)
    two = group_train_test_split(["a"] * 9 + ["b"], test_fraction=0.5, seed=0)
    assert two.train_group_count == 1 and two.test_group_count == 1


def test_row_split_treats_each_row_as_a_group() -> None:
    split = row_train_test_split(20, test_fraction=0.25, seed=5)
    assert len(split.test_indices) == 5
    assert split.test_group_count == 5


@pytest.mark.parametrize(
    "groups,kwargs,error",
    [
        (["a"] * 5, {}, ValueError),
        (["a", "b"], {"test_fraction": 0.05}, ValueError),
        (["a", "b"], {"test_fraction": 0.6}, ValueError),
        (["a", "b"], {"seed": True}, TypeError),
        (["a", 1.5], {}, TypeError),
        (["a", None], {}, TypeError),
    ],
)
def test_invalid_split_inputs(groups, kwargs, error) -> None:
    with pytest.raises(error):
        group_train_test_split(groups, **kwargs)
