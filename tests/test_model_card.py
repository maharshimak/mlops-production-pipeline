import json
import re

import pytest
from conftest import make_records

from mlops_pipeline.datasets import DatasetSchema
from mlops_pipeline.experiment import record_sha256, run_experiment
from mlops_pipeline.model_card import model_card_markdown

SCHEMA = DatasetSchema(features=("x1", "x2"), label="y", group="customer")


def _record() -> dict:
    return run_experiment(make_records(), SCHEMA, code_version="card-test").record


def _leaf_values(value) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _leaf_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _leaf_values(child)]
    if isinstance(value, float):
        return [repr(value)]
    return [str(value)]


def test_model_card_states_record_values() -> None:
    record = _record()
    card = model_card_markdown(record)
    assert card.startswith(f"# Model card: {record['model_id']}")
    for expected in (
        record["dataset"]["sha256"],
        record["config_sha256"],
        record["artifact"]["sha256"],
        record["record_sha256"],
        "card-test",
        repr(record["metrics"]["test_mae"]),
    ):
        assert expected in card


def test_every_number_in_card_comes_from_the_record() -> None:
    record = _record()
    card = model_card_markdown(record)
    known = set(_leaf_values(record))
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])", card)
    unexplained = {n for n in numbers if n not in known and not any(n in k for k in known)}
    assert not unexplained


def test_model_card_is_deterministic_and_lists_unrecorded_sections() -> None:
    record = _record()
    assert model_card_markdown(record) == model_card_markdown(json.loads(json.dumps(record)))
    assert "## Not recorded" in model_card_markdown(record)


def test_model_card_refuses_tampered_record() -> None:
    record = _record()
    record["gate"]["allowed"] = not record["gate"]["allowed"]
    with pytest.raises(ValueError, match="does not match"):
        model_card_markdown(record)


def test_model_card_escapes_table_breaking_values() -> None:
    record = _record()
    record["model_name"] = "a|b\nc"
    record["record_sha256"] = record_sha256(record)
    card = model_card_markdown(record)
    assert "| model_name | a\\|b c |" in card
