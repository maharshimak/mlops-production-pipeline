import pytest


def make_records(entities: int = 20, rows_per_entity: int = 2) -> list[dict[str, object]]:
    """Synthetic, exactly linear grouped dataset: y = 2*x1 - 0.5*x2 + 3."""
    records: list[dict[str, object]] = []
    for index in range(entities * rows_per_entity):
        x1 = float(index)
        x2 = float(index % 3)
        records.append(
            {
                "x1": x1,
                "x2": x2,
                "y": 2.0 * x1 - 0.5 * x2 + 3.0,
                "customer": f"c{index // rows_per_entity:03d}",
            }
        )
    return records


SCHEMA_DICT = {"features": ["x1", "x2"], "label": "y", "group": "customer"}


@pytest.fixture
def records() -> list[dict[str, object]]:
    return make_records()


@pytest.fixture
def schema_dict() -> dict[str, object]:
    return {**SCHEMA_DICT, "features": list(SCHEMA_DICT["features"])}
