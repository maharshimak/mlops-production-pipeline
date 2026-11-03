"""Markdown model card rendered strictly from a verified run record.

Every value in the card is read from the run record; sections for information
the record does not contain (intended use, owners, fairness analysis, ...) are
listed explicitly as "not recorded" rather than filled with invented text.
"""

from __future__ import annotations

from collections.abc import Mapping

from mlops_pipeline.experiment import verify_record

NOT_RECORDED = (
    "Intended use, out-of-scope use, owners, training-data provenance beyond the "
    "dataset hash, fairness and subgroup analysis are not part of the run record "
    "and are therefore not stated here."
)


def _section(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"run record section {key!r} must be an object")
    return value


def _fmt(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[tuple[str, object]]) -> list[str]:
    lines = ["| Field | Value |", "| --- | --- |"]
    lines.extend(f"| {_fmt(name)} | {_fmt(value)} |" for name, value in rows)
    return lines


def model_card_markdown(record: object) -> str:
    verified = verify_record(record)
    dataset = _section(verified, "dataset")
    split = _section(verified, "split")
    metrics = _section(verified, "metrics")
    gate = _section(verified, "gate")
    artifact = _section(verified, "artifact")
    config = _section(verified, "config")
    drift = verified.get("drift")
    if not isinstance(drift, list):
        raise TypeError("run record section 'drift' must be a list")
    features = dataset.get("features")
    if not isinstance(features, list):
        raise TypeError("run record dataset.features must be a list")
    reasons = gate.get("reasons")
    if not isinstance(reasons, list):
        raise TypeError("run record gate.reasons must be a list")

    lines = [f"# Model card: {_fmt(verified.get('model_id'))}", ""]
    lines += ["## Model", ""]
    lines += _table(
        [
            ("model_id", verified.get("model_id")),
            ("model_name", verified.get("model_name")),
            ("model_type", verified.get("model_type")),
            ("artifact_sha256", artifact.get("sha256")),
            ("artifact_size_bytes", artifact.get("size_bytes")),
            ("code_version", verified.get("code_version")),
            ("code_version_source", verified.get("code_version_source")),
            ("code_tree_dirty", verified.get("code_tree_dirty")),
            ("run_record_sha256", verified.get("record_sha256")),
        ]
    )
    lines += ["", "## Training data", ""]
    lines += _table(
        [
            ("dataset_sha256", dataset.get("sha256")),
            ("row_count", dataset.get("row_count")),
            ("features", ", ".join(_fmt(name) for name in features)),
            ("label", dataset.get("label")),
            ("group_column", dataset.get("group")),
        ]
    )
    lines += ["", "## Split", ""]
    lines += _table([(str(key), split[key]) for key in sorted(split)])
    lines += ["", "## Configuration", "", f"config_sha256: `{_fmt(verified.get('config_sha256'))}`", ""]
    lines += _table([(str(key), config[key]) for key in sorted(config)])
    lines += ["", "## Evaluation metrics (held-out test split)", ""]
    lines += _table([(str(key), metrics[key]) for key in sorted(metrics)])
    lines += ["", "## Drift (PSI, train reference vs test)", ""]
    lines += ["| Feature | PSI |", "| --- | --- |"]
    for item in drift:
        if not isinstance(item, Mapping):
            raise TypeError("run record drift entries must be objects")
        lines.append(f"| {_fmt(item.get('feature'))} | {_fmt(item.get('psi'))} |")
    lines += ["", "## Quality gate", "", f"allowed: {_fmt(gate.get('allowed'))}", ""]
    if reasons:
        lines += [f"- {_fmt(reason)}" for reason in reasons]
    else:
        lines.append("- no gate failures recorded")
    lines += ["", "## Not recorded", "", NOT_RECORDED, ""]
    return "\n".join(lines)
