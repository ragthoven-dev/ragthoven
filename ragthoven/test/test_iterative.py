import csv
import hashlib
import json
import math
from pathlib import Path

from ragthoven.models.iter_matrix import IterationMatrix
from ragthoven.test import load_config
from ragthoven.test.test_intensive import run_with_cfg
from ragthoven.utils import stringify_obj


def _coerce_number(value: str):
    stripped = value.strip()
    if stripped.lower().startswith("result:"):
        stripped = stripped.split(":", 1)[1].strip()
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def _read_expected():
    data_path = Path("ragthoven/test/test_data/math_expressions.csv")
    with data_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [_coerce_number(row["expected"]) for row in reader]


def _output_path_for(cfg_path: str) -> Path:
    """
    Recompute the deterministic output filename the same way run_with_cfg does.
    """
    config = load_config(cfg_path)

    if config.training_data is not None:
        config.training_data.dataset = config.training_data.dataset.replace(
            "csv:./data/", "csv:./ragthoven/test/test_data/"
        )
    config.validation_data.dataset = config.validation_data.dataset.replace(
        "csv:./data/", "csv:./ragthoven/test/test_data/"
    )

    iter_matrix = IterationMatrix(config)
    iter_matrix.build_config()
    current_config = iter_matrix.get_config()

    object_stringified = stringify_obj(current_config)
    config_hash = hashlib.sha256(object_stringified.encode())
    hex_dig = str(config_hash.hexdigest())[:12]
    output_base_name = current_config.results.output_filename
    return Path(f"{output_base_name}.{hex_dig}.jsonl")


def _read_predictions(output_path: Path):
    preds = []
    with output_path.open() as jsonl_file:
        for line in jsonl_file:
            payload = json.loads(line)
            preds.append(
                (
                    int(payload["id"]),
                    _coerce_number(str(payload["label"])),
                )
            )
    preds.sort(key=lambda item: item[0])
    return [pred for _, pred in preds]


def _assert_predictions_match(cfg_path: str):
    output_path = _output_path_for(cfg_path)
    run_with_cfg(cfg_path, max_validation=None)
    predictions = _read_predictions(output_path)
    expected = _read_expected()

    assert len(predictions) == len(expected)

    for pred, exp in zip(predictions, expected):
        if isinstance(exp, float):
            assert isinstance(pred, (int, float))
            assert math.isclose(float(pred), exp, rel_tol=1e-4, abs_tol=1e-4)
        else:
            assert pred == exp


def test_iterative_math_plus_self_verification():
    _assert_predictions_match("ragthoven/test/test_config/example_math_self_verification.yaml")


def test_iterative_math_run_produces_jsonl():
    _assert_predictions_match("ragthoven/test/test_config/example_math_calculator.yaml")
