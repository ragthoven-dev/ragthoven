import csv
import json

from ragthoven.executors.output_writer import CSVOutputWriter, JSONLOutputWriter
from ragthoven.models.base import Config, LLM, Results, ValidationData


def _config(output_field: str | None = None) -> Config:
    results = Results(output_field=output_field) if output_field is not None else Results()
    return Config(
        name="output-writer-test",
        training_data=None,
        validation_data=ValidationData(input_feature="text", split_name="test"),
        results=results,
        embed=None,
        rerank=None,
        llm=LLM(),
        preprocessor=None,
        iterative=None,
    )


def test_jsonl_output_field_override(tmp_path):
    config = _config(output_field="prediction")
    output_path = tmp_path / "out.jsonl"
    writer = JSONLOutputWriter(str(output_path), config)
    writer.append("yes", "1")
    writer.close()

    data = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert data["prediction"] == "yes"
    assert data["id"] == "1"
    assert "label" not in data


def test_csv_output_field_override(tmp_path):
    config = _config(output_field="prediction")
    output_path = tmp_path / "out.csv"
    writer = CSVOutputWriter(str(output_path), config)
    writer.append("yes", "1")
    writer.close()

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows[0] == ["id", "prediction"]
    assert rows[1] == ["1", "yes"]


def test_jsonl_output_field_default_label(tmp_path):
    config = _config()
    output_path = tmp_path / "default.jsonl"
    writer = JSONLOutputWriter(str(output_path), config)
    writer.append("yes", "1")
    writer.close()

    data = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert data["label"] == "yes"
    assert data["id"] == "1"
