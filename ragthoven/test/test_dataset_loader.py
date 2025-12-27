from pathlib import Path

from ragthoven.utils.dataset_loader import dataset_load


def test_dataset_load_tsv():
    data_path = Path("ragthoven/test/test_data/math_expressions.tsv")
    dataset = dataset_load(f"tsv:{data_path}", version="", split="train")

    assert dataset.num_rows == 2
    assert dataset.column_names == ["expression", "expected"]


def test_dataset_load_csv():
    data_path = Path("ragthoven/test/test_data/math_expressions.csv")
    dataset = dataset_load(f"csv:{data_path}", version="", split="train")

    assert dataset.num_rows >= 2
    assert dataset.column_names == ["expression", "expected"]
