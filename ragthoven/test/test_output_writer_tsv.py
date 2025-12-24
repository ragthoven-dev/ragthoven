from ragthoven.executors.output_writer import TSVOutputWriter
from ragthoven.test import load_config


def test_tsv_output_writer_respects_output_field(tmp_path):
    cfg_path = "ragthoven/test/test_config/example_ag_news.yaml"
    config = load_config(cfg_path)
    config.results.output_field = "text"

    output_path = tmp_path / "example.tsv"
    writer = TSVOutputWriter(str(output_path), config)
    writer.append("hello", "abc")
    writer.close()

    lines = output_path.read_text().splitlines()
    assert lines[0] == "id\ttext"
    assert lines[1] == "abc\thello"


def test_tsv_output_writer_respects_output_cached(tmp_path):
    cfg_path = "ragthoven/test/test_config/example_ag_news.yaml"
    config = load_config(cfg_path)
    config.results.output_field = "text"
    config.results.output_cached = True

    output_path = tmp_path / "cached.tsv"
    output_path.write_text("id\ttext\nabc\thello\n", encoding="utf-8")

    writer = TSVOutputWriter(str(output_path), config)
    assert writer.get_processed_ids() == {"abc"}
    writer.append("world", "def")
    writer.close()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["id\ttext", "abc\thello", "def\tworld"]
