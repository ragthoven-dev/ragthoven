from ragthoven import Ragthoven
from ragthoven.executors.output_writer import CSVOutputWriter, SupportedOutputFormats
from ragthoven.test import load_config


def test_example_ag_news():
    cfg_path = "ragthoven/test/test_config/example_ag_news.yaml"
    config = load_config(cfg_path)

    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_ag_news():
    cfg_path = "ragthoven/test/test_config/example_ag_news_cde.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_bin_mgtd():
    cfg_path = "ragthoven/test/test_config/example_bin_mgtd.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_bin_mgtd_out_f():
    cfg_path = "ragthoven/test/test_config/example_bin_mgtd_out_f.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cnn_daily():
    cfg_path = "ragthoven/test/test_config/example_cnn_daily.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cnn_daily_batched():
    cfg_path = "ragthoven/test/test_config/example_cnn_daily_batch_size.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_dialogsum():
    cfg_path = "ragthoven/test/test_config/example_dialogsum.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_imdb():
    cfg_path = "ragthoven/test/test_config/example_imdb.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_stock_tweets():
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cache_stock_tweets():
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_csv_stock_tweets():
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    filename = f"{config.results.output_filename}.{SupportedOutputFormats.CSV.value}"
    output_writer = CSVOutputWriter(filename, config)
    r = Ragthoven(config, output_write=output_writer)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cache_csv_stock_tweets():
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    filename = f"{config.results.output_filename}.{SupportedOutputFormats.CSV.value}"
    output_writer = CSVOutputWriter(filename, config)
    r = Ragthoven(config, output_write=output_writer)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_ag_news_alternative_models():
    cfg_path = "ragthoven/test/test_config/example_ag_news_alt_model.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_config_single_shot_example():
    cfg_path = "ragthoven/test/test_config/single-shot-example.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_config_single_shot_example():
    cfg_path = "ragthoven/test/test_config/single-shot-example-preprocessor.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()
