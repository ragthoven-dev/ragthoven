import pytest
from dotenv import load_dotenv

from ragthoven import Ragthoven
from ragthoven.executors.output_writer import CSVOutputWriter, SupportedOutputFormats
from ragthoven.utils.config_loader import load_config


@pytest.fixture(scope="session")
def setup_env():
    load_dotenv(override=True)


def test_example_ag_news(setup_env):
    cfg_path = "ragthoven/test/test_config/example_ag_news.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_ag_news(setup_env):
    cfg_path = "ragthoven/test/test_config/example_ag_news_cde.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_bin_mgtd(setup_env):
    cfg_path = "ragthoven/test/test_config/example_bin_mgtd.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_bin_mgtd_out_f(setup_env):
    cfg_path = "ragthoven/test/test_config/example_bin_mgtd_out_f.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cnn_daily(setup_env):
    cfg_path = "ragthoven/test/test_config/example_cnn_daily.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cnn_daily_batched(setup_env):
    cfg_path = "ragthoven/test/test_config/example_cnn_daily_batch_size.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_dialogsum(setup_env):
    cfg_path = "ragthoven/test/test_config/example_dialogsum.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_imdb(setup_env):
    cfg_path = "ragthoven/test/test_config/example_imdb.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_stock_tweets(setup_env):
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cache_stock_tweets(setup_env):
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_csv_stock_tweets(setup_env):
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    filename = f"{config.results.output_filename}.{SupportedOutputFormats.CSV.value}"
    output_writer = CSVOutputWriter(filename, config)
    r = Ragthoven(config, output_write=output_writer)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_cache_csv_stock_tweets(setup_env):
    cfg_path = "ragthoven/test/test_config/example_stocks_tweets.yaml"
    config = load_config(cfg_path)
    filename = f"{config.results.output_filename}.{SupportedOutputFormats.CSV.value}"
    output_writer = CSVOutputWriter(filename, config)
    r = Ragthoven(config, output_write=output_writer)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_example_ag_news_alternative_models(setup_env):
    cfg_path = "ragthoven/test/test_config/example_ag_news_alt_model.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_config_single_shot_example(setup_end):
    cfg_path = "ragthoven/test/test_config/single-shot-example.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()


def test_config_single_shot_example(setup_end):
    cfg_path = "ragthoven/test/test_config/single-shot-example-preprocessor.yaml"
    config = load_config(cfg_path)
    r = Ragthoven(config)
    r.validation_dataset = r.validation_dataset.select(range(2))
    r.execute()
