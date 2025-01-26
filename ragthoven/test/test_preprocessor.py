from ragthoven import Ragthoven
from ragthoven.executors.preprocessor import DataPreprocessor
from ragthoven.test.test_intensive import setup_env
from ragthoven.utils.config_loader import load_config


def test_config_single_shot_example(setup_env):
    cfg_path = "ragthoven/test/test_config/single-shot-example-preprocessor.yaml"
    config = load_config(cfg_path)

    mocked_preprocessor = MockedPreprocessor(config.preprocessor)
    r = Ragthoven(config, data_preprocessor=mocked_preprocessor)
    r.validation_dataset = r.validation_dataset.select(range(1))
    r.execute()


class MockedPreprocessor(DataPreprocessor):
    def preprocess(self, args):
        res = super().preprocess(args)

        assert res["fizzbuzz"] == "fizz"
        assert res["article"] == 4290
        assert res["and_countes"] == 0
