import hashlib

from ragthoven import Ragthoven
from ragthoven.executors.output_writer import JSONLOutputWriter, SupportedOutputFormats
from ragthoven.models.base import Config
from ragthoven.models.iter_matrix import IterationMatrix
from ragthoven.utils import stringify_obj, stringify_obj_beautiful
from ragthoven.test import load_config, setup_env


def run_with_cfg(cfg_path: str, max_validation: int | None = 2):
    config = load_config(cfg_path)

    # Mocking the data
    if config.training_data is not None:
        config.training_data.dataset = config.training_data.dataset.replace(
            "csv:./data/", "csv:./ragthoven/test/test_data/"
        )

    config.validation_data.dataset = config.validation_data.dataset.replace(
        "csv:./data/", "csv:./ragthoven/test/test_data/"
    )
    iter_matrix = IterationMatrix(config)

    cont = True
    while cont:
        iter_matrix.build_config()
        current_config: Config = iter_matrix.get_config()

        # Process terminal options
        output_writer = None
        output_base_name = current_config.results.output_filename

        # write metadata
        object_stringified = stringify_obj(current_config)
        config_hash = hashlib.sha256(object_stringified.encode())
        hex_dig = str(config_hash.hexdigest())[:12]
        with open(f"{output_base_name}.{hex_dig}.metadata.json", "w") as metadata_file:
            metadata_file.write(stringify_obj_beautiful(current_config))

        filename = f"{output_base_name}.{hex_dig}.{SupportedOutputFormats.JSONL.value}"
        output_writer = JSONLOutputWriter(filename, current_config)

        r = Ragthoven(current_config, output_write=output_writer)

        if r.train_dataset is not None:
            r.train_dataset = r.train_dataset.select(range(20))

        if max_validation is not None:
            r.validation_dataset = r.validation_dataset.select(range(max_validation))
        r.execute()
        cont = iter_matrix.inc()


def test_intensive_comp_case2024_climate(setup_env):
    run_with_cfg("ragthoven/test/test_config/comp_case2024-climate.yaml")


def test_intensive_comp_case2024_climate_matrix_multiprompt_custom_examples(setup_env):
    run_with_cfg(
        "ragthoven/test/test_config/comp_case2024-climate_matrix_multiprompt_custom_examples.yaml"
    )


def test_intensive_comp_case2024_climate_matrix_multiprompt(setup_env):
    run_with_cfg(
        "ragthoven/test/test_config/comp_case2024-climate_matrix_multiprompt.yaml"
    )


def test_intensive_comp_case2024_climate_matrix_no_rerank(setup_env):
    run_with_cfg(
        "ragthoven/test/test_config/comp_case2024-climate_matrix_no_rerank.yaml"
    )


def test_intensive_comp_case2024_climate_matrix_no_retrieval(setup_env):
    run_with_cfg(
        "ragthoven/test/test_config/comp_case2024-climate_matrix_no_retrieval.yaml"
    )


def test_intensive_comp_case2024_climate_matrix_with_custom_examples(setup_env):
    run_with_cfg(
        "ragthoven/test/test_config/comp_case2024-climate_matrix_with_custom_examples.yaml"
    )


def test_intensive_comp_case2024_climate_matrix(setup_env):
    run_with_cfg("ragthoven/test/test_config/comp_case2024-climate_matrix.yaml")


def test_intensive_comp_case2024_climate(setup_env):
    run_with_cfg("ragthoven/test/test_config/comp_case2024-climate.yaml")


def single_shot_only(setup_env):
    run_with_cfg("ragthoven/test/test_config/single-shot-example.yaml")


def single_shot_only(setup_env):
    run_with_cfg("ragthoven/test/test_config/single-shot-example-preprocessor.yaml")
