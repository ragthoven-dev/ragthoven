import yaml

from ragthoven.models.base import (
    LLM,
    Config,
    Embed,
    Prompt,
    Rerank,
    Results,
    TrainingData,
    ValidationData,
)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        return process_config(config_dict)


def process_config(config_dict: dict):
    # read nested keys
    config_dict["training_data"] = TrainingData(**config_dict["training_data"])
    config_dict["validation_data"] = ValidationData(**config_dict["validation_data"])
    config_dict["embed"] = (
        Embed(**config_dict["embed"]) if config_dict.get("embed") is not None else None
    )
    config_dict["rerank"] = (
        Rerank(**config_dict["rerank"])
        if config_dict.get("rerank") is not None
        else None
    )
    config_dict["llm"] = LLM(**config_dict["llm"])
    config_dict["llm"].prompts = (
        [Prompt(**pvals) for pvals in config_dict["llm"].prompts]
        if config_dict["llm"].prompts is not None
        else None
    )
    # 'results' key needs special handling because all of its children are optional
    if config_dict["results"] is None:
        config_dict["results"] = Results()
    else:
        config_dict["results"] = Results(**config_dict["results"])

    return Config(**config_dict)
