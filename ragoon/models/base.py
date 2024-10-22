from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EmbedderType(Enum):
    DEFAULT_CHROMA_EMBEDDER = "sbert"
    CDE_EMBEDDER = "cde"


@dataclass
class TrainingData:
    dataset: str
    input_feature: str
    split_name: str
    textual_labels: list[str]
    dataset_version: Optional[str] = None
    label_feature: Optional[str] = None
    output_feature: Optional[str] = None

    def __post_init__(self):

        # Output feature and label feature are alises
        if self.label_feature is None and self.output_feature is None:
            raise ValueError("You must provide label_feature or output_feature!")
        elif self.label_feature is None:
            self.label_feature = self.output_feature
        else:
            self.output_feature = self.label_feature


@dataclass
class ValidationData:
    input_feature: str
    split_name: str
    dataset: Optional[str] = (
        None  # this might look like the example: "json:data/data.jsonl"
    )


@dataclass
class Results:
    output_cached: Optional[str] = None
    output_cache_id: Optional[str] = None
    bad_request_default_value: Optional[str] = None
    output_filename: Optional[str] = (
        "results"  # should have no extension (i.e. no "results.jsonl")
    )


@dataclass
class Embed:
    k: int | list[int]
    training_size_limit: Optional[int] = None
    model: Optional[str] | Optional[list[str]] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedder: Optional[str] = EmbedderType.DEFAULT_CHROMA_EMBEDDER.value
    docs_embedding_count: Optional[int] = 10

    def __post_init__(self):
        valid_options = [e.value for e in EmbedderType]
        if self.embedder not in valid_options:
            raise ValueError(f"Wrong embedder specified {self.embedder}, valid options: {valid_options}")


@dataclass
class Rerank:
    k: int | list[int]
    model: Optional[str] | Optional[list[str]] = "ms-marco-MiniLM-L-12-v2"


@dataclass
class Prompt:
    name: str
    role: str
    prompt: str
    out: Optional[str] = None


@dataclass
class LLM:
    sprompt: Optional[str | list[str]] = None
    uprompt: Optional[str | list[str]] = None
    prompts: Optional[list[Prompt]] = None
    examples: Optional[str] = None
    model: Optional[str] | list[str] = "gpt-4o"
    base_url: Optional[str] = None
    temperature: Optional[float] | Optional[list[float]] = 0


@dataclass
class Config:
    name: str
    training_data: TrainingData
    validation_data: ValidationData
    results: Results
    embed: Embed | None
    rerank: Rerank | None
    llm: LLM
