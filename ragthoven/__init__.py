import logging

import tqdm

from ragthoven.constants import PROMPT_LOGGING
from ragthoven.executors.cde_embedder import CDEEmbedder
from ragthoven.executors.embedder import BaseEmbedder, ChromaEmbedder
from ragthoven.executors.output_writer import (
    BaseOutputWriter,
    JSONLOutputWriter,
    SupportedOutputFormats,
)
from ragthoven.executors.preprocessor import BaseDataPreprocessor, DataPreprocessor
from ragthoven.executors.prompt_executor import (
    BasePromptExecutor,
    LiteLLMPromptExecutor,
)
from ragthoven.executors.prompt_formatter import (
    BaseExamplePromptFormatter,
    BasePromptFormatter,
)
from ragthoven.executors.reranker import BaseReranker, FlashRanker
from ragthoven.models.base import Config, EmbedderType
from ragthoven.utils.dataset_loader import dataset_load

logger = logging.getLogger()
logger.propagate = False
logger.setLevel(logging.ERROR)


class Ragthoven:
    def __init__(
        self,
        config: Config,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
        prompt_formatter: BasePromptFormatter | None = None,
        prompt_executor: BasePromptExecutor | None = None,
        data_preprocessor: BaseDataPreprocessor | None = None,
        output_write: BaseOutputWriter | None = None,
    ):
        self.config = config
        self.train_dataset = None
        self.validation_dataset = None

        self.train_dataset = None
        if self.config.embed is not None:
            if self.config.training_data is None:
                raise ValueError(
                    "When using embedder the training dataset must be provided"
                )

            self.train_dataset = dataset_load(
                self.config.training_data.dataset,
                self.config.training_data.dataset_version,
                split=self.config.training_data.split_name,
            )

        if (
            self.config.validation_data.dataset is None
        ):  # use training dataset as val src
            self.validation_dataset = dataset_load(
                self.config.training_data.dataset,
                self.config.training_data.dataset_version,
                split=self.config.validation_data.split_name,
            )
        else:
            self.validation_dataset = dataset_load(
                self.config.validation_data.dataset,
                (
                    self.config.validation_data.dataset_version
                    if self.config.validation_data.dataset_version is not None
                    else ""
                ),
                split=self.config.validation_data.split_name,
            )

        # Embedding texts into db
        self.embedder = None
        if embedder is None and self.config.embed is not None:
            if self.config.embed.embedder == EmbedderType.CDE_EMBEDDER.value:
                self.embedder = CDEEmbedder(self.config)
            else:
                self.embedder = ChromaEmbedder(self.config)
        else:
            self.embedder = embedder

        # Reranker
        self.reranker = None
        if reranker is None and self.config.rerank is not None:
            self.reranker = FlashRanker(self.config)
        else:
            self.reranker = reranker

        # Data preprocessor
        self.data_preprocessor = None
        if data_preprocessor is None and self.config.preprocessor is not None:
            self.data_preprocessor = DataPreprocessor(self.config.preprocessor)
        else:
            self.data_preprocessor = data_preprocessor

        # Prompt formatter
        if prompt_formatter is None:
            self.pformater = BaseExamplePromptFormatter(self.config)
        else:
            self.pformater = prompt_formatter

        self.pformater.set_train_dataset(self.train_dataset)

        if self.config.llm.sprompt is not None and self.config.llm.uprompt is not None:
            self.pformater.set_prompts(
                [self.config.llm.sprompt, self.config.llm.uprompt]
            )
        else:
            self.pformater.set_prompts(self.config.llm.prompts)

        # Prompt executor
        if prompt_executor is None:
            self.pexecutor = LiteLLMPromptExecutor(
                config=self.config,
            )
        else:
            self.pexecutor = prompt_executor

        # Output Writer
        if output_write is None:
            output_file = (
                self.config.results.output_filename
                + "."
                + SupportedOutputFormats.JSONL.value
            )
            self.output_write = JSONLOutputWriter(output_file, self.config)
        else:
            self.output_write = output_write

    def execute(self):
        processed_ids = self.output_write.get_processed_ids()

        if self.embedder is not None:
            self.embedder.set_training_dataset(self.train_dataset)
            self.embedder.embedd()

        for i in tqdm.tqdm(
            range(
                len(self.validation_dataset[self.config.validation_data.input_feature])
            )
        ):
            if self.config.results.output_cache_id is not None:
                id = self.validation_dataset[self.config.results.output_cache_id][i]
            else:
                id = str(i)

            if self.config.results.output_cached and id in processed_ids:
                continue

            text = self.validation_dataset[self.config.validation_data.input_feature][i]
            all_features = {
                key: self.validation_dataset[key][i]
                for key in self.validation_dataset.features.keys()
            }

            if self.data_preprocessor is not None:
                all_features = self.data_preprocessor.preprocess(all_features)

            if self.config.llm.prompts is not None:
                named_prompts_with_output = {p.name: p for p in self.config.llm.prompts}
                for prompt in self.config.llm.prompts:
                    if prompt.name == "system":
                        continue

                    sprompt, uprompt = self.pformater.format_multiple(
                        text,
                        all_features,
                        prompt.name,
                        named_prompts_with_output,
                        self.embedder,
                        self.reranker,
                        self.config,
                    )
                    pres = self.pexecutor.get_prompt_results(
                        sprompt, uprompt, prompt.tools
                    )

                    if i == 0 and self.config.llm.log_first:
                        print(f"Logging on first data: {sprompt}, {uprompt}, {pres}")

                    named_prompts_with_output[prompt.name].out = pres
                self.output_write.append(pres, id)

            else:
                sprompt, uprompt = self.pformater.format_simple(
                    text, all_features, self.embedder, self.reranker, self.config
                )
                pres = self.pexecutor.get_prompt_results(sprompt, uprompt)
                if i == 0 and self.config.llm.log_first:
                    print(
                        PROMPT_LOGGING.format(
                            sprompt=sprompt, uprompt=uprompt, pres=pres
                        )
                    )
                self.output_write.append(pres, id)

        self.output_write.close()
