import abc

from datasets import Dataset

from ragthoven.executors.embedder import BaseEmbedder
from ragthoven.executors.reranker import BaseReranker
from ragthoven.models.base import Config, Prompt


class BasePromptFormatter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def format_simple(self):
        raise NotImplementedError(
            f"PromptFormatter {self} did not implement embed method!"
        )

    @abc.abstractmethod
    def format_multiple(self):
        raise NotImplementedError(
            f"PromptFormatter {self} did not implement embed method!"
        )

    @abc.abstractmethod
    def set_train_dataset(self, tr_dataset: Dataset):
        raise NotImplementedError(
            f"PromptFormatter {self} did not implement embed method!"
        )

    @abc.abstractmethod
    def set_prompts(self):
        raise NotImplementedError(
            f"PromptFormatter {self} did not implement embed method!"
        )


class BaseExamplePromptFormatter(BasePromptFormatter):
    def __init__(self, config: Config, sort_asc: bool = True) -> None:
        self.sort_asc = sort_asc
        self.config = config
        self.examples_cache = None
        self.examples_cache_id = None

    def set_prompts(self, prompts: dict[str, Prompt] | dict[str, str] | None):
        from jinja2 import Template

        self.prompt_templates = {}

        if self.config.llm.uprompt is not None and self.config.llm.sprompt is not None:
            self.prompt_templates["system"] = Template(prompts[0])
            self.prompt_templates["user"] = Template(prompts[1])
        else:
            for prompt in prompts:
                self.prompt_templates[prompt.name] = Template(prompt.prompt)

        self.examples_template = None
        if self.config.llm.examples is not None:
            self.examples_template = Template(self.config.llm.examples)

    def set_train_dataset(self, tr_dataset: Dataset):
        self.training_dataset = tr_dataset

    def build_examples(
        self, text: str, embedder: BaseEmbedder, reranker: BaseReranker, config: Config
    ):
        if self.examples_cache_id == text:
            return self.examples_cache

        examples = ""
        if embedder is not None:
            res = embedder.get_similar(text, config.embed.k)
            c_ids = res["ids"][0]
            c_docs = res["documents"][0]
            passages = [{"id": c_ids[i], "text": c_docs[i]} for i in range(len(c_docs))]

            if reranker is not None:
                reranked_resutls = reranker.get_similar(text, passages, config.rerank.k)
            else:
                reranked_resutls = passages

            ids = list(map(lambda x: int(x["id"]), reranked_resutls))
            texts = list(map(lambda x: x["text"], reranked_resutls))
            label_ids = [
                self.training_dataset[config.training_data.label_feature][i]
                for i in ids
            ]

            passage_all_values = [self.training_dataset[i] for i in ids]

            # Change label ids to (numerical index)
            if (
                len(config.training_data.textual_labels) > 0
            ):  # to label specified in config
                labels_textual = list(
                    map(lambda x: config.training_data.textual_labels[x], label_ids)
                )
            else:  # or don't if no are provided
                labels_textual = label_ids

            examples = ""
            if self.config.llm.examples is None:
                for i in range(len(texts)):
                    idx = i if self.sort_asc else len(texts) - 1 - i
                    examples += f"text: {texts[idx]}\nlabel:{labels_textual[idx]}\n\n"
            else:
                examples_enumerated = [
                    {
                        "text": texts[i],
                        "label": labels_textual[i],
                        "data": passage_all_values[i],
                    }
                    for i in range(len(texts))
                ]
                examples = self.examples_template.render(examples=examples_enumerated)

            self.examples_cache = examples
            self.examples_cache_id = text
            return self.examples_cache

    def format_prompt(
        self, text, all_features, prompt_id, prev_prompt_out_states, examples
    ):
        return self.prompt_templates[prompt_id].render(
            examples=examples, text=text, data=all_features, **prev_prompt_out_states
        )

    def format_multiple(
        self,
        text: str,
        all_features: dict[str, any],
        prompt_id: str,
        prev_prompt_out_states: dict[str, Prompt],
        examples,
    ):
        sprompt = self.prompt_templates["system"].render(examples=examples, text=text)
        uprompt = self.prompt_templates[prompt_id].render(
            examples=examples, text=text, **prev_prompt_out_states, data=all_features
        )

        return sprompt, uprompt

    def format_simple(self, text: str, all_features: dict[str, any], examples):
        sprompt = self.prompt_templates["system"].render(
            text=text, examples=examples, data=all_features
        )
        uprompt = self.prompt_templates["user"].render(
            text=text, examples=examples, data=all_features
        )
        return [sprompt, uprompt]
