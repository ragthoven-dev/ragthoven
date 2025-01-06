import abc
import os

import litellm

from ragthoven.models.base import Config


class BasePromptExecutor(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_prompt_results(self):
        raise NotImplementedError(f"Prompt {self} did not implement promot")


class LiteLLMPromptExecutor(BasePromptExecutor):
    def __init__(self, config: Config = None):
        self.config = config
        self.model_params = {
            "temperature": self.config.llm.temperature,
            "base_url": self.config.llm.base_url,
        }

    def get_prompt_results(self, sprompt, uprompt):
        from litellm import completion

        messages = [
            {"content": sprompt, "role": "system"},
            {"content": uprompt, "role": "user"},
        ]
        try:
            response = completion(
                model=self.config.llm.model, messages=messages, **self.model_params
            )
        except litellm.BadRequestError as e:
            return self.config.bad_request_default_value

        return response.choices[0].message.content
