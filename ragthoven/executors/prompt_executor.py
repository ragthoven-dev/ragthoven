import abc
import json

import litellm

from ragthoven.models.base import Config
from ragthoven.tools import BaseFunCalling
from ragthoven.utils import get_class, get_class_func_name_only


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

        self.available_tools = {}
        if self.config.llm.tools is not None and len(self.config.llm.tools) > 0:
            for tool in self.config.llm.tools:
                tool_name = get_class_func_name_only(tool)
                self.available_tools[tool_name] = get_class("ragthoven.tools", tool)()

    def get_prompt_results(self, sprompt, uprompt, tools=None):
        from litellm import completion

        messages = [
            {"content": sprompt, "role": "system"},
            {"content": uprompt, "role": "user"},
        ]

        to_model_tools = None
        if tools is not None:
            to_model_tools = []
            for tool in tools:
                tool_class: BaseFunCalling = self.available_tools[tool]
                to_model_tools.append(tool_class.get_json_schema())

        try:
            response = completion(
                model=self.config.llm.model,
                messages=messages,
                tools=to_model_tools,
                **self.model_params,
            )
        except litellm.BadRequestError as e:
            return self.config.bad_request_default_value

        tool_calls = response.choices[0].message.tool_calls
        results = ""
        if tool_calls:
            for tool_call in tool_calls:
                fn_to_call = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                function_result = self.available_tools[fn_to_call](arguments)
                results += f"\nFunction {fn_to_call}, resulted in following result: {function_result}\n"

        results += (
            response.choices[0].message.content
            if response.choices[0].message.content is not None
            else ""
        )
        return results
