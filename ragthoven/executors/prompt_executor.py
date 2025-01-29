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

    def get_tools(self, tools):
        to_model_tools = None
        if tools is not None:
            to_model_tools = []
            for tool in tools:
                tool_class: BaseFunCalling = self.available_tools[tool]
                to_model_tools.append(tool_class.get_json_schema())

        return to_model_tools

    def get_messages_prompt_results(self, messages, tools):
        from litellm import completion

        to_model_tools = self.get_tools(tools)
        response = None

        try:
            response = completion(
                model=self.config.llm.model,
                messages=messages,
                tools=to_model_tools,
                **self.model_params,
            )
        except litellm.BadRequestError as e:
            print(f"What went wrong??? {e}")
            return self.config.results.bad_request_default_value

        return response

    def get_all_function_calls(self, tool_calls):
        resulting_calls = []
        if tool_calls:
            for tool_call in tool_calls:
                fn_to_call = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                function_result = self.available_tools[fn_to_call](arguments)
                resulting_calls.append((fn_to_call, function_result, tool_call.id))
        return resulting_calls

    def get_prompt_results(self, sprompt, uprompt, tools=None):
        messages = [
            {"content": sprompt, "role": "system"},
            {"content": uprompt, "role": "user"},
        ]

        response = self.get_messages_prompt_results(messages, tools)
        if response == self.config.results.bad_request_default_value:
            return response

        tool_calls = response.choices[0].message.tool_calls
        tool_called = self.get_all_function_calls(tool_calls)
        results = "".join(
            [
                f"\nFunction {fn_to_call}, resulted in following result: {function_result}\n"
                for fn_to_call, function_result, _ in tool_called
            ]
        )

        results += (
            response.choices[0].message.content
            if response.choices[0].message.content is not None
            else ""
        )
        return results
