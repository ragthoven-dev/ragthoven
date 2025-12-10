import inspect
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from ragthoven.tools.reasoning_tools import ReturnResult, ReturnResultWrapper
from ragthoven.utils.dataset_loader import dataset_load
from ragthoven.utils import get_class, get_class_func_name_only

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

    def _instantiate_tools(self, tool_configs):
        """
        Instantiate tools defined in config.iterative.tools with dependency injection.
        """
        tools = {}
        if tool_configs is None:
            return tools

        available_deps = {
            "prompt_executor": self.pexecutor,
            "embedder": self.embedder,
            "reranker": self.reranker,
        }

        for cfg in tool_configs:
            model_override = None
            cls = None

            if isinstance(cfg, dict):
                name = cfg.get("name")
                model_override = cfg.get("model")
            elif inspect.isclass(cfg):
                cls = cfg
                # Normalize to the dotted path expected by get_class
                name = f"{cls.__module__.split('ragthoven.tools.', 1)[-1]}.{cls.__name__}"
            else:
                name = cfg

            if name is None:
                raise ValueError("Tool configuration is missing 'name'")

            if "." not in name:
                raise ValueError(
                    "Tool name must include its module prefix, e.g. 'reasoning_tools.Calculator'"
                )

            if cls is None:
                cls = get_class("ragthoven.tools", name)
                class_name = get_class_func_name_only(name)
            else:
                class_name = cls.__name__

            requires = getattr(cls, "requires", [])
            kwargs = {}
            for dep in requires:
                if dep not in available_deps:
                    raise ValueError(f"Unsupported dependency requested: {dep}")
                kwargs[dep] = available_deps[dep]
            if model_override is not None:
                kwargs["model_override"] = model_override

            tools[class_name] = cls(**kwargs)

        return tools

    def _get_tool_schemas(self, tools: dict):
        return [tool.get_json_schema() for tool in tools.values()]

    def _execute_tool(self, tool_call, tools: dict):
        try:
            tool = tools[tool_call.function.name]
        except KeyError:
            return f"ERROR: Unknown tool '{tool_call.function.name}'"

        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as exc:
            return f"ERROR: InvalidArguments: {exc}"

        print(
            f"[RAGTHOVEN][TOOL] call={tool_call.function.name} args={args}"
        )

        try:
            result = tool(args)
            print(f"[RAGTHOVEN][TOOL] result={result}")

            # When the ReturnResult tool is invoked, record the final answer so the loop can stop.
            if tool_call.function.name == ReturnResult.__name__:
                return result

            return result
        except Exception as exc:
            print(f"[RAGTHOVEN][TOOL] error={type(exc).__name__}: {exc}")
            return f"ERROR: {type(exc).__name__}: {exc}"

    def execute_iterative_loop(self, i, text, all_features):
        """
        Iterative execution mode: let the LLM call tools in a loop until it stops.
        """
        tool_cfgs = []
        if self.config.iterative and self.config.iterative.tools:
            tool_cfgs = list(self.config.iterative.tools)
        tool_cfgs.append(ReturnResult)

        tools = self._instantiate_tools(tool_cfgs if self.config.iterative else None)
        max_iter = (
            self.config.iterative.max_iterations
            if self.config.iterative is not None
            else 1
        )
        
        examples = self.pformater.build_examples(
            text, self.embedder, self.reranker, self.config
        )

        sprompt, uprompt = self.pformater.format_simple(text, all_features, examples)
        messages = [
            {"role": "system", "content": sprompt},
            {"role": "user", "content": uprompt},
        ]

        last_message = None
        for _ in range(max_iter):
            response = self.pexecutor.get_messages_prompt_results(
                messages, tools=self._get_tool_schemas(tools)
            )

            if response == self.config.results.bad_request_default_value:
                return str(self.config.results.bad_request_default_value)

            last_message = response.choices[0].message

            if not last_message.tool_calls:
                return last_message.content

            messages.append(last_message)

            for tool_call in last_message.tool_calls:
                result = self._execute_tool(tool_call, tools)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

                if isinstance(result, ReturnResultWrapper):
                    return str(result.result)

        # Return the last assistant content as a fallback if max_iter reached
        return last_message.content if last_message else None

    def execute_single_prompt(self, i, text, all_features):
        examples = self.pformater.build_examples(
            text, self.embedder, self.reranker, self.config
        )
        sprompt, uprompt = self.pformater.format_simple(text, all_features, examples)
        pres = self.pexecutor.get_prompt_results(sprompt, uprompt)
        if i == 0 and self.config.llm.log_first:
            print(PROMPT_LOGGING.format(sprompt=sprompt, uprompt=uprompt, pres=pres))
        return pres

    def execute_messages_prompts(self, j, text, all_features):
        examples = self.pformater.build_examples(
            text, self.embedder, self.reranker, self.config
        )
        prompts = self.config.llm.prompts
        system_prompt = prompts[0]
        named_prompts_with_output = {p.name: p for p in self.config.llm.prompts}

        if system_prompt.name != "system" or system_prompt.role != "system":
            raise ValueError("First prompt is not a system prompt")

        messages = [
            {
                "role": system_prompt.role,
                "content": self.pformater.format_prompt(
                    text,
                    all_features,
                    system_prompt.name,
                    named_prompts_with_output,
                    examples,
                ),
            },
        ]

        for i in range(1, len(prompts)):
            messages.append(
                {
                    "role": prompts[i].role,
                    "content": self.pformater.format_prompt(
                        text,
                        all_features,
                        prompts[i].name,
                        named_prompts_with_output,
                        examples,
                    ),
                }
            )

            response = self.pexecutor.get_messages_prompt_results(
                messages, tools=prompts[i].tools
            )

            if response == self.config.results.bad_request_default_value:
                messages.append({"role": "assistant", "content": str(response)})

                # return response if the prompt is last
                if i == len(prompts) - 1:
                    return str(self.config.results.bad_request_default_value)
                continue

            model_response = response.choices[0].message
            named_prompts_with_output[prompts[i].name].out = model_response

            tool_calls = model_response.tool_calls
            processed_tool_calls = self.pexecutor.get_all_function_calls(tool_calls)

            if tool_calls:
                messages.append(model_response)
                for tool_call in processed_tool_calls:
                    _, function_result, tool_call_id = tool_call
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": function_result,
                        }
                    )
            else:
                messages.append(
                    {"role": "assistant", "content": model_response.content}
                )

        # In case the last prompt has a tool call, similar to the example from OpenAI
        # https://platform.openai.com/docs/guides/function-calling?example=get-weather
        if model_response.tool_calls is not None and len(model_response.tool_calls) > 0:
            response = self.pexecutor.get_messages_prompt_results(
                messages, tools=prompts[len(prompts) - 1].tools
            )
            if response == self.config.results.bad_request_default_value:
                messages.append({"role": "assistant", "content": str(response)})
            else:
                model_response = response.choices[0].message
                messages.append(
                    {"role": "assistant", "content": model_response.content}
                )

        if j == 0 and self.config.llm.log_first:
            debug_messages = []
            for message in messages:
                if type(message) is not dict:
                    debug_messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                tool_call.function.name
                                for tool_call in message.tool_calls
                            ],
                            "arguments": [
                                tool_call.function.arguments
                                for tool_call in message.tool_calls
                            ],
                        }
                    )
                else:
                    debug_messages.append(message)
            print(json.dumps(debug_messages, indent=2))

        if response == self.config.results.bad_request_default_value:
            return str(self.config.results.bad_request_default_value)

        return model_response.content

    def execute_sequential_prompts(self, i, text, all_features):
        named_prompts_with_output = {p.name: p for p in self.config.llm.prompts}
        examples = self.pformater.build_examples(
            text, self.embedder, self.reranker, self.config
        )

        for prompt in self.config.llm.prompts:
            if prompt.name == "system":
                continue
            sprompt, uprompt = self.pformater.format_multiple(
                text, all_features, prompt.name, named_prompts_with_output, examples
            )
            pres = self.pexecutor.get_prompt_results(sprompt, uprompt, prompt.tools)

            if i == 0 and self.config.llm.log_first:
                print(
                    PROMPT_LOGGING.format(sprompt=sprompt, uprompt=uprompt, pres=pres)
                )
            named_prompts_with_output[prompt.name].out = pres

        return pres

    def process_validation_example(self, index, processed_ids):
        if self.config.results.output_cache_id is not None:
            example_id = self.validation_dataset[self.config.results.output_cache_id][
                index
            ]
        else:
            example_id = str(index)

        if self.config.results.output_cached and example_id in processed_ids:
            return None

        text = self.validation_dataset[self.config.validation_data.input_feature][index]
        all_features = {
            key: self.validation_dataset[key][index]
            for key in self.validation_dataset.features.keys()
        }

        if self.data_preprocessor is not None:
            all_features = self.data_preprocessor.preprocess(all_features)

        pres = None
        if self.config.iterative is not None and self.config.iterative.enabled:
            pres = self.execute_iterative_loop(index, text, all_features)
        elif self.config.llm.prompts is not None:
            use_messages = self.config.llm.messages
            if use_messages:
                pres = self.execute_messages_prompts(index, text, all_features)
            else:
                pres = self.execute_sequential_prompts(index, text, all_features)
        else:
            pres = self.execute_single_prompt(index, text, all_features)

        return (pres, example_id)

    def process_batch_parallel(
        self, start_index, end_index, processed_ids, max_workers=20
    ):
        """
        Processes a batch of examples from the validation set. The batch is defined by the start_index and end_index.

        Args:
            start_index (int): The index of the first example in the batch
            end_index (int): The index of the last example in the batch
            processed_ids (set): A set containing the ids of the examples that have already been processed
            max_workers (int): The max number of threads to use for parallel processing
        """

        # We are using ThreadPoolExecutor to parallelize the processing of the examples in the batch
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self.process_validation_example, index, processed_ids
                ): index
                for index in range(start_index, end_index + 1)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    prediction = future.result()
                    results.append({"index": index, "prediction": prediction})
                except Exception as exc:
                    # TODO @trimitris: printing the index doesn't make much sense. Ideally we want to print the example itself
                    print(
                        f"Validation example with index: {index}, threw the exception: {exc}"
                    )

                    # Throw the exception again, because we want the user to know that something went wrong.
                    # Otherwise, the failure is going to get lost in the sea of logs in stdout, and the user
                    # might never realise that some of the examples failed. They will just have gaps in the output
                    # file which can cause lower evaluation scores which they might not be able to explain.
                    raise exc

        # results are appended to the results list in the order the threads are completed, so they need to be sorted
        results = sorted(results, key=lambda x: x["index"])

        # write to output file
        for res in results:
            if res["prediction"] is not None:
                self.output_write.append(res["prediction"][0], res["prediction"][1])
            else:
                print(f"Skipping already processed index: {res['index']}")

    def execute(self):
        # First embed the training dataset and prepare for the processing
        processed_ids = self.output_write.get_processed_ids()

        if self.embedder is not None:
            self.embedder.set_training_dataset(self.train_dataset)
            self.embedder.embedd()

        # From this point the processing of each validation example begins
        start_time = time.time()

        batch_size = self.config.results.batch_size

        if batch_size <= 0:
            # this is useful if we expose the BATCH_SIZE in config
            raise ValueError("Batch size must be greater than 0")

        array_to_process = self.validation_dataset[
            self.config.validation_data.input_feature
        ]
        num_batches = len(array_to_process) // batch_size + (
            0 if (len(array_to_process) % batch_size == 0) else 1
        )

        for batch in tqdm.tqdm(range(num_batches)):
            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size - 1, len(array_to_process) - 1)

            print(
                f"Processing batch: {batch} with start_index: {start_index} and end_index: {end_index}"
            )
            self.process_batch_parallel(start_index, end_index, processed_ids)

        self.output_write.close()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time (validation set only): {elapsed_time} seconds")
