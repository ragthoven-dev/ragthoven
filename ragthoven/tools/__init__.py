from dataclasses import dataclass


class BaseFunCalling:
    # Dependencies that should be injected on instantiation (e.g. prompt_executor)
    requires: list[str] = []

    def __init__(self) -> None:
        self._type = "function"
        self.name = "empty"
        self.description = None
        self.parameters = None

    def __call__(self, args):
        raise NotImplemented(
            f"Call function not implemented for function calling class: {self.name}"
        )

    def get_json_schema(self):
        assert self.description is not None
        assert self.parameters is not None

        tool_schema = {
            "type": self._type,
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": True,
            },
        }

        return tool_schema


@dataclass
class ReturnResultWrapper:
    result: str


class ReturnResult(BaseFunCalling):
    """Signal the final answer for the current example."""

    requires: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.description = "Finish the task by returning the final answer text. If the response is a number, keep it simple and return only a number."
        self.parameters = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to return for this example. If it is a number, only a number is returned without any additional characters.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        }

    def __call__(self, args):
        return ReturnResultWrapper(result=args["answer"])