class BaseFunCalling:
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
