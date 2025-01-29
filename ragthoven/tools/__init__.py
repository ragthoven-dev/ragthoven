class BaseFunCalling:
    def __init__(self) -> None:
        self._type = "function"
        self.nane = "empty"

    def __call__(self, args):
        raise NotImplemented(
            f"Call function not implemented for function calling class: {self.name}"
        )

    def get_json_schema(self):
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
