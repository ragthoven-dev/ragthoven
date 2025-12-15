import ast
from dataclasses import dataclass
from typing import Any

from ragthoven.executors.prompt_executor import BasePromptExecutor
from ragthoven.tools import BaseFunCalling


class Calculator(BaseFunCalling):
    """Deterministic math evaluator."""

    requires: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.description = "Evaluate a mathematical expression using safe arithmetic. Use Python syntax."
        self.parameters = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2*(3+4)'. Use Python syntax.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    def _eval(self, node: ast.AST) -> Any:
        """Safely evaluate an expression AST."""
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    def __call__(self, args):
        expr = args["expression"]
        try:
            parsed = ast.parse(expr, mode="eval")
            result = self._eval(parsed)
            return f"Result: {result}"
        except Exception as exc:  # catch and hand control back to LLM
            return f"ERROR: {type(exc).__name__}: {exc}"


class SelfVerification(BaseFunCalling):
    """LLM sub-agent that checks a claim + reasoning pair."""

    requires: list[str] = ["prompt_executor"]

    def __init__(
        self,
        prompt_executor: BasePromptExecutor,
        model_override: str | None = None,
        llm_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self.prompt_executor = prompt_executor
        self.model_override = model_override
        self.llm_overrides = llm_overrides
        self.name = type(self).__name__
        self.description = (
            "Verify a claim for correctness. Replies with VERDICT: VALID or INVALID."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["claim", "reasoning"],
            "additionalProperties": False,
        }

    def __call__(self, args):
        system_prompt = (
            "You are a verification expert. Check the reasoning for errors. "
            "Respond succinctly and end with 'VERDICT: VALID' or 'VERDICT: INVALID'."
        )
        user_prompt = f"Claim: {args['claim']}\nReasoning: {args['reasoning']}"

        response = self.prompt_executor.get_messages_prompt_results(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=None,
            model=self.model_override,
            llm_overrides=self.llm_overrides,
        )

        if (
            response is None
            or not getattr(response, "choices", None)
            or len(response.choices) == 0
        ):
            return "ERROR: No response from verification model"

        content = response.choices[0].message.content
        return content if content else "ERROR: Empty response"


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
