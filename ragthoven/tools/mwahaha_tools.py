import json
import re

from ragthoven.executors.prompt_executor import BasePromptExecutor
from ragthoven.tools import BaseFunCalling


class _BaseMwahahaSubagent(BaseFunCalling):
    """Common helper for LLM-backed MWAHAHA subagent tools."""

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

    @staticmethod
    def _norm_arg(value, fallback="-") -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text if text else fallback

    @staticmethod
    def _input_block(headline: str, word1: str, word2: str) -> str:
        return f"- headline: {headline}\n- word1: {word1}\n- word2: {word2}"

    def _run_subagent(self, system_prompt: str, user_prompt: str) -> str:
        response = self.prompt_executor.get_messages_prompt_results(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=None,
            model=self.model_override,
            llm_overrides=self.llm_overrides,
        )

        if response is None:
            return "ERROR: BadRequest"

        default_bad_request = self.prompt_executor.config.results.bad_request_default_value
        if response == default_bad_request:
            return "ERROR: BadRequest"

        if not getattr(response, "choices", None):
            return "ERROR: No response"

        content = response.choices[0].message.content
        if content is None:
            return "ERROR: Empty response"

        return str(content).strip()


class PlannerSubagent(_BaseMwahahaSubagent):
    """Planner stage from exp08 as a callable subagent tool."""

    def __init__(
        self,
        prompt_executor: BasePromptExecutor,
        model_override: str | None = None,
        llm_overrides: dict | None = None,
    ) -> None:
        super().__init__(
            prompt_executor=prompt_executor,
            model_override=model_override,
            llm_overrides=llm_overrides,
        )
        self.name = type(self).__name__
        self.description = (
            "Planner subagent: create a structured joke plan using script "
            "opposition and anchors."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "word1": {"type": "string"},
                "word2": {"type": "string"},
                "inspiration": {
                    "type": "string",
                    "description": "Retrieved examples block for mechanism inspiration.",
                },
            },
            "required": ["headline", "word1", "word2", "inspiration"],
            "additionalProperties": False,
        }

    def __call__(self, args):
        headline = self._norm_arg(args.get("headline"))
        word1 = self._norm_arg(args.get("word1"))
        word2 = self._norm_arg(args.get("word2"))
        inspiration = self._norm_arg(args.get("inspiration"), fallback="")

        system_prompt = (
            "You are PlannerSubagent in an exp08-style humor pipeline. "
            "Create a concise but useful plan that maximizes surprise while "
            "staying safe and constraint-aware."
        )
        user_prompt = (
            f"Inputs:\n{self._input_block(headline, word1, word2)}\n\n"
            f"Inspiration examples (mechanisms only, do not copy wording/entities):\n"
            f"{inspiration}\n\n"
            "Output exactly these sections:\n"
            "PITFALLS:\n"
            "STRATEGY:\n"
            "TRICK_TYPE:\n"
            "CHECKPOINTS:\n"
            "SCRIPT_A:\n"
            "SCRIPT_B:\n"
            "VIOLATION:\n"
            "BENIGNING:\n"
            "PIVOT:\n"
            "SETUP_GIST:\n"
            "PUNCHLINE_GIST:\n"
            "ANCHOR_TOKENS:\n"
            "WORDPAIR_LINK:\n"
            "PREMISES:\n"
            "- ...\n"
            "MECHANISMS:\n"
            "- ...\n"
            "SAFETY_NOTE:\n"
        )
        return self._run_subagent(system_prompt, user_prompt)


class WriterSubagent(_BaseMwahahaSubagent):
    """Writer stage from exp08 as a callable subagent tool."""

    def __init__(
        self,
        prompt_executor: BasePromptExecutor,
        model_override: str | None = None,
        llm_overrides: dict | None = None,
    ) -> None:
        super().__init__(
            prompt_executor=prompt_executor,
            model_override=model_override,
            llm_overrides=llm_overrides,
        )
        self.name = type(self).__name__
        self.description = (
            "Writer subagent: generate diverse candidate jokes from planner output."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "word1": {"type": "string"},
                "word2": {"type": "string"},
                "planner_note": {"type": "string"},
                "inspiration": {"type": "string"},
            },
            "required": ["headline", "word1", "word2", "planner_note", "inspiration"],
            "additionalProperties": False,
        }

    def __call__(self, args):
        headline = self._norm_arg(args.get("headline"))
        word1 = self._norm_arg(args.get("word1"))
        word2 = self._norm_arg(args.get("word2"))
        planner_note = self._norm_arg(args.get("planner_note"), fallback="")
        inspiration = self._norm_arg(args.get("inspiration"), fallback="")

        system_prompt = (
            "You are WriterSubagent in an exp08-style humor pipeline. "
            "Write multiple candidates with strong twist quality and constraint discipline."
        )
        user_prompt = (
            f"Inputs:\n{self._input_block(headline, word1, word2)}\n\n"
            f"Planner output:\n{planner_note}\n\n"
            f"Inspiration examples (mechanisms only, no wording/entity reuse):\n"
            f"{inspiration}\n\n"
            "Requirements:\n"
            "- Generate 10 candidates.\n"
            "- 1-3 sentences each.\n"
            "- No semicolons.\n"
            "- Keep candidates diverse.\n"
            "- If headline is present, reference it with anchor tokens.\n"
            "- If word1/word2 are not '-', include both exactly.\n\n"
            "Output format:\n"
            "C1: ...\nCHECK1: ...\n"
            "C2: ...\nCHECK2: ...\n"
            "...\n"
            "C10: ...\nCHECK10: ...\n"
        )
        return self._run_subagent(system_prompt, user_prompt)


class ReflectorSubagent(_BaseMwahahaSubagent):
    """Reflector stage from exp08 as a callable subagent tool."""

    def __init__(
        self,
        prompt_executor: BasePromptExecutor,
        model_override: str | None = None,
        llm_overrides: dict | None = None,
    ) -> None:
        super().__init__(
            prompt_executor=prompt_executor,
            model_override=model_override,
            llm_overrides=llm_overrides,
        )
        self.name = type(self).__name__
        self.description = (
            "Reflector subagent: diagnose candidate weaknesses and provide rewrites."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "word1": {"type": "string"},
                "word2": {"type": "string"},
                "planner_note": {"type": "string"},
                "candidates": {"type": "string"},
            },
            "required": ["headline", "word1", "word2", "planner_note", "candidates"],
            "additionalProperties": False,
        }

    def __call__(self, args):
        headline = self._norm_arg(args.get("headline"))
        word1 = self._norm_arg(args.get("word1"))
        word2 = self._norm_arg(args.get("word2"))
        planner_note = self._norm_arg(args.get("planner_note"), fallback="")
        candidates = self._norm_arg(args.get("candidates"), fallback="")

        system_prompt = (
            "You are ReflectorSubagent in an exp08-style humor pipeline. "
            "Diagnose what fails and produce stronger rewrites."
        )
        user_prompt = (
            f"Inputs:\n{self._input_block(headline, word1, word2)}\n\n"
            f"Planner output:\n{planner_note}\n\n"
            f"Candidates:\n{candidates}\n\n"
            "Tasks:\n"
            "- Identify key failure modes (weak twist, overlap, missing anchors, etc.).\n"
            "- Produce 2 rewrites that fix the most likely issues.\n"
            "- Keep 1-3 sentences, no semicolons.\n\n"
            "Output format:\n"
            "DIAGNOSE:\n- ...\n"
            "R1: ...\n"
            "R2: ...\n"
        )
        return self._run_subagent(system_prompt, user_prompt)


class JudgeSubagent(_BaseMwahahaSubagent):
    """Judge stage from exp08 as a callable subagent tool."""

    def __init__(
        self,
        prompt_executor: BasePromptExecutor,
        model_override: str | None = None,
        llm_overrides: dict | None = None,
    ) -> None:
        super().__init__(
            prompt_executor=prompt_executor,
            model_override=model_override,
            llm_overrides=llm_overrides,
        )
        self.name = type(self).__name__
        self.description = (
            "Judge subagent: choose/polish the best candidate and return one final joke."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "word1": {"type": "string"},
                "word2": {"type": "string"},
                "candidates": {"type": "string"},
                "rewrites": {"type": "string"},
                "audit_feedback": {
                    "type": "string",
                    "description": "Optional ConstraintAudit failures to fix before finalizing.",
                },
            },
            "required": [
                "headline",
                "word1",
                "word2",
                "candidates",
                "rewrites",
                "audit_feedback",
            ],
            "additionalProperties": False,
        }

    def __call__(self, args):
        headline = self._norm_arg(args.get("headline"))
        word1 = self._norm_arg(args.get("word1"))
        word2 = self._norm_arg(args.get("word2"))
        candidates = self._norm_arg(args.get("candidates"), fallback="")
        rewrites = self._norm_arg(args.get("rewrites"), fallback="")
        audit_feedback = self._norm_arg(args.get("audit_feedback"), fallback="")

        system_prompt = (
            "You are JudgeSubagent in an exp08-style humor pipeline. "
            "Select or polish the best valid candidate and return only final joke text."
        )
        user_prompt = (
            f"Inputs:\n{self._input_block(headline, word1, word2)}\n\n"
            f"Candidates:\n{candidates}\n\n"
            f"Reflector rewrites:\n{rewrites}\n\n"
            f"ConstraintAudit feedback (if any):\n{audit_feedback}\n\n"
            "Decision rules:\n"
            "- Keep 1-3 sentences, no semicolons.\n"
            "- Preserve required anchors/word constraints.\n"
            "- Prefer strongest twist and clean landing.\n\n"
            "Output only the final joke text.\n"
        )
        return self._run_subagent(system_prompt, user_prompt)


class ConstraintAudit(BaseFunCalling):
    """Deterministic checker for MWAHAHA task-A style joke constraints."""

    requires: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.description = (
            "Check a candidate joke against deterministic constraints and return "
            "issues plus simple metrics."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "candidate": {
                    "type": "string",
                    "description": "Candidate joke text to audit.",
                },
                "headline": {
                    "type": "string",
                    "description": "Headline constraint or '-' when missing.",
                },
                "word1": {
                    "type": "string",
                    "description": "Required word1 token or '-'.",
                },
                "word2": {
                    "type": "string",
                    "description": "Required word2 token or '-'.",
                },
            },
            "required": ["candidate", "headline", "word1", "word2"],
            "additionalProperties": False,
        }

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    @staticmethod
    def _longest_common_token_span(a: str, b: str) -> int:
        a_tokens = ConstraintAudit._tokens(a)
        b_tokens = ConstraintAudit._tokens(b)
        if not a_tokens or not b_tokens:
            return 0

        longest = 0
        for i in range(len(a_tokens)):
            for j in range(len(b_tokens)):
                length = 0
                while (
                    i + length < len(a_tokens)
                    and j + length < len(b_tokens)
                    and a_tokens[i + length] == b_tokens[j + length]
                ):
                    length += 1
                if length > longest:
                    longest = length
        return longest

    @staticmethod
    def _same_clause(text: str, word1: str, word2: str) -> bool:
        clauses = re.split(r"[.!?;,:]+", text)
        for clause in clauses:
            if word1 in clause and word2 in clause:
                return True
        return False

    def __call__(self, args):
        candidate = str(args["candidate"]).strip()
        headline = str(args["headline"]).strip()
        word1 = str(args["word1"]).strip()
        word2 = str(args["word2"]).strip()

        issues = []
        metrics = {
            "candidate_len": len(candidate),
            "has_semicolon": ";" in candidate,
            "exact_headline_copy": False,
            "headline_anchor_hits": 0,
            "headline_longest_common_span": 0,
            "has_word1": None,
            "has_word2": None,
            "wordpair_same_clause": None,
        }

        if not candidate:
            issues.append("empty_candidate")

        if ";" in candidate:
            issues.append("has_semicolon")

        if headline != "-":
            headline_norm = self._norm(headline)
            cand_norm = self._norm(candidate)
            if cand_norm == headline_norm:
                metrics["exact_headline_copy"] = True
                issues.append("headline_exact_copy")

            headline_tokens = {
                tok for tok in self._tokens(headline) if len(tok) >= 4
            }
            candidate_tokens = set(self._tokens(candidate))
            anchor_hits = len(headline_tokens.intersection(candidate_tokens))
            metrics["headline_anchor_hits"] = anchor_hits
            if anchor_hits < 1:
                issues.append("missing_headline_anchor")

            longest_span = self._longest_common_token_span(candidate, headline)
            metrics["headline_longest_common_span"] = longest_span
            if longest_span >= 6:
                issues.append("headline_overlap_too_high")

        if word1 != "-":
            has_word1 = word1 in candidate
            metrics["has_word1"] = has_word1
            if not has_word1:
                issues.append("missing_word1")

        if word2 != "-":
            has_word2 = word2 in candidate
            metrics["has_word2"] = has_word2
            if not has_word2:
                issues.append("missing_word2")

        if word1 != "-" and word2 != "-":
            same_clause = self._same_clause(candidate, word1, word2)
            metrics["wordpair_same_clause"] = same_clause
            if not same_clause:
                issues.append("wordpair_not_same_clause")

        result = {
            "ok": len(issues) == 0,
            "issues": issues,
            "metrics": metrics,
            "suggestion": (
                "Fix issues, then call ConstraintAudit again. Use ReturnResult only "
                "after the candidate passes."
            ),
        }
        return json.dumps(result, ensure_ascii=True)
