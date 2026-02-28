import json
import re

from ragthoven.tools import BaseFunCalling


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
