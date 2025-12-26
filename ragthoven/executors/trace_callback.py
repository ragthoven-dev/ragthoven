import json
from datetime import datetime, timezone

from litellm.integrations.custom_logger import CustomLogger

from ragthoven.executors.trace_writer import JSONLTraceWriter


class JsonlTraceCallback(CustomLogger):
    def __init__(self, trace_writer: JSONLTraceWriter) -> None:
        super().__init__()
        self.trace_writer = trace_writer

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._log_event("llm_success", kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._log_event("llm_failure", kwargs, response_obj, start_time, end_time)

    def _log_event(self, event_type, kwargs, response_obj, start_time, end_time):
        metadata = self._extract_metadata(kwargs)
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "run_id": metadata.get("run_id"),
            "example_id": metadata.get("example_id"),
            "prompt_name": metadata.get("prompt_name"),
            "mode": metadata.get("mode"),
            "iteration": metadata.get("iteration"),
            "metadata": metadata,
            "request": self._serialize(self._redact_kwargs(kwargs)),
            "response": self._serialize(response_obj),
        }

        latency_ms = self._compute_latency_ms(start_time, end_time)
        if latency_ms is not None:
            event["latency_ms"] = latency_ms

        self.trace_writer.append(event)

    @staticmethod
    def _extract_metadata(kwargs):
        if not kwargs:
            return {}
        if "metadata" in kwargs and kwargs["metadata"] is not None:
            return kwargs["metadata"]
        litellm_params = kwargs.get("litellm_params") or {}
        return litellm_params.get("metadata") or {}

    @staticmethod
    def _compute_latency_ms(start_time, end_time):
        if start_time is None or end_time is None:
            return None
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            return (end_time - start_time).total_seconds() * 1000
        try:
            return (end_time - start_time) * 1000
        except Exception:
            return None

    @staticmethod
    def _redact_kwargs(kwargs):
        if not kwargs:
            return {}
        redacted = dict(kwargs)
        for key in ("api_key", "headers", "extra_headers", "organization", "project"):
            if key in redacted:
                redacted[key] = "***"
        return redacted

    @staticmethod
    def _serialize(obj):
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)
