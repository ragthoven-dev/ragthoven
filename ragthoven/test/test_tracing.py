import json
from datetime import datetime

from ragthoven import Ragthoven
from ragthoven.executors.output_writer import JSONLOutputWriter
from ragthoven.executors.prompt_executor import BasePromptExecutor
from ragthoven.executors.trace_callback import JsonlTraceCallback
from ragthoven.executors.trace_writer import JSONLTraceWriter
from ragthoven.models.base import Config, LLM, Results, ValidationData


class DummyPromptExecutor(BasePromptExecutor):
    def get_prompt_results(self, sprompt, uprompt, tools=None, model=None):
        return f"DUMMY: {uprompt[:20]}"


def test_trace_callback_writes_success_event(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    writer = JSONLTraceWriter(str(trace_path))
    callback = JsonlTraceCallback(writer)

    callback.log_success_event(
        kwargs={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"run_id": "r1", "example_id": "e1"},
        },
        response_obj={"id": "resp1", "choices": []},
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    writer.close()

    line = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert line["event"] == "llm_success"
    assert line["response"]["id"] == "resp1"
    assert line["metadata"]["run_id"] == "r1"


def test_ragthoven_emits_output_trace(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "expression,expected\n\"1 + 1\",2\n\"2 + 2\",4\n", encoding="utf-8"
    )

    output_base = tmp_path / "results"
    results = Results(
        output_filename=str(output_base),
        batch_size=2,
        trace_enabled=True,
        trace_output_filename=str(output_base),
    )
    config = Config(
        name="trace-test",
        training_data=None,
        validation_data=ValidationData(
            dataset=f"csv:{csv_path}",
            input_feature="expression",
            split_name="train",
        ),
        results=results,
        embed=None,
        rerank=None,
        llm=LLM(sprompt="System: {{text}}", uprompt="User: {{text}}"),
        preprocessor=None,
        iterative=None,
    )

    output_writer = JSONLOutputWriter(f"{output_base}.jsonl", config)
    r = Ragthoven(
        config,
        prompt_executor=DummyPromptExecutor(),
        output_write=output_writer,
    )
    r.execute()

    trace_path = tmp_path / "results.traces.jsonl"
    first_line = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_line["event"] == "ragthoven_output"
    assert first_line["output"].startswith("DUMMY:")
