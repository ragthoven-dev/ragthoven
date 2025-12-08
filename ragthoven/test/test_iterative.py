from pathlib import Path

from ragthoven.test.test_intensive import run_with_cfg


def test_iterative_math_plus_self_verification():
    run_with_cfg("ragthoven/test/test_config/example_math_self_verification.yaml", max_validation=None) 

def test_iterative_math_run_produces_jsonl():
    run_with_cfg("ragthoven/test/test_config/example_math_calculator.yaml", max_validation=None)
