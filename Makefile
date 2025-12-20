format:
	isort . && \
	black ragthoven

test-fast:
	mkdir -p test_output && \
	PYTHONPATH=ragthoven && \
	poetry run pytest ragthoven/test -k "test_example" && \
	mv example_*_results.jsonl test_output/ && \
	mv example_*_results.csv test_output/

test-intensive:
	mkdir -p test_output && \
	PYTHONPATH=ragthoven && \
	poetry run pytest ragthoven/test -k "test_intensive" && \
	mv results.*.json test_output/ && \
	mv results.*.jsonl test_output/

test-preproc:
	mkdir -p test_output && \
	PYTHONPATH=ragthoven && \
	poetry run pytest ragthoven/test -k "test_preprocessor" && \
	mv results.jsonl test_output

test-iterative:
	mkdir -p test_output && \
	PYTHONPATH=ragthoven && \
	poetry run pytest ragthoven/test/test_iterative.py && \
	mv math_calc_results* test_output && \
	mv math_calc_self_verification_results.* test_output

test-output-writer:
	PYTHONPATH=ragthoven && \
	poetry run pytest ragthoven/test/test_output_writer.py

clean:
	rm -rf test_output/* && \
	rm -rf .pytest_cache

test:
	make test-fast
	make test-intensive
	make test-preproc
	make test-iterative
	make test-output-writer
