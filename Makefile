format:
	isort . && \
	black ragthoven/ && \
	black main.py

test-fast:
	mkdir -p test_output && \
	PYTHONPATH=ragthoven && \
	pytest ragthoven/test -k "test_example" && \
	mv example_*_results.jsonl test_output/ && \
	mv example_*_results.csv test_output/

test-intensive:
	mkdir -p test_output && \
	PYTHONPATH=ragthoven && \
	pytest ragthoven/test -k "test_intensive" && \
	mv results.*.json test_output/ && \
	mv results.*.jsonl test_output/

clean:
	rm -rf test_output/* && \
	rm -rf .pytest_cache

test:
	make test-fast
	make test-intensive
