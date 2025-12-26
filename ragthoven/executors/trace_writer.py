import json
import threading


class JSONLTraceWriter:
    def __init__(self, path: str) -> None:
        self.results_file = open(path, "w", encoding="utf-8")
        self._lock = threading.Lock()

    def append(self, event: dict):
        with self._lock:
            self.results_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.results_file.flush()

    def close(self):
        with self._lock:
            self.results_file.close()
