import abc
import csv
import json
import os
from enum import Enum

from ragthoven.models.base import Config


class SupportedOutputFormats(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"


class BaseOutputWriter(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def append(self, response: str, id: str):
        pass

    @abc.abstractmethod
    def get_processed_ids(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class JSONLOutputWriter(BaseOutputWriter):
    def __init__(self, path, config: Config) -> None:
        self.config = config
        self.processed_ids = set()
        if self.config.results.output_cached == True and os.path.exists(path):
            with open(path, "r") as already_processed:
                for line in already_processed.readlines():
                    processed_id = json.loads(line)[
                        (
                            "id"
                            if self.config.results.output_cache_id is None
                            else self.config.results.output_cache_id
                        )
                    ]
                    self.processed_ids.add(processed_id)
            self.results_file = open(path, "a")
        else:
            self.results_file = open(path, "w")

    def get_processed_ids(self):
        return self.processed_ids

    def append(self, response: str, id: str):
        data = {
            "label": response,
            (
                "id"
                if self.config.results.output_cache_id is None
                else self.config.results.output_cache_id
            ): id,
        }
        self.results_file.write(json.dumps(data) + "\n")

    def close(self):
        self.results_file.close()


class CSVOutputWriter(BaseOutputWriter):
    def __init__(self, path, config: Config) -> None:
        self.config = config
        self.processed_ids = set()

        if self.config.results.output_cached == True and os.path.exists(path):
            with open(path, "r", newline="") as already_processed:
                csv_reader = csv.DictReader(already_processed)
                for line in csv_reader:
                    processed_id = line.get(
                        [
                            (
                                "id"
                                if self.config.results.output_cache_id is None
                                else self.config.results.output_cache_id
                            )
                        ]
                    )
                    self.processed_ids.add(processed_id)
            self.results_file = open(path, "a", newline="")
            self.csv_writer = csv.writer(self.results_file)
        else:
            self.results_file = open(path, "w", newline="")
            self.csv_writer = csv.writer(self.results_file)
            self.csv_writer.writerow(
                [
                    (
                        "id"
                        if self.config.results.output_cache_id is None
                        else self.config.results.output_cache_id
                    ),
                    "label",
                ]
            )

    def get_processed_ids(self):
        return self.processed_ids

    def append(self, response: str, id: str):
        data = [id, response]
        self.csv_writer.writerow(data)

    def close(self):
        self.results_file.close()
