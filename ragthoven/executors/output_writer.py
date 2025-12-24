import abc
import csv
import json
import os
from enum import Enum

from ragthoven.models.base import Config


class SupportedOutputFormats(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"


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
        self.output_field = self.config.results.output_field or "label"
        if self.config.results.output_cached == True and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as already_processed:
                for line in already_processed.readlines():
                    processed_id = json.loads(line)[
                        (
                            "id"
                            if self.config.results.output_cache_id is None
                            else self.config.results.output_cache_id
                        )
                    ]
                    self.processed_ids.add(processed_id)
            self.results_file = open(path, "a", encoding="utf-8")
        else:
            self.results_file = open(path, "w", encoding="utf-8")

    def get_processed_ids(self):
        return self.processed_ids

    def append(self, response: str, id: str):
        data = {
            self.output_field: response,
            (
                "id"
                if self.config.results.output_cache_id is None
                else self.config.results.output_cache_id
            ): id,
        }
        self.results_file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def close(self):
        self.results_file.close()


class DelimitedOutputWriter(BaseOutputWriter):
    def __init__(self, path: str, config: Config, delimiter: str) -> None:
        self.config = config
        self.processed_ids = set()
        self.output_field = self.config.results.output_field or "label"
        self.id_field = (
            "id"
            if self.config.results.output_cache_id is None
            else self.config.results.output_cache_id
        )
        self.delimiter = delimiter

        if self.config.results.output_cached == True and os.path.exists(path):
            with open(path, "r", newline="", encoding="utf-8") as already_processed:
                reader = csv.DictReader(already_processed, delimiter=self.delimiter)
                for line in reader:
                    processed_id = line.get(self.id_field)
                    if processed_id is not None:
                        self.processed_ids.add(processed_id)
            self.results_file = open(path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.results_file, delimiter=self.delimiter)
        else:
            self.results_file = open(path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.results_file, delimiter=self.delimiter)
            self.csv_writer.writerow([self.id_field, self.output_field])

    def get_processed_ids(self):
        return self.processed_ids

    def append(self, response: str, id: str):
        self.csv_writer.writerow([id, response])

    def close(self):
        self.results_file.close()


class CSVOutputWriter(DelimitedOutputWriter):
    def __init__(self, path, config: Config) -> None:
        super().__init__(path, config, delimiter=",")


class TSVOutputWriter(DelimitedOutputWriter):
    def __init__(self, path, config: Config) -> None:
        super().__init__(path, config, delimiter="\t")
