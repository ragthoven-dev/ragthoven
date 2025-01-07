from datasets import load_dataset

VALID_PREFIXES = [("json", "json:"), ("csv", "csv:")]


def dataset_load(dataset_name: str, version: str, split: str):
    for prefix, fprefix in VALID_PREFIXES:
        if dataset_name.startswith(fprefix):
            data_file = dataset_name.removeprefix(fprefix)

            return load_dataset(
                prefix,
                data_files=data_file,
                split=split,
            )

    return load_dataset(dataset_name, version, split=split)
