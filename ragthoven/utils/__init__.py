import importlib
import json
import random
import re
import string


# hacky way
def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def stringify_obj(obj):
    return json.dumps(obj, default=lambda o: o.__dict__)


def stringify_obj_beautiful(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, indent=4)


def chromadb_normalize_name(collection_name: str):
    normalized = collection_name.lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+$", "", normalized)
    while ".." in normalized:
        normalized = normalized.replace("..", ".")
    if not normalized:
        normalized = "collection"
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    max_base_len = 63 - len(suffix)
    normalized = normalized[:max_base_len]
    normalized = re.sub(r"[^a-z0-9]+$", "", normalized)
    normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
    if not normalized:
        normalized = "collection"
    return normalized + suffix


def get_class_func_name_only(path: str):
    whole_hierarchy = path.split(".")

    return whole_hierarchy[len(whole_hierarchy) - 1]


def get_func(module_name: str, func_name: str):
    whole_hierarchy = func_name.split(".")

    module_full_name = module_name
    if len(whole_hierarchy) > 1:
        for package in whole_hierarchy[0 : len(whole_hierarchy) - 1]:
            module_full_name += "." + package

    module = importlib.import_module(module_full_name)
    func = getattr(module, whole_hierarchy[len(whole_hierarchy) - 1])
    return func


def get_class(module_name: str, class_name: str):
    return get_func(module_name, class_name)
