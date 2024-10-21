import json
import random
import string


# hacky way
def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def stringify_obj(obj):
    return json.dumps(obj, default=lambda o: o.__dict__)


def stringify_obj_beautiful(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, indent=4)


def chromadb_normalize_name(collection_name: str):
    return collection_name.lower().replace(" ", "-").replace("/", "-").replace(
        ":", "-"
    )[:32] + "".join(random.choices(string.ascii_letters, k=5))
