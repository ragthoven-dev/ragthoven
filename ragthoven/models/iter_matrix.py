from copy import deepcopy
from enum import Enum

from ragthoven.models.base import Config
from ragthoven.utils import to_dict
from ragthoven.utils.config_loader import process_config


class IterParam(Enum):
    EMBED_K = "embed.k"
    EMBED_MODEL = "embed.model"
    RERANK_K = "rerank.k"
    RERANK_MODEL = "rerank.model"
    LLM_MODE = "llm.model"
    LLM_TEMPERATURE = "llm.temperature"


class IterationMatrix:
    def __init__(self, config: Config):
        self.conf_dict = to_dict(config)
        self.current_config = deepcopy(config)

        self.to_check = [
            IterParam.EMBED_K,
            IterParam.EMBED_MODEL,
            IterParam.RERANK_K,
            IterParam.RERANK_MODEL,
            IterParam.LLM_MODE,
            IterParam.LLM_TEMPERATURE,
        ]

        self.params_counter = {}
        self.params_max_index = {}
        for item in self.to_check:
            keys = item.value.split(".")
            if self.conf_dict[keys[0]] is None:
                continue
            val = self.conf_dict[keys[0]][keys[1]]
            self.params_counter[item.value] = 0 if type(val) is list else None
            self.params_max_index[item.value] = (
                len(val) - 1 if type(val) is list else None
            )

    def inc(self):
        to_inc = None
        i = 0
        for i in range(len(self.to_check)):
            if self.to_check[i] is not None:
                to_inc = self.to_check[i]
                break

        if to_inc is None:
            return False

        carry = True
        while carry:
            if self.params_counter.get(to_inc.value, None) is None:
                i += 1
                if i >= len(self.to_check):
                    break
                to_inc = self.to_check[i]
                continue
            else:
                self.params_counter[to_inc.value], carry = (
                    (self.params_counter[to_inc.value] + 1, False)
                    if self.params_counter[to_inc.value]
                    < self.params_max_index[to_inc.value]
                    else (0, True)
                )
                i += 1
                to_inc = self.to_check[i]

        return not (carry and len(self.to_check) >= i)

    def build_config(self):
        cconfig_dict = to_dict(self.current_config)
        for k, v in self.params_counter.items():
            keys = k.split(".")
            values = self.conf_dict[keys[0]][keys[1]]
            cconfig_dict[keys[0]][keys[1]] = values if v is None else values[v]

        self.current_config = process_config(cconfig_dict)

    def get_config(self):
        return self.current_config
