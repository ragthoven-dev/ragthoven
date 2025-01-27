import abc

from ragthoven.models.base import Preprocessor
from ragthoven.utils import get_func


class BaseDataPreprocessor(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def preprocess(self, **kwargs):
        pass


class DataPreprocessor(BaseDataPreprocessor):
    def __init__(self, config: Preprocessor):
        self.config = config

        self.preprocessors = []
        for preprocessor in self.config.entries:
            self.preprocessors.append(get_func("ragthoven.tools", preprocessor))

    def preprocess(self, args):
        for preprocessor in self.preprocessors:
            args = preprocessor(args)

        return args
