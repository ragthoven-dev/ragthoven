import abc

from ragthoven.models.base import Config


class BaseReranker(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_similar(self, query, passages, k):
        raise NotImplementedError(f"Reranker {self} did not implement rerank")


class FlashRanker(BaseReranker):
    def __init__(self, config: Config):
        from flashrank import Ranker

        self.ranker = Ranker(model_name=config.rerank.model)

    def get_similar(self, query, passages, k):
        from flashrank import RerankRequest

        rerankrequest = RerankRequest(query=query, passages=passages)
        return self.ranker.rerank(rerankrequest)[:k]
