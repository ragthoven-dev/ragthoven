import abc

from datasets import Dataset

from ragthoven.models.base import Config
from ragthoven.utils import chromadb_normalize_name


class BaseEmbedder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_similar(self):
        raise NotImplementedError(f"Embedder {self} did not implement get_similar!")

    @abc.abstractmethod
    def set_training_dataset(self, training_dataset: Dataset):
        raise NotImplementedError(
            f"Embedder {self} did not implement set_training_dataset!"
        )

    @abc.abstractmethod
    def embedd(self):
        raise NotImplementedError(f"Embedder {self} did not implement embedd!")


class ChromaEmbedder(BaseEmbedder):
    def __init__(self, config: Config):
        import chromadb
        from chromadb.utils import embedding_functions

        self.input_feature = config.training_data.input_feature
        self.embedding_limit = config.embed.training_size_limit
        self.chroma_client = chromadb.Client()
        self.em_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embed.model, device=config.embed.device
        )

        self.collection = self.chroma_client.create_collection(
            name=chromadb_normalize_name(config.name), embedding_function=self.em_fn
        )

    def set_training_dataset(self, training_dataset: Dataset):
        self.train_dataset = training_dataset
        self.max_range = (
            self.embedding_limit
            if self.embedding_limit is not None and int(self.embedding_limit) > 0
            else len(self.train_dataset[self.input_feature])
        )

    def embedd(self):
        self.collection.add(
            documents=self.train_dataset[self.input_feature][: self.max_range],
            ids=list(map(lambda x: str(x), range(0, self.max_range))),
        )

    def get_similar(self, query: str, k: int):
        return self.collection.query(query_texts=query, n_results=k)
