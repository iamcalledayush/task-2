from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class ModelWrapper(ABC):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    @abstractmethod
    def load(self):
        """Load model and move to specified device"""
        pass

    @abstractmethod
    def encode_data(self, texts, normalize_embeddings=True):
        """Encode data with appropriate prompt template"""
        pass

    @abstractmethod
    def encode_query(self, query, normalize_embeddings=True):
        """Encode query with appropriate prompt template"""
        pass


class BaseWrapper(ModelWrapper):
    def load(self):
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True).to(self.device)
        return self

    def encode_data(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)

    def encode_query(self, query, normalize_embeddings=True):
        return self.model.encode(query, normalize_embeddings=normalize_embeddings)


class CDEWrapper(ModelWrapper):
    def load(self):
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True).to(self.device)
        return self

    def encode_data(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings, prompt_name="document")

    def encode_query(self, query, normalize_embeddings=True):
        return self.model.encode(query, normalize_embeddings=normalize_embeddings, prompt_name="query")


class StellaWrapper(ModelWrapper):
    def load(self):
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True).to(self.device)
        return self

    def encode_data(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)

    def encode_query(self, query, normalize_embeddings=True):
        return self.model.encode(query, normalize_embeddings=normalize_embeddings, prompt_name="s2s_query")


class NomicWrapper(ModelWrapper):
    def load(self):
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True).to(self.device)
        return self

    def encode_data(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)

    def encode_query(self, query, normalize_embeddings=True):
        return self.model.encode(query, normalize_embeddings=normalize_embeddings)


class ModelRegistry:
    def __init__(self):
        self._registry = {
            "intfloat/e5-large-v2": BaseWrapper,  # NEW best model (Default added)
            "all-MiniLM-L12-v2": BaseWrapper,     # 33 MP | 0.12 GB
            "infgrad/stella-base-en-v2": BaseWrapper,  # 55 MP | 0.20 GB
            "Alibaba-NLP/gte-base-en-v1.5": BaseWrapper,  # 137 MP | 0.51 GB
            "nomic-ai/nomic-embed-text-v1.5": NomicWrapper,  # 137 MP | 0.51 GB
            "jxm/cde-small-v2": CDEWrapper,       # 150 MP | 0.56 GB
            "Alibaba-NLP/gte-large-en-v1.5": BaseWrapper,  # 434 MP | 1.62 GB
            "jinaai/jina-embeddings-v3": BaseWrapper,  # 572 MP | 2.13 GB
            "BAAI/bge-large-en-v1.5": BaseWrapper   # 1340 MP | 4.99 GB
        }

    def available_models(self):
        return list(self._registry.keys())

    def load(self, model_name: str, device: str = "cpu"):
        if model_name not in self._registry:
            raise ValueError(f"Model {model_name} not found in registry")
        return self._registry[model_name](model_name, device).load()