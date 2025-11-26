from abc import ABC, abstractmethod
from typing import Optional, Any, List


class BaseEmb(ABC):
    def __init__(self, model_name: str, model_params: Optional[dict[str, Any]] = None, **kwargs: Any):
        self.model_name = model_name
        self.model_params = model_params or {}

    @abstractmethod
    def get_emb(self, input: str) -> List[float]:
        pass


from langchain.embeddings import HuggingFaceEmbeddings
class BGEEmbedding(BaseEmb):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.embed_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda', 'local_files_only': True},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_emb(self, text: str) -> List[float]:
        return self.embed_model.get_text_embedding(text)

# emb = BGEEmbedding(r"C:\Users\chao\Downloads\bge-m3")
emb = BGEEmbedding(r"D:\移动硬盘\下载\all-MiniLM-L6-v2")
print(emb.get_emb("建筑结构的安全性检查包括哪些方面？"))