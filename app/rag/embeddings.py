from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingModel(Embeddings):

    def __init__(self, model_name):
      self.model = GoogleGenerativeAIEmbeddings(
    model=model_name
)

    def embed_documents(self, chunks: list[str]):
        return self.model.embed_documents(chunks)

    def embed_query(self, query: str):
        return self.model.embed_query(query)