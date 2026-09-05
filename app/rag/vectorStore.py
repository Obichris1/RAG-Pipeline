from langchain_community.vectorstores import FAISS


class VectorStore:

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None

    def create(self, chunks: list[str]):
        self.db = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )

    def search(self, query: str, k: int = 3):
        if self.db is None:
            raise ValueError("Vector store has not been created yet.")
        return self.db.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int = 3):
        """Returns [(Document, l2_distance), ...] — lower distance = closer match."""
        if self.db is None:
            raise ValueError("Vector store has not been created yet.")
        return self.db.similarity_search_with_score(query, k=k)