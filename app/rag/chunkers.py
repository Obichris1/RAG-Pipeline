
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainTextChunker:

    def __init__(self, CHUNK_SIZE, CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

    def chunk_text(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)

        return chunks

