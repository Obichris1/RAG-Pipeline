from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


DATA_DIR =BASE_DIR / "data"
PDF_PATH = DATA_DIR / "knowledge.pdf"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL_NAME = "gemini-embedding-2"

TOP_K = 3
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY")


print("base directory" ,BASE_DIR)
print("data folder", DATA_DIR)
print("pdf file", PDF_PATH)

# existing constants...
MAX_DOCUMENT_CHARS = 50_000  # ~15-20 pages; keeps this test app responsive