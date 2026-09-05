# Import Python's built-in math module.
#
# We use it to calculate vector magnitudes
# when calculating cosine similarity.
import math


# Import ChatGroq, LangChain's integration for Groq's LLM API.
from langchain_groq import ChatGroq


# Import configuration values from our config file.
#
# CHUNK_SIZE
# → how large each text chunk should be.
#
# CHUNK_OVERLAP
# → how much neighboring chunks should overlap.
#
# TOP_K
# → how many chunks we retrieve for a question.
#
# EMBEDDING_MODEL_NAME
# → the embedding model being used.
#
# GROQ_API_KEY
# → API key used to access Groq.
#
# MAX_DOCUMENT_CHARS
# → maximum amount of extracted document text allowed.
from app.core.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    MAX_DOCUMENT_CHARS,
)


# Import the class responsible for extracting text from PDFs.
from app.rag.loader import PDFLoader

# Import our text chunking class.
from app.rag.chunkers import LangChainTextChunker

# Import our embedding model wrapper.
from app.rag.embeddings import EmbeddingModel

# Import our FAISS vector store wrapper.
from app.rag.vectorStore import VectorStore


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures how similar two vectors
    are based on their direction.

    A value close to:
        1  → very similar
        0  → unrelated
       -1  → opposite directions
    """

    # Calculate the dot product of the two vectors.
    #
    # zip(a, b) pairs corresponding values:
    #
    # a[0] with b[0]
    # a[1] with b[1]
    # ...
    #
    # Then we multiply each pair and add everything together.
    dot = sum(x * y for x, y in zip(a, b))

    # Calculate the magnitude/norm of vector a.
    #
    # Formula:
    #
    # sqrt(x1² + x2² + ... + xn²)
    norm_a = math.sqrt(sum(x * x for x in a))

    # Calculate the magnitude/norm of vector b.
    norm_b = math.sqrt(sum(y * y for y in b))

    # Prevent division by zero.
    #
    # If either vector has a magnitude of zero,
    # cosine similarity cannot be calculated normally.
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Cosine similarity formula:
    #
    #              A · B
    # cosine =  -----------
    #           ||A|| ||B||
    #
    return dot / (norm_a * norm_b)


class RagEngine:
    """
    Main class responsible for coordinating the entire RAG pipeline.

    It connects:

    PDF Loader
        ↓
    Chunker
        ↓
    Embedding Model
        ↓
    Vector Store
        ↓
    Retrieval
        ↓
    LLM
    """

    def __init__(self):

        # Initially, there is no vector store.
        #
        # It will be created when the user uploads/processes a PDF.
        self.vector_store = None

        # This will later hold the embedding model instance.
        self.embedding_model = None

        # This list will store the chunks created from the document.
        self.chunks = []

        # This list will store the embedding vector
        # for every chunk.
        self.chunk_embeddings = []

        # Create the Groq LLM.
        self.llm = ChatGroq(

            # Specify the model we want Groq to run.
            model="openai/gpt-oss-20b",

            # Temperature controls randomness.
            #
            # 0 means we want deterministic/consistent responses.
            temperature=0,

            # Pass the Groq API key.
            groq_api_key=GROQ_API_KEY
        )

    def index_document(self, pdf_path: str) -> dict:
        """
        Process and index a PDF.

        Pipeline:

        PDF
         ↓
        Extract text
         ↓
        Check document size
         ↓
        Split text into chunks
         ↓
        Generate embeddings
         ↓
        Create vector store
        """

        # Create a PDFLoader and extract all text from the PDF.
        text = PDFLoader(pdf_path).load()

        # Check whether the document exceeds our configured limit.
        if len(text) > MAX_DOCUMENT_CHARS:

            # Return an error response instead of continuing.
            return {
                "ok": False,

                # Tell the user how large the document is
                # and what the maximum allowed size is.
                "message": (
                    f"This document is {len(text):,} characters, which is over the "
                    f"{MAX_DOCUMENT_CHARS:,} character limit for this demo. "
                    "Please try a shorter document."
                ),
            }

        # Create our text chunker.
        #
        # CHUNK_SIZE controls the approximate maximum size
        # of each chunk.
        #
        # CHUNK_OVERLAP preserves some context between neighboring chunks.
        chunks = LangChainTextChunker(
            CHUNK_SIZE=CHUNK_SIZE,
            CHUNK_OVERLAP=CHUNK_OVERLAP
        ).chunk_text(text)

        # Create the embedding model.
        #
        # This model converts text into numerical vectors.
        embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)

        # Generate one embedding vector for every chunk.
        #
        # Example:
        #
        # Chunk 1 → [0.12, -0.43, ...]
        # Chunk 2 → [0.51,  0.18, ...]
        # Chunk 3 → [...]
        vectors = embedding_model.embed_documents(chunks)

        # Create our vector store using the embedding model.
        self.vector_store = VectorStore(embedding_model)

        # Add the chunks to the vector store.
        #
        # The vector store will use the embedding model
        # to create its searchable vector index.
        self.vector_store.create(chunks)

        # Keep a reference to the embedding model.
        #
        # We need this later when a user asks a question
        # so we can embed the question.
        self.embedding_model = embedding_model

        # Save the chunks so the UI can display them
        # and so we can map retrieved chunks back to their indexes.
        self.chunks = chunks

        # Save the chunk embeddings.
        #
        # These are used later to calculate cosine similarity
        # for our visual explanation.
        self.chunk_embeddings = vectors

        # Return information that the Gradio UI can use.
        return {
            "ok": True,

            # Message displayed in the status box.
            "message": (
                f"Document indexed successfully. "
                f"Created {len(chunks)} chunks."
            ),

            # Return all chunks to the UI.
            "chunks": chunks,

            # Return all embeddings to the UI.
            "embeddings": vectors,
        }

    def compare(self, question: str, k: int = None) -> dict:
        """
        Embed the user's question, retrieve the top-K chunks,
        and calculate similarity information for visualization.

        This function is particularly useful for your RAG visualizer
        because it exposes what happens inside retrieval.
        """

        # Make sure a document has been indexed first.
        if self.vector_store is None:
            raise ValueError("No document has been uploaded yet.")

        # If k wasn't explicitly provided,
        # use TOP_K from the config file.
        k = k or TOP_K

        # Convert the user's question into an embedding vector.
        #
        # Example:
        #
        # "What is the refund policy?"
        #
        # becomes something like:
        #
        # [0.12, -0.44, 0.08, ...]
        query_embedding = self.embedding_model.embed_query(question)

        # Search the vector store for the k most relevant chunks.
        #
        # search_with_score returns:
        #
        # [
        #     (Document, distance),
        #     (Document, distance),
        #     ...
        # ]
        results = self.vector_store.search_with_score(question, k=k)

        # Create a list that will contain detailed information
        # about every retrieved chunk.
        matches = []

        # Loop through the retrieved results.
        #
        # enumerate(..., start=1) gives us:
        #
        # rank 1
        # rank 2
        # rank 3
        # ...
        for rank, (doc, l2_distance) in enumerate(results, start=1):

            # Find which original chunk this retrieved document came from.
            #
            # `.index()` searches self.chunks for the exact chunk text.
            try:
                chunk_index = self.chunks.index(doc.page_content)

                # Once we know the chunk index,
                # retrieve its original embedding.
                chunk_embedding = self.chunk_embeddings[chunk_index]

            except ValueError:

                # If the chunk cannot be found,
                # we don't know its index.
                chunk_index = None

                # Therefore we also don't have its embedding.
                chunk_embedding = None

            # Calculate cosine similarity between:
            #
            # query embedding
            #       VS
            # retrieved chunk embedding
            #
            # This gives us an easier-to-understand similarity
            # measurement for the visualizer.
            similarity = (
                cosine_similarity(
                    query_embedding,
                    chunk_embedding
                )
                if chunk_embedding is not None
                else None
            )

            # Store all the useful information about this match.
            matches.append({

                # Position of this chunk in the retrieval results.
                "rank": rank,

                # Original index of the chunk in the document.
                "chunk_index": chunk_index,

                # Actual text of the retrieved chunk.
                "text": doc.page_content,

                # FAISS L2 distance returned by the vector store.
                #
                # Smaller generally means the vectors are closer.
                "l2_distance": l2_distance,

                # Our manually calculated cosine similarity.
                "cosine_similarity": similarity,

                # The actual embedding vector of the chunk.
                "chunk_embedding": chunk_embedding,
            })

        # Return everything needed by the UI.
        return {

            # The embedding of the user's question.
            "query_embedding": query_embedding,

            # Information about the retrieved chunks.
            "matches": matches,
        }

    def generate(self, question: str, documents):
        """
        Send the retrieved documents and user's question
        to the LLM and generate the final answer.
        """

        # Extract the actual text from every retrieved document.
        #
        # If we retrieved 3 chunks:
        #
        # chunk1 text
        # chunk2 text
        # chunk3 text
        #
        # they are joined together into one context string.
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # Build the prompt that will be sent to the LLM.
        prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that you could not find the answer in the document.

Context:
{context}

Question:
{question}

Answer:
"""

        # Send the prompt to Groq.
        #
        # invoke() makes the actual LLM request.
        response = self.llm.invoke(prompt)

        # Return only the text generated by the LLM.
        return response.content

    def ask(self, question: str) -> dict:
        """
        Complete RAG question-answering pipeline.

        Question
           ↓
        Compare/retrieve
           ↓
        Retrieved chunks
           ↓
        Context
           ↓
        LLM
           ↓
        Answer
        """

        # First perform retrieval and similarity comparison.
        comparison = self.compare(question)

        # Convert our comparison results back into simple
        # document-like objects.
        #
        # The generate() function expects objects containing
        # a `.page_content` property.
        documents = [
            type(
                "Doc",
                (),
                {
                    "page_content": m["text"]
                }
            )()

            # Create one document object for every retrieved match.
            for m in comparison["matches"]
        ]

        # Send the retrieved documents and question to the LLM.
        answer = self.generate(question, documents)

        # Return both the final answer and all the retrieval
        # information needed by the Gradio visualizer.
        return {
            "answer": answer,
            "comparison": comparison,
        }