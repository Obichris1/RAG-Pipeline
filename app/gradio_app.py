# Import Gradio, which we use to build the web interface.
import gradio as gr

# Import Path so we can convert the uploaded file path
# into a Python Path object.
from pathlib import Path

# Import our RAG engine, which handles:
# PDF processing → chunking → embeddings → vector search → LLM answer.
from app.rag.engine import RagEngine


# Create one instance of the RAG engine.
# This object will hold the currently indexed document,
# its chunks, embeddings, vector store, and LLM.
rag_engine = RagEngine()


def find_overlap(prev_chunk: str, chunk: str) -> str:
    """
    Find the text that appears at the end of the previous chunk
    and at the beginning of the current chunk.

    This is only used for visualization so we can see
    whether chunk overlap is actually happening.
    """

    # We don't need to check for an overlap longer than
    # the shorter of the two chunks.
    max_check = min(len(prev_chunk), len(chunk))

    # Start from the largest possible overlap and work downward.
    # This allows us to find the longest matching section.
    for length in range(max_check, 0, -1):

        # Take the last `length` characters from the previous chunk
        # and compare them with the first `length` characters
        # of the current chunk.
        if prev_chunk[-length:] == chunk[:length]:

            # If they match, return that overlapping text.
            return chunk[:length]

    # If no overlap is found, return an empty string.
    return ""


def preview_vector(vector, n=6):
    """
    Show only the first few values of an embedding vector.

    Embeddings can contain hundreds of numbers, so displaying
    the entire vector would make the UI extremely large.
    """

    # Take the first `n` values from the vector.
    # Format each value to 4 decimal places.
    # Join them together with commas.
    return ", ".join(f"{v:.4f}" for v in vector[:n]) + " …"


def process_pdf(pdf_path):
    """
    Called when the user clicks "Process Document".

    It sends the uploaded PDF through the RAG indexing pipeline:
    
    PDF
      ↓
    Text extraction
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    Vector store
    """

    # If the user hasn't uploaded a PDF,
    # show a message and return empty tables.
    if pdf_path is None:
        return "Please upload a PDF.", [], []

    try:

        # Convert the uploaded file path into a Path object
        # and send it to the RAG engine.
        result = rag_engine.index_document(Path(str(pdf_path)))

        # If indexing failed, return the error message
        # and empty tables.
        if not result["ok"]:
            return result["message"], [], []

        # Get the chunks created during indexing.
        chunks = result["chunks"]

        # Get the embedding vectors created for each chunk.
        vectors = result["embeddings"]

        # This list will contain rows for the chunks table.
        chunk_rows = []

        # Loop through every chunk.
        for i, chunk in enumerate(chunks):

            # For the first chunk there is no previous chunk,
            # so we don't calculate overlap.
            #
            # For every other chunk, compare it with the
            # immediately preceding chunk.
            overlap = find_overlap(chunks[i - 1], chunk) if i > 0 else ""

            # Add one row to the Gradio table.
            #
            # i + 1 makes the displayed chunk number start at 1
            # instead of Python's normal 0-based indexing.
            chunk_rows.append([
                i + 1,
                chunk,
                len(chunk),
                overlap
            ])

        # Create rows for the embeddings table.
        embedding_rows = [

            # Display:
            # 1. Embedding number
            # 2. Number of dimensions
            # 3. First few embedding values
            [i + 1, len(vector), preview_vector(vector)]

            # Repeat this for every embedding vector.
            for i, vector in enumerate(vectors)
        ]

        # Return:
        # 1. Status message
        # 2. Chunk table data
        # 3. Embedding table data
        return result["message"], chunk_rows, embedding_rows

    except Exception as e:

        # Print the actual exception to the terminal.
        # This is useful when debugging.
        print("INDEX ERROR:", repr(e))

        # Show the error in the Gradio UI.
        return f"Error processing document: {str(e)}", [], []


def ask_question(question):
    """
    Called when the user asks a question.

    The question goes through:

    Question
       ↓
    Query embedding
       ↓
    Vector similarity search
       ↓
    Top-K chunks
       ↓
    Retrieved context
       ↓
    LLM
       ↓
    Answer
    """

    # Remove whitespace from the question.
    #
    # If nothing remains, the user didn't actually enter a question.
    if not question.strip():
        return "Please enter a question.", "", []

    try:

        # Send the question to the RAG engine.
        #
        # The engine performs the retrieval and LLM generation.
        result = rag_engine.ask(question)

        # Get the comparison information returned by the RAG engine.
        #
        # This contains:
        # - query embedding
        # - retrieved chunks
        # - cosine similarity
        # - L2 distance
        # - chunk embeddings
        comparison = result["comparison"]

        # Create a small preview of the query embedding.
        #
        # The full embedding could contain hundreds of values,
        # so we only display the first few.
        query_preview = preview_vector(comparison["query_embedding"])

        # This will hold the rows displayed in the retrieval table.
        match_rows = []

        # Loop through every retrieved chunk.
        for m in comparison["matches"]:

            # Add one row containing information about the match.
            match_rows.append([

                # Rank of the retrieved chunk.
                m["rank"],

                # Original chunk number.
                #
                # We add 1 because chunk indexes internally start at 0.
                #
                # If the chunk index couldn't be found,
                # display "?" instead.
                m["chunk_index"] + 1
                if m["chunk_index"] is not None
                else "?",

                # The actual chunk text.
                m["text"],

                # Cosine similarity between the question embedding
                # and the chunk embedding.
                #
                # Higher cosine similarity generally means
                # the vectors are more similar.
                f"{m['cosine_similarity']:.4f}"
                if m["cosine_similarity"] is not None
                else "—",

                # L2 distance between the vectors.
                #
                # Smaller distance means the vectors are closer.
                f"{m['l2_distance']:.4f}",

                # Show a preview of the retrieved chunk's embedding.
                preview_vector(m["chunk_embedding"])
                if m["chunk_embedding"] is not None
                else "—",
            ])

        # Return:
        # 1. Final LLM answer
        # 2. Query embedding preview
        # 3. Retrieval table
        return result["answer"], query_preview, match_rows

    except Exception as e:

        # Print the error to the terminal for debugging.
        print("ASK ERROR:", repr(e))

        # Show the error in the Gradio interface.
        return f"Error: {str(e)}", "", []


# Create the main Gradio application.
#
# Everything indented inside this block becomes part of the UI.
with gr.Blocks() as demo:

    # Display the main title.
    gr.Markdown("# 📚 Chat With Your PDF")

    # Display a warning explaining the current document-size limitation.
    gr.Markdown(
        "⚠️ This is a test build — please upload shorter documents "
        "(roughly under 15-20 pages). Very long PDFs will be rejected."
    )

    # -------------------------
    # PDF UPLOAD
    # -------------------------

    # Create a file upload component.
    #
    # file_types=[".pdf"]
    # means only PDF files should be accepted.
    #
    # type="filepath"
    # means Gradio gives our Python function the path
    # to the uploaded file.
    pdf_upload = gr.File(
        label="Upload your PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    # Create the button the user clicks to process the PDF.
    process_button = gr.Button("Process Document")

    # Create a textbox for displaying processing status.
    status = gr.Textbox(label="Status")

    # Create a table for displaying chunks.
    chunks_table = gr.Dataframe(
        headers=[
            "#",
            "Chunk Text",
            "Characters",
            "Overlap with Previous"
        ],
        label="Chunks",
        wrap=True,
    )

    # Create a table for displaying embedding information.
    embeddings_table = gr.Dataframe(
        headers=[
            "#",
            "Dimensions",
            "Preview (first 6 values)"
        ],
        label="Embeddings",
        wrap=True,
    )

    # Tell Gradio what should happen when
    # the "Process Document" button is clicked.
    #
    # fn=process_pdf
    # → call this function.
    #
    # inputs=pdf_upload
    # → give the uploaded PDF path to the function.
    #
    # outputs=[...]
    # → put the returned values into these UI components.
    process_button.click(
        fn=process_pdf,
        inputs=pdf_upload,
        outputs=[
            status,
            chunks_table,
            embeddings_table
        ]
    )

    # -------------------------
    # ASK QUESTION
    # -------------------------

    # Create a textbox where the user can type a question.
    question = gr.Textbox(
        label="Ask a question",
        placeholder="What would you like to know about the document?"
    )

    # Create the button used to submit the question.
    ask_button = gr.Button("Ask")

    # Create a Markdown component where the final LLM answer
    # will be displayed.
    answer = gr.Markdown(label="Answer")

    

    # Display a preview of the embedding generated
    # from the user's question.
    query_embedding_preview = gr.Textbox(
        label="Your question, as an embedding (preview)",
        interactive=False,
    )

    # Create a table explaining which chunks were retrieved.
    retrieval_table = gr.Dataframe(
        headers=[
            "Rank",
            "Chunk #",
            "Chunk Text",
            "Cosine Similarity ↑",
            "L2 Distance ↓",
            "Chunk Embedding Preview"
        ],

        # This explains what the table represents.
        label=(
            "Why these chunks were retrieved "
            "(ranked by similarity to your question)"
        ),

        # Allow long chunk text to wrap inside the cells.
        wrap=True,
    )

    # Tell Gradio what should happen when
    # the "Ask" button is clicked.
    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=[
            answer,
            query_embedding_preview,
            retrieval_table
        ]
    )


# This checks whether this Python file is being run directly.
#
# If the file is imported by another Python file,
# this block will not execute.
import os

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )