import re
import hashlib

import streamlit as st
import pymupdf
import faiss

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================================================
# Configuration
# =========================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-5.6-luna"


# =========================================================
# Model Initialization
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# =========================================================
# 1. PDF Text Extraction
# =========================================================

def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from the PDF while preserving page numbers.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document, start=1):

        pages.append({
            "page": page_number,
            "text": page.get_text()
        })

    document.close()

    return pages


# =========================================================
# 2. Text Chunking
# =========================================================

def split_text(
    pages,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):
    """
    Split document pages into chunks while preserving
    page numbers and avoiding cuts in the middle of words
    whenever possible.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"].strip()

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            # Try to end the chunk at a natural boundary
            if end < len(text):

                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind(" ", start, end)
                )

                # Avoid creating very small chunks
                if boundary > start + chunk_size // 2:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start = max(
                end - chunk_overlap,
                start + 1
            )

    return chunks


# =========================================================
# 3. Embedding Generation
# =========================================================

def generate_embeddings(chunks):
    """
    Convert chunk text into embedding vectors.
    """

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=False
    )

    return embeddings


# =========================================================
# 4. FAISS Vector Index
# =========================================================

def build_faiss_index(embeddings):
    """
    Build a FAISS index for semantic search.
    """

    normalized_embeddings = embeddings.copy()

    faiss.normalize_L2(
        normalized_embeddings
    )

    dimension = normalized_embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        normalized_embeddings
    )

    return index


# =========================================================
# 5. Exact Keyword Search
# =========================================================

def search_keyword(keyword, chunks):
    """
    Find chunks containing an exact keyword or phrase.
    """

    keyword = keyword.strip().lower()

    results = []

    for chunk in chunks:

        text = chunk["text"]

        if keyword in text.lower():

            results.append({
                "page": chunk["page"],
                "text": text
            })

    return results


# =========================================================
# 6. Exact Keyword Count
# =========================================================

def count_keyword_occurrences(keyword, pages):
    """
    Count exact keyword occurrences in original pages.

    Using original pages prevents duplicate counting
    caused by overlapping chunks.
    """

    keyword = keyword.strip()

    total_count = 0
    found_pages = []

    for page_data in pages:

        text = page_data["text"]

        matches = re.findall(
            re.escape(keyword),
            text,
            flags=re.IGNORECASE
        )

        count = len(matches)

        if count > 0:

            total_count += count

            found_pages.append(
                page_data["page"]
            )

    return total_count, found_pages


# =========================================================
# 7. Semantic Retrieval
# =========================================================

def retrieve_chunks(
    question,
    chunks,
    faiss_index,
    k=TOP_K
):
    """
    Retrieve semantically relevant document chunks.
    """

    query_vector = embedding_model.encode(
        question
    ).reshape(1, -1)

    faiss.normalize_L2(
        query_vector
    )

    # Do not request more results than available chunks
    k = min(
        k,
        len(chunks)
    )

    scores, indices = faiss_index.search(
        query_vector,
        k
    )

    retrieved_chunks = [
        chunks[index]
        for index in indices[0]
        if index >= 0
    ]

    return retrieved_chunks, scores[0]


# =========================================================
# 8. Answer Generation
# =========================================================

def generate_answer(
    question,
    retrieved_chunks,
    client
):
    """
    Generate a grounded answer using only retrieved context.
    """

    context_parts = []

    for chunk in retrieved_chunks:

        page = chunk["page"]
        text = chunk["text"]

        context_parts.append(
            f"Page {page}:\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
Answer the question using only the document context provided below.

Do not use outside knowledge.

If the answer cannot be found in the context, say:
"I cannot find this information in the provided document."

When relevant, mention the page number where the information was found.

Document Context:
{context}

Question:
{question}

Answer:
"""

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )

    return response.output_text


# =========================================================
# 9. Complete Hybrid RAG Pipeline
# =========================================================

def ask_document(
    question,
    pages,
    chunks,
    faiss_index,
    client,
    k=TOP_K
):
    """
    Handle:
    - exact keyword searches
    - exact phrase searches
    - year searches
    - count requests
    - page requests
    - semantic RAG questions
    """

    question_clean = question.strip()
    question_lower = question_clean.lower()


    # -----------------------------------------------------
    # A. Short Keyword / Phrase Search
    # -----------------------------------------------------

    if len(question_clean.split()) <= 3:

        total_count, found_pages = (
            count_keyword_occurrences(
                question_clean,
                pages
            )
        )

        if total_count > 0:

            answer = (
                f'"{question_clean}" appears '
                f'{total_count} times in the document.\n\n'
                f'Pages: '
                f'{", ".join(map(str, found_pages))}'
            )

            return answer, [], None


    # -----------------------------------------------------
    # B. Detect Year Inside a Longer Question
    # -----------------------------------------------------

    years = re.findall(
        r"\b\d{4}\b",
        question_lower
    )

    if years:

        year = years[0]

        total_count, found_pages = (
            count_keyword_occurrences(
                year,
                pages
            )
        )


        # -------------------------------------------------
        # Count Request
        # -------------------------------------------------

        count_phrases = [
            "how many",
            "how often",
            "number of times",
            "count",
            "times mentioned",
            "how many times"
        ]

        if any(
            phrase in question_lower
            for phrase in count_phrases
        ):

            if total_count > 0:

                answer = (
                    f'"{year}" appears '
                    f'{total_count} times in the document.\n\n'
                    f'Pages: '
                    f'{", ".join(map(str, found_pages))}'
                )

            else:

                answer = (
                    f'"{year}" was not found '
                    f'in the document.'
                )

            return answer, [], None


        # -------------------------------------------------
        # Page Request
        # -------------------------------------------------

        page_phrases = [
            "which page",
            "which pages",
            "what page",
            "what pages",
            "page number",
            "pages mention",
            "where is"
        ]

        if any(
            phrase in question_lower
            for phrase in page_phrases
        ):

            if found_pages:

                answer = (
                    f'"{year}" appears on these pages:\n\n'
                    f'{", ".join(map(str, found_pages))}'
                )

            else:

                answer = (
                    f'"{year}" was not found '
                    f'in the document.'
                )

            return answer, [], None


        # -------------------------------------------------
        # Year-Related Semantic Question
        # -------------------------------------------------

        keyword_results = search_keyword(
            year,
            chunks
        )

        if keyword_results:

            if client is None:

                answer = (
                    "Please enter your OpenAI API key "
                    "to ask AI questions."
                )

                return answer, [], None

            retrieved_chunks = (
                keyword_results[:k]
            )

            answer = generate_answer(
                question,
                retrieved_chunks,
                client
            )

            return (
                answer,
                retrieved_chunks,
                None
            )


    # -----------------------------------------------------
    # C. Normal Semantic RAG Search
    # -----------------------------------------------------

    if client is None:

        answer = (
            "Please enter your OpenAI API key "
            "to ask AI questions."
        )

        return answer, [], None


    retrieved_chunks, scores = retrieve_chunks(
        question,
        chunks,
        faiss_index,
        k
    )

    answer = generate_answer(
        question,
        retrieved_chunks,
        client
    )

    return (
        answer,
        retrieved_chunks,
        scores
    )


# =========================================================
# Streamlit User Interface
# =========================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="centered"
)


st.title(
    "📄 AI Document Assistant"
)

st.markdown(
    """
    Upload a PDF and interact with its content using:

    - Exact keyword and phrase search
    - Year and occurrence counting
    - Semantic search with FAISS
    - AI-powered document questions using RAG
    """
)


st.write(
    "Upload a PDF and either search for an exact keyword "
    "or ask a question about the document."
)


# =========================================================
# User OpenAI API Key
# =========================================================

user_api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="Enter your OpenAI API key"
)


if user_api_key:

    client = OpenAI(
        api_key=user_api_key
    )

else:

    client = None


st.caption(
    "An OpenAI API key is only required for AI-powered questions. "
    "Exact searches work without an API key. "
    "Your key is used only for the current session and is not stored."
)


# =========================================================
# PDF Upload
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# =========================================================
# Document Processing
# =========================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()


    # Create a unique fingerprint for the uploaded PDF
    file_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()


    # Process only when a new document is uploaded
    if (
        "file_hash" not in st.session_state
        or st.session_state.file_hash != file_hash
    ):

        with st.spinner(
            "Processing document..."
        ):

            # Extract pages
            pages = extract_text_from_pdf(
                pdf_bytes
            )


            # Check whether PDF has extractable text
            full_document_text = "".join(
                page["text"]
                for page in pages
            )


            if not full_document_text.strip():

                st.error(
                    "No extractable text was found in this PDF."
                )

                st.stop()


            # Create document chunks
            chunks = split_text(
                pages
            )


            if not chunks:

                st.error(
                    "No text chunks could be created."
                )

                st.stop()


            # Generate embeddings
            chunk_embeddings = (
                generate_embeddings(
                    chunks
                )
            )


            # Build FAISS index
            faiss_index = (
                build_faiss_index(
                    chunk_embeddings
                )
            )


            # Store processed document
            st.session_state.pages = pages

            st.session_state.chunks = chunks

            st.session_state.faiss_index = (
                faiss_index
            )

            st.session_state.file_hash = (
                file_hash
            )


    # =====================================================
    # Reuse Processed Document
    # =====================================================

    pages = st.session_state.pages

    chunks = st.session_state.chunks

    faiss_index = (
        st.session_state.faiss_index
    )


    st.success(
        f"Document ready: "
        f"{len(pages)} pages, "
        f"{len(chunks)} chunks."
    )


    # =====================================================
    # Search / Question Input
    # =====================================================

    question = st.text_input(
        "Enter a keyword or ask a question:",
        placeholder=(
            "Ask a question about the document:"
        )
    )


    # =====================================================
    # Ask Button
    # =====================================================

    if st.button("Search / Ask"):

        if question.strip():

            with st.spinner(
                "Searching document..."
            ):

                answer, retrieved_chunks, scores = (
                    ask_document(
                        question,
                        pages,
                        chunks,
                        faiss_index,
                        client
                    )
                )


            # =================================================
            # Display Answer
            # =================================================

            st.subheader(
                "Result"
            )

            st.markdown(
                answer
            )


            # =================================================
            # Display Source Pages for RAG Answers
            # =================================================

            if retrieved_chunks:

                source_pages = sorted(
                    set(
                        chunk["page"]
                        for chunk
                        in retrieved_chunks
                    )
                )

                st.caption(
                    "Source pages: "
                    + ", ".join(
                        map(
                            str,
                            source_pages
                        )
                    )
                )


                # =============================================
                # Display Retrieved Document Context
                # =============================================

                with st.expander(
                    "View retrieved context"
                ):

                    for i, chunk in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        st.markdown(
                            f"**Source {i} — "
                            f"Page {chunk['page']}**"
                        )

                        st.write(
                            chunk["text"]
                        )

                        st.divider()

        else:

            st.warning(
                "Please enter a keyword or question."
            )
