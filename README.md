# AI Document Assistant using RAG

An AI-powered document assistant that allows users to upload PDF documents, search for exact information, and ask natural-language questions about the document.

The application combines semantic search, vector embeddings, FAISS, and a Large Language Model (LLM) using a Retrieval-Augmented Generation (RAG) architecture.

## Features

- Upload and process PDF documents
- Extract text while preserving page numbers
- Intelligent text chunking with overlap
- Generate semantic embeddings using Sentence Transformers
- Perform vector similarity search with FAISS
- Search for exact keywords and phrases
- Count keyword and year occurrences
- Identify pages containing specific information
- Ask natural-language questions about the document
- Generate answers grounded only in retrieved document context
- Display source page numbers
- Inspect the retrieved context used to generate answers
- Cache processed documents for improved performance

## Live Demo

Try the deployed application here:

[Open AI Document Assistant](https://ai-document-assistant-v4ydtjhbpqky25bdaq5wjy.streamlit.app/)

## How It Works

The application follows a hybrid Retrieval-Augmented Generation pipeline:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Exact Search or Semantic Retrieval
    ↓
Relevant Document Context
    ↓
LLM
    ↓
Grounded Answer + Source Pages
```

For exact keyword, phrase, year, and count requests, the application searches the original extracted document text.

For natural-language questions, the system converts the question into an embedding and uses FAISS to retrieve the most semantically relevant document chunks. These chunks are then provided to the LLM as context for generating a grounded answer.

## Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- OpenAI API

## Embedding Model

The project uses:

`all-MiniLM-L6-v2`

to convert document chunks and user questions into numerical embedding vectors for semantic similarity search.

## RAG Architecture

The system uses Retrieval-Augmented Generation (RAG) to reduce unsupported answers.

Instead of asking the language model to answer directly, the application first retrieves relevant information from the uploaded PDF and provides that information to the model as context.

The model is instructed to answer only from the retrieved document context and to indicate when the requested information cannot be found.

## Project Structure

```text
ai-document-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## OpenAI API Key

Exact keyword, phrase, year, and occurrence searches can be performed without an OpenAI API key.

AI-powered questions using the RAG pipeline require an OpenAI API key. Users can enter their own API key securely through the password-protected input field in the Streamlit application.

The API key is used only for the current session and is not stored by the application.

Do not include or commit API keys directly in the source code.


## Run the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local Streamlit address in your browser.

## Example Use Cases

Users can perform exact searches such as:

```text
climate
2024
New Inhabitant
```

They can also ask natural-language questions such as:

```text
What are the main climate risks facing cities?
```

The system retrieves relevant sections of the uploaded document and generates an answer based on that context.

## Limitations

- Scanned PDFs without extractable text require OCR, which is not currently implemented.
- Retrieval quality depends on document structure, chunking, and embedding quality.
- Exact PDF text extraction may differ from the search behavior of some PDF viewers.
- The application currently uses an in-memory FAISS index and is designed primarily for individual document analysis.

## Future Improvements

Potential improvements include:

- OCR support for scanned documents
- Support for multiple documents
- Improved document-aware and sentence-aware chunking
- Conversation history
- Persistent vector storage
- Advanced retrieval and reranking
- Automated RAG evaluation

## Author

**Saliha Hedayat**

Machine Learning & AI Portfolio Project
