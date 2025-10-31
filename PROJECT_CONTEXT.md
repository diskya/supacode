# Project Context: CodexBot

## 1. Project Overview

CodexBot is a command-line tool that serves as a code-aware chatbot. It allows a user to ask natural language questions about a codebase. The tool works in two phases:
1.  **Indexing:** It scans a source code repository, breaks the code into logical chunks (functions, classes), and stores semantic vector embeddings for each chunk in a local knowledge base.
2.  **Chatting:** When a user asks a question, it finds the most relevant code chunks from the knowledge base, combines them into a context, and sends this context along with the user's question to an LLM to generate an answer.

## 2. Key Technologies

-   **Language:** Python
-   **CLI:** `argparse`
-   **LLM Integration:** `openai` library (configured for a custom endpoint)
-   **Embeddings:**
    -   `sentence-transformers` for high-quality semantic embeddings.
    -   A custom `HashingEmbedder` for a fully offline, dependency-light fallback.
-   **Storage:** `SQLite` for metadata and `NumPy` flat files for vector storage.

## 3. Project Structure

-   `codexbot/`: Main package directory.
    -   `__main__.py`: Makes the package executable with `python -m codexbot`.
    -   `cli.py`: Defines the command-line interface (`index`, `query`, `chat`).
    -   `indexer.py`: Handles scanning the repository and parsing code into chunks.
    -   `embedder.py`: Manages the creation of vector embeddings from code chunks.
    -   `store.py`: Handles storing and retrieving data from SQLite and NumPy files.
    -   `retrieval.py`: Implements the core semantic and symbol search logic.
    -   `context.py`: Assembles the final prompt to be sent to the LLM.
    -   `llm_client.py`: Manages communication with the LLM API.
-   `requirements.txt`: Project dependencies.

## 4. How to Run

1.  `pip install -r requirements.txt`
2.  `export DASHSCOPE_API_KEY='...'`
3.  `python -m codexbot index --repo . --embedder sentence-transformers`
4.  `python -m codexbot chat --q "Your question about the code"`
