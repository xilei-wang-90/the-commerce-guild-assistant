# Sandrock Knowledge Assistant

Welcome! I'm a RAG-powered assistant that answers questions about **My Time at Sandrock** and **My Time at Portia** using wiki knowledge.

## How to use

1. **Ask a question** — type anything about the game (items, characters, locations, missions, etc.).
2. **Choose a retrieval mode** — click the gear icon to switch between:
   - **summary** — retrieves full wiki pages for broad context.
   - **section-reverse-hyde** — retrieves per-section chunks for more precise answers.

## Requirements

- **Ollama** must be running locally with the `all-minilm` embedding model pulled.
- A valid **`GEMINI_API_KEY`** must be set in your environment (or `.env` file).
- The **ChromaDB** vector store (`sandrock_db/`) must be populated via the embedding scripts.
