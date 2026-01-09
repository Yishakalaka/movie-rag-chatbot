# Movie RAG Chatbot

## Overview
A retrieval-augmented generation (RAG) chatbot that answers natural language questions about movies using semantic search and a large language model.

The system embeds movie metadata into vector representations, retrieves the most relevant context for a user query, and uses an LLM to generate grounded, context-aware responses.

## Tech Stack
- Python
- SentenceTransformers
- PyTorch
- Vector-based semantic search
- Ollama or OpenAI-compatible APIs
- Databricks (development environment)

## Project Structure
chatbot/        # Core chatbot logic
scripts/        # Data ingestion and embedding generation
data/           # Dataset instructions (no large files stored)

## How It Works
1. Movie metadata is ingested and cleaned
2. Text is converted into embeddings using a sentence transformer model
3. Embeddings are stored locally for similarity search
4. User queries are embedded and matched against stored vectors
5. Retrieved context is passed to an LLM to generate a response

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Set up environment variables:
   cp .env.example .env

3. Ingest data and build embeddings:
   python scripts/ingest_data.py
   python scripts/create_embeddings.py

4. Run the chatbot:
   python chatbot/chatbot.py

## Example Queries
- Recommend a suspenseful movie with a strong female lead
- What are some critically acclaimed sci-fi films from the 2000s?
- Find movies similar to Inception but less action-heavy

## Design Overview
This project follows a modular retrieval-augmented generation architecture, separating ingestion, embedding, retrieval, and response generation for clarity and extensibility.

## Notes
- Large datasets and generated embeddings are intentionally excluded from the repository
- All secrets are managed via environment variables
