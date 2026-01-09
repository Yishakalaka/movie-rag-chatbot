# Data

This directory documents the data used by the Movie RAG Chatbot.
Large datasets and generated artifacts are intentionally excluded from the repository.

## Expected Local Files
- imdb_data.csv
  Raw movie metadata used for ingestion

- embeddings.parquet
  Generated embeddings created from the movie dataset

## Dataset Source
The dataset is derived from a public IMDb-style movie metadata source.
Any dataset with similar fields such as title, description, and genre can be used.

## How to Generate Data
From the repository root:

python scripts/ingest_data.py
python scripts/create_embeddings.py

## Notes
- Do not commit raw datasets or embeddings to GitHub
- This folder exists only to document data requirements
