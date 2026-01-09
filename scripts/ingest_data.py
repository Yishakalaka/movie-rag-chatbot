import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data/movies_metadata.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "./data/embeddings.parquet")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./data/embeddings.parquet")

# === FUNCTIONS ===

def load_or_download_model():
    """Load the model from cache or download it if missing."""
    if os.path.exists(MODEL_DIR):
        print(f"✅ Loading cached model from {MODEL_DIR}")
        return SentenceTransformer(MODEL_DIR)
    else:
        print(f"⬇️ Downloading model {MODEL_NAME} and caching to {MODEL_DIR}")
        model = SentenceTransformer(MODEL_NAME)
        os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
        model.save(MODEL_DIR)
        return model


def generate_embeddings(model, texts, batch_size=64):
    """Generate embeddings in batches."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def main():
    print("🚀 Starting embedding pipeline...")

    # Step 1: Check if cached embeddings exist
    if os.path.exists(EMBEDDINGS_PATH):
        print(f"✅ Found cached embeddings at {EMBEDDINGS_PATH}")
        df = pd.read_parquet(EMBEDDINGS_PATH)
        print(f"Loaded {len(df)} rows with cached embeddings.")
        return df

    # Step 2: Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"📄 Loaded {len(df)} rows from {DATA_PATH}")

    # Step 3: Prepare text data
    if "overview" not in df.columns:
        raise KeyError("❌ Missing 'overview' column in dataset.")
    texts = df["overview"].fillna("").astype(str).tolist()

    # Step 4: Load or download model
    model = load_or_download_model()

    # Step 5: Generate embeddings
    print("⚙️ Generating new embeddings...")
    embeddings = generate_embeddings(model, texts, batch_size=64)

    # Step 6: Save embeddings
    df["embeddings"] = embeddings.tolist()
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    df.to_parquet(EMBEDDINGS_PATH, index=False)
    print(f"✅ Saved embeddings to {EMBEDDINGS_PATH}")

    return df


if __name__ == "__main__":
    main()
