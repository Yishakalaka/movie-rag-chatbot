import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import torch

# === CONFIG ===
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data/imdb_data.csv")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./data/embeddings.parquet")
MODEL_DIR = os.getenv("MODEL_DIR", "/Workspace/Shared/models/all-MiniLM-L6-v2")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # change if you prefer mistral, etc.

# === MOVIE CHATBOT ===
class MovieChatbot:
    def __init__(self):
        print("🤖 Initializing MovieChatbot...")
        self.df, self.embeddings = self.load_data()
        self.model = self.load_model()
        print(f"✅ Chatbot ready! Loaded {len(self.df)} movies.\n")

    def load_model(self):
        """Load sentence-transformer model (cached if available)."""
        if os.path.exists(MODEL_DIR):
            print(f"✅ Using cached model from {MODEL_DIR}")
            return SentenceTransformer(MODEL_DIR)
        else:
            print(f"⬇️ Downloading model {MODEL_NAME}...")
            model = SentenceTransformer(MODEL_NAME)
            os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
            model.save(MODEL_DIR)
            return model

    def load_data(self):
        """Load precomputed embeddings and dataset."""
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(
                f"❌ Embeddings not found at {EMBEDDINGS_PATH}. Run `ingest_data.py` first."
            )

        print(f"📂 Loading cached embeddings from {EMBEDDINGS_PATH}")
        df = pd.read_parquet(EMBEDDINGS_PATH)

        # Detect whether embeddings are in separate numeric columns or a list column
        if "embeddings" in df.columns:
            embeddings = np.array(df["embeddings"].tolist())
        else:
            numeric_cols = [c for c in df.columns if str(c).isdigit()]
            embeddings = df[numeric_cols].to_numpy()

        # Normalize column names for easier access
        df.columns = df.columns.map(lambda c: str(c).lower().strip())

        # Ensure we have a title and overview-like field
        possible_title_cols = ["title", "movie_title", "name"]
        possible_overview_cols = ["overview", "plot", "description"]

        df["title"] = df[
            next((c for c in possible_title_cols if c in df.columns), df.columns[0])
        ]
        df["overview"] = df[
            next((c for c in possible_overview_cols if c in df.columns), df.columns[1])
        ]

        return df, embeddings

    def search_movies(self, query, top_k=5):
        """Return top_k similar movies for a given query."""
        # Encode query as float32
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Ensure embeddings are float32 for consistent dtype
        if self.embeddings.dtype != np.float32:
            self.embeddings = self.embeddings.astype(np.float32)
        
        embeddings_tensor = torch.tensor(self.embeddings, dtype=torch.float32)
        
        # Compute cosine similarity safely
        cos_scores = util.cos_sim(query_embedding, embeddings_tensor)[0].cpu().numpy()
        top_results = np.argsort(-cos_scores)[:top_k]

        results = []
        for idx in top_results:
            movie = self.df.iloc[int(idx)]
            results.append({
                "title": movie["title"],
                "overview": movie["overview"],
                "score": float(cos_scores[idx])
            })

        return results


    def generate_llm_response(self, query, results):
        """Use Ollama LLM (if available) to summarize recommendations."""
        import subprocess, json

        # Build context for LLM
        context = "\n".join([
            f"Title: {r['title']}\nOverview: {r['overview']}" for r in results
        ])
        prompt = f"""You are a movie expert assistant.
Based on the query "{query}", here are some candidate movies:

{context}

Provide a friendly, concise recommendation summary for the user."""

        try:
            response = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL, prompt],
                capture_output=True, text=True, timeout=60
            )
            if response.returncode == 0:
                print("\n🧠 LLM Response:\n")
                print(response.stdout.strip())
            else:
                print("\n⚠️ Error generating LLM response:", response.stderr.strip())
        except Exception as e:
            print(f"\n⚠️ Ollama not available: {e}")

    def run_query(self, query):
        """Run chatbot once (non-interactive)."""
        print(f"🎬 Query: {query}\n")
        results = self.search_movies(query)
        print("🍿 Top movie recommendations:")
        for i, r in enumerate(results, 1):
            print(f"{i}. 🎥 {r['title']}  (score: {r['score']:.3f})")
            print(f"   📝 {r['overview'][:250]}...\n")

        self.generate_llm_response(query, results)

    def chat(self):
        """Adaptive chat mode: works locally or in Databricks."""
        print("\n🎬 Welcome to the Movie Recommendation Chatbot!")
        print("Type a movie description or 'exit' to quit.\n")

        # Detect Databricks environment
        is_databricks = bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))

        if is_databricks:
            print("💻 Running in Databricks interactive mode. Type your query below (rerun cell to continue).")
            return  # We'll handle user input externally in notebook cells

        while True:
            try:
                query = input("You: ").strip()
            except EOFError:
                break

            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            results = self.search_movies(query)
            print("\n🍿 Top movie recommendations:")
            for r in results:
                print(f"🎥 {r['title']} (score: {r['score']:.3f})")
                print(f"   {r['overview'][:150]}...\n")


if __name__ == "__main__":
    bot = MovieChatbot()

    # Databricks mode → just print a message
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        print("🧠 Databricks environment detected.")
        print("👉 To chat, run this in a notebook cell:\n")
        print("from chatbot.chatbot import MovieChatbot\nbot = MovieChatbot()\nbot.search_movies('your query here')")
    else:
        bot.chat()
