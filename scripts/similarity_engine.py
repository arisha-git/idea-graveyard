import pandas as pd
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import pickle
import os

# ── 1. Load only failed startups from database ──────────────────
print("Loading failed startups from database...")
conn = sqlite3.connect('database/startups.db')
df = pd.read_sql("SELECT * FROM startups WHERE status='failed'", conn)
conn.close()

# Keep only rows that have a useful description
df = df[df['description'].str.strip().str.len() > 10].reset_index(drop=True)
print(f"✓ {len(df)} failed startups with descriptions")

# ── 2. Load the NLP model ────────────────────────────────────────
print("\nLoading NLP model (first time may take 1-2 mins)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded")

# ── 3. Generate embeddings for all failed startups ───────────────
print("\nGenerating embeddings for all startups...")
print("(This may take a few minutes...)")

descriptions = df['description'].tolist()
embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)

print(f"✓ Embeddings shape: {embeddings.shape}")

# ── 4. Save embeddings to disk so we don't recompute every time ──
os.makedirs('models', exist_ok=True)
np.save('models/embeddings.npy', embeddings)
df.to_csv('models/failed_startups_index.csv', index=False)
print("✓ Saved embeddings to models/embeddings.npy")
print("✓ Saved index to models/failed_startups_index.csv")

# ── 5. Test it with a sample idea ───────────────────────────────
print("\n--- TEST RUN ---")
from sklearn.metrics.pairwise import cosine_similarity

test_idea = "AI tutor for coding interviews"
print(f"Input idea: '{test_idea}'")

idea_embedding = model.encode([test_idea])
similarities = cosine_similarity(idea_embedding, embeddings)[0]

# Get top 5 most similar
top_indices = np.argsort(similarities)[::-1][:5]

print("\nTop 5 similar failed startups:")
for i, idx in enumerate(top_indices):
    row = df.iloc[idx]
    score = similarities[idx]
    print(f"\n#{i+1} — {row['name']} (similarity: {score:.2f})")
    print(f"  Category: {row['category']}")
    print(f"  Description: {row['description'][:100]}...")
    print(f"  Failure reason: {row['failure_reason']}")