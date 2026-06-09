import pandas as pd
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ── Load everything once at startup ─────────────────────────────
print("Loading model and embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load('models/embeddings.npy')
df = pd.read_csv('models/failed_startups_index.csv')
print("✓ Ready")

def analyze_idea(idea: str, top_n: int = 10):
    # 1. Embed the input idea
    idea_embedding = model.encode([idea])

    # 2. Find similar failed startups
    similarities = cosine_similarity(idea_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # 3. Build similar startups list
    similar = []
    for idx in top_indices:
        row = df.iloc[idx]
        similar.append({
            "name": row['name'],
            "category": row['category'],
            "similarity_score": round(float(similarities[idx]), 2),
            "failure_reason": row['failure_reason'] if str(row['failure_reason']) != 'nan' else "Unknown"
        })

    # 4. Extract most common failure reasons
    all_reasons = [s['failure_reason'] for s in similar if s['failure_reason'] != 'Unknown']
    
    # Split reasons that contain semicolons or commas into individual reasons
    split_reasons = []
    for r in all_reasons:
        parts = str(r).replace(';', ',').split(',')
        split_reasons.extend([p.strip() for p in parts if len(p.strip()) > 3])

    reason_counts = Counter(split_reasons)
    top_reasons = [reason for reason, count in reason_counts.most_common(3)]

    # 5. Calculate success probability
    # Based on: average similarity score of matches (higher similarity = more competition = lower chance)
    avg_similarity = np.mean([s['similarity_score'] for s in similar])
    known_failures = len([s for s in similar if s['failure_reason'] != 'Unknown'])
    
    # Higher similarity to failures = lower success probability
    base_score = 100 - int(avg_similarity * 100)
    # Penalize if many similar startups have known failure reasons
    penalty = known_failures * 3
    success_probability = max(10, min(90, base_score - penalty))

    # 6. Return final output
    return {
        "idea": idea,
        "similar_failed_startups": len(similar),
        "success_probability": success_probability,
        "common_failure_reasons": top_reasons if top_reasons else ["No specific reasons found"],
        "top_matches": similar[:5]
    }


# ── Test it ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_ideas = [
        "AI tutor for coding interviews",
        "food delivery app for busy professionals",
        "social network for book lovers"
    ]

    for idea in test_ideas:
        print(f"\n{'='*50}")
        result = analyze_idea(idea)
        print(json.dumps(result, indent=2))