from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ── App setup ────────────────────────────────────────────────────
app = FastAPI(
    title="Idea Graveyard API",
    description="Find similar failed startups for any idea",
    version="1.0.0"
)

# ── Load model and embeddings once at startup ────────────────────
print("Loading model and embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load('models/embeddings.npy')
df = pd.read_csv('models/failed_startups_index.csv')
print("✓ API ready")

# ── Request/Response models ──────────────────────────────────────
class IdeaRequest(BaseModel):
    idea: str

class StartupMatch(BaseModel):
    name: str
    category: str
    similarity_score: float
    failure_reason: str

class AnalysisResponse(BaseModel):
    idea: str
    similar_failed_startups: int
    success_probability: int
    common_failure_reasons: list[str]
    top_matches: list[StartupMatch]

# ── Core logic ───────────────────────────────────────────────────
def analyze_idea(idea: str, top_n: int = 10):
    idea_embedding = model.encode([idea])
    similarities = cosine_similarity(idea_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]

    similar = []
    for idx in top_indices:
        row = df.iloc[idx]
        similar.append({
            "name": row['name'],
            "category": row['category'],
            "similarity_score": round(float(similarities[idx]), 2),
            "failure_reason": row['failure_reason'] if str(row['failure_reason']) != 'nan' else "Unknown"
        })

    all_reasons = [s['failure_reason'] for s in similar if s['failure_reason'] != 'Unknown']
    split_reasons = []
    for r in all_reasons:
        parts = str(r).replace(';', ',').split(',')
        split_reasons.extend([p.strip() for p in parts if len(p.strip()) > 3])

    reason_counts = Counter(split_reasons)
    top_reasons = [reason for reason, count in reason_counts.most_common(3)]

    avg_similarity = np.mean([s['similarity_score'] for s in similar])
    known_failures = len([s for s in similar if s['failure_reason'] != 'Unknown'])
    base_score = 100 - int(avg_similarity * 100)
    penalty = known_failures * 3
    success_probability = max(10, min(90, base_score - penalty))

    return {
        "idea": idea,
        "similar_failed_startups": len(similar),
        "success_probability": success_probability,
        "common_failure_reasons": top_reasons if top_reasons else ["No specific reasons found"],
        "top_matches": similar[:5]
    }

# ── Routes ───────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Idea Graveyard API is running"}

@app.post("/analyze-idea", response_model=AnalysisResponse)
def analyze(request: IdeaRequest):
    if not request.idea or len(request.idea.strip()) < 5:
        raise HTTPException(status_code=400, detail="Idea is too short")
    result = analyze_idea(request.idea)
    return result