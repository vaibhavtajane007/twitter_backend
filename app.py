# app.py
import os
import json
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# -------------------------
# Config paths
# -------------------------
MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"
HIST_PATH = "bert_embeddings.csv"   # your big training CSV with emb0..emb767

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI(title="Twitter Trend Predictor")

# TODO: in production, set this to your frontend URL instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g. ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TweetInput(BaseModel):
    tweet: str


# -------------------------
# Load model + columns
# -------------------------
print("✅ Loading model...")
MODEL = joblib.load(MODEL_PATH)

print("✅ Loading model columns...")
with open(COLS_PATH, "r") as f:
    MODEL_COLUMNS = json.load(f)
print(f"✅ Model loaded with {len(MODEL_COLUMNS)} columns")

# -------------------------
# Load historical table (= bert_embeddings.csv)
# -------------------------
print("🔹 Loading historical table (bert_embeddings.csv)...")
HIST_DF = pd.read_csv(HIST_PATH)
print("✅ Historical table loaded:", HIST_DF.shape)

# Ensure consistent types
if "trend_name" in HIST_DF.columns:
    HIST_DF["trend_name"] = HIST_DF["trend_name"].astype(str)

# -------------------------
# Load sentence-transformer for new hashtags
# -------------------------
print("Loading sentence-transformer for new hashtag embeddings...")
EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()
print(f"✅ Embedding model ready, dim = {EMB_DIM}")

# Figure out which embedding columns exist in HIST_DF
EMB_COLS = [c for c in HIST_DF.columns if c.startswith("emb")]
EMB_COLS = sorted(EMB_COLS, key=lambda x: int(x[3:]))  # emb0, emb1, ...
print(f"✅ Found {len(EMB_COLS)} embedding cols in history")


# -------------------------
# Helpers
# -------------------------
def extract_hashtag(text: str) -> str | None:
    """Extract first hashtag (without #)."""
    match = re.findall(r"#(\w+)", text)
    return match[0] if match else None


def get_embedding_for_text(text: str) -> np.ndarray:
    """Compute new sentence-transformer embedding for unseen hashtag."""
    emb = EMB_MODEL.encode([text], convert_to_numpy=True)[0]
    return emb


def find_closest_historical(hashtag: str) -> pd.Series | None:
    """
    Try to find matching or similar historical trend by name.

    Strategy:
    1) Exact "trend_name" match if present
    2) Case-insensitive match
    3) Fallback: None
    """
    if "trend_name" not in HIST_DF.columns:
        return None

    # exact match
    exact = HIST_DF[HIST_DF["trend_name"] == hashtag]
    if not exact.empty:
        return exact.iloc[-1]

    # lowercase match
    lower = HIST_DF["trend_name"].str.lower()
    mask = lower == hashtag.lower()
    if mask.any():
        return HIST_DF[mask].iloc[-1]

    return None


def build_feature_row(hashtag: str) -> tuple[pd.DataFrame, str]:
    """
    Build a single-row DataFrame matching MODEL_COLUMNS.
    Returns: (df_row, explanation_str)
    """
    row = {c: 0.0 for c in MODEL_COLUMNS}
    explanation_parts = []

    # ----- Step 1: try to use historical row -----
    hist_row = find_closest_historical(hashtag)
    if hist_row is not None:
        # Copy any overlapping columns
        for col in MODEL_COLUMNS:
            if col in hist_row and col not in EMB_COLS:
                val = hist_row[col]
                # avoid copying NaNs
                if isinstance(val, (int, float, np.number)) and pd.notna(val):
                    row[col] = float(val)
        explanation_parts.append("Using historical stats for a known/similar hashtag.")
    else:
        explanation_parts.append("No direct history found; using generic defaults.")

    # ----- Step 2: embedding features -----
    # Check if we already have embeddings in HIST_DF for this hashtag
    if hist_row is not None and all(c in hist_row.index for c in EMB_COLS):
        # Use stored embeddings
        for col in EMB_COLS:
            if col in MODEL_COLUMNS:
                row[col] = float(hist_row[col])
        explanation_parts.append("Used stored BERT embedding from training data.")
    else:
        # Compute new embedding
        emb = get_embedding_for_text("#" + hashtag)
        # emb is 384 dim; your model was trained on 768-dim from BERT-base.
        # To keep things consistent, we just place it into the first EMB_DIM columns.
        for i in range(min(len(EMB_COLS), len(emb))):
            col_name = EMB_COLS[i]
            if col_name in MODEL_COLUMNS:
                row[col_name] = float(emb[i])
        explanation_parts.append("Computed fresh embedding for this hashtag.")

    # ----- Step 3: simple runtime features -----
    # These depend on "now"
    now = datetime.utcnow()
    if "day_of_week" in row:
        row["day_of_week"] = float(now.weekday())
        explanation_parts.append(f"day_of_week={now.weekday()} (UTC).")
    if "is_weekend" in row:
        row["is_weekend"] = float(1 if now.weekday() >= 5 else 0)
    if "month" in row:
        row["month"] = float(now.month)

    # basic text-based features
    if "trend_length" in row:
        row["trend_length"] = float(len(hashtag))
    if "has_numbers" in row:
        row["has_numbers"] = float(any(ch.isdigit() for ch in hashtag))

    df = pd.DataFrame([row], columns=MODEL_COLUMNS)
    explanation = " ".join(explanation_parts)
    return df, explanation


# -------------------------
# Health-check route
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Twitter Trend Predictor backend running"}


# -------------------------
# OPTIONS handler for CORS preflight (extra safe)
# -------------------------
from fastapi import Response

@app.options("/predict")
def options_predict():
    # CORS middleware will still add the proper headers
    return Response(status_code=200)


# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict")
def predict_trend(data: TweetInput):
    hashtag = extract_hashtag(data.tweet)
    if not hashtag:
        raise HTTPException(400, "No valid hashtag found in tweet text.")

    features_df, explanation = build_feature_row(hashtag)

    try:
        proba = float(MODEL.predict_proba(features_df)[0][1])
    except Exception as e:
        raise HTTPException(500, f"Model prediction failed: {e}")

    will = int(proba >= 0.5)

    return {
        "hashtag": hashtag,
        "probability": proba,
        "will_trend_tomorrow": will,
        "reason": explanation,
    }
