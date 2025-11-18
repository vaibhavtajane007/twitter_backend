# app.py
import os
import json
import re
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# For embeddings of NEW hashtags
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Twitter Trend Predictor")
origins = [
    "http://localhost:3000",      # your local frontend dev server
    "https://your-frontend.com",  # your deployed frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# PATHS
# --------------------------
MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"
BERT_EMBED_CSV = "bert_embeddings.csv"

# --------------------------
# LOAD MODEL + COLUMNS
# --------------------------
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"❌ Model file not found: {MODEL_PATH}")

MODEL = joblib.load(MODEL_PATH)

if not os.path.exists(COLS_PATH):
    raise SystemExit(f"❌ Columns file not found: {COLS_PATH}")

with open(COLS_PATH, "r") as f:
    MODEL_COLUMNS = json.load(f)

print("✅ Model loaded with", len(MODEL_COLUMNS), "columns")

# --------------------------
# LOAD HISTORICAL TABLE (bert_embeddings.csv)
# --------------------------
if os.path.exists(BERT_EMBED_CSV):
    hist_df = pd.read_csv(BERT_EMBED_CSV)
    if "trend_name" not in hist_df.columns:
        raise SystemExit("❌ 'trend_name' not found in bert_embeddings.csv")
    # Normalize for lookups
    hist_df["trend_name_norm"] = hist_df["trend_name"].astype(str).str.lower().str.strip()
    print("✅ Historical table loaded:", hist_df.shape)
else:
    print("⚠️ bert_embeddings.csv not found; will treat all hashtags as new.")
    hist_df = pd.DataFrame(columns=["trend_name_norm"])

# --------------------------
# LOAD EMBEDDING MODEL (for NEW hashtags)
# --------------------------
print("Loading sentence-transformer for new hashtag embeddings...")
EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()
print("✅ Embedding model ready, dim =", EMB_DIM)

# --------------------------
# INPUT SCHEMA
# --------------------------
class TweetInput(BaseModel):
    tweet: str

# --------------------------
# HASHTAG EXTRACTOR
# --------------------------
def extract_hashtag(tweet: str):
    # returns first hashtag without '#'
    match = re.findall(r"#(\w+)", tweet)
    return match[0] if match else None

# --------------------------
# HISTORICAL LOOKUP
# --------------------------
def get_historical_row(hashtag: str):
    """
    Try to find this hashtag in bert_embeddings.csv (case-insensitive),
    return last matching row or None.
    """
    if hist_df.empty:
        return None

    cand = hashtag.lower().strip()
    # Check both 'tag' and '#tag' variants
    candidates = {cand, f"#{cand}"}
    mask = hist_df["trend_name_norm"].isin(candidates)

    if not mask.any():
        return None

    # Take the last row for this hashtag
    return hist_df[mask].iloc[-1]

# --------------------------
# BUILD FEATURE VECTOR
# --------------------------
def build_features(hashtag: str):
    # start with all zeros
    row = {c: 0 for c in MODEL_COLUMNS}

    # basic runtime features if present
    dow = datetime.utcnow().weekday()
    if "day_of_week" in row:
        row["day_of_week"] = dow
    if "is_weekend" in row:
        row["is_weekend"] = int(dow >= 5)
    if "trend_length" in row:
        row["trend_length"] = len(hashtag)
    if "has_numbers" in row:
        row["has_numbers"] = int(any(ch.isdigit() for ch in hashtag))

    # 1) Fill from historical row if exists
    hist = get_historical_row(hashtag)
    if hist is not None:
        for col in MODEL_COLUMNS:
            if col in hist.index:
                row[col] = hist[col]

    # 2) Ensure embedding columns are set
    emb_cols = [c for c in MODEL_COLUMNS if c.startswith("emb")]
    if emb_cols:
        # If historical row didn't provide embeddings, compute new one
        need_new_emb = True
        if hist is not None:
            # check if at least one emb col is non-zero
            if any(abs(hist.get(c, 0)) > 1e-9 for c in emb_cols):
                need_new_emb = False

        if need_new_emb:
            emb_vec = EMB_MODEL.encode([hashtag])[0]  # shape (dim,)
            # pad or trim to fit emb_cols length (safety)
            vec = np.zeros(len(emb_cols), dtype=float)
            L = min(len(emb_vec), len(emb_cols))
            vec[:L] = emb_vec[:L]
        else:
            # take from hist row
            vec = np.array([hist.get(c, 0.0) for c in emb_cols], dtype=float)

        # put into row
        for c, v in zip(emb_cols, vec):
            row[c] = float(v)

    return row

# --------------------------
# PREDICT ENDPOINT
# --------------------------
@app.post("/predict")
def predict_trend(data: TweetInput):
    hashtag = extract_hashtag(data.tweet)
    if not hashtag:
        raise HTTPException(status_code=400, detail="No valid hashtag found in tweet.")

    feats = build_features(hashtag)
    # Ensure exact column order
    X = pd.DataFrame([feats], columns=MODEL_COLUMNS)

    proba = float(MODEL.predict_proba(X)[0][1])
    THRESHOLD = 2.5e-05   # 0.000025
    will = int(proba >= THRESHOLD)

    return {
        "hashtag": hashtag,
        "probability": proba,
        "will_trend_tomorrow": will,
    }

# --------------------------
# SIMPLE ROOT
# --------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Twitter Trend Predictor is running."}
