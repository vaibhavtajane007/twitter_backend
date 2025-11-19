# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import re
import os
from typing import List, Dict, Any

# try/except for optional feature pipeline import
try:
    from feature_engineering import extract_features
except Exception:
    # fallback minimal extractor (keeps server usable even if your feature module missing)
    def extract_features(tag: str, model_columns: List[str]) -> Dict[str, Any]:
        now = pd.Timestamp.utcnow()
        row = {c: 0.0 for c in model_columns}
        if "trend_length" in row:
            row["trend_length"] = len(tag)
        if "has_numbers" in row:
            row["has_numbers"] = int(bool(re.search(r"\d", tag)))
        if "day_of_week" in row:
            row["day_of_week"] = float(now.dayofweek)
        if "is_weekend" in row:
            row["is_weekend"] = float(int(now.dayofweek >= 5))
        # fill embedding columns with zeros (if any)
        for c in model_columns:
            if c.startswith("emb"):
                row[c] = 0.0
        return row

app = FastAPI(title="Twitter Trend Predictor")

# CORS — restrict to your frontend or use "*" for testing
origins = [
    "https://twitterfronten.netlify.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",  # remove in production if you want tighter security
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model files
MODEL_PATH = os.environ.get("MODEL_PATH", "model/final_twitter_model.pkl")
COLS_PATH = os.environ.get("COLS_PATH", "model/model_columns.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLS_PATH):
    raise RuntimeError(f"Missing model files. Place final_twitter_model.pkl and model_columns.json into model/ or set MODEL_PATH/COLS_PATH env vars.")

# load model + columns
model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    model_columns = json.load(f)

# detect embedding prefix list for any extra handling (if needed)
emb_cols = [c for c in model_columns if c.startswith("emb")]

# Rule-based static trending keywords (expandable). These are high-confidence triggers.
# Make them lower case for case-insensitive checks.
ALWAYS_TREND_KEYWORDS = {
    "indvsaus", "indvspak", "india", "cricket", "ipl", "ipl2025",
    "budget2025", "budget", "jadesha", "samson", "csk", "rr", "worldcup",
    "electionresults", "breakingnews", "cloudflare", "trailer", "viralvideo",
    "lok sabha", "loksabha", "modi"
}

# Provide a human-readable reason mapping if you want
KEYWORD_REASON = "High-confidence keyword match (static boost from recent India trends)"

class Input(BaseModel):
    # Accept either 'trend_name' (your frontend) or 'tweet' (older code)
    trend_name: str | None = None
    tweet: str | None = None

def normalize_tag_from_payload(payload: Input):
    # Prefer trend_name, else extract first hashtag from tweet
    if payload.trend_name:
        return payload.trend_name.strip().lstrip("#")
    if payload.tweet:
        hashtags = re.findall(r"#([\w\d_]+)", payload.tweet)
        return hashtags[0] if hashtags else payload.tweet.strip()[:100]
    raise HTTPException(status_code=400, detail="No trend_name or tweet provided.")

@app.post("/predict")
def predict(payload: Input):
    try:
        tag = normalize_tag_from_payload(payload)
        tag_clean = tag.strip()
        tag_lower = tag_clean.lower()

        # RULE-BASED STATIC CHECK (exact/contains)
        # If any keyword in ALWAYS_TREND_KEYWORDS is contained in tag, return immediate high-probability
        if any(k in tag_lower.replace(" ", "") or k in tag_lower for k in ALWAYS_TREND_KEYWORDS):
            # return the stable shape the frontend expects
            return {
                "hashtag": tag_clean,
                "trend_name": tag_clean,
                "probability": 0.48,               # keep moderate numeric probability (frontend shows %)
                "probability_pct": 48.0,           # convenience field (optional)
                "will_trend_tomorrow": 1,
                "rule_based": True,
                "reason": KEYWORD_REASON,
                "threshold_used": 0.5,
                "adjustments": []
            }

        # ML MODEL PATH
        # build features using extract_features (should match model_columns)
        row = extract_features(tag_clean, model_columns)
        # ensure all expected columns exist in row (fill zeros if missing)
        for c in model_columns:
            if c not in row:
                row[c] = 0.0

        df = pd.DataFrame([row], columns=model_columns).fillna(0)
        probs = model.predict_proba(df)[:, 1]
        prob = float(probs[0])
        label = int(prob >= 0.5)

        return {
            "hashtag": tag_clean,
            "trend_name": tag_clean,
            "probability": prob,
            "probability_pct": round(prob * 100, 4),
            "will_trend_tomorrow": label,
            "rule_based": False,
            "reason": "ML model prediction",
            "threshold_used": 0.5,
            "adjustments": []
        }

    except HTTPException:
        raise
    except Exception as e:
        # return a consistent error payload for frontend debugging
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
