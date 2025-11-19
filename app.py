from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import joblib
from bert_encoder import extract_hashtag
from feature_engineering import add_features
from pydantic import BaseModel

CALI_MODEL_PATH = "model/final_twitter_model_calibrated.pkl"
SCALER_PATH = "model/scaler.pkl"
THRESHOLD_PATH = "model/threshold.json"
COLUMNS_PATH = "model/model_columns.json"

print("🔹 Loading model...")
try:
    model = joblib.load(CALI_MODEL_PATH)
except Exception:
    # Fall back to the old model path for compatibility
    model = joblib.load("model/final_twitter_model.pkl")

print("🔹 Loading scaler...")
scaler = None
try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

# Load saved threshold and optional serve-time offset
threshold = 0.5
threshold_offset = 0.0
try:
    with open(THRESHOLD_PATH, "r") as _f:
        _j = json.load(_f)
        if _j.get("threshold") is not None:
            threshold = float(_j.get("threshold"))
        threshold_offset = float(_j.get("threshold_offset", 0.0))
except Exception:
    pass

print("🔹 Loading feature columns...")
with open(COLUMNS_PATH, "r") as f:
    MODEL_COLUMNS = json.load(f)

print("✅ Backend ready")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TweetInput(BaseModel):
    tweet: str


@app.post("/predict")
def predict_trend(data: TweetInput):

    hashtag, emb = extract_hashtag(data.tweet)

    if hashtag is None:
        raise HTTPException(status_code=400, detail="No valid hashtag found.")

    try:
        feats = add_features(hashtag, emb)
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    X = pd.DataFrame([feats], columns=MODEL_COLUMNS).fillna(0)

    # If scaler available, scale features in the same order as MODEL_COLUMNS
    X_values = X.values
    if scaler is not None:
        X_values = scaler.transform(X_values)

    try:
        proba = float(model.predict_proba(X_values)[0][1])
    except Exception:
        # If the model expects a 2D array of shaped dataframe columns, try fallback
        proba = float(model.predict_proba(X)[0][1])

    # Apply lightweight heuristics to reduce false-positives on generic greetings
    # and to boost well-known high-popularity tags when historical signals exist.
    adjustments = []
    try:
        # feats is a dict we built earlier
        if feats.get("is_generic", 0) == 1:
            old = proba
            proba = float(proba * 0.2)
            adjustments.append({"action": "downweight_generic", "from": old, "to": proba})

        gp = float(feats.get("global_popularity_pct", 0) or 0)
        gtot = float(feats.get("global_total_volume", 0) or 0)
        if gp > 0.8 or gtot > 1_000_000:
            old = proba
            proba = float(min(0.99, proba + 0.35))
            adjustments.append({"action": "boost_high_popularity", "from": old, "to": proba})
    except Exception:
        pass

    # apply a small offset to the saved threshold to make serving slightly more conservative
    effective_threshold = min(0.99, threshold + (threshold_offset or 0.0))
    will = int(proba > effective_threshold)

    return {
        "hashtag": hashtag,
        "probability": proba,
        "will_trend_tomorrow": will,
        "threshold_used": effective_threshold,
        "adjustments": adjustments,
        "features_used": feats
    }


@app.get("/")
def home():
    return {"message": "Twitter Trend Model API Running"}
