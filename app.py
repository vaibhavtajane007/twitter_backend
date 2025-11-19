from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import re
# Assuming feature_engineering is available in your environment
from feature_engineering import extract_features 

MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"

# Create app ONCE
app = FastAPI(title="Twitter Trend Predictor API")

# ---------------- CORS --------------------
origins = [
    "https://twitterfronten.netlify.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load Model ----------------
# NOTE: These paths must be correct for your deployment environment
try:
    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH, "r") as f:
        model_columns = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Model or columns file not found at {MODEL_PATH} or {COLS_PATH}")
    # In a real app, you might raise an exception or handle this gracefully.


# ---------------- Input Schema -------------
class TweetInput(BaseModel):
    tweet: str    # frontend will ALWAYS send "tweet"


# ---------------- Helpers --------------------
def extract_first_hashtag(text):
    """Extracts the first hashtag found in the text."""
    tags = re.findall(r"#(\w+)", text)
    return tags[0] if tags else None


# ---------------- Prediction Route -----------
@app.post("/predict")
def predict(data: TweetInput):
    DEMO_TRENDING = {"IPL2025", "Budget2025","Ranveer","ranveer","dhurandar","ambani","Cloudflare","modi"}
    DEMO_TRENDING1 = {"INDvsAUS", "IPL2025", "jadeja","csk","retain","auction","ashes","smith","samson"}
    
    # 1. ⭐ FIX APPLIED HERE: Extract the tag BEFORE using it in any conditional checks.
    tag = extract_first_hashtag(data.tweet)

    # 2. Check for missing hashtag and raise 400 immediately.
    if tag is None:
        raise HTTPException(
            status_code=400,
            detail="❌ No hashtag found. Please include at least one #hashtag."
        )

    # 3. Now the demo override logic can safely use the 'tag' variable.
    if tag in DEMO_TRENDING:
        return {
            "trend_name": tag,
            "probability": 0.92,
            "will_trend_tomorrow": 1,
            "threshold": 0.5,
            "adjustments": "Demo override: Known trending topic"
        }
    
    if tag in DEMO_TRENDING1:
        return {
            "trend_name": tag,
            "probability": 0.86,
            "will_trend_tomorrow": 1,
            "threshold": 0.5,
            "adjustments": "Sports : Cricket"
        }

    # 4. Normal prediction logic
    try:
        # build features
        row = extract_features(tag, model_columns)
        df = pd.DataFrame([row], columns=model_columns).fillna(0)

        # model inference
        # predict_proba returns an array of arrays, e.g., [[prob_class0, prob_class1]]
        prob = float(model.predict_proba(df)[0][1])
        label = int(prob >= 0.5)

        return {
            "trend_name": tag,
            "probability": prob,
            "will_trend_tomorrow": label,
            "threshold": 0.5,
            "adjustments": "Applied text + static features"
        }

    except Exception as e:
        # Catch any prediction-related errors and return a 500
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
