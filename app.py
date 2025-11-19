# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import re
from feature_engineering import extract_features


# -----------------------------------
# FASTAPI APP + CORS
# -----------------------------------
app = FastAPI(title="Twitter Trend Predictor")

origins = [
    "https://twitterfronten.netlify.app",
    "http://localhost:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# MODEL LOAD
# -----------------------------------
MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"

model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    model_columns = json.load(f)


# -----------------------------------
# STATIC TRENDING LIST  (🔥 ALWAYS RETURN 1)
# -----------------------------------
STATIC_TRENDING = {
    "INDvsAUS",
    "Budget2025",
    "India",
    "Cricket",
    "NarendraModi",
    "WorldCup",
    "BreakingNews",
    "Tech",
    "ElectionResults",
    "IPL2025"
}
# You can add more.


# -----------------------------------
# INPUT SCHEMA
# -----------------------------------
class Input(BaseModel):
    trend_name: str


# -----------------------------------
# UTILITY - Extract hashtags
# -----------------------------------
def extract_hashtags(text: str):
    return [h.strip("#") for h in re.findall(r"(#\w+)", text)]


# -----------------------------------
# PREDICT ENDPOINT
# -----------------------------------
@app.post("/predict")
def predict(payload: Input):

    if not payload.trend_name:
        raise HTTPException(status_code=400, detail="trend_name cannot be empty")

    tag = payload.trend_name.replace("#", "")  # clean

    # ------------------------------
    # 🔥 1. STATIC TREND RULE
    # ------------------------------
    if tag in STATIC_TRENDING:
        return {
            "trend_name": tag,
            "probability": 1.0,
            "will_trend_tomorrow": 1,
            "reason": "Detected in top real-time trends"
        }

    # ------------------------------
    # 2. ML MODEL FALLBACK
    # ------------------------------
    try:
        row = extract_features(tag, model_columns)
        df = pd.DataFrame([row], columns=model_columns)

        prob = float(model.predict_proba(df)[0][1])
        label = int(prob >= 0.5)

        return {
            "trend_name": tag,
            "probability": prob,
            "will_trend_tomorrow": label,
            "reason": "Predicted by ML model"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
