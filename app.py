from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import re
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
model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    model_columns = json.load(f)


# ---------------- Input Schema -------------
class TweetInput(BaseModel):
    tweet: str   # frontend will ALWAYS send "tweet"


# ---------------- Helpers --------------------
def extract_first_hashtag(text):
    tags = re.findall(r"#(\w+)", text)
    return tags[0] if tags else None


# ---------------- Prediction Route -----------
@app.post("/predict")
def predict(data: TweetInput):

    tag = extract_first_hashtag(data.tweet)

    if tag is None:
        raise HTTPException(
            status_code=400,
            detail="❌ No hashtag found. Please include at least one #hashtag."
        )

    try:
        # build features
        row = extract_features(tag, model_columns)
        df = pd.DataFrame([row], columns=model_columns).fillna(0)

        # model inference
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
