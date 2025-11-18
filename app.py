from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import joblib
from bert_encoder import extract_hashtag
from feature_engineering import add_features
from pydantic import BaseModel

MODEL_PATH = "model/final_twitter_model.pkl"
COLUMNS_PATH = "model/model_columns.json"

print("🔹 Loading model...")
model = joblib.load(MODEL_PATH)

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

    feats = add_features(hashtag, emb)

    X = pd.DataFrame([feats], columns=MODEL_COLUMNS).fillna(0)

    proba = float(model.predict_proba(X)[0][1])
    will = int(proba > 0.25)

    return {
        "hashtag": hashtag,
        "probability": proba,
        "will_trend_tomorrow": will
    }

@app.get("/")
def home():
    return {"message": "Twitter Trend Model API Running"}
