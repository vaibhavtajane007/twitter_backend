from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
import re
import os
from textblob import TextBlob
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Twitter Trend Predictor")

MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"

# -----------------------------
# Load model + columns
# -----------------------------
model = joblib.load(MODEL_PATH)

with open(COLS_PATH, "r") as f:
    model_columns = json.load(f)

# -----------------------------
# Basic text feature extractor
# -----------------------------
def compute_text_features(tag: str):
    t = tag.lower()

    return {
        "trend_length": len(tag),
        "has_numbers": int(any(ch.isdigit() for ch in tag)),
        "sentiment_score": TextBlob(tag).sentiment.polarity,
        "is_sports": int(any(w in t for w in ["vs","match","cup","league","ipl","wc","ind","aus","final"])),
        "is_political": int(any(w in t for w in ["modi","bjp","election","congress","minister"])),
        "is_event": int(any(w in t for w in ["day","festival","birthday","anniversary"]))
    }

# -----------------------------
# Input model
# -----------------------------
class TweetInput(BaseModel):
    tweet: str

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_trend(data: TweetInput):
    hashtags = re.findall(r"#([\w\d_]+)", data.tweet)
    if not hashtags:
        raise HTTPException(400, "No hashtags found")

    tag = hashtags[0]
    feats = compute_text_features(tag)

    # Build input row
    row = []
    for col in model_columns:
        val = feats.get(col, 0.0)
        row.append(val)

    X = pd.DataFrame([row], columns=model_columns)

    proba = float(model.predict_proba(X)[0][1])
    label = int(proba >= 0.5)

    return {
        "hashtag": tag,
        "probability": proba,
        "will_trend_tomorrow": label
    }
