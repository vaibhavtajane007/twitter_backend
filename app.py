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
async def predict(data: dict):
    try:
        features = extract_features(data["trend_name"])   # your function
        prediction = model.predict([features])[0]
        
        return {"prediction": int(prediction)}  # ALWAYS JSON

    except Exception as e:
        print("Prediction error:", str(e))
        return {"error": str(e)}
