# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from feature_engineering import extract_features
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Twitter Trend Predictor")

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

MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"

model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    model_columns = json.load(f)

class Input(BaseModel):
    trend_name: str

@app.post("/predict")
def predict(payload: Input):
    try:
        print("Received:", payload)   # DEBUG

        tag = payload.trend_name.replace("#", "")

        row = extract_features(tag, model_columns)
        df = pd.DataFrame([row], columns=model_columns)

        prob = float(model.predict_proba(df)[0][1])
        label = int(prob >= 0.5)

        return {
            "trend_name": tag,
            "probability": prob,
            "will_trend_tomorrow": label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
