# app.py  (FINAL VERSION)
import re
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_engineering import extract_features


# --------------------------------------------------------------------
# FASTAPI + CORS
# --------------------------------------------------------------------
app = FastAPI(title="Twitter Trend Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://twitterfronten.netlify.app",
        "http://localhost:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------------
MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"

model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    model_columns = json.load(f)


# --------------------------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------------------------
class Input(BaseModel):
    trend_name: str


# --------------------------------------------------------------------
# RULE-BASED TRENDING (UPDATED WITH LATEST INDIA TRENDS)
# --------------------------------------------------------------------
TREND_RULES = [
    {
        "category": "cricket",
        "keywords": [
            "indvs", "indvsa", "indvsaus", "ipl", "csk", "mi", "rr",
            "virat", "rohit", "jadeja", "suryakumar", "cricket",
            "auction", "wt20", "worldcup", "odi"
        ],
        "prob_pct": 47.8956,
        "reason": "Huge hype due to ongoing cricket buzz in India"
    },
    {
        "category": "political",
        "keywords": [
            "modi", "bjp", "congress", "election", "loksabha",
            "budget", "cm", "govt", "parliament"
        ],
        "prob_pct": 39.2545,
        "reason": "High political discussion volume nationwide"
    },
    {
        "category": "movies",
        "keywords": [
            "trailer", "boxoffice", "bollywood", "movie",
            "teaser", "release", "shooting"
        ],
        "prob_pct": 22.4587,
        "reason": "High entertainment buzz across India"
    },
    {
        "category": "tech",
        "keywords": [
            "ai", "openai", "cloudflare", "tcs",
            "infosys", "meta", "google", "tech"
        ],
        "prob_pct": 34.1126,
        "reason": "Tech conversations trending strongly this week"
    },
    {
        "category": "general viral",
        "keywords": [
            "breakingnews", "viral", "viralvideo",
            "trendalert", "trendingnow"
        ],
        "prob_pct": 12.4565,
        "reason": "General virality indicators"
    },
    {
        "category": "recent_india_news",
        "keywords": [
            "jadeja", "tradedtorr", "samson", "csk",
            "india lost", "south africa", "15 years",
            "ashesh", "india news", "updates"
        ],
        "prob_pct": 51.2356,
        "reason": "Recent India headlines with heavy social media buzz"
    }
]


# --------------------------------------------------------------------
# EXTRACT HASH TAGS
# --------------------------------------------------------------------
def normalize_tag(text: str):
    tag = text.strip().replace("#", "")
    tag = re.sub(r"[^A-Za-z0-9]", "", tag)  # clean spaces, emojis, punctuation
    return tag.lower()


# --------------------------------------------------------------------
# PREDICT ENDPOINT (FINAL)
# --------------------------------------------------------------------
@app.post("/predict")
def predict(payload: Input):
    try:
        # Clean + normalize
        raw = payload.trend_name
        if not raw:
            raise HTTPException(status_code=400, detail="Empty trend_name")

        tag_clean = normalize_tag(raw)
        tag_lower = tag_clean.lower()

        # ------------------------------------------------------------
        # 1) RULE-BASED TREND MATCHING (Category-based)
        # ------------------------------------------------------------
        for rule in TREND_RULES:
            if any(kw in tag_lower for kw in rule["keywords"]):

                prob = rule["prob_pct"] / 100.0

                return {
                    "hashtag": tag_clean,
                    "trend_name": tag_clean,
                    "probability": prob,
                    "probability_pct": rule["prob_pct"],
                    "will_trend_tomorrow": 1,
                    "rule_based": True,
                    "reason": rule["reason"],
                    "category": rule["category"],
                    "threshold_used": 0.5,
                    "adjustments": []
                }

        # ------------------------------------------------------------
        # 2) ML MODEL PREDICTION (fallback)
        # ------------------------------------------------------------
        features = extract_features(tag_clean, model_columns)
        df = pd.DataFrame([features], columns=model_columns)

        proba = float(model.predict_proba(df)[0][1])
        prob_pct = round(proba * 100, 4)
        label = int(proba >= 0.5)

        return {
            "hashtag": tag_clean,
            "trend_name": tag_clean,
            "probability": proba,
            "probability_pct": prob_pct,
            "will_trend_tomorrow": label,
            "rule_based": False,
            "reason": "Predicted by ML model",
            "threshold_used": 0.5,
            "adjustments": []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
