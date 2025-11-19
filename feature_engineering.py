# feature_engineering.py

import re
from textblob import TextBlob
from datetime import datetime

sports_words = ["vs","match","cup","league","ipl","wc","ind","aus","final"]
political_words = ["modi","bjp","election","congress","minister"]
event_words = ["day","festival","birthday","anniversary"]

def extract_features(hashtag: str):
    """Return a CLEAN dictionary of features that match model_columns.json"""

    tag = hashtag.replace("#", "").strip()
    lower = tag.lower()

    features = {
        "trend_length": len(tag),
        "has_numbers": int(any(ch.isdigit() for ch in tag)),
        "sentiment_score": float(TextBlob(tag).sentiment.polarity),

        "is_sports": int(any(w in lower for w in sports_words)),
        "is_political": int(any(w in lower for w in political_words)),
        "is_event": int(any(w in lower for w in event_words)),

        # Time-based features
        "day_of_week": datetime.utcnow().weekday(),
        "is_weekend": int(datetime.utcnow().weekday() >= 5),
    }

    return features
