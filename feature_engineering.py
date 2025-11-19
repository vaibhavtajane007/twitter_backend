# feature_engineering.py
import re
from textblob import TextBlob
from datetime import datetime

sports_words = ["vs", "match", "cup", "league", "ipl", "wc", "ind", "aus", "final"]
political_words = ["modi", "bjp", "trump", "election", "congress", "minister"]
event_words = ["day", "festival", "birthday", "anniversary"]

def extract_features(tag: str, model_columns: list):
    """
    Builds a full feature vector for the model,
    filling missing values with 0.
    """

    tag_lower = tag.lower()

    features = {
        "trend_length": len(tag),
        "has_numbers": int(any(ch.isdigit() for ch in tag)),
        "sentiment_score": TextBlob(tag).sentiment.polarity,
        "is_sports": int(any(w in tag_lower for w in sports_words)),
        "is_political": int(any(w in tag_lower for w in political_words)),
        "is_event": int(any(w in tag_lower for w in event_words)),
        "day_of_week": datetime.utcnow().weekday(),
        "is_weekend": int(datetime.utcnow().weekday() >= 5),
    }

    # Create full row in correct order
    final = []
    for col in model_columns:
        final.append(features.get(col, 0))

    return final
