import numpy as np
import re
from datetime import datetime

# -------------------------
# Basic NLP helpers
# -------------------------
def detect_category(tag):
    tag_l = tag.lower()
    return {
        "is_sports": int(any(w in tag_l for w in ["vs", "match", "fc", "cricket", "cup", "league"])),
        "is_political": int(any(w in tag_l for w in ["modi", "bjp", "congress", "election", "vote"])),
        "is_event": int(any(w in tag_l for w in ["festival", "day", "birthday", "anniversary", "award", "launch"]))
    }

def sentiment_from_text(tag):
    pos = ["love", "great", "happy", "win", "amazing"]
    neg = ["hate", "bad", "anger", "worst", "sad"]
    score = 0
    tl = tag.lower()
    for p in pos:
        if p in tl:
            score += 1
    for n in neg:
        if n in tl:
            score -= 1
    return score

# -------------------------
# Main feature builder
# -------------------------
def add_features(hashtag, emb):

    today = datetime.now()

    cat = detect_category(hashtag)

    features = {
        "avg_rank": 10,
        "min_rank": 10,
        "max_rank": 10,
        "total_volume": 50000,
        "appearances": 1,
        "is_most_tweeted": 0,
        "is_longest_trending": 0,
        "days_diff": 1,
        "prev_rank": 20,
        "prev_volume": 20000,
        "rank_change": -10,
        "volume_change": 30000,
        "day_of_week": today.weekday(),
        "is_weekend": int(today.weekday() in [5, 6]),
        "month": today.month,
        "days_since_last_trend": 999,
        "recent_streak": 1,
        "momentum_score": 1.2,
        "volatility": 0.1,
        "trend_length": 1,
        "has_numbers": int(bool(re.search(r"\d", hashtag))),
        "sentiment_score": sentiment_from_text(hashtag),
        "rolling_avg_volume_3": 50000,
        "rolling_avg_rank_3": 15,
        "trend_freq_7d": 1,
        "is_sports": cat["is_sports"],
        "is_political": cat["is_political"],
        "is_event": cat["is_event"]
    }

    # add embedding
    for i in range(len(emb)):
        features[f"emb{i}"] = float(emb[i])

    return features
