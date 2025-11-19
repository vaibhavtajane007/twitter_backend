# feature_engineering.py

import re
import numpy as np
import pandas as pd
from textblob import TextBlob
from datetime import datetime


def clean_text(text: str) -> str:
    """Basic cleaning: remove URLs, symbols, extra spaces."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9# ]+", " ", text)
    return text.strip().lower()


def compute_text_features(tag: str) -> dict:
    """Extracts simple interpretable features."""
    t = tag.lower()

    return {
        "trend_length": len(tag),
        "has_numbers": int(any(ch.isdigit() for ch in tag)),
        "sentiment_score": TextBlob(tag).sentiment.polarity,

        # Keyword-based binary flags
        "is_sports": int(any(w in t for w in [
            "vs", "match", "cup", "league", "ipl", "wc", "ind", "aus", "final"
        ])),
        "is_political": int(any(w in t for w in [
            "modi", "bjp", "election", "congress", "minister", "trump", "biden"
        ])),
        "is_event": int(any(w in t for w in [
            "day", "festival", "birthday", "anniversary"
        ])),
    }


def align_features_to_model(features: dict, model_columns: list) -> np.ndarray:
    """
    Ensures the final feature vector exactly matches the columns
    used during training (order + missing columns filled with 0).
    """
    aligned = []

    for col in model_columns:
        aligned.append(features.get(col, 0))

    return np.array(aligned)


def extract_features(trend_name: str, model_columns=None) -> np.ndarray:
    """
    Main function: clean → extract → align.
    Used directly by /predict endpoint.
    """
    trend_name = clean_text(trend_name)

    feat = compute_text_features(trend_name)

    if model_columns is None:
        raise ValueError("model_columns not provided to extract_features()")

    final_vector = align_features_to_model(feat, model_columns)
    return final_vector
