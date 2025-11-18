import numpy as np
import pandas as pd

def explain(hashtag, proba, similar_trends):

    reasons = []

    if proba > 0.7:
        reasons.append("High similarity to previously trending high-volume hashtags.")

    if any("India" in s for s in similar_trends):
        reasons.append("Related to national political or social topics which trend often.")

    if any("Day" in hashtag):
        reasons.append("Hashtag contains day/event keywords which usually spike in trends.")

    if len(reasons) == 0:
        reasons.append("Moderate trend probability due to semantic similarity and pattern behavior.")

    return " ".join(reasons)
