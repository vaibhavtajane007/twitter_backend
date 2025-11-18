import pandas as pd
import joblib
import json
import os

MODEL_DIR = "model"
EMBED_PATH = "bert_embeddings.csv"

print("🔹 Loading model...")
model = joblib.load(os.path.join(MODEL_DIR, "final_twitter_model.pkl"))

with open(os.path.join(MODEL_DIR, "model_columns.json"), "r") as f:
    MODEL_COLUMNS = json.load(f)

print("🔹 Loading embeddings (memory-safe)...")

# Load ONLY needed columns (reduces memory by 15x)
usecols = ["trend_name", "will_trend_tomorrow"] + MODEL_COLUMNS

df = pd.read_csv(EMBED_PATH, usecols=usecols)
print("Loaded:", df.shape)

# remove date, name from X (if present)
X = df[MODEL_COLUMNS].fillna(0)
y = df["will_trend_tomorrow"]

print("\n📊 Start predicting probabilities (batch mode)...")

# predict in batches to avoid RAM overload
batch_size = 5000
probas = []

for i in range(0, len(X), batch_size):
    batch = X.iloc[i:i+batch_size]
    p = model.predict_proba(batch)[:, 1]
    probas.extend(p)

df["proba"] = probas

print("\n🔥 Top hashtags predicted most likely to trend tomorrow:")
top = df.sort_values("proba", ascending=False).head(30)
print(top[["trend_name", "will_trend_tomorrow", "proba"]])
