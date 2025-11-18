import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

EMBED_PATH = "bert_embeddings.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("🔹 Loading training data from bert_embeddings.csv ...")
df = pd.read_csv(EMBED_PATH)
print("bert_embeddings.csv shape:", df.shape)

# ----------------------------
# Target
# ----------------------------
if "will_trend_tomorrow" not in df.columns:
    raise SystemExit("❌ 'will_trend_tomorrow' not found in bert_embeddings.csv")

y = df["will_trend_tomorrow"].astype(int)

# ----------------------------
# Build X: drop target + non-feature cols
# ----------------------------
X = df.drop(columns=["will_trend_tomorrow"], errors="ignore")

# Drop ID / text columns
for col in ["trend_date", "trend_name"]:
    if col in X.columns:
        X = X.drop(columns=[col])

# Drop unnamed index-like columns
unnamed_cols = [c for c in X.columns if c.startswith("Unnamed")]
if unnamed_cols:
    X = X.drop(columns=unnamed_cols, errors="ignore")

# Convert any remaining object cols to numeric (coerce)
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = pd.to_numeric(X[c], errors="coerce")

# Keep only numeric
numeric_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
X = X[numeric_cols].fillna(0)

print(f"✅ Using {len(numeric_cols)} numeric feature columns for training.")

# Save columns for app.py
with open(os.path.join(MODEL_DIR, "model_columns.json"), "w") as f:
    json.dump(list(X.columns), f)
print(f"💾 Saved {len(numeric_cols)} model columns → model/model_columns.json")

print("\nTARGET VALUE COUNTS:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=(len(y_train) / y_train.sum()) if y_train.sum() > 0 else 1,
)

print("\n🔹 Training model...")
model.fit(X_train, y_train)
print("✅ Training complete.")

acc = model.score(X_test, y_test)
print(f"\n📊 Model Accuracy: {acc:.4f}")

joblib.dump(model, os.path.join(MODEL_DIR, "final_twitter_model.pkl"))
print("\n💾 Model saved to model/final_twitter_model.pkl")
print("✅ Done.")
