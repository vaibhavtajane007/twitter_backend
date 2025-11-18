import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ----------------------------
# Config
# ----------------------------
EMBED_PATH = "bert_embeddings.csv"   # using this directly
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
# Build X (drop non-feature columns)
# ----------------------------
X = df.drop(columns=["will_trend_tomorrow"], errors="ignore")

# Drop obviously non-numeric ID/text columns
for col in ["trend_date", "trend_name"]:
    if col in X.columns:
        X = X.drop(columns=[col])

# Drop any "Unnamed" index-like columns
unnamed_cols = [c for c in X.columns if c.startswith("Unnamed")]
if unnamed_cols:
    X = X.drop(columns=unnamed_cols, errors="ignore")

# Convert any remaining object columns to numeric (coerce errors to NaN)
non_numeric = [c for c in X.columns if X[c].dtype == "object"]
for c in non_numeric:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# Keep only numeric columns
numeric_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
X = X[numeric_cols]

# Fill NaNs
X = X.fillna(0)

print(f"✅ Using {len(numeric_cols)} numeric feature columns for training.")

# ----------------------------
# Save model columns
# ----------------------------
model_columns = list(X.columns)
with open(os.path.join(MODEL_DIR, "model_columns.json"), "w") as f:
    json.dump(model_columns, f)
print(f"💾 Saved {len(model_columns)} model columns → model/model_columns.json")

# ----------------------------
# Train-test split
# ----------------------------
print("\nTARGET VALUE COUNTS:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ----------------------------
# XGBoost model (no scaler, since features are already numeric)
# ----------------------------
model = XGBClassifier(
    n_estimators=300,
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

# ----------------------------
# Evaluate
# ----------------------------
acc = model.score(X_test, y_test)
print(f"\n📊 Model Accuracy: {acc:.4f}")

# ----------------------------
# Save model
# ----------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "final_twitter_model.pkl"))
print("\n💾 Model saved to model/final_twitter_model.pkl")
print("✅ Done.")
