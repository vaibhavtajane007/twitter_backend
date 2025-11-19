import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

# further split train into train/val for calibration & threshold tuning
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    # set scale_pos_weight = (num_negative / num_positive)
    scale_pos_weight=( (len(y_train_sub) - y_train_sub.sum()) / y_train_sub.sum() ) if y_train_sub.sum() > 0 else 1,
)

print("\n🔹 Training model...")
model.fit(X_train_sub, y_train_sub)
print("✅ Base model training complete.")

# Fit a scaler on training data and save it for the app
scaler = StandardScaler()
scaler.fit(X_train_sub)

# Calibrate probabilities using a held-out validation set
print("\n🔹 Calibrating probabilities (sigmoid)...")
calibrator = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
calibrator.fit(scaler.transform(X_val), y_val)
print("✅ Calibration complete.")

X_test_scaled = scaler.transform(X_test)
probs = calibrator.predict_proba(X_test_scaled)[:, 1]
preds = (probs > 0.5).astype(int)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
auc = roc_auc_score(y_test, probs)

print(f"\n📊 Model on test set — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# Save calibrated model and scaler
joblib.dump(calibrator, os.path.join(MODEL_DIR, "final_twitter_model_calibrated.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("\n💾 Calibrated model saved to model/final_twitter_model_calibrated.pkl")
print("💾 Scaler saved to model/scaler.pkl")

# Find best probability threshold on validation set (maximize F1)
val_probs = calibrator.predict_proba(scaler.transform(X_val))[:, 1]
best_thr = 0.5
best_f1 = 0.0
for thr in np.linspace(0.05, 0.95, 91):
    thr_pred = (val_probs > thr).astype(int)
    f1s = f1_score(y_val, thr_pred, zero_division=0)
    if f1s > best_f1:
        best_f1 = f1s
        best_thr = thr

with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
    json.dump({"threshold": float(best_thr), "best_val_f1": float(best_f1)}, f)
print(f"💾 Saved threshold {best_thr:.3f} (val F1={best_f1:.4f}) to model/threshold.json")

print("\n✅ Done.")
print("✅ Done.")
