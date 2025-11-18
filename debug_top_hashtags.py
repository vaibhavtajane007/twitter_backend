import pandas as pd
import numpy as np
import joblib
import json

MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"
EMBED_PATH = "bert_embeddings.csv"

def main():
    print("🔹 Loading model...")
    model = joblib.load(MODEL_PATH)

    print("🔹 Loading model columns...")
    with open(COLS_PATH, "r") as f:
        model_columns = json.load(f)

    print("🔹 Loading bert_embeddings.csv (this may take a bit)...")
    df = pd.read_csv(EMBED_PATH)

    if "will_trend_tomorrow" not in df.columns:
        raise SystemExit("❌ 'will_trend_tomorrow' not found in bert_embeddings.csv")

    y = df["will_trend_tomorrow"].astype(int)

    # Build X exactly like in train_model.py
    X = df.drop(columns=["will_trend_tomorrow"], errors="ignore")

    # Drop non-feature columns
    for col in ["trend_date", "trend_name"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    unnamed_cols = [c for c in X.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        X = X.drop(columns=unnamed_cols, errors="ignore")

    # Keep only model_columns (to ensure same order)
    X = X[model_columns]

    print("🔹 Computing probabilities on training data...")
    probs = model.predict_proba(X)[:, 1]

    df_result = pd.DataFrame({
        "trend_name": df.get("trend_name", pd.Series(["?"] * len(df))),
        "trend_date": df.get("trend_date", pd.Series(["?"] * len(df))),
        "will_trend_tomorrow": y,
        "proba": probs,
    })

    # Show basic stats
    print("\n📊 Probability stats:")
    print(df_result["proba"].describe())

    # Show some of the top 20 highest predicted
    top = df_result.sort_values("proba", ascending=False).head(20)
    print("\n🔥 Top 20 hashtags by predicted probability:")
    print(top[["trend_date", "trend_name", "will_trend_tomorrow", "proba"]])

if __name__ == "__main__":
    main()
