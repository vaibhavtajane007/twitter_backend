# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

DATA = "cleaned.csv"
MODEL_PATH = "model/final_twitter_model.pkl"
COLS_PATH = "model/model_columns.json"

df = pd.read_csv(DATA)

TARGET = "will_trend_tomorrow"

y = df[TARGET]
X = df.drop(columns=[TARGET])

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)

with open(COLS_PATH, "w") as f:
    json.dump(list(X.columns), f, indent=4)

print("Model trained & saved.")
