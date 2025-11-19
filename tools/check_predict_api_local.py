import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

samples = [
    "Check this out! #ayush",
    "Breaking news #election",
    "I love this! #music",
    "Random chat without hashtag",
    "Early morning vibes #GoodMorning",
    "Major headline #Trump",
]

for s in samples:
    resp = client.post("/predict", json={"tweet": s})
    print(s, "->", resp.status_code, resp.json())
