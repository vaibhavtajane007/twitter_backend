import requests

API_URL = "http://127.0.0.1:8000/predict"

samples = [
    "Check this out! #ayush",
    "Breaking news #election",
    "I love this! #music",
    "Random chat without hashtag",
]

for s in samples:
    payload = {"tweet": s}
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        if r.status_code == 200:
            print(s, "->", r.json())
        else:
            print(s, "-> HTTP", r.status_code, r.text)
    except Exception as e:
        print(s, "-> Error:", e)
