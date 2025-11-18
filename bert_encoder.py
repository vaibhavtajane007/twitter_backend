from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_hashtag(text):
    tags = re.findall(r"#\w+", text)
    if len(tags) == 0:
        return None, None
    return tags[0], model.encode([tags[0]])[0]
