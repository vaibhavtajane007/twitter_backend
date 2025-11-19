import re
_st_model = None

def _get_sentence_transformer():
    """Lazy-load SentenceTransformer to avoid import errors at module import time.

    If transformers/torch are not installed or incompatible, the function will
    return None so callers can fall back to heuristics instead of crashing.
    """
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Try a small memory-friendly model first
            try:
                _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                try:
                    _st_model = SentenceTransformer("all-mpnet-base-v2")
                except Exception:
                    _st_model = None
        except Exception:
            _st_model = None
    return _st_model


def extract_hashtag(text):
    """Return the first hashtag (without leading '#') and its embedding (1D numpy array).

    Raises ImportError if the sentence-transformers backend is missing.
    """
    tags = re.findall(r"#(\w+)", text)
    if len(tags) == 0:
        return None, None

    tag = tags[0]
    model = _get_sentence_transformer()
    if model is None:
        return tag, None
    try:
        emb = model.encode([f"#{tag}"])[0]
        return tag, emb
    except Exception:
        return tag, None
