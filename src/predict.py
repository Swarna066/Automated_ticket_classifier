"""
Step 3: Classify a brand-new support ticket / complaint.

Loads the artifacts saved by train.py and runs the same preprocessing
pipeline used at training time before predicting.
"""

import pickle

from preprocessing import preprocess

MODELS_DIR = "../models"

_model = None
_count_vect = None
_tfidf_trans = None


def _load_artifacts():
    """Load the pickled model/vectorizers once, then reuse them."""
    global _model, _count_vect, _tfidf_trans
    if _model is None:
        _model = pickle.load(open(f"{MODELS_DIR}/model.pkl", "rb"))
        _count_vect = pickle.load(open(f"{MODELS_DIR}/count.pkl", "rb"))
        _tfidf_trans = pickle.load(open(f"{MODELS_DIR}/tfidf.pkl", "rb"))


def classify_ticket(raw_text: str) -> str:
    """Return the predicted category for one piece of raw ticket text."""
    _load_artifacts()

    cleaned = preprocess(raw_text)
    counts = _count_vect.transform([cleaned])
    vector = _tfidf_trans.transform(counts)

    return _model.predict(vector)[0]


if __name__ == "__main__":
    sample = (
        "I kindly request a full refund of the $50 service fee and an "
        "explanation of why this fee was charged. Additionally, I would "
        "appreciate confirmation that no further unauthorized charges "
        "will be applied to my account."
    )
    print("Ticket: ", sample)
    print("Predicted category:", classify_ticket(sample))
