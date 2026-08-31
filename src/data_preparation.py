"""
Step 1: Build a labeled training set from the raw complaints JSON.

The raw dataset (CFPB consumer complaints) does NOT come with a ready
"ticket category" column. The original notebook creates one using
NMF (Non-negative Matrix Factorization) topic modeling: it groups
similar complaints into 5 topics based on their words, then a human
looks at the top words of each topic and gives it a readable name.

IMPORTANT / honest note for a portfolio project:
These are "pseudo-labels" discovered automatically, not human-verified
ground truth. That's a perfectly reasonable way to bootstrap a labeled
dataset when no categories exist yet, but it means the classifier we
train later is really learning to reproduce these topic clusters
rather than a truly independent judgment of "customer intent". If you
later get access to real agent-assigned categories, swap those in for
better and more meaningful accuracy numbers.

Run this script once to produce data/tickets_labeled.csv, which
train.py then uses.
"""

import json

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from preprocessing import preprocess

RAW_JSON_PATH = "../data/complaints.json"     # put your raw file here
OUTPUT_CSV_PATH = "../data/tickets_labeled.csv"

# Human-readable names for the 5 discovered topics. If you re-run NMF
# on different data, re-check these - the topic *numbers* NMF assigns
# are arbitrary and can come out in a different order each time.
TOPIC_NAMES = {
    4: "others",
    2: "Mortgage/Loan management",
    1: "Bank Account services",
    3: "Credit card / Prepaid card",
    0: "Theft/Dispute Reporting",
}


def load_raw_complaints(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    df = df[["_source.complaint_id", "_source.complaint_what_happened"]]
    df.columns = ["complaint_id", "complaint_what_happened"]

    # Drop complaints with no actual text
    df = df[df["complaint_what_happened"] != ""]
    return df


def build_labeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning text (this can take a couple of minutes)...")
    tqdm.pandas()
    df["complaint_clean"] = df["complaint_what_happened"].progress_apply(preprocess)

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    tfidf = vectorizer.fit_transform(df["complaint_clean"])

    print("Discovering topics with NMF...")
    nmf_model = NMF(n_components=5, init="random", random_state=40)
    doc_topic_matrix = nmf_model.fit_transform(tfidf)

    df["Label"] = doc_topic_matrix.argmax(axis=1)
    df["Label"] = df["Label"].map(TOPIC_NAMES)

    return df[["complaint_id", "complaint_clean", "Label"]]


if __name__ == "__main__":
    raw_df = load_raw_complaints(RAW_JSON_PATH)
    labeled_df = build_labeled_dataset(raw_df)
    labeled_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(labeled_df)} labeled tickets to {OUTPUT_CSV_PATH}")
    print(labeled_df["Label"].value_counts())