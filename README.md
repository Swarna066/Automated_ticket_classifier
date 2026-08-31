# Automated Ticket Classifier

A text classifier that reads a customer support ticket / complaint and
predicts which category it belongs to (e.g. *Bank Account services*,
*Credit card / Prepaid card*, *Mortgage/Loan management*,
*Theft/Dispute Reporting*, or *others*), using classic NLP + TF-IDF +
Logistic Regression.

This project is built from an existing modeling notebook, reorganized
into a clean, runnable project with a small demo app.

## How it works (pipeline)

```
raw ticket text
      │
      ▼
1. Clean text        lowercase, tokenize, drop punctuation/symbols
2. Lemmatize          reduce words to root form, drop stopwords
3. Keep key nouns     only common nouns (POS tag 'NN') survive
      │
      ▼
4. TF-IDF vectorize   turn cleaned text into numeric features
      │
      ▼
5. Logistic Regression   predicts the ticket category
```

Steps 1-3 live in `src/preprocessing.py` and are used both when
building the training data and when classifying a brand-new ticket -
that consistency matters, since the model can only work with words in
the same form it was trained on.

## Project structure

```
ticket-classifier/
├── data/                     # put your dataset here (not included)
├── models/                   # trained model + vectorizers get saved here
├── src/
│   ├── preprocessing.py      # shared text-cleaning functions
│   ├── data_preparation.py   # raw complaints -> labeled CSV
│   ├── train.py              # trains and saves the classifier
│   └── predict.py            # loads the classifier, predicts one ticket
├── app.py                    # Streamlit demo
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

(NLTK's `punkt` tokenizer downloads automatically the first time you
import `preprocessing.py`.)

## Running it

1. **Prepare labeled data** (skip this if you already have a CSV with
   `complaint_clean` and `Label` columns - just place it at
   `data/tickets_labeled.csv`):
   ```bash
   cd src
   python data_preparation.py
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```
   This saves `model.pkl`, `count.pkl`, and `tfidf.pkl` into `models/`.

3. **Try it from the command line:**
   ```bash
   python predict.py
   ```

4. **Or launch the demo app:**
   ```bash
   cd ..
   streamlit run app.py
   ```

## Where the categories came from (important context)

The raw dataset (consumer complaints) doesn't ship with ready-made
ticket categories. `data_preparation.py` creates them using **NMF
topic modeling**: it groups similar complaints by their words into 5
clusters, and a human then names each cluster after inspecting its
top words (e.g. words like "mortgage, loan, foreclosure" → "Mortgage/
Loan management").

Worth knowing for a portfolio project: these are **pseudo-labels**,
not human-verified ground truth. The classifier is essentially
learning to reproduce the same TF-IDF-based clusters it was labeled
with, which is part of why accuracy on this data looks very high. If
you get access to real, human-assigned ticket categories, swapping
those in as your `Label` column will give you a more meaningful
accuracy number.

## About the models

The original notebook compared 5 classifiers (Logistic Regression,
Naive Bayes, Decision Tree, Random Forest, Gradient Boosting) and
found Logistic Regression performed best, so that's what `train.py`
uses.

One fix worth flagging: the original hyperparameter search
(`GridSearchCV`) was accidentally called with an empty parameter grid,
so no tuning actually happened — every model just ran once with
default settings inside cross-validation. `train.py` fixes this so the
parameter grid you define is actually searched. This does mean
training takes noticeably longer than the empty-grid version did;
shrink `PARAM_GRID` in `train.py` if you want a faster first run.

## Extending this project

- Swap in real ticket categories if you have them, instead of NMF topics.
- Try `TfidfVectorizer` directly instead of `CountVectorizer` +
  `TfidfTransformer` (mathematically equivalent, one less step).
- Add a confidence score to predictions (`model.predict_proba`).
- Try a small transformer model (e.g. DistilBERT) for a stronger baseline.
