"""
Step 2: Train the ticket classifier.

This mirrors the modeling notebook: TF-IDF features (via
CountVectorizer + TfidfTransformer) feeding a Logistic Regression
model, tuned with GridSearchCV + StratifiedKFold. Logistic Regression
is used here because, in the original comparison across 5 models
(Logistic Regression, Naive Bayes, Decision Tree, Random Forest,
Gradient Boosting), it came out on top.

NOTE on a bug fix vs. the original notebook:
The original `model_build` helper called
    GridSearchCV(model, param_grid={}, ...)
which ignores the `param_grid` argument and always passes an empty
grid - so no hyperparameter tuning ever actually happened, only a
single fit with default settings, wrapped in cross-validation. That's
fixed below (`param_grid=param_grid`) so the search space you define
is actually used. Because of that, this WILL take a while to run on
the full logistic regression grid - shrink `PARAM_GRID` below if you
want a quicker first run.
"""

import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

DATA_PATH = "../data/tickets_labeled.csv"
MODELS_DIR = "../models"

PARAM_GRID = {
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "max_iter": [200, 500],
    "class_weight": [None, "balanced"],
}


def build_model(X_train, y_train, param_grid):
    """Cross-validated hyperparameter search.

    StratifiedKFold keeps the class balance the same in every fold,
    which matters here since the ticket categories aren't evenly sized.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
    grid = GridSearchCV(
        LogisticRegression(),
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    return grid.best_estimator_


def main():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)

    X = df["complaint_clean"]
    y = df["Label"]

    # Bag-of-words counts, then re-weighted by TF-IDF.
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(X)

    tfidf_trans = TfidfTransformer()
    X_vec = tfidf_trans.fit_transform(counts)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=40, stratify=y
    )

    model = build_model(X_train, y_train, PARAM_GRID)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\nClassification report - training data:")
    print(classification_report(y_train, y_pred_train))
    print("\nClassification report - test data:")
    print(classification_report(y_test, y_pred_test))

    pickle.dump(model, open(f"{MODELS_DIR}/model.pkl", "wb"))
    pickle.dump(count_vect, open(f"{MODELS_DIR}/count.pkl", "wb"))
    pickle.dump(tfidf_trans, open(f"{MODELS_DIR}/tfidf.pkl", "wb"))
    print(f"\nSaved model and vectorizers to {MODELS_DIR}/")


if __name__ == "__main__":
    main()