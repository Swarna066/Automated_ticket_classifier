"""
Text preprocessing utilities for the ticket classifier.

This is the exact cleaning pipeline used in the original notebooks,
just packaged into reusable functions so it can be called both when
building the training data AND when classifying a brand-new ticket
(both sides MUST use the same cleaning steps, or the model will see
words it doesn't recognise).

Pipeline:
    1. clean_text      -> lowercase, tokenize, drop punctuation/symbols
    2. lemmatize_text   -> reduce words to root form, drop stopwords (spaCy)
    3. pos_tags         -> keep only singular common nouns (tag 'NN')
    4. remove the "xxxx" placeholder the original dataset uses to mask
       personal info (names, account numbers, etc.)
"""

import string

import nltk
import spacy

# These downloads are cached after the first run, so this is cheap on
# every later import.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Loaded once at import time - loading a spaCy model is slow, so we
# don't want to redo it for every single ticket.
_nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """Lowercase the text, tokenize it, and keep only alphanumeric words."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t.isalnum()]          # drop punctuation/symbols
    tokens = [t for t in tokens if t not in string.punctuation]

    return " ".join(tokens)


def lemmatize_text(text: str) -> str:
    """Reduce each word to its base form and drop common stopwords."""
    doc = _nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(lemmatized)


def pos_tags(text: str) -> str:
    """Keep only singular common nouns (POS tag 'NN') - these carry most
    of the topic signal (e.g. 'card', 'account', 'payment', 'fee')."""
    doc = _nlp(text)
    nouns = [tok.lemma_ for tok in doc if tok.tag_ == "NN"]
    return " ".join(nouns)


def preprocess(raw_text: str) -> str:
    """Run the full cleaning pipeline on one piece of text.

    Use this single function everywhere (training data prep AND
    prediction time) so the two never drift apart.
    """
    text = clean_text(raw_text)
    text = lemmatize_text(text)
    text = pos_tags(text)
    text = text.replace("xxxx", "")
    return text
