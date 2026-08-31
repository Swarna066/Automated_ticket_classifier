"""
Simple Streamlit demo for the ticket classifier.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Let this file import from src/ regardless of where streamlit is launched from
sys.path.append(str(Path(__file__).parent / "src"))

from predict import classify_ticket  # noqa: E402

st.set_page_config(page_title="Ticket Classifier", page_icon="🎫")

st.title("🎫 Automated Financial Complaint Classification ")
st.write(
    "Paste in a customer ticket / complaint below and the model will "
    "predict which category it belongs to."
)

ticket_text = st.text_area(
    "Ticket text",
    height=180,
    placeholder="e.g. I was charged an overdraft fee I don't recognise on my checking account...",
)

if st.button("Classify", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter some ticket text first.")
    else:
        with st.spinner("Classifying..."):
            try:
                category = classify_ticket(ticket_text)
                st.success(f"Predicted category: **{category}**")
            except FileNotFoundError:
                st.error(
                    "No trained model found. Run `python src/train.py` "
                    "first to create the model files in `models/`."
                )

with st.expander("How does this work?"):
    st.write(
        "The ticket text is cleaned (lowercased, tokenized, lemmatized, "
        "and reduced to its key nouns), converted into TF-IDF features, "
        "and passed to a Logistic Regression model trained on labeled "
        "example tickets."
    )
