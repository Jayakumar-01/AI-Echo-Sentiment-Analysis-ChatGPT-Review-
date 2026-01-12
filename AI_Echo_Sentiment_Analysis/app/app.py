import streamlit as st
import joblib
import numpy as np

# Load components
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

# Strong keyword rules
negative_keywords = ["worst", "waste", "useless", "bad", "terrible", "hate", "poor", "slow", "bug", "not working"]
positive_keywords = ["amazing", "excellent", "great", "awesome", "love", "perfect", "good", "fast", "useful"]

st.set_page_config(page_title="AI Echo - Sentiment Analysis", layout="centered")
st.title("🧠 AI Echo – Hybrid Sentiment Analysis")
st.write("Classifies reviews into Positive, Neutral, or Negative using NLP + ML.")

text = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_lower = text.lower()

        # 1. Rule-based override
        if any(word in text_lower for word in negative_keywords):
            st.error("😡 Negative Sentiment")
        elif any(word in text_lower for word in positive_keywords):
            st.success("😊 Positive Sentiment")
        else:
            # 2. ML-based prediction
            vec = vectorizer.transform([text_lower])
            probs = model.predict_proba(vec)[0]
            pred_index = np.argmax(probs)
            sentiment = label_encoder.inverse_transform([pred_index])[0]

            # 3. Confidence-based correction
            max_prob = np.max(probs)
            if max_prob < 0.50:
                sentiment = "Neutral"

            # Output
            if sentiment == "Positive":
                st.success("😊 Positive Sentiment")
            elif sentiment == "Neutral":
                st.info("😐 Neutral Sentiment")
            else:
                st.error("😡 Negative Sentiment")
