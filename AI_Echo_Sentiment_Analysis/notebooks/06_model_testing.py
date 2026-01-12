import joblib
import numpy as np

# Load components
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

# Strong keyword rules
negative_keywords = ["worst", "waste", "useless", "bad", "terrible", "hate", "poor", "slow", "bug", "not working"]
positive_keywords = ["amazing", "excellent", "great", "awesome", "love", "perfect", "good", "fast", "useful"]

print("Hybrid Sentiment Analysis (Positive / Neutral / Negative)")
print("Type 'exit' to stop\n")

while True:
    text = input("Enter a review text: ").lower()

    if text == "exit":
        print("Exiting...")
        break

    # 1. Rule-based override (highest priority)
    if any(word in text for word in negative_keywords):
        print("Sentiment: Negative 😡\n")
        continue

    if any(word in text for word in positive_keywords):
        print("Sentiment: Positive 😊\n")
        continue

    # 2. ML-based prediction
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    pred_index = np.argmax(probs)
    sentiment = label_encoder.inverse_transform([pred_index])[0]

    # 3. Confidence-based correction
    max_prob = np.max(probs)
    if max_prob < 0.50:
        sentiment = "Neutral"

    # Output
    if sentiment == "Positive":
        print("Sentiment: Positive 😊\n")
    elif sentiment == "Neutral":
        print("Sentiment: Neutral 😐\n")
    else:
        print("Sentiment: Negative 😡\n")
