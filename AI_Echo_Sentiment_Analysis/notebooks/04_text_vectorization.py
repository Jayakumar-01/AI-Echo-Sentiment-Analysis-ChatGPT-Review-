import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load labeled data
file_path = "../data/labeled_reviews.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded:", df.shape)

# Use the 'review' column for text (change if your column name is different)
text_column = "review"

# Check missing values
df = df.dropna(subset=[text_column, "sentiment"])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df[text_column])
y = df["sentiment"]

print("TF-IDF Shape:", X.shape)
print("Target Shape:", y.shape)

# Save vectorized data
import joblib

joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(X, "../models/X_tfidf.pkl")
joblib.dump(y, "../models/y.pkl")

print("\nTF-IDF Vectorizer and data saved in models/ folder")
