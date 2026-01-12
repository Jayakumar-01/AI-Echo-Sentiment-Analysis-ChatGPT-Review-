import pandas as pd
from textblob import TextBlob

df = pd.read_csv("../data/cleaned_reviews.csv")

def assign_sentiment(row):
    rating = row["rating"]
    text = str(row["review"])

    polarity = TextBlob(text).sentiment.polarity

    # Strong negative
    if polarity <= -0.3:
        return "Negative"

    # Strong positive
    elif polarity >= 0.3:
        return "Positive"

    # Neutral text zone
    else:
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

df["sentiment"] = df.apply(assign_sentiment, axis=1)

print(df["sentiment"].value_counts())
df.to_csv("../data/labeled_reviews.csv", index=False)
print("Corrected sentiment labels saved.")
