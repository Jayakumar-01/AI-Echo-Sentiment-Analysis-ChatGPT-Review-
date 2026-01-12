import pandas as pd
import re

# Load dataset
file_path = "../data/datachatgpt_style_reviews_dataset.csv"
df = pd.read_csv(file_path)

print("Before Cleaning:")
print(df.shape)

# 1. Remove duplicate rows
df.drop_duplicates(inplace=True)

# 2. Handle missing values
df = df.dropna(subset=["review", "rating"])

# 3. Convert rating to integer
df["rating"] = df["rating"].astype(int)

# 4. Clean review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)     # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text) # remove special characters
    text = re.sub(r"\s+", " ", text)        # remove extra spaces
    return text.strip()

df["cleaned_review"] = df["review"].apply(clean_text)

# 5. Save cleaned dataset
cleaned_path = "../data/cleaned_reviews.csv"
df.to_csv(cleaned_path, index=False)

print("\nAfter Cleaning:")
print(df.shape)
print("\nCleaned data saved to:", cleaned_path)
