import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load labeled dataset
df = pd.read_csv("../data/labeled_reviews.csv")

print("Dataset loaded successfully")
print(df.head())
# 1. Distribution of Review Ratings
plt.figure(figsize=(8,5))
sns.countplot(x="rating", data=df, palette="viridis")
plt.title("Distribution of Review Ratings (1 to 5 Stars)")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show(block=False)


# ============================
# EDA Step 2: Helpful Votes Analysis
# ============================

helpful_threshold = 10

df["helpful_flag"] = df["helpful_votes"].apply(
    lambda x: "Helpful (>=10 votes)" if x >= helpful_threshold else "Not Helpful (<10 votes)"
)

helpful_counts = df["helpful_flag"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(helpful_counts,
        labels=helpful_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Helpful vs Not Helpful Reviews")
plt.tight_layout()
plt.show()
# ============================
# EDA Step 3: Word Clouds (Positive vs Negative)
# ============================

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Separate positive and negative reviews using ratings
positive_reviews = df[df["rating"] >= 4]["review"].dropna()
negative_reviews = df[df["rating"] <= 2]["review"].dropna()

# Combine text
positive_text = " ".join(positive_reviews)
negative_text = " ".join(negative_reviews)

# Generate word clouds
positive_wc = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
negative_wc = WordCloud(width=800, height=400, background_color="black").generate(negative_text)

# Plot
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.imshow(positive_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews (4–5 Stars)")

plt.subplot(1,2,2)
plt.imshow(negative_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews (1–2 Stars)")

plt.tight_layout()
plt.show()

# ============================
# EDA Step 4: Average Rating Over Time
# ============================

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Group by date and calculate average rating
rating_trend = df.groupby(df["date"].dt.to_period("M"))["rating"].mean()
rating_trend = rating_trend.reset_index()
rating_trend["date"] = rating_trend["date"].astype(str)

# Plot line chart
plt.figure(figsize=(10,5))
plt.plot(rating_trend["date"], rating_trend["rating"], marker="o")
plt.xticks(rotation=45)
plt.title("Average Rating Over Time")
plt.xlabel("Date (Month)")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()


# ============================
# EDA Step 5: Ratings by User Location
# ============================

# Take top 10 locations by number of reviews
top_locations = df["location"].value_counts().head(10).index
location_ratings = df[df["location"].isin(top_locations)]

avg_rating_by_location = (
    location_ratings
    .groupby("location")["rating"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,6))
avg_rating_by_location.plot(kind="bar")
plt.title("Average Rating by Top 10 User Locations")
plt.xlabel("Location")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================
# EDA Step 6: Ratings by Platform (Web vs Mobile)
# ============================

platform_avg_rating = df.groupby("platform")["rating"].mean()

plt.figure(figsize=(6,5))
platform_avg_rating.plot(kind="bar")
plt.title("Average Rating by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================
# EDA Step 7: Verified vs Non-Verified User Satisfaction
# ============================

verified_rating = df.groupby("verified_purchase")["rating"].mean()

plt.figure(figsize=(6,5))
verified_rating.plot(kind="bar")
plt.title("Average Rating: Verified vs Non-Verified Users")
plt.xlabel("Verified Purchase")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================
# EDA Step 8: Average Review Length per Rating
# ============================

avg_length_per_rating = df.groupby("rating")["review_length"].mean()

plt.figure(figsize=(8,5))
avg_length_per_rating.plot(kind="bar")
plt.title("Average Review Length per Rating")
plt.xlabel("Rating (1 to 5 Stars)")
plt.ylabel("Average Review Length")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================
# EDA Step 9: Most Mentioned Words in 1-Star Reviews
# ============================

from wordcloud import WordCloud

# Filter 1-star reviews
one_star_reviews = df[df["rating"] == 1]["review"].dropna()

# Combine all 1-star review text
one_star_text = " ".join(one_star_reviews)

# Generate word cloud
one_star_wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="Reds"
).generate(one_star_text)

# Plot word cloud
plt.figure(figsize=(10,5))
plt.imshow(one_star_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most Mentioned Words in 1-Star Reviews")
plt.tight_layout()
plt.show()

# ============================
# EDA Step 10: Average Rating by ChatGPT Version
# ============================

# Group by version and calculate average rating
version_avg_rating = df.groupby("version")["rating"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
version_avg_rating.plot(kind="bar")
plt.title("Average Rating by ChatGPT Version")
plt.xlabel("ChatGPT Version")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
