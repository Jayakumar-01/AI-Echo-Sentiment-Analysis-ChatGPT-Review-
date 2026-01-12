import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Echo – Sentiment Insights Dashboard", layout="wide")

st.title("📊 AI Echo – Sentiment Insights Dashboard")
st.write("This dashboard provides analytical insights from user reviews of ChatGPT.")

# Load dataset
df = pd.read_csv("../data/labeled_reviews.csv")

st.success("Dataset loaded successfully!")
st.write("Preview of Dataset:")
st.dataframe(df.head())

# ============================
# 1. Distribution of Review Ratings
# ============================

st.subheader("📊 1. Distribution of Review Ratings")

fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x="rating", data=df, ax=ax)
ax.set_title("Distribution of Review Ratings (1 to 5 Stars)")
ax.set_xlabel("Rating")
ax.set_ylabel("Number of Reviews")

st.pyplot(fig)

# ============================
# 2. Helpful Reviews Analysis
# ============================

st.subheader("👍👎 2. Helpful vs Not Helpful Reviews")

helpful_threshold = 10

df["helpful_flag"] = df["helpful_votes"].apply(
    lambda x: "Helpful (>=10 votes)" if x >= helpful_threshold else "Not Helpful (<10 votes)"
)

helpful_counts = df["helpful_flag"].value_counts()

fig2, ax2 = plt.subplots(figsize=(5,5))
ax2.pie(helpful_counts, labels=helpful_counts.index, autopct='%1.1f%%', startangle=90)
ax2.set_title("Helpful vs Not Helpful Reviews")

st.pyplot(fig2)

# ============================
# 3. Keywords in Positive vs Negative Reviews (Word Clouds)
# ============================

from wordcloud import WordCloud

st.subheader("🧭 3. Common Keywords in Positive vs Negative Reviews")

# Separate reviews based on rating
positive_reviews = df[df["rating"] >= 4]["review"].dropna()
negative_reviews = df[df["rating"] <= 2]["review"].dropna()

# Combine text
positive_text = " ".join(positive_reviews)
negative_text = " ".join(negative_reviews)

# Generate word clouds
positive_wc = WordCloud(width=400, height=300, background_color="white").generate(positive_text)
negative_wc = WordCloud(width=400, height=300, background_color="black").generate(negative_text)

# Display side by side using columns
col1, col2 = st.columns(2)

with col1:
    st.write("Positive Reviews (4–5 Stars)")
    fig_pos, ax_pos = plt.subplots(figsize=(5,4))
    ax_pos.imshow(positive_wc, interpolation="bilinear")
    ax_pos.axis("off")
    st.pyplot(fig_pos)

with col2:
    st.write("Negative Reviews (1–2 Stars)")
    fig_neg, ax_neg = plt.subplots(figsize=(5,4))
    ax_neg.imshow(negative_wc, interpolation="bilinear")
    ax_neg.axis("off")
    st.pyplot(fig_neg)

# ============================
# 4. Average Rating Over Time
# ============================

st.subheader("📆 4. Average Rating Over Time")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Group by month and calculate average rating
rating_trend = (
    df.groupby(df["date"].dt.to_period("M"))["rating"]
    .mean()
    .reset_index()
)
rating_trend["date"] = rating_trend["date"].astype(str)

fig4, ax4 = plt.subplots(figsize=(8,4))
ax4.plot(rating_trend["date"], rating_trend["rating"], marker="o")
ax4.set_title("Average Rating Over Time")
ax4.set_xlabel("Date (Month)")
ax4.set_ylabel("Average Rating")
ax4.tick_params(axis='x', rotation=45)

st.pyplot(fig4)

# ============================
# 5. Ratings by User Location
# ============================

st.subheader("🌍 5. Average Rating by User Location")

# Take top 10 locations by number of reviews
top_locations = df["location"].value_counts().head(10).index
location_data = df[df["location"].isin(top_locations)]

avg_rating_by_location = (
    location_data.groupby("location")["rating"]
    .mean()
    .sort_values(ascending=False)
)

fig5, ax5 = plt.subplots(figsize=(8,4))
avg_rating_by_location.plot(kind="bar", ax=ax5)
ax5.set_title("Average Rating by Top 10 User Locations")
ax5.set_xlabel("Location")
ax5.set_ylabel("Average Rating")
ax5.tick_params(axis='x', rotation=45)

st.pyplot(fig5)

# ============================
# 6. Ratings by Platform (Web vs Mobile)
# ============================

st.subheader("🧑‍💻 6. Average Rating by Platform")

platform_avg_rating = df.groupby("platform")["rating"].mean()

fig6, ax6 = plt.subplots(figsize=(5,4))
platform_avg_rating.plot(kind="bar", ax=ax6)
ax6.set_title("Average Rating by Platform")
ax6.set_xlabel("Platform")
ax6.set_ylabel("Average Rating")
ax6.tick_params(axis='x', rotation=0)

st.pyplot(fig6)

# ============================
# 7. Verified vs Non-Verified User Satisfaction
# ============================

st.subheader("✅❌ 7. Verified vs Non-Verified User Satisfaction")

verified_avg_rating = df.groupby("verified_purchase")["rating"].mean()

fig7, ax7 = plt.subplots(figsize=(5,4))
verified_avg_rating.plot(kind="bar", ax=ax7)
ax7.set_title("Average Rating: Verified vs Non-Verified Users")
ax7.set_xlabel("Verified Purchase")
ax7.set_ylabel("Average Rating")
ax7.tick_params(axis='x', rotation=0)

st.pyplot(fig7)

# ============================
# 8. Average Review Length per Rating
# ============================

st.subheader("🔠 8. Average Review Length per Rating")

avg_length_per_rating = df.groupby("rating")["review_length"].mean()

fig8, ax8 = plt.subplots(figsize=(6,4))
avg_length_per_rating.plot(kind="bar", ax=ax8)
ax8.set_title("Average Review Length per Rating")
ax8.set_xlabel("Rating (1 to 5 Stars)")
ax8.set_ylabel("Average Review Length")
ax8.tick_params(axis='x', rotation=0)

st.pyplot(fig8)

# ============================
# 9. Most Mentioned Words in 1-Star Reviews
# ============================

st.subheader("💬 9. Most Mentioned Words in 1-Star Reviews")

from wordcloud import WordCloud

# Filter only 1-star reviews
one_star_reviews = df[df["rating"] == 1]["review"].dropna()

# Combine all text
one_star_text = " ".join(one_star_reviews)

# Generate word cloud
one_star_wc = WordCloud(
    width=600,
    height=400,
    background_color="white",
    colormap="Reds"
).generate(one_star_text)

# Plot
fig9, ax9 = plt.subplots(figsize=(6,4))
ax9.imshow(one_star_wc, interpolation="bilinear")
ax9.axis("off")
ax9.set_title("Most Mentioned Words in 1-Star Reviews")

st.pyplot(fig9)

# ============================
# 10. Average Rating by ChatGPT Version
# ============================

st.subheader("📱🧪 10. Average Rating by ChatGPT Version")

version_avg_rating = (
    df.groupby("version")["rating"]
    .mean()
    .sort_values(ascending=False)
)

fig10, ax10 = plt.subplots(figsize=(6,4))
version_avg_rating.plot(kind="bar", ax=ax10)
ax10.set_title("Average Rating by ChatGPT Version")
ax10.set_xlabel("ChatGPT Version")
ax10.set_ylabel("Average Rating")
ax10.tick_params(axis='x', rotation=0)

st.pyplot(fig10)
