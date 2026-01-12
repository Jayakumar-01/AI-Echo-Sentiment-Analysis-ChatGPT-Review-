import pandas as pd

file_path = "../data/datachatgpt_style_reviews_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)
