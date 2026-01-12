import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load vectorized data
X = joblib.load("../models/X_tfidf.pkl")
y = joblib.load("../models/y.pkl")

# Encode sentiment labels (Positive, Neutral, Negative → 0,1,2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Multiclass Logistic Regression
model = LogisticRegression(max_iter=2000, solver='lbfgs')

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and encoder
joblib.dump(model, "../models/sentiment_model.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")

print("\nModel and Label Encoder saved.")

# ============================
# Save Model Performance Report
# ============================

report_text = f"""
MODEL PERFORMANCE REPORT

Accuracy:
{accuracy_score(y_test, y_pred)}

Classification Report:
{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}

Confusion Matrix:
{confusion_matrix(y_test, y_pred)}
"""

with open("../reports/model_performance_report.txt", "w") as f:
    f.write(report_text)

print("Model performance report saved in reports/model_performance_report.txt")

