import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("readme", sep="\t", header=None, names=["label", "message"])

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df.drop_duplicates(inplace=True)
df["message"] = df["message"].fillna("")
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# -----------------------------
# 3. Define Features & Target
# -----------------------------
X = df["message"]
y = df["label"]

# -----------------------------
# 4. Train-Test Split (Stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),   # Better feature extraction
    max_features=5000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# -----------------------------
# 6. Models
# -----------------------------
lr = LogisticRegression(class_weight="balanced")
rf = RandomForestClassifier(n_estimators=200)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# -----------------------------
# 7. Evaluation Function
# -----------------------------
def evaluate(y_true, y_pred, model_name):
    print(f"\n📌 {model_name}")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("-"*40)

evaluate(y_test, lr_pred, "Logistic Regression")
evaluate(y_test, rf_pred, "Random Forest")

# -----------------------------
# 8. Save Best Model
# -----------------------------
joblib.dump(lr, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
