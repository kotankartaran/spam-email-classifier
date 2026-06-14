import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

# Keep only required columns
df = df[["v1", "v2"]]

df.columns = ["label", "message"]

print("Dataset Shape:", df.shape)

# ==========================
# ENCODE TARGET
# ==========================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# ==========================
# FEATURES & TARGET
# ==========================

X = df["message"]
y = df["label"]

# ==========================
# TF-IDF VECTORIZATION
# ==========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# MODEL
# ==========================

model = MultinomialNB()

model.fit(X_train, y_train)

# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(
    model,
    "models/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("\nModel saved successfully!")