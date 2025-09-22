#!/usr/bin/python3
import argparse, json, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--test_data", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

# Load dataset
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    X, y = [], []
    for d in data:
        X.append(d["sentence"])
        y.append(d["label"])
    return X, y

X_train, y_train = load_data(args.train_data)
X_test, y_test = load_data(args.test_data)

# Define model
clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("logreg", LogisticRegression(max_iter=300))
])

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
joblib.dump(clf, args.model_path)
print(f"✅ Model saved to {args.model_path}")
