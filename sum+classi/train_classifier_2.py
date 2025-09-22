import json, argparse, joblib, os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load dataset
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    X, y = [], []
    for d in data:
        X.append(d["tokens"])  # use pre-tokenized tokens
        y.append(d["label"])
    return X, y

# Custom analyzer (instead of lambda) so it can be pickled
def identity(x):
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # Load train/test sets
    X_train, y_train = load_data(args.train_data)
    X_test, y_test = load_data(args.test_data)

    # Vectorizer with custom analyzer
    vectorizer = CountVectorizer(analyzer=identity)

    clf = Pipeline([
        ("vec", vectorizer),
        ("logreg", LogisticRegression(max_iter=300))
    ])

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(clf, args.model_path)
    print(f"\n✅ Model saved to {args.model_path}")
