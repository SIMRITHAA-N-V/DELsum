#!/usr/bin/python3
import argparse, os, json, nltk
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Folder with labeled documents")
parser.add_argument("--prep_path", type=str, required=True, help="Where to save train/test json")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.prep_path, exist_ok=True)

# Each file: one sentence per line -> "sentence<TAB>label"
all_sentences, all_labels = [], []
for fname in os.listdir(args.data_path):
    if not fname.endswith(".txt"): 
        continue
    with open(os.path.join(args.data_path, fname), "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2: 
                continue
            sentence, label = parts
            tokens = nltk.word_tokenize(sentence)
            all_sentences.append((sentence, tokens))
            all_labels.append(label)

# Split
train_X, test_X, train_y, test_y = train_test_split(
    all_sentences, all_labels, 
    test_size=args.test_size, random_state=args.random_state, stratify=all_labels
)

# Save
train_data, test_data = [], []
for (s, tokens), label in zip(train_X, train_y):
    train_data.append({"sentence": s, "tokens": tokens, "label": label})
for (s, tokens), label in zip(test_X, test_y):
    test_data.append({"sentence": s, "tokens": tokens, "label": label})

with open(os.path.join(args.prep_path, "train_data.json"), "w") as f:
    json.dump(train_data, f, indent=4)
with open(os.path.join(args.prep_path, "test_data.json"), "w") as f:
    json.dump(test_data, f, indent=4)

print(f"✅ Data prepared:\n  Train -> {len(train_data)} samples\n  Test -> {len(test_data)} samples")
