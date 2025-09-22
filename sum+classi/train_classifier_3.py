import json
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ------------------------------
# Load dataset
# ------------------------------
train_dataset = load_dataset("json", data_files={"train": "sum+classi/train_data.json"})["train"]
test_dataset = load_dataset("json", data_files={"test": "sum+classi/test_data.json"})["test"]

# ------------------------------
# Prepare label mappings
# ------------------------------
labels = list(set(train_dataset["label"]))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("Label mapping:", label2id)

# ------------------------------
# Load tokenizer & model
# ------------------------------
tokenizer = BertTokenizerFast.from_pretrained("nlpaueb/legal-bert-base-uncased")

def preprocess(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",  # ensures equal length
        max_length=512
    )

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# ------------------------------
# Encode labels into integers
# ------------------------------
def encode_labels(batch):
    batch["labels"] = label2id[batch["label"]]
    return batch

train_dataset = train_dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------
# Load model
# ------------------------------
model = BertForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# ------------------------------
# Training arguments
# ------------------------------
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",   # use "steps" instead of old argument
    eval_steps=500,                # evaluate every 500 steps
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

# ------------------------------
# Data collator
# ------------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ------------------------------
# Train
# ------------------------------
trainer.train()

# ------------------------------
# Save model
# ------------------------------
model.save_pretrained("./legal-bert-classifier")
tokenizer.save_pretrained("./legal-bert-classifier")

print("✅ Training complete! Model saved in ./legal-bert-classifier")
