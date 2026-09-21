########################### Main training script ##########################


import pandas as pd
import torch
import os
import re

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import FakeNewsDataset
from model import load_model
from trainer import train_model
from evaluator import evaluate_model

from config import *
from transformers import DistilBertTokenizer


# ============================================================
# LEAK REMOVAL
# ============================================================
# True.csv is ~99.8% Reuters-sourced, and ~91% of its rows start
# with a "CITY (Reuters) -" dateline that never appears in Fake.csv.
# Left in, the model learns to key off that phrase/style instead of
# actual content (verified: prepending "(Reuters)" to a real BBC
# article flipped a 99% FAKE prediction to 100% REAL).
#
# We strip:
#   1. The leading "CITY (Reuters) -" dateline
#   2. Any remaining "(Reuters)" / "Reuters" mentions elsewhere in
#      the body (e.g. "...the Reuters poll showed...")

REUTERS_DATELINE_PATTERN = re.compile(
    r'^[A-Za-z][A-Za-z\.,/\-\s]{0,50}\(Reuters\)\s*-\s*'
)
REUTERS_INLINE_PATTERN = re.compile(
    r'\(?\bReuters\b\)?[,:]?\s*'
)


def strip_reuters_leak(text):
    text = REUTERS_DATELINE_PATTERN.sub("", text, count=1)
    text = REUTERS_INLINE_PATTERN.sub("", text)
    return text.strip()


# Load datasets
print("\nLoading Dataset...")

fake_df = pd.read_csv("model_training/Fake.csv")
true_df = pd.read_csv("model_training/True.csv")

# Remove the Reuters dateline/citation leak from the real-news class
print("\nStripping Reuters dateline/citation leak from True.csv...")

before_sample = true_df["text"].iloc[0][:80]

true_df["text"] = true_df["text"].apply(strip_reuters_leak)
fake_df["text"] = fake_df["text"].apply(strip_reuters_leak)

print("Before:", before_sample)
print("After :", true_df["text"].iloc[0][:80])

# ============================================================
# ADD SHORT-CLAIM DATA (LIAR / PolitiFact)
# ============================================================
# Fake.csv/True.csv are both full multi-paragraph articles (avg
# ~420 words). Real-world testing showed the model has no idea what
# to do with short, single-sentence claims (avg ~18 words) like
# those from PolitiFact/Snopes fact-checks - it just guesses REAL
# for anything short and formally-worded, fake or not.
#
# liar_binary.csv (built by model_training/build_liar_binary.py from
# the LIAR dataset, Wang 2017 ACL) is ~12.8K short PolitiFact
# statements already labeled real/fake in matching format, so we can
# fold it in directly to teach the model short-claim inputs too.

liar_path = "model_training/liar_binary.csv"

if os.path.exists(liar_path):
    print("\nLoading LIAR short-claim dataset...")
    liar_df = pd.read_csv(liar_path)
    print(f"LIAR rows: {len(liar_df)}  "
          f"(Real: {(liar_df['label'] == 1).sum()}, "
          f"Fake: {(liar_df['label'] == 0).sum()})")
else:
    print(f"\nNOTE: {liar_path} not found - skipping short-claim data. "
          f"Run model_training/build_liar_binary.py first if you want it included.")
    liar_df = pd.DataFrame(columns=["title", "text", "label"])

# Also drop 'subject' as a leak: True.csv only ever uses
# {politicsNews, worldnews} while Fake.csv uses a disjoint set of
# categories, so 'subject' alone can act as a near-perfect shortcut.
# (We already only keep ["content", "label"] below, so this is just
# documented here as a reminder not to reintroduce it as a feature.)

# Labels
fake_df["label"] = 0
true_df["label"] = 1

# Merge (articles + short claims together)
df = pd.concat([fake_df, true_df, liar_df], ignore_index=True)

# Shuffle
df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Combine title + text
df["content"] = (df["title"].fillna("") + " " + df["text"].fillna(""))

# Keep only required columns
df = df[["content", "label"]]

print("\nDataset Shape:", df.shape)

print("\nFake News:", (df["label"] == 0).sum())
print("Real News:", (df["label"] == 1).sum())

# Small dataset for testing
# df = df.sample(n=2000, random_state=42).reset_index(drop=True)

# Split
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    df["content"],
    df["label"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts,
    train_val_labels,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=train_val_labels
)

# TRAIN / VALIDATION SPLIT
print("\n================ DATA SPLIT ================")

print("Training Samples   :", len(train_texts))
print("Validation Samples :", len(val_texts))
print("Testing Samples    :", len(test_texts))


# Load DistilBERT Tokenizer
print("\nLoading DistilBERT Tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer Loaded Successfully!")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("Tokenizer Saved Successfully!")


# Tokenize training data
print("\nTokenizing Training Data...")

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

# TOKENIZE VALIDATION DATA
print("Tokenizing Validation Data...")

val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

# Tokenize testing data
print("Tokenizing Testing Data...")

test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=256
)

print("\nTokenization Completed!")

print("Training Samples Tokenized :", len(train_encodings["input_ids"]))
print("Validation Samples Tokenized :",len(val_encodings["input_ids"]))
print("Testing Samples Tokenized  :", len(test_encodings["input_ids"]))

# Create PyTorch Dataset
train_dataset = FakeNewsDataset(train_encodings, train_labels)
val_dataset = FakeNewsDataset(val_encodings,val_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

print("\nDatasets Created Successfully!")

# Create DataLoaders
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

print("DataLoaders Created Successfully!")

print("Training Batches   :", len(train_loader))
print("Validation Batches :", len(val_loader))
print("Testing Batches    :", len(test_loader))

# Load DistilBERT Model

print("\nLoading DistilBERT Model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

model = load_model()

model.to(device)

print("Model Loaded Successfully!")

# TRAIN MODEL

print("\nStarting Training...\n")

loss = train_model(model, train_loader, val_loader, device)

print(f"\nTraining Completed!")

print(f"Final Training Loss: {loss:.4f}")

print("\nStarting Final Test Evaluation...\n")

accuracy = evaluate_model(model,test_loader,device)

print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

print("\nSaving Tokenizer...")

tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("Tokenizer Saved Successfully!")

print("TRAINING PIPELINE COMPLETED")