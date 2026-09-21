import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import MODEL_SAVE_PATH, MAX_LENGTH
from model import load_model
from config import *

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "real_world_data",
    "external_test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_model"
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nLoading Real-World Test Dataset...")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!\n")

print(f"Total Samples: {len(df)}")
print(f"Fake News: {(df['label'] == 0).sum()}")
print(f"Real News: {(df['label'] == 1).sum()}")

print(f"\nUsing Device: {device}")

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading Tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(
    MODEL_SAVE_PATH
)

print("Tokenizer Loaded Successfully!")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Trained DistilBERT Model...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH,
    num_labels=2
)

model.to(device)
model.eval()

print("Model Loaded Successfully!")

# ============================================================
# EVALUATION
# ============================================================

print("\nStarting Real-World Evaluation...\n")

predictions = []
actual_labels = []
confidences = []

correct_predictions = 0
incorrect_predictions = 0

for index, row in df.iterrows():

    text = str(row["text"])
    actual_label = int(row["label"])

    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Get logits from model
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get predicted class and confidence
    confidence, prediction = torch.max(probabilities, dim=1)

    predicted_label = prediction.item()
    confidence_value = confidence.item() * 100

    # Convert label to text
    predicted_text = "FAKE" if predicted_label == 0 else "REAL"
    actual_text = "FAKE" if actual_label == 0 else "REAL"

    # Check correctness
    correct = predicted_label == actual_label

    if correct:
        correct_predictions += 1
        status = "✓ CORRECT"
    else:
        incorrect_predictions += 1
        status = "✗ WRONG"

    # Store results for metrics computed after the loop
    predictions.append(predicted_label)
    actual_labels.append(actual_label)
    confidences.append(confidence_value)

    print(
        f"[{index + 1:02d}/{len(df)}] "
        f"Actual: {actual_text} | "
        f"Predicted: {predicted_text} | "
        f"Confidence: {confidence_value:6.2f}% | "
        f"{status}"
    )

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predictions
)

precision = precision_score(
    actual_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predictions,
    zero_division=0
)

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("REAL-WORLD EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_labels,
    predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix")
print()
print("              Predicted")
print("              Fake   Real")
print(
    f"Actual Fake    {cm[0][0]:2d}     {cm[0][1]:2d}"
)
print(
    f"Actual Real    {cm[1][0]:2d}     {cm[1][1]:2d}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report\n")

print(
    classification_report(
        actual_labels,
        predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

# ============================================================
# ERROR ANALYSIS
# ============================================================

false_positives = sum(
    (actual_labels[i] == 1 and predictions[i] == 0)
    for i in range(len(actual_labels))
)

false_negatives = sum(
    (actual_labels[i] == 0 and predictions[i] == 1)
    for i in range(len(actual_labels))
)

true_positives = sum(
    (actual_labels[i] == 1 and predictions[i] == 1)
    for i in range(len(actual_labels))
)

true_negatives = sum(
    (actual_labels[i] == 0 and predictions[i] == 0)
    for i in range(len(actual_labels))
)

print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

print(
    f"\nFalse Positives (REAL predicted as FAKE): "
    f"{false_positives}"
)

print(
    f"False Negatives (FAKE predicted as REAL): "
    f"{false_negatives}"
)

print(
    f"True Positives (REAL predicted as REAL): "
    f"{true_positives}"
)

print(
    f"True Negatives (FAKE predicted as FAKE): "
    f"{true_negatives}"
)

# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

correct_confidences = [
    confidences[i]
    for i in range(len(actual_labels))
    if actual_labels[i] == predictions[i]
]

wrong_confidences = [
    confidences[i]
    for i in range(len(actual_labels))
    if actual_labels[i] != predictions[i]
]

print("\n")
print("=" * 60)
print("CONFIDENCE ANALYSIS")
print("=" * 60)

print(
    f"\nAverage Confidence: "
    f"{sum(confidences) / len(confidences):.2f}%"
)

print(
    f"Minimum Confidence: "
    f"{min(confidences):.2f}%"
)

print(
    f"Maximum Confidence: "
    f"{max(confidences):.2f}%"
)

if correct_confidences:
    print(
        f"\nAverage Confidence (Correct): "
        f"{sum(correct_confidences) / len(correct_confidences):.2f}%"
    )

if wrong_confidences:
    print(
        f"Average Confidence (Wrong): "
        f"{sum(wrong_confidences) / len(wrong_confidences):.2f}%"
    )

# ============================================================
# PERFORMANCE BY SOURCE
# ============================================================

if "source_type" in df.columns:

    print("\n")
    print("=" * 60)
    print("PERFORMANCE BY SOURCE")
    print("=" * 60)

    for source in df["source_type"].dropna().unique():

        source_indices = df.index[
            df["source_type"] == source
        ].tolist()

        source_correct = sum(
            actual_labels[i] == predictions[i]
            for i in source_indices
        )

        source_accuracy = (
            source_correct / len(source_indices)
        )

        print(f"\n{source}")
        print(f"Samples  : {len(source_indices)}")
        print(f"Accuracy : {source_accuracy:.2%}")

# ============================================================
# PERFORMANCE BY CATEGORY
# ============================================================

if "category" in df.columns:

    print("\n")
    print("=" * 60)
    print("PERFORMANCE BY CATEGORY")
    print("=" * 60)

    for category in df["category"].dropna().unique():

        category_indices = df.index[
            df["category"] == category
        ].tolist()

        category_correct = sum(
            actual_labels[i] == predictions[i]
            for i in category_indices
        )

        category_accuracy = (
            category_correct / len(category_indices)
        )

        print(f"\n{category}")
        print(f"Samples  : {len(category_indices)}")
        print(f"Accuracy : {category_accuracy:.2%}")

# ============================================================
# MOST CONFIDENT WRONG PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("MOST CONFIDENT WRONG PREDICTIONS")
print("=" * 60)

wrong_indices = [
    i
    for i in range(len(actual_labels))
    if actual_labels[i] != predictions[i]
]

wrong_indices.sort(
    key=lambda i: confidences[i],
    reverse=True
)

for number, i in enumerate(wrong_indices[:10], start=1):

    actual_name = (
        "FAKE"
        if actual_labels[i] == 0
        else "REAL"
    )

    predicted_name = (
        "FAKE"
        if predictions[i] == 0
        else "REAL"
    )

    article = str(df.iloc[i]["text"])

    print(f"\nError #{number}")
    print(f"Actual    : {actual_name}")
    print(f"Predicted : {predicted_name}")
    print(f"Confidence: {confidences[i]:.2f}%")
    print(f"Article   : {article[:300]}...")

# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

results_df = df.copy()

results_df["prediction"] = predictions
results_df["confidence"] = confidences

results_path = os.path.join(
    BASE_DIR,
    "real_world_data",
    "evaluation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n")
print("=" * 60)
print("Evaluation results saved successfully!")
print(f"Saved to: {results_path}")

# ============================================================
# SAVE REAL NEWS ERRORS
# ============================================================

error_df = results_df[
    (results_df["label"] == 1) &
    (results_df["prediction"] == 0)
].copy()

error_path = os.path.join(
    BASE_DIR,
    "real_world_data",
    "real_news_errors.csv"
)

error_df.to_csv(
    error_path,
    index=False
)

print(
    f"Real-news error analysis saved successfully!"
)
print(f"Saved to: {error_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nTotal Articles       : {len(df)}")
print(
    f"Correct Predictions : "
    f"{sum(actual_labels[i] == predictions[i] for i in range(len(df)))}/{len(df)}"
)
print(
    f"Incorrect Predictions : "
    f"{sum(actual_labels[i] != predictions[i] for i in range(len(df)))}/{len(df)}"
)

print(f"\nFinal Accuracy       : {accuracy * 100:.2f}%")
print(f"False Positives      : {false_positives}")
print(f"False Negatives      : {false_negatives}")

print("\nREAL-WORLD EVALUATION COMPLETED!")
