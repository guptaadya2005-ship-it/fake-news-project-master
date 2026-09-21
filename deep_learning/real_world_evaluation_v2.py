## New evaluation script that runs claims through the full pipeline (style classifier + evidence check + fusion), instead of just the style classifier alone ##

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from fusion import fused_prediction

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "real_world_data",
    "external_test.csv"
)

print("\nLoading Real-World Test Dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!\n")

print(f"Total Samples: {len(df)}")
print(f"Fake News: {(df['label'] == 0).sum()}")
print(f"Real News: {(df['label'] == 1).sum()}")

# ============================================================
# EVALUATION - runs each claim through the full pipeline
# ============================================================

print("\nStarting Full-Pipeline Evaluation (style classifier + evidence check)...")
print("This calls a web search and an NLI model per claim, so it will be")
print("slower than the style-only script - expect several minutes, not hours.\n")

fused_predictions = []
style_only_predictions = []
decision_sources = []
actual_labels = []
fused_confidences = []

for index, row in df.iterrows():

    text = str(row["text"])
    actual_label = int(row["label"])

    result = fused_prediction(text)

    fused_predictions.append(result["final_label"])
    style_only_predictions.append(result["style_result"]["label"])
    decision_sources.append(result["source"])
    actual_labels.append(actual_label)
    fused_confidences.append(result["final_confidence"])

    actual_text = "FAKE" if actual_label == 0 else "REAL"
    predicted_text = "FAKE" if result["final_label"] == 0 else "REAL"
    correct = "CORRECT" if result["final_label"] == actual_label else "WRONG"
    status = "\u2713" if correct == "CORRECT" else "\u2717"

    print(
        f"[{index+1:02d}/{len(df)}] "
        f"Actual: {actual_text} | "
        f"Predicted: {predicted_text} | "
        f"Decided by: {result['source']:<16} | "
        f"Confidence: {result['final_confidence']*100:6.2f}% | "
        f"{status} {correct}"
    )

# ============================================================
# METRICS - fused pipeline
# ============================================================

fused_accuracy = accuracy_score(actual_labels, fused_predictions)
fused_precision = precision_score(actual_labels, fused_predictions, zero_division=0)
fused_recall = recall_score(actual_labels, fused_predictions, zero_division=0)
fused_f1 = f1_score(actual_labels, fused_predictions, zero_division=0)

# ============================================================
# METRICS - style classifier alone (baseline, for comparison)
# ============================================================

style_accuracy = accuracy_score(actual_labels, style_only_predictions)
style_precision = precision_score(actual_labels, style_only_predictions, zero_division=0)
style_recall = recall_score(actual_labels, style_only_predictions, zero_division=0)
style_f1 = f1_score(actual_labels, style_only_predictions, zero_division=0)

print("\n")
print("=" * 60)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 60)

print(f"\n{'Metric':<12} {'Style only':>12} {'Full pipeline':>15} {'Change':>10}")
print(
    f"{'Accuracy':<12} {style_accuracy:>12.4f} {fused_accuracy:>15.4f} "
    f"{fused_accuracy - style_accuracy:>+10.4f}"
)
print(
    f"{'Precision':<12} {style_precision:>12.4f} {fused_precision:>15.4f} "
    f"{fused_precision - style_precision:>+10.4f}"
)
print(
    f"{'Recall':<12} {style_recall:>12.4f} {fused_recall:>15.4f} "
    f"{fused_recall - style_recall:>+10.4f}"
)
print(
    f"{'F1 Score':<12} {style_f1:>12.4f} {fused_f1:>15.4f} "
    f"{fused_f1 - style_f1:>+10.4f}"
)

# How often did evidence override the style classifier?
evidence_decided = sum(1 for s in decision_sources if s == "evidence")
style_decided = sum(1 for s in decision_sources if s == "style_classifier")

print(f"\nDecisions made by evidence check : {evidence_decided}/{len(df)}")
print(f"Decisions made by style classifier: {style_decided}/{len(df)}")

# ============================================================
# FULL PIPELINE RESULTS (same format as original script)
# ============================================================

print("\n")
print("=" * 60)
print("FULL PIPELINE - DETAILED RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {fused_accuracy:.4f} ({fused_accuracy * 100:.2f}%)")
print(f"Precision : {fused_precision:.4f}")
print(f"Recall    : {fused_recall:.4f}")
print(f"F1 Score  : {fused_f1:.4f}")

cm = confusion_matrix(actual_labels, fused_predictions, labels=[0, 1])

print("\nConfusion Matrix")
print()
print("              Predicted")
print("              Fake   Real")
print(f"Actual Fake    {cm[0][0]:2d}     {cm[0][1]:2d}")
print(f"Actual Real    {cm[1][0]:2d}     {cm[1][1]:2d}")

print("\nClassification Report\n")
print(
    classification_report(
        actual_labels,
        fused_predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

# ============================================================
# WHERE EVIDENCE FIXED A STYLE-CLASSIFIER MISTAKE
# ============================================================

print("=" * 60)
print("CASES WHERE EVIDENCE OVERRODE A WRONG STYLE PREDICTION")
print("=" * 60)

fixed_count = 0

for i in range(len(df)):
    style_was_wrong = style_only_predictions[i] != actual_labels[i]
    fused_is_right = fused_predictions[i] == actual_labels[i]
    decided_by_evidence = decision_sources[i] == "evidence"

    if style_was_wrong and fused_is_right and decided_by_evidence:
        fixed_count += 1
        article = str(df.iloc[i]["text"])
        print(f"\n[{i+1}] {article[:150]}...")
        print(
            f"    Style classifier said: "
            f"{'REAL' if style_only_predictions[i] == 1 else 'FAKE'} (WRONG)"
        )
        print(
            f"    Evidence check said   : "
            f"{'REAL' if fused_predictions[i] == 1 else 'FAKE'} (CORRECT)"
        )

print(f"\nTotal cases fixed by evidence check: {fixed_count}")

# ============================================================
# CASES WHERE EVIDENCE MADE THINGS WORSE
# ============================================================

print("\n")
print("=" * 60)
print("CASES WHERE EVIDENCE OVERRODE A CORRECT STYLE PREDICTION")
print("=" * 60)

broken_count = 0

for i in range(len(df)):
    style_was_right = style_only_predictions[i] == actual_labels[i]
    fused_is_wrong = fused_predictions[i] != actual_labels[i]
    decided_by_evidence = decision_sources[i] == "evidence"

    if style_was_right and fused_is_wrong and decided_by_evidence:
        broken_count += 1
        article = str(df.iloc[i]["text"])
        print(f"\n[{i+1}] {article[:150]}...")
        print(
            f"    Style classifier said: "
            f"{'REAL' if style_only_predictions[i] == 1 else 'FAKE'} (CORRECT)"
        )
        print(
            f"    Evidence check said   : "
            f"{'REAL' if fused_predictions[i] == 1 else 'FAKE'} (WRONG)"
        )

print(f"\nTotal cases broken by evidence check: {broken_count}")

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = df.copy()
results_df["style_prediction"] = style_only_predictions
results_df["fused_prediction"] = fused_predictions
results_df["decided_by"] = decision_sources
results_df["confidence"] = fused_confidences

results_path = os.path.join(
    BASE_DIR,
    "real_world_data",
    "evaluation_results_v2.csv"
)

results_df.to_csv(results_path, index=False)

print("\n")
print("=" * 60)
print("Evaluation results saved successfully!")
print(f"Saved to: {results_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nTotal Articles      : {len(df)}")
print(f"Style-only Accuracy : {style_accuracy * 100:.2f}%")
print(f"Full Pipeline Accuracy: {fused_accuracy * 100:.2f}%")
print(f"Net Improvement     : {(fused_accuracy - style_accuracy) * 100:+.2f} points")
print(f"Fixed by evidence   : {fixed_count}")
print(f"Broken by evidence  : {broken_count}")

print("\nFULL PIPELINE EVALUATION COMPLETED!")