"""
Converts the LIAR dataset (short PolitiFact statements) into the same
binary real/fake format used by Fake.csv / True.csv, so it can be
merged into training.

Run this ONCE, locally, after downloading and extracting
liar_dataset.zip into model_training/liar_dataset/
(https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

Usage (from project root):
    python model_training/build_liar_binary.py
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIAR_DIR = os.path.join(BASE_DIR, "liar_dataset")

COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job",
    "state", "party", "barely_true_c", "false_c", "half_true_c",
    "mostly_true_c", "pants_on_fire_c", "venue"
]

# LIAR's 6-way truthfulness -> binary real/fake
# (This mapping is a judgment call - "half-true" and "barely-true"
#  sit in a gray zone. Mapping half-true -> real and barely-true -> fake
#  is a common choice in prior work using this dataset.)
LABEL_MAP = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0,
}


def load_split(filename):
    path = os.path.join(LIAR_DIR, filename)
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)
    return df


def main():
    train = load_split("train.tsv")
    valid = load_split("valid.tsv")
    test = load_split("test.tsv")

    combined = pd.concat([train, valid, test], ignore_index=True)

    combined["label_bin"] = combined["label"].map(LABEL_MAP)

    # Drop anything that didn't map (shouldn't happen, but just in case)
    before = len(combined)
    combined = combined.dropna(subset=["label_bin"])
    dropped = before - len(combined)
    if dropped:
        print(f"Dropped {dropped} rows with unmapped labels.")

    combined["label_bin"] = combined["label_bin"].astype(int)

    # Match the column shape your pipeline expects: title + text + label
    # LIAR statements have no separate title, so title is left empty -
    # train.py already does title.fillna("") + " " + text, so this is safe.
    out = pd.DataFrame({
        "title": "",
        "text": combined["statement"],
        "label": combined["label_bin"]
    })

    print("Label distribution:")
    print(out["label"].value_counts())
    print(f"\nTotal rows: {len(out)}")
    print(f"Avg statement length (words): {out['text'].str.split().str.len().mean():.1f}")

    out_path = os.path.join(BASE_DIR, "liar_binary.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
