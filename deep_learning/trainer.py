########################### Training loop ##########################

import os
import torch
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    EPOCHS,
    PATIENCE
)


def validate_model(model, val_loader, device):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            val_loader,
            desc="Validation",
            leave=True
        )

        for batch in progress_bar:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    average_loss = total_loss / len(val_loader)

    return average_loss


def train_model(
    model,
    train_loader,
    val_loader,
    device
):

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    patience_counter = 0

    best_model_saved = False

    last_train_loss = 0.0

    # EPOCH LOOP

    for epoch in range(EPOCHS):

        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 50}")

        # TRAINING

        model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=True
        )


        for batch in progress_bar:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # TRAINING LOSS

        average_train_loss = (
            total_loss / len(train_loader)
        )

        last_train_loss = average_train_loss

        print(
            f"\nTraining Loss: "
            f"{average_train_loss:.4f}"
        )

        # VALIDATION

        print("\nRunning Validation...")

        validation_loss = validate_model(
            model,
            val_loader,
            device
        )

        print(
            f"\nValidation Loss: "
            f"{validation_loss:.4f}"
        )

        # BEST MODEL CHECK

        if validation_loss < best_val_loss:

            best_val_loss = validation_loss

            patience_counter = 0

            os.makedirs(
                MODEL_SAVE_PATH,
                exist_ok=True
            )

            model.save_pretrained(
                MODEL_SAVE_PATH
            )

            best_model_saved = True

            print(
                "\n✓ Validation loss improved!"
            )

            print(
                "✓ Best model saved!"
            )

        else:

            patience_counter += 1

            print(
                "\n⚠ Validation loss did not improve."
            )

            print(
                f"Patience: "
                f"{patience_counter}/{PATIENCE}"
            )

    # EARLY STOPPING

        if patience_counter >= PATIENCE:

            print(
                "\n🛑 Early stopping triggered."
            )

            break

 # TRAINING SUMMARY

    print(
        "\n=========================================="
    )

    if best_model_saved:

        print(
            "✓ Best model saved successfully."
        )

    print(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        "=========================================="
    )

    return last_train_loss