########################### Prediction ##########################

import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

from config import MODEL_SAVE_PATH, MAX_LENGTH

# DEVICE

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# LOAD TOKENIZER

print("\nLoading Tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(
    MODEL_SAVE_PATH
)

print("Tokenizer Loaded Successfully!")

# LOAD TRAINED MODEL

print("\nLoading Trained DistilBERT Model...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)

model.to(device)

model.eval()

print("Trained Model Loaded Successfully!")

# PREDICTION FUNCTION

def predict_news(news):

    encoding = tokenizer(
        news,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)

    attention_mask = encoding["attention_mask"].to(device)

    # MODEL PREDICTION

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    prediction_value = prediction.item()

    confidence_value = confidence.item()

    # LABEL

    if prediction_value == 1:

        label = "REAL NEWS"

    else:

        label = "FAKE NEWS"

    # RETURN RESULT

    return (
        label,
        confidence_value * 100
    )

# TERMINAL TESTING

if __name__ == "__main__":
  
    print("\n====================================")
    print("   FAKE NEWS DETECTION SYSTEM")
    print("====================================")

    while True:

        news = input(
            "\nEnter a news article:\n"
        )

        if news.lower().strip() == "exit":

            print("\nGoodbye!")

            break

        if not news.strip():

            print(
                "\nPlease enter some news text."
            )

            continue

        prediction, confidence = predict_news(
            news
        )

        print(
            "\nPrediction :",
            prediction
        )

        print(
            f"Confidence : "
            f"{confidence:.2f}%"
        )