########################### AI expert ##########################


import torch
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification

MODEL_PATH = "bert_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading DistilBERT...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)

model.eval()

print("DistilBERT Loaded Successfully!")

# This function takes text as input and returns both the predicted label and its confidence.

def predict_news(text):      

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    prediction = prediction.item()
    confidence = confidence.item() * 100

    if prediction == 1:
        label = "Real News"
    else:
        label = "Fake News"

    return label, confidence