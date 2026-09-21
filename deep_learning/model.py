########################### DistilBERT model ##########################


from transformers import DistilBertForSequenceClassification
from config import MODEL_NAME

def load_model():

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    return model