#### Combines the style classifier's prediction with the evidence verifier's verdict using the fallback rule we discussed ####

########################### Fusion ##########################
#
# Step 4 - combines two independent signals into one final verdict:
#
#   1. Style classifier (your trained DistilBERT model)
#      - good at: writing style, phrasing patterns, tone
#      - bad at:  claims that need real-world fact checking
#
#   2. Evidence verifier (web search + NLI, see evidence_retrieval.py /
#      evidence_verifier.py)
#      - good at:  claims where search results clearly agree or disagree
#      - bad at:   claims with no/weak search coverage (too recent,
#                   too obscure, or genuinely ambiguous)
#
# Fusion rule (simple and explainable, not a black box):
#   - If the evidence verifier reaches a confident REAL or FAKE verdict,
#     trust it - it's checking actual facts, not just style.
#   - Otherwise, fall back to the style classifier's prediction.

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

try:
    # Used when imported by the Flask app as deep_learning.fusion
    from .config import MODEL_SAVE_PATH, MAX_LENGTH
    from .evidence_retrieval import get_evidence
    from .evidence_verifier import verify_claim
except ImportError:
    # Keeps `python fusion.py` working from the deep_learning folder
    from config import MODEL_SAVE_PATH, MAX_LENGTH
    from evidence_retrieval import get_evidence
    from evidence_verifier import verify_claim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = None
_model = None


def _load_style_model():
    """Lazy-load the trained DistilBERT model + tokenizer."""
    global _tokenizer, _model
    if _model is None:
        print("Loading style classifier (DistilBERT)...")
        _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_SAVE_PATH)
        _model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_SAVE_PATH,
            num_labels=2
        )
        _model.to(device)
        _model.eval()
    return _tokenizer, _model


def predict_style(text):
    """
    Run the existing trained model on a single piece of text.

    Returns:
        label (int): 0 = FAKE, 1 = REAL
        confidence (float): 0-1
    """
    tokenizer, model = _load_style_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)

    return prediction.item(), confidence.item()


def fused_prediction(claim, max_evidence=5):
    """
    Run the full pipeline on a single claim: style classifier + evidence
    check, combined with the fallback rule.

    Returns a dict:
        final_label (int): 0 = FAKE, 1 = REAL
        final_confidence (float): 0-1
        source (str): "evidence" or "style_classifier" - which signal decided
        style_result (dict): {"label": int, "confidence": float}
        evidence_result (dict): raw output of evidence_verifier.verify_claim()
    """

    # Signal 1: style classifier
    style_label, style_confidence = predict_style(claim)
    style_result = {"label": style_label, "confidence": style_confidence}

    # Signal 2: evidence check
    evidence = get_evidence(claim, max_results=max_evidence)
    evidence_result = verify_claim(claim, evidence)

    # Fusion rule
    if evidence_result["verdict"] == "REAL":
        return {
            "final_label": 1,
            "final_confidence": evidence_result["confidence"],
            "source": "evidence",
            "style_result": style_result,
            "evidence_result": evidence_result
        }

    if evidence_result["verdict"] == "FAKE":
        return {
            "final_label": 0,
            "final_confidence": evidence_result["confidence"],
            "source": "evidence",
            "style_result": style_result,
            "evidence_result": evidence_result
        }

    # Evidence was UNCERTAIN (no evidence found, or NLI wasn't confident
    # either way) - fall back to the style classifier
    return {
        "final_label": style_label,
        "final_confidence": style_confidence,
        "source": "style_classifier",
        "style_result": style_result,
        "evidence_result": evidence_result
    }


if __name__ == "__main__":
    # Quick manual test on a couple of claims
    test_claims = [
        ("Los Angeles mayoral candidate Spencer Pratt got zero out of "
         "24,000 votes in a late-night ballot drop.", 0),  # actually FAKE
        ("President Donald Trump signed a law increasing the minimum "
         "wage to $25 per hour beginning June 1.", 0),  # actually FAKE
    ]

    for claim, actual_label in test_claims:
        result = fused_prediction(claim)

        final_text = "REAL" if result["final_label"] == 1 else "FAKE"
        actual_text = "REAL" if actual_label == 1 else "FAKE"
        correct = "CORRECT" if result["final_label"] == actual_label else "WRONG"

        print(f"\nClaim: {claim}")
        print(f"Actual    : {actual_text}")
        print(f"Predicted : {final_text}  ({correct})")
        print(f"Decided by: {result['source']}  "
              f"(confidence {result['final_confidence']:.2%})")
        print(f"  Style classifier alone said: "
              f"{'REAL' if result['style_result']['label'] == 1 else 'FAKE'} "
              f"({result['style_result']['confidence']:.2%})")
        print(f"  Evidence verdict: {result['evidence_result']['verdict']}")
