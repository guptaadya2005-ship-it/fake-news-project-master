"""Application-facing adapter for the full fake-news detection pipeline."""


def _label_text(label):
    return "Real News" if label == 1 else "Fake News"


def _public_evidence(item):
    """Return only fields the web client can safely display."""
    if not item:
        return None

    return {
        "title": item.get("title", ""),
        "snippet": item.get("snippet", ""),
        "url": item.get("url", ""),
    }


def predict_with_evidence(text):
    """
    Run the style classifier and evidence verifier, returning a stable
    presentation-layer result for the Flask API.

    The import is deliberately lazy: starting the web server should not load
    the transformer models until somebody submits their first article.
    """
    from deep_learning.fusion import fused_prediction

    result = fused_prediction(text)
    style = result["style_result"]
    evidence = result["evidence_result"]

    return {
        "prediction": _label_text(result["final_label"]),
        "confidence": result["final_confidence"] * 100,
        "source": result["source"],
        "style_analysis": {
            "prediction": _label_text(style["label"]),
            "confidence": style["confidence"] * 100,
        },
        "evidence": {
            "verdict": evidence["verdict"],
            "confidence": evidence["confidence"] * 100,
            "method": evidence["method"],
            "best_source": _public_evidence(evidence.get("best_evidence")),
        },
    }
