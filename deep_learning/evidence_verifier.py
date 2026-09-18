########################### Evidence verifier ##########################
#
# Step 3 of the retrieval-augmented fact-check pipeline.
#
# Checks a claim two ways, checked in order:
#
#   A. Fact-check rating lookup (checked FIRST)
#      Known fact-check sites (PolitiFact, Snopes, etc.) put an explicit
#      verdict in the article title/URL ("...rated False", "pants-fire").
#      If we find one, trust it directly - it's far more reliable than
#      NLI here, because fact-check articles open by RESTATING the claim
#      almost verbatim before debunking it, which fools generic NLI into
#      reading the restatement as agreement (near-identical wording =
#      high entailment) when it's actually about to be refuted.
#
#   B. NLI entailment/contradiction (fallback)
#      For regular news coverage (BBC, Reuters, etc.) with no explicit
#      verdict, fall back to the pretrained NLI model: does the article
#      support or contradict the claim?
#
# Uses roberta-large-mnli, a standard pretrained NLI model - no training
# required, just downloaded and used as-is.
#
# Install:
#   pip install transformers torch

import re
from transformers import pipeline

NLI_MODEL_NAME = "roberta-large-mnli"

# How confident the strongest entailment/contradiction score needs to be
# before we trust the evidence over the style classifier. Below this,
# treat the evidence as inconclusive.
CONFIDENCE_THRESHOLD = 0.60

# Domains known to publish explicit claim ratings (title/URL usually
# contains a rating word). Add more as you find them in your data.
FACT_CHECK_DOMAINS = [
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "apnews.com/hub/ap-fact-check",
    "fullfact.org",
    "reuters.com/fact-check",
    "poynter.org/fact-checking",
]

# Rating PHRASES (not bare words) -> verdict. Bare words like "false" or
# "true" match all over ordinary prose and caused false positives on
# unrelated articles - require an actual rating-style phrase instead.
# Checked longest/most-specific first.
RATING_PATTERNS = [
    (r"\bpants[\s-]on[\s-]fire\b", "FAKE"),
    (r"\brated?\s+(?:as\s+)?mostly[\s-]false\b", "FAKE"),
    (r"\brated?\s+(?:as\s+)?false\b", "FAKE"),
    (r"\bfact[\s-]check:?\s*false\b", "FAKE"),
    (r"\bdebunked\b", "FAKE"),
    (r"\bis a hoax\b", "FAKE"),
    (r"\bdid not\b.{0,20}\breceive\b", "FAKE"),  # narrow, claim-specific pattern
    (r"\brated?\s+(?:as\s+)?mostly[\s-]true\b", "REAL"),
    (r"\brated?\s+(?:as\s+)?true\b", "REAL"),
    (r"\bfact[\s-]check:?\s*true\b", "REAL"),
]

# Minimum fraction of the claim's significant words that must also
# appear in the article title before we trust its rating - stops an
# unrelated fact-check article (found by a noisy site-search) from
# being treated as evidence for a claim it never discusses.
RELEVANCE_THRESHOLD = 0.4

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "of",
    "in", "on", "at", "to", "for", "and", "or", "but", "with", "by",
    "as", "that", "this", "it", "from", "has", "have", "had", "will",
    "would", "could", "should", "did", "does", "do", "not", "no"
}

_nli_pipeline = None


def _significant_words(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _is_relevant(claim, item):
    """Check the article's title actually overlaps with the claim's
    key words, so we don't trust a rating from an unrelated article."""
    claim_words = _significant_words(claim)
    title_words = _significant_words(item.get("title", ""))

    if not claim_words:
        return False

    overlap = len(claim_words & title_words) / len(claim_words)
    return overlap >= RELEVANCE_THRESHOLD


def _check_fact_check_rating(claim, item):
    """
    Look for an explicit rating in a known fact-check source, but ONLY
    if the article is actually about this claim (see _is_relevant) -
    otherwise a noisy site-search result for an unrelated article could
    be mistaken for evidence.

    Returns "REAL", "FAKE", or None.
    """
    url = item.get("url", "").lower()

    is_fact_check_source = any(domain in url for domain in FACT_CHECK_DOMAINS)
    if not is_fact_check_source:
        return None

    if not _is_relevant(claim, item):
        return None

    text_to_check = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    rating = _find_rating_keyword(text_to_check)
    if rating is not None:
        return rating

    try:
        from .evidence_retrieval import fetch_full_text
    except ImportError:
        from evidence_retrieval import fetch_full_text
    full_text = fetch_full_text(item.get("url", ""))
    if full_text:
        return _find_rating_keyword(full_text.lower())

    return None


def _find_rating_keyword(text):
    """Search text for a rating PHRASE pattern, most-specific first."""
    for pattern, verdict in RATING_PATTERNS:
        if re.search(pattern, text):
            return verdict
    return None


def _get_pipeline():
    """Lazy-load the NLI model so importing this file doesn't trigger
    a slow model download until it's actually used."""
    global _nli_pipeline
    if _nli_pipeline is None:
        print(f"Loading NLI model ({NLI_MODEL_NAME})... this may take a moment.")
        _nli_pipeline = pipeline(
            "text-classification",
            model=NLI_MODEL_NAME,
            top_k=None
        )
    return _nli_pipeline


def _score_pair(premise, hypothesis):
    """
    Run one (evidence snippet, claim) pair through the NLI model.
    Truncates automatically - premise can sometimes be full article text
    (thousands of characters) which would otherwise exceed the model's
    512-token limit and crash.

    Returns a dict like:
        {"CONTRADICTION": 0.05, "NEUTRAL": 0.10, "ENTAILMENT": 0.85}
    """
    nli = _get_pipeline()
    results = nli(
        premise,
        text_pair=hypothesis,
        truncation=True,
        max_length=512
    )[0]
    return {r["label"]: r["score"] for r in results}


def verify_claim(claim, evidence_list):
    """
    Check a claim against a list of evidence snippets.

    Args:
        claim (str): the statement being checked
        evidence_list (list[dict]): output of evidence_retrieval.get_evidence()
                                     each item has "title", "snippet", "url"

    Returns:
        dict with:
            verdict (str): "REAL", "FAKE", or "UNCERTAIN"
            confidence (float): 0-1
            best_evidence (dict or None): the snippet that drove the verdict
            method (str): "fact_check_rating" or "nli" or "none"
            all_scores (list[dict]): per-snippet NLI scores, for debugging
    """

    if not evidence_list:
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "best_evidence": None,
            "method": "none",
            "all_scores": []
        }

    # --- Stage A: explicit fact-check rating lookup ---
    for item in evidence_list:
        rating = _check_fact_check_rating(claim, item)
        if rating is not None:
            return {
                "verdict": rating,
                "confidence": 0.95,  # high, fixed confidence - explicit rating
                "best_evidence": item,
                "method": "fact_check_rating",
                "all_scores": []
            }

    # --- Stage B: NLI fallback ---
    # Only consider evidence that's actually about this claim - an
    # irrelevant search result (found by a noisy query) shouldn't be
    # able to sway the verdict just because NLI finds some surface-level
    # semantic overlap with unrelated text.
    relevant_evidence = [item for item in evidence_list if _is_relevant(claim, item)]

    best_entailment = {"score": 0.0, "evidence": None}
    best_contradiction = {"score": 0.0, "evidence": None}
    all_scores = []

    for item in relevant_evidence:
        premise = f"{item['title']}. {item['snippet']}".strip()

        if not premise:
            continue

        scores = _score_pair(premise, claim)
        all_scores.append({"evidence": item, "scores": scores})

        if scores.get("ENTAILMENT", 0.0) > best_entailment["score"]:
            best_entailment = {"score": scores["ENTAILMENT"], "evidence": item}

        if scores.get("CONTRADICTION", 0.0) > best_contradiction["score"]:
            best_contradiction = {"score": scores["CONTRADICTION"], "evidence": item}

    if (best_entailment["score"] >= CONFIDENCE_THRESHOLD
            and best_entailment["score"] > best_contradiction["score"]):
        return {
            "verdict": "REAL",
            "confidence": best_entailment["score"],
            "best_evidence": best_entailment["evidence"],
            "method": "nli",
            "all_scores": all_scores
        }

    if (best_contradiction["score"] >= CONFIDENCE_THRESHOLD
            and best_contradiction["score"] > best_entailment["score"]):
        return {
            "verdict": "FAKE",
            "confidence": best_contradiction["score"],
            "best_evidence": best_contradiction["evidence"],
            "method": "nli",
            "all_scores": all_scores
        }

    return {
        "verdict": "UNCERTAIN",
        "confidence": max(best_entailment["score"], best_contradiction["score"]),
        "best_evidence": None,
        "method": "nli",
        "all_scores": all_scores
    }


if __name__ == "__main__":
    # Quick manual test - pairs with evidence_retrieval.py
    from evidence_retrieval import get_evidence

    test_claim = (
        "U.S. Sen. Jon Ossoff cast the deciding vote for the inflation "
        "disaster, referring to the 2022 Inflation Reduction Act."
    )

    print(f"Claim: {test_claim}\n")

    evidence = get_evidence(test_claim)
    result = verify_claim(test_claim, evidence)

    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Method: {result['method']}")

    if result["best_evidence"]:
        print(f"Best evidence: {result['best_evidence']['title']}")
        print(f"  {result['best_evidence']['snippet'][:200]}")
        print(f"  {result['best_evidence']['url']}")
