############################# Takes a claim, runs a web search, pulls back top snippets ################################

########################### Evidence retrieval ##########################
#
# Given a claim, search the web and return a handful of short text
# snippets that might support or contradict it. This is step 2 of the
# retrieval-augmented fact-check pipeline: it doesn't judge the claim
# itself, it just gathers evidence for evidence_verifier.py to compare
# against.
#
# Uses ddgs (no API key required) - this is the renamed/maintained
# successor to the old duckduckgo-search package:
#   pip install ddgs

from ddgs import DDGS
import time
import requests
from bs4 import BeautifulSoup

# How many times to retry a failed search before giving up on this claim
MAX_RETRIES = 3

# Seconds to wait between retries (gives a flaky backend a moment to recover)
RETRY_DELAY = 2

# Fact-check domains worth searching directly and fetching in full -
# their short search snippets are usually just the claim restated, but
# their full articles contain the actual verdict.
FACT_CHECK_DOMAINS = [
    "politifact.com",
    "snopes.com",
    "factcheck.org",
]


def get_evidence(claim, max_results=5):
    """
    Search the web for a claim and return a list of evidence snippets.
    Retries a few times on failure, since free search backends
    (Bing/Yahoo/etc, rotated automatically by ddgs) occasionally time
    out or rate-limit - a single failure doesn't mean no evidence exists.

    Also runs targeted "site:domain claim" searches against known
    fact-check sites and puts those results FIRST in the returned list,
    since they're the most reliable sources for evidence_verifier.py's
    rating lookup.

    Args:
        claim (str): the statement to check
        max_results (int): how many general search results to pull

    Returns:
        list[dict]: each dict has "title", "snippet", "url"
                     Empty list if every retry fails or nothing is found.
    """

    fact_check_evidence = []
    for domain in FACT_CHECK_DOMAINS:
        fact_check_evidence.extend(
            _search(f"site:{domain} {claim}", max_results=2)
        )

    general_evidence = _search(claim, max_results=max_results)

    # Fact-check results first, then general results, de-duplicated by URL
    seen_urls = set()
    combined = []
    for item in fact_check_evidence + general_evidence:
        if item["url"] not in seen_urls:
            combined.append(item)
            seen_urls.add(item["url"])

    return combined


def _search(query, max_results):
    """Run one search query with retries. Returns a list of evidence dicts
    (possibly empty)."""

    evidence = []
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)

                for r in results:
                    evidence.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", "")
                    })

            return evidence

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(
                    f"Search attempt {attempt}/{MAX_RETRIES} failed for "
                    f"query: '{query[:60]}...' - retrying..."
                )
                time.sleep(RETRY_DELAY)

    print(
        f"Search failed after {MAX_RETRIES} attempts for "
        f"query: '{query[:60]}...' Error: {last_error}"
    )
    return evidence


def fetch_full_text(url, max_chars=4000, timeout=10):
    """
    Fetch and extract the readable text of a full article page.

    Used when a short search snippet isn't enough (e.g. fact-check
    articles that open by restating the claim before giving the verdict
    later in the article body).

    Args:
        url (str): page to fetch
        max_chars (int): truncate returned text to this length
        timeout (int): request timeout in seconds

    Returns:
        str: extracted plain text, or "" if the fetch/parse fails
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (fake-news-project research bot)"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]

    except Exception as e:
        print(f"Full-text fetch failed for {url}: {e}")
        return ""


if __name__ == "__main__":
    # Quick manual test
    test_claim = (
        "Los Angeles mayoral candidate Spencer Pratt got zero out of "
        "24,000 votes in a late-night ballot drop."
    )

    print(f"Claim: {test_claim}\n")

    results = get_evidence(test_claim)

    if not results:
        print("No evidence found.")
    else:
        for i, item in enumerate(results, start=1):
            print(f"[{i}] {item['title']}")
            print(f"    {item['snippet'][:200]}")
            print(f"    {item['url']}\n")