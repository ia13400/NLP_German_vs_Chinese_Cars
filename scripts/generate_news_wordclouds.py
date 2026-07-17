from __future__ import annotations

from car_interest_nlp.nlp.corpus import load_gdelt_corpus
from car_interest_nlp.nlp.wordclouds import generate_all_word_clouds


def main() -> int:
    """Generate all 7 required GDELT news word cloud categories from the real article corpus.

    German-brand and Chinese-brand word clouds (frequency + TF-IDF, 4 total), one word
    cloud per year present in the corpus, one technology-related word cloud, and one
    regulation/tariff word cloud. Runs over whatever article/scraped-text coverage exists
    so far -- with real, partial GDELT coverage it is entirely normal for some categories
    to have no matching articles yet even when others do, so `generate_all_word_clouds`
    generates each category independently and skips (logs, does not fail) any with no
    input text rather than aborting the whole run.
    """
    corpus = load_gdelt_corpus()
    if corpus.empty:
        print(
            "No GDELT article corpus available yet -- run scripts/download_gdelt_news.py "
            "(and optionally scripts/fetch_article_texts.py) first."
        )
        return 0

    generated = generate_all_word_clouds(corpus, "artifacts/figures/gdelt")
    print(f"Generated word clouds: {sorted(generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
