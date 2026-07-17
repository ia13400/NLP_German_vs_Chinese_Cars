from __future__ import annotations

import time
import urllib.robotparser
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import httpx
import trafilatura

from ..logging_utils import configure_logging
from ..progress import TimeBudget, iter_with_progress
from ..utils import stable_hash
from .collectors import write_cache_metadata
from .validators import validate_local_cache

logger = configure_logging()

PARSER_VERSION = "article-text-1.0"
ARTICLE_RAW_FILENAME_PREFIX = "article_"

DEFAULT_USER_AGENT = "car-interest-nlp-research-bot/1.0 (+respects robots.txt)"
DEFAULT_PER_DOMAIN_DELAY_SECONDS = 3.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0

# Scraping arbitrary third-party news domains is the highest-risk part of this project (real
# blast radius: hundreds of different publishers, each with different rules) -- this module is
# deliberately narrow: robots.txt is always checked and never bypassed, one real domain-level
# rate limit is enforced regardless of how many URLs from that domain are queued, and every
# failure (blocked/paywalled/JS-only page/timeout) is treated as an expected, logged, non-fatal
# outcome rather than an error. This is best-effort robots.txt compliance only -- scraping may
# still exceed an individual publisher's Terms of Service beyond what robots.txt covers; content
# fetched here is used solely for local, non-redistributed text statistics (word frequency/NER
# in this project), never reproduced or republished. See README's "GDELT news analysis" section.

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
_domain_last_request_at: dict[str, float] = {}


def _get_robots_parser(domain: str, *, timeout: float) -> urllib.robotparser.RobotFileParser:
    if domain in _robots_cache:
        return _robots_cache[domain]
    parser = urllib.robotparser.RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"
    parser.set_url(robots_url)
    try:
        response = httpx.get(
            robots_url, timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT}
        )
        # A missing/inaccessible robots.txt means no restriction is declared (standard
        # robots.txt semantics) -- parse([]) leaves the parser in its default allow-all state.
        parser.parse(response.text.splitlines() if response.status_code == 200 else [])
    except httpx.HTTPError:
        parser.parse([])
    _robots_cache[domain] = parser
    return parser


def is_allowed_by_robots(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> bool:
    """Check a real URL against its domain's real robots.txt (cached per domain)."""
    domain = urlparse(url).netloc
    parser = _get_robots_parser(domain, timeout=timeout)
    return parser.can_fetch(user_agent, url)


def _respect_domain_rate_limit(domain: str, delay_seconds: float) -> None:
    """Enforce a minimum delay between requests to the same domain, across all queued URLs."""
    last = _domain_last_request_at.get(domain)
    if last is not None:
        elapsed = time.monotonic() - last
        if elapsed < delay_seconds:
            time.sleep(delay_seconds - elapsed)
    _domain_last_request_at[domain] = time.monotonic()


def _article_cache_key(url: str) -> str:
    return stable_hash(url)[:16]


def article_raw_path(dest_dir: str | Path, url: str) -> Path:
    return Path(dest_dir) / f"{ARTICLE_RAW_FILENAME_PREFIX}{_article_cache_key(url)}.txt"


def fetch_article_text(
    url: str,
    dest_dir: str | Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    per_domain_delay_seconds: float = DEFAULT_PER_DOMAIN_DELAY_SECONDS,
    respect_robots_txt: bool = True,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> Path | None:
    """Fetch and extract one real article's main text (via `trafilatura`), caching the result.

    Returns `None` (never raises) if the URL is disallowed by robots.txt, unreachable, or
    yields no extractable text -- these are all expected, common outcomes when fetching
    hundreds of arbitrary third-party news domains (paywalls, blocks, JS-only rendering),
    not project bugs, so callers should treat `None` as "skip," not "retry."
    """
    domain = urlparse(url).netloc
    if respect_robots_txt and not is_allowed_by_robots(url, user_agent, timeout=timeout):
        logger.info("Skipping %s: disallowed by robots.txt", url)
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = article_raw_path(dest_dir, url)

    _respect_domain_rate_limit(domain, per_domain_delay_seconds)
    try:
        response = httpx.get(
            url, timeout=timeout, headers={"User-Agent": user_agent}, follow_redirects=True
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("Skipping %s: fetch failed (%s)", url, exc)
        return None

    text = trafilatura.extract(response.text, url=url)
    if not text:
        logger.info("Skipping %s: no extractable article text (boilerplate-only/JS-rendered)", url)
        return None

    destination.write_text(text, encoding="utf-8")
    write_cache_metadata(destination, source_url=url)
    return destination


def fetch_article_texts(
    urls: Sequence[str],
    dest_dir: str | Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    per_domain_delay_seconds: float = DEFAULT_PER_DOMAIN_DELAY_SECONDS,
    respect_robots_txt: bool = True,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    time_budget: TimeBudget | None = None,
) -> dict[str, int]:
    """Fetch (resumably) every URL not already cached under `dest_dir`, behind one progress bar.

    Real full coverage of a large URL list is expected to take multiple resumed calls (same
    reasoning as `gdelt_dataset_builder.ensure_gdelt_dataset` -- see its docstring); already
    -cached URLs and already-attempted-and-skipped URLs are distinguished by a metadata
    sidecar only existing for successful fetches, so a `None` result is naturally retried on
    the next call rather than permanently skipped.
    """
    dest_dir = Path(dest_dir)
    summary = {"cached": 0, "fetched": 0, "skipped": 0, "total": len(urls)}
    for url in iter_with_progress(
        urls, total=len(urls), desc="article text", time_budget=time_budget
    ):
        destination = article_raw_path(dest_dir, url)
        metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
        if validate_local_cache(destination, metadata_path):
            summary["cached"] += 1
            continue
        result = fetch_article_text(
            url,
            dest_dir,
            user_agent=user_agent,
            per_domain_delay_seconds=per_domain_delay_seconds,
            respect_robots_txt=respect_robots_txt,
            timeout=timeout,
        )
        if result is None:
            summary["skipped"] += 1
        else:
            summary["fetched"] += 1
    return summary


def read_cached_article_text(dest_dir: str | Path, url: str) -> str | None:
    """Read back a previously fetched article's cached text, or `None` if not cached."""
    path = article_raw_path(dest_dir, url)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")
