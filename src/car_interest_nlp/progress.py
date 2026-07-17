from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

from .logging_utils import configure_logging

logger = configure_logging()

T = TypeVar("T")


class TimeBudget:
    """Wall-clock deadline tracker for long-running, resumable operations.

    Every long loop in this project that touches a rate-limited real API (GDELT, Google
    Trends) or is inherently slow (scraping hundreds of article URLs, NER training epochs)
    accepts a `TimeBudget` rather than looping unconditionally -- a single invocation is
    expected to make partial progress and be re-run rather than run to completion in one
    sitting. `minutes` is always supplied by the caller (the notebook sets the actual
    values for each phase), never hardcoded deep inside a function.
    """

    def __init__(self, minutes: float = 15.0) -> None:
        self.minutes = minutes
        self.started_at = time.monotonic()
        self.deadline = self.started_at + minutes * 60

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at


def iter_with_progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    time_budget: TimeBudget | None = None,
) -> Iterator[T]:
    """Yield items from `iterable` behind a single live progress bar, not repeated log lines.

    Long operations in this project must show one meaningful progress bar (via `tqdm`)
    rather than a `logger.info` line printed per item -- callers should never additionally
    log inside the loop body for routine per-item status.

    If `time_budget` is given and expires mid-iteration, this stops yielding (does not
    raise) and logs a single summary line. Every caller of this function processes items
    through its own per-item cache (write-as-you-go), so whatever was already processed
    before the budget ran out is preserved on disk, and the next call resumes from there.
    """
    # tqdm.auto (not plain tqdm) picks a real ipywidgets-based live bar when running inside
    # a Jupyter kernel and the plain ASCII/carriage-return bar otherwise -- confirmed
    # directly that the plain ASCII bar's `\r` redraws don't reliably re-render in some
    # notebook/captured-output contexts, showing a frozen initial "0/N" instead of live
    # progress even while items (including near-instant cache hits) are really being
    # processed underneath.
    bar = tqdm(iterable, total=total, desc=desc, unit="item")
    processed = 0
    try:
        for item in bar:
            if time_budget is not None and time_budget.expired:
                logger.info(
                    "%s: time budget of %.1f min reached after %s item(s); stopping -- "
                    "re-run to continue.",
                    desc,
                    time_budget.minutes,
                    processed,
                )
                return
            yield item
            processed += 1
    finally:
        bar.close()
