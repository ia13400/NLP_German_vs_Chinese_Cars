from __future__ import annotations


class SourceUnavailableError(RuntimeError):
    """Raised when a real data source cannot be collected and no silent fallback is allowed.

    Live mode must never substitute synthetic data for a source it cannot reach; instead it
    raises this error so the caller learns exactly what is missing and what to do about it.
    """

    def __init__(
        self, source: str, reason: str, required_action: str, accepted_fallback: str
    ) -> None:
        self.source = source
        self.reason = reason
        self.required_action = required_action
        self.accepted_fallback = accepted_fallback
        super().__init__(
            f"Source '{source}' is unavailable: {reason} "
            f"Required action: {required_action} "
            f"Accepted fallback: {accepted_fallback}"
        )
