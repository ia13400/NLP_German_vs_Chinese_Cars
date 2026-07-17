from __future__ import annotations

from pathlib import Path

from car_interest_nlp.analysis.media_attention import build_attention_over_time


def main() -> int:
    """Build the media-attention-over-time table from cached GDELT `timelinevol` data.

    Built entirely from GDELT's own `timelinevol` aggregate mode, not from tallying
    individual articles -- works over whatever timeline chunks are cached so far (see
    `media_attention` module docstrings).
    """
    Path("artifacts/tables").mkdir(parents=True, exist_ok=True)

    over_time = build_attention_over_time()
    over_time.to_csv(
        "artifacts/tables/gdelt_attention_over_time.csv", index=False, encoding="utf-8"
    )

    print(f"attention-over-time rows: {len(over_time)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
