from datetime import datetime, timezone


def utc_now_naive() -> datetime:
    """Return UTC time without tzinfo for the existing timezone-naive DB columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
