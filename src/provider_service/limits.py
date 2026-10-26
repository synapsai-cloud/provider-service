"""In-flight requests one replica accepts when the caller does not send a limit."""

DEFAULT_REQUEST_LIMIT = 16


def request_limit(value: int | None = None) -> int:
    if value is None:
        return DEFAULT_REQUEST_LIMIT
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("request_limit must be a positive integer")
    return parsed
