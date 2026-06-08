"""Filename date-token inference helpers for registry scanning."""

from __future__ import annotations

import re

# Date tokens in filenames.
#
# Allow common no-separator data names such as ro2001.nc, tas201001.nc,
# and pr20100101.nc. Exclude version-ish tokens where the date is directly
# prefixed by a standalone "v" or "r" marker, e.g. ET_v2010.nc or MOD_r2050.nc.
_DATE_TOKEN = re.compile(
    r"(?<!\d)"
    r"(?P<token>"
    r"(?P<year>(?:19|20)\d{2})"
    r"(?:[-_/]?(?P<month>0[1-9]|1[0-2])"
    r"(?:[-_/]?(?P<day>0[1-9]|[12]\d|3[01]))?"
    r")?"
    r")"
    r"(?!\d)"
)


def _is_version_prefixed_date(stem: str, start: int) -> bool:
    """Return True for date tokens like v2010 or _r2050."""
    if start <= 0:
        return False
    marker = stem[start - 1].lower()
    if marker not in {"v", "r"}:
        return False
    if start == 1:
        return True
    return not stem[start - 2].isalnum()


def _iter_date_tokens(stem: str):
    """Yield non-version date regex matches from a filename stem."""
    for match in _DATE_TOKEN.finditer(stem):
        if _is_version_prefixed_date(stem, match.start("token")):
            continue
        yield match


def _date_granularity(match) -> str:
    if match.group("day"):
        return "day"
    if match.group("month"):
        return "month"
    return "year"


_DATE_GRANULARITY_RANK = {"none": 0, "year": 1, "month": 2, "day": 3}


def _most_specific_date_matches(stem: str):
    """Return date matches with the most-specific granularity in ``stem``."""
    matches = list(_iter_date_tokens(stem))
    if not matches:
        return []
    best_rank = max(_DATE_GRANULARITY_RANK[_date_granularity(match)] for match in matches)
    return [match for match in matches if _DATE_GRANULARITY_RANK[_date_granularity(match)] == best_rank]


def _is_year_range_endpoint(stem: str, match) -> bool:
    """Return True when a year-level token is part of an inline ``YYYY[-_]YYYY`` range."""
    if _date_granularity(match) != "year":
        return False
    start = match.start("token")
    end = match.end("token")
    if start >= 1 and end <= len(stem):
        before = stem[start - 1] if start >= 1 else ""
        if before in {"-", "_"}:
            preceding = stem[: start - 1]
            tail = re.search(r"(?<!\d)((?:19|20)\d{2})$", preceding)
            if tail:
                return True
    if end < len(stem) and stem[end] in {"-", "_"}:
        rest = stem[end + 1 :]
        head = re.match(r"(?:19|20)\d{2}(?!\d)", rest)
        if head:
            return True
    return False


def _filename_split_match(stem: str):
    """Pick the date token that should split a filename into prefix/suffix.

    Among the most-specific matches:
      * year-level tokens that are part of an inline ``YYYY-YYYY`` range
        (experiment time range) are excluded;
      * the rightmost remaining token wins so a single year/month/day token
        following the experiment name acts as the file's own date marker.

    Returns ``None`` when no match is available.
    """
    matches = _most_specific_date_matches(stem)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    filtered = [m for m in matches if not _is_year_range_endpoint(stem, m)]
    candidates = filtered or matches
    return max(candidates, key=lambda m: m.start("token"))


def _classify_filename_date(stem: str) -> str:
    """Return the most-specific date granularity in a filename stem.

    Order: day > month > year > none. Tokens preceded by letters (e.g., the
    "v" in "v2010") are excluded so version markers don't masquerade as years.
    """
    matches = _most_specific_date_matches(stem)
    if not matches:
        return "none"
    return _date_granularity(matches[0])


__all__ = [
    "_classify_filename_date",
    "_filename_split_match",
    "_is_year_range_endpoint",
    "_most_specific_date_matches",
]
