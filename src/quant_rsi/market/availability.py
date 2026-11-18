"""Shared point-in-time conventions for research inputs (#159 D3).

This module is the data layer's single source for when a dated value becomes usable.
It imports only the core contracts, never evaluation, forecast or research code, so
every research source can depend on it without pulling in another pipeline.

The older forecast/execution stack (``market_data.knowledge_time`` and
``market.asof``) keeps its own conventions until #159 D2 decides that pipeline's
future; nothing here changes it.
"""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

# Versioned so a run can record which rule derived each availability time.
DATE_ONLY_RULE = "date_only_next_day_start_v1"

# With no declared source timezone, assume the latest local midnight on Earth (UTC-12):
# the start of the next calendar day there is next day 12:00 UTC.
_UNKNOWN_ZONE_OFFSET = timedelta(hours=12)


def date_only_available_at(day: date | str, timezone: str | None = None) -> datetime:
    """When a value published with a date but no time becomes usable (D3.2).

    A date-only release may have appeared at any time that day, so it is usable from the
    start of the next calendar day in the source's declared timezone. Without a declared
    timezone the most conservative zone (UTC-12) applies.
    """
    day = date.fromisoformat(day) if isinstance(day, str) else day
    following = day + timedelta(days=1)
    if timezone is None:
        return datetime.combine(following, datetime.min.time(), UTC) + _UNKNOWN_ZONE_OFFSET
    local = datetime.combine(following, datetime.min.time(), ZoneInfo(timezone))
    return local.astimezone(UTC)


# A value is usable at a decision instant only if it was available strictly before it
# (D3.1). Under the default close-to-close label, information stamped exactly at the close
# describes the same post-close move the label measures.
AT_INSTANT_RULE = "strictly_before_decision_instant_v1"
# Freshness is measured from publication, never from receipt, version or extraction, so
# re-fetching an old release cannot renew it (D3.3).
MAX_AGE_RULE = "publication_anchored_exact_age_v1"


def usable_at(available_at: datetime, instant: datetime) -> bool:
    return available_at < instant


def within_max_age(published_at: datetime, instant: datetime, max_age_days: int) -> bool:
    """Exact duration: no flooring to whole days."""
    return instant - published_at <= timedelta(days=max_age_days)
