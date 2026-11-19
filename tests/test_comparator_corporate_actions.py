"""Comparators truncate at an unreviewed corporate action; targets still refuse (#159).

Measured on the TSM run with fits: SOXX's 3:1 split on 2026-11-05 made the sector
comparator unavailable, although the split followed the last development row.
"""

from datetime import datetime

import numpy as np
import pytest
from test_daily_data import install

from quant_rsi.research import demo_prices
from quant_rsi.research.study_peers import aligned_closes, peer_feature


def test_target_prices_still_refuse_an_unreviewed_split(tmp_path, monkeypatch):
    install(monkeypatch)  # the SOXL fixture has a split inside its price window
    with pytest.raises(ValueError, match="Unresolved corporate action"):
        demo_prices.load_daily_prices("SOXL", tmp_path)
    with pytest.raises(ValueError, match="Unknown unresolved corporate-action policy"):
        demo_prices.load_daily_prices("SOXL", tmp_path, unresolved_actions="adjust")


def test_comparator_closes_from_the_action_on_are_missing_never_adjusted(tmp_path, monkeypatch):
    install(monkeypatch)
    data, info = demo_prices.load_daily_prices("SOXL", tmp_path, unresolved_actions="truncate")
    action = datetime.fromisoformat(info["unresolved_corporate_action_at"])
    assert info["price_basis"] == "raw"
    closes = info["session_close_times"]
    aligned = aligned_closes(data.dates, closes, data, info)
    before = [datetime.fromisoformat(c) < action for c in closes]
    assert any(before) and not all(before)
    for index, usable in enumerate(before):
        if usable:
            assert aligned[index] == data.closes[index, 0]  # raw, unadjusted
        else:
            assert np.isnan(aligned[index])
    # No trailing return spans the action: windows ending after it are missing.
    returns = peer_feature(np.ones(len(aligned)), aligned, "peer_return", 10)
    last = max(i for i, usable in enumerate(before) if usable)
    assert np.isfinite(returns[10 : last + 1]).all()
    assert np.isnan(returns[last + 1 :]).all()
