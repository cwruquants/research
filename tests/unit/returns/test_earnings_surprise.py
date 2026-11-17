"""Checks when the earnings-surprise WRDS cache can be reused."""

import pandas as pd

from research.returns.earnings_surprise import covers, pull_scope


def _calls(permnos, days):
    return pd.DataFrame({"permno": permnos, "call_day": pd.to_datetime(days)})


def test_cache_covers_a_subset_of_its_calls():
    cached = pull_scope(_calls([10001, 10002], ["2010-01-15", "2021-11-01"]))
    assert covers(cached, pull_scope(_calls([10002], ["2015-06-01"])))


def test_cache_does_not_cover_newer_calls_or_new_stocks():
    cached = pull_scope(_calls([10001, 10002], ["2010-01-15", "2021-11-01"]))
    assert not covers(cached, pull_scope(_calls([10001], ["2023-05-01"])))
    assert not covers(cached, pull_scope(_calls([10003], ["2015-06-01"])))
