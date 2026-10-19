"""Checks day-0 timing, window compounding, delisting and market-model returns."""

import numpy as np
import pandas as pd
import pytest

from src.returns.event_returns import compute, day0_index, call_timing, window_label

# Weekdays only; holidays don't matter for these checks
CAL = pd.bdate_range("2019-01-01", "2020-03-31")


def _dsi(m=0.001):
    return pd.DataFrame({"date": CAL, "vwretd": m})


def _event(ts, doc=1, permno=10, title="Q4 2019 Acme Earnings Call"):
    return pd.DataFrame({"document_id": [doc], "permno": [permno],
                         "event_date": [ts], "event_title": [title]})


def test_window_label():
    assert window_label(-1, 1) == "m1_p1"
    assert window_label(2, 61) == "p2_p61"


@pytest.mark.parametrize("ts, timing, day0", [
    ("2020-01-08 08:00:00-05", "pre_open", "2020-01-08"),
    ("2020-01-08 11:00:00-05", "market_hours", "2020-01-08"),
    ("2020-01-08 16:00:00-05", "after_close", "2020-01-09"),
    ("2020-01-10 17:00:00-05", "after_close", "2020-01-13"),   # Friday evening -> Monday
    ("2020-01-11 10:00:00-05", "market_hours", "2020-01-13"),  # Saturday
    ("2020-01-08 21:30:00+00", "after_close", "2020-01-09"),   # UTC stamp, 4:30pm New York
])
def test_day0(ts, timing, day0):
    day, t = call_timing(pd.Series([ts]))
    assert t[0] == timing
    cal = CAL.to_numpy("datetime64[D]")
    assert cal[day0_index(cal, day, t)[0]] == np.datetime64(day0)


def test_raw_and_market_adjusted_compounding():
    rets = {"2020-01-07": 0.10, "2020-01-08": -0.05, "2020-01-09": 0.02}
    dsf = pd.DataFrame({"permno": 10, "date": CAL, "ret": 0.0})
    for d, r in rets.items():
        dsf.loc[dsf["date"] == d, "ret"] = r
    out = compute(_event("2020-01-08 10:00:00-05"), dsf, _dsi(0.001)).iloc[0]
    assert out["n_m1_p1"] == 3
    expect = 1.10 * 0.95 * 1.02 - 1
    assert out["ret_m1_p1"] == pytest.approx(expect)
    assert out["bhar_m1_p1"] == pytest.approx(expect - (1.001 ** 3 - 1))
    assert out["ret_p0_p1"] == pytest.approx(0.95 * 1.02 - 1)
    assert out["is_earnings_call"]


def test_missing_days_are_skipped_and_empty_window_is_nan():
    dsf = pd.DataFrame({"permno": 10, "date": CAL[CAL <= "2020-01-08"], "ret": 0.01})
    out = compute(_event("2020-01-08 10:00:00-05"), dsf, _dsi()).iloc[0]
    assert out["n_m1_p1"] == 2
    assert out["ret_m1_p1"] == pytest.approx(1.01 ** 2 - 1)
    assert out["n_p2_p61"] == 0
    assert np.isnan(out["ret_p2_p61"]) and np.isnan(out["car_mm_p2_p61"])


def test_delisting_return_is_compounded():
    dsf = pd.DataFrame({"permno": 10, "date": CAL[CAL <= "2020-01-09"], "ret": 0.0})
    delist = pd.DataFrame({"permno": [10], "dlstdt": [pd.Timestamp("2020-01-10")],
                           "dlret": [-0.30], "dlstcd": [552]})
    out = compute(_event("2020-01-08 10:00:00-05"), dsf, _dsi(), delist).iloc[0]
    assert out["n_p2_p61"] == 1
    assert out["ret_p2_p61"] == pytest.approx(-0.30)


def test_market_model_recovers_alpha_beta():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, len(CAL))
    r = 0.0002 + 1.5 * m
    day0 = CAL.get_loc(pd.Timestamp("2020-01-08"))
    r[day0] += 0.05  # the event-day surprise
    dsi = pd.DataFrame({"date": CAL, "vwretd": m})
    dsf = pd.DataFrame({"permno": 10, "date": CAL, "ret": r})
    out = compute(_event("2020-01-08 10:00:00-05"), dsf, dsi).iloc[0]
    assert out["mm_beta"] == pytest.approx(1.5)
    assert out["mm_alpha"] == pytest.approx(0.0002)
    assert out["car_mm_m1_p1"] == pytest.approx(0.05)
    assert out["mm_est_days"] == 221


def test_short_history_gives_no_market_model():
    dsf = pd.DataFrame({"permno": 10, "date": CAL[CAL >= "2019-11-01"], "ret": 0.01})
    out = compute(_event("2020-01-08 10:00:00-05"), dsf, _dsi()).iloc[0]
    assert np.isnan(out["mm_beta"]) and np.isnan(out["car_mm_m1_p1"])
    assert not np.isnan(out["ret_m1_p1"])
