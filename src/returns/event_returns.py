# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.1", "pandas==2.3.0", "pyarrow==21.0.0"]
# ///
"""
Event-window stock returns around each linked earnings call.

Inputs are the linked calls (document_id, permno, event_date, event_title from
document_crsp_links joined to documents) and the CRSP files fetch_crsp.py writes.
Nothing is written to the DB.

Day 0
    The first trading session that can react to the call. A call starting at or
    after the 4pm New York close has day 0 on the next trading day; any earlier
    call (pre-open or during market hours) has day 0 on its own date, or on the
    next trading day if that date is not one. Trading days are the dates in the
    CRSP index file.

Windows
    Trading-day offsets from day 0, inclusive (WINDOWS): run-up [-20,-2];
    reaction [0,0], [0,+1], [-1,+1], [-1,+2], [0,+2]; drift [+2,+21],
    [+22,+42], [+43,+63], [+2,+61], [+2,+121]. A day with no CRSP return for the stock is skipped, and n_<w> counts
    the days used; a window with no days is missing, not zero.

Returns, per window w
    ret_<w>      buy-and-hold raw return, prod(1 + r) - 1
    bhar_<w>     ret_<w> minus the buy-and-hold CRSP value-weighted index return
                 over the same days
    car_mm_<w>   market-model cumulative abnormal return, sum(r - alpha - beta*m),
                 with alpha and beta from OLS on days [-250,-30], at least 100
                 days required

Delisting returns from CRSP's delisting file are compounded into the return on
the delisting date (or the next trading day), so a stock that delists inside a
window carries its last return instead of dropping out.

Usage
    uv run src/returns/event_returns.py --events results/event_returns/events.csv.gz \\
        --crsp-dir results/event_returns/crsp --out results/event_returns/event_returns.csv.gz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = [
    (-20, -2),                      # pre-call run-up
    (0, 0), (0, 1), (-1, 1), (-1, 2), (0, 2),  # reaction
    (2, 21), (22, 42), (43, 63),    # drift, month by month
    (2, 61), (2, 121),              # drift, three and six months
]
ESTIMATION = (-250, -30)
MIN_EST_DAYS = 100
CLOSE_HOUR = 16  # New York, 4pm


def window_label(start: int, end: int) -> str:
    def one(x):
        return f"m{-x}" if x < 0 else f"p{x}"
    return f"{one(start)}_{one(end)}"


def call_timing(event_date: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """New York call date and timing label (pre_open, market_hours, after_close)."""
    ny = pd.to_datetime(event_date, utc=True).dt.tz_convert("America/New_York")
    minutes = ny.dt.hour * 60 + ny.dt.minute
    timing = np.where(minutes >= CLOSE_HOUR * 60, "after_close",
                      np.where(minutes < 9 * 60 + 30, "pre_open", "market_hours"))
    return ny.dt.tz_localize(None).dt.normalize().to_numpy("datetime64[D]"), timing


def day0_index(cal: np.ndarray, call_day: np.ndarray, timing: np.ndarray) -> np.ndarray:
    """Position of day 0 in the trading calendar, -1 when past its end."""
    left = np.searchsorted(cal, call_day, side="left")
    right = np.searchsorted(cal, call_day, side="right")
    idx = np.where(timing == "after_close", right, left)
    return np.where(idx < len(cal), idx, -1)


def fold_delistings(dsf: pd.DataFrame, delist: pd.DataFrame, cal: np.ndarray) -> pd.DataFrame:
    """Compound each non-missing delisting return into the return on its trading day."""
    d = delist.dropna(subset=["dlret", "dlstdt"]).copy()
    pos = np.searchsorted(cal, pd.to_datetime(d["dlstdt"]).to_numpy("datetime64[D]"), side="left")
    d = d[pos < len(cal)]
    d["date"] = cal[pos[pos < len(cal)]]
    d = d[["permno", "date", "dlret"]].drop_duplicates(["permno", "date"])
    out = dsf.merge(d, on=["permno", "date"], how="outer")
    has_dl = out["dlret"].notna()
    out.loc[has_dl, "ret"] = (1 + out.loc[has_dl, "ret"].fillna(0)) * (1 + out.loc[has_dl, "dlret"]) - 1
    return out.drop(columns="dlret")


def _prefix(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x)])


def _span_sum(c: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Sum of the series behind prefix array c over positions [start, end], clipped."""
    n = len(c) - 1
    s = np.clip(start, 0, n)
    e = np.clip(end + 1, 0, n)
    return c[np.maximum(e, s)] - c[s]


def stock_windows(r: np.ndarray, m: np.ndarray, idx0: np.ndarray,
                  windows=WINDOWS, estimation=ESTIMATION, min_est=MIN_EST_DAYS) -> dict:
    """All window statistics for one stock's events.

    r is the stock's daily return on every calendar trading day (NaN when CRSP
    has none), m the index return on the same days, idx0 each event's day 0.
    """
    ok = ~np.isnan(r)
    rz = np.where(ok, r, 0.0)
    mz = np.where(ok, m, 0.0)
    c_n = _prefix(ok.astype(float))
    c_lr = _prefix(np.log1p(rz))
    c_lm = _prefix(np.log1p(mz))
    c_r, c_m = _prefix(rz), _prefix(mz)
    c_rm, c_mm = _prefix(rz * mz), _prefix(mz * mz)

    out = {}
    es, ee = idx0 + estimation[0], idx0 + estimation[1]
    n_est = _span_sum(c_n, es, ee)
    sr, sm = _span_sum(c_r, es, ee), _span_sum(c_m, es, ee)
    srm, smm = _span_sum(c_rm, es, ee), _span_sum(c_mm, es, ee)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_m = smm - sm * sm / n_est
        beta = (srm - sr * sm / n_est) / var_m
        alpha = (sr - beta * sm) / n_est
    bad = (n_est < min_est) | ~(var_m > 0)
    beta[bad] = np.nan
    alpha[bad] = np.nan
    out["mm_est_days"] = n_est.astype(int)
    out["mm_alpha"], out["mm_beta"] = alpha, beta

    for a, b in windows:
        lab = window_label(a, b)
        s, e = idx0 + a, idx0 + b
        n = _span_sum(c_n, s, e)
        ret = np.expm1(_span_sum(c_lr, s, e))
        mkt = np.expm1(_span_sum(c_lm, s, e))
        car = _span_sum(c_r, s, e) - n * alpha - beta * _span_sum(c_m, s, e)
        none = n == 0
        ret[none] = np.nan
        mkt[none] = np.nan
        car[none] = np.nan
        out[f"n_{lab}"] = n.astype(int)
        out[f"ret_{lab}"] = ret
        out[f"bhar_{lab}"] = ret - mkt
        out[f"car_mm_{lab}"] = car
    return out


def compute(events: pd.DataFrame, dsf: pd.DataFrame, dsi: pd.DataFrame,
            delist: pd.DataFrame | None = None, windows=WINDOWS) -> pd.DataFrame:
    dsi = dsi.sort_values("date")
    cal = pd.to_datetime(dsi["date"]).to_numpy("datetime64[D]")
    m = dsi["vwretd"].to_numpy(float)

    dsf = dsf[["permno", "date", "ret"]].copy()
    dsf["date"] = pd.to_datetime(dsf["date"]).to_numpy("datetime64[D]")
    if delist is not None:
        dsf = fold_delistings(dsf, delist, cal)
    pos = np.searchsorted(cal, dsf["date"].to_numpy("datetime64[D]"))
    on_cal = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == dsf["date"].to_numpy("datetime64[D]"))
    dsf = dsf[on_cal].assign(pos=pos[on_cal])

    ev = events[["document_id", "permno", "event_date", "event_title"]].copy()
    ev["call_day"], ev["call_timing"] = call_timing(ev["event_date"])
    ev["idx0"] = day0_index(cal, ev["call_day"].to_numpy("datetime64[D]"), ev["call_timing"].to_numpy())
    ev["day0"] = pd.NaT
    has0 = ev["idx0"] >= 0
    ev.loc[has0, "day0"] = cal[ev.loc[has0, "idx0"].to_numpy()]
    ev["is_earnings_call"] = ev["event_title"].str.contains("earnings", case=False, na=False)

    by_permno = {p: g for p, g in dsf.groupby("permno")}
    parts = []
    for permno, g in ev[has0].groupby("permno"):
        r = np.full(len(cal), np.nan)
        s = by_permno.get(permno)
        if s is not None:
            r[s["pos"].to_numpy()] = s["ret"].to_numpy(float)
        stats = stock_windows(r, m, g["idx0"].to_numpy(), windows=windows)
        parts.append(pd.DataFrame(stats, index=g.index))
    res = pd.concat(parts) if parts else pd.DataFrame(index=ev.index[:0])
    cols = ["document_id", "permno", "call_day", "call_timing", "day0", "is_earnings_call"]
    return ev[cols].join(res).sort_values("document_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--crsp-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    crsp = Path(args.crsp_dir)
    events = pd.read_csv(args.events)
    out = compute(events,
                  pd.read_parquet(crsp / "dsf.parquet"),
                  pd.read_parquet(crsp / "dsi.parquet"),
                  pd.read_parquet(crsp / "delist.parquet"))
    out.to_csv(args.out, index=False)

    print(f"events {len(out):,} | day 0 found {out['day0'].notna().sum():,}")
    print(out["call_timing"].value_counts().to_string())
    for a, b in WINDOWS:
        lab = window_label(a, b)
        print(f"{lab:>7}: ret {out[f'ret_{lab}'].notna().sum():,} | "
              f"car_mm {out[f'car_mm_{lab}'].notna().sum():,} | "
              f"mean bhar {out[f'bhar_{lab}'].mean():.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
