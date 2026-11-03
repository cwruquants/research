"""
Earnings surprise (SUE) for each earnings call in event_returns.py's output.

    sue = (actual EPS - median consensus) / share price

Each call gets its IBES ticker from the CRSP link table (the link covering the
call date with the best score), then the quarterly announcement closest to the
call date, from 3 days before to 1 day after it, then the last monthly fpi 6
consensus for that fiscal period dated before the announcement. Only USD actuals
and forecasts are used, since the price is in USD. The price is |prc| on the same
day mcap_pre uses, the last trading day at least 2 calendar days before day 0.

Actuals and forecasts come from the unadjusted I/B/E/S files, so they are on the
share count of their own date. When CRSP cfacpr changes before the announcement,
the forecast and the price are rescaled to the announcement's share basis.

sue_fresh is the same measure on a fresher consensus from ibes.detu_epsus: the
median of each analyst's latest forecast in the 90 days before the announcement.

release_day is the first trading session able to react to the release, by the
same 4pm New York rule that sets day 0 for the call, and early_release marks calls
where it comes before day 0. Both are empty when I/B/E/S has no announcement time.

Pulls everything in one WRDS connection (credentials from ~/.pgpass, as in
fetch_crsp.py) and caches the raw tables as Parquet in --cache-dir, so a rerun
does not reconnect. Writes one row per matched call to --out and the match rate
by call year to --coverage. Nothing is written to the DB.

Usage
    uv run src/returns/earnings_surprise.py --returns results/event_returns/event_returns.csv.gz \\
        --cache-dir results/event_returns/ibes --out results/event_returns/earnings_surprise.csv \\
        --coverage results/event_returns/earnings_surprise_coverage.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from fetch_crsp import WRDS_HOST, WRDS_PORT, copy_frame, wrds_user  # noqa: E402
from market_cap import LAG, TOLERANCE  # noqa: E402

DAYS_BEFORE_CALL = 3
DAYS_AFTER_CALL = 1
FRESH_DAYS = 90
MARKET_CLOSE = pd.Timedelta(hours=16)
UNKNOWN_TIME = "00:00:00"
TABLES = ["link", "actu", "consensus", "detail", "prices"]


def pull(calls: pd.DataFrame, cache: Path) -> dict[str, pd.DataFrame]:
    paths = {name: cache / f"{name}.parquet" for name in TABLES}
    if all(p.exists() for p in paths.values()):
        return {name: pd.read_parquet(p) for name, p in paths.items()}

    cache.mkdir(parents=True, exist_ok=True)
    permnos = ",".join(map(str, sorted(calls.permno.unique())))
    start = (calls.call_day.min() - pd.Timedelta(days=200)).date()
    end = (calls.call_day.max() + pd.Timedelta(days=10)).date()
    with psycopg.connect(host=WRDS_HOST, port=WRDS_PORT, dbname="wrds", user=wrds_user(),
                         sslmode="require", connect_timeout=170) as conn:
        conn.read_only = True
        link = copy_frame(conn, "select ticker, permno, sdate, edate, score from wrdsapps_link_crsp_ibes.ibcrsphist "
                                f"where permno in ({permnos})").dropna(subset=["ticker"])
        tickers = ",".join(f"'{t}'" for t in sorted(link.ticker.unique()))
        queries = {
            "actu": "select ticker, pends, anndats, anntims, value as actual, curr_act from ibes.actu_epsus "
                    f"where ticker in ({tickers}) and pdicity = 'QTR' and measure = 'EPS' "
                    f"and anndats between '{start}' and '{end}'",
            "consensus": "select ticker, fpedats, statpers, medest, meanest, numest, stdev, curcode "
                         "from ibes.statsumu_epsus where fpi = '6' and measure = 'EPS' and fiscalp = 'QTR' "
                         f"and ticker in ({tickers}) and statpers between '{start}' and '{end}'",
            # fpedats alone would also match annual forecasts at fiscal year end, so keep quarterly fpi only
            "detail": "select ticker, fpedats, estimator, analys, anndats as forecast_date, value, report_curr "
                      "from ibes.detu_epsus where fpi in ('6', '7') and measure = 'EPS' "
                      f"and ticker in ({tickers}) and anndats between '{start}' and '{end}'",
            "prices": f"select permno, date, prc, cfacpr from crsp.dsf where permno in ({permnos}) "
                      f"and date between '{start}' and '{end}'",
        }
        tables = {"link": link}
        for name, query in queries.items():
            tables[name] = copy_frame(conn, query)
            print(f"{name}: {len(tables[name]):,} rows")
    for name, df in tables.items():
        df.to_parquet(paths[name], index=False)
    return tables


def to_days(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    return df


def match_ticker(calls: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    m = calls.merge(link, on="permno")
    m = m[(m.sdate <= m.call_day) & (m.edate.isna() | (m.call_day <= m.edate))]
    return m.sort_values(["document_id", "score", "ticker"]).drop_duplicates("document_id")


def match_announcement(m: pd.DataFrame, actu: pd.DataFrame) -> pd.DataFrame:
    m = m.merge(actu, on="ticker")
    m["gap"] = (m.anndats - m.call_day).dt.days
    m = m[m.gap.between(-DAYS_BEFORE_CALL, DAYS_AFTER_CALL)].assign(abs_gap=lambda d: d.gap.abs())
    return m.sort_values(["document_id", "abs_gap", "gap"]).drop_duplicates("document_id")


def match_consensus(m: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    m = m.merge(consensus, left_on=["ticker", "pends"], right_on=["ticker", "fpedats"])
    m = m[m.statpers < m.anndats]
    return m.sort_values(["document_id", "statpers"]).drop_duplicates("document_id", keep="last")


def asof(df: pd.DataFrame, date_col: str, prices: pd.DataFrame, col: str, tolerance=None) -> np.ndarray:
    """The value of a CRSP column on or before each row's date, in df's row order."""
    left = df[["permno", date_col]].assign(row=np.arange(len(df))).sort_values(date_col)
    hit = pd.merge_asof(left, prices[["permno", "date", col]], left_on=date_col, right_on="date", by="permno",
                        direction="backward", tolerance=tolerance)
    return hit.sort_values("row")[col].to_numpy()


def fresh_consensus(m: pd.DataFrame, detail: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    keys = ["permno", "ticker", "pends", "anndats"]
    f = m[keys + ["factor_ann"]].drop_duplicates().merge(detail, left_on=["ticker", "pends"], right_on=["ticker", "fpedats"])
    f = f[(f.forecast_date < f.anndats) & (f.forecast_date >= f.anndats - pd.Timedelta(days=FRESH_DAYS))]
    f = f.sort_values("forecast_date").drop_duplicates(keys + ["estimator", "analys"], keep="last")
    f["value"] = f.value * f.factor_ann / asof(f, "forecast_date", factors, "cfacpr")
    return f.groupby(keys).value.agg(medest_fresh="median", numest_fresh="size").reset_index()


def release_day(m: pd.DataFrame, cal: np.ndarray) -> pd.Series:
    """First trading session able to react to the release, NaT without a release time."""
    days = m.anndats.to_numpy("datetime64[ns]")
    after_close = (pd.to_timedelta(m.anntims) >= MARKET_CLOSE).to_numpy()
    idx = np.where(after_close, np.searchsorted(cal, days, side="right"), np.searchsorted(cal, days, side="left"))
    day = pd.Series(cal[np.minimum(idx, len(cal) - 1)], index=m.index)
    return day.where((idx < len(cal)) & (m.anntims != UNKNOWN_TIME))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--returns", required=True, help="event_returns.py output (document_id, permno, call_day, day0, is_earnings_call)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", required=True)
    args = ap.parse_args()

    calls = pd.read_csv(args.returns, usecols=["document_id", "permno", "call_day", "day0", "is_earnings_call"])
    calls = calls[calls.is_earnings_call.astype(str).str.lower().isin(["t", "true"])].dropna(subset=["day0"])
    calls = to_days(calls.drop(columns="is_earnings_call"), "call_day", "day0")
    calls["permno"] = calls.permno.astype(int)

    tables = pull(calls, Path(args.cache_dir))
    link = to_days(tables["link"], "sdate", "edate")
    actu = to_days(tables["actu"], "pends", "anndats")
    actu = actu[(actu.curr_act == "USD") & actu.actual.notna()].drop_duplicates(["ticker", "pends", "anndats"])
    consensus = to_days(tables["consensus"], "fpedats", "statpers")
    consensus = consensus[(consensus.curcode == "USD") & consensus.medest.notna()]
    detail = to_days(tables["detail"], "fpedats", "forecast_date")
    detail = detail[(detail.report_curr == "USD") & detail.value.notna()]
    prices = to_days(tables["prices"], "date").sort_values("date")
    cal = np.sort(prices.date.unique())
    factors = prices[prices.cfacpr > 0]
    quotes = prices[prices.prc.notna() & (prices.prc != 0)]

    ticker = match_ticker(calls, link)
    announced = match_announcement(ticker, actu)
    m = match_consensus(announced, consensus)

    m["price_date"] = m.day0 - LAG
    m["price"] = np.abs(asof(m, "price_date", quotes, "prc", TOLERANCE))
    m["quote_factor"] = asof(m, "price_date", quotes, "cfacpr", TOLERANCE)
    m["factor_ann"] = asof(m, "anndats", factors, "cfacpr")
    m["factor_stat"] = asof(m, "statpers", factors, "cfacpr")
    m["split_ratio"] = (m.factor_ann / m.factor_stat).fillna(1.0)
    m["price"] = m.price * (m.factor_ann / m.quote_factor).fillna(1.0)
    m["sue"] = (m.actual - m.medest * m.split_ratio) / m.price

    m = m.merge(fresh_consensus(m, detail, factors), on=["permno", "ticker", "pends", "anndats"], how="left")
    m["sue_fresh"] = (m.actual - m.medest_fresh) / m.price

    m["release_day"] = release_day(m, cal)
    m["early_release"] = (m.release_day < m.day0).where(m.release_day.notna())

    split = m[m.split_ratio != 1]
    closer = ((split.actual - split.medest * split.split_ratio).abs() < (split.actual - split.medest).abs()).mean()
    print(f"split between consensus and announcement: {len(split):,} calls, rescaled forecast closer in {closer:.1%}")
    print(f"release before day 0: {(m.early_release == True).sum():,}, after day 0: {(m.release_day > m.day0).sum():,}, "  # noqa: E712
          f"no release time: {m.release_day.isna().sum():,}")
    print(f"consensus age in days: {(m.anndats - m.statpers).dt.days.describe(percentiles=[.5, .9, .99]).round(0).to_dict()}")
    print(f"calls sharing an announcement with another call: {m.duplicated(['ticker', 'pends'], keep=False).sum():,}")

    out = m.dropna(subset=["sue"]).rename(columns={"ticker": "ibes_ticker"})
    cols = ["document_id", "ibes_ticker", "anndats", "anntims", "pends", "statpers", "actual", "medest", "meanest", "numest",
            "stdev", "split_ratio", "price", "sue", "medest_fresh", "numest_fresh", "sue_fresh", "release_day", "early_release"]
    out = out[cols].sort_values("document_id")
    out["early_release"] = out.early_release.astype("boolean")
    out[["sue", "sue_fresh"]] = out[["sue", "sue_fresh"]].round(8)
    out.to_csv(args.out, index=False)
    print(f"{len(out):,} of {len(calls):,} calls have sue -> {args.out}")

    cov = calls.assign(year=calls.call_day.dt.year, ticker=calls.document_id.isin(ticker.document_id),
                       announcement=calls.document_id.isin(announced.document_id),
                       consensus=calls.document_id.isin(m.document_id), sue=calls.document_id.isin(out.document_id))
    stages = ["ticker", "announcement", "consensus", "sue"]
    by_year = cov.groupby("year")[stages].sum()
    by_year.insert(0, "calls", cov.groupby("year").size())
    by_year.loc["all"] = by_year.sum()
    for s in stages:
        by_year[s + "_pct"] = (by_year[s] / by_year.calls * 100).round(1)
    by_year.to_csv(args.coverage)
    print(by_year.to_string())


if __name__ == "__main__":
    main()
