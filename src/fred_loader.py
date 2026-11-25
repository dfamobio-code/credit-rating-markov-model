"""Retrieve FRED's monthly USREC indicator and classify calendar years."""

import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


def load_fred_data(start_year=1981, end_year=2025, api_key=None):
    """Return monthly recession observations with columns date and USREC.

    If FRED_API_KEY is set (or api_key is supplied), use the FRED API.
    Otherwise use FRED's public CSV download, which needs no API key.
    """
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")

    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    api_key = api_key or os.getenv("FRED_API_KEY")

    if api_key:
        params = urlencode({
            "series_id": "USREC", "api_key": api_key,
            "file_type": "json", "observation_start": start,
            "observation_end": end,
        })
        url = "https://api.stlouisfed.org/fred/series/observations?" + params
        with urlopen(url, timeout=30) as response:
            observations = json.load(response)["observations"]
        monthly = pd.DataFrame(observations)[["date", "value"]]
        monthly = monthly.rename(columns={"value": "USREC"})
    else:
        params = urlencode({"id": "USREC", "cosd": start, "coed": end})
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + params
        with urlopen(url, timeout=30) as response:
            monthly = pd.read_csv(response)
        monthly = monthly.rename(columns={monthly.columns[0]: "date"})

    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["USREC"] = pd.to_numeric(monthly["USREC"], errors="coerce")
    monthly = monthly.loc[monthly["date"].dt.year.between(start_year, end_year)]
    return monthly[["date", "USREC"]].sort_values("date").reset_index(drop=True)


def classify_regimes(monthly):
    """Label a year recession if at least one month has USREC = 1.

    Incomplete calendar years are rejected, not treated as expansions.
    """
    data = monthly[["date", "USREC"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.to_period("M")
    data["USREC"] = pd.to_numeric(data["USREC"], errors="coerce")
    if data.empty or data["date"].isna().any():
        raise ValueError("No usable monthly dates.")
    if data["month"].duplicated().any():
        raise ValueError("Duplicate monthly observations.")
    if not data["USREC"].isin([0, 1]).all():
        raise ValueError("USREC must contain only 0 or 1, with no missing values.")

    data["year"] = data["date"].dt.year
    annual = data.groupby("year")["USREC"].agg(["sum", "count"])
    if (annual["count"] != 12).any():
        raise ValueError("Every included year must have all 12 monthly observations.")

    annual["regime"] = annual["sum"].gt(0).map({True: "recession", False: "expansion"})
    return (annual.rename(columns={"sum": "recession_months"})
            .reset_index()[["year", "regime", "recession_months"]])


def load_annual_regimes(start_year=1981, end_year=2025, api_key=None):
    """Convenience function: download monthly data and return annual regimes."""
    return classify_regimes(load_fred_data(start_year, end_year, api_key))
