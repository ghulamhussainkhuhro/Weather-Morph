# climatology.py
# NASA POWER → Climatology likelihoods for “very hot / wet / windy / uncomfortable”
# Dependencies: requests, pandas, numpy

import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import numpy as np


# ---------------------------
# 0) Configuration & Types
# ---------------------------

POWER_DAILY_POINT = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Default variables we’ll request. Add RH2M for Heat Index.
DEFAULT_VARS = ("T2M", "PRECTOTCORR", "WS10M", "RH2M")

@dataclass
class ConditionDef:
    """Defines a condition like 'very_hot' or 'very_wet'."""
    label: str        # e.g., "very_hot"
    var: str          # e.g., "T2M" or "HI" (derived)
    op: str           # ">=" or "<="
    value: float      # threshold (ignored if percentile_mode=True)
    unit: str         # display unit
    percentile_mode: bool = False   # if True, compute var-specific percentile as threshold
    percentile_p: float = 0.9       # used only when percentile_mode=True

# Recommended baseline condition set
DEFAULT_CONDITIONS = [
    ConditionDef(label="very_hot",      var="T2M", op=">=", value=35.0, unit="°C", percentile_mode=False),
    ConditionDef(label="very_cold",     var="T2M", op="<=", value=5.0,  unit="°C", percentile_mode=False),
    ConditionDef(label="very_wet",      var="PRECTOTCORR", op=">=", value=10.0, unit="mm/day", percentile_mode=False),
    ConditionDef(label="very_windy",    var="WS10M", op=">=", value=8.0, unit="m/s", percentile_mode=False),
    ConditionDef(label="uncomfortable", var="HI", op=">=", value=40.0, unit="°C HI", percentile_mode=False),
]


# ------------------------------------------
# 1) NASA POWER client: fetch & to DataFrame
# ------------------------------------------

def fetch_power_daily(lat: float, lon: float, start_yyyymmdd: int, end_yyyymmdd: int,
                      params: Tuple[str, ...] = DEFAULT_VARS) -> dict:
    """Fetch daily data from NASA POWER for a point and date range."""
    query = {
        "parameters": ",".join(params),
        "community": "SB",
        "longitude": lon,
        "latitude": lat,
        "start": start_yyyymmdd,
        "end": end_yyyymmdd,
        "format": "JSON",
    }
    r = requests.get(POWER_DAILY_POINT, params=query, timeout=60)
    r.raise_for_status()
    return r.json()


def power_to_df(power_json: dict) -> pd.DataFrame:
    """
    Convert POWER JSON to a tidy DataFrame with a Date index and columns per variable.
    Missing values (-999) become NaN.
    Adds helper columns: year, month, day, doy.
    """
    params_obj = power_json["properties"]["parameter"]  # dict: var -> {datestr: value}
    df = pd.DataFrame({var: pd.Series(vals) for var, vals in params_obj.items()})

    # Index is YYYYMMDD; convert to datetime
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"

    # Convert -999 to NaN and coerce numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -998.9, c] = np.nan  # handle -999 fill value

    # Add helpers
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["doy"] = df.index.dayofyear

    return df


# ------------------------------------------
# 2) Derived metrics: Heat Index, Wind Chill
# ------------------------------------------

def heat_index_c_from_t_rh(t_c: float, rh_pct: float) -> Optional[float]:
    """
    Rothfusz regression for Heat Index.
    Input: temperature in °C, relative humidity in %.
    Returns HI in °C. Returns None if inputs invalid.
    Notes:
      - Formula is defined for T >= 26.7°C (80°F). For cooler temps, HI ≈ T.
      - We’ll still compute for all, but clamp RH range sensibly.
    """
    if np.isnan(t_c) or np.isnan(rh_pct):
        return None
    t_f = t_c * 9/5 + 32
    rh = max(0.0, min(100.0, rh_pct))

    # Rothfusz regression in Fahrenheit
    hi_f = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh
        - 0.22475541 * t_f * rh - 6.83783e-3 * t_f**2
        - 5.481717e-2 * rh*2 + 1.22874e-3 * t_f*2 * rh
        + 8.5282e-4 * t_f * rh*2 - 1.99e-6 * t_f**2 * rh*2
    )

    # Simple adjustments commonly used
    if (rh < 13) and (80 <= t_f <= 112):
        adjustment = ((13 - rh) / 4) * math.sqrt((17 - abs(t_f - 95)) / 17)
        hi_f -= adjustment
    elif (rh > 85) and (80 <= t_f <= 87):
        hi_f += 0.02 * (rh - 85) * (87 - t_f)

    hi_c = (hi_f - 32) * 5/9
    return hi_c


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (HI, etc.) when source variables are present."""
    df = df.copy()
    if "T2M" in df.columns and "RH2M" in df.columns:
        df["HI"] = [
            heat_index_c_from_t_rh(t, rh) if (pd.notna(t) and pd.notna(rh)) else np.nan
            for t, rh in zip(df["T2M"].values, df["RH2M"].values)
        ]
    return df


# ----------------------------------------------------
# 3) DOY window selection & utility for wrap-around
# ----------------------------------------------------

def doy_window_mask(dates: pd.DatetimeIndex, month: int, day: int, window_days: int = 7) -> pd.Series:
    """
    For a (month, day) target, return a boolean mask True for dates within ±window_days
    around the target day-of-year, with wrap-around (Jan/Dec boundary).
    Uses a LEAP reference year (2000) to safely handle Feb 29.
    """
    REF_YEAR = 2000  # leap year

    # target DOY on leap year ref (handles Feb 29 just fine)
    target_doy = date(REF_YEAR, month, day).timetuple().tm_yday

    # map each timestamp's month/day onto the same leap reference year
    def to_doy(ts: pd.Timestamp) -> int:
        return date(REF_YEAR, ts.month, ts.day).timetuple().tm_yday

    doys = pd.Series([to_doy(d) for d in dates], index=dates)

    # circular distance on 366-day circle
    dist = np.minimum((doys - target_doy) % 366, (target_doy - doys) % 366)
    return dist <= window_days



# ----------------------------------------------------
# 4) Probability machinery
# ----------------------------------------------------

def wilson_ci(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion (95% by default).
    Returns (low, high) as percentages (0..100).
    """
    if n == 0:
        return (np.nan, np.nan)
    p = hits / n
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    margin = z * math.sqrt( (p*(1-p)/n) + (z*2)/(4*n*2) ) / denom
    low = max(0.0, center - margin) * 100
    high = min(1.0, center + margin) * 100
    return (low, high)


def apply_operator(series: pd.Series, op: str, thr: float) -> pd.Series:
    if op == ">=":
        return series >= thr
    elif op == "<=":
        return series <= thr
    else:
        raise ValueError(f"Unsupported operator: {op}")


def compute_condition_probability(
    df_window: pd.DataFrame,
    cond: ConditionDef,
    percentile_thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute probability + CI for one condition on a DOY-windowed dataframe.
    Returns a dict with n, hits, prob, ci_low, ci_high, threshold_used.
    """
    data_col = cond.var
    if data_col not in df_window.columns:
        return {
            "label": cond.label, "n": 0, "hits": 0, "prob": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "threshold_used": np.nan, "unit": cond.unit
        }

    series = df_window[data_col].dropna()
    n = int(series.shape[0])
    if n == 0:
        return {
            "label": cond.label, "n": 0, "hits": 0, "prob": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "threshold_used": np.nan, "unit": cond.unit
        }

    # Threshold selection
    if cond.percentile_mode:
        if percentile_thresholds is None or data_col not in percentile_thresholds:
            thr = float(np.nanpercentile(series.values, cond.percentile_p * 100))
        else:
            thr = percentile_thresholds[data_col]
    else:
        thr = cond.value

    flags = apply_operator(series, cond.op, thr)
    hits = int(flags.sum())
    prob_pct = (hits / n) * 100.0
    ci_low, ci_high = wilson_ci(hits, n)

    return {
        "label": cond.label,
        "n": n,
        "hits": hits,
        "prob": round(prob_pct, 1),
        "ci_low": round(ci_low, 1),
        "ci_high": round(ci_high, 1),
        "threshold_used": round(thr, 3),
        "unit": cond.unit
    }


# ----------------------------------------------------
# 5) Trend (year-by-year probability around same DOY)
# ----------------------------------------------------

def yearly_probability_series(
    df: pd.DataFrame,
    month: int, day: int, window_days: int,
    cond: ConditionDef
) -> pd.DataFrame:
    """
    For each year, compute probability of the condition within the same DOY window.
    Returns a DataFrame with columns: year, n, hits, prob.
    """
    years = sorted(df["year"].unique())
    rows = []
    for y in years:
        df_y = df[df["year"] == y]
        mask = doy_window_mask(df_y.index, month, day, window_days)
        df_win = df_y.loc[mask]
        # Add derived metrics inside loop too (if not already)
        df_win = add_derived_metrics(df_win)

        res = compute_condition_probability(df_win, cond)
        rows.append({"year": y, "n": res["n"], "hits": res["hits"], "prob": res["prob"]})

    return pd.DataFrame(rows).dropna(subset=["prob"])


def linear_trend_pct_per_decade(years: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """
    Simple linear trend: slope in percentage points per decade.
    Also returns R^2 for a quick quality hint (no p-values to avoid SciPy dependency).
    """
    if len(years) < 3:
        return {"slope_pp_per_decade": np.nan, "r2": np.nan}

    # Fit prob = a*year + b
    coeffs = np.polyfit(years, probs, 1)
    slope_per_year = coeffs[0]  # percentage points per calendar year
    slope_per_decade = slope_per_year * 10.0

    # R^2
    y_pred = np.polyval(coeffs, years)
    ss_res = np.sum((probs - y_pred)**2)
    ss_tot = np.sum((probs - np.mean(probs))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return {"slope_pp_per_decade": round(float(slope_per_decade), 2), "r2": round(float(r2), 3)}


# ----------------------------------------------------
# 6) Main analysis function
# ----------------------------------------------------

def analyze_climatology(
    lat: float,
    lon: float,
    month: int,
    day: int,
    start_year: int = 2001,
    end_year: Optional[int] = None,  # default: last complete/plausible year
    window_days: int = 7,
    conditions: List[ConditionDef] = DEFAULT_CONDITIONS,
    percentile_mode: bool = False
) -> Dict:
    """
    End-to-end analysis:
      - fetch POWER daily data (multi-year)
      - build DOY window sample across all years
      - compute derived metrics (HI)
      - compute probabilities (with Wilson CI)
      - compute descriptive stats
      - compute simple trends per condition
    Returns a structured dict for your UI/backend.
    """

    if end_year is None:
        end_year = datetime.now().year  # include current year data where available

    start = int(f"{start_year}0101")
    end = int(f"{end_year}1231")

    # 1) Fetch & clean
    raw = fetch_power_daily(lat, lon, start, end, params=DEFAULT_VARS)
    df = power_to_df(raw)
    df = add_derived_metrics(df)

    # 2) Build DOY window sample across all years
    mask = doy_window_mask(df.index, month, day, window_days)
    df_win = df.loc[mask].copy()

    # 3) Percentile thresholds (if global percentile mode is desired)
    # Compute once per variable from the DOY window sample
    percentile_thresholds = None
    if percentile_mode:
        percentile_thresholds = {}
        for cond in conditions:
            if cond.var in df_win.columns:
                series = df_win[cond.var].dropna()
                if series.empty:
                    percentile_thresholds[cond.var] = np.nan
                else:
                    percentile_thresholds[cond.var] = float(np.nanpercentile(series, cond.percentile_p * 100))

    # 4) Probabilities for each condition
    results = []
    for cond in conditions:
        # clone condition with requested percentile mode (global switch)
        c = ConditionDef(
            label=cond.label, var=cond.var, op=cond.op, value=cond.value, unit=cond.unit,
            percentile_mode=percentile_mode if cond.var != "HI" else cond.percentile_mode,  # allow HI to keep absolute if desired
            percentile_p=cond.percentile_p
        )
        results.append(compute_condition_probability(df_win, c, percentile_thresholds))

    # 5) Descriptive statistics for UI
    def describe_safe(series: pd.Series) -> Dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {"mean": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan, "std": np.nan}
        return {
            "mean": round(float(s.mean()), 2),
            "p10": round(float(np.nanpercentile(s, 10)), 2),
            "p50": round(float(np.nanpercentile(s, 50)), 2),
            "p90": round(float(np.nanpercentile(s, 90)), 2),
            "std": round(float(s.std(ddof=1)), 2) if len(s) > 1 else 0.0,
        }

    stats = {}
    for var in ["T2M", "PRECTOTCORR", "WS10M", "HI"]:
        if var in df_win.columns:
            stats[var] = describe_safe(df_win[var])

    # 6) Trend per condition (yearly probs)
    trends = {}
    for cond in conditions:
        ys = yearly_probability_series(df, month, day, window_days, cond)
        if not ys.empty:
            t = linear_trend_pct_per_decade(ys["year"].values.astype(float), ys["prob"].values.astype(float))
            trends[cond.label] = {
                "trend_pp_per_decade": t["slope_pp_per_decade"],
                "r2": t["r2"],
                "series": ys.to_dict(orient="records")
            }
        else:
            trends[cond.label] = {"trend_pp_per_decade": np.nan, "r2": np.nan, "series": []}

    # 7) Build response dict (perfect for UI & download)
    header = raw.get("header", {})
    meta = {
        "source": "NASA POWER (MERRA-2), Daily Point API",
        "api_version": header.get("api", {}).get("version", ""),
        "time_standard": header.get("time_standard", "LST"),
        "lat": lat, "lon": lon,
        "years": [int(df["year"].min()), int(df["year"].max())] if not df.empty else [start_year, end_year],
        "target_month": month, "target_day": day, "window_days": window_days,
        "percentile_mode": percentile_mode,
        "units": {
            "T2M": "°C", "PRECTOTCORR": "mm/day", "WS10M": "m/s", "RH2M": "%", "HI": "°C HI"
        }
    }

    return {
        "meta": meta,
        "probabilities": results,
        "stats": stats,
        "trends": trends
    }


# ----------------------------------------------------
# 7) Example usage (CLI / quick test)
# ----------------------------------------------------

if __name__ == "_main_":
    # Example: Karachi region (your coords), target July 4th, 2001..2024, ±7-day window
    lat, lon = 24.86, 67.01
    month, day = 7, 4

    # Absolute thresholds mode (default)
    out_abs = analyze_climatology(
        lat=lat, lon=lon,
        month=month, day=day,
        start_year=2001, end_year=2024,
        window_days=7,
        conditions=DEFAULT_CONDITIONS,
        percentile_mode=False
    )

    print("\n=== ABSOLUTE THRESHOLDS ===")
    print(pd.DataFrame(out_abs["probabilities"]))
    print("Stats:", out_abs["stats"])
    print("Meta:", out_abs["meta"])

    # Percentile mode (e.g., 'very_hot' meaning ≥ 90th percentile for that DOY window)
    # Flip percentile_mode=True to compute thresholds from local climatology.
    out_pct = analyze_climatology(
        lat=lat, lon=lon,
        month=month, day=day,
        start_year=2001, end_year=2024,
        window_days=7,
        conditions=[
            # Example: same labels but percentile-driven (90th / 10th)
            ConditionDef("very_hot", "T2M", ">=", 0.0, "°C", percentile_mode=True, percentile_p=0.9),
            ConditionDef("very_cold", "T2M", "<=", 0.0, "°C", percentile_mode=True, percentile_p=0.1),
            ConditionDef("very_wet", "PRECTOTCORR", ">=", 0.0, "mm/day", percentile_mode=True, percentile_p=0.9),
            ConditionDef("very_windy", "WS10M", ">=", 0.0, "m/s", percentile_mode=True, percentile_p=0.9),
            # For HI, you may prefer an absolute comfort threshold rather than percentile
            ConditionDef("uncomfortable", "HI", ">=", 40.0, "°C HI", percentile_mode=False),
        ],
        percentile_mode=True
    )

    print("\n=== PERCENTILE MODE ===")
    print(pd.DataFrame(out_pct["probabilities"]))
    print("Stats:", out_pct["stats"])


def get_status(title: str, value: float, probability: float) -> str:
    title = title.lower()
    if title in ["t2m", "temperature", "temperature at 2m (°c)"]:
        if value >= 35: return "hot"
        elif value <= 10: return "cold"
        return "moderate"
    elif title in ["ws10m", "wind speed", "wind speed 10m (m/s)"]:
        if value >= 8: return "very windy"
        return "normal"
    elif title in ["prectotcorr", "precipitation", "precipitation corrected (mm/day)"]:
        if value >= 10: return "rainy"
        return "dry"
    elif title in ["hi", "heat index"]:
        if value >= 40: return "uncomfortable"
        return "comfortable"
    return "unknown"
