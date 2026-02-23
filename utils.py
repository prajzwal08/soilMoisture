from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd
import numpy as np
import xarray as xr


def missing_days_per_year(
    ds: xr.Dataset,
    var: str = "soil_moisture",
    time_dim: str = "date_time",
) -> Dict[str, Dict[int, int]]:
    """
    Count number of missing DAYS (NaNs) per year, per depth.

    Returns: {depth_value: {year: n_missing_days}}
    """
    da = ds[var]
    miss = da.isnull()  # True where missing

    # count missing days per year (per depth)
    n_miss = miss.groupby(f"{time_dim}.year").sum(time_dim).astype(int)  # (year, depth)

    out: Dict[str, Dict[int, int]] = {}
    for d in n_miss["depth"].values:
        depth_key = str(d)
        out[depth_key] = {
            int(y): int(v)
            for y, v in zip(n_miss["year"].values, n_miss.sel(depth=d).values)
        }
    return out


def longest_missing_run_with_dates(
    is_nan: np.ndarray,
    dates: np.ndarray | pd.DatetimeIndex,
) -> Tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Parameters
    ----------
    is_nan : 1D bool array
        True where missing (NaN)
    dates : 1D datetime-like, same length as is_nan

    Returns
    -------
    (length_days, start_date, end_date)
        start/end are None if there is no missing run.
    """
    if is_nan.size == 0 or not is_nan.any():
        return 0, None, None

    dates = pd.to_datetime(dates)

    # pad to catch runs at boundaries
    y = np.r_[False, is_nan, False]

    # transition indices in padded array
    idx = np.flatnonzero(y[1:] != y[:-1])  # pairs: start, end (in padded coords)

    # starts/ends in original coords
    starts = idx[::2]          # start index in original array
    ends_excl = idx[1::2]      # end index (exclusive) in original array

    lens = ends_excl - starts
    k = int(lens.argmax())

    s_i = int(starts[k])
    e_i = int(ends_excl[k] - 1)

    return int(lens[k]), pd.Timestamp(dates[s_i]), pd.Timestamp(dates[e_i])

# longest_gap_dict = {str(d): longest_missing_run_with_dates(result_ds["soil_moisture"].sel(depth=d).isnull().values,
#                                                           result_ds["soil_moisture"].sel(depth=d)["date_time"].values)
#                     for d in result_ds["soil_moisture"]["depth"].values}



def _fill_short_nan_runs(arr: np.ndarray, max_gap_days: int) -> np.ndarray:
    """
    Return a boolean mask 'available' after treating NaN-runs <= max_gap_days as filled.
    arr: 1D float array (daily). NaNs indicate gaps.
    """
    is_nan = np.isnan(arr)
    if is_nan.size == 0:
        return np.array([], dtype=bool)

    # find NaN runs [s, e] inclusive in index space
    y = np.r_[False, is_nan, False]
    idx = np.flatnonzero(y[1:] != y[:-1])
    starts = idx[::2]
    ends_excl = idx[1::2]
    lens = ends_excl - starts

    # available = not NaN, plus NaNs in short runs (<= max_gap_days)
    available = ~is_nan
    for s, e_excl, L in zip(starts, ends_excl, lens):
        if L <= max_gap_days:
            available[s:e_excl] = True
    return available

def longest_available_after_removing_long_gaps(
    ds: xr.Dataset,
    depth_dim: str = "depth",
    time_dim: str = "date_time",
    var: str = "soil_moisture",
    max_gap_days: int = 7,
) -> Dict[str, Tuple[int, pd.Timestamp | None, pd.Timestamp | None]]:
    """
    Treat gaps (NaN-runs) <= max_gap_days as 'removed' (bridged), but keep gaps > max_gap_days.
    Then find the longest consecutive available period per depth.

    Returns: {depth: (length_days, start_date, end_date)}
    """
    da = ds[var]
    dates = pd.to_datetime(da[time_dim].values)

    out: Dict[str, Tuple[int, pd.Timestamp | None, pd.Timestamp | None]] = {}

    for d in da[depth_dim].values:
        x = da.sel({depth_dim: d}).values.astype("float64")
        avail = _fill_short_nan_runs(x, max_gap_days=max_gap_days)

        if avail.size == 0 or not avail.any():
            out[str(d)] = (0, None, None)
            continue

        # longest True-run in avail
        y = np.r_[False, avail, False]
        idx = np.flatnonzero(y[1:] != y[:-1])
        starts = idx[::2]
        ends_excl = idx[1::2]
        lens = ends_excl - starts

        k = int(lens.argmax())
        s_i = int(starts[k])
        e_i = int(ends_excl[k] - 1)

        out[str(d)] = (int(lens[k]), pd.Timestamp(dates[s_i]), pd.Timestamp(dates[e_i]))

    return out



def trim_to_common_continuous_period(
    ds: xr.Dataset,
    longest_available: dict,        # {depth: (len_days, start, end)}
    var: str = "soil_moisture",
    depth_dim: str = "depth",
    time_dim: str = "date_time",
    min_frac: float = 0.95,
) -> xr.Dataset:
    """
    What it does
    -----------
    Uses the output of `longest_available_after_removing_long_gaps(...)` to:

    1) Drop depths that do not have "almost as long" continuous availability as the best depth.
       - "almost as long" is controlled by `min_frac` (e.g. 0.95 = 95%)

    2) Slice the dataset to the time range where ALL kept depths overlap
       (intersection of their longest-available [start, end] periods).

    Parameters
    ----------
    ds:
        Your xarray Dataset (e.g., result_ds) containing `var` with dims (depth, date_time).
    longest_available:
        Dict from your function:
            longest_available[depth] = (L, start, end)
        where:
            L     = length of longest available segment in days (after "removing" short gaps)
            start = start date of that segment (pd.Timestamp) or None
            end   = end date of that segment (pd.Timestamp) or None
    var:
        Variable name inside ds, default "soil_moisture".
    depth_dim:
        Name of depth dimension, default "depth".
    time_dim:
        Name of time dimension, default "date_time".
    min_frac:
        Fraction threshold relative to the best depth.
        Example: 0.95 keeps depths with L >= 0.95 * (maximum L across depths).

    Returns
    -------
    xr.Dataset:
        A dataset that:
          - contains only the selected depths
          - is time-sliced to the overlapping part of their longest-available periods
    """

    # max_len = the best (longest) continuous available length among all depths
    max_len = max(v[0] for v in longest_available.values())

    # keep = list of depths that:
    #  - have a longest segment length >= min_frac * max_len
    #  - have a valid start and end date (not None)
    keep = [
        d for d, (L, s, e) in longest_available.items()
        if (L >= min_frac * max_len) and (L > 0) and (s is not None) and (e is not None)
    ]

    # select only the kept depths in the dataset
    ds2 = ds.sel({depth_dim: keep})

    # intersection of longest-available periods across kept depths:
    # - common start must be the latest of all starts
    # - common end must be the earliest of all ends
    start_common = max(pd.Timestamp(longest_available[d][1]) for d in keep)
    end_common   = min(pd.Timestamp(longest_available[d][2]) for d in keep)

    # slice the dataset to [start_common, end_common]
    # (this ensures all kept depths are within their "best continuous" overlap window)
    return ds2.sel({time_dim: slice(start_common, end_common)})



def trim_to_surface_valid_period_and_keep_well_covered_depths(
    ds: xr.Dataset,
    longest_avail: dict,      # {depth: (L, start, end)}
    surface_depth: str = "0-10",
    var: str = "soil_moisture",
    min_frac: float = 0.95,   # At least 95% of the surface depth's longest available length should be available (time).
    depth_dim: str = "depth",
    time_dim: str = "date_time",
) -> xr.Dataset:
    # 1) reference window from surface depth (after removing long gaps)
    L, start, end = longest_avail[surface_depth]
    t = ds[time_dim]
    ds_ref = ds.sel({time_dim: (t >= start) & (t < end)})
    # ds_ref = ds.sel({time_dim: slice(start, end)}) # this adds nan to end. 

    # 2) coverage per depth within that window (raw non-NaN)
    frac = ds_ref[var].notnull().mean(time_dim)  # (depth,)

    keep_depths = frac[depth_dim].where(frac >= min_frac, drop=True)
    return ds_ref.sel({depth_dim: keep_depths})

# usage:
# longest_avail = longest_available_after_removing_long_gaps(trial_ds, max_gap_days=7)
# clean_ds = slice_to_surface_window_and_keep_depths(trial_ds, longest_avail, surface_depth="0-5", min_frac=0.10)


def gapfill_by_monthday_mean_with_feb29_fallback(
    ds: xr.Dataset,
    var: str = "soil_moisture",
    time_dim: str = "date_time",
    flag_name: str = "soil_moisture_qc",
    keep_original_flag: bool = True,
) -> xr.Dataset:
    """
    Gap-fill NaNs using mean for same month-day across years (per depth).
    Feb-29 fallback: use Feb-28 (else Mar-01) if 02-29 climatology is unavailable.

    Flag values:
      0 = original observed
      1 = gap-filled
      2 = still missing
    """
    da = ds[var]
    orig_nan = da.isnull()

    md = da[time_dim].dt.strftime("%m-%d")                 # e.g., "02-28"
    clim = da.groupby(md).mean(time_dim, skipna=True)      # dims: (strftime, depth)

    # Feb-29 fallback: if "02-29" exists but is all-NaN, replace with nearest (02-28 then 03-01)
    if "02-29" in clim[md.name].values:
        feb29 = clim.sel({md.name: "02-29"})
        if bool(feb29.isnull().all().item()):
            repl = clim.sel({md.name: "02-28"})
            if bool(repl.isnull().all().item()) and ("03-01" in clim[md.name].values):
                repl = repl.fillna(clim.sel({md.name: "03-01"}))
            clim.loc[{md.name: "02-29"}] = repl

    fill_vals = clim.sel({md.name: md})                    # aligns to (date_time, depth)
    filled = da.where(~orig_nan, fill_vals)

    still_nan = filled.isnull()
    flag = xr.zeros_like(da, dtype=np.int8)
    flag = flag.where(orig_nan, other=0)       # observed -> 0
    flag = flag.where(~orig_nan, other=1)      # filled -> 1 (tentative)
    flag = flag.where(~still_nan, other=2)     # still missing -> 2

    out = ds.copy()
    out[var] = filled

    if keep_original_flag and (flag_name in out):
        out[f"{flag_name}_gapfill"] = flag
    else:
        out[flag_name] = flag

    return out