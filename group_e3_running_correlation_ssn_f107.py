"""
Group E – Plot 3: Running Correlation — Sunspot Number vs F10.7 Over Time
=========================================================================
Shows how the correlation between the monthly sunspot number and the
F10.7 solar radio flux index evolves across different epochs of the
solar record, using a sliding window of configurable width.

Datasets used:
  - Sunspot Number : downloaded live from SILSO (sidc.be)
  - F10.7 index    : local file  (columns: year month day f10.7_adj)

Method:
  A sliding window (default 36 months = 3 years) is moved along the
  overlapping time series one step at a time.  At each position, the
  Pearson correlation coefficient between SSN and F10.7 within that
  window is stored.  The result is plotted together with the sunspot
  number time series to contextualise any changes.

Dependencies: numpy, pandas, scipy, matplotlib

Usage:
  1. Set the correct path for the F10.7 file below.
  2. Optionally adjust WINDOW_MONTHS.
  3. Run:  python group_e3_running_correlation_ssn_f107.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import urllib.request
import io

# ──────────────────────────────────────────────────────────────────────────────
# DATA FILE  — edit this path to match your local F10.7 file
# ──────────────────────────────────────────────────────────────────────────────
F107_FILE = "f107.txt"   # columns: year  month  day  f107_adj

# ── Analysis parameter ────────────────────────────────────────────────────────
WINDOW_MONTHS = 36   # width of the sliding window (months)
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_sunspot():
    url = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    print("  Downloading SILSO monthly sunspot data …")
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(
        io.StringIO(raw), sep=";", header=None,
        usecols=[0, 1, 3], names=["year", "month", "ssn"],
    )
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )
    df = df.dropna(subset=["ssn"])
    df = df[df["ssn"] >= 0].set_index("date")[["ssn"]]
    df.index = df.index.to_period("M").to_timestamp()
    print(f"    SSN: {len(df)} months ({df.index[0].year}–{df.index[-1].year})")
    return df


def load_f107(filepath):
    df = pd.read_csv(
        filepath, header=None, comment="#", delim_whitespace=True,
        names=["year", "month", "day", "value"], usecols=[0, 1, 2, 3],
    )
    df = df.dropna()
    df = df[df["value"] > 0]
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].astype(int).rename(
            columns={"year": "year", "month": "month", "day": "day"}
        )
    )
    monthly = (
        df.set_index("date")[["value"]]
        .resample("MS")
        .mean()
        .rename(columns={"value": "f107"})
    )
    print(f"    F10.7: {len(monthly)} months ({monthly.index[0].year}–{monthly.index[-1].year})")
    return monthly


def running_correlation(series_a, series_b, window):
    """
    Compute Pearson r between series_a and series_b in a sliding window.

    Parameters
    ----------
    series_a, series_b : pd.Series with identical DatetimeIndex
    window             : int, window width in months

    Returns
    -------
    dates : array of window-centre dates
    r_arr : array of Pearson r values
    p_arr : array of p-values
    """
    n = len(series_a)
    dates = []
    r_arr = []
    p_arr = []

    for start in range(0, n - window + 1):
        end = start + window
        win_a = series_a.iloc[start:end].values
        win_b = series_b.iloc[start:end].values

        # Skip window if either series has constant value (r undefined)
        if win_a.std() == 0 or win_b.std() == 0:
            continue

        r, p = pearsonr(win_a, win_b)
        centre_idx = start + window // 2
        dates.append(series_a.index[centre_idx])
        r_arr.append(r)
        p_arr.append(p)

    return np.array(dates, dtype="datetime64[ns]"), np.array(r_arr), np.array(p_arr)


def solar_cycle_boundaries():
    """
    Approximate solar minimum years for cycles 19–25 for reference lines.
    Source: NOAA / SIDC cycle records.
    """
    return {
        "SC19→20": 1954,
        "SC20→21": 1964,
        "SC21→22": 1976,
        "SC22→23": 1986,
        "SC23→24": 1996,
        "SC24→25": 2008,
        "SC25→26": 2019,
    }


# ── Main plot ─────────────────────────────────────────────────────────────────
def plot_running_correlation(df, window, out_file="running_correlation_ssn_f107.png"):
    ssn  = df["ssn"]
    f107 = df["f107"]

    dates, r_arr, p_arr = running_correlation(ssn, f107, window)

    # Significance mask  (p < 0.05)
    sig = p_arr < 0.05

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.08)

    # ── Top panel: SSN + F10.7 time series ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1r = ax1.twinx()

    l1, = ax1.plot(ssn.index,  ssn.values,  color="royalblue", lw=0.8,
                   alpha=0.7, label="Sunspot Number (SSN)")
    l2, = ax1r.plot(f107.index, f107.values, color="darkorange", lw=0.8,
                    alpha=0.8, label="F10.7 Index")

    ax1.set_ylabel("Sunspot Number", color="royalblue", fontsize=10)
    ax1r.set_ylabel("F10.7 (sfu)", color="darkorange",  fontsize=10)
    ax1.tick_params(axis="y", colors="royalblue")
    ax1r.tick_params(axis="y", colors="darkorange")
    ax1.set_xticklabels([])
    ax1.set_xlim(pd.Timestamp(dates[0]) - pd.DateOffset(months=window),
                 pd.Timestamp(dates[-1]) + pd.DateOffset(months=window))

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    ax1.set_title(
        f"Running Correlation between Sunspot Number and F10.7 Index\n"
        f"(Sliding window = {window} months)",
        fontsize=13, fontweight="bold",
    )

    # Solar cycle boundary lines
    for label, yr in solar_cycle_boundaries().items():
        ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey",
                    lw=0.6, ls="--", alpha=0.5)

    # ── Bottom panel: running Pearson r ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Shade by significance
    ax2.fill_between(
        pd.to_datetime(dates), r_arr,
        where=sig,  color="steelblue", alpha=0.30, label="p < 0.05 (significant)"
    )
    ax2.fill_between(
        pd.to_datetime(dates), r_arr,
        where=~sig, color="lightgrey", alpha=0.50, label="p ≥ 0.05 (not significant)"
    )
    ax2.plot(pd.to_datetime(dates), r_arr, color="navy", lw=1.2)

    # Reference lines
    ax2.axhline(0,    color="black", lw=0.8, ls="-")
    ax2.axhline(0.7,  color="green", lw=0.8, ls="--", alpha=0.6, label="r = +0.70")
    ax2.axhline(-0.7, color="red",   lw=0.8, ls="--", alpha=0.6, label="r = −0.70")
    ax2.axhline(1.0,  color="grey",  lw=0.4, ls=":")
    ax2.axhline(-1.0, color="grey",  lw=0.4, ls=":")

    for label, yr in solar_cycle_boundaries().items():
        vl = ax2.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey",
                         lw=0.6, ls="--", alpha=0.5)
        ax2.text(
            pd.Timestamp(f"{yr}-01-01"), -0.95,
            label, fontsize=6, rotation=90, va="bottom", ha="right",
            color="grey", alpha=0.8,
        )

    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel("Pearson r", fontsize=11)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.legend(loc="lower right", fontsize=8, ncol=2)

    # Format x-axis
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out_file}")
    plt.show()


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Group E – Plot 3: Running Correlation (SSN vs F10.7)")
    print("=" * 60)

    print("\nLoading datasets …")
    ssn  = load_sunspot()
    f107 = load_f107(F107_FILE)

    # Align to common monthly grid
    df = ssn.join(f107, how="inner").dropna()
    print(f"\n  Common overlap: {df.index[0].date()} → {df.index[-1].date()} "
          f"({len(df)} months)")

    if len(df) < WINDOW_MONTHS * 2:
        raise ValueError(
            f"Only {len(df)} overlapping months — increase overlap or reduce WINDOW_MONTHS "
            f"(currently {WINDOW_MONTHS})."
        )

    plot_running_correlation(df, WINDOW_MONTHS)

    # ── Print summary statistics ──────────────────────────────────────────────
    _, r_all, _ = running_correlation(df["ssn"], df["f107"], WINDOW_MONTHS)
    print("\n── Summary of Running Correlation Values ──────────────────")
    print(f"  Window width : {WINDOW_MONTHS} months")
    print(f"  Mean r       : {np.mean(r_all):.3f}")
    print(f"  Std  r       : {np.std(r_all):.3f}")
    print(f"  Min  r       : {np.min(r_all):.3f}  (weakest coupling)")
    print(f"  Max  r       : {np.max(r_all):.3f}  (strongest coupling)")
    print(f"  Fraction with |r| > 0.7 : {np.mean(np.abs(r_all) > 0.7)*100:.1f}%")


if __name__ == "__main__":
    main()
