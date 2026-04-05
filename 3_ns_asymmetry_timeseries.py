"""
Script 3: North–South Hemispheric Asymmetry Time Series
========================================================
Computes and plots the normalised North–South asymmetry index:

    A(t) = [ N(t) − S(t) ] / [ N(t) + S(t) ]

where N and S are the monthly hemispheric sunspot numbers from SILSO.

A > 0  → northern hemisphere more active
A < 0  → southern hemisphere more active

Data source (automatic download, no registration needed):
  SILSO World Data Center — monthly hemispheric sunspot numbers V2
  https://www.sidc.be/SILSO/DATA/SN_hem_m_tot_V2.0.txt

  File columns (space-separated):
    0 : Year
    1 : Month
    2 : Year.fraction (decimal date, e.g. 1992.04)
    3 : SN_North (monthly mean)
    4 : SN_South (monthly mean)
    5 : SN_Total
    6 : SN_North std-dev  (or -1 if unavailable)
    7 : SN_South std-dev
    8 : SN_Total std-dev
    9 : Number of observations
   10 : Provisional marker (1=provisional)

Requirements:
  pip install requests pandas matplotlib numpy
"""

import io
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ─────────────────────────────────────────────────────────────
SILSO_HEM_URL   = (
    "https://www.sidc.be/SILSO/DATA/SN_hem_m_tot_V2.0.txt"
)
SILSO_LOCAL_FILE = "SN_hem_m_tot_V2.0.txt"   # local copy of the SILSO file
SMOOTH_MONTHS = 13   # running mean for the smoothed overlay

CYCLE_MINIMA = {
    12: 1878, 13: 1889, 14: 1901, 15: 1913, 16: 1923,
    17: 1933, 18: 1944, 19: 1954, 20: 1964, 21: 1976,
    22: 1986, 23: 1996, 24: 2008, 25: 2019,
}


# ── Data helpers ──────────────────────────────────────────────────────────────
def download_silso_hemispheric() -> pd.DataFrame:
    """Load and parse SILSO monthly hemispheric sunspot numbers.

    Reads from a local file (SILSO_LOCAL_FILE) when available;
    falls back to downloading from the SILSO server.
    """
    if os.path.exists(SILSO_LOCAL_FILE):
        print(f"Loading SILSO hemispheric data from local file: {SILSO_LOCAL_FILE}")
        with open(SILSO_LOCAL_FILE, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        print(f"Downloading SILSO hemispheric data from:\n  {SILSO_HEM_URL}")
        try:
            r = requests.get(SILSO_HEM_URL, timeout=30)
            r.raise_for_status()
            text = r.text
        except Exception as e:
            raise RuntimeError(
                f"Download failed: {e}\n"
                f"Alternatively, place '{SILSO_LOCAL_FILE}' in the working directory."
            )

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            sn_n  = float(parts[3])
            sn_s  = float(parts[4])
            total = float(parts[5])
            # Reject fill values (-1 used for missing data in SILSO)
            if sn_n < 0 or sn_s < 0:
                continue
            rows.append((year, month, sn_n, sn_s, total))
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows, columns=["year", "month", "SN_N", "SN_S", "SN_total"])
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=15))
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} monthly records ({df['year'].min()}–{df['year'].max()}).")
    return df


def compute_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised N–S asymmetry column; avoid division by zero."""
    denom = df["SN_N"] + df["SN_S"]
    df = df.copy()
    df["asymmetry"] = np.where(
        denom > 0,
        (df["SN_N"] - df["SN_S"]) / denom,
        np.nan,
    )
    # 13-month running mean (centred)
    df["asym_smooth"] = (
        df["asymmetry"]
        .rolling(SMOOTH_MONTHS, center=True, min_periods=6)
        .mean()
    )
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_ns_asymmetry(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12),
        gridspec_kw={"height_ratios": [2, 2, 1.8]},
        sharex=True,
    )

    # ── Panel 1: Raw hemispheric sunspot numbers ──────────────────────────
    ax = axes[0]
    ax.fill_between(df["date"], df["SN_N"], alpha=0.50,
                    color="tomato", label="Northern hemisphere (N)")
    ax.fill_between(df["date"], df["SN_S"], alpha=0.50,
                    color="royalblue", label="Southern hemisphere (S)")
    ax.set_ylabel("Monthly Sunspot Number", fontsize=11)
    ax.set_title(
        "North & South Hemispheric Sunspot Numbers and N–S Asymmetry",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Raw asymmetry index ──────────────────────────────────────
    ax = axes[1]
    ax.bar(df["date"], df["asymmetry"],
           width=20, color=np.where(df["asymmetry"] >= 0, "tomato", "royalblue"),
           alpha=0.55, label="Monthly A(t)")
    ax.plot(df["date"], df["asym_smooth"],
            color="black", lw=1.6, label=f"{SMOOTH_MONTHS}-month smooth")
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_ylabel("Asymmetry  A = (N−S)/(N+S)", fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Cumulative asymmetry ────────────────────────────────────
    ax = axes[2]
    cum = df["asymmetry"].cumsum()
    ax.fill_between(df["date"], cum, alpha=0.4, color="darkgreen")
    ax.plot(df["date"], cum, color="darkgreen", lw=1.5, label="Cumulative A(t)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Cumulative Asymmetry", fontsize=11)
    ax.set_xlabel("Year", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate cycle minima on all panels
    for cyc, yr in CYCLE_MINIMA.items():
        ts = pd.Timestamp(year=yr, month=1, day=1)
        for a in axes:
            a.axvline(ts, color="gray", lw=0.6, ls=":", alpha=0.7)
        axes[0].text(ts, axes[0].get_ylim()[1] * 0.92,
                     f"SC{cyc}", fontsize=7, color="dimgray",
                     ha="left", va="top", rotation=45)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out = "3_ns_asymmetry_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = download_silso_hemispheric()
    df = compute_asymmetry(df)
    plot_ns_asymmetry(df)
