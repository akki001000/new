"""
Script 6: Active Latitude Band Width per Solar Cycle
=====================================================
For each solar cycle, computes the time-evolution of the **width** of the
active latitude band (separately for each hemisphere), defined as:

    width(t) = 90th‐percentile latitude − 10th‐percentile latitude

(of the area-weighted distribution of |latitude| within that time bin)

Then plots:
  - Band-width time series for N and S hemispheres for every cycle
  - Summary bar chart: mean band-width per cycle and hemisphere
  - Overlay: whether band width narrows over the cycle (Spörer-consistent)

A narrowing band width with cycle phase is a key indicator of Spörer's Law.

Data source:
  NASA MSFC combined RGO + USAF/NOAA sunspot group data
  https://solarscience.msfc.nasa.gov/greenwch/<YYYY>.txt
  (cached in 'msfc_sunspot_groups.csv')

Requirements:
  pip install requests pandas matplotlib numpy scipy
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

# ── Configuration ─────────────────────────────────────────────────────────────
FIRST_YEAR = 1874
LAST_YEAR  = 2024
BASE_URL   = "https://solarscience.msfc.nasa.gov/greenwch/{year}.txt"
CACHE_FILE = "msfc_sunspot_groups.csv"

CYCLE_BOUNDS = {
    12: (1878, 1889), 13: (1889, 1901), 14: (1901, 1913),
    15: (1913, 1923), 16: (1923, 1933), 17: (1933, 1944),
    18: (1944, 1954), 19: (1954, 1964), 20: (1964, 1976),
    21: (1976, 1986), 22: (1986, 1996), 23: (1996, 2008),
    24: (2008, 2019), 25: (2019, 2026),
}
BIN_MONTHS = 3       # width of temporal bin in months


# ── Data helpers ──────────────────────────────────────────────────────────────
def download_year(year: int, session: requests.Session) -> pd.DataFrame:
    url = BASE_URL.format(year=year)
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()

    rows = []
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            yr   = int(parts[0])
            mo   = int(parts[1])
            da   = int(parts[2])
            lat  = float(parts[5])
            area = float(parts[7])
            if area <= 0:
                continue
            rows.append((yr, mo, da, lat, area))
        except (ValueError, IndexError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["year", "month", "day", "latitude", "area"])


def load_data() -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        print(f"Loading cache from '{CACHE_FILE}' ...")
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        if "area" not in df.columns:
            df["area"] = 100.0
        return df[df["area"] > 0].copy()

    print(f"Downloading MSFC data {FIRST_YEAR}–{LAST_YEAR} ...")
    frames, session = [], requests.Session()
    for yr in range(FIRST_YEAR, LAST_YEAR + 1):
        if yr % 10 == 0:
            print(f"  ... {yr}")
        df = download_year(yr, session)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No data downloaded.")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data[["year", "month", "day"]])
    data = data[data["latitude"].between(-90, 90) & (data["area"] > 0)].copy()
    data.to_csv(CACHE_FILE, index=False)
    return data


# ── Analysis ──────────────────────────────────────────────────────────────────
def weighted_percentile(values: np.ndarray, weights: np.ndarray,
                         perc: float) -> float:
    """Compute the weighted percentile of values."""
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cumw = np.cumsum(w_sorted)
    cumw /= cumw[-1]
    return float(np.interp(perc / 100.0, cumw, v_sorted))


def band_width_series(subset: pd.DataFrame) -> pd.DataFrame:
    """
    For a single-cycle DataFrame, compute quarterly band width for N and S.
    Returns DataFrame with columns: t_yr, width_N, width_S.
    """
    subset = subset.copy()
    subset["t_yr"] = subset["year"] + (subset["month"] - 1) / 12.0
    # Quarterly bins aligned to cycle start
    t_start = subset["t_yr"].min()
    bin_size = BIN_MONTHS / 12.0
    subset["bin"] = ((subset["t_yr"] - t_start) / bin_size).astype(int)

    rows = []
    for b, grp in subset.groupby("bin"):
        t_mid = t_start + (b + 0.5) * bin_size
        grp_n = grp[grp["latitude"] >  0]
        grp_s = grp[grp["latitude"] <= 0]

        def _width(g):
            if len(g) < 5:
                return np.nan
            lats    = g["latitude"].abs().values
            weights = g["area"].values
            p90     = weighted_percentile(lats, weights, 90)
            p10     = weighted_percentile(lats, weights, 10)
            return p90 - p10

        rows.append((t_mid, _width(grp_n), _width(grp_s)))

    return pd.DataFrame(rows, columns=["t_yr", "width_N", "width_S"])


def compute_all_band_widths(data: pd.DataFrame) -> dict:
    results = {}
    for cyc, (yr_start, yr_end) in CYCLE_BOUNDS.items():
        t0 = pd.Timestamp(year=yr_start, month=1, day=1)
        t1 = pd.Timestamp(year=yr_end,   month=1, day=1)
        subset = data[(data["date"] >= t0) & (data["date"] < t1)]
        if len(subset) < 50:
            continue
        bw = band_width_series(subset)
        results[cyc] = bw
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_band_widths(all_bw: dict) -> None:
    n = len(all_bw)
    ncols = 4
    nrows_panels = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(16, nrows_panels * 2.8 + 5))
    gs  = fig.add_gridspec(nrows_panels + 1, ncols, hspace=0.55, wspace=0.35)
    fig.suptitle(
        "Active Latitude Band Width per Solar Cycle\n"
        f"(90th − 10th percentile of area-weighted |latitude|, "
        f"{BIN_MONTHS}-month bins)",
        fontsize=13, fontweight="bold",
    )

    mean_n, mean_s, cycles = [], [], []
    for idx, (cyc, bw) in enumerate(sorted(all_bw.items())):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        t_phase = bw["t_yr"] - bw["t_yr"].min()    # years since cycle start
        ax.plot(t_phase, bw["width_N"], color="tomato",     lw=1.4, label="N")
        ax.plot(t_phase, bw["width_S"], color="royalblue",  lw=1.4, label="S")

        # Trend lines
        for col_data, color in [("width_N", "darkred"), ("width_S", "darkblue")]:
            y = bw[col_data].dropna()
            x = t_phase[y.index]
            if len(x) >= 4:
                slope, intercept, *_ = linregress(x, y)
                ax.plot(x, intercept + slope * x, color=color,
                        lw=1.0, ls="--", alpha=0.7)

        ax.set_title(f"SC {cyc}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Years since minimum", fontsize=7)
        ax.set_ylabel("Width (°)", fontsize=7)
        ax.set_ylim(0, 40)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        cycles.append(cyc)
        mean_n.append(bw["width_N"].mean())
        mean_s.append(bw["width_S"].mean())

    # ── Summary bar chart ─────────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[nrows_panels, :])
    x = np.arange(len(cycles))
    ax_sum.bar(x - 0.2, mean_n, width=0.38, color="tomato",
               label="Northern hemisphere", edgecolor="black", lw=0.5)
    ax_sum.bar(x + 0.2, mean_s, width=0.38, color="royalblue",
               label="Southern hemisphere", edgecolor="black", lw=0.5)
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels([f"SC{c}" for c in cycles], fontsize=9, rotation=45)
    ax_sum.set_ylabel("Mean Band Width (°)", fontsize=11)
    ax_sum.set_title(
        "Mean Active Latitude Band Width per Cycle and Hemisphere", fontsize=10)
    ax_sum.legend(fontsize=9)
    ax_sum.grid(True, alpha=0.3, axis="y")

    out = "6_active_latitude_bandwidth.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    print(f"Total observations: {len(data):,}")
    all_bw = compute_all_band_widths(data)
    print(f"Cycles analysed: {len(all_bw)}")
    plot_band_widths(all_bw)
