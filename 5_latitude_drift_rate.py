"""
Script 5: Latitude Drift Rate — Equatorward Migration per Solar Cycle
======================================================================
For each solar cycle, computes the mean heliographic latitude of sunspot
groups in monthly bins, then fits a linear regression to the latitude
centroid over time to quantify the equatorward migration speed
(Spörer's Law).

Outputs:
  - One subplot per cycle showing latitude centroid + linear fit
  - Summary bar chart: drift rate (°/yr) for each cycle
  - Summary bar chart: total latitudinal range traversed per cycle

Data source:
  NASA MSFC combined RGO + USAF/NOAA sunspot group data
  https://solarscience.msfc.nasa.gov/greenwch/<YYYY>.txt
  (shared cache 'msfc_sunspot_groups.csv' from Script 1/2)

Requirements:
  pip install requests pandas matplotlib numpy scipy
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress

# ── Configuration ─────────────────────────────────────────────────────────────
FIRST_YEAR = 1874
LAST_YEAR  = 2016          # last year of locally available data
BASE_URL   = "https://solarscience.msfc.nasa.gov/greenwch/{year}.txt"
CACHE_FILE = "msfc_sunspot_groups.csv"
DATA_DIR   = "greenwch"    # directory containing local <YYYY>.txt files

# Solar cycle number → (minimum year, next minimum year)
# Use mid-cycle activity window to avoid contamination from next cycle
CYCLE_BOUNDS = {
    12: (1878, 1889), 13: (1889, 1901), 14: (1901, 1913),
    15: (1913, 1923), 16: (1923, 1933), 17: (1933, 1944),
    18: (1944, 1954), 19: (1954, 1964), 20: (1964, 1976),
    21: (1976, 1986), 22: (1986, 1996), 23: (1996, 2008),
    24: (2008, 2019), 25: (2019, 2026),
}


# ── Data helpers ──────────────────────────────────────────────────────────────
def download_year(year: int, session: requests.Session) -> pd.DataFrame:
    """Return one year of MSFC sunspot group data as a DataFrame.

    Reads from a local file (DATA_DIR/<year>.txt) when available;
    falls back to downloading from MSFC over the network.
    """
    local_path = os.path.join(DATA_DIR, f"{year}.txt")
    if os.path.exists(local_path):
        with open(local_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        url = BASE_URL.format(year=year)
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            text = r.text
        except Exception:
            return pd.DataFrame()

    rows = []
    for line in text.splitlines():
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
            df["area"] = 100.0   # fallback if cache lacks area
        return df[df["area"] > 0].copy()

    print(f"Reading MSFC data {FIRST_YEAR}–{LAST_YEAR} ...")
    frames, session = [], requests.Session()
    for yr in range(FIRST_YEAR, LAST_YEAR + 1):
        if yr % 10 == 0:
            print(f"  ... {yr}")
        df = download_year(yr, session)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No data found. Place yearly .txt files in '{DATA_DIR}/' "
            "or check your internet connection."
        )

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data[["year", "month", "day"]])
    data = data[data["latitude"].between(-90, 90) & (data["area"] > 0)].copy()
    data.to_csv(CACHE_FILE, index=False)
    return data


# ── Analysis ──────────────────────────────────────────────────────────────────
def compute_cycle_drift(data: pd.DataFrame) -> list[dict]:
    """For each hemisphere separately, fit linear drift within each cycle."""
    results = []
    for cyc, (yr_start, yr_end) in CYCLE_BOUNDS.items():
        t0 = pd.Timestamp(year=yr_start, month=1, day=1)
        t1 = pd.Timestamp(year=yr_end,   month=1, day=1)

        subset = data[(data["date"] >= t0) & (data["date"] < t1)].copy()
        if len(subset) < 50:
            continue

        # Use absolute latitude (fold hemispheres together — Spörer's law is
        # symmetric N & S), weighted by area
        subset["abs_lat"] = subset["latitude"].abs()
        subset["t_yr"] = subset["year"] + (subset["month"] - 1) / 12.0

        # Monthly weighted-mean latitude
        monthly = (
            subset.groupby(["year", "month"])
            .apply(lambda g: np.average(g["abs_lat"], weights=g["area"]))
            .reset_index()
        )
        monthly.columns = ["year", "month", "mean_abs_lat"]
        monthly["t_yr"] = monthly["year"] + (monthly["month"] - 1) / 12.0

        if len(monthly) < 6:
            continue

        slope, intercept, r, p, se = linregress(monthly["t_yr"], monthly["mean_abs_lat"])
        results.append({
            "cycle":     cyc,
            "yr_start":  yr_start,
            "yr_end":    yr_end,
            "slope":     slope,       # degrees per year (negative = equatorward)
            "intercept": intercept,
            "r":         r,
            "p":         p,
            "monthly":   monthly,
            "lat_start": monthly["mean_abs_lat"].iloc[:3].mean(),
            "lat_end":   monthly["mean_abs_lat"].iloc[-3:].mean(),
            "range":     monthly["mean_abs_lat"].iloc[:3].mean() -
                         monthly["mean_abs_lat"].iloc[-3:].mean(),
        })

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_drift_results(results: list[dict]) -> None:
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows + 2, ncols,
                             figsize=(16, nrows * 2.8 + 7))
    axes_flat = axes[:nrows].flatten()
    fig.suptitle(
        "Equatorward Latitude Drift (Spörer's Law) — Per Solar Cycle\n"
        "Weighted Mean |Latitude| with Linear Regression",
        fontsize=13, fontweight="bold",
    )

    for idx, res in enumerate(results):
        ax = axes_flat[idx]
        mo = res["monthly"]
        slope  = res["slope"]
        intcpt = res["intercept"]
        t_fit  = np.linspace(mo["t_yr"].min(), mo["t_yr"].max(), 100)
        lat_fit = slope * t_fit + intcpt

        ax.scatter(mo["t_yr"], mo["mean_abs_lat"],
                   color="steelblue", s=12, alpha=0.7, zorder=3)
        ax.plot(t_fit, lat_fit, color="red", lw=1.5,
                label=f"{slope:.2f} °/yr")
        ax.set_title(f"SC {res['cycle']}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Year", fontsize=7)
        ax.set_ylabel("|Lat| (°)", fontsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 45)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Hide unused subplots in individual panel rows
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # ── Summary: drift rate per cycle ─────────────────────────────────────
    cycles = [r["cycle"] for r in results]
    slopes = [r["slope"] for r in results]
    ranges = [r["range"]  for r in results]

    ax_rate = axes[nrows][0:2].flatten()   # spans 2 columns
    ax_rate = fig.add_axes([0.08, 0.08, 0.40, 0.16])
    colors  = ["tomato" if s < 0 else "steelblue" for s in slopes]
    ax_rate.bar(cycles, slopes, color=colors, edgecolor="black", linewidth=0.5)
    ax_rate.axhline(0, color="black", lw=0.8)
    ax_rate.set_xlabel("Solar Cycle", fontsize=10)
    ax_rate.set_ylabel("Drift Rate (°/yr)", fontsize=10)
    ax_rate.set_title("Equatorward Drift Rate per Cycle\n(negative = equatorward)", fontsize=9)
    ax_rate.set_xticks(cycles)
    ax_rate.grid(True, alpha=0.3, axis="y")

    ax_range = fig.add_axes([0.55, 0.08, 0.40, 0.16])
    ax_range.bar(cycles, ranges, color="mediumseagreen",
                 edgecolor="black", linewidth=0.5)
    ax_range.set_xlabel("Solar Cycle", fontsize=10)
    ax_range.set_ylabel("Latitude Range Traversed (°)", fontsize=10)
    ax_range.set_title("Total Latitudinal Range per Cycle", fontsize=9)
    ax_range.set_xticks(cycles)
    ax_range.grid(True, alpha=0.3, axis="y")

    # Hide built-in summary rows
    for ax in axes[nrows:].flatten():
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.27, 1, 1])
    out = "5_latitude_drift_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n{'Cycle':>6} {'Slope(°/yr)':>12} {'R':>6} {'p':>8} {'Range(°)':>10}")
    print("-" * 48)
    for r in results:
        print(f"SC{r['cycle']:>4}  {r['slope']:>10.3f}  {r['r']:>6.3f}  {r['p']:>8.4f}  {r['range']:>9.2f}")
    print(f"\nFigure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    print(f"Total observations: {len(data):,}")
    results = compute_cycle_drift(data)
    print(f"Cycles analysed: {len(results)}")
    plot_drift_results(results)
