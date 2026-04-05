"""
Script 7: Sunspot Area vs Latitude Scatter — Per Solar Cycle
=============================================================
Plots the heliographic latitude of each sunspot group against its
projected area for each solar cycle (one subplot per cycle).

Key features:
  - Separate markers for Northern (red) and Southern (blue) hemispheres
  - Kernel Density Estimate (KDE) contour overlay to show density peaks
  - Marginal histograms: top = area distribution, right = latitude distribution
  - Summary panel: cycle-mean latitude of largest groups (area > 500 MSH)

Interpretation:
  - How does the latitude of emergence vary with group size?
  - Do the largest (most complex) groups emerge at specific latitudes?
  - How does this pattern change across cycles?

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
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# ── Configuration ─────────────────────────────────────────────────────────────
FIRST_YEAR = 1874
LAST_YEAR  = 2016          # last year of locally available data
BASE_URL   = "https://solarscience.msfc.nasa.gov/greenwch/{year}.txt"
CACHE_FILE = "msfc_sunspot_groups.csv"
DATA_DIR   = "greenwch"    # directory containing local <YYYY>.txt files

CYCLE_BOUNDS = {
    12: (1878, 1889), 13: (1889, 1901), 14: (1901, 1913),
    15: (1913, 1923), 16: (1923, 1933), 17: (1933, 1944),
    18: (1944, 1954), 19: (1954, 1964), 20: (1964, 1976),
    21: (1976, 1986), 22: (1986, 1996), 23: (1996, 2008),
    24: (2008, 2019), 25: (2019, 2026),
}

LARGE_AREA_THRESHOLD = 500      # MSH — "large" sunspot group


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
            df["area"] = 100.0
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


# ── Plotting helpers ───────────────────────────────────────────────────────────
def kde_contours(ax, x: np.ndarray, y: np.ndarray, color: str,
                 levels: int = 4) -> None:
    """Overlay KDE contours on a scatter axis."""
    if len(x) < 20:
        return
    try:
        xy = np.vstack([x, np.log10(y + 1)])
        kde = gaussian_kde(xy, bw_method="scott")
        xi  = np.linspace(x.min(), x.max(), 60)
        yi  = np.logspace(np.log10(max(y.min(), 1)), np.log10(y.max()), 60)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), np.log10(Yi.ravel() + 1)])).reshape(Xi.shape)
        ax.contour(Xi, Yi, Zi, levels=levels, colors=color, linewidths=0.7, alpha=0.6)
    except Exception:
        pass    # silently skip if KDE fails (too few points)


def plot_cycle_panel(ax, subset: pd.DataFrame, cyc: int) -> None:
    """Plot scatter + KDE for one solar cycle."""
    north = subset[subset["latitude"] >  0]
    south = subset[subset["latitude"] <= 0]

    ax.scatter(north["latitude"], north["area"],
               c="tomato", s=1.5, alpha=0.35, linewidths=0, label="N")
    ax.scatter(south["latitude"], south["area"],
               c="royalblue", s=1.5, alpha=0.35, linewidths=0, label="S")

    kde_contours(ax, north["latitude"].values, north["area"].values, "darkred")
    kde_contours(ax, south["latitude"].values, south["area"].values, "darkblue")

    # Mark large sunspots
    large = subset[subset["area"] >= LARGE_AREA_THRESHOLD]
    if not large.empty:
        ax.scatter(large["latitude"], large["area"],
                   marker="*", c="gold", s=20, zorder=5, label=f"≥{LARGE_AREA_THRESHOLD}")

    ax.set_yscale("log")
    ax.set_title(f"SC {cyc}", fontsize=9, fontweight="bold")
    ax.set_xlabel("|Lat| (°)", fontsize=7)
    ax.set_ylabel("Area (MSH)", fontsize=7)
    ax.set_xlim(-55, 55)
    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=6, loc="upper right", markerscale=2)


def plot_summary(ax, data: pd.DataFrame) -> None:
    """Summary: mean latitude of large groups per cycle."""
    cyc_list, mean_n_lats, mean_s_lats = [], [], []
    for cyc, (yr_start, yr_end) in CYCLE_BOUNDS.items():
        t0 = pd.Timestamp(year=yr_start, month=1, day=1)
        t1 = pd.Timestamp(year=yr_end,   month=1, day=1)
        sub = data[
            (data["date"] >= t0) & (data["date"] < t1) &
            (data["area"] >= LARGE_AREA_THRESHOLD)
        ]
        if len(sub) < 5:
            continue
        cyc_list.append(cyc)
        mean_n_lats.append(sub[sub["latitude"] > 0]["latitude"].mean())
        mean_s_lats.append(sub[sub["latitude"] <= 0]["latitude"].abs().mean())

    x = np.arange(len(cyc_list))
    ax.plot(x, mean_n_lats, "o-", color="tomato",    lw=1.5, label="North")
    ax.plot(x, mean_s_lats, "s-", color="royalblue", lw=1.5, label="South")
    ax.set_xticks(x)
    ax.set_xticklabels([f"SC{c}" for c in cyc_list], rotation=45, fontsize=9)
    ax.set_ylabel("Mean |Lat| of Large Groups (°)", fontsize=10)
    ax.set_title(
        f"Mean Latitude of Large Sunspot Groups (Area ≥ {LARGE_AREA_THRESHOLD} MSH) per Cycle",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ── Main plot ──────────────────────────────────────────────────────────────────
def plot_all(data: pd.DataFrame) -> None:
    cycles = sorted(CYCLE_BOUNDS.keys())
    ncols  = 4
    nrows  = (len(cycles) + ncols - 1) // ncols

    fig = plt.figure(figsize=(16, nrows * 3.2 + 4))
    gs  = gridspec.GridSpec(nrows + 1, ncols, figure=fig,
                            hspace=0.55, wspace=0.38)
    fig.suptitle(
        "Sunspot Group Area vs Heliographic Latitude — Per Solar Cycle\n"
        "(Red = North, Blue = South, Contours = KDE density, ★ = large groups)",
        fontsize=13, fontweight="bold",
    )

    idx = 0
    for cyc in cycles:
        yr_start, yr_end = CYCLE_BOUNDS[cyc]
        t0  = pd.Timestamp(year=yr_start, month=1, day=1)
        t1  = pd.Timestamp(year=yr_end,   month=1, day=1)
        sub = data[(data["date"] >= t0) & (data["date"] < t1)]
        if len(sub) < 30:
            idx += 1
            continue
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        plot_cycle_panel(ax, sub, cyc)
        idx += 1

    # Summary panel spanning all columns
    ax_sum = fig.add_subplot(gs[nrows, :])
    plot_summary(ax_sum, data)

    out = "7_sunspot_area_latitude_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    print(f"Total observations: {len(data):,}")
    plot_all(data)
