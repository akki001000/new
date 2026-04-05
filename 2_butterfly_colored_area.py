"""
Script 2: Butterfly Diagram Colored by Sunspot Area
=====================================================
Same as the classic butterfly diagram but the colour and size of each
point encodes the sunspot group's projected area (in millionths of the
solar hemisphere, MSH), so the viewer sees both the spatial distribution
AND the intensity of solar activity at once.

Data source:
  NASA Marshall Space Flight Center (MSFC) combined RGO + USAF/NOAA
  sunspot group dataset:
  https://solarscience.msfc.nasa.gov/greenwch/<YYYY>.txt

  Relevant columns (0-based after whitespace split):
    col 0 : year
    col 1 : month
    col 2 : day
    col 5 : Stonyhurst latitude  (degrees; +N / -S)
    col 7 : projected area (MSH) — whole-spot area in col 8 if preferred

Requirements:
  pip install requests pandas matplotlib numpy
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

# ── Configuration ─────────────────────────────────────────────────────────────
FIRST_YEAR = 1874
LAST_YEAR  = 2024
BASE_URL   = "https://solarscience.msfc.nasa.gov/greenwch/{year}.txt"
CACHE_FILE = "msfc_sunspot_groups.csv"      # shared cache with Script 1

CYCLE_MINIMA = {
    12: 1878, 13: 1889, 14: 1901, 15: 1913, 16: 1923,
    17: 1933, 18: 1944, 19: 1954, 20: 1964, 21: 1976,
    22: 1986, 23: 1996, 24: 2008, 25: 2019,
}


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
            area = float(parts[7])    # projected area (MSH)
            if area <= 0:
                continue
            rows.append((yr, mo, da, lat, area))
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["year", "month", "day", "latitude", "area"])


def load_data() -> pd.DataFrame:
    """Load from shared cache (with area column) or re-download."""
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        if "area" in df.columns:
            print(f"Loading cached data from '{CACHE_FILE}' ...")
            return df[df["area"] > 0].copy()
        # Cache lacks area column → re-download
        print("Cache does not contain 'area' column — re-downloading ...")

    print(f"Downloading MSFC sunspot group data {FIRST_YEAR}–{LAST_YEAR} ...")
    frames, session = [], requests.Session()
    for yr in range(FIRST_YEAR, LAST_YEAR + 1):
        if yr % 10 == 0:
            print(f"  ... {yr}")
        df = download_year(yr, session)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded. Check your internet connection.")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data[["year", "month", "day"]])
    data = data[data["latitude"].between(-90, 90) & (data["area"] > 0)].copy()
    data.to_csv(CACHE_FILE, index=False)
    print(f"Saved cache to '{CACHE_FILE}'.")
    return data


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_butterfly_area(data: pd.DataFrame) -> None:
    # Clamp area to a reasonable range to prevent extreme outliers dominating the colormap
    area_min, area_max = 10.0, data["area"].quantile(0.99)
    data = data[data["area"] >= area_min].copy()
    data["area_clamped"] = data["area"].clip(upper=area_max)

    # Point size proportional to log(area)
    sizes = 0.5 + 10 * (np.log10(data["area_clamped"]) - np.log10(area_min)) / \
                        (np.log10(area_max) - np.log10(area_min))

    norm  = LogNorm(vmin=area_min, vmax=area_max)
    cmap  = plt.cm.hot_r        # bright = large area

    fig, ax = plt.subplots(figsize=(18, 7))

    sc = ax.scatter(
        data["date"], data["latitude"],
        c=data["area_clamped"], cmap=cmap, norm=norm,
        s=sizes, alpha=0.55, linewidths=0, rasterized=True,
    )

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax,
        orientation="vertical", pad=0.01, fraction=0.02,
    )
    cbar.set_label("Sunspot Group Projected Area (MSH)", fontsize=11)

    ax.axhline(0, color="cyan", lw=0.9, ls="--", alpha=0.8, label="Equator")

    for cyc, yr in CYCLE_MINIMA.items():
        ts = pd.Timestamp(year=yr, month=1, day=1)
        ax.axvline(ts, color="white", lw=0.6, ls=":", alpha=0.6)
        ax.text(ts, 44, f"SC{cyc}", fontsize=7.5, color="lightgray",
                ha="left", va="bottom", rotation=45)

    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_xlabel("Year", fontsize=13, color="white")
    ax.set_ylabel("Heliographic Latitude (°)", fontsize=13, color="white")
    ax.set_title(
        "Butterfly Diagram — Coloured by Sunspot Group Area\n"
        "(Brighter / Larger Point = Greater Area in MSH)",
        fontsize=14, fontweight="bold", color="white",
    )
    ax.set_ylim(-50, 50)
    ax.set_yticks(range(-40, 41, 10))
    ax.tick_params(colors="white")
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45, ha="right", color="white")
    ax.grid(True, alpha=0.15, lw=0.5, color="gray")
    ax.legend(fontsize=10, labelcolor="white",
              facecolor="#1a1a1a", edgecolor="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    plt.tight_layout()
    out = "2_butterfly_colored_area.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    print(f"Observations with area data: {len(data):,}")
    plot_butterfly_area(data)
