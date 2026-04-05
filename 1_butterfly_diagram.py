"""
Script 1: Butterfly Diagram (Spörer Diagram)
============================================
Classic plot of sunspot heliographic latitude vs time.

Data source:
  NASA Marshall Space Flight Center (MSFC) combined RGO + USAF/NOAA
  sunspot group dataset, available per-year at:
  https://solarscience.msfc.nasa.gov/greenwch/<YYYY>.txt

  Each row = one daily observation of a sunspot group.
  Relevant columns (0-based after splitting whitespace):
    col 0 : year
    col 1 : month
    col 2 : day
    col 5 : latitude  (Stonyhurst heliographic, degrees; +N / -S)

Requirements:
  pip install requests pandas matplotlib numpy
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ─────────────────────────────────────────────────────────────
FIRST_YEAR = 1874          # RGO data starts here
LAST_YEAR  = 2024
BASE_URL   = "https://solarscience.msfc.nasa.gov/greenwch/{year}.txt"
CACHE_FILE = "msfc_sunspot_groups.csv"   # local cache so re-runs are fast

# Solar cycle minima years (for annotation)
CYCLE_MINIMA = {
    12: 1878, 13: 1889, 14: 1901, 15: 1913, 16: 1923,
    17: 1933, 18: 1944, 19: 1954, 20: 1964, 21: 1976,
    22: 1986, 23: 1996, 24: 2008, 25: 2019,
}


# ── Data download & parsing ────────────────────────────────────────────────────
def download_year(year: int, session: requests.Session) -> pd.DataFrame:
    """Download one year of MSFC sunspot group data and return a DataFrame."""
    url = BASE_URL.format(year=year)
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()

    rows = []
    for line in r.text.splitlines():
        parts = line.split()
        # Expect at least 6 whitespace-separated fields; skip header/comments
        if len(parts) < 6:
            continue
        try:
            yr  = int(parts[0])
            mo  = int(parts[1])
            da  = int(parts[2])
            lat = float(parts[5])   # Stonyhurst latitude
            rows.append((yr, mo, da, lat))
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["year", "month", "day", "latitude"])
    return df


def load_data() -> pd.DataFrame:
    """Load group data from local cache or download from MSFC."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from '{CACHE_FILE}' ...")
        return pd.read_csv(CACHE_FILE, parse_dates=["date"])

    print(f"Downloading MSFC sunspot group data {FIRST_YEAR}–{LAST_YEAR} ...")
    frames = []
    session = requests.Session()
    for yr in range(FIRST_YEAR, LAST_YEAR + 1):
        if yr % 10 == 0:
            print(f"  ... {yr}")
        df = download_year(yr, session)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data could be downloaded. Check your internet connection.")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data[["year", "month", "day"]])
    data = data[data["latitude"].between(-90, 90)].copy()
    data.to_csv(CACHE_FILE, index=False)
    print(f"Saved cache to '{CACHE_FILE}'.")
    return data


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_butterfly(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(18, 7))

    ax.scatter(
        data["date"], data["latitude"],
        s=0.8, alpha=0.35, color="steelblue", linewidths=0, rasterized=True,
    )

    ax.axhline(0, color="black", lw=0.8, ls="--", label="Equator")

    # Shade northern / southern activity bands
    ax.axhspan( 5,  35, alpha=0.04, color="tomato")
    ax.axhspan(-35, -5, alpha=0.04, color="royalblue")

    # Annotate cycle numbers and mark minima
    for cyc, yr in CYCLE_MINIMA.items():
        ts = pd.Timestamp(year=yr, month=1, day=1)
        ax.axvline(ts, color="gray", lw=0.6, ls=":", alpha=0.7)
        ax.text(ts, 44, f"SC{cyc}", fontsize=7.5, color="dimgray",
                ha="left", va="bottom", rotation=45)

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Heliographic Latitude (°)", fontsize=13)
    ax.set_title(
        "Butterfly Diagram (Spörer Diagram)\nSunspot Heliographic Latitude vs Time  (1874 – 2024)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(-50, 50)
    ax.set_yticks(range(-40, 41, 10))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    out = "1_butterfly_diagram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    print(f"Total group observations loaded: {len(data):,}")
    plot_butterfly(data)
