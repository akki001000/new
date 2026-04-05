"""
Group E – Plot 1: Heatmap of Dominant Periods Across All Solar Datasets
=======================================================================
Datasets used:
  - Sunspot Number  : downloaded live from SILSO (sidc.be)
  - F10.7 index     : local file  (columns: year month day f10.7_adj)
  - Total Solar Irr : local file  (columns: year month day tsi)
  - Cosmic Ray Flux : local file  (columns: year month day cr_flux)
  - aa-index        : local file  (columns: year month day aa)

Method: Lomb-Scargle periodogram is used for all datasets so that
        unevenly-sampled or gapped data is handled correctly.
        The top-N dominant periods for each dataset are collected and
        displayed as a colour-coded heatmap (power at each period band).

Dependencies: numpy, scipy, pandas, matplotlib, astropy (pip install astropy)

Usage:
  1. Set the correct path for each local data file in the DATA FILES section.
  2. Run:  python group_e1_dominant_periods_heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
import urllib.request
import io

# ──────────────────────────────────────────────────────────────────────────────
# DATA FILES  — edit these paths to match your local files
# ──────────────────────────────────────────────────────────────────────────────
F107_FILE   = "f107.txt"          # columns: year  month  day  f107_adj
TSI_FILE    = "tsi.txt"           # columns: year  month  day  tsi
COSMIC_FILE = "cosmic_ray.txt"    # columns: year  month  day  cr_flux
AA_FILE     = "aa_index.txt"      # columns: year  month  day  aa
# ──────────────────────────────────────────────────────────────────────────────


# ── Period bands (years) that are physically meaningful in solar physics ──────
PERIOD_BANDS = {
    "~27-day\n(rotation)":    (0.060,  0.090),
    "~150-day\n(Rieger)":     (0.35,   0.50),
    "~1-2 yr\n(QBO)":         (1.0,    2.5),
    "~5-6 yr\n(mid-term)":    (4.5,    6.5),
    "~11 yr\n(Schwabe)":      (9.0,   13.0),
    "~22 yr\n(Hale)":         (18.0,  26.0),
    "~80 yr\n(Gleissberg)":   (70.0,  95.0),
}

DATASET_NAMES = ["Sunspot\nNumber", "F10.7\nIndex", "TSI", "Cosmic\nRay", "aa-Index"]


# ── Helper: fractional year → decimal ────────────────────────────────────────
def to_decimal_year(df, date_col="date"):
    return df[date_col].dt.year + (df[date_col].dt.dayofyear - 1) / 365.25


# ── 1. Load Sunspot Number from SILSO (monthly means) ─────────────────────────
def load_sunspot():
    url = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    print("Downloading SILSO monthly sunspot data …")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(
            io.StringIO(raw),
            sep=";",
            header=None,
            usecols=[0, 1, 3],
            names=["year", "month", "ssn"],
        )
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
        )
        df = df.dropna(subset=["ssn"])
        df = df[df["ssn"] >= 0]
        t = to_decimal_year(df)
        y = df["ssn"].values.astype(float)
        print(f"  Sunspot: {len(df)} monthly values ({df['year'].min()}–{df['year'].max()})")
        return t.values, y
    except Exception as exc:
        raise RuntimeError(f"Could not download sunspot data: {exc}")


# ── 2. Load a generic 4-column local file ─────────────────────────────────────
def load_local(filepath, value_col=3, sep=None):
    """
    Expects columns: year  month  day  value
    Lines starting with # are treated as comments.
    """
    df = pd.read_csv(
        filepath,
        sep=sep,
        header=None,
        comment="#",
        delim_whitespace=(sep is None),
        names=["year", "month", "day", "value"],
        usecols=[0, 1, 2, value_col],
    )
    df = df.dropna()
    df = df[df["value"] > 0]
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].astype(int).rename(
            columns={"year": "year", "month": "month", "day": "day"}
        )
    )
    t = to_decimal_year(df)
    y = df["value"].values.astype(float)
    print(f"  {filepath}: {len(df)} values ({df['year'].min():.0f}–{df['year'].max():.0f})")
    return t.values, y


# ── 3. Compute Lomb-Scargle power in each period band ─────────────────────────
def ls_band_power(t, y, band_min_yr, band_max_yr, n_grid=500):
    """
    Returns the maximum LS normalised power within [band_min_yr, band_max_yr].
    t : time in decimal years
    y : signal (mean-subtracted internally)
    """
    y = y - np.mean(y)
    freq_max = 1.0 / band_min_yr
    freq_min = 1.0 / band_max_yr
    frequency = np.linspace(freq_min, freq_max, n_grid)
    ls = LombScargle(t, y)
    power = ls.power(frequency)
    return float(np.max(power))


# ── 4. Build the power matrix  [n_datasets × n_period_bands] ──────────────────
def build_power_matrix(datasets):
    """
    datasets : list of (t_array, y_array) tuples, one per dataset
    Returns  : numpy array shape (n_datasets, n_bands)
               values are LS normalised power (0–1 range approximately)
    """
    band_labels = list(PERIOD_BANDS.keys())
    n_ds = len(datasets)
    n_b  = len(band_labels)
    mat  = np.zeros((n_ds, n_b))

    for i, (t, y) in enumerate(datasets):
        for j, (bl, (bmin, bmax)) in enumerate(PERIOD_BANDS.items()):
            t_span = t.max() - t.min()
            if t_span < bmin * 0.5:
                mat[i, j] = np.nan   # dataset too short for this period
            else:
                mat[i, j] = ls_band_power(t, y, bmin, bmax)
        print(f"  Computed LS powers for dataset {i+1}/{n_ds}")

    return mat, band_labels


# ── 5. Plot heatmap ────────────────────────────────────────────────────────────
def plot_heatmap(mat, band_labels, dataset_labels, out_file="heatmap_dominant_periods.png"):
    fig, ax = plt.subplots(figsize=(13, 5))

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=np.nanmax(mat),
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Lomb-Scargle Normalised Power", fontsize=11)

    # Annotate each cell with the power value
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                txt = "N/A"
                col = "grey"
            else:
                txt = f"{val:.3f}"
                col = "black" if val < 0.6 * np.nanmax(mat) else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)

    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels, fontsize=10)
    ax.set_yticks(range(len(dataset_labels)))
    ax.set_yticklabels(dataset_labels, fontsize=11)
    ax.set_xlabel("Period Band", fontsize=12)
    ax.set_title(
        "Heatmap of Dominant Periods Across Solar Activity Datasets\n"
        "(Lomb-Scargle Normalised Power)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out_file}")
    plt.show()


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Group E – Plot 1: Dominant Period Heatmap")
    print("=" * 60)

    # Load datasets
    datasets = []

    print("\n[1/5] Loading Sunspot Number …")
    datasets.append(load_sunspot())

    print("\n[2/5] Loading F10.7 …")
    datasets.append(load_local(F107_FILE))

    print("\n[3/5] Loading TSI …")
    datasets.append(load_local(TSI_FILE))

    print("\n[4/5] Loading Cosmic Ray Flux …")
    datasets.append(load_local(COSMIC_FILE))

    print("\n[5/5] Loading aa-Index …")
    datasets.append(load_local(AA_FILE))

    # Build power matrix
    print("\nComputing Lomb-Scargle power in each period band …")
    mat, band_labels = build_power_matrix(datasets)

    # Plot
    plot_heatmap(mat, band_labels, DATASET_NAMES)


if __name__ == "__main__":
    main()
