"""
Group E – Plot 2: Cross-Correlation Matrix of All Solar Indices
===============================================================
Datasets used:
  - Sunspot Number  : downloaded live from SILSO (sidc.be)
  - F10.7 index     : local file  (columns: year month day f10.7_adj)
  - Total Solar Irr : local file  (columns: year month day tsi)
  - Cosmic Ray Flux : local file  (columns: year month day cr_flux)
  - aa-index        : local file  (columns: year month day aa)

Method:
  All datasets are resampled to a common monthly time grid via linear
  interpolation over their overlapping date range.  Pearson correlation
  coefficients are then computed for every pair and displayed as a
  triangular heatmap with the coefficient values annotated.

Dependencies: numpy, pandas, scipy, matplotlib

Usage:
  1. Set the correct path for each local data file below.
  2. Run:  python group_e2_cross_correlation_matrix.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import urllib.request
import io

# ──────────────────────────────────────────────────────────────────────────────
# DATA FILES  — edit these paths to match your local files
# ──────────────────────────────────────────────────────────────────────────────
F107_FILE   = "f107.txt"
TSI_FILE    = "tsi.txt"
COSMIC_FILE = "cosmic_ray.txt"
AA_FILE     = "aa_index.txt"
# ──────────────────────────────────────────────────────────────────────────────

DATASET_LABELS = ["Sunspot\nNumber", "F10.7\nIndex", "TSI", "Cosmic\nRay", "aa-Index"]
SHORT_LABELS   = ["SSN", "F10.7", "TSI", "CR", "aa"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_decimal_year(df):
    return df["date"].dt.year + (df["date"].dt.dayofyear - 1) / 365.25


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
    print(f"    SSN: {len(df)} months")
    return df.rename(columns={"ssn": "SSN"})


def load_local(filepath, col_name):
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
        .rename(columns={"value": col_name})
    )
    print(f"    {col_name}: {len(monthly)} monthly means")
    return monthly


def align_datasets(series_list):
    """
    Inner-join all monthly series to their common overlapping date range,
    drop rows with any NaN, and return a single aligned DataFrame.
    """
    combined = series_list[0]
    for s in series_list[1:]:
        combined = combined.join(s, how="inner")
    combined = combined.dropna()
    print(f"\n  Common overlap: {combined.index[0].date()} → {combined.index[-1].date()} "
          f"({len(combined)} months)")
    return combined


def compute_corr_matrix(df):
    """
    Returns (corr, pval) DataFrames for all column pairs.
    """
    cols = df.columns.tolist()
    n = len(cols)
    corr_arr = np.ones((n, n))
    pval_arr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r, p = pearsonr(df.iloc[:, i], df.iloc[:, j])
                corr_arr[i, j] = r
                pval_arr[i, j] = p
    return (
        pd.DataFrame(corr_arr, index=cols, columns=cols),
        pd.DataFrame(pval_arr, index=cols, columns=cols),
    )


def plot_corr_matrix(corr_df, pval_df, labels, out_file="cross_correlation_matrix.png"):
    n     = len(labels)
    short = SHORT_LABELS

    fig, ax = plt.subplots(figsize=(8, 7))

    # Full symmetric matrix — use a diverging colourmap centred at 0
    corr_vals = corr_df.values.copy()

    im = ax.imshow(corr_vals, cmap="RdBu_r", vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=11)

    # Annotate every cell
    for i in range(n):
        for j in range(n):
            r = corr_vals[i, j]
            p = pval_df.values[i, j]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            txt = f"{r:.2f}{sig}"
            text_col = "white" if abs(r) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color=text_col, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, fontsize=12)
    ax.set_yticklabels(short, fontsize=12)
    ax.set_title(
        "Cross-Correlation Matrix of Solar Activity Indices\n"
        "(Pearson r; * p<0.05  ** p<0.01  *** p<0.001)",
        fontsize=12, fontweight="bold",
    )

    # Legend for full dataset names
    patches = [
        mpatches.Patch(color="none", label=f"{s} = {l.replace(chr(10), ' ')}")
        for s, l in zip(short, labels)
    ]
    ax.legend(
        handles=patches, loc="upper left", fontsize=8,
        bbox_to_anchor=(1.15, 1.0), title="Index",
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out_file}")
    plt.show()


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Group E – Plot 2: Cross-Correlation Matrix")
    print("=" * 60)

    print("\nLoading datasets …")
    ssn   = load_sunspot()
    f107  = load_local(F107_FILE,   "F10.7")
    tsi   = load_local(TSI_FILE,    "TSI")
    cr    = load_local(COSMIC_FILE, "CR")
    aa    = load_local(AA_FILE,     "aa")

    df = align_datasets([ssn, f107, tsi, cr, aa])

    print("\nComputing Pearson correlations …")
    corr_df, pval_df = compute_corr_matrix(df)

    print("\nCorrelation matrix:\n", corr_df.round(3).to_string())

    plot_corr_matrix(corr_df, pval_df, DATASET_LABELS)


if __name__ == "__main__":
    main()
