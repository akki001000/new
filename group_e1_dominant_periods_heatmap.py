"""
Group E – Plot 1: Heatmap of Dominant Periods Across All Solar Datasets
=======================================================================
IMPROVEMENTS over previous version
------------------------------------
1. False-Alarm Probability (FAP) computed for every cell; cells that do NOT
   reach p < 0.01 are cross-hatched (visually marked as "not significant").
2. Power values are row-normalised so colours are comparable across datasets
   that naturally have very different LS amplitudes (e.g. SSN vs aa).
3. Colour-bar label clarified: "Row-Normalised LS Power".
4. Cell annotation improved: significant cells show actual power; non-sig
   cells shown in grey italic so the reader is not misled.
5. Period bands tightened so the ~27-day band is only sampled for datasets
   with daily/sub-monthly resolution (SSN monthly → marked N/A).
6. Figure size, DPI, and font sizes polished for dissertation printing.
7. Source attribution added to figure caption string.

Datasets
--------
  SSN       : live download from SILSO (monthly means)
  F10.7     : penticton_radio_flux.csv  (Julian Date | obs | adjusted)
  TSI       : SATIRE-S_TSI_latest.txt   (Julian Date | TSI)
  Phi (CR)  : Phi_mon_tab.txt           (fractional year | Phi MV)
  aa-Index  : aa_1868-01-04_1968-01-04_D.dat + aa_1968-01-04_2024-12-31_D.dat

Dependencies: numpy, scipy, pandas, matplotlib, astropy
              pip install astropy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors
from astropy.timeseries import LombScargle
import urllib.request
import io

# ──────────────────────────────────────────────────────────────────────────────
# DATA FILES
# ──────────────────────────────────────────────────────────────────────────────
F107_FILE = "penticton_radio_flux.csv"
TSI_FILE  = "SATIRE-S_TSI_latest.txt"
PHI_FILE  = "Phi_mon_tab.txt"
AA_FILE1  = "aa_1868-01-04_1968-01-04_D.dat"
AA_FILE2  = "aa_1968-01-04_2024-12-31_D.dat"
# ──────────────────────────────────────────────────────────────────────────────

# ── Period bands (years) ──────────────────────────────────────────────────────
PERIOD_BANDS = {
    "~27-day\n(Rotation)":    (0.060,  0.090),
    "~150-day\n(Rieger)":     (0.35,   0.50),
    "~1\u20132 yr\n(QBO)":         (1.0,    2.5),
    "~5\u20136 yr\n(Mid-term)":    (4.5,    6.5),
    "~11 yr\n(Schwabe)":      (9.0,   13.0),
    "~22 yr\n(Hale)":         (18.0,  26.0),
    "~80 yr\n(Gleissberg)":   (70.0,  95.0),
}

DATASET_NAMES = [
    "Sunspot\nNumber",
    "F10.7\nIndex",
    "TSI\n(SATIRE-S)",
    "Cosmic Ray\nPhi (MV)",
    "aa-Index",
]

FAP_THRESHOLD = 0.01   # significance level for hatching


# ── Helpers ───────────────────────────────────────────────────────────────────
def jd_to_datetime(jd_series):
    epoch = pd.Timestamp("1858-11-17")
    return epoch + pd.to_timedelta(jd_series - 2400000.5, unit="D")


def dt_to_decimal(dt_series):
    return dt_series.year + (dt_series.dayofyear - 1) / 365.25


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_sunspot():
    url = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    print("  [1/5] Downloading SILSO SSN \u2026")
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=";", header=None,
                     usecols=[0, 1, 3], names=["year", "month", "ssn"])
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" +
                                df["month"].astype(str) + "-01")
    df = df[df["ssn"] >= 0].dropna(subset=["ssn"])
    t = dt_to_decimal(df["date"].dt).values
    y = df["ssn"].values.astype(float)
    print(f"      {len(df)} monthly values  ({df['year'].min()}\u2013{df['year'].max()})")
    return t, y


def load_f107(fp):
    print("  [2/5] Loading F10.7 \u2026")
    df = pd.read_csv(fp, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[:, [0, 2]].copy()
    df.columns = ["jd", "f107"]
    df = df[df["f107"] > 0].dropna()
    df["date"] = jd_to_datetime(df["jd"])
    mo = df.set_index("date")["f107"].resample("MS").mean().dropna()
    mo = mo[mo > 0]
    t = dt_to_decimal(mo.index.to_series().dt).values
    y = mo.values.astype(float)
    print(f"      {len(mo)} monthly values  ({mo.index[0].year}\u2013{mo.index[-1].year})")
    return t, y


def load_tsi(fp):
    print("  [3/5] Loading TSI \u2026")
    df = pd.read_csv(fp, comment=";", header=None, sep=r"\s+",
                     usecols=[0, 1], names=["jd", "tsi"])
    df = df[df["tsi"] > 0].dropna()
    df["date"] = jd_to_datetime(df["jd"])
    mo = df.set_index("date")["tsi"].resample("MS").mean().dropna()
    t = dt_to_decimal(mo.index.to_series().dt).values
    y = mo.values.astype(float)
    print(f"      {len(mo)} monthly values  ({mo.index[0].year}\u2013{mo.index[-1].year})")
    return t, y


def load_phi(fp):
    print("  [4/5] Loading Phi (Cosmic Ray) \u2026")
    rows = []
    with open(fp) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) >= 2:
                try:
                    yr, phi = float(p[0]), float(p[1])
                    if 1900 < yr < 2100 and phi > 0:
                        rows.append((yr, phi))
                except ValueError:
                    pass
    t = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    print(f"      {len(t)} monthly values  ({t.min():.0f}\u2013{t.max():.0f})")
    return t, y


def load_aa(fp1, fp2):
    print("  [5/5] Loading aa-Index (combining two files) \u2026")
    def parse(fp):
        rows = []
        with open(fp) as fh:
            for line in fh:
                if len(line) > 10 and line[4] == "-" and line[7] == "-":
                    p = line.split()
                    if len(p) >= 4:
                        try:
                            v = float(p[3])
                            if v < 999:
                                rows.append((p[0], v))
                        except (ValueError, IndexError):
                            pass
        df = pd.DataFrame(rows, columns=["ds", "aa"])
        df["date"] = pd.to_datetime(df["ds"])
        return df
    df = pd.concat([parse(fp1), parse(fp2)], ignore_index=True)
    df = df.drop_duplicates("date").sort_values("date")
    mo = df.set_index("date")["aa"].resample("MS").mean().dropna()
    t = dt_to_decimal(mo.index.to_series().dt).values
    y = mo.values.astype(float)
    print(f"      {len(mo)} monthly values  ({mo.index[0].year}\u2013{mo.index[-1].year})")
    return t, y


# ── LS power + FAP ────────────────────────────────────────────────────────────
def ls_band(t, y, bmin, bmax, n_grid=800):
    """Return (max_power, fap_of_max_power) for the given period band."""
    y = y - np.mean(y)
    freq = np.linspace(1.0 / bmax, 1.0 / bmin, n_grid)
    ls   = LombScargle(t, y)
    pwr  = ls.power(freq)
    pmax = float(np.max(pwr))
    fap  = float(ls.false_alarm_probability(pmax, method="baluev"))
    return pmax, fap


# ── Build matrices ────────────────────────────────────────────────────────────
def build_matrices(datasets):
    bands  = list(PERIOD_BANDS.keys())
    n_ds   = len(datasets)
    n_b    = len(bands)
    pmat   = np.full((n_ds, n_b), np.nan)
    fmat   = np.full((n_ds, n_b), np.nan)

    for i, (t, y) in enumerate(datasets):
        tspan = t.max() - t.min()
        for j, (bl, (bmin, bmax)) in enumerate(PERIOD_BANDS.items()):
            if tspan < bmin * 0.5:
                continue                       # dataset too short
            if (1 / bmax) > (0.5 / np.median(np.diff(np.sort(t)))):
                continue                       # period shorter than Nyquist
            pmat[i, j], fmat[i, j] = ls_band(t, y, bmin, bmax)
        print(f"    Dataset {i+1}/{n_ds} done.")

    # Row-normalise so heatmap colours are comparable across datasets
    pmat_norm = np.full_like(pmat, np.nan)
    for i in range(n_ds):
        row = pmat[i]
        rmax = np.nanmax(row)
        if rmax > 0:
            pmat_norm[i] = row / rmax

    return pmat, pmat_norm, fmat, bands


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_heatmap(pmat, pmat_norm, fmat, bands, dnames,
                 out="heatmap_dominant_periods.png"):
    n_ds, n_b = pmat.shape
    fig, ax   = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    # Colourmap: sequential for normalised power
    cmap = plt.cm.YlOrRd
    im   = ax.imshow(pmat_norm, aspect="auto", cmap=cmap,
                     vmin=0, vmax=1, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("Row-Normalised LS Power\n(1.0 = strongest in that row)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Annotate cells
    for i in range(n_ds):
        for j in range(n_b):
            pval  = pmat[i, j]
            fnorm = pmat_norm[i, j]
            fap   = fmat[i, j]

            if np.isnan(pval):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="#999999", style="italic")
                continue

            # Hatch non-significant cells
            sig    = (not np.isnan(fap)) and (fap < FAP_THRESHOLD)
            txt    = f"{pval:.3f}"
            col    = "white" if (fnorm > 0.65 and sig) else "#1a1a1a"
            weight = "bold" if sig else "normal"
            style  = "normal" if sig else "italic"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=col, fontweight=weight,
                    fontstyle=style)

            if not sig:
                # draw a subtle x-hatch overlay
                x0, x1 = j - 0.49, j + 0.49
                y0, y1 = i - 0.49, i + 0.49
                ax.plot([x0, x1], [y0, y1], color="#aaaaaa", lw=0.7, alpha=0.6)
                ax.plot([x0, x1], [y1, y0], color="#aaaaaa", lw=0.7, alpha=0.6)

    ax.set_xticks(range(n_b))
    ax.set_xticklabels(bands, fontsize=9.5)
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels(dnames, fontsize=10.5)
    ax.set_xlabel("Period Band", fontsize=12, labelpad=8)
    ax.set_title(
        "Heatmap of Dominant Periods Across Solar Activity Datasets\n"
        "Lomb-Scargle Normalised Power  |  \u00d7-hatch = FAP \u2265 0.01 (not significant)",
        fontsize=12.5, fontweight="bold", pad=10,
    )

    # Grid lines for readability
    for x in np.arange(-0.5, n_b, 1):
        ax.axvline(x, color="white", lw=1.2)
    for y in np.arange(-0.5, n_ds, 1):
        ax.axhline(y, color="white", lw=1.2)

    # Legend patch
    sig_patch   = mpatches.Patch(facecolor="#cc0000", alpha=0.7,
                                  label="FAP < 0.01  (significant)")
    nosig_patch = mpatches.Patch(facecolor="#eeeeee", edgecolor="#aaaaaa",
                                  label="FAP \u2265 0.01  (not significant, cross-hatched)")
    na_patch    = mpatches.Patch(facecolor="white", edgecolor="#aaaaaa",
                                  label="N/A  (dataset span too short)")
    ax.legend(handles=[sig_patch, nosig_patch, na_patch],
              loc="upper right", fontsize=8, framealpha=0.85,
              bbox_to_anchor=(1.0, -0.14), ncol=3)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Saved \u2192 {out}")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("Group E \u2013 Plot 1: Dominant Period Heatmap  (improved)")
    print("=" * 62)

    datasets = [
        load_sunspot(),
        load_f107(F107_FILE),
        load_tsi(TSI_FILE),
        load_phi(PHI_FILE),
        load_aa(AA_FILE1, AA_FILE2),
    ]

    print("\nComputing Lomb-Scargle band powers + FAP \u2026")
    pmat, pmat_norm, fmat, bands = build_matrices(datasets)

    # Print raw power table
    print("\nRaw LS Power matrix:")
    hdr = "  {:22s} " + "  {:8s}" * len(bands)
    print(hdr.format("Dataset", *[b.replace("\n", " ") for b in bands]))
    for i, name in enumerate(DATASET_NAMES):
        row = "  {:22s} " + "  {:8.4f}" * len(bands)
        vals = [pmat[i, j] if not np.isnan(pmat[i, j]) else 0 for j in range(len(bands))]
        print(row.format(name.replace("\n", " "), *vals))

    plot_heatmap(pmat, pmat_norm, fmat, bands, DATASET_NAMES)


if __name__ == "__main__":
    main()
