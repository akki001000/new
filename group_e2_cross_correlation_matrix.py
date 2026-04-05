"""
Group E – Plot 2: Cross-Correlation Matrix of All Solar Indices
===============================================================
IMPROVEMENTS over previous version
------------------------------------
1. Spearman rho added alongside Pearson r — important because SSN-TSI and
   Phi-aa relationships are known to be nonlinear (literature confirms ~0.87
   for SSN-TSI Pearson; ~0.80 for SSN-Phi).
2. Upper triangle shows Pearson r  |  Lower triangle shows Spearman rho.
3. Diagonal shows the dataset name + data range (years).
4. Significance stars correct (Bonferroni-corrected for 10 unique pairs).
5. Colourbar anchored at +/-1 with diverging palette (blue=negative,
   red=positive), correctly conveys the Phi/aa anti-correlation sign.
6. Time-lag awareness note: Pearson r here is zero-lag; see Plot 3 for
   lead-lag analysis.
7. All-positive palette issue fixed -- Phi (cosmic ray) is ANTI-correlated
   with SSN/F10.7/TSI, which should appear BLUE. This is now correct because
   we use the actual sign of r (not abs).
8. Caption / subtitle clarified for dissertation.

Datasets: same as Plot 1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr
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

LONG_LABELS  = ["Sunspot Number\n(SSN)", "F10.7 Index\n(SFU)",
                 "TSI\n(W m\u207b\u00b2)", "Cosmic Ray Phi\n(MV)", "aa-Index\n(nT)"]
SHORT_LABELS  = ["SSN", "F10.7", "TSI", "Phi", "aa"]

# Bonferroni-corrected alpha for 10 unique pairs
N_PAIRS = 10
SIG_LEVELS = {
    "***": 0.001 / N_PAIRS,
    "**":  0.01  / N_PAIRS,
    "*":   0.05  / N_PAIRS,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def jd_to_datetime(jd_series):
    return pd.Timestamp("1858-11-17") + pd.to_timedelta(jd_series - 2400000.5, unit="D")


def sig_stars(p, bonferroni=True):
    if p < 0.001 / N_PAIRS:
        return "***"
    if p < 0.01  / N_PAIRS:
        return "**"
    if p < 0.05  / N_PAIRS:
        return "*"
    return ""


# ── Loaders (monthly Series with DatetimeIndex) ───────────────────────────────
def load_sunspot():
    url = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    print("  Loading SSN \u2026")
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=";", header=None,
                     usecols=[0, 1, 3], names=["year", "month", "ssn"])
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" +
                                df["month"].astype(str) + "-01")
    df = df[df["ssn"] >= 0].dropna(subset=["ssn"]).set_index("date")["ssn"]
    df.index = df.index.to_period("M").to_timestamp()
    return df.rename("SSN")


def load_f107(fp):
    print("  Loading F10.7 \u2026")
    df = pd.read_csv(fp, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[:, [0, 2]].copy()
    df.columns = ["jd", "v"]
    df = df[df["v"] > 0].dropna()
    df["date"] = jd_to_datetime(df["jd"])
    mo = df.set_index("date")["v"].resample("MS").mean()
    return mo[mo > 0].dropna().rename("F10.7")


def load_tsi(fp):
    print("  Loading TSI \u2026")
    df = pd.read_csv(fp, comment=";", header=None, sep=r"\s+",
                     usecols=[0, 1], names=["jd", "v"])
    df = df[df["v"] > 0].dropna()
    df["date"] = jd_to_datetime(df["jd"])
    mo = df.set_index("date")["v"].resample("MS").mean().dropna()
    return mo.rename("TSI")


def load_phi(fp):
    print("  Loading Phi \u2026")
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
    dates = []
    for fy in [r[0] for r in rows]:
        yr  = int(fy)
        doy = int((fy - yr) * 365.25) + 1
        dates.append(pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=doy - 1))
    df = pd.DataFrame({"date": dates, "v": [r[1] for r in rows]})
    mo = df.set_index("date")["v"].resample("MS").mean().dropna()
    return mo.rename("Phi")


def load_aa(fp1, fp2):
    print("  Loading aa \u2026")
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
    return mo.rename("aa")


def align(series_list):
    combined = series_list[0].to_frame()
    for s in series_list[1:]:
        combined = combined.join(s.to_frame(), how="inner")
    combined = combined.dropna()
    y0, y1 = combined.index[0].year, combined.index[-1].year
    print(f"\n  Common period: {y0}\u2013{y1}  ({len(combined)} months)")
    return combined


# ── Correlation matrices ──────────────────────────────────────────────────────
def build_corr(df):
    cols = df.columns.tolist()
    n    = len(cols)
    pr   = np.ones((n, n))
    pp   = np.zeros((n, n))
    sr   = np.ones((n, n))
    sp   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r, p = pearsonr(df.iloc[:, i], df.iloc[:, j])
                pr[i, j], pp[i, j] = r, p
                r2, p2 = spearmanr(df.iloc[:, i], df.iloc[:, j])
                sr[i, j], sp[i, j] = r2, p2
    return (pd.DataFrame(pr, index=cols, columns=cols),
            pd.DataFrame(pp, index=cols, columns=cols),
            pd.DataFrame(sr, index=cols, columns=cols),
            pd.DataFrame(sp, index=cols, columns=cols))


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_matrix(pr_df, pp_df, sr_df, sp_df, df_aligned, year_range):
    """
    Upper triangle : Pearson r
    Diagonal       : label + year range
    Lower triangle : Spearman rho
    """
    n     = len(SHORT_LABELS)
    short = SHORT_LABELS

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    # Build colour matrix: upper = Pearson, lower = Spearman, diag = 1
    cmat = np.eye(n)
    for i in range(n):
        for j in range(n):
            if j > i:
                cmat[i, j] = pr_df.values[i, j]   # upper
            elif i > j:
                cmat[i, j] = sr_df.values[i, j]   # lower

    im = ax.imshow(cmat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation coefficient", fontsize=11)
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.ax.tick_params(labelsize=9)

    # Annotate
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: name + range
                ax.text(j, i, f"{short[i]}\n{year_range[short[i]]}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
            elif j > i:
                # Upper: Pearson
                r     = pr_df.values[i, j]
                p     = pp_df.values[i, j]
                stars = sig_stars(p)
                col   = "white" if abs(r) > 0.65 else "#111111"
                ax.text(j, i, f"r = {r:.2f}{stars}", ha="center", va="center",
                        fontsize=10, color=col, fontweight="bold")
            else:
                # Lower: Spearman
                r     = sr_df.values[i, j]
                p     = sp_df.values[i, j]
                stars = sig_stars(p)
                col   = "white" if abs(r) > 0.65 else "#111111"
                ax.text(j, i, f"\u03c1 = {r:.2f}{stars}", ha="center", va="center",
                        fontsize=10, color=col, style="italic")

    # Grid
    for k in np.arange(-0.5, n, 1):
        ax.axvline(k, color="white", lw=1.5)
        ax.axhline(k, color="white", lw=1.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, fontsize=12)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=12)

    ax.set_title(
        "Cross-Correlation Matrix of Solar Activity Indices\n"
        "Upper: Pearson r  |  Lower: Spearman \u03c1  |  "
        "Bonferroni-corrected: * p<0.05  ** p<0.01  *** p<0.001",
        fontsize=11, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    out = "cross_correlation_matrix.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Saved \u2192 {out}")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("Group E \u2013 Plot 2: Cross-Correlation Matrix  (improved)")
    print("=" * 62)

    ssn  = load_sunspot()
    f107 = load_f107(F107_FILE)
    tsi  = load_tsi(TSI_FILE)
    phi  = load_phi(PHI_FILE)
    aa   = load_aa(AA_FILE1, AA_FILE2)

    df = align([ssn, f107, tsi, phi, aa])

    # Year ranges for diagonal labels
    year_range = {
        col: f"{df[col].dropna().index[0].year}\u2013{df[col].dropna().index[-1].year}"
        for col in df.columns
    }

    print("\nComputing Pearson r and Spearman \u03c1 \u2026")
    pr_df, pp_df, sr_df, sp_df = build_corr(df)

    print("\nPearson r matrix:")
    print(pr_df.round(3).to_string())
    print("\nSpearman \u03c1 matrix:")
    print(sr_df.round(3).to_string())

    # Flag the known sign issue: Phi should be anti-correlated with SSN
    phi_ssn_r = pr_df.loc["Phi", "SSN"]
    sign_note = "\u2713 correct sign" if phi_ssn_r < 0 else "\u26a0 CHECK SIGN \u2013 Phi should be NEGATIVE"
    print(f"\n  Phi vs SSN Pearson r = {phi_ssn_r:.3f}  "
          f"(expected ~-0.80 from literature; {sign_note})")

    plot_matrix(pr_df, pp_df, sr_df, sp_df, df, year_range)


if __name__ == "__main__":
    main()
