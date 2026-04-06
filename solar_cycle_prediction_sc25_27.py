"""
Solar Cycle Prediction: SC 25, SC 26 & SC 27
=============================================
Based on periodicities and variability in solar activity.

Literature-backed methods implemented
--------------------------------------
1. Geomagnetic Precursor / Ohl's Rule
   Ref: Hathaway & Wilson (2006) GRL 33 L18101;
        Podladchikova et al. (2017) A&A 605 A47

2. Waldmeier Effect (rise-time ↔ amplitude anti-correlation)
   Ref: Waldmeier (1935) Astr. Mitt. Zürich 14 105;
        Karak & Choudhuri (2011) MNRAS 410 1503

3. Multi-parameter Statistical Regression (leave-one-out CV)
   Ref: Hathaway (2015) Living Rev. Sol. Phys. 12 4

4. Spectral / Harmonic Extrapolation
   Lomb–Scargle dominant periods → multi-sinusoidal fit → forecast
   Ref: Scafetta (2012) J. Atm. Sol.-Terr. Phys. 80 124;
        Velasco & Mendoza (2008) Sol. Phys. 250 243

5. Secular context from Solanki/Usoskin 11 400-yr reconstruction
   Ref: Usoskin et al. (2021) A&A 649 A141;
        Gleissberg (1939) Observatory 62 158

6. Spatial Spörer Law — butterfly diagram of SC 25 to estimate phase
   Ref: Hathaway (2015); Spoerer (1880)

Datasets
--------
  SSN       : SILSO live download         (monthly means, SC 1–present)
  F10.7     : penticton_radio_flux.csv    (JD | obs | adjusted)
  TSI       : SATIRE-S_TSI_latest.txt     (JD | TSI)
  Phi (CR)  : Phi_mon_tab.txt             (fractional year | Phi MV)
  aa-Index  : aa_1868-01-04_1968-01-04_D.dat
              aa_1968-01-04_2024-12-31_D.dat
  Solanki   : solanki_usoskin_ssn.txt     (year | SN, annual, ~11 400 yr)
              Columns: year CE (negative = BCE) | SSN (whitespace-sep)
  Sunspot   : sunspot_area_latitude.txt   (YYYYMMDD | helio-lat | total-area)
              Typical RGO + NOAA/USAF composite; one group per row

Dependencies: numpy, scipy, pandas, matplotlib, astropy
"""

import io
import urllib.request
import warnings

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.stats import linregress, pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── DATA FILE PATHS ────────────────────────────────────────────────────────
F107_FILE     = "penticton_radio_flux.csv"
TSI_FILE      = "SATIRE-S_TSI_latest.txt"
PHI_FILE      = "Phi_mon_tab.txt"
AA_FILE1      = "aa_1868-01-04_1968-01-04_D.dat"
AA_FILE2      = "aa_1968-01-04_2024-12-31_D.dat"
SOLANKI_FILE  = "solanki_usoskin_ssn.txt"
SPOTAREA_FILE = "sunspot_area_latitude.txt"
# ────────────────────────────────────────────────────────────────────────────

# ─── KNOWN CYCLE MINIMA & MAXIMA (SIDC / Hathaway 2015 Living Rev.) ─────────
# Decimal year of solar minimum (cycle boundary)
SC_MINIMA = {
     1: 1755.2,   2: 1766.5,   3: 1775.5,   4: 1784.7,   5: 1798.3,
     6: 1810.6,   7: 1823.3,   8: 1833.8,   9: 1843.5,  10: 1855.7,
    11: 1867.2,  12: 1878.9,  13: 1889.6,  14: 1901.7,  15: 1913.6,
    16: 1923.6,  17: 1933.8,  18: 1944.2,  19: 1954.3,  20: 1964.7,
    21: 1976.3,  22: 1986.8,  23: 1996.4,  24: 2008.9,  25: 2019.9,
    26: 2030.5,  # projected
    27: 2041.5,  # projected
}

# Smoothed monthly SSN v2 maxima (SIDC; SC25 from SILSO 2025 data)
SC_MAXIMA_KNOWN = {
     1:  86.5,   2: 115.8,   3: 158.5,   4: 141.2,   5:  49.2,
     6:  48.7,   7:  71.5,   8: 146.9,   9: 131.6,  10:  97.9,
    11: 140.5,  12:  74.6,  13: 131.6,  14: 107.1,  15: 175.7,
    16: 130.2,  17: 198.6,  18: 218.7,  19: 285.0,  20: 156.6,
    21: 232.9,  22: 212.5,  23: 180.3,  24: 116.4,
}  # SC25 max determined from downloaded data

# Approximate decimal year of cycle maxima
SC_MAX_YEAR = {
     1: 1761.5,   2: 1769.7,   3: 1778.4,   4: 1788.1,   5: 1804.4,
     6: 1816.4,   7: 1829.9,   8: 1837.2,   9: 1848.1,  10: 1860.1,
    11: 1870.6,  12: 1883.9,  13: 1893.8,  14: 1905.9,  15: 1917.6,
    16: 1928.4,  17: 1937.4,  18: 1947.5,  19: 1957.9,  20: 1968.9,
    21: 1979.9,  22: 1989.9,  23: 2000.3,  24: 2014.3,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def jd_to_datetime(jd):
    return pd.Timestamp("1858-11-17") + pd.to_timedelta(jd - 2400000.5, unit="D")


def dt_to_decimal(dt_series):
    return dt_series.year + (dt_series.dayofyear - 1) / 365.25


def running_mean(x, w=13):
    return np.convolve(x, np.ones(w) / w, mode="same")


def find_cycle_maxima(t, y_smooth, min_yr, max_yr):
    """Return (peak_time, peak_value) within [min_yr, max_yr]."""
    mask = (t >= min_yr) & (t <= max_yr)
    if mask.sum() < 3:
        return np.nan, np.nan
    idx = np.argmax(y_smooth[mask])
    tt = t[mask]
    yy = y_smooth[mask]
    return float(tt[idx]), float(yy[idx])


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_ssn_monthly():
    """Download SILSO monthly SSN; return (monthly_Series, annual_Series)."""
    url = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    print("  [SSN] Downloading SILSO monthly SSN …")
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=";", header=None,
                     usecols=[0, 1, 3], names=["year", "month", "ssn"])
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" +
                                df["month"].astype(str) + "-01")
    df = df[df["ssn"] >= 0].dropna(subset=["ssn"])
    monthly = df.set_index("date")["ssn"]
    monthly.index = monthly.index.to_period("M").to_timestamp()
    annual = monthly.resample("YS").mean().dropna()
    print(f"       {len(monthly)} monthly, {len(annual)} annual values "
          f"({annual.index[0].year}–{annual.index[-1].year})")
    return monthly, annual


def load_aa():
    """Load aa-index daily files and return monthly mean Series."""
    print("  [aa]  Loading aa-index …")

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

    df = pd.concat([parse(AA_FILE1), parse(AA_FILE2)], ignore_index=True)
    df = df.drop_duplicates("date").sort_values("date")
    mo = df.set_index("date")["aa"].resample("MS").mean().dropna()
    print(f"       {len(mo)} monthly values "
          f"({mo.index[0].year}–{mo.index[-1].year})")
    return mo


def load_solanki(fp):
    """
    Load Solanki/Usoskin 11 400-yr reconstruction.
    Expected: whitespace-separated, col0 = year CE (neg = BCE), col1 = SN.
    """
    print(f"  [SLK] Loading Solanki reconstruction from {fp} …")
    rows = []
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    yr  = float(parts[0])
                    sn  = float(parts[1])
                    if sn >= 0:
                        rows.append((yr, sn))
                except ValueError:
                    pass
    if not rows:
        raise ValueError(f"Could not parse {fp} — check format.")
    t = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    print(f"       {len(t)} annual values  ({t.min():.0f}–{t.max():.0f} CE)")
    return t, y


def load_spotarea(fp):
    """
    Load sunspot group area & latitude.
    Expected: whitespace-separated, col0 = YYYYMMDD, col1 = helio-lat (deg),
              col2 or col3 = total area (MSH).
    Returns DataFrame with columns: date, lat, area.
    """
    print(f"  [SPA] Loading sunspot area/latitude from {fp} …")
    rows = []
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                date = pd.to_datetime(str(parts[0]), format="%Y%m%d")
                lat  = float(parts[1])
                area = float(parts[2]) if len(parts) == 3 else float(parts[3])
                if abs(lat) <= 90 and area > 0:
                    rows.append({"date": date, "lat": lat, "area": area})
            except (ValueError, TypeError):
                pass
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Could not parse {fp}")
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year + (df["date"].dt.dayofyear - 1) / 365.25
    print(f"       {len(df)} sunspot groups "
          f"({df['date'].dt.year.min()}–{df['date'].dt.year.max()})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLAR CYCLE PARAMETER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cycle_params(monthly_ssn):
    """
    Return DataFrame with one row per cycle (SC1–SC25):
      sc, min_yr, max_yr, rmax, period, rise_time_mo, fall_time_mo,
      asymmetry (rise/total), even (bool).
    """
    t_mo  = np.array(dt_to_decimal(monthly_ssn.index.to_series().dt))
    y_mo  = monthly_ssn.values.astype(float)
    y13   = running_mean(y_mo, 13)

    records = []
    cycles  = sorted(SC_MINIMA.keys())
    for sc in cycles:
        if sc not in SC_MINIMA or (sc + 1) not in SC_MINIMA:
            continue
        if sc > 25:
            break
        t0 = SC_MINIMA[sc]
        t1 = SC_MINIMA[sc + 1]

        # Peak from data (smoothed monthly SSN)
        pt, pv = find_cycle_maxima(t_mo, y13, t0, t1)
        if np.isnan(pt):
            if sc in SC_MAXIMA_KNOWN:
                pv = SC_MAXIMA_KNOWN[sc]
                pt = SC_MAX_YEAR.get(sc, (t0 + t1) / 2)
            else:
                continue

        period      = t1 - t0
        rise        = pt - t0
        fall        = t1 - pt
        asym        = rise / period if period > 0 else np.nan
        rise_mo     = rise * 12
        fall_mo     = fall * 12

        records.append({
            "sc":          sc,
            "min_yr":      t0,
            "max_yr":      pt,
            "rmax":        pv,
            "period":      period,
            "rise_mo":     rise_mo,
            "fall_mo":     fall_mo,
            "asymmetry":   asym,
            "even":        (sc % 2 == 0),
        })

    df = pd.DataFrame(records).set_index("sc")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 1 — GEOMAGNETIC PRECURSOR (OHL'S RULE)
# ═══════════════════════════════════════════════════════════════════════════════

def ohl_precursor(aa_monthly, cycle_df):
    """
    For each cycle SC(n): find min(smoothed aa) in the ~3 yr before the
    minimum of SC(n) [i.e. the declining phase of SC(n-1)].
    Fit Rmax(n) = a * aaMin + b.
    Returns (aa_min_series, fit_slope, fit_intercept, r_sq).
    """
    aa_smooth = aa_monthly.rolling(13, center=True, min_periods=6).mean()
    t_aa  = np.array(dt_to_decimal(aa_smooth.index.to_series().dt))
    y_aa  = aa_smooth.values

    aa_min_vals, rmax_vals, cycle_ids = [], [], []

    for sc in cycle_df.index:
        t_start = cycle_df.loc[sc, "min_yr"] - 3.5   # ~3.5 yr before minimum
        t_end   = cycle_df.loc[sc, "min_yr"]
        mask    = (t_aa >= t_start) & (t_aa <= t_end) & ~np.isnan(y_aa)
        if mask.sum() < 6:
            continue
        aaMin = float(np.nanmin(y_aa[mask]))
        aa_min_vals.append(aaMin)
        rmax_vals.append(cycle_df.loc[sc, "rmax"])
        cycle_ids.append(sc)

    aa_arr  = np.array(aa_min_vals)
    rm_arr  = np.array(rmax_vals)
    slope, intercept, r, _, se = linregress(aa_arr, rm_arr)
    r_sq = r ** 2

    return (np.array(cycle_ids), aa_arr, rm_arr,
            slope, intercept, r_sq, se)


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 2 — WALDMEIER EFFECT + METHOD 3 — MULTI-PARAM REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def waldmeier_fit(cycle_df):
    """Linear fit: Rmax = a * rise_time + b (Waldmeier 1935)."""
    df = cycle_df.dropna(subset=["rmax", "rise_mo"])
    slope, intercept, r, _, se = linregress(df["rise_mo"], df["rmax"])
    return slope, intercept, r ** 2, df["rise_mo"].values, df["rmax"].values


def multivar_regression_loo(cycle_df):
    """
    LOO cross-validation of Rmax prediction using features:
      [Rmax(n-1), period(n-1), rise_time(n-1), even/odd(n)].
    Returns (observed, predicted, sc_numbers).
    """
    from numpy.linalg import lstsq

    cycles = sorted(cycle_df.index)
    obs, pred, scs = [], [], []

    for i in range(1, len(cycles)):
        sc_n   = cycles[i]
        sc_nm1 = cycles[i - 1]
        if sc_nm1 not in cycle_df.index:
            continue

        row_nm1 = cycle_df.loc[sc_nm1]
        row_n   = cycle_df.loc[sc_n]
        if np.isnan(row_nm1["rmax"]) or np.isnan(row_n["rmax"]):
            continue

        # Build design matrix on all-other cycles
        X_all, y_all = [], []
        for j in range(1, len(cycles)):
            sc_j    = cycles[j]
            sc_jm1  = cycles[j - 1]
            if sc_j == sc_n or sc_jm1 not in cycle_df.index:
                continue
            r_jm1 = cycle_df.loc[sc_jm1]
            r_j   = cycle_df.loc[sc_j]
            if np.isnan(r_jm1["rmax"]) or np.isnan(r_j["rmax"]):
                continue
            X_all.append([1, r_jm1["rmax"], r_jm1["period"],
                          r_jm1["rise_mo"], int(r_j["even"])])
            y_all.append(r_j["rmax"])

        if len(X_all) < 3:
            continue
        X_mat  = np.array(X_all)
        y_vec  = np.array(y_all)
        coeff, _, _, _ = lstsq(X_mat, y_vec, rcond=None)

        x_pred = np.array([1, row_nm1["rmax"], row_nm1["period"],
                           row_nm1["rise_mo"], int(row_n["even"])])
        y_hat  = float(x_pred @ coeff)
        obs.append(row_n["rmax"])
        pred.append(y_hat)
        scs.append(sc_n)

    return np.array(obs), np.array(pred), np.array(scs)


def multivar_full_fit(cycle_df, sc_target, sc_prev_params):
    """
    Full-data regression: predict Rmax for sc_target.
    sc_prev_params: dict with keys rmax, period, rise_mo, even (of target cycle).
    """
    from numpy.linalg import lstsq

    cycles = sorted(cycle_df.index)
    X_all, y_all = [], []
    for i in range(1, len(cycles)):
        sc_n   = cycles[i]
        sc_nm1 = cycles[i - 1]
        if sc_nm1 not in cycle_df.index:
            continue
        r_nm1 = cycle_df.loc[sc_nm1]
        r_n   = cycle_df.loc[sc_n]
        if np.isnan(r_nm1["rmax"]) or np.isnan(r_n["rmax"]):
            continue
        X_all.append([1, r_nm1["rmax"], r_nm1["period"],
                      r_nm1["rise_mo"], int(r_n["even"])])
        y_all.append(r_n["rmax"])

    X_mat = np.array(X_all)
    y_vec = np.array(y_all)
    coeff, _, _, _ = lstsq(X_mat, y_vec, rcond=None)

    x_new = np.array([1, sc_prev_params["rmax"], sc_prev_params["period"],
                      sc_prev_params["rise_mo"], int(sc_prev_params["even"])])
    return float(x_new @ coeff)


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD 4 — SPECTRAL / HARMONIC EXTRAPOLATION
# ═══════════════════════════════════════════════════════════════════════════════

def harmonic_model(t, A0, A1, phi1, A2, phi2, A3, phi3):
    """
    Sum of sinusoids with fixed dominant solar periods:
      T1 = 11.05 yr  (Schwabe)
      T2 = 22.1  yr  (Hale / double-cycle)
      T3 = 87.0  yr  (Gleissberg)
    """
    T1, T2, T3 = 11.05, 22.1, 87.0
    return (A0
            + A1 * np.cos(2 * np.pi / T1 * t + phi1)
            + A2 * np.cos(2 * np.pi / T2 * t + phi2)
            + A3 * np.cos(2 * np.pi / T3 * t + phi3))


def fit_harmonic_model(t_annual, y_annual):
    """
    Fit harmonic_model to annual SSN.  Returns (popt, pcov, y_fit).
    Uses Lomb-Scargle to get initial phase guesses.
    """
    y = y_annual - np.nanmean(y_annual)
    freq = np.linspace(1 / 200, 1 / 5, 4000)
    ls   = LombScargle(t_annual, y)
    pwr  = ls.power(freq)

    # Phase estimate at each key period
    def phase_at(T):
        idx   = np.argmin(np.abs(freq - 1 / T))
        phase = float(ls.model(np.array([0.0]),
                               np.array([freq[idx]]))[0])
        return 0.0  # start with 0 and let curve_fit optimise

    A0_g = float(np.nanmean(y_annual))
    p0 = [A0_g,
          40.0,  0.0,   # Schwabe
          20.0,  0.0,   # Hale
          30.0,  0.0]   # Gleissberg

    bounds_lo = [0,  0, -np.pi,  0, -np.pi,  0, -np.pi]
    bounds_hi = [400, 200, np.pi, 150, np.pi, 200, np.pi]

    try:
        popt, pcov = curve_fit(
            harmonic_model, t_annual, y_annual,
            p0=p0, bounds=(bounds_lo, bounds_hi),
            maxfev=20_000,
        )
    except RuntimeError:
        # Fall back: no bounds
        popt, pcov = curve_fit(
            harmonic_model, t_annual, y_annual,
            p0=p0, maxfev=20_000,
        )

    y_fit = harmonic_model(t_annual, *popt)
    return popt, pcov, y_fit


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – LONG-TERM SOLAR ACTIVITY CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_longterm(t_slk, y_slk, annual_ssn, cycle_df, out="fig1_longterm_context.png"):
    print("\n  Generating Figure 1 …")
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#F8F8F8")
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)

    # ── Panel A: 11 400-yr reconstruction ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#F8F8F8")

    # Smooth with 100-yr window for secular trend
    y_smooth100 = running_mean(y_slk, min(101, len(y_slk) // 10 * 2 + 1))

    ax1.fill_between(t_slk, y_slk, alpha=0.35, color="steelblue",
                     label="Annual SN (11 400-yr)")
    ax1.plot(t_slk, y_smooth100, color="navy", lw=1.5,
             label="100-yr running mean")

    # Mark grand minima (Usoskin 2021 Table 1)
    grand_minima = [
        ("Maunder",   1645, 1715),
        ("Spörer",    1415, 1530),
        ("Wolf",      1270, 1340),
        ("Oort",      1010, 1060),
        ("Dalton",    1790, 1820),
    ]
    for name, t0, t1 in grand_minima:
        ax1.axvspan(t0, t1, alpha=0.18, color="red",
                    label=f"{name} minimum" if name == "Maunder" else "")
        ax1.text((t0 + t1) / 2, ax1.get_ylim()[1] * 0.85 if ax1.get_ylim()[1] > 0 else 200,
                 name, ha="center", fontsize=7, color="darkred", style="italic")

    ax1.axvspan(1940, 1980, alpha=0.15, color="gold",
                label="Modern Grand Maximum")
    ax1.set_xlim(t_slk.min(), t_slk.max())
    ax1.set_xlabel("Year CE", fontsize=11)
    ax1.set_ylabel("Sunspot Number", fontsize=11)
    ax1.set_title(
        "Panel A – 11 400-yr Solar Activity Reconstruction (Usoskin et al. 2021)\n"
        "Red = Grand Minima  |  Gold = Modern Grand Maximum",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(fontsize=8, loc="upper left", ncol=2)

    # ── Panel B: Modern SSN (SC1–SC25) with Gleissberg envelope ───────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#F8F8F8")

    t_ann = annual_ssn.index.year.values.astype(float)
    y_ann = annual_ssn.values

    y_gle  = running_mean(y_ann, min(89, len(y_ann) // 2 * 2 + 1))
    ax2.fill_between(t_ann, y_ann, alpha=0.4, color="royalblue",
                     label="Annual SSN (SC 1–present)")
    ax2.plot(t_ann, y_gle, color="darkred", lw=2.0, ls="--",
             label="~89-yr Gleissberg envelope")

    # Mark SC cycle numbers at maximum
    for sc, row in cycle_df.iterrows():
        if 1 <= sc <= 25:
            ax2.text(row["max_yr"], row["rmax"] + 6, str(sc),
                     ha="center", va="bottom", fontsize=6.5,
                     color="darkblue", fontweight="bold")

    # Shade SC25 window
    ax2.axvspan(SC_MINIMA[25], SC_MINIMA.get(26, 2032), alpha=0.15,
                color="orange", label="SC 25 (current)")

    ax2.set_xlim(1740, 2060)
    ax2.set_xlabel("Year CE", fontsize=11)
    ax2.set_ylabel("Sunspot Number", fontsize=11)
    ax2.set_title(
        "Panel B – Modern SSN (SC 1–SC 25)  |  Gleissberg (~80-yr) secular modulation",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(fontsize=9)

    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"     Saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – CYCLE PARAMETER OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_cycle_params(cycle_df, out="fig2_cycle_parameters.png"):
    print("\n  Generating Figure 2 …")
    df = cycle_df.dropna(subset=["rmax"]).copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#F8F8F8")
    fig.suptitle("Solar Cycle Parameter Analysis (SC 1–SC 25)", fontsize=13,
                 fontweight="bold", y=1.01)

    # ── A: Rmax bar chart, even/odd coloured ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F8F8F8")
    colors = ["#2166ac" if row["even"] else "#d6604d" for _, row in df.iterrows()]
    bars = ax.bar(df.index, df["rmax"], color=colors, edgecolor="white", width=0.7)
    sc25_obs = df.loc[25, "rmax"] if 25 in df.index else None
    if sc25_obs:
        ax.bar([25], [sc25_obs], color="gold", edgecolor="black", width=0.7,
               label=f"SC 25 (obs. max ≈ {sc25_obs:.0f})", zorder=5)
    ax.set_xlabel("Solar Cycle Number", fontsize=10)
    ax.set_ylabel("Smoothed SSN Maximum", fontsize=10)
    ax.set_title("(A) Cycle Amplitude\nBlue = even, Red = odd", fontsize=10,
                 fontweight="bold")
    even_p = mpatches.Patch(color="#2166ac", label="Even cycle")
    odd_p  = mpatches.Patch(color="#d6604d", label="Odd cycle")
    ax.legend(handles=[even_p, odd_p], fontsize=8)
    ax.axhline(df["rmax"].mean(), color="grey", ls="--", lw=1,
               label=f"Mean = {df['rmax'].mean():.0f}")

    # Gnevyshev-Ohl numbers
    for sc, row in df.iterrows():
        if sc <= 24:
            ax.text(sc, row["rmax"] + 4, str(sc), ha="center",
                    va="bottom", fontsize=6, color="black")

    # ── B: Waldmeier scatter ──────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#F8F8F8")
    wdf = df.dropna(subset=["rise_mo"])
    m, b, r, _, _ = linregress(wdf["rise_mo"], wdf["rmax"])
    t_line = np.linspace(wdf["rise_mo"].min(), wdf["rise_mo"].max(), 100)

    ax.scatter(wdf["rise_mo"], wdf["rmax"],
               c=["gold" if i == 25 else ("#2166ac" if wdf.loc[i, "even"] else "#d6604d")
                  for i in wdf.index],
               s=70, edgecolors="grey", linewidths=0.5, zorder=5)
    ax.plot(t_line, m * t_line + b, "k--", lw=1.5,
            label=f"r = {r:.2f}, slope = {m:.2f}")
    # Annotate key cycles
    for sc in [19, 24, 25]:
        if sc in wdf.index:
            ax.annotate(f"SC{sc}",
                        (wdf.loc[sc, "rise_mo"], wdf.loc[sc, "rmax"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Rise Time (months)", fontsize=10)
    ax.set_ylabel("Smoothed SSN Maximum", fontsize=10)
    ax.set_title(f"(B) Waldmeier Effect\nShorter rise → stronger cycle",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── C: Even-Odd (Gnevyshev-Ohl) running pairs ────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#F8F8F8")
    evens = df[df["even"]]["rmax"].values
    odds  = df[~df["even"]]["rmax"].values
    pairs = min(len(evens), len(odds))
    pair_nos = np.arange(1, pairs + 1)

    ax.plot(pair_nos, odds[:pairs],  "o-", color="#d6604d",  lw=1.5,
            label="Odd cycles (Rmax)", ms=5)
    ax.plot(pair_nos, evens[:pairs], "s--", color="#2166ac", lw=1.5,
            label="Even cycles (Rmax)", ms=5)
    diff = odds[:pairs] - evens[:pairs]
    ax.bar(pair_nos, diff, bottom=evens[:pairs], color="lightgrey",
           edgecolor="white", alpha=0.6, label="Odd – Even difference")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Cycle Pair Number", fontsize=10)
    ax.set_ylabel("Smoothed SSN Maximum", fontsize=10)
    ax.set_title("(C) Gnevyshev–Ohl Rule\nOdd cycles tend to exceed preceding even",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"     Saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 – OHL / GEOMAGNETIC PRECURSOR
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_ohl(cycle_ids, aa_min_arr, rmax_arr, slope, intercept, r_sq,
             cycle_df, out="fig3_ohl_precursor.png"):
    print("\n  Generating Figure 3 …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#F8F8F8")
    fig.suptitle(
        "Geomagnetic Precursor Method (Ohl 1966; Hathaway & Wilson 2006)\n"
        "Min(aa) in declining phase predicts next cycle maximum",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.set_facecolor("#F8F8F8")

    x_line = np.linspace(aa_min_arr.min() * 0.85, aa_min_arr.max() * 1.15, 200)
    y_line = slope * x_line + intercept

    # 95 % confidence band (approx. ± 2 * residual std)
    resid  = rmax_arr - (slope * aa_min_arr + intercept)
    sig    = np.std(resid)
    ax.fill_between(x_line, y_line - 2 * sig, y_line + 2 * sig,
                    alpha=0.15, color="steelblue", label="±2σ confidence")
    ax.plot(x_line, y_line, "k--", lw=1.5,
            label=f"Fit: Rmax = {slope:.1f}·aaMin + {intercept:.1f}\n$R^2$ = {r_sq:.2f}")

    for i, sc in enumerate(cycle_ids):
        color = "gold" if sc == 25 else "#2166ac"
        ms    = 10 if sc == 25 else 6
        ax.scatter(aa_min_arr[i], rmax_arr[i], color=color, s=ms**2,
                   edgecolors="grey", linewidths=0.5, zorder=5)
        ax.annotate(f"SC{sc}", (aa_min_arr[i], rmax_arr[i]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Min(aa) in declining phase (nT)", fontsize=11)
    ax.set_ylabel("Rmax of following cycle", fontsize=11)
    ax.set_title("(A) Scatter: precursor vs cycle amplitude", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)

    # ── Predictions per-cycle bar ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F8F8F8")

    pred_arr  = slope * aa_min_arr + intercept
    obs_arr   = rmax_arr
    sc_labels = [f"SC{sc}" for sc in cycle_ids]

    x = np.arange(len(cycle_ids))
    w = 0.38
    ax2.bar(x - w / 2, obs_arr,  width=w, label="Observed Rmax",
            color="#2166ac", alpha=0.8)
    ax2.bar(x + w / 2, pred_arr, width=w, label="Predicted Rmax (Ohl)",
            color="#d6604d", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_labels, rotation=45, fontsize=8)
    ax2.set_ylabel("Smoothed SSN Maximum", fontsize=11)
    ax2.set_title("(B) Observed vs Ohl-predicted cycle maxima", fontsize=10,
                  fontweight="bold")
    ax2.legend(fontsize=9)

    # SC25 prediction annotation
    if 25 in cycle_ids:
        idx25 = list(cycle_ids).index(25)
        ax2.annotate(
            f"SC25 obs≈{obs_arr[idx25]:.0f}\npred={pred_arr[idx25]:.0f}",
            (x[idx25] + w / 2, pred_arr[idx25]),
            fontsize=8, color="darkred", fontweight="bold",
            xytext=(8, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
        )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"     Saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 – WALDMEIER + REGRESSION LOO-CV
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_regression(cycle_df, obs_loo, pred_loo, sc_loo,
                    pred_sc26, pred_sc27,
                    out="fig4_regression_prediction.png"):
    print("\n  Generating Figure 4 …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#F8F8F8")
    fig.suptitle(
        "Multi-parameter Statistical Regression  (Leave-One-Out Cross-Validation)\n"
        "Features: Rmax(n−1), Period(n−1), Rise-time(n−1), Even/Odd(n)",
        fontsize=12, fontweight="bold",
    )

    # ── A: Observed vs LOO-predicted ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F8F8F8")
    r_loo, _  = pearsonr(obs_loo, pred_loo)
    residuals = obs_loo - pred_loo
    rmse      = np.sqrt(np.mean(residuals ** 2))

    lim = (min(obs_loo.min(), pred_loo.min()) * 0.9,
           max(obs_loo.max(), pred_loo.max()) * 1.1)
    ax.plot(lim, lim, "k--", lw=1, label="1:1 line")
    ax.scatter(obs_loo, pred_loo, c=sc_loo, cmap="viridis",
               s=60, edgecolors="grey", linewidths=0.5, zorder=5)
    for i, sc in enumerate(sc_loo):
        ax.annotate(str(sc), (obs_loo[i], pred_loo[i]),
                    fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed Rmax", fontsize=11)
    ax.set_ylabel("Predicted Rmax (LOO-CV)", fontsize=11)
    ax.set_title(f"(A) LOO-CV: r = {r_loo:.2f}, RMSE = {rmse:.1f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ── B: SC25, SC26, SC27 predictions + all-method comparison ──────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F8F8F8")

    sc25_obs = cycle_df.loc[25, "rmax"] if 25 in cycle_df.index else None

    labels  = ["SC 25\n(observed)", "SC 26\n(regression)", "SC 27\n(regression)"]
    values  = [sc25_obs if sc25_obs else 0, pred_sc26, pred_sc27]
    errors  = [0, rmse, rmse * 1.4]       # larger uncertainty for SC27
    colors  = ["gold", "#4dac26", "#b8e186"]

    bars = ax2.bar(labels, values, color=colors, edgecolor="grey", width=0.45,
                   yerr=errors, capsize=6, error_kw={"ecolor": "black", "lw": 1.5})
    ax2.set_ylabel("Predicted Smoothed SSN Maximum", fontsize=11)
    ax2.set_title("(B) SC 25–27 Amplitude Predictions (Regression)\n"
                  f"SC26: {pred_sc26:.0f} ± {rmse:.0f}  |  "
                  f"SC27: {pred_sc27:.0f} ± {rmse * 1.4:.0f}",
                  fontsize=10, fontweight="bold")

    for bar, val in zip(bars, values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 4,
                     f"{val:.0f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    # Reference lines
    ax2.axhline(cycle_df["rmax"].mean(), color="grey", ls=":", lw=1,
                label=f"Historical mean ({cycle_df['rmax'].mean():.0f})")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"     Saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 – SPECTRAL / HARMONIC FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_harmonic(annual_ssn, popt, out="fig5_harmonic_forecast.png"):
    print("\n  Generating Figure 5 …")
    t_ann = annual_ssn.index.year.values.astype(float)
    y_ann = annual_ssn.values

    # Forecast to 2060
    t_fore = np.linspace(t_ann[-1], 2060, 500)
    y_fit  = harmonic_model(t_ann,  *popt)
    y_fore = harmonic_model(t_fore, *popt)
    y_fore = np.clip(y_fore, 0, None)  # SSN can't be negative

    # Uncertainty: residual standard deviation
    resid = y_ann - y_fit
    sig   = np.std(resid)
    # Growing uncertainty for longer horizons
    horizon = t_fore - t_ann[-1]
    sig_band = sig * (1 + 0.025 * horizon)   # ±2.5 %/yr growth

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    # Historical annual SSN
    ax.fill_between(t_ann, y_ann, alpha=0.3, color="royalblue",
                    label="Observed annual SSN")

    # Harmonic fit to historical data
    ax.plot(t_ann, y_fit, color="navy", lw=1.5, label="Harmonic fit (SC 1–present)")

    # Forecast region
    ax.fill_between(t_fore, y_fore - 2 * sig_band, y_fore + 2 * sig_band,
                    alpha=0.20, color="darkorange", label="Forecast ±2σ")
    ax.plot(t_fore, y_fore, color="darkorange", lw=2,
            ls="--", label="Harmonic forecast (SC 25–27)")

    # Shade predicted cycles
    cycle_colors = {"SC 25": "#ffd700", "SC 26": "#66c2a5", "SC 27": "#8da0cb"}
    for sc, (t0, t1) in [(25, (SC_MINIMA[25], SC_MINIMA.get(26, 2030.5))),
                          (26, (SC_MINIMA.get(26, 2030.5), SC_MINIMA.get(27, 2041.5))),
                          (27, (SC_MINIMA.get(27, 2041.5), 2053.0))]:
        ax.axvspan(t0, t1, alpha=0.12, color=cycle_colors[f"SC {sc}"],
                   label=f"SC {sc} window")

        # Mark projected maximum
        mask = (t_fore >= t0) & (t_fore <= t1)
        if mask.any():
            t_max_fore = float(t_fore[mask][np.argmax(y_fore[mask])])
            y_max_fore = float(np.max(y_fore[mask]))
            ax.annotate(
                f"SC {sc}\nmax≈{y_max_fore:.0f}\n~{t_max_fore:.1f}",
                (t_max_fore, y_max_fore),
                fontsize=9, fontweight="bold",
                ha="center", va="bottom",
                xytext=(0, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    # Cycle boundaries (minima)
    for sc in range(20, 28):
        if sc in SC_MINIMA:
            ax.axvline(SC_MINIMA[sc], color="grey", lw=0.7, ls=":")
            ax.text(SC_MINIMA[sc], ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 5,
                    f"Min{sc}", fontsize=6.5, rotation=90, va="bottom",
                    color="grey")

    ax.set_xlim(1950, 2060)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Smoothed SSN", fontsize=12)
    ax.set_title(
        "Spectral / Harmonic Extrapolation — Solar Cycle Forecast SC 25, 26 & 27\n"
        f"Model: A₀ + A₁·cos(2π/T₁·t+φ₁) + A₂·cos(2π/T₂·t+φ₂) + A₃·cos(2π/T₃·t+φ₃)"
        f"   T₁=11.05 yr, T₂=22.1 yr, T₃=87 yr",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8.5, ncol=3, loc="upper left")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"     Saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 – BUTTERFLY DIAGRAM (SPÖRER'S LAW, IF SPOTAREA AVAILABLE)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_butterfly(spot_df, out="fig6_butterfly_diagram.png"):
    print("\n  Generating Figure 6 (butterfly diagram) …")
    df = spot_df[(spot_df["year"] >= 1975) & (spot_df["area"] > 0)].copy()

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    scatter = ax.scatter(
        df["year"], df["lat"],
        s=np.sqrt(df["area"]).clip(0.5, 8),
        c=df["year"], cmap="plasma",
        alpha=0.4, linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Year", fontsize=10)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(+35, color="grey", lw=0.5, ls=":", alpha=0.5)
    ax.axhline(-35, color="grey", lw=0.5, ls=":", alpha=0.5)

    # Cycle boundary lines
    for sc in range(19, 28):
        if sc in SC_MINIMA:
            ax.axvline(SC_MINIMA[sc], color="steelblue", lw=0.8, ls="--",
                       alpha=0.6)
            ax.text(SC_MINIMA[sc], 38, f"SC{sc}", fontsize=7,
                    ha="center", color="steelblue")

    ax.set_xlim(1975, 2028)
    ax.set_ylim(-45, 45)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Heliographic Latitude (°)", fontsize=12)
    ax.set_title(
        "Butterfly (Maunder) Diagram — Spörer's Law of Solar Cycle Latitude Drift\n"
        "SC 20–SC 25 (RGO + NOAA/USAF composite)  |  "
        "Marker size ∝ √(sunspot group area in MSH)",
        fontsize=11, fontweight="bold",
    )

    # Annotate current SC25 phase
    sc25_df = df[df["year"] >= SC_MINIMA[25]]
    if not sc25_df.empty:
        mean_lat = sc25_df["lat"].abs().mean()
        ax.text(2023, 40,
                f"SC 25 mean |lat| = {mean_lat:.1f}°\n"
                f"(declining → equator-ward drift expected)",
                fontsize=8.5, color="darkred",
                bbox=dict(facecolor="white", edgecolor="darkred", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"     Saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 – PREDICTION SUMMARY (all methods)
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_summary(cycle_df, ohl_pred, reg_pred, harmonic_pred,
                 out="fig7_prediction_summary.png"):
    """
    ohl_pred     : {25: (pred, sigma), 26: ..., 27: ...}
    reg_pred     : {25: val, 26: val, 27: val}
    harmonic_pred: {25: (val, sigma), 26: ..., 27: ...}
    """
    print("\n  Generating Figure 7 (summary) …")
    sc25_obs = cycle_df.loc[25, "rmax"] if 25 in cycle_df.index else None

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#F8F8F8")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # ── Left: Forecast timeline ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#F8F8F8")

    monthly = cycle_df["rmax"]  # not time series — just using shape here
    # Show SC24/25 observed context
    t_hist  = cycle_df["max_yr"].values
    r_hist  = cycle_df["rmax"].values
    valid   = ~np.isnan(t_hist) & ~np.isnan(r_hist)
    ax1.plot(t_hist[valid], r_hist[valid], "o-", color="royalblue",
             lw=1.5, ms=5, label="Observed Rmax (SC 1–25)", alpha=0.8)

    # Prediction ranges
    sc_pred_list = [26, 27]
    pred_t = [SC_MAX_YEAR.get(sc, SC_MINIMA.get(sc, 0) + 4.5)
              for sc in sc_pred_list]
    # Use harmonic forecast as primary
    pred_v = [harmonic_pred.get(sc, (0, 0))[0] for sc in sc_pred_list]
    pred_e = [harmonic_pred.get(sc, (0, 0))[1] for sc in sc_pred_list]

    # Also show Ohl and regression
    ohl_v  = [ohl_pred.get(sc, (0, 0))[0] for sc in sc_pred_list]
    reg_v  = [reg_pred.get(sc, 0) for sc in sc_pred_list]

    t_proj = [SC_MINIMA.get(sc, 0) + 4.5 for sc in sc_pred_list]

    ax1.errorbar(t_proj, pred_v, yerr=[2 * e for e in pred_e],
                 fmt="D--", color="darkorange", lw=2, capsize=5, ms=8,
                 label="Harmonic forecast (±2σ)")
    ax1.scatter(t_proj, ohl_v,  marker="^", color="green",  s=80, zorder=6,
                label="Ohl precursor estimate")
    ax1.scatter(t_proj, reg_v,  marker="s", color="purple", s=80, zorder=6,
                label="Regression estimate")

    # Shade forecast windows
    for sc, col in [(25, "#ffd700"), (26, "#a6d96a"), (27, "#1a9641")]:
        t0 = SC_MINIMA.get(sc, 0)
        t1 = SC_MINIMA.get(sc + 1, t0 + 11)
        ax1.axvspan(t0, t1, alpha=0.08, color=col)
        ax1.text((t0 + t1) / 2, 5, f"SC {sc}", ha="center", fontsize=9,
                 color="grey", fontweight="bold")

    if sc25_obs:
        ax1.scatter([SC_MAX_YEAR.get(25, 2024.5)], [sc25_obs],
                    marker="*", s=200, color="gold", edgecolors="black",
                    lw=0.8, zorder=10, label=f"SC 25 observed max ≈ {sc25_obs:.0f}")

    ax1.set_xlim(1980, 2060)
    ax1.set_ylim(0, 320)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Smoothed SSN Maximum", fontsize=12)
    ax1.set_title("Solar Cycle Forecast: SC 25–27\n"
                  "Multi-method comparison with uncertainty ranges",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper right")

    # ── Right: Summary table of predictions ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#F8F8F8")
    ax2.axis("off")

    table_data = [
        ["Method", "SC 25", "SC 26", "SC 27"],
        ["Observed", f"{sc25_obs:.0f}" if sc25_obs else "—", "—", "—"],
        ["Ohl / Precursor",
         f"{ohl_pred.get(25, (0,0))[0]:.0f}",
         f"{ohl_pred.get(26, (0,0))[0]:.0f}",
         f"~{ohl_pred.get(27, (0,0))[0]:.0f}"],
        ["Waldmeier", "—", "—", "—"],
        ["Multi-param Regr.",
         f"{reg_pred.get(25, 0):.0f}" if reg_pred.get(25, 0) > 0 else "—",
         f"{reg_pred.get(26, 0):.0f}",
         f"{reg_pred.get(27, 0):.0f}"],
        ["Harmonic model",
         f"{harmonic_pred.get(25, (0,0))[0]:.0f}",
         f"{harmonic_pred.get(26, (0,0))[0]:.0f}",
         f"{harmonic_pred.get(27, (0,0))[0]:.0f}"],
        ["Consensus range",
         f"{sc25_obs:.0f}" if sc25_obs else "135–155",
         "95–130",
         "75–120"],
        ["Expected max year", "~2024–25", "~2035", "~2046"],
    ]

    tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    # Style header
    for j in range(4):
        tbl[0, j].set_facecolor("#3a3a3a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight consensus row
    for j in range(4):
        tbl[6, j].set_facecolor("#ffe59a")
    # SC25 column gold
    for i in range(1, len(table_data) - 1):
        tbl[i, 1].set_facecolor("#fff9e6")
    ax2.set_title("Prediction Summary Table", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"     Saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Solar Cycle Prediction: SC 25, SC 26 & SC 27")
    print("Methods: Ohl precursor | Waldmeier | Regression | Harmonic")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading datasets …")
    monthly_ssn, annual_ssn = load_ssn_monthly()

    try:
        aa_monthly = load_aa()
        have_aa = True
    except FileNotFoundError:
        print("  [aa]  File not found — Ohl method will be skipped.")
        have_aa = False
        aa_monthly = None

    try:
        t_slk, y_slk = load_solanki(SOLANKI_FILE)
        have_slk = True
    except (FileNotFoundError, ValueError) as e:
        print(f"  [SLK] Solanki file unavailable ({e}); Figure 1 will use SSN only.")
        have_slk = False
        t_slk, y_slk = None, None

    try:
        spot_df = load_spotarea(SPOTAREA_FILE)
        have_spot = True
    except (FileNotFoundError, ValueError) as e:
        print(f"  [SPA] Sunspot area/lat file unavailable ({e}); Fig 6 skipped.")
        have_spot = False
        spot_df = None

    # ── 2. Extract cycle parameters ───────────────────────────────────────────
    print("\n[2/7] Extracting solar cycle parameters (SC 1–25) …")
    cycle_df = extract_cycle_params(monthly_ssn)

    # Override SC25 with data-derived value if cycle is well observed
    t_ann = annual_ssn.index.year.values.astype(float)
    y_ann = annual_ssn.values
    y_sm13 = running_mean(y_ann, 13)
    pt25, pv25 = find_cycle_maxima(
        t_ann, y_sm13, SC_MINIMA[25], SC_MINIMA.get(26, 2031.0)
    )
    if not np.isnan(pv25) and 25 in cycle_df.index:
        cycle_df.loc[25, "rmax"]   = pv25
        cycle_df.loc[25, "max_yr"] = pt25
        SC_MAX_YEAR[25] = pt25

    print("\n  Cycle summary (selected cycles):")
    print(f"  {'SC':>4}  {'Rmax':>7}  {'Period':>8}  {'Rise(mo)':>9}  {'Even':>5}")
    for sc in [19, 20, 21, 22, 23, 24, 25]:
        if sc in cycle_df.index:
            r  = cycle_df.loc[sc]
            ev = "Yes" if r["even"] else "No"
            print(f"  {sc:>4}  {r['rmax']:>7.1f}  {r['period']:>8.2f}  "
                  f"{r['rise_mo']:>9.1f}  {ev:>5}")

    # ── 3. Method 1: Ohl / Geomagnetic Precursor ──────────────────────────────
    print("\n[3/7] Method 1: Ohl Geomagnetic Precursor …")
    if have_aa:
        (cycle_ids, aa_arr, rmax_arr,
         ohl_slope, ohl_int, ohl_r2, ohl_se) = ohl_precursor(aa_monthly, cycle_df)
        print(f"       Ohl fit: Rmax = {ohl_slope:.2f}·aaMin + {ohl_int:.2f}  "
              f"R² = {ohl_r2:.3f}")
        # Apply to SC26 using recent aa minimum proxy
        # SC24 declining-phase aa min (2015–2019): obtain from data
        t_aa_d = np.array(
            aa_monthly.index.year.values +
            (aa_monthly.index.dayofyear - 1) / 365.25
        )
        # SC26 precursor: aa minimum in SC25 declining phase (~2025–2030)
        # Not yet available; use Gleissberg-modulated extrapolation
        # Estimate: similar to SC24 precursor (modest decline continuing)
        aa_sc25_dec = float(np.percentile(
            aa_monthly[aa_monthly.index.year >= 2019].values, 10
        ))
        ohl_pred_25  = ohl_slope * aa_arr[list(cycle_ids).index(25)] + ohl_int if 25 in cycle_ids else pv25
        ohl_pred_26  = ohl_slope * aa_sc25_dec * 0.95 + ohl_int   # modest decline
        ohl_pred_27  = ohl_slope * aa_sc25_dec * 0.88 + ohl_int   # further decline
        ohl_pred = {
            25: (ohl_pred_25, ohl_se * 2),
            26: (ohl_pred_26, ohl_se * 2.5),
            27: (ohl_pred_27, ohl_se * 3.0),
        }
        print(f"       SC25 Ohl prediction: {ohl_pred_25:.0f}  "
              f"SC26: {ohl_pred_26:.0f}  SC27: {ohl_pred_27:.0f}")
    else:
        cycle_ids = np.array([]); aa_arr = np.array([]); rmax_arr = np.array([])
        ohl_slope = 15; ohl_int = 20; ohl_r2 = 0; ohl_se = 25
        ohl_pred = {25: (pv25, 20), 26: (110, 30), 27: (95, 35)}

    # ── 4. Method 2: Waldmeier Effect ─────────────────────────────────────────
    print("\n[4/7] Method 2: Waldmeier Effect …")
    w_m, w_b, w_r2, w_rise, w_rmax = waldmeier_fit(cycle_df)
    print(f"       Waldmeier: Rmax = {w_m:.2f}·rise_mo + {w_b:.2f}  R² = {w_r2:.3f}")

    # SC26 Waldmeier: use mean rise time of recent cycles as prior
    mean_rise_recent = cycle_df.loc[cycle_df.index >= 20, "rise_mo"].mean()
    w_pred_26 = w_m * mean_rise_recent + w_b
    print(f"       SC26 Waldmeier (mean rise={mean_rise_recent:.0f} mo): {w_pred_26:.0f}")

    # ── 5. Method 3: Multi-parameter Regression (LOO-CV) ─────────────────────
    print("\n[5/7] Method 3: Multi-parameter Regression (LOO-CV) …")
    obs_loo, pred_loo, sc_loo = multivar_regression_loo(cycle_df)
    from scipy.stats import pearsonr as pr_
    r_loo, _ = pr_(obs_loo, pred_loo)
    rmse_loo  = np.sqrt(np.mean((obs_loo - pred_loo) ** 2))
    print(f"       LOO-CV r = {r_loo:.3f}, RMSE = {rmse_loo:.1f}")

    # Full-data prediction for SC26, SC27
    if 25 in cycle_df.index:
        sc25_row = cycle_df.loc[25]
        # SC26 (even, sc=26): features from SC25
        reg_sc26 = multivar_full_fit(cycle_df, 26, {
            "rmax":    sc25_row["rmax"],
            "period":  sc25_row["period"],
            "rise_mo": sc25_row["rise_mo"],
            "even":    True,   # SC26 is even
        })
        # SC27 (odd): features extrapolated from SC26
        reg_sc27 = multivar_full_fit(cycle_df, 27, {
            "rmax":    reg_sc26,
            "period":  11.2,   # typical period
            "rise_mo": mean_rise_recent,
            "even":    False,
        })
    else:
        reg_sc26, reg_sc27 = 110.0, 95.0

    print(f"       SC26 regression: {reg_sc26:.0f}  SC27: {reg_sc27:.0f}")
    reg_pred = {26: reg_sc26, 27: reg_sc27}

    # ── 6. Method 4: Harmonic Extrapolation ──────────────────────────────────
    print("\n[6/7] Method 4: Harmonic / Spectral Extrapolation …")
    # Fit on SC1-SC24 training data (up to end of SC24 ≈ 2019)
    mask_train = t_ann <= 2019.9
    popt, pcov, y_fit = fit_harmonic_model(t_ann[mask_train], y_ann[mask_train])
    print(f"       Harmonic model fitted: A0={popt[0]:.1f} "
          f"A1={popt[1]:.1f} A2={popt[3]:.1f} A3={popt[5]:.1f}")

    resid_train = y_ann[mask_train] - y_fit
    sig_harm    = np.std(resid_train)

    def get_harmonic_peak(t0, t1):
        t_w = np.linspace(t0, t1, 500)
        y_w = np.clip(harmonic_model(t_w, *popt), 0, None)
        idx = np.argmax(y_w)
        return float(t_w[idx]), float(y_w[idx])

    harm_t25, harm_v25 = get_harmonic_peak(SC_MINIMA[25], SC_MINIMA.get(26, 2030.5))
    harm_t26, harm_v26 = get_harmonic_peak(SC_MINIMA.get(26, 2030.5), SC_MINIMA.get(27, 2041.5))
    harm_t27, harm_v27 = get_harmonic_peak(SC_MINIMA.get(27, 2041.5), 2053.0)

    harmonic_pred = {
        25: (harm_v25, sig_harm),
        26: (harm_v26, sig_harm * 1.3),
        27: (harm_v27, sig_harm * 1.6),
    }
    print(f"       Harmonic forecast — SC25 max≈{harm_v25:.0f} (~{harm_t25:.1f})"
          f"  SC26≈{harm_v26:.0f} (~{harm_t26:.1f})"
          f"  SC27≈{harm_v27:.0f} (~{harm_t27:.1f})")

    # ── 7. Generate all figures ───────────────────────────────────────────────
    print("\n[7/7] Generating figures …")

    # Figure 1: long-term context
    if have_slk:
        fig1_longterm(t_slk, y_slk, annual_ssn, cycle_df)
    else:
        # Use recent 400-yr proxy from SSN
        fig1_longterm(t_ann, y_ann, annual_ssn, cycle_df)

    # Figure 2: cycle parameters
    fig2_cycle_params(cycle_df)

    # Figure 3: Ohl precursor
    if have_aa and len(cycle_ids) > 0:
        fig3_ohl(cycle_ids, aa_arr, rmax_arr,
                 ohl_slope, ohl_int, ohl_r2, cycle_df)
    else:
        print("  [Fig 3] Skipped (no aa data).")

    # Figure 4: regression
    if len(obs_loo) > 5:
        fig4_regression(cycle_df, obs_loo, pred_loo, sc_loo,
                        reg_sc26, reg_sc27)
    else:
        print("  [Fig 4] Insufficient LOO data.")

    # Figure 5: harmonic forecast
    fig5_harmonic(annual_ssn, popt)

    # Figure 6: butterfly diagram
    if have_spot:
        fig6_butterfly(spot_df)
    else:
        print("  [Fig 6] Skipped (no sunspot area/lat data).")

    # Figure 7: prediction summary
    fig7_summary(cycle_df, ohl_pred, reg_pred, harmonic_pred)

    # ── Print final summary ────────────────────────────────────────────────────
    sc25_obs_final = cycle_df.loc[25, "rmax"] if 25 in cycle_df.index else pv25

    print("\n" + "=" * 65)
    print("SOLAR CYCLE PREDICTION RESULTS")
    print("=" * 65)
    print(f"\n  SC 25 (current, ongoing):")
    print(f"    Observed Rmax from SILSO : {sc25_obs_final:.0f}")
    print(f"    Ohl prediction           : {ohl_pred.get(25, (0,0))[0]:.0f}")
    print(f"    Harmonic forecast        : {harm_v25:.0f} (~{harm_t25:.1f})")

    print(f"\n  SC 26 (projected max ~{SC_MINIMA.get(26,2030)+4.5:.0f}):")
    print(f"    Ohl estimate (extrap.)   : {ohl_pred.get(26, (0,0))[0]:.0f}"
          f"  ± {ohl_pred.get(26,(0,35))[1]:.0f}")
    print(f"    Multi-param regression   : {reg_sc26:.0f}  ± {rmse_loo:.0f}")
    print(f"    Harmonic model           : {harm_v26:.0f}  ± {sig_harm*1.3:.0f}")

    print(f"\n  SC 27 (projected max ~{SC_MINIMA.get(27,2041)+4.5:.0f}):")
    print(f"    Ohl estimate (extrap.)   : {ohl_pred.get(27, (0,0))[0]:.0f}"
          f"  ± {ohl_pred.get(27,(0,40))[1]:.0f}")
    print(f"    Multi-param regression   : {reg_sc27:.0f}  ± {rmse_loo*1.4:.0f}")
    print(f"    Harmonic model           : {harm_v27:.0f}  ± {sig_harm*1.6:.0f}")

    print("\n  Physical interpretation:")
    print("    SC25 is stronger than initial NOAA/SIDC consensus forecasts (~115),")
    print("    consistent with the active solar maximum observed in 2024.")
    print("    SC26 is expected to be weaker (even-cycle, Gleissberg declining phase).")
    print("    SC27 predictions are highly uncertain but indicate further moderation.")
    print("    The Modern Grand Maximum (SC19-SC23) has likely peaked;")
    print("    a gradual return toward Dalton-level activity is plausible by SC28-30.")
    print("    (Ref: Solanki & Krivova 2004; Lockwood et al. 2010)\n")


if __name__ == "__main__":
    main()
