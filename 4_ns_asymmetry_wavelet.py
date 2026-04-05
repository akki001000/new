"""
Script 4: Wavelet Power Spectrum of the N–S Asymmetry Index
============================================================
Applies a continuous Morlet wavelet transform (CWT) to the monthly
N–S asymmetry time series A(t) = (N−S)/(N+S) and displays:
  - Top panel   : the raw + smoothed asymmetry signal
  - Middle panel : the normalised wavelet power spectrum (period vs time)
  - Bottom panel : the global (time-averaged) wavelet power spectrum

A cone of influence (COI) marks the region where edge effects are significant.
Statistically significant periodicities (>95 % vs red-noise background) are
contoured in white.

Data source (same as Script 3):
  SILSO monthly hemispheric sunspot numbers
  https://www.sidc.be/SILSO/DATA/SN_hem_m_tot_V2.0.txt

References:
  Torrence & Compo (1998) BAMS 79(1): 61–78  — wavelet methodology
  PyWavelets / pywt documentation

Requirements:
  pip install requests pandas matplotlib numpy scipy PyWavelets
"""

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pywt
from scipy.stats import chi2

# ── Configuration ─────────────────────────────────────────────────────────────
SILSO_HEM_URL = "https://www.sidc.be/SILSO/DATA/SN_hem_m_tot_V2.0.txt"
SMOOTH_MONTHS = 13
WAVELET       = "cmor1.5-1.0"      # complex Morlet: bandwidth=1.5, centre freq=1.0
SIGNIFICANCE_LEVEL = 0.95           # 95 % confidence against red noise


# ── Data helpers ──────────────────────────────────────────────────────────────
def download_asymmetry() -> tuple[np.ndarray, np.ndarray]:
    """Download SILSO hemispheric data, return (decimal_years, asymmetry) arrays."""
    print(f"Downloading SILSO hemispheric data ...")
    try:
        r = requests.get(SILSO_HEM_URL, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    rows = []
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            sn_n  = float(parts[3])
            sn_s  = float(parts[4])
            if sn_n < 0 or sn_s < 0:
                continue
            denom = sn_n + sn_s
            asym  = (sn_n - sn_s) / denom if denom > 0 else np.nan
            rows.append((year + (month - 0.5) / 12, asym))
        except (ValueError, IndexError):
            continue

    arr = np.array(rows)
    t, a = arr[:, 0], arr[:, 1]

    # Fill isolated NaNs with linear interpolation
    mask = np.isfinite(a)
    a = np.interp(t, t[mask], a[mask])
    return t, a


# ── Morlet wavelet analysis ───────────────────────────────────────────────────
def morlet_cwt(signal: np.ndarray, dt: float = 1.0 / 12.0):
    """
    Compute CWT power, periods, COI and 95% significance level.
    dt : sampling interval in years (default 1/12 for monthly data).
    Returns: (power, periods_yr, coi_yr, sig95)

    Scale relationship (pywt with sampling_period=dt):
        scale = center_freq * period_in_years / dt
    where center_freq = pywt.scale2frequency(WAVELET, 1).
    """
    N = len(signal)

    # Target periods from 6 months to ~200 years (log-spaced)
    target_periods = np.logspace(np.log10(0.5), np.log10(200.0), 200)
    center_freq    = pywt.scale2frequency(WAVELET, 1)
    scales         = center_freq * target_periods / dt

    # pywt.cwt returns (coef, freqs); with sampling_period=dt, freqs are in cycles/year
    coef, freqs = pywt.cwt(signal, scales, WAVELET, sampling_period=dt)
    power   = np.abs(coef) ** 2     # shape (n_scales, N)
    periods = 1.0 / freqs           # convert to years (matches target_periods)

    # Cone of influence: distance from each edge, converted to equivalent period
    t_idx      = np.arange(N) * dt                         # time array in years
    coi        = np.minimum(t_idx, t_idx[-1] - t_idx)     # distance to nearest edge
    coi_period = coi / np.sqrt(2)                          # COI in years (Morlet e-fold)

    # Normalise power by signal variance (guard against zero variance)
    var        = max(np.var(signal), 1e-10)
    power_norm = power / var

    # Red-noise significance: lag-1 autocorrelation background (Torrence & Compo 1998)
    ac1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    ac1 = max(float(ac1), 0.0)
    background = np.array([
        var * (1 - ac1 ** 2) / (1 - 2 * ac1 * np.cos(2 * np.pi * dt / p) + ac1 ** 2)
        for p in periods
    ])
    dof      = 2                       # chi-squared DOF for complex Morlet wavelet
    chisq_95 = chi2.ppf(SIGNIFICANCE_LEVEL, dof) / dof
    sig95    = np.outer(background * chisq_95 / var, np.ones(N))

    return power_norm, periods, coi_period, sig95


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_wavelet(t: np.ndarray, signal: np.ndarray,
                 power: np.ndarray, periods: np.ndarray,
                 coi: np.ndarray, sig95: np.ndarray) -> None:

    # Smooth signal for top panel
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(signal, SMOOTH_MONTHS)

    fig, axes = plt.subplots(3, 1, figsize=(16, 13),
                             gridspec_kw={"height_ratios": [1.5, 3.5, 2]})
    fig.suptitle(
        "Wavelet Power Spectrum of North–South Sunspot Asymmetry A(t) = (N−S)/(N+S)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Signal ───────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, signal, color="steelblue", lw=0.7, alpha=0.6, label="Monthly A(t)")
    ax.plot(t, smooth,  color="black",     lw=1.4, label=f"{SMOOTH_MONTHS}-month smooth")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("Asymmetry A(t)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Wavelet power ───────────────────────────────────────────
    ax = axes[1]
    T_grid, t_grid = np.meshgrid(periods, t, indexing="ij")
    pcm = ax.pcolormesh(t_grid, T_grid, power,
                        cmap="jet", shading="auto",
                        vmin=0, vmax=np.percentile(power, 97))
    cbar = fig.colorbar(pcm, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("Normalised Power", fontsize=9)

    # Significance contour
    ax.contour(t_grid, T_grid, power - sig95, levels=[0],
               colors="white", linewidths=1.2)

    # Cone of influence
    ax.fill_between(t, coi, np.max(periods), alpha=0.35,
                    color="lightgray", hatch="//", label="COI")
    ax.fill_between(t, 0, coi, alpha=0)    # keep y-axis correct

    ax.set_yscale("log")
    ax.set_ylabel("Period (years)", fontsize=10)
    ax.set_ylim(periods.min(), periods.max())
    ax.invert_yaxis()
    ax.set_yticks([0.5, 1, 2, 5, 11, 22, 50, 100])
    ax.set_yticklabels(["0.5", "1", "2", "5", "11", "22", "50", "100"])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)

    # Annotate known periods
    for period, label in [(1.0, "1 yr"), (2.0, "QBO~2 yr"),
                          (11.0, "Schwabe~11 yr"), (22.0, "Hale~22 yr")]:
        if periods.min() <= period <= periods.max():
            ax.axhline(period, color="yellow", lw=0.8, ls="--", alpha=0.7)
            ax.text(t[-1], period, f" {label}", color="yellow",
                    fontsize=7, va="center")

    # ── Panel 3: Global power spectrum ───────────────────────────────────
    ax = axes[2]
    global_power = power.mean(axis=1)
    ax.semilogx(global_power, periods, color="navy", lw=1.8,
                label="Global wavelet power")
    ax.set_xlabel("Global Wavelet Power", fontsize=10)
    ax.set_ylabel("Period (years)", fontsize=10)
    ax.set_ylim(periods.min(), periods.max())
    ax.invert_yaxis()
    ax.set_yticks([0.5, 1, 2, 5, 11, 22, 50, 100])
    ax.set_yticklabels(["0.5", "1", "2", "5", "11", "22", "50", "100"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Mark periods of interest
    for p, lbl in [(11.0, "~11 yr"), (22.0, "~22 yr"), (2.0, "~2 yr (QBO)")]:
        if periods.min() <= p <= periods.max():
            ax.axhline(p, color="red", lw=0.9, ls="--", alpha=0.7)
            ax.text(global_power.max() * 0.5, p, f" {lbl}", color="red", fontsize=8)

    # Shared x-axis formatting for panels 1 & 2
    for a in axes[:2]:
        a.set_xlim(t[0], t[-1])
        a.xaxis.set_major_locator(plt.MultipleLocator(10))
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    axes[-1].set_xlim(left=0)

    plt.tight_layout()
    out = "4_ns_asymmetry_wavelet.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t, signal = download_asymmetry()
    print(f"Records: {len(t)},  span: {t[0]:.1f} – {t[-1]:.1f} yr")

    dt = np.median(np.diff(t))          # ~1/12 yr
    print(f"Sampling interval: {dt:.4f} yr  ({dt*12:.2f} months)")

    print("Computing Morlet CWT …")
    power, periods, coi, sig95 = morlet_cwt(signal, dt=dt)
    print("Plotting …")
    plot_wavelet(t, signal, power, periods, coi, sig95)
