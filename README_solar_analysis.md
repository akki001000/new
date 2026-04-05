# MSc Thesis — Solar Activity Spatio-Temporal Analysis Scripts
## Group C: Spatio-Temporal Plots

These 7 standalone Python scripts produce publication-quality figures for an
MSc thesis on *"Investigation of Spatio-Temporal Periodicities and Variability
in Solar Activity"*.

Each script **automatically downloads** real data from publicly available
archives (NASA MSFC, SILSO/SIDC) — no manual data preparation needed.

---

## Quick Start

```bash
# Install dependencies (once)
pip install requests pandas matplotlib numpy scipy PyWavelets

# Run any script individually
python3 1_butterfly_diagram.py
python3 2_butterfly_colored_area.py
python3 3_ns_asymmetry_timeseries.py
python3 4_ns_asymmetry_wavelet.py
python3 5_latitude_drift_rate.py
python3 6_active_latitude_bandwidth.py
python3 7_sunspot_area_latitude_scatter.py
```

> **Tip — caching:** Scripts 1, 2, 5, 6 and 7 all share a local cache file
> `msfc_sunspot_groups.csv`. Run **Script 1 first** to populate the cache;
> subsequent scripts load it instantly without re-downloading.

---

## Script Reference

| # | Script | What it shows | Data source |
|---|--------|--------------|-------------|
| 1 | `1_butterfly_diagram.py` | Classic Spörer butterfly diagram (sunspot latitude vs time, 1874–2024) | NASA MSFC combined RGO+USAF |
| 2 | `2_butterfly_colored_area.py` | Butterfly diagram where point colour/size encodes sunspot area (MSH) | NASA MSFC combined RGO+USAF |
| 3 | `3_ns_asymmetry_timeseries.py` | N–S hemispheric asymmetry index A(t) = (N−S)/(N+S) + cumulative | SILSO/SIDC hemispheric SN |
| 4 | `4_ns_asymmetry_wavelet.py` | Morlet CWT power spectrum of A(t) with COI and 95% significance | SILSO/SIDC hemispheric SN |
| 5 | `5_latitude_drift_rate.py` | Equatorward drift rate (°/yr) per solar cycle via linear regression | NASA MSFC combined RGO+USAF |
| 6 | `6_active_latitude_bandwidth.py` | Active latitude band width (90th–10th %ile) per cycle per hemisphere | NASA MSFC combined RGO+USAF |
| 7 | `7_sunspot_area_latitude_scatter.py` | Sunspot group area vs latitude scatter + KDE contours per cycle | NASA MSFC combined RGO+USAF |

---

## Data Sources

### 1. NASA MSFC Combined Sunspot Group Data (Scripts 1, 2, 5, 6, 7)
- **URL pattern:** `https://solarscience.msfc.nasa.gov/greenwch/<YYYY>.txt`
- **Coverage:** 1874 (RGO) – present (USAF/NOAA)
- **Key columns used:** year, month, day, Stonyhurst latitude, projected area (MSH)
- **Reference:** Hathaway, D.H. (2015), *Living Reviews in Solar Physics*, 12(1)

### 2. SILSO Hemispheric Monthly Sunspot Numbers (Scripts 3, 4)
- **URL:** `https://www.sidc.be/SILSO/DATA/SN_hem_m_tot_V2.0.txt`
- **Coverage:** 1992 – present
- **Key columns:** Year, Month, SN_North, SN_South, SN_Total
- **Reference:** Clette, F. & Lefèvre, L. (2016), *Solar Physics*, 291(9–10)

---

## Output Files

Each script saves a high-resolution PNG:

| File | Description |
|------|-------------|
| `1_butterfly_diagram.png` | Classic butterfly (blue dots) |
| `2_butterfly_colored_area.png` | Dark-background butterfly, hot-colourmap by area |
| `3_ns_asymmetry_timeseries.png` | 3-panel: hemispheres, asymmetry, cumulative |
| `4_ns_asymmetry_wavelet.png` | 3-panel: signal, wavelet power, global spectrum |
| `5_latitude_drift_rate.png` | Per-cycle panels + summary drift-rate bar chart |
| `6_active_latitude_bandwidth.png` | Per-cycle panels + summary band-width bar chart |
| `7_sunspot_area_latitude_scatter.png` | Per-cycle scatter + summary mean-latitude panel |
| `msfc_sunspot_groups.csv` | Local cache (auto-created; delete to force re-download) |

---

## Thesis Chapter Mapping

| Script | Relevant Thesis Chapter |
|--------|------------------------|
| 1, 2 | Chapter 6 — Spatio-Temporal Analysis (Butterfly Diagrams) |
| 3 | Chapter 6 — North–South Asymmetry |
| 4 | Chapter 5 — Periodicity Analysis (Wavelet) |
| 5, 6 | Chapter 6 — Spörer's Law & Latitude Drift |
| 7 | Chapter 6 — Area-Latitude Relationships |

---

## Requirements

```
requests>=2.28
pandas>=1.5
matplotlib>=3.6
numpy>=1.23
scipy>=1.9
PyWavelets>=1.4
```
