# Orbital Deep-UV Raman Spectroscopy for Asteroid Organic Detection

Can you detect organics on an asteroid from orbit using Raman spectroscopy? This project tries to answer that quantitatively.

## What this is

A photon budget model for 248nm deep-UV resonance-enhanced Raman spectroscopy at asteroid Bennu, calibrated with actual OSIRIS-REx flight data (OVIRS organic maps + OTES thermal measurements). The model evaluates detection feasibility across all OSIRIS-REx mission phase distances (50m to 1600m) and includes Monte Carlo uncertainty propagation, a quantum squeezing analysis, and a g⁽²⁾ photon correlation pathway.

## What I found

- Detection works at close range. SNR ≈ 11 at 375m (Reconnaissance), SNR > 500 at 50m (Sample Collection). At 680m (Orbital B) it's marginal — SNR ≈ 3.4 for 5% organics.
- 248nm DUV excitation gives ~200× enhancement over 532nm visible Raman through combined ν⁴ scaling and resonance enhancement of aromatic organics. Also eliminates fluorescence interference entirely.
- Organic concentration dominates everything. Monte Carlo sensitivity analysis (10,000 trials) gives Spearman ρ = 0.79 — knowing what's on the surface matters more than any instrument improvement.
- Quantum squeezing doesn't help. At the ~4.5% total optical efficiency of a space instrument, 10 dB of ideal squeezing yields 0.18 dB effective. You need η > 30% for it to matter at all. Verified analytically and with QuTiP.
- g⁽²⁾ photon correlation is theoretically interesting but needs ~10⁷ hours at 680m. Future tech, not current.

## Running it

```
pip install numpy matplotlib scipy qutip netCDF4
python src.py
```

Generates 8 figures into `Outputs/figures/`:
- `fig1` – Feasibility contour map (distance × organic concentration)
- `fig2` – Organic detection threshold at three instrument configs
- `fig3` – Quantum squeezing threshold
- `fig4` – Mission phase feasibility matrix
- `fig5` – Bennu realistic environment (6-panel, uses OVIRS data)
- `fig6` – Monte Carlo UQ (4-panel: SNR distribution, detection range, sensitivity, signal vs background)
- `fig7` – Squeezing analysis (effective dB vs efficiency + SNR improvement)
- `fig8` – g⁽²⁾ correlation (source signatures, mixed signal, integration time)

## Data

The `Data/` folder expects three files from NASA PDS and LASP:
- OVIRS organic band area map (`.fits`, ~19 MB, 786,432 spectral records)
- OTES thermal emission temperatures (`.hdf5`)
- TSIS-1 solar spectral irradiance (`.nc`)

If the data files aren't present, the model falls back to synthetic distributions.

## Web simulator

`simulator.html` — open it in any browser. No setup, no dependencies. Includes live SNR calculation, mission phase matrix, squeezing analysis, and NASA JPL asteroid database search.

## Background

Started this in November 2025 after self-studying quantum optics through Saleh & Teich's *Fundamentals of Photonics* and Susskind's *Theoretical Minimum*. The question of whether Raman could work from orbit seemed like it should have a quantitative answer, and the OSIRIS-REx dataset made it possible to calibrate against real measurements instead of assumptions.

## Structure

```
├── src.py                  # Everything — model, figures, MC, squeezing, g²
├── simulator.html          # Standalone web simulator
├── Data/                   # OVIRS, OTES, TSIS-1 flight data
├── Outputs/
│   ├── figures/            # All 8 figure outputs
│   └── results/            # Coverage CSVs
└── paper/
    ├── paper_final.md      # Manuscript draft
    └── paper_final.docx    # Formatted version
```
