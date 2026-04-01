# Orbital Deep-UV Raman Spectroscopy for Asteroid Organic Detection

Photon budget model for assessing the feasibility of detecting organics
on asteroid Bennu from orbit using 248nm resonance-enhanced Raman
spectroscopy. Calibrated with OSIRIS-REx OVIRS and OTES flight data.

## Key Results
- Detection feasible at <500m (Reconnaissance phase)
- Marginal at 680m (Orbital B), not feasible beyond ~800m
- Quantum squeezing ineffective at <30% optical efficiency
- Organic concentration dominates detection uncertainty (ρ = 0.79)

## Quick Start
```
pip install -r requirements.txt
python src.py
```
Generates 8 figures into `Outputs/figures/`.

## Data
OVIRS and OTES data products from NASA PDS Small Bodies Node.
TSIS-1 solar spectral irradiance from LASP.

## Paper
See `paper/paper_final.md` for the full manuscript draft.

## Web Simulator
Interactive browser-based tool: `simulator.jsx`
EOF
