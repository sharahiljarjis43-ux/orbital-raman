# Research portfolio

**Sharahil Jarjis · Scientific computing, spectroscopy, and mathematical modeling**

This collection brings together four current research projects, with a short reading route, executable checks, and explicit limits on the conclusions. The repository began with orbital Raman modeling; the additional projects live in `projects/`.

## Projects

| Project | Research question | Methods and evidence | Start here |
| --- | --- | --- | --- |
| **Orbital Raman spectroscopy** | How do photon budget, instrument losses, surface heterogeneity, and observing geometry constrain a Raman campaign at asteroid Bennu? | Python instrument model, uncertainty propagation, public OSIRIS-REx map analysis, and spatial-sampling studies. Results are conditional feasibility estimates. | [Project overview](README.md) · [Methods and limitations](docs/METHODS_AND_LIMITATIONS.md) |
| **Lattice assembly** | Which local bonding patterns can form connected, fully bonded structures in a specified three-patch lattice model? | Finite combinatorial classification, exact integer witnesses, and reproducible verification. Conclusions apply to the stated model and candidate family. | [Overview, proof, and checks](projects/lattice-assembly/README.md) |
| **Quantum binding memory** | Can higher-order temporal measurements distinguish hidden kinetic models that agree at lower order? | Finite-state forward models, exact rational checks, moment-order analysis, and synthetic inference examples. Exploratory model study. | [Overview, assumptions, and checks](projects/quantum-binding/README.md) |
| **Potts/TRG prototype** | How can an exact Potts-model benchmark constrain a tensor-renormalization calculation? | Exact checks and finite-bond-dimension numerical experiments. Precision calibration remains unfinished. | [Prototype and validation limits](projects/potts-trg/README.md) |

## Suggested reading route

1. Begin with **orbital Raman**, the closest connection to remote sensing and mission design. The README identifies runnable entry points and the role of public mission data.
2. Open **lattice assembly** for an example of mathematical analysis tied to executable verification.
3. Open **quantum binding memory** for an example of identifiability, stochastic dynamics, and a carefully bounded synthetic study.
4. Open **Potts/TRG** for an earlier numerical prototype with exact controls and a documented convergence limit.

Each project documents its own dependencies and verification commands. Saved numerical outputs are examples from specified models; reproducing a calculation does not establish that its physical assumptions are experimentally validated.

## Research status and contributions

**Project owner and maintainer: Sharahil Jarjis.** These are working research materials. Each project's development notes record source credits, tool assistance and verification scope. See [project attribution and tool use](AUTHORSHIP.md).

The Raman package connects to separate laboratory work with meteorite spectroscopy. Unpublished laboratory measurements and third-party research drafts are outside this public collection. The other projects use mathematical models and computational examples.

## Repository map

- [`README.md`](README.md): Raman overview and setup.
- `analysis/`, `tests/`, and the root Python entry points: Raman model and analyses.
- [`Data/README.md`](Data/README.md): public mission-data context.
- [`projects/lattice-assembly/`](projects/lattice-assembly/README.md): self-contained lattice study.
- [`projects/quantum-binding/`](projects/quantum-binding/README.md): self-contained quantum-binding study.
- [`projects/potts-trg/`](projects/potts-trg/README.md): exact benchmarks and the numerical prototype.
- [`archive/2026-04/`](archive/2026-04/README.md): preserved earlier figures, demonstrations, and manuscript draft.

**Updated:** October 26, 2026.
