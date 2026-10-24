"""Parse the OSIRIS-REx Spectral Analysis Bundle 3.2-3.6 μm band-area product.

The product `l_1600mm_sp_ovirs_reca_bandarea3200to3600nm_allsites_wavc_*.fits`
is the per-facet integrated band area from OVIRS Recon-A spectra mapped onto
Bennu's 1.6 m global digital terrain model (786,432 facets, GSD 1600 mm).

Header documents this product as the OVIRS Level-3 pipeline output for the
Bennu Spectral Analysis Working Group (SAWG). Each facet record holds:

  FACET_NUM  : i4   shape-model facet index
  LATITUDE   : f4   degrees, body-fixed
  LONGITUDE  : f4   degrees, body-fixed (0-360 East)
  RADIUS     : f4   km from Bennu center to the facet
  VALUE      : f4   band area integrated 3200-3600 nm (-9999 = no obs)
  SIGMA      : f4   1-σ uncertainty on VALUE

This script:
1. Loads the FITS table.
2. Drops fill-value (-9999) facets.
3. Returns a tidy pandas DataFrame plus a header dict.

The fraction of valid facets (~6%) reflects Recon-A targeted coverage of the
four candidate sample sites (Nightingale, Osprey, Kingfisher, Sandpiper). This
is by design and matches the published Ferrone (2021) / Praet (2021) datasets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

log = logging.getLogger(__name__)

FILL_VALUE = -9999.0


@dataclass
class OVIRSDataset:
    """Container for parsed OVIRS band-area data."""

    df: pd.DataFrame
    header: dict
    n_facets_total: int
    n_facets_valid: int
    mission_phase: str
    map_name: str
    gsd_mm: float
    source_path: str

    @property
    def coverage_fraction(self) -> float:
        return self.n_facets_valid / self.n_facets_total


def load_band_area_fits(path: str | Path) -> OVIRSDataset:
    """Load the OVIRS Recon-A 3.2-3.6 μm band-area FITS product.

    Parameters
    ----------
    path : str or Path
        Path to the FITS file.

    Returns
    -------
    OVIRSDataset

    Raises
    ------
    FileNotFoundError, ValueError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OVIRS FITS not found: {path}")

    log.info("Loading OVIRS band-area product: %s", path.name)
    with fits.open(path) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise ValueError("Expected a binary table HDU at index 1; got none.")
        primary_hdr = dict(hdul[0].header)
        tbl_hdr = dict(hdul[1].header)
        tbl = hdul[1].data

        required = {"LATITUDE", "LONGITUDE", "RADIUS", "VALUE", "SIGMA"}
        missing = required - set(tbl.names)
        if missing:
            raise ValueError(f"FITS missing required columns: {missing}")

        df_all = pd.DataFrame(
            {
                "facet_num": np.asarray(tbl["FACET_NUM"], dtype=np.int64),
                "latitude_deg": np.asarray(tbl["LATITUDE"], dtype=np.float64),
                "longitude_deg": np.asarray(tbl["LONGITUDE"], dtype=np.float64),
                "radius_km": np.asarray(tbl["RADIUS"], dtype=np.float64),
                "band_area": np.asarray(tbl["VALUE"], dtype=np.float64),
                "band_area_sigma": np.asarray(tbl["SIGMA"], dtype=np.float64),
            }
        )

    n_total = len(df_all)
    valid = df_all["band_area"] > FILL_VALUE + 1.0
    df = df_all.loc[valid].copy().reset_index(drop=True)
    n_valid = len(df)

    df["snr_band_area"] = df["band_area"] / np.maximum(df["band_area_sigma"], 1e-6)

    header = {**primary_hdr, **tbl_hdr}
    ds = OVIRSDataset(
        df=df,
        header=header,
        n_facets_total=n_total,
        n_facets_valid=n_valid,
        mission_phase=str(primary_hdr.get("MPHASE", "Recon A")).strip(),
        map_name=str(primary_hdr.get("MAP_NAME", "Band Area 3.2 - 3.6 um from OVIRS")).strip(),
        gsd_mm=float(primary_hdr.get("GSD", 1600.0)),
        source_path=str(path),
    )

    log.info(
        "OVIRS facets: total=%d valid=%d (coverage %.1f%%); band_area median=%.3f "
        "[5/95 pct = %.3f / %.3f]; median SNR=%.1f",
        n_total,
        n_valid,
        100 * ds.coverage_fraction,
        df["band_area"].median(),
        df["band_area"].quantile(0.05),
        df["band_area"].quantile(0.95),
        df["snr_band_area"].median(),
    )
    return ds


def apply_quality_filter(
    df: pd.DataFrame,
    min_snr: float = 5.0,
    lat_range: tuple[float, float] = (-90.0, 90.0),
) -> pd.DataFrame:
    """Apply photometric / SNR quality cuts.

    The SAB Level-3 product has already been photometrically corrected; the
    upstream OVIRS Level-2 pipeline rejects shadowed and high-emission-angle
    observations (Reuter et al. 2018, Simon et al. 2020). The remaining quality
    handle is per-facet SNR on the band-area measurement itself.

    Parameters
    ----------
    df : DataFrame
        Output of `load_band_area_fits`.
    min_snr : float, default 5
        Minimum band-area SNR (= VALUE / SIGMA on the SAB product). The
        nominal OVIRS spectral SNR in the 3.1-3.6 μm region is ~50
        (Kaplan et al. 2021 §2; Reuter et al. 2018); on the *integrated*
        band area the noise reduces by ~√N_channels, so a band-area SNR
        cut of 5 corresponds to a per-channel spectral SNR comfortably
        above the OVIRS detection threshold for a 5% absorption feature
        (Reuter et al. 2018).
    lat_range : (lo, hi)
        Optional latitude restriction. Default keeps all valid facets.
    """
    mask = (
        (df["snr_band_area"] >= min_snr)
        & (df["latitude_deg"] >= lat_range[0])
        & (df["latitude_deg"] <= lat_range[1])
    )
    out = df.loc[mask].copy().reset_index(drop=True)
    log.info(
        "Quality cut: %d → %d facets (SNR ≥ %.1f, lat in [%g, %g])",
        len(df),
        len(out),
        min_snr,
        lat_range[0],
        lat_range[1],
    )
    return out


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Wrote %d rows to %s", len(df), path)
