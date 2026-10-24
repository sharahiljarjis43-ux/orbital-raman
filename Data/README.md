# External data

This source update adds no new raw mission or laboratory files. Previously published mission files are retained here with their original contents; that history does not establish a complete input set or independent scientific validation.

For the optional OVIRS scenario pipeline, obtain this product from the OSIRIS-REx Spectral Analysis Bundle in the NASA Planetary Data System:

```text
l_1600mm_sp_ovirs_reca_bandarea3200to3600nm_allsites_wavc_0000n00000.fits
```

Local source metadata identifies OSIRIS-REx Recon A, the OVIRS Level 3 Pipeline v1, product date 2019-11-17, map “Band Area 3.2 - 3.6 um from OVIRS”, and shape file `g_01600mm_spc_obj_0000n00000_v042.obj`. The FITS binary table has 786,432 rows with `FACET_NUM`, `LATITUDE`, `LONGITUDE`, `RADIUS`, `VALUE` and `SIGMA` columns. These are product-identification details, not a chemical calibration.

Use the retained FITS file only if its filename and metadata match this product. Otherwise obtain the identified product from the archive, place it here or pass its path with `main.py --fits`. See [reproduction instructions](../docs/REPRODUCING.md). Use the archive's original labels, attribution and terms, and record the downloaded version and checksum.

The historical standalone model also contains loaders for an OTES file named `20190926T173403S025_ote_emiss_temp.hdf5` and TSIS-1 irradiance named `tsis_ssi_L3_c24h_latest.nc`. Their presence is optional for the documented data-free example. The name “latest” does not fix a version; an exact checksum and acquisition date are required for a published replay.

Unpublished meteorite measurements, raw instrument exports and laboratory collaborators' software require separate provenance and sharing decisions. They have not been added here.
