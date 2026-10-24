# Provenance and assistance

This is a curated computational companion to *Connectivity and finite-cluster assembly in three-patch lattice gases*. It uses the classification revision dated 11 October 2026 and was prepared for the research portfolio on 24 October 2026.

## Origin of the material

- The 15-family table is extracted from `evidence/classification/family_classification.json` in the manuscript archive.
- Integer witness coordinates are extracted from `evidence/classification/screen_reproduction.json`. Solver timing, finite-box status and objectives are omitted because they are not required to check the saved existence witnesses. Their omission does not turn finite-box solver results into global minimum-size claims.
- The proof and base coordinate tables are adapted from `evidence/classification/publication_proof.md`. The separate thermodynamic corollary is omitted because its supporting component-expansion argument is outside this focused package.
- The checker is reorganized from the archive's `source/verify_classification.py`, retaining the same reciprocal-patch and connectivity checks. The SVG drawing script was prepared for this portfolio release.

The [source manifest](data/source-manifest.json) records hashes of those source artifacts. The generated verification file separately hashes the packaged inputs. Original research files were preserved.

## AI assistance and interpretation

The October classification theorem, supporting arguments and computational review were developed with AI assistance. The current README, package curation, checker refactoring and SVG script were also prepared with AI assistance. These additions should not be represented as work completed independently before the assisted sessions.

The supplied manuscript records earlier student work separately from the October additions. Independent refereeing and literature-priority assessment remain outstanding.

## Scope of the release

The package contains mathematical definitions, derived coordinates, source code and generated checks. Private author fields, submission forms, supplied manuscript PDFs, downloaded articles, conversation transcripts and unrelated numerical archives are excluded. No experimental data or collaborator-owned measurements are included.
