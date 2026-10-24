# Source and data provenance

## Source snapshot

The instrument and mission calculation source in this update is curated from local Git commit `cf215304f26567e5449b3bb67f542c82bf8719be` (18 July 2026), which contains the later active-illumination, solar-background, pointing-campaign and Raman-coefficient work. It supersedes the older local `main` source for this package. No calculation behavior was intentionally changed by packaging.

Included source consists of `src_fixed.py`, the three analysis entry points, the `analysis` package, and their inherited tests. A small data-free example, environment record and explanatory documents were added for review. The artifact-existence test skips when its intentionally unbundled generated reports are absent.

## AI assistance and authorship

The October 2026 repository audit, documentation, packaging and smoke-check wrapper were prepared with Codex assistance. The source snapshot also has an AI-assisted development history. Repository presentation and passing tests are not independent validation of the research. Scientific interpretation, attribution, authorship and claims remain the researcher's responsibility and should be reviewed before submission.

This update does not claim sole authorship of third-party scientific methods or collaborator software. It does not invent a license for material whose original terms have not been established.

## Included and excluded material

This update adds source and documentation, without adding new raw mission or laboratory inputs. It omits local credentials, editor history, agent worktrees, virtual environments, duplicate local drafts, unrelated Olympiad/statistical-mechanics projects, unpublished meteorite files, collaborator pipeline copies and third-party papers. Original mission product identifiers are documented so users can obtain and cite inputs from their source archive.

Previously published mission files are retained under `Data/`, and older figures, manuscript drafts and interactive demos are retained under `archive/2026-04/`. Their contents and history are preserved; they are not promoted as current validated evidence. Raw-file redistribution and laboratory collaborator permission have not been newly established by this audit.

## Verification scope

The release record in `docs/VERIFICATION.md` reports exactly which checks were run. Data-free checks evaluate numerical and geometric behavior. Optional mission-data replay, laboratory fitting, external-mesh ray tracing and hardware validation are separate checks; they must not be implied by the source-only result.
