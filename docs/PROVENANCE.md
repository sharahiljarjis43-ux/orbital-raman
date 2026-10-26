# Source and data provenance

## Source snapshot

The instrument and mission calculation source in this update is curated from local Git commit `cf215304f26567e5449b3bb67f542c82bf8719be` (18 July 2026), which contains the later active-illumination, solar-background, pointing-campaign and Raman-coefficient work. It supersedes the older local `main` source for this package. No calculation behavior was intentionally changed by packaging.

Included source consists of `src_fixed.py`, the three analysis entry points, the `analysis` package, and their inherited tests. A small data-free example, environment record and explanatory documents were added for review. The artifact-existence test skips when its intentionally unbundled generated reports are absent.

## Development tools

Codex assisted the October repository documentation, packaging, checks and smoke-check wrapper. Earlier model development also used AI tools. The selected scientific calculation source was preserved unchanged; the verification record defines the checks performed.

Project ownership and maintenance are attributed to Sharahil Jarjis; see [project attribution](../AUTHORSHIP.md). Third-party methods, data and software retain their source credits and applicable terms.

## Included and excluded material

This update adds source and documentation, without adding new raw mission or laboratory inputs. It omits local credentials, editor history, agent worktrees, virtual environments, duplicate local drafts, unrelated Olympiad/statistical-mechanics projects, unpublished meteorite files, collaborator pipeline copies and third-party papers. Original mission product identifiers are documented so users can obtain and cite inputs from their source archive.

Previously published mission files are retained under `Data/`, and older figures, manuscript drafts and interactive demos are retained under `archive/2026-04/`. Their contents and history are preserved; they are not promoted as current validated evidence. Raw-file redistribution and laboratory collaborator permission have not been newly established by this audit.

## Verification scope

The release record in `docs/VERIFICATION.md` reports exactly which checks were run. Data-free checks evaluate numerical and geometric behavior. Optional mission-data replay, laboratory fitting, external-mesh ray tracing and hardware validation are separate checks; they must not be implied by the source-only result.
