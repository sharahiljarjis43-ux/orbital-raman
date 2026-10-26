# Provenance

The source and four saved critical-line logs come from the local `Yau/potts_kappa` archive, imported on 9 September 2026 in commit `613a00a`. The original computation files were preserved in that archive. [source-manifest.json](source-manifest.json) records their original hashes.

**Project owner and maintainer: Sharahil Jarjis.** AI tools assisted implementation, numerical investigation and preparation of this release, including documentation and the smoke check. See [project attribution](../../AUTHORSHIP.md).

For this release, `critline.py`'s module description was revised to distinguish the exact derivative identity from unconverged numerical approximations. `selfdual.py` was given a main guard and its stale statement that the triangular critical line was unavailable was corrected. Other numerical source routines and the four included logs were copied unchanged.

The README and small smoke check were prepared for this release. The smoke check rechecks algebraic/numerical controls and summarizes existing logs; no high-D campaign, extrapolation or new precision result was generated.

The original narrative report, conversation transcript, unrelated project plans, plotting dependencies, duplicate ZIPs and broader historical run collections are excluded. The README carries forward the source import's concrete uncertainty caveat: the triangular D32-to-D48 change exceeds the older ±0.00003 claim.
