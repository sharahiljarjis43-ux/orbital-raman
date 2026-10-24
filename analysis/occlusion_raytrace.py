"""Line-of-sight mesh ray-trace of monostatic occlusion on Bennu.

Purpose: empirically anchor the small monostatic self-occlusion allowance in
`src_fixed.SpaceEnvironment` -- the residual signal-yield loss that survives
active illumination (a facet returns nothing if the spacecraft's own near-nadir
line of sight to it is physically blocked by local relief). Source (laser) and
detector are co-aligned, so it is a single term.

Method (addresses adversarial-review must-fixes)
-----------------------------------------------
- Manual OBJ parse (robust to irregular whitespace; asserts triangulated).
- Orient face normals OUTWARD (flip where n . radial < 0) -- do NOT trust raw
  OBJ winding.
- Sample facets; place spacecraft at 375 m near-nadir (per-facet radial, or a
  single pass above a local patch centroid).
- SEGMENT-bounded LOS: cast facet->spacecraft, count a hit only if it is on a
  DIFFERENT triangle (exclude self) AND closer than the spacecraft.
- Report BOTH face-count and AREA-weighted fractions (area-weighted is the
  physical surface fraction), plus an artifact diagnostic.

Meshes (NASA OSIRIS-REx, https://www.asteroidmission.org/updated-bennu-shape-model-3d-files/;
not committed -- ~7 MB / ~80 MB):
  - 75 cm global : Bennu_v20_200k.obj           (OLA v20, 196,608 faces)
  - 5 cm Nightingale Pre-TAG DTM : Bennu-Pre-TAG-DTM.obj  (OLA v20 PTM, Poisson
    reconstruction; the only product fine enough to resolve boulder-scale relief)

FINDING (not a clean single value -- and that is the point)
-----------------------------------------------------------
  75 cm global mesh (2 m faces)       : 0.0%  -- occlusion is sub-resolution.
  5 cm Nightingale DTM (4 cm faces)   : few %, but WEIGHTING-DEPENDENT:
      face-count weighted : ~1.5%   (under-weights larger occluded faces)
      area-weighted (raw)  : ~8.9%   (but 5.0 pts come from large INWARD faces:
                                      Poisson fill-in over unseen boulder
                                      undersides -- reconstruction artifacts,
                                      not real measurable surface)
      artifact-corrected   : ~2-4%   (raw area-weighted minus the inward-face
                                      artifact contribution)

=> The ray-trace CONFIRMS occlusion is a few percent and that the physical value
sits inside the 1-8% sensitivity band -- but it does NOT collapse to a precise
point value, because the boulder DTM's Poisson reconstruction makes the exact
number weighting- and artifact-dependent. The correct treatment is therefore the
retained bracketed band: 3% nominal, 1-8% sensitivity (now empirically anchored
by this ray-trace rather than assumed). Feasibility conclusions are insensitive
across 0-8%. Nightingale is one boulder-rich TAG site, so even this bracket is
site-specific, not a proven global value.

Reproduce:
    python analysis/occlusion_raytrace.py <global.obj> <nightingale.obj>
"""
from __future__ import annotations

import sys
import time

import numpy as np
import trimesh

ALT_KM = 0.375


def load_obj(path: str) -> trimesh.Trimesh:
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            s = line.split()
            if not s:
                continue
            if s[0] == "v":
                verts.append((float(s[1]), float(s[2]), float(s[3])))
            elif s[0] == "f":
                if len(s) != 4:
                    raise ValueError(f"non-triangular face: {line!r}")
                faces.append(tuple(int(t.split("/")[0]) for t in s[1:4]))
    return trimesh.Trimesh(
        vertices=np.asarray(verts, float),
        faces=np.asarray(faces, int) - 1,
        process=False,
    )


def occlusion(mesh: trimesh.Trimesh, kind: str, n_sample=40000, seed=42) -> dict:
    F = len(mesh.faces)
    fc = np.asarray(mesh.triangles_center)
    fn = np.asarray(mesh.face_normals).copy()
    fa = np.asarray(mesh.area_faces)
    radial = fc / np.linalg.norm(fc, axis=1, keepdims=True)

    # unit sanity: centroid radius should be ~0.24 km (Bennu). Assert to catch a
    # metre-vs-km OBJ, which would scale ALT_KM and eps by 1000x.
    r_mean = float(np.linalg.norm(fc, axis=1).mean())
    assert 0.15 < r_mean < 0.35, f"unexpected mesh units: centroid radius {r_mean} km"
    # orient normals outward
    inward = np.sum(fn * radial, axis=1) < 0
    fn[inward] *= -1.0
    edge = float(np.sqrt(mesh.area / F))
    eps = 2.0 * edge

    rng = np.random.default_rng(seed)
    idx = rng.choice(F, min(n_sample, F), replace=False)
    c, nrm, area = fc[idx], fn[idx], fa[idx]

    if kind == "local_pass":
        cen = fc.mean(axis=0)
        sc = np.broadcast_to(cen + ALT_KM * cen / np.linalg.norm(cen), c.shape)
    else:  # per-facet near-nadir
        up = c / np.linalg.norm(c, axis=1, keepdims=True)
        sc = c + ALT_KM * up

    look = sc - c
    sc_dist = np.linalg.norm(look, axis=1)
    look /= sc_dist[:, None]
    self_facing = np.sum(nrm * look, axis=1) <= 0.0

    locs, iray, itri = mesh.ray.intersects_location(
        c + eps * nrm, look, multiple_hits=True
    )
    blocked = np.zeros(len(idx), bool)
    if len(iray):
        d = np.linalg.norm(locs - (c + eps * nrm)[iray], axis=1)
        valid = (itri != idx[iray]) & (d < sc_dist[iray])   # exclude self, bound to spacecraft
        blocked[iray[valid]] = True

    occ = self_facing | blocked
    aw = lambda mask: float((area * mask).sum() / area.sum())
    fsr = (float(area[occ].mean() / area[~occ].mean())
           if occ.any() and (~occ).any() else float("nan"))
    return {
        "faces": F, "sampled": len(idx), "r_mean_km": r_mean,
        "inward_count_frac": float(inward.mean()),
        "inward_area_frac": float(fa[inward].sum() / fa.sum()),
        "occ_face_size_ratio": fsr,
        "facecount": float(occ.mean()),
        "area_weighted": aw(occ),
        # numerator excludes the inward (Poisson-artifact) faces; denominator is
        # the full sampled area (conservative -- artifact area stays in the total).
        "area_weighted_artifact_corrected": aw(occ & ~inward[idx]),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    m_g = load_obj(sys.argv[1])
    m_l = load_obj(sys.argv[2])
    print("global 75cm  :", {k: round(v, 4) for k, v in occlusion(m_g, "per_facet").items()
                             if k in ("facecount", "area_weighted")})
    r = occlusion(m_l, "local_pass")
    print(f"Nightingale 5cm (single near-nadir pass):")
    print(f"  face-count={100*r['facecount']:.2f}%  area-weighted(raw)={100*r['area_weighted']:.2f}%  "
          f"artifact-corrected={100*r['area_weighted_artifact_corrected']:.2f}%")
    print(f"  occluded faces {r['occ_face_size_ratio']:.1f}x larger than visible; "
          f"inward/artifact faces = {100*r['inward_count_frac']:.2f}% by count, "
          f"{100*r['inward_area_frac']:.2f}% by area; centroid r={r['r_mean_km']:.3f} km")
