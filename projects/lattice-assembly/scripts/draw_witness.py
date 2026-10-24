"""Draw the checked AD witness as a standalone SVG, without plotting packages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from verify import DIRECTIONS, ORBITS, ROOT, check_witness


def render() -> str:
    witnesses = json.loads((ROOT / "data/witnesses.json").read_text())["witnesses"]
    witness = next(w for w in witnesses if w["family"] == "AD")
    checked = check_witness(witness)
    patterns = {tuple(r["coordinate"]): frozenset(r["directions"]) for r in witness["sites"]}
    points = {v: (v[0] + v[1] / 2, math.sqrt(3) * v[1] / 2) for v in patterns}
    xmin, xmax = min(p[0] for p in points.values()), max(p[0] for p in points.values())
    ymin, ymax = min(p[1] for p in points.values()), max(p[1] for p in points.values())
    scale = min(650 / (xmax - xmin), 320 / (ymax - ymin))
    xy = {v: (400 + scale * (p[0] - (xmin + xmax) / 2),
              245 - scale * (p[1] - (ymin + ymax) / 2)) for v, p in points.items()}
    edges = set()
    for site, pattern in patterns.items():
        for direction in pattern:
            step = DIRECTIONS[direction]
            neighbour = site[0] + step[0], site[1] + step[1]
            edges.add(tuple(sorted((site, neighbour))))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="540" viewBox="0 0 800 540" role="img" aria-labelledby="title desc">',
             '<title id="title">A finite fully bonded A plus D cluster</title>',
             '<desc id="desc">Sixteen triangular lattice particles joined by twenty-four reciprocal bonds. Blue circles use consecutive patch directions; orange triangles use alternating directions.</desc>',
             '<rect width="800" height="540" fill="#f6f8fb" rx="18"/>',
             '<g font-family="Arial, Helvetica, sans-serif" text-anchor="middle">',
             '<text x="400" y="40" font-size="23" font-weight="700" fill="#172a3a">Finite saturation with A + D</text>',
             '<text x="400" y="66" font-size="14" fill="#4c6072">16 particles · 24 reciprocal bonds · all patches bonded</text>']
    for a, b in sorted(edges):
        x1, y1 = xy[a]
        x2, y2 = xy[b]
        parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#6c7f90" stroke-width="3"/>')
    for index, site in enumerate(sorted(patterns), 1):
        x, y = xy[site]
        if patterns[site] in ORBITS[0]:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="15" fill="#176b8c" stroke="white" stroke-width="2"/>')
        else:
            vertices = f'{x:.2f},{y-19:.2f} {x-19:.2f},{y+15:.2f} {x+19:.2f},{y+15:.2f}'
            parts.append(f'<polygon points="{vertices}" fill="#b7601e" stroke="white" stroke-width="2"/>')
        parts.append(f'<text x="{x:.2f}" y="{y+4:.2f}" fill="white" font-size="11">{index}</text>')
    parts.extend(['<circle cx="207" cy="465" r="10" fill="#176b8c"/>',
                  '<text x="337" y="470" font-size="15" fill="#172a3a">A: consecutive directions</text>',
                  '<polygon points="500,453 488,475 512,475" fill="#b7601e"/>',
                  '<text x="630" y="470" font-size="15" fill="#172a3a">D: alternating directions</text>',
                  '<text x="400" y="510" font-size="13" fill="#4c6072">An existence witness; global minimum size is not claimed.</text>',
                  '</g></svg>'])
    if len(edges) != checked["bonds"]:
        raise ValueError("drawing edge count disagrees with checked witness")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/ad-witness.svg")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render())
    print("Saved the checked 16-particle AD witness SVG.")


if __name__ == "__main__":
    main()
