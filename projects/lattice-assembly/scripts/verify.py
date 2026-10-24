"""Check integer existence witnesses; analytical exclusions are in the proof.

Run from any directory. Only the generated results file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
REPRESENTATIVES = ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 4))
ORBITS = tuple(
    {frozenset((direction + rotation) % 6 for direction in pattern)
     for rotation in range(6)}
    for pattern in REPRESENTATIVES
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_witness(witness: dict) -> dict:
    """Validate one finite component using exact integer lattice arithmetic."""
    family = witness["family"]
    mask = witness["mask"]
    require(type(mask) is int and 1 <= mask <= 15, f"{family}: invalid mask")
    allowed = set().union(*(orbit for j, orbit in enumerate(ORBITS) if mask >> j & 1))
    patterns = {}
    for site in witness["sites"]:
        coordinate, directions = site["coordinate"], site["directions"]
        require(len(coordinate) == 2 and all(type(v) is int for v in coordinate),
                f"{family}: coordinates must be two integers")
        require(len(directions) == 3 and len(set(directions)) == 3
                and all(type(v) is int and 0 <= v < 6 for v in directions),
                f"{family}: each site needs three distinct direction indices")
        coordinate = tuple(coordinate)
        pattern = frozenset(directions)
        require(coordinate not in patterns, f"{family}: duplicate site {coordinate}")
        require(pattern in allowed, f"{family}: disallowed pattern at {coordinate}")
        patterns[coordinate] = pattern
    require(bool(patterns), f"{family}: empty witness")

    graph = {coordinate: set() for coordinate in patterns}
    for coordinate, pattern in patterns.items():
        for direction in pattern:
            step = DIRECTIONS[direction]
            neighbour = coordinate[0] + step[0], coordinate[1] + step[1]
            require(neighbour in patterns, f"{family}: missing neighbour {neighbour}")
            require((direction + 3) % 6 in patterns[neighbour],
                    f"{family}: nonreciprocal bond {coordinate} -> {neighbour}")
            graph[coordinate].add(neighbour)

    visited, pending = set(), [next(iter(graph))]
    while pending:
        site = pending.pop()
        if site not in visited:
            visited.add(site)
            pending.extend(graph[site] - visited)
    require(len(visited) == len(graph), f"{family}: disconnected component")
    bonds = sum(map(len, graph.values())) // 2
    require(2 * bonds == 3 * len(graph), f"{family}: incomplete saturation")
    contacts = sum((b[0] - a[0], b[1] - a[1]) in DIRECTIONS
                   for a, b in combinations(graph, 2))
    return {"mask": mask, "particles": len(graph), "bonds": bonds,
            "nearest_neighbour_contacts": contacts, "connected": True,
            "all_patches_reciprocal": True, "deficit": 0}


def verify() -> dict:
    require(tuple(map(len, ORBITS)) == (6, 6, 6, 2), "incorrect rotation orbits")
    all_patterns = set().union(*ORBITS)
    require(all_patterns == {frozenset(p) for p in combinations(range(6), 3)},
            "orbits do not partition all twenty three-patch patterns")
    require(sum(map(len, ORBITS)) == len(all_patterns), "overlapping orbits")

    family_path = ROOT / "data/families.json"
    witness_path = ROOT / "data/witnesses.json"
    families = json.loads(family_path.read_text())["families"]
    witnesses = json.loads(witness_path.read_text())["witnesses"]
    require(len(families) == 15 and {r["mask"] for r in families} == set(range(1, 16)),
            "family table must cover all fifteen masks exactly once")
    for row in families:
        mask = row["mask"]
        name = "".join(letter for j, letter in enumerate("ABCD") if mask >> j & 1)
        require(row["family"] == name, f"incorrect family name for mask {mask}")
        require(row["pattern_count"] == sum(len(o) for j, o in enumerate(ORBITS)
                                             if mask >> j & 1),
                f"{name}: incorrect pattern count")
        # This checks transcription of the analytical theorem, not its proof.
        require(row["finite_saturated_cluster_exists"] == bool(mask & 1 and mask & 14),
                f"{name}: classification disagrees with the stated theorem")

    positive = {r["family"] for r in families if r["finite_saturated_cluster_exists"]}
    require(len(witnesses) == 7 and {w["family"] for w in witnesses} == positive,
            "witnesses must cover exactly the seven positive families")
    checks = {w["family"]: check_witness(w) for w in witnesses}
    require(all(checks[r["family"]]["mask"] == r["mask"] for r in families
                if r["family"] in checks), "witness and family masks disagree")

    return {
        "patterns_partitioned": 20,
        "rotation_orbit_sizes": list(map(len, ORBITS)),
        "classified_families": 15,
        "verified_witnesses": 7,
        "negative_families": 8,
        "negative_cases_basis": "Analytical all-size proof; not established by this checker.",
        "global_minimum_sizes_claimed": False,
        "checks": checks,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (family_path, witness_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/verification.json")
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: 7 verified witnesses / 15 classified families; 20 patterns partitioned.")
    print("The eight negative cases depend on the analytical proof.")


if __name__ == "__main__":
    main()
