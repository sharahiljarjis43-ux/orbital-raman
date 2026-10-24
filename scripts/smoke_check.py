"""Print and check an explicit photon-budget scenario without external data."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src_fixed as model  # noqa: E402


def main() -> None:
    # `organic_pct` is a supplied scenario parameter, not an OVIRS measurement.
    scenario_pct = 5.0
    records = []
    for distance_m in (375.0, 680.0):
        budget = model.calculate_snr_full(distance_m, scenario_pct)
        for key in ("N_sig", "N_solar", "N_dark", "N_read", "N_bg", "snr"):
            if not math.isfinite(float(budget[key])) or budget[key] < 0:
                raise RuntimeError(f"Invalid {key} at {distance_m} m")
        if not math.isclose(
            budget["snr"], model.calculate_snr(distance_m, scenario_pct), rel_tol=1e-9
        ):
            raise RuntimeError("SNR entry points disagree")
        records.append({"distance_m": distance_m, **budget})
    if records[1]["snr"] >= records[0]["snr"]:
        raise RuntimeError("Expected lower SNR at the longer range")
    print(json.dumps({
        "status": "numerical checks passed",
        "interpretation": "Assumed photon-budget scenarios; not measured detection",
        "scenario_organic_pct": scenario_pct,
        "budgets": records,
    }, indent=2))


if __name__ == "__main__":
    main()
