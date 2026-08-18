# -*- coding: utf-8 -*-
"""
Generates the golden reference results of the 5R1C thermal zone from the
PRE-MIGRATION `tsib.energysystem` stack.

This script is the oracle capture step of the oemof-solph migration: it is
run once, while the old framework still exists, so that the new solph-based
zone can be checked against it after the old stack has been deleted. It is
kept in the repository for provenance - re-running it requires checking out
the commit recorded in golden_meta.json.

Both the model *inputs* (the stochastic occupancy-derived profiles) and the
*outputs* are stored, so the parity test is hermetic: it does not need to
re-run tsorb and cannot drift if the occupancy model changes.

Usage
-----
    SOLVER=highs uv run python test/data/golden/_generate.py
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, TEST_DIR)

import tsib.energysystem as es  # noqa: E402
from conftest import zone_cfg_with_occupancy  # noqa: E402

# reference building of the whole heat load test suite
BUILDING_IX = 24
SHORT_STEPS = 168

# profiles which are stochastic (tsorb) and therefore pinned by the fixture
INPUT_KEYS = ["Q_ig", "occ_nothome", "occ_sleeping"]
# full result time series stored for the short horizon
TIMESERIES_COLUMNS = ["Heating Load", "Cooling Load", "T_air", "T_s", "T_m", "T_e"]


def _solve(cfg):
    """Runs the pre-migration zone exactly as Building._get_heatload_profile
    does: no refurbishment, no attached buses, heat priced internally."""
    model = es.EnergySystemModel(cfg["weather"].index, wacc=cfg["WACC"])
    zone = model.add(es.ThermalZone5R1C("thermalzone", cfg, refurbishment=False))
    model.solve(tee=False)
    results = model.results("thermalzone")
    results["objective"] = model.objective_value
    results["n_variables"] = sum(
        1 for _ in model.pyomo_model.component_data_objects(ctype=None, active=True)
    )
    return zone, results


def _inputs_frame(cfg, n):
    frame = pd.DataFrame(index=cfg["weather"].index[:n])
    for key in INPUT_KEYS:
        values = cfg[key]
        values = values.values if hasattr(values, "values") else np.asarray(values)
        frame[key] = values[:n]
    frame["T_e"] = cfg["weather"]["T"].values[:n]
    return frame


def main():
    os.makedirs(HERE, exist_ok=True)

    # --- full year, the production path ------------------------------------
    cfg, building_id = zone_cfg_with_occupancy(ix=BUILDING_IX)
    n_year = len(cfg["weather"])
    print("Solving full year ({} steps) with the pre-migration stack...".format(n_year))
    _, year = _solve(cfg)

    # gzipped and rounded to 6 significant digits: the year-long series are
    # only compared on aggregates, and full precision would cost ~800 kB
    _inputs_frame(cfg, n_year).to_csv(
        os.path.join(HERE, "zone_year_inputs.csv.gz"), float_format="%.6g"
    )
    year["timeseries"][["Heating Load"]].to_csv(
        os.path.join(HERE, "zone_year.csv.gz"), float_format="%.6g"
    )

    heat = year["timeseries"]["Heating Load"]
    cool = year["timeseries"]["Cooling Load"]
    monthly = heat.groupby(heat.index.month).sum()

    aggregates = {
        "annual_heat_kWh": float(heat.sum()),
        "annual_cool_kWh": float(cool.sum()),
        "specific_heat_kWh_per_m2": float(heat.sum() / cfg["A_ref"]),
        "monthly_heat_kWh": {str(m): float(v) for m, v in monthly.items()},
        "design_heat_load_kW": float(year["static"]["Capacity"]),
        "objective": float(year["objective"]),
        "max_load_violation_kW": float(year["max_load_violation"] or 0.0),
        "T_air_min": float(year["timeseries"]["T_air"].min()),
        "T_air_max": float(year["timeseries"]["T_air"].max()),
    }
    with open(os.path.join(HERE, "zone_year_aggregates.json"), "w") as handle:
        json.dump(aggregates, handle, indent=2, sort_keys=True)

    # --- short horizon, full detail ----------------------------------------
    cfg_short, _ = zone_cfg_with_occupancy(ix=BUILDING_IX, n_steps=SHORT_STEPS)
    print("Solving {} h horizon...".format(SHORT_STEPS))
    _, short = _solve(cfg_short)

    _inputs_frame(cfg_short, SHORT_STEPS).to_csv(
        os.path.join(HERE, "zone_168h_inputs.csv")
    )
    short["timeseries"][TIMESERIES_COLUMNS].to_csv(
        os.path.join(HERE, "zone_168h.csv"), float_format="%.9g"
    )

    # --- provenance ---------------------------------------------------------
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TEST_DIR
    ).decode().strip()
    meta = {
        "generated_from_commit": commit,
        "generated_by": os.path.relpath(__file__, TEST_DIR),
        "stack": "tsib.energysystem (pre-migration)",
        "building_ix": BUILDING_IX,
        "building_id": str(building_id),
        "weather": "TRY_4",
        "solver": os.environ.get("SOLVER", "auto-detected"),
        "n_steps_year": int(n_year),
        "n_steps_short": SHORT_STEPS,
        "A_ref_m2": float(cfg["A_ref"]),
        "comfortT_lb": float(cfg["comfortT_lb"]),
        "comfortT_ub": float(cfg["comfortT_ub"]),
        "WACC": float(cfg["WACC"]),
        "note": (
            "Inputs are pinned alongside the outputs so the parity test does "
            "not depend on the stochastic tsorb occupancy simulation."
        ),
    }
    with open(os.path.join(HERE, "golden_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    print("\nWritten to {}:".format(HERE))
    for name in sorted(os.listdir(HERE)):
        if name.startswith("_") or name.endswith(".py"):
            continue
        size = os.path.getsize(os.path.join(HERE, name)) / 1024.0
        print("  {:28s} {:8.1f} kB".format(name, size))
    print("\nSpec. heat demand: {:.1f} kWh/m2/a".format(
        aggregates["specific_heat_kWh_per_m2"]))
    print("Design heat load : {:.3f} kW".format(aggregates["design_heat_load_kW"]))


if __name__ == "__main__":
    main()
