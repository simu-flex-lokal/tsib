# -*- coding: utf-8 -*-
"""
What one building hands to an energy system model.

tsib describes a building and simulates its time series; the model that
dispatches or sizes the equipment lives elsewhere. This script shows the
whole handover for one TABULA archetype:

    equipment sheet  ->  spec (a dict)  ->  inputs (arrays)  ->  JSON on disk

Nothing here solves anything, and tsib has no solver. Feed the two files
this writes to `esmkit.build_system(spec, inputs, index)` - or to any other
model that reads them - to get a dispatch out.

Run:
    uv run python examples/buildingsystem/system_export_demo.py
"""

import json
import os

import numpy as np

import tsib
from tsib.system import BuildingSystemParameters, required_inputs


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def building():
    """One EPISCOPE single family house in TRY zone 4."""
    try_data, loc = tsib.readTRY(try_num=4)
    return tsib.Building(configurator=tsib.BuildingConfiguration({
        "ID": "DE.N.SFH.06.Gen.ReEx.001.001",
        "weatherData": try_data,
        "weatherID": "TRY_4",
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "n_persons": 2,
        "n_apartments": 1,
        "comfortT_lb": 20.0,
        "comfortT_ub": 26.0,
        "seed": 42,
    }))


def main():
    bdg = building()

    # 1. what the building has. Three lines of data, and the only thing that
    #    changes between scenarios - the building itself stays untouched.
    sheet = BuildingSystemParameters(
        equipment={
            "heat_pump": {"capacity_kw": 12.0},
            "pv": {"kwp": 8.0},
            "battery": {"capacity_kwh": 10.0, "power_kw": 3.0},
        },
        tariff={"import": "@elecPrice", "export": 0.08},
        meta={"bus_id": "pylovo-42", "freq": "h"},
    )
    print("sheet:", sheet)
    print("it will need:", ", ".join(sheet.required_inputs()))

    # 2. how it is wired. The template is the same for every tsib building;
    #    the thermal zone's ten scalars are computed from the configuration
    #    and written into the spec, its five series stay references.
    spec = bdg.system_spec(sheet)
    zone = spec["components"]["thermalzone"]
    print("\nspec version {}, {} buses, {} components".format(
        spec["version"], len(spec["buses"]), len(spec["components"])))
    print("the zone carries C_m = {:.2f} kWh/K and H_Walls = {:.4f} kW/K"
          .format(zone["C_m"], zone["H"]["Walls"]))
    print("and refers to", ", ".join(
        v for v in zone.values() if isinstance(v, str) and v.startswith("@")))

    # 3. the time series it refers to. Occupancy and the renewable
    #    potentials are simulated here if they are missing - this is the
    #    expensive step, and the reason a building is worth caching.
    index = bdg.cfg["weather"].index
    price = np.where((index.hour >= 17) & (index.hour < 21), 0.60, 0.20)
    inputs, index = bdg.system_inputs(spec, elecPrice=price)

    print("\ninputs for {} steps:".format(len(index)))
    for key in sorted(inputs):
        values = np.atleast_1d(inputs[key])
        print("  {:<12} {:>10.2f} .. {:<10.2f} mean {:.2f}".format(
            key, values.min(), values.max(), values.mean()))
    assert sorted(inputs) == required_inputs(spec)

    # 4. on disk. The spec is small enough to read; the series are not.
    spec_path = os.path.join(OUT_DIR, "system_spec.json")
    with open(spec_path, "w") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)

    series_path = os.path.join(OUT_DIR, "system_inputs.csv")
    frame = {key: np.broadcast_to(np.atleast_1d(values), (len(index),))
             for key, values in inputs.items()}
    import pandas as pd

    pd.DataFrame(frame, index=index).to_csv(series_path, float_format="%.6g")

    print("\nwrote {} ({:.1f} kB) and {} ({:.0f} kB)".format(
        os.path.basename(spec_path), os.path.getsize(spec_path) / 1e3,
        os.path.basename(series_path), os.path.getsize(series_path) / 1e3))
    print("the building is now describable without tsib.")


if __name__ == "__main__":
    main()
