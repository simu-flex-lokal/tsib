# -*- coding: utf-8 -*-
"""
Describing a building by its equipment instead of wiring it.

This is the worked example for `docs/parameters.md` §9. A
`BuildingSystemParameters` sheet says *what a building has*; `build_spec()`
turns it into a `SystemSpec` through the one template every tsib building
shares. Nothing here authors a topology - that is the point, because a
synthetic low-voltage grid has a few hundred buildings and hand-wiring each
of them does not scale.

The sheet is also the exchange format when tsib is coupled to another
energy system model: topology is deliberately *not* exchanged, only the
equipment sheet plus the input time series. Two models otherwise have to
agree on ports, buses and connection semantics they represent differently.

Parts 1-4 are pure data and run instantly. Part 5 simulates and solves a
real building three times - once per sheet - and needs a solver.

Run:
    SOLVER=highs uv run python examples/energysystem/parameterization_demo.py

Plots of the results are a separate example: `plotting_demo.py`.
"""

import json

import numpy as np
import pandas as pd

import tsib
from tsib.optimization import (
    BuildingSystemParameters,
    build_spec,
    presets,
    required_inputs,
)


#: a heat pump building with PV, a battery and a heat buffer
HP_PV_BATTERY = BuildingSystemParameters(
    equipment={
        "heat_pump": {"capacity_kw": 25.0},
        "pv": {"kwp": 8.0},
        "battery": {"capacity_kwh": 10.0, "power_kw": 3.0,
                    "roundtrip_efficiency": 0.95},
        "buffer": {"capacity_kwh": 20.0, "standby_loss_kW": 0.02},
    },
    tariff={"import": "@elecPrice", "export": 0.08},
    meta={"bus_id": "pylovo-42", "freq": "h"},
)

#: the same building with a gas boiler and PV, no electric heat
GAS_WITH_PV = BuildingSystemParameters(
    equipment={"pv": {"kwp": 8.0}},
    tariff={"import": "@elecPrice", "export": 0.08},
    meta={"bus_id": "pylovo-43", "freq": "h"},
)

#: nothing at all - the status quo of most buildings in a grid
GAS_ONLY = BuildingSystemParameters(
    tariff={"import": "@elecPrice"},
    meta={"bus_id": "pylovo-44", "freq": "h"},
)

SHEETS = [("heat pump + PV + battery", HP_PV_BATTERY),
          ("gas boiler + PV", GAS_WITH_PV),
          ("gas boiler only", GAS_ONLY)]


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --- 1. the sheet -------------------------------------------------------


def show_sheet():
    rule("1. The equipment sheet")
    print(HP_PV_BATTERY.to_json(indent=2))
    print(
        "\nPlain JSON, so it round-trips and can be handed to another model."
    )


# --- 2. the spec it builds ----------------------------------------------


def show_spec():
    rule("2. The spec the template builds from it")
    spec = build_spec(HP_PV_BATTERY)
    print(json.dumps(spec.to_dict()["components"], indent=2, sort_keys=True))

    # meta is descriptive: two identical archetypes at different grid nodes
    # are the same building and must share a cache entry
    print("\ngrid connection point reached the model:",
          HP_PV_BATTERY.meta["bus_id"] in spec.to_json(), "(must be False)")


# --- 3. what the spec demands from the configuration --------------------


def show_required_inputs():
    rule("3. What each spec demands from the building configuration")
    for label, params in SHEETS:
        print("  {:26s} {}".format(label, params.required_inputs()))
    print("  {:26s} {}".format("heat_load_only (preset)",
                               required_inputs(presets.heat_load_only())))
    print(
        "\nThis is the contract with BuildingConfiguration, and it is what\n"
        "lets Building.optimize simulate only the profiles a spec reads - a\n"
        "gas heated building needs neither a COP nor a PV yield."
    )


# --- 4. equipment decides which components exist ------------------------


def show_components_per_sheet():
    rule("4. Same template, different equipment")
    for label, params in SHEETS:
        built = sorted(build_spec(params).components)
        print("  {:26s} {}".format(label, ", ".join(built)))
    print(
        "\nA building with no heat supplying equipment gets a priced,\n"
        "non-electric 'heat_supply' instead of an infeasible model: a gas\n"
        "heated house still belongs in a grid study, it just contributes\n"
        "household load and PV rather than heat pump electricity."
    )


# --- 5. what that means for the grid ------------------------------------


def dynamic_tariff(index):
    """A crude day-ahead-like tariff: cheap at night, peak in the evening."""
    price = np.full(len(index), 0.28)
    price[(index.hour >= 0) & (index.hour < 6)] = 0.18
    price[(index.hour >= 17) & (index.hour < 21)] = 0.55
    return price


def solve_all():
    rule("5. The same building under three sheets")

    bdg = tsib.Building(
        tsib.BuildingConfiguration({"ID": "DE.N.SFH.06.Gen.ReEx.001.001",
                                    "n_persons": 2, "seed": 42})
    )
    index = bdg.cfg["weather"].index
    price = dynamic_tariff(index)
    bdg.cfg["elecPrice"] = price
    expensive = price > 0.5

    print("building : {} m2, design heat load {:.1f} kW".format(
        bdg.cfg["A_ref"], bdg.zone_config.calcDesignHeatLoad()))
    print("tariff   : 0.18 / 0.28 / 0.55 EUR/kWh, evening peak 17-21 h")
    print("\nthe first run simulates the occupancy profile and takes minutes\n")

    header = "{:26s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
        "sheet", "import", "export", "peak", "in peak h")
    print(header)
    print("-" * len(header))

    solved = {}
    for label, params in SHEETS:
        params.check_index(index)
        results = bdg.optimize(build_spec(params))
        solved[label] = results

        imported = results["grid"]["out_elec"]
        exported = results["grid"].get(
            "in_elec", pd.Series(0.0, index=imported.index))

        print("{:26s} {:9.0f}  {:9.0f}  {:8.1f}  {:11.1%}".format(
            label,
            imported.sum(),
            exported.sum(),
            imported.max(),
            imported.values[expensive].sum() / imported.sum(),
        ))

    print("\nunits: kWh over the year, peak in kW, last column the share of\n"
          "the annual import drawn in the 17% of hours that are expensive.\n"
          "The net exchange (import - export) per building is what a grid\n"
          "powerflow computation consumes, keyed by meta['bus_id'].")

    return bdg, solved, price


def main():
    show_sheet()
    show_spec()
    show_required_inputs()
    show_components_per_sheet()
    solve_all()


if __name__ == "__main__":
    main()
