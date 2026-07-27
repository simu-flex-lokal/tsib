# -*- coding: utf-8 -*-
"""
Adding a component to the tsib building block kit.

This is the worked example from the tutorial in `docs/energysystem.md` §4:
a gas-fired CHP unit. It is deliberately unremarkable, and that is the
point - since the migration to oemof-solph, adding a technology to the kit
is *one factory function* mapping spec parameters onto stock solph objects.
No constraints, no variables, no solver knowledge.

Contrast with the 5R1C thermal zone (`tsib/optimization/zone5r1c.py`), the
only component that needs a real custom solph block, because its physics is
not expressible as flows between buses.

Run:
    SOLVER=highs uv run python examples/energysystem/chp_component.py
"""

import numpy as np
import pandas as pd
from oemof import solph

from tsib.optimization import SystemSpec, build_system, node_results, objective_value, solve
from tsib.optimization.registry import _bus, _capacity, factory


# --- the whole addition -------------------------------------------------


@factory("chp")
def _chp(name, params, buses, cfg, n_steps, step_size_h):
    """
    Gas-fired combined heat and power unit.

    Spec parameters
    ---------------
    bus_fuel, bus_elec, bus_heat: str, required
        Bus names.
    capacity: float, optional
        Fuel input capacity [kW]. Alternatively capex_per_unit/lifetime/
        max_capacity for an investment decision.
    electrical_efficiency, thermal_efficiency: float, optional
    """
    fuel = _bus(buses, params, "bus_fuel", name)
    elec = _bus(buses, params, "bus_elec", name)
    heat = _bus(buses, params, "bus_heat", name)
    wacc = params.pop("wacc", 0.0)
    capacity = _capacity(params, n_steps * step_size_h, wacc)

    return [
        solph.components.Converter(
            label=name,
            inputs={fuel: solph.Flow(nominal_capacity=capacity)},
            outputs={elec: solph.Flow(), heat: solph.Flow()},
            conversion_factors={
                elec: params.pop("electrical_efficiency", 0.35),
                heat: params.pop("thermal_efficiency", 0.50),
            },
        )
    ]


# --- using it -----------------------------------------------------------


def main():
    n = 48
    index = pd.date_range("2010-01-01", periods=n, freq="h")

    # expensive grid electricity in the evening: the CHP should run then
    elec_price = np.where((index.hour >= 17) & (index.hour < 21), 0.45, 0.12)

    cfg = {
        "weather": pd.DataFrame({"T": np.zeros(n)}, index=index),
        "elecPrice": elec_price,
        "heatDemand": np.full(n, 4.0),
        "elecDemand": np.full(n, 1.5),
    }

    spec = SystemSpec()
    spec.add_bus("gas", carrier="gas")
    spec.add_bus("elec", carrier="electricity")
    spec.add_bus("heat", carrier="heat")

    spec.add_component("gas_grid", "source", bus="gas", price=0.08)
    spec.add_component("grid", "grid", bus="elec", import_price="@elecPrice")
    spec.add_component("boiler_backup", "source", bus="heat", price=0.20)
    spec.add_component("heat_demand", "demand", bus="heat", profile="@heatDemand")
    spec.add_component("elec_demand", "demand", bus="elec", profile="@elecDemand")

    # the new component, used exactly like any built-in one
    spec.add_component(
        "chp", "chp",
        bus_fuel="gas", bus_elec="elec", bus_heat="heat",
        capacity=12.0, electrical_efficiency=0.35, thermal_efficiency=0.50,
    )

    es, nodes = build_system(spec, cfg)
    model, _ = solve(es)
    results = node_results(model, nodes, index=index)

    fuel = results["chp"]["in_gas"]
    power = results["chp"]["out_elec"]
    heat = results["chp"]["out_heat"]
    imported = results["grid"]["out_elec"]
    backup = results["boiler_backup"]["out_heat"]

    print("objective            : {:8.2f} EUR".format(objective_value(model)))
    print("gas input            : {:8.2f} kWh".format(fuel.sum()))
    print("electricity generated: {:8.2f} kWh".format(power.sum()))
    print("heat generated       : {:8.2f} kWh".format(heat.sum()))
    print("backup boiler heat   : {:8.2f} kWh".format(backup.sum()))
    print("grid import          : {:8.2f} kWh".format(imported.sum()))
    print()
    # the conversion factors hold by construction - solph derives both
    # outputs from the single fuel input
    print("electricity == fuel * 0.35 : {:.6f} == {:.6f}".format(
        power.sum(), fuel.sum() * 0.35))
    print("heat        == fuel * 0.50 : {:.6f} == {:.6f}".format(
        heat.sum(), fuel.sum() * 0.50))
    print("heat covered by the CHP    : {:.1%}".format(
        heat.sum() / (heat.sum() + backup.sum())))


if __name__ == "__main__":
    main()
