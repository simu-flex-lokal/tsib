# -*- coding: utf-8 -*-
"""
The 5R1C thermal zone as an oemof-solph component.

Two independent anchors:

1. the IWU/TABULA reference heat demand - an *external* validation which
   does not depend on any tsib implementation;
2. the golden fixtures in test/data/golden/, captured from the
   pre-migration `tsib.energysystem` stack, which pin the migration itself.

The golden fixtures store the stochastic occupancy inputs alongside the
results, so these tests do not depend on the tsorb occupancy model.
"""

import json
import os

import numpy as np
import pandas as pd
import pyomo.environ as po
import pytest

from tsib.optimization import (
    build_system,
    node_results,
    objective_value,
    presets,
    solve,
    zone_results,
)

from conftest import BUILDING_SET, golden, golden_zone_cfg, zone_cfg_with_occupancy


def _solve_zone(cfg):
    es, nodes = build_system(presets.heat_load_only(), cfg)
    model, _ = solve(es)
    return model, nodes, zone_results(model, nodes["thermalzone"])


# --- external anchor -------------------------------------------------


def test_heatload_reference():
    """Full-year heat demand of the reference building against the IWU
    value, with the same +-20 kWh/m^2/a tolerance as the pre-migration test."""
    cfg, ID = zone_cfg_with_occupancy()

    # manipulate internal gains with tabula mean value
    cfg["Q_ig"] = cfg["Q_ig"] * 15.552 / (cfg["Q_ig"].sum() / cfg["A_ref"])

    _, _, results = _solve_zone(cfg)
    q_sim = results["timeseries"]["Heating Load"].sum() / cfg["A_ref"]
    q_iwu = BUILDING_SET.loc[ID, "q_h_nd"]

    print("Spec. heat demand IWU [kWh/m²/a]: " + str(round(q_iwu)))
    print("Spec. heat demand 5R1C [kWh/m²/a]: " + str(round(q_sim)))

    assert abs(q_sim - q_iwu) <= 20, (
        "The difference between simulation and the values listed by "
        "the IWU is too high."
    )
    assert results["static"]["Capacity"] > 0
    assert results["timeseries"]["T_air"].min() >= cfg["comfortT_lb"] - 1e-4


# --- migration parity ------------------------------------------------


def test_parity_short_horizon():
    """Every state and flow of the 168 h horizon reproduces the
    pre-migration model, not just the aggregates."""
    cfg = golden_zone_cfg(n_steps=168)
    _, _, results = _solve_zone(cfg)

    expected = pd.read_csv(golden("zone_168h.csv"), index_col=0)
    actual = results["timeseries"]
    expected.index = actual.index

    for column in ["Heating Load", "Cooling Load", "T_air", "T_s", "T_m", "T_e"]:
        deviation = np.abs(actual[column].values - expected[column].values).max()
        assert deviation < 1e-5, "{} deviates by {:.2e}".format(column, deviation)


def test_parity_full_year_aggregates():
    """Annual and monthly heat demand reproduce the pre-migration model.

    Per-time-step equality is deliberately not asserted here: with a smart
    thermostat installed the comfort band is open, so at a constant heat
    price the T_m trajectory is degenerate and a different solver path may
    pick another optimum of equal cost. The aggregates are what the
    building results are used for.
    """
    cfg = golden_zone_cfg()
    _, _, results = _solve_zone(cfg)
    expected = json.load(open(golden("zone_year_aggregates.json")))

    heat = results["timeseries"]["Heating Load"]
    assert heat.sum() == pytest.approx(expected["annual_heat_kWh"], rel=1e-3)
    assert results["timeseries"]["Cooling Load"].sum() == pytest.approx(
        expected["annual_cool_kWh"], rel=1e-3
    )
    assert results["static"]["Capacity"] == pytest.approx(
        expected["design_heat_load_kW"], rel=1e-9
    )

    monthly = heat.groupby(heat.index.month).sum()
    for month, value in expected["monthly_heat_kWh"].items():
        assert monthly[int(month)] == pytest.approx(value, rel=1e-2)


def test_parity_full_year_series():
    """The full-year heating load series itself, at the 6 significant
    digits the golden fixture stores."""
    cfg = golden_zone_cfg()
    _, _, results = _solve_zone(cfg)

    expected = pd.read_csv(golden("zone_year.csv.gz"), index_col=0)["Heating Load"]
    actual = results["timeseries"]["Heating Load"]
    assert len(actual) == len(expected) == 8760

    deviation = np.abs(actual.values - expected.values)
    assert deviation.max() < 1e-3, "max deviation {:.2e} kW".format(deviation.max())


# --- structure -------------------------------------------------------


def test_zone_is_an_lp():
    """The zone creates no binary variables: the heat load simulation
    stays a pure LP."""
    cfg = golden_zone_cfg(n_steps=168)
    es, nodes = build_system(presets.heat_load_only(), cfg)
    model, _ = solve(es)

    for var in model.component_data_objects(po.Var):
        assert not var.is_binary(), "unexpected binary variable {}".format(var.name)


def test_comfort_band_respected():
    """The comfort band bounds the air temperature in both directions."""
    cfg = golden_zone_cfg(n_steps=168)
    _, _, results = _solve_zone(cfg)
    T_air = results["timeseries"]["T_air"]

    assert T_air.max() <= cfg["comfortT_ub"] + 1e-4
    assert T_air.min() >= cfg["comfortT_lb"] - 1e-4


def test_heat_flows_through_the_bus():
    """The zone's heating is the solph edge from the heat bus, so the bus
    balance ties it to whatever supplies it."""
    cfg = golden_zone_cfg(n_steps=48)
    model, nodes, results = _solve_zone(cfg)

    zone = nodes["thermalzone"]
    supply = nodes["heat_supply"]
    heat_bus = zone.heat_bus

    from_supply = np.array(
        [po.value(model.flow[supply, heat_bus, t]) for t in model.TIMESTEPS]
    )
    into_zone = results["timeseries"]["Heating Load"].values
    assert np.abs(from_supply - into_zone).max() < 1e-6


def test_objective_is_energy_cost():
    """With no load violation the objective is exactly the priced energy
    of the two supply sources."""
    cfg = golden_zone_cfg(n_steps=168)
    model, _, results = _solve_zone(cfg)

    heat = results["timeseries"]["Heating Load"].sum()
    cool = results["timeseries"]["Cooling Load"].sum()
    violation = results["max_load_violation"] or 0.0

    expected = (
        heat * presets.DEFAULT_HEAT_COST
        + cool * presets.DEFAULT_COOL_COST
        + violation * 100.0
    )
    assert objective_value(model) == pytest.approx(expected, rel=1e-6)
