# -*- coding: utf-8 -*-
"""
Phase 2 tests: the 5R1C thermal zone as fixed-parameter energy system
component reproduces the IWU/TABULA reference heat demand.
"""

import pandas as pd
import pyomo.environ as pyomo

import tsib.energysystem as es

from conftest import BUILDING_SET, zone_cfg_with_occupancy


def test_heatload_reference():
    """Full-year heat demand of the reference building against the IWU
    value, with the same +-20 kWh/m^2/a tolerance as the previous
    Building5R1C test."""
    cfg, ID = zone_cfg_with_occupancy()

    # manipulate internal gains with tabula mean value
    cfg["Q_ig"] = cfg["Q_ig"] * 15.552 / (cfg["Q_ig"].sum() / cfg["A_ref"])

    model = es.EnergySystemModel(cfg["weather"].index, wacc=cfg["WACC"])
    model.add(es.ThermalZone5R1C("zone", cfg, refurbishment=False))
    model.solve(tee=False)

    results = model.results("zone")
    q_sim = results["timeseries"]["Heating Load"].sum() / cfg["A_ref"]
    q_iwu = BUILDING_SET.loc[ID, "q_h_nd"]

    print("Spec. heat demand IWU [kWh/m²/a]: " + str(round(q_iwu)))
    print("Spec. heat demand 5R1C [kWh/m²/a]: " + str(round(q_sim)))

    assert abs(q_sim - q_iwu) <= 20, (
        "The difference between simulation and the values listed by "
        "the IWU is too high."
    )

    # design heat load is reported as static capacity
    assert results["static"]["Capacity"] > 0
    # comfort band is respected
    assert results["timeseries"]["T_air"].min() >= cfg["comfortT_lb"] - 1e-4


def test_zone_without_investment_is_lp():
    """Without attached investment the zone creates no binary variables
    and no Big-M flow variables at all."""
    cfg, _ = zone_cfg_with_occupancy(n_steps=168)

    model = es.EnergySystemModel(cfg["weather"].index, wacc=cfg["WACC"])
    zone = model.add(es.ThermalZone5R1C("zone", cfg, refurbishment=False))
    M = model.build()

    assert not zone.has_free_investment()
    for var in M.component_data_objects(pyomo.Var):
        assert not var.is_binary(), "unexpected binary variable {}".format(var.name)
    # the envelope heat flows collapse into the balances: only the
    # temperature/heating/cooling/violation variables remain
    n_vars = sum(1 for _ in M.component_data_objects(pyomo.Var))
    assert n_vars == 5 * 168 + 1


def test_zone_comfort_band_bounds():
    """The smart thermostat setting spans the comfort band: without it
    the zone is pinned to the lower comfort temperature."""
    cfg, _ = zone_cfg_with_occupancy(n_steps=168)

    model = es.EnergySystemModel(cfg["weather"].index, wacc=cfg["WACC"])
    model.add(es.ThermalZone5R1C("zone", cfg, refurbishment=False))
    model.solve(tee=False)
    ts = model.results("zone")["timeseries"]

    # winter week: heating keeps the zone exactly at the lower comfort bound
    assert ts["T_air"].max() <= cfg["comfortT_ub"] + 1e-4
    assert ts["T_air"].min() >= cfg["comfortT_lb"] - 1e-4
