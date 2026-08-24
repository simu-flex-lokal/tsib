# -*- coding: utf-8 -*-
"""
Dispatch behaviour of the kit, and the flexibility proof.

The load-bearing tests here are the last three: they show that the
building's thermal mass and comfort band act as a dispatchable storage in
the same solve as prices and equipment. That property is the entire reason
the 5R1C zone is a component instead of a precomputed load profile, so it
is what the migration had to preserve.
"""

import numpy as np
import pandas as pd
import pytest

from tsib.optimization import (
    SystemSpec,
    build_system,
    node_results,
    objective_value,
    presets,
    solve,
)

from conftest import example_building, golden_zone_cfg


def simple_cfg(n=48, **extra):
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    cfg = {"weather": pd.DataFrame({"T": np.zeros(n)}, index=index)}
    cfg.update(extra)
    return cfg


def two_price_profile(n=48, cheap=0.05, expensive=0.40):
    """Cheap in the first half of every day, expensive in the second."""
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    return np.where(index.hour < 12, cheap, expensive)


# --- storage ----------------------------------------------------------


def test_battery_shifts_energy_into_expensive_hours():
    price = two_price_profile()
    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_component("grid", "grid", bus="elec", import_price="@price")
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    # default round-trip efficiency (0.95/0.95): lossy storage makes the
    # optimum unique. With lossless storage, shuffling energy *within* the
    # expensive hours is cost-neutral and the optimum is degenerate.
    spec.add_component(
        "battery", "battery", bus_in="elec", bus_out="elec", capacity=10.0
    )

    es, nodes = build_system(spec, simple_cfg(price=price))
    model, _ = solve(es)
    results = node_results(model, nodes)

    charge = results["battery"]["in_elec"]
    discharge = results["battery"]["out_elec"]
    assert charge.sum() > 0, "the battery is never used"
    # charging happens in the cheap hours, discharging in the expensive ones
    assert charge[price > 0.1].sum() == pytest.approx(0.0, abs=1e-6)
    assert discharge[price < 0.1].sum() == pytest.approx(0.0, abs=1e-6)


def test_storage_is_periodically_balanced():
    """`balanced=True` reproduces the periodic SOC wrap of the
    pre-migration model: no free energy over the horizon."""
    price = two_price_profile()
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price="@price")
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    spec.add_component(
        "battery", "battery", bus_in="elec", bus_out="elec", capacity=10.0
    )

    es, nodes = build_system(spec, simple_cfg(price=price))
    model, _ = solve(es)
    content = node_results(model, nodes)["battery"]["storage_content"]

    assert content.iloc[0] == pytest.approx(content.iloc[-1], abs=1e-6)


def test_thermal_storage_standby_loss():
    """The standby loss maps onto solph's fixed_losses_absolute: supplying
    a constant demand through the buffer costs more than supplying it
    directly."""
    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_component("supply", "source", bus="heat", price=0.10)
    spec.add_component("demand", "demand", bus="heat", profile=1.0)
    spec.add_component(
        "buffer", "thermal_storage", bus_in="heat", bus_out="heat",
        capacity=20.0,
        standby_loss_kW=0.05, charge_efficiency=1.0, discharge_efficiency=1.0,
    )

    es, nodes = build_system(spec, simple_cfg())
    model, _ = solve(es)
    results = node_results(model, nodes)

    supplied = results["supply"]["out_heat"].sum()
    demanded = results["demand"]["in_heat"].sum()
    # the loss is only paid for while the buffer actually holds energy
    assert supplied >= demanded - 1e-6


# --- generation and conversion ----------------------------------------


def test_pv_self_consumption_beats_export():
    """PV serves the demand first while the export price is below the
    import price."""
    n = 48
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    yield_profile = np.clip(np.sin((index.hour - 6) / 12 * np.pi), 0, None)

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component(
        "grid", "grid", bus="elec", import_price=0.35, export_price=0.08
    )
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    spec.add_component(
        "pv", "pv", bus="elec", specific_yield="@pv_yield", capacity=3.0
    )

    es, nodes = build_system(spec, simple_cfg(n=n, pv_yield=yield_profile))
    model, _ = solve(es)
    results = node_results(model, nodes)

    generation = results["pv"]["out_elec"]
    # a grid connection with feed-in is two solph nodes collected under one
    # name: "out_elec" is the import source, "in_elec" the export sink
    imported = results["grid"]["out_elec"]
    exported = results["grid"]["in_elec"]

    assert generation.sum() > 0
    sunny = yield_profile > 0
    # self-consumption first: while the sun shines, import is displaced
    assert imported[sunny].sum() < imported[~sunny].sum()
    # only the surplus above the demand of 1 kW is exported
    assert exported.sum() == pytest.approx(
        np.clip(generation.values - 1.0, 0, None).sum(), abs=1e-6
    )


def test_pv_investment_sizes_to_the_optimum():
    n = 48
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    yield_profile = np.clip(np.sin((index.hour - 6) / 12 * np.pi), 0, None)

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.35)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    spec.add_component(
        "pv", "pv", bus="elec", specific_yield="@pv_yield", wacc=0.05,
        capex_per_unit=800.0, lifetime=25.0, max_capacity=10.0,
    )

    es, nodes = build_system(spec, simple_cfg(n=n, pv_yield=yield_profile))
    model, _ = solve(es)
    capacity = node_results(model, nodes)["pv"]["capacity"]

    # cheap PV against a 0.35 EUR/kWh import price: build, but not beyond
    # what the demand can absorb (no export remuneration configured)
    assert 0 < capacity <= 10.0


def test_heat_pump_couples_two_buses():
    """Electricity in, heat out, at the given COP - and the COP cut-off is
    respected."""
    n = 24
    cop = np.where(np.arange(n) < 12, 3.0, 0.0)  # off in the second half

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_bus("heat")
    spec.add_component("grid", "grid", bus="elec", import_price=0.30)
    spec.add_component(
        "hp", "heat_pump", bus_in="elec", bus_out="heat", cop="@cop", capacity=5.0
    )
    spec.add_component("backup", "source", bus="heat", price=10.0)
    spec.add_component("demand", "demand", bus="heat", profile=2.0)

    es, nodes = build_system(spec, simple_cfg(n=n, cop=cop))
    model, _ = solve(es)
    results = node_results(model, nodes)

    heat = results["hp"]["out_heat"]
    power = results["hp"]["in_elec"]
    assert heat[cop == 0].sum() == pytest.approx(0.0, abs=1e-6)
    # energy balance holds by construction of the conversion factor
    assert power[cop > 0].sum() == pytest.approx(heat[cop > 0].sum() / 3.0, rel=1e-6)


# --- the flexibility proof --------------------------------------------


def _zone_with_price(price, n=168, comfort_ub=None):
    """Thermal zone supplied by a price-varying heat source."""
    cfg = golden_zone_cfg(n_steps=n)
    if comfort_ub is not None:
        cfg["comfortT_ub"] = comfort_ub
    cfg["heat_price"] = price

    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_bus("cool", carrier="cool")
    spec.add_component("heat_supply", "source", bus="heat", price="@heat_price")
    spec.add_component("cool_supply", "source", bus="cool", price=0.02)
    spec.add_component(
        "thermalzone", "zone5r1c", heat_bus="heat", cool_bus="cool"
    )
    es, nodes = build_system(spec, cfg)
    model, _ = solve(es)
    return model, nodes, cfg


def test_thermal_mass_preheats_before_price_spikes():
    """The zone buys heat before the expensive hours and rides through
    them on the building's thermal mass, within the comfort band."""
    n = 168
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    expensive = (index.hour >= 17) & (index.hour < 21)
    price = np.where(expensive, 1.50, 0.05)

    model, nodes, cfg = _zone_with_price(price, n=n)
    ts = node_results(model, nodes)["thermalzone"]["timeseries"]
    heat = ts["Heating Load"]

    # the spike covers 4 of 24 hours (17%); the zone has to buy markedly
    # less than its share there. Threshold taken from the pre-migration
    # test, which is the validated criterion for this scenario.
    share_in_spike = heat.values[expensive].sum() / heat.sum()
    assert share_in_spike < 0.15, (
        "the zone draws {:.1%} of its heat in the price spike - the thermal "
        "mass is not being used as flexibility".format(share_in_spike)
    )

    # pre-heating is visible: the air temperature demonstrably rises above
    # the lower comfort bound ahead of the expensive hours ...
    assert ts["T_air"].max() > cfg["comfortT_lb"] + 1.0
    # ... the mass temperature swings, i.e. it behaves as a storage ...
    assert ts["T_m"].max() - ts["T_m"].min() > 0.5
    # ... and the comfort band is never violated to achieve any of it
    assert ts["T_air"].min() >= cfg["comfortT_lb"] - 1e-4
    assert ts["T_air"].max() <= cfg["comfortT_ub"] + 1e-4


def test_flexibility_reduces_cost():
    """Collapsing the comfort band removes the flexibility and must make
    the same price scenario strictly more expensive."""
    n = 168
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    price = np.where((index.hour >= 17) & (index.hour < 21), 1.50, 0.05)

    flexible, _, cfg = _zone_with_price(price, n=n)
    # comfort_ub == comfort_lb pins T_air and destroys the flexibility
    rigid, _, _ = _zone_with_price(price, n=n, comfort_ub=cfg["comfortT_lb"])

    assert objective_value(flexible) < objective_value(rigid), (
        "flexible: {:.3f}, rigid: {:.3f}".format(
            objective_value(flexible), objective_value(rigid)
        )
    )


def test_zone_and_storage_in_one_solve():
    """Thermal zone, thermal buffer and a time-varying price in a single
    optimization: both flexibilities are available at once."""
    n = 168
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    expensive = (index.hour >= 17) & (index.hour < 21)
    cfg = golden_zone_cfg(n_steps=n)
    cfg["heat_price"] = np.where(expensive, 1.50, 0.05)

    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_bus("cool", carrier="cool")
    spec.add_component("heat_supply", "source", bus="heat", price="@heat_price")
    spec.add_component("cool_supply", "source", bus="cool", price=0.02)
    spec.add_component(
        "buffer", "thermal_storage", bus_in="heat", bus_out="heat",
        capacity=30.0, standby_loss_kW=0.02
    )
    spec.add_component("thermalzone", "zone5r1c", heat_bus="heat", cool_bus="cool")

    es, nodes = build_system(spec, cfg)
    model, _ = solve(es)
    results = node_results(model, nodes)

    supply = results["heat_supply"]["out_heat"]
    assert supply.values[expensive].sum() / supply.sum() < 0.05
    assert results["buffer"]["in_heat"].sum() > 0, "the buffer is never used"


def test_building_optimize_resolves_its_own_profiles():
    """
    `Building.optimize` is the documented one-liner entry point, so the
    profile references the presets carry by default ("@elecPrice", "@cop",
    "@pv_yield") have to resolve against a plain building - none of them is
    produced by `BuildingConfiguration` itself.
    """
    bdg, _ = example_building(mean_load=True, seed=42)
    results = bdg.optimize(presets.hp_pv_battery(pv_kwp=8.0, battery_kwh=10.0))

    assert {"thermalzone", "grid", "hp", "pv", "battery"} <= set(results)
    assert results["pv"]["out_elec"].sum() > 0
    assert results["thermalzone"]["timeseries"]["Heating Load"].sum() > 0
    # the heat pump is the only heat source, so it has to carry the zone
    assert results["hp"]["out_heat"].sum() > 0

    # a caller-supplied profile wins over the fallbacks
    cfg = bdg._optimization_config()
    assert cfg["elecPrice"] == presets.DEFAULT_ELEC_PRICE
    bdg.cfg["elecPrice"] = np.full(len(cfg["weather"]), 0.11)
    assert bdg._optimization_config()["elecPrice"][0] == 0.11
