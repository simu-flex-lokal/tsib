# -*- coding: utf-8 -*-
"""
Phase 4 tests: storage/PV/grid components and the core architectural
proof - the 5R1C zone's thermal mass acts as a flexibility resource in
one shared solve with storage and time-varying prices.
"""

import numpy as np
import pandas as pd
import pytest

import tsib.energysystem as es
from tsib.energysystem.components.storage import StorageComponent

from conftest import zone_cfg_with_occupancy


def hourly_index(n, start="2010-01-01"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def test_storage_roundtrip():
    """A battery shifts energy from cheap night hours into an expensive
    evening; charge/discharge are linked by the round-trip efficiency."""
    n = 48
    index = hourly_index(n)
    hour = index.hour
    price = pd.Series(np.where((hour >= 17) & (hour <= 21), 0.50, 0.05), index=index)
    demand = pd.Series(1.0, index=index)

    model = es.EnergySystemModel(index)
    bus = model.add(es.Bus("elec"))
    model.add(es.FixedDemand("demand", bus, demand))
    model.add(es.GridConnection("grid", bus, import_price=price))
    model.add(
        es.ElectricalStorage(
            "battery",
            bus,
            capacity=20.0,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
            charge_power_limit=5.0,
            discharge_power_limit=5.0,
        )
    )
    model.solve(tee=False)

    battery = model.results("battery")
    grid = model.results("grid")

    # the expensive evening is served from the battery, not the grid
    spike = (hour >= 17) & (hour <= 21)
    np.testing.assert_allclose(grid["import"].values[spike], 0.0, atol=1e-6)
    np.testing.assert_allclose(battery["discharge"].values[spike], 1.0, atol=1e-6)

    # energy conservation including round-trip losses (periodic horizon)
    charged = battery["charge"].sum()
    discharged = battery["discharge"].sum()
    assert discharged / charged == pytest.approx(0.9 * 0.9, rel=1e-4)
    assert grid["import"].sum() == pytest.approx(
        demand.sum() + charged - discharged, rel=1e-4
    )

    # SOC respects the capacity
    assert battery["soc"].max() <= 20.0 + 1e-6
    assert battery["soc"].min() >= -1e-6


def test_thermal_storage_standby_loss():
    """The thermal storage loses its standby heat via the
    _external_energy_delta hook, which the grid has to resupply."""
    n = 24
    index = hourly_index(n)
    demand = pd.Series(0.5, index=index)

    model = es.EnergySystemModel(index)
    bus = model.add(es.Bus("heat"))
    model.add(es.FixedDemand("demand", bus, demand))
    model.add(es.GridConnection("grid", bus, import_price=0.10))
    model.add(
        es.ThermalStorage(
            "tank",
            bus,
            capacity=50.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            standby_loss_kW=0.2,
        )
    )
    model.solve(tee=False)

    grid_import = model.results("grid")["import"].sum()
    # demand plus the constant standby loss of the tank
    assert grid_import == pytest.approx(demand.sum() + 0.2 * n, rel=1e-4)


def test_storage_subclasses_share_base():
    """Electrical and thermal storage are thin specializations of the
    generic StorageComponent - the SOC/dispatch logic lives once in the
    base class (EV-readiness of the interface)."""
    assert issubclass(es.ElectricalStorage, StorageComponent)
    assert issubclass(es.ThermalStorage, StorageComponent)

    electrical_overrides = {
        k
        for k in vars(es.ElectricalStorage)
        if not k.startswith("__") and not k.startswith("_abc")
    }
    assert electrical_overrides == {"carrier"}

    thermal_overrides = {
        k
        for k in vars(es.ThermalStorage)
        if not k.startswith("__") and not k.startswith("_abc")
    }
    # only the standby-loss hook and its parameter handling are added
    assert thermal_overrides <= {
        "carrier",
        "build_parameters",
        "_external_energy_delta",
    }

    # time-series power limits are part of the base interface (future
    # EVBattery availability profiles)
    limits = pd.Series(np.tile([0.0, 11.0], 12), index=hourly_index(24))
    storage = es.ElectricalStorage(
        "ev_like", es.Bus("elec"), capacity=40.0, charge_power_limit=limits
    )
    assert storage.charge_power_limit is limits


def test_pv_self_consumption():
    """PV feeds the demand first as long as the export price is below
    the import price; the bus balance closes every time step."""
    n = 48
    index = hourly_index(n)
    hour = index.hour
    yield_profile = pd.Series(
        np.where((hour >= 8) & (hour <= 16), 0.8, 0.0), index=index
    )
    demand = pd.Series(1.0, index=index)

    model = es.EnergySystemModel(index)
    bus = model.add(es.Bus("elec"))
    model.add(es.FixedDemand("demand", bus, demand))
    model.add(
        es.GridConnection("grid", bus, import_price=0.30, export_price=0.08)
    )
    model.add(es.PVGenerator("pv", bus, yield_profile, capacity=2.0))
    model.solve(tee=False)

    pv = model.results("pv")
    grid = model.results("grid")

    generation = pv["generation"].values
    balance = grid["import"].values - grid["export"].values + generation - demand.values
    np.testing.assert_allclose(balance, 0.0, atol=1e-6)

    # at 1.6 kW PV potential and 1 kW load: no import, 0.6 kW export
    daylight = (hour >= 8) & (hour <= 16)
    np.testing.assert_allclose(grid["import"].values[daylight], 0.0, atol=1e-6)
    np.testing.assert_allclose(grid["export"].values[daylight], 0.6, atol=1e-6)
    assert pv["capacity"] == pytest.approx(2.0)


def test_pv_investment_decision():
    """A cheap PV investment is built up to the demand-driven optimum,
    an absurdly expensive one is not built at all."""
    n = 168
    index = hourly_index(n)
    hour = index.hour
    yield_profile = pd.Series(
        np.where((hour >= 8) & (hour <= 16), 0.5, 0.0), index=index
    )
    demand = pd.Series(1.0, index=index)

    def run(capex):
        model = es.EnergySystemModel(index, wacc=0.05)
        bus = model.add(es.Bus("elec"))
        model.add(es.FixedDemand("demand", bus, demand))
        model.add(es.GridConnection("grid", bus, import_price=0.30))
        model.add(
            es.PVGenerator(
                "pv",
                bus,
                yield_profile,
                capacity=es.ContinuousInvestment(
                    capex_per_unit=capex, lifetime=25, max_capacity=100.0
                ),
            )
        )
        model.solve(tee=False)
        return model.results("pv")["capacity"]

    assert run(capex=100.0) > 1.0
    assert run(capex=1e6) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------
# core architectural proof (more important than the envelope-investment
# path): the comfort band and thermal mass of the 5R1C zone are a
# dispatchable flexibility resource in ONE shared solve with storage and
# a time-varying price
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def price_scenario():
    cfg, _ = zone_cfg_with_occupancy(n_steps=168)
    index = cfg["weather"].index
    hour = pd.DatetimeIndex(index).hour
    # pronounced daily price structure: cheap night, expensive evening
    price = pd.Series(
        np.where((hour >= 17) & (hour <= 21), 0.30, np.where(hour < 6, 0.02, 0.10)),
        index=index,
    )
    return cfg, price, hour


def build_flex_model(cfg, price, comfort_ub=None, with_storage=False):
    cfg = dict(cfg)
    if comfort_ub is not None:
        cfg["comfortT_ub"] = comfort_ub
    model = es.EnergySystemModel(cfg["weather"].index, wacc=cfg["WACC"])
    heat_bus = model.add(es.Bus("heat"))
    zone = model.add(
        es.ThermalZone5R1C("zone", cfg, heat_bus=heat_bus, refurbishment=False)
    )
    model.add(es.GridConnection("heat_grid", heat_bus, import_price=price))
    if with_storage:
        model.add(
            es.ThermalStorage(
                "tank",
                heat_bus,
                capacity=30.0,
                charge_efficiency=0.98,
                discharge_efficiency=0.98,
                standby_loss_kW=0.05,
            )
        )
    model.solve(tee=False)
    return model, zone


def test_thermal_mass_flexibility(price_scenario):
    """The zone pre-heats within its comfort band before price spikes:
    the same MILP dispatches building mass like a storage."""
    cfg, price, hour = price_scenario
    model, _ = build_flex_model(cfg, price)

    ts = model.results("zone")["timeseries"]
    imports = model.results("heat_grid")["import"].values

    spike = (hour >= 17) & (hour <= 21)
    night = hour < 6

    # pre-heating: the air temperature demonstrably rises above the
    # lower comfort bound ahead of expensive hours ...
    assert ts["T_air"].max() > cfg["comfortT_lb"] + 1.0
    # ... while the comfort band is never violated
    assert ts["T_air"].min() >= cfg["comfortT_lb"] - 1e-4
    assert ts["T_air"].max() <= cfg["comfortT_ub"] + 1e-4

    # the heat purchase concentrates in cheap night hours and avoids
    # the expensive evening: night hours cover 25% of the day but
    # noticeably more of the energy, spike hours (21% of the day) far
    # less than their share
    assert imports[night].sum() / imports.sum() > 0.35
    assert imports[spike].sum() / imports.sum() < 0.15

    # the mass temperature swings: thermal storage behavior
    assert ts["T_m"].max() - ts["T_m"].min() > 0.5


def test_flexibility_reduces_cost(price_scenario):
    """Collapsing the comfort band (removing the flexibility) has to
    make the identical system strictly more expensive - the monetary
    proof that thermal mass flexibility is exploited."""
    cfg, price, _ = price_scenario
    model_flex, _ = build_flex_model(cfg, price)
    model_rigid, _ = build_flex_model(
        cfg, price, comfort_ub=cfg["comfortT_lb"] + 1e-6
    )
    assert model_flex.objective_value < 0.95 * model_rigid.objective_value


def test_zone_and_storage_one_solve(price_scenario):
    """Thermal zone + thermal storage + time-varying price in ONE
    solve: both flexibilities are dispatched together and the bus
    balance closes."""
    cfg, price, hour = price_scenario
    model, zone = build_flex_model(cfg, price, with_storage=True)

    ts = model.results("zone")["timeseries"]
    tank = model.results("tank")
    imports = model.results("heat_grid")["import"].values

    # bus balance: import + tank discharge - tank charge == zone heating
    balance = (
        imports
        + tank["discharge"].values
        - tank["charge"].values
        - ts["Heating Load"].values
    )
    np.testing.assert_allclose(balance, 0.0, atol=1e-5)

    # both flexibilities are used
    assert tank["charge"].sum() > 1.0
    assert ts["T_m"].max() - ts["T_m"].min() > 0.1

    # no import during the price spike: tank + pre-heated mass carry
    # the zone through the expensive evening
    spike = (hour >= 17) & (hour <= 21)
    assert imports[spike].sum() / imports.sum() < 0.05
