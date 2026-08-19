# -*- coding: utf-8 -*-
"""
The equipment sheet: the layer between what a building *is* and how its
energy system is wired.

The point of these tests is that a building is described by *parameters*
only - which equipment it has and how big it is - and that the same fixed
template turns any such sheet into a solvable system. That is what makes a
few hundred buildings of a synthetic grid tractable, and what a partner
model can consume without knowing anything about solph.
"""

import numpy as np
import pandas as pd
import pytest

import tsib
from tsib.optimization import (
    BuildingSystemParameters,
    build_spec,
    build_system,
    equipment,
    node_results,
    presets,
    required_inputs,
    solve,
)

from conftest import example_building, golden_zone_cfg


FULL_SHEET = {
    "equipment": {
        "heat_pump": {"capacity_kw": 8.0},
        "pv": {"kwp": 8.0},
        "battery": {"capacity_kwh": 10.0, "power_kw": 3.0,
                    "roundtrip_efficiency": 0.95},
        "buffer": {"capacity_kwh": 20.0, "standby_loss_kW": 0.02},
    },
    "tariff": {"import": "@elecPrice", "export": 0.08},
    "meta": {"bus_id": "pylovo-42", "freq": "h"},
}


# --- the sheet as data -------------------------------------------------


def test_sheet_survives_a_json_round_trip():
    """The sheet is what gets exchanged, so it has to be plain JSON."""
    params = BuildingSystemParameters.from_dict(FULL_SHEET)
    restored = BuildingSystemParameters.from_json(params.to_json())

    assert restored.to_dict() == params.to_dict()
    assert restored.meta["bus_id"] == "pylovo-42"


def test_unknown_equipment_is_rejected():
    with pytest.raises(ValueError, match="Unknown equipment"):
        BuildingSystemParameters(equipment={"windturbine": {"capacity_kw": 5}})


def test_unknown_equipment_parameter_is_rejected():
    """A typo must not silently size nothing."""
    params = BuildingSystemParameters(equipment={"battery": {"capacity": 10.0}})
    with pytest.raises(ValueError, match="Unknown parameter"):
        build_spec(params)


def test_new_equipment_needs_only_a_builder():
    """
    The adaptability claim: EV charging and whatever follows it must be one
    registered function, not a change to the exchange format or the
    template.
    """
    assert "ev_charger" not in tsib.optimization.EQUIPMENT_BUILDERS

    @equipment("ev_charger")
    def _ev(name, params, spec):
        spec.add_component(name, "battery", bus="elec",
                           capacity=params["capacity_kwh"])

    try:
        spec = build_spec(
            BuildingSystemParameters(
                equipment={"ev_charger": {"capacity_kwh": 60.0}}
            )
        )
        assert "ev_charger" in spec.components
    finally:
        del tsib.optimization.EQUIPMENT_BUILDERS["ev_charger"]


# --- the template ------------------------------------------------------


def test_absent_equipment_is_not_built():
    full = build_spec(BuildingSystemParameters.from_dict(FULL_SHEET))
    bare = build_spec(BuildingSystemParameters())

    assert {"heat_pump", "pv", "battery", "buffer"} <= set(full.components)
    assert not {"heat_pump", "pv", "battery", "buffer"} & set(bare.components)
    # the loads and the zone are the template, not equipment
    assert {"grid", "household_load", "dhw_load", "thermalzone"} <= set(
        bare.components
    )


def test_a_building_without_a_heat_pump_gets_a_non_electric_heat_source():
    """
    Most buildings in a low voltage grid are not electrically heated. They
    still belong in the study - they contribute household load, PV and
    battery - so their model has to be feasible.
    """
    bare = build_spec(BuildingSystemParameters())
    with_hp = build_spec(
        BuildingSystemParameters(equipment={"heat_pump": {"capacity_kw": 8.0}})
    )

    assert bare.components["heat_supply"]["bus"] == "heat"
    assert "heat_supply" not in with_hp.components


def test_both_pv_forms_describe_the_same_generator():
    """
    A pre-computed kW profile is what gets exchanged (no second engine
    recomputes the yield); kWp against a specific yield stays available for
    tsib-internal use. Given a profile, kwp is provenance and must not
    scale it again.
    """
    by_kwp = build_spec(
        BuildingSystemParameters(equipment={"pv": {"kwp": 8.0}})
    ).components["pv"]
    by_profile = build_spec(
        BuildingSystemParameters(
            equipment={"pv": {"profile": "@pv_kw", "kwp": 8.0}}
        )
    ).components["pv"]

    assert by_kwp["capacity"] == 8.0
    assert by_kwp["specific_yield"] == "@pv_yield"
    assert by_profile["capacity"] == 1.0
    assert by_profile["specific_yield"] == "@pv_kw"


def test_pv_without_a_size_or_a_profile_is_an_error():
    with pytest.raises(ValueError, match="needs either"):
        build_spec(BuildingSystemParameters(equipment={"pv": {}}))


def test_roundtrip_efficiency_splits_over_both_directions():
    battery = build_spec(
        BuildingSystemParameters(
            equipment={"battery": {"capacity_kwh": 10.0,
                                   "roundtrip_efficiency": 0.9}}
        )
    ).components["battery"]

    assert battery["charge_efficiency"] == pytest.approx(0.9 ** 0.5)
    assert (
        battery["charge_efficiency"] * battery["discharge_efficiency"]
        == pytest.approx(0.9)
    )


def test_capacity_may_be_an_investment_decision():
    pv = build_spec(
        BuildingSystemParameters(
            equipment={"pv": {"kwp": {"capex_per_unit": 800.0, "lifetime": 25.0,
                                      "max_capacity": 10.0}}}
        )
    ).components["pv"]

    assert pv["capex_per_unit"] == 800.0
    assert "capacity" not in pv


def test_meta_stays_out_of_the_system():
    """
    The Pylovo bus id travels with the building but must never reach the
    model - two identical archetypes at different grid nodes are the same
    building.
    """
    spec = build_spec(BuildingSystemParameters.from_dict(FULL_SHEET))
    assert "pylovo-42" not in spec.to_json()


def test_check_index_catches_a_resolution_mismatch():
    params = BuildingSystemParameters.from_dict(FULL_SHEET)
    hourly = pd.date_range("2010-01-01", periods=24, freq="h")
    quarterly = pd.date_range("2010-01-01", periods=24, freq="15min")

    assert params.check_index(hourly)
    with pytest.raises(ValueError, match="15min"):
        params.check_index(quarterly)


# --- the contract with the building configuration ----------------------


def test_required_inputs_lists_what_the_spec_will_look_up():
    full = BuildingSystemParameters.from_dict(FULL_SHEET)
    bare = BuildingSystemParameters()

    assert required_inputs(build_spec(full)) == [
        "cop", "elecLoad", "elecPrice", "hotWaterLoad", "pv_yield"
    ]
    # no heat pump and no PV: neither renewable profile is needed
    assert bare.required_inputs() == ["elecLoad", "elecPrice", "hotWaterLoad"]


def test_required_inputs_accepts_a_serialized_spec():
    spec = build_spec(BuildingSystemParameters.from_dict(FULL_SHEET))
    assert required_inputs(spec.to_dict()) == required_inputs(spec)


def test_heat_load_only_needs_no_renewable_simulation():
    """
    The gating this enables: a pure heat load run used to pay for a PV
    simulation whose result it never read.
    """
    bdg, _ = example_building()

    def fail():
        raise AssertionError("getRenewables was called")

    bdg.getRenewables = fail
    cfg = bdg._optimization_config(presets.heat_load_only())

    assert "pv_yield" not in cfg
    assert required_inputs(presets.heat_load_only()) == []


# --- end to end --------------------------------------------------------


def _price_profile(index):
    """Cheap by day, expensive in the evening peak."""
    return np.where((index.hour >= 17) & (index.hour < 21), 0.60, 0.10)


def test_a_parameterized_building_solves_and_serves_its_hot_water():
    n = 168
    cfg = golden_zone_cfg(n_steps=n)
    cfg["elecPrice"] = _price_profile(cfg["weather"].index)
    cfg["cop"] = np.full(n, 3.0)
    cfg["pv_yield"] = np.zeros(n)

    params = BuildingSystemParameters(
        equipment={"heat_pump": {"capacity_kw": 20.0},
                   "buffer": {"capacity_kwh": 20.0}},
        tariff={"import": "@elecPrice"},
    )
    es, nodes = build_system(build_spec(params), cfg)
    model, _ = solve(es)
    results = node_results(model, nodes, index=cfg["weather"].index)

    assert results["thermalzone"]["timeseries"]["Heating Load"].sum() > 0
    # the heat pump is the only heat source, so it carries zone and hot water
    assert results["heat_pump"]["out_heat"].sum() > 0
    assert results["dhw_load"]["in_heat"].sum() == pytest.approx(
        np.asarray(cfg["hotWaterLoad"])[:n].sum(), rel=1e-6
    )


def test_a_gas_heated_building_still_solves():
    n = 168
    cfg = golden_zone_cfg(n_steps=n)
    cfg["elecPrice"] = _price_profile(cfg["weather"].index)

    params = BuildingSystemParameters(tariff={"import": "@elecPrice"})
    es, nodes = build_system(build_spec(params), cfg)
    model, _ = solve(es)
    results = node_results(model, nodes, index=cfg["weather"].index)

    # its heat comes from outside the electricity system, so the grid only
    # sees the household load
    assert results["heat_supply"]["out_heat"].sum() > 0
    assert results["grid"]["out_elec"].sum() == pytest.approx(
        np.asarray(cfg["elecLoad"])[:n].sum(), rel=1e-6
    )


def test_building_optimize_accepts_a_parameterized_spec():
    """`Building.optimize` resolves every reference the template carries,
    including the hot water profile no preset used before."""
    bdg, _ = example_building(mean_load=True, seed=42)
    params = BuildingSystemParameters(
        equipment={"heat_pump": {"capacity_kw": 20.0}, "pv": {"kwp": 8.0}},
    )
    results = bdg.optimize(build_spec(params))

    assert {"thermalzone", "grid", "heat_pump", "pv", "dhw_load"} <= set(results)
    assert results["pv"]["out_elec"].sum() > 0
    assert results["dhw_load"]["in_heat"].sum() > 0
