# -*- coding: utf-8 -*-
"""
The equipment sheet: the layer between what a building *is* and how its
energy system is wired.

The point of these tests is that a building is described by *parameters*
only - which equipment it has and how big it is - and that the same fixed
template turns any such sheet into a complete system description. That is
what makes a few hundred buildings of a synthetic grid tractable, and what
a partner model can consume without knowing anything about tsib.

Nothing is solved here, and nothing can be: the spec is plain data on its
way out of tsib.
"""

import json

import numpy as np
import pandas as pd
import pytest

import tsib
from tsib.system import (
    SPEC_VERSION,
    BuildingSystemParameters,
    build_spec,
    equipment,
    required_inputs,
)

from tsib.envelope import ZONE_SERIES

from conftest import example_building


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
    assert "ev_charger" not in tsib.system.EQUIPMENT_BUILDERS

    @equipment("ev_charger")
    def _ev(name, params, spec):
        spec.add_component(name, "battery", bus_in="elec", bus_out="elec",
                           capacity=params["capacity_kwh"])

    try:
        spec = build_spec(
            BuildingSystemParameters(
                equipment={"ev_charger": {"capacity_kwh": 60.0}}
            )
        )
        assert "ev_charger" in spec["components"]
    finally:
        del tsib.system.EQUIPMENT_BUILDERS["ev_charger"]


# --- the template ------------------------------------------------------


def test_absent_equipment_is_not_built():
    full = build_spec(BuildingSystemParameters.from_dict(FULL_SHEET))
    bare = build_spec(BuildingSystemParameters())

    assert {"heat_pump", "pv", "battery", "buffer"} <= set(full["components"])
    assert not {"heat_pump", "pv", "battery", "buffer"} & set(bare["components"])
    # the loads and the zone are the template, not equipment
    assert {"grid", "household_load", "dhw_load", "thermalzone"} <= set(
        bare["components"]
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

    assert bare["components"]["heat_supply"]["bus"] == "heat"
    assert "heat_supply" not in with_hp["components"]


def test_both_pv_forms_describe_the_same_generator():
    """
    A pre-computed kW profile is what gets exchanged (no second engine
    recomputes the yield); kWp against a specific yield stays available for
    tsib-internal use. Given a profile, kwp is provenance and must not
    scale it again.
    """
    by_kwp = build_spec(
        BuildingSystemParameters(equipment={"pv": {"kwp": 8.0}})
    )["components"]["pv"]
    by_profile = build_spec(
        BuildingSystemParameters(
            equipment={"pv": {"profile": "@pv_kw", "kwp": 8.0}}
        )
    )["components"]["pv"]

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
    )["components"]["battery"]

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
    )["components"]["pv"]

    assert pv["capex_per_unit"] == 800.0
    assert "capacity" not in pv


def test_meta_stays_out_of_the_system():
    """
    The Pylovo bus id travels with the building but must never reach the
    model - two identical archetypes at different grid nodes are the same
    building.
    """
    spec = build_spec(BuildingSystemParameters.from_dict(FULL_SHEET))
    assert "pylovo-42" not in json.dumps(spec)


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

    equipment_inputs = ["cop", "elecLoad", "elecPrice", "hotWaterLoad",
                        "pv_yield"]
    assert required_inputs(build_spec(full)) == sorted(
        equipment_inputs + list(ZONE_SERIES)
    )
    # no heat pump and no PV: neither renewable profile is needed, but the
    # zone is part of every building and always asks for its five series
    assert bare.required_inputs() == sorted(
        ["elecLoad", "elecPrice", "hotWaterLoad"] + list(ZONE_SERIES)
    )


# --- the spec as plain data --------------------------------------------


def test_the_spec_is_json():
    """
    It leaves tsib and is read by something else, so it must survive a file
    on the way. Nothing in it may be a numpy array or an object.
    """
    spec = build_spec(BuildingSystemParameters.from_dict(FULL_SHEET))
    restored = json.loads(json.dumps(spec))

    assert restored == spec
    assert restored["version"] == SPEC_VERSION
    assert set(restored) == {"version", "buses", "components"}
    assert all("type" in c for c in restored["components"].values())


def test_the_zone_carries_its_scalars_and_references_its_series():
    """
    The seam itself: the physics travels as numbers inside the spec, the
    time series travel separately and are named here.
    """
    bdg, _ = example_building()
    zone_params = bdg.zone_parameters()
    spec = build_spec(BuildingSystemParameters(), zone_params)
    zone = spec["components"]["thermalzone"]

    assert zone["type"] == "zone5r1c"
    assert zone["C_m"] == zone_params["C_m"]
    assert zone["H"] == zone_params["H"]
    assert zone["max_load"] == zone_params["max_load"]
    for key in ZONE_SERIES:
        assert zone[key] == "@" + key
        # the values themselves are not in the spec
        assert not isinstance(zone[key], (list, tuple))


def test_the_sheet_overrides_the_derived_zone_parameters():
    bdg, _ = example_building()
    sheet = BuildingSystemParameters(zone={"max_load": 4.0})
    zone = build_spec(sheet, bdg.zone_parameters())["components"]["thermalzone"]

    assert zone["max_load"] == 4.0


def test_a_spec_without_zone_parameters_still_names_its_inputs():
    """
    Asking what a sheet needs must not require a building: the template
    alone answers it, and the incomplete spec says so by carrying no
    physics.
    """
    spec = build_spec(BuildingSystemParameters.from_dict(FULL_SHEET))
    zone = spec["components"]["thermalzone"]

    assert "C_m" not in zone
    assert zone["T_e"] == "@T_e"


# --- the contract with the building ------------------------------------


def test_a_building_supplies_every_input_its_spec_asks_for():
    bdg, _ = example_building(mean_load=True, seed=42)
    sheet = BuildingSystemParameters(
        equipment={"heat_pump": {"capacity_kw": 20.0}, "pv": {"kwp": 8.0}},
    )
    spec = bdg.system_spec(sheet)
    inputs, index = bdg.system_inputs(spec)

    assert sorted(inputs) == required_inputs(spec)
    assert len(index) == len(bdg.cfg["weather"].index)
    for key, values in inputs.items():
        if key == "elecPrice":
            continue
        assert len(values) == len(index), key
    assert inputs["pv_yield"].sum() > 0
    assert inputs["hotWaterLoad"].sum() > 0


def test_a_sheet_without_renewables_pays_for_no_simulation():
    """
    The gating this enables: a building whose system has neither a heat
    pump nor PV must not run the renewable simulation to be described.
    """
    bdg, _ = example_building()

    def fail():
        raise AssertionError("getRenewables was called")

    bdg.getRenewables = fail
    spec = bdg.system_spec(BuildingSystemParameters())
    inputs, _ = bdg.system_inputs(spec)

    assert "pv_yield" not in inputs
    assert "cop" not in inputs


def test_the_tariff_can_be_handed_in():
    bdg, _ = example_building()
    spec = bdg.system_spec(BuildingSystemParameters())
    price = np.linspace(0.1, 0.5, len(bdg.cfg["weather"].index))

    inputs, _ = bdg.system_inputs(spec, elecPrice=price)
    assert inputs["elecPrice"][0] == pytest.approx(0.1)

    inputs, _ = bdg.system_inputs(spec)
    assert inputs["elecPrice"] == tsib.system.DEFAULT_ELEC_PRICE
