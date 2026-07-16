# -*- coding: utf-8 -*-
"""
Phase 3 tests: envelope refurbishment and comfort control as discrete
investment decisions inside the MILP (previously untested
refurbishment=True path).
"""

import pytest

import tsib.energysystem as es

from conftest import zone_cfg_with_occupancy

# a few winter days suffice to trigger heating-driven investment
# decisions (investment cost is scaled by the year fraction), while
# keeping the Big-M MILP small enough for fast tests
N_STEPS = 72


@pytest.fixture(scope="module")
def refurb_cfg():
    cfg, _ = zone_cfg_with_occupancy(
        n_steps=N_STEPS, refurbishment=True, force_refurbishment=False
    )
    return cfg


def solve_zone(cfg, **zone_kwargs):
    model = es.EnergySystemModel(cfg["weather"].index, wacc=cfg["WACC"])
    zone = model.add(es.ThermalZone5R1C("zone", cfg, **zone_kwargs))
    model.solve(tee=False)
    return model, zone


@pytest.fixture(scope="module")
def solved_free(refurb_cfg):
    """Free refurbishment decision at the default heat price."""
    return solve_zone(refurb_cfg, refurbishment=True)


@pytest.fixture(scope="module")
def solved_forced(refurb_cfg):
    """Refurbishment enforced: the 'Nothing' options are excluded."""
    return solve_zone(refurb_cfg, refurbishment=True, force_refurbishment=True)


def test_choose_exactly_one_option(solved_free):
    """For every envelope element exactly one refurbishment option has
    to be chosen (Schuetz et al. 2017 - eq. 2)."""
    model, zone = solved_free
    assert zone.has_free_investment()

    block = getattr(model.pyomo_model, "zone")
    for element, inv in zone.envelope_investments.items():
        selections = [inv.selection_value(block, o) for o in inv.option_names]
        # binary integrality
        for value in selections:
            assert min(abs(value), abs(value - 1)) < 1e-4, (element, selections)
        assert sum(selections) == pytest.approx(1.0, abs=1e-4), element

    # results report the chosen options
    decisions = model.results("zone")["refurbishment"]
    assert decisions.loc["Capacity"].max() <= 1.0 + 1e-4


def test_force_refurbishment(solved_forced):
    """force_refurbishment removes the 'Nothing' option, so a real
    measure has to be chosen for every element."""
    model, zone = solved_forced
    block = getattr(model.pyomo_model, "zone")
    for element, inv in zone.envelope_investments.items():
        assert "Nothing" not in inv.option_names, element
        assert inv.selected_option(block) != "Nothing", element


def test_heat_price_drives_insulation(refurb_cfg, solved_free):
    """With a near-zero heat price no envelope measure pays off; the
    default price already triggers insulation investment."""
    model_cheap, zone_cheap = solve_zone(
        refurb_cfg, refurbishment=True, heat_cost=0.001
    )
    block_cheap = getattr(model_cheap.pyomo_model, "zone")

    # at near-zero heat price nothing is insulated
    for element, inv in zone_cheap.envelope_investments.items():
        assert inv.selected_option(block_cheap) == "Nothing", element

    # at the default heat price part of the envelope gets refurbished
    # and the heating demand drops
    model_default, zone_default = solved_free
    block_default = getattr(model_default.pyomo_model, "zone")
    chosen = {
        element: inv.selected_option(block_default)
        for element, inv in zone_default.envelope_investments.items()
    }
    assert any(option != "Nothing" for option in chosen.values()), chosen

    heat_cheap = model_cheap.results("zone")["timeseries"]["Heating Load"].sum()
    heat_default = model_default.results("zone")["timeseries"]["Heating Load"].sum()
    assert heat_default < heat_cheap


def test_design_load_reduced_by_insulation(refurb_cfg, solved_forced):
    """The design heat load reported after the solve reflects the
    chosen refurbishment measures."""
    model_forced, _ = solved_forced
    model_nothing, _ = solve_zone(refurb_cfg, refurbishment=False)
    capacity_forced = model_forced.results("zone")["static"]["Capacity"]
    capacity_nothing = model_nothing.results("zone")["static"]["Capacity"]
    assert capacity_forced < capacity_nothing


def test_control_investments(solved_free):
    """Already installed control features are fixed active with zero
    cost, missing ones are free binary decisions."""
    model, zone = solved_free
    block = getattr(model.pyomo_model, "zone")

    # capControl=True in the reference cfg: fixed active
    inv_smart = zone.control.investments["SmartThermostat"]
    assert inv_smart.is_fixed
    assert inv_smart.selection(block, "install") == 1.0
    assert inv_smart.options["install"]["capex"] == 0.0
    # the other features are free binary decisions
    for feature in ["Occupancy", "NightReduction"]:
        inv = zone.control.investments[feature]
        assert not inv.is_fixed
        value = inv.selection_value(block, "install")
        assert min(abs(value), abs(value - 1)) < 1e-4
