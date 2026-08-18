# -*- coding: utf-8 -*-
"""
Ready-made system configurations of the tsib building block kit.

A preset returns a `SystemSpec`, not a built system, so presets are simply
named specs: there is one build path (`build_system`), and a preset can be
serialized, diffed, or modified before building. That is what makes them
usable as the parameterization unit when orchestrating many buildings.

    spec = presets.hp_pv_battery(pv_kwp=8.0)
    es, nodes = build_system(spec, cfg)
"""

from .spec import SystemSpec

#: default energy prices [EUR/kWh] of the pre-migration ThermalZone5R1C
DEFAULT_HEAT_COST = 0.08
DEFAULT_COOL_COST = 0.02

#: fallback electricity tariff [EUR/kWh] used by `Building.optimize` when the
#: building configuration carries no `elecPrice` profile of its own
DEFAULT_ELEC_PRICE = 0.35


def heat_load_only(heat_cost=DEFAULT_HEAT_COST, cool_cost=DEFAULT_COOL_COST,
                   **zone_kwargs):
    """
    Bare heat load simulation: a thermal zone supplied by priced heat and
    cooling sources. This is the production path of `Building.getHeatLoad`
    and the direct replacement of the pre-migration standalone zone.

    Parameters
    ----------
    heat_cost, cool_cost: float, optional
        Energy prices [EUR/kWh]. Their absolute level is irrelevant for the
        resulting load as long as both are positive; they exist so the
        objective has something to minimize.
    zone_kwargs: passed through to ThermalZone5R1C (max_load, initial_T_m).
    """
    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_bus("cool", carrier="cool")
    spec.add_component("heat_supply", "source", bus="heat", price=heat_cost)
    spec.add_component("cool_supply", "source", bus="cool", price=cool_cost)
    spec.add_component(
        "thermalzone", "zone5r1c", heat_bus="heat", cool_bus="cool", **zone_kwargs
    )
    return spec


def hp_pv_battery(
    import_price="@elecPrice",
    export_price=None,
    cop="@cop",
    specific_yield="@pv_yield",
    elec_demand="@elecLoad",
    pv_kwp=None,
    battery_kwh=None,
    hp_kw=None,
    buffer_kwh=None,
    cool_cost=DEFAULT_COOL_COST,
    wacc=0.06,
    **zone_kwargs
):
    """
    Building with a heat pump, PV, a battery and a heat buffer, all on a
    priced grid connection.

    Capacities may be given as a number (fixed) or left None, in which case
    pass a `*_invest` dict instead via `invest` to size them. This is the
    showcase of what the migration buys: the thermal zone's comfort band,
    the buffer and the battery trade off against the price signal in one
    solve, and everything except the zone is a stock solph component.

    Parameters
    ----------
    import_price, export_price, cop, specific_yield, elec_demand: str or
        array-like - profile references resolved against the building cfg.
    pv_kwp, battery_kwh, hp_kw, buffer_kwh: float or dict, optional
        Fixed capacity, or a dict of investment parameters
        ({"capex_per_unit":…, "lifetime":…, "max_capacity":…}).
    """
    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_bus("heat", carrier="heat")
    spec.add_bus("cool", carrier="cool")

    grid = {"bus": "elec", "import_price": import_price}
    if export_price is not None:
        grid["export_price"] = export_price
    spec.add_component("grid", "grid", **grid)

    spec.add_component("demand", "demand", bus="elec", profile=elec_demand)
    spec.add_component("cool_supply", "source", bus="cool", price=cool_cost)

    spec.add_component(
        "hp", "heat_pump", bus_in="elec", bus_out="heat", cop=cop, wacc=wacc,
        **_capacity_params(hp_kw)
    )
    if pv_kwp is not None:
        spec.add_component(
            "pv", "pv", bus="elec", specific_yield=specific_yield, wacc=wacc,
            **_capacity_params(pv_kwp)
        )
    if battery_kwh is not None:
        spec.add_component(
            "battery", "battery", bus="elec", wacc=wacc,
            **_capacity_params(battery_kwh)
        )
    if buffer_kwh is not None:
        spec.add_component(
            "buffer", "thermal_storage", bus="heat", wacc=wacc,
            **_capacity_params(buffer_kwh)
        )

    spec.add_component(
        "thermalzone", "zone5r1c", heat_bus="heat", cool_bus="cool", **zone_kwargs
    )
    return spec


def _capacity_params(value):
    """A number becomes a fixed capacity, a dict an investment decision."""
    if isinstance(value, dict):
        return dict(value)
    return {"capacity": value}
