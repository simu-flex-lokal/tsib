# -*- coding: utf-8 -*-
"""
Component factories of the tsib building block kit.

Every entry maps a spec `type` onto stock oemof-solph objects. Adding a new
kind of component to the kit means adding one factory here - no changes to
the builder, the presets or the solver.

Each factory returns a *list* of solph nodes, because some kit components
expand into more than one: a grid connection with feed-in is a Source plus a
Sink, since solph prices energy per directed edge.

The only factory which does not return stock solph objects is `zone5r1c` -
the 5R1C thermal zone is the single piece of building physics solph has no
equivalent for.
"""

from oemof import solph

from .investment import annuity_factor

#: registered factories, filled by @factory below
COMPONENT_FACTORIES = {}


def factory(name):
    """Registers a component factory under its spec type name."""

    def register(func):
        if name in COMPONENT_FACTORIES:
            raise ValueError("Duplicate component factory '{}'".format(name))
        COMPONENT_FACTORIES[name] = func
        return func

    return register


def build_component(ctype, name, params, buses, cfg, n_steps):
    """Dispatches to the registered factory of `ctype`."""
    if ctype not in COMPONENT_FACTORIES:
        raise ValueError(
            "Unknown component type '{}' for '{}'. Available: {}".format(
                ctype, name, sorted(COMPONENT_FACTORIES)
            )
        )
    return COMPONENT_FACTORIES[ctype](name, params, buses, cfg, n_steps)


def _bus(buses, params, key, name):
    """Looks up a bus by the name given in the spec."""
    if key not in params:
        raise ValueError("Component '{}' needs a '{}'".format(name, key))
    label = params.pop(key)
    if label not in buses:
        raise ValueError(
            "Component '{}' refers to unknown bus '{}'".format(name, label)
        )
    return buses[label]


def _profile(params, key, cfg, n_steps, name, default=None):
    from .spec import resolve_profile

    value = params.pop(key, default)
    if value is None:
        return None
    return resolve_profile(value, getattr(cfg, "cfg", cfg), n_steps,
                           "{}.{}".format(name, key))


def _investment(params, n_steps, wacc):
    """
    Turns capex/lifetime spec entries into a solph Investment.

    solph expects an already annualized cost per unit (`ep_costs`), so the
    annuity and the horizon scaling stay on the tsib side - the same
    convention the pre-migration framework used, which is what keeps a
    72 hour design study trading capex against opex correctly.
    """
    capex = params.pop("capex_per_unit", None)
    if capex is None:
        return None
    lifetime = params.pop("lifetime")
    opex_fix_share = params.pop("opex_fix_share", 0.0)
    year_fraction = n_steps / 8760.0
    ep_costs = capex * (annuity_factor(lifetime, wacc) + opex_fix_share) * year_fraction
    return solph.Investment(
        ep_costs=ep_costs,
        minimum=params.pop("min_capacity", 0.0),
        maximum=params.pop("max_capacity", None),
    )


def _capacity(params, n_steps, wacc):
    """Either a fixed nominal capacity or a free Investment."""
    investment = _investment(params, n_steps, wacc)
    if investment is not None:
        return investment
    return params.pop("capacity", None)


# ---------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------


@factory("demand")
def _demand(name, params, buses, cfg, n_steps):
    """Inflexible load profile [kW]."""
    bus = _bus(buses, params, "bus", name)
    profile = _profile(params, "profile", cfg, n_steps, name)
    # solph expresses a fixed profile as fix * nominal_capacity
    return [
        solph.components.Sink(
            label=name, inputs={bus: solph.Flow(fix=profile, nominal_capacity=1.0)}
        )
    ]


@factory("source")
def _source(name, params, buses, cfg, n_steps):
    """Unlimited supply at a price [EUR/kWh] - the generic heat/cool supply."""
    bus = _bus(buses, params, "bus", name)
    price = _profile(params, "price", cfg, n_steps, name, default=0.0)
    capacity = params.pop("capacity", None)
    return [
        solph.components.Source(
            label=name,
            outputs={bus: solph.Flow(variable_costs=price, nominal_capacity=capacity)},
        )
    ]


@factory("grid")
def _grid(name, params, buses, cfg, n_steps):
    """
    Grid connection: import at a price and optionally export at a
    remuneration. Two solph nodes, because solph prices directed edges.
    """
    bus = _bus(buses, params, "bus", name)
    import_price = _profile(params, "import_price", cfg, n_steps, name, default=0.0)
    export_price = _profile(params, "export_price", cfg, n_steps, name)
    max_import = params.pop("max_import", None)
    max_export = params.pop("max_export", None)

    nodes = [
        solph.components.Source(
            label=name + "_import",
            outputs={
                bus: solph.Flow(
                    variable_costs=import_price, nominal_capacity=max_import
                )
            },
        )
    ]
    if export_price is not None:
        # negative variable costs = revenue
        nodes.append(
            solph.components.Sink(
                label=name + "_export",
                inputs={
                    bus: solph.Flow(
                        variable_costs=-1.0 * export_price
                        if not hasattr(export_price, "__len__")
                        else [-p for p in export_price],
                        nominal_capacity=max_export,
                    )
                },
            )
        )
    return nodes


@factory("meter")
def _meter(name, params, buses, cfg, n_steps):
    """
    Sub-meter between two buses: a lossless (or efficiency-scaled) transfer
    which prices the metered throughput. This is the primitive of a
    cascading metering concept (e.g. the reduced grid fee branch of
    section 14a EnWG) - see backlog/14a-cascading-metering.md.
    """
    up = _bus(buses, params, "bus_up", name)
    down = _bus(buses, params, "bus_down", name)
    price = _profile(params, "price", cfg, n_steps, name, default=0.0)
    efficiency = params.pop("efficiency", 1.0)
    max_power = params.pop("max_power", None)
    return [
        solph.components.Converter(
            label=name,
            inputs={up: solph.Flow(variable_costs=price)},
            outputs={down: solph.Flow(nominal_capacity=max_power)},
            conversion_factors={down: efficiency},
        )
    ]


@factory("pv")
def _pv(name, params, buses, cfg, n_steps):
    """PV generator driven by a specific yield profile [kW/kWp]."""
    bus = _bus(buses, params, "bus", name)
    yield_profile = _profile(params, "specific_yield", cfg, n_steps, name)
    curtailable = params.pop("curtailable", True)
    wacc = params.pop("wacc", 0.0)
    capacity = _capacity(params, n_steps, wacc)
    # curtailable: the profile is an upper bound; otherwise it is fixed
    limit = {"maximum": yield_profile} if curtailable else {"fix": yield_profile}
    return [
        solph.components.Source(
            label=name,
            outputs={bus: solph.Flow(nominal_capacity=capacity, **limit)},
        )
    ]


@factory("heat_pump")
def _heat_pump(name, params, buses, cfg, n_steps):
    """
    Heat pump coupling an electricity and a heat bus at a time varying COP.

    tsib.simHeatpump returns cop == 0 below the cut-off temperature; solph
    divides by the conversion factor, so those hours are handled by forcing
    the heat output to zero instead.
    """
    bus_in = _bus(buses, params, "bus_in", name)
    bus_out = _bus(buses, params, "bus_out", name)
    cop = _profile(params, "cop", cfg, n_steps, name)
    wacc = params.pop("wacc", 0.0)
    capacity = _capacity(params, n_steps, wacc)

    if hasattr(cop, "__len__"):
        conversion = [1.0 / c if c > 0 else 0.0 for c in cop]
        available = [1.0 if c > 0 else 0.0 for c in cop]
    else:
        conversion = 1.0 / cop if cop > 0 else 0.0
        available = 1.0 if cop > 0 else 0.0

    return [
        solph.components.Converter(
            label=name,
            inputs={bus_in: solph.Flow()},
            outputs={
                bus_out: solph.Flow(nominal_capacity=capacity, maximum=available)
                if capacity is not None
                else solph.Flow()
            },
            # conversion_factors are given per input: electricity = heat / COP
            conversion_factors={bus_in: conversion},
        )
    ]


@factory("battery")
def _battery(name, params, buses, cfg, n_steps):
    """Electrical storage. `balanced=True` reproduces the periodic SOC wrap."""
    return _storage(name, params, buses, cfg, n_steps)


@factory("thermal_storage")
def _thermal_storage(name, params, buses, cfg, n_steps):
    """
    Hot water storage. The standby heat loss maps onto solph's
    `fixed_losses_absolute`, which is strictly more capable than the
    pre-migration `standby_loss_kW` (it also supports relative losses).
    """
    params.setdefault("fixed_losses_absolute", params.pop("standby_loss_kW", 0.0))
    return _storage(name, params, buses, cfg, n_steps)


def _storage(name, params, buses, cfg, n_steps):
    bus = _bus(buses, params, "bus", name)
    wacc = params.pop("wacc", 0.0)
    capacity = _capacity(params, n_steps, wacc)
    losses = _profile(params, "fixed_losses_absolute", cfg, n_steps, name, default=0.0)

    kwargs = dict(
        loss_rate=params.pop("self_discharge_per_h", 0.0),
        fixed_losses_absolute=losses,
        inflow_conversion_factor=params.pop("charge_efficiency", 0.95),
        outflow_conversion_factor=params.pop("discharge_efficiency", 0.95),
        min_storage_level=params.pop("soc_min", 0.0),
        max_storage_level=params.pop("soc_max", 1.0),
        balanced=params.pop("balanced", True),
        inputs={bus: solph.Flow(nominal_capacity=params.pop("charge_power_limit", None))},
        outputs={
            bus: solph.Flow(nominal_capacity=params.pop("discharge_power_limit", None))
        },
    )
    initial = params.pop("initial_soc", None)
    if initial is not None:
        kwargs["initial_storage_level"] = initial
        kwargs["balanced"] = False

    if isinstance(capacity, solph.Investment):
        kwargs["nominal_capacity"] = capacity
    else:
        kwargs["nominal_capacity"] = capacity

    return [solph.components.GenericStorage(label=name, **kwargs)]


@factory("zone5r1c")
def _zone5r1c(name, params, buses, cfg, n_steps):
    """The 5R1C thermal zone - the one custom component."""
    from .zone5r1c import ThermalZone5R1C

    heat_bus = _bus(buses, params, "heat_bus", name)
    cool_bus = None
    if params.get("cool_bus") is not None:
        cool_bus = _bus(buses, params, "cool_bus", name)
    else:
        params.pop("cool_bus", None)
    return [
        ThermalZone5R1C(name, cfg, heat_bus=heat_bus, cool_bus=cool_bus, **params)
    ]
