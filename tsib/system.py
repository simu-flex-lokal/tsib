# -*- coding: utf-8 -*-
"""
Per-building equipment sheet and the template which turns it into a spec.

tsib has three separate descriptions of a building and they answer three
different questions:

===================================  ==================================
`BuildingConfiguration`              what is *true* of the building
`BuildingSystemParameters` (here)    what the building *has*
the spec (`build_spec`)              how it is *wired*
===================================  ==================================

The middle one is the exchange format. Topology is never exchanged: every
engine keeps its own predefined building network and only the equipment
sheet plus the input time series travel between them. That is what makes a
tsib building parameterizable at grid scale - hand-authoring a spec per
building does not survive a few hundred of them - and what keeps the
description meaningful to a partner model whose components and connections
look nothing like the one tsib assumes.

The spec is a plain dictionary and nothing here solves anything: tsib
states what the building is made of and hands over the time series, the
energy system model lives elsewhere.

    params = BuildingSystemParameters(
        equipment={"heat_pump": {"capacity_kw": 8.0}, "pv": {"kwp": 8.0}},
        tariff={"import": "@elecPrice", "export": 0.08},
    )
    spec = build_spec(params, bdg.zone_parameters())
    inputs = bdg.system_inputs(spec)

New equipment is added by registering one `@equipment` builder; neither the
exchange format nor the template knows it happened.
"""

import copy
import json

from .envelope.gains import ZONE_SERIES

#: version of the spec schema emitted by `build_spec`
SPEC_VERSION = "1"

#: default energy prices [EUR/kWh] carried over from the 5R1C heat load run
DEFAULT_HEAT_COST = 0.08
DEFAULT_COOL_COST = 0.02

#: fallback electricity tariff [EUR/kWh] when the building configuration
#: carries no `elecPrice` profile of its own
DEFAULT_ELEC_PRICE = 0.35

# ---------------------------------------------------------------------
# the spec: plain data, no framework
# ---------------------------------------------------------------------


class SpecBuilder(object):
    """
    Collects buses and components into the spec dictionary.

    Only `build_spec` uses this; it exists so the equipment builders can be
    written as `spec.add_component(...)` instead of assembling nested dicts
    by hand. The product is a plain dict - topology plus scalars, never a
    time series - which any energy system model can read.
    """

    def __init__(self):
        self.buses = {}
        self.components = {}

    def add_bus(self, name, carrier=None):
        self.buses[name] = {"carrier": carrier}

    def add_component(self, name, ctype, **params):
        if name in self.components:
            raise ValueError("Duplicate component '{}'".format(name))
        self.components[name] = dict(params, type=ctype)

    def to_dict(self):
        return {
            "version": SPEC_VERSION,
            "buses": copy.deepcopy(self.buses),
            "components": copy.deepcopy(self.components),
        }


def capacity_params(value):
    """
    A number becomes a fixed capacity, a dict an investment decision.

    The same choice is offered per equipment entry of the sheet.
    """
    if isinstance(value, dict):
        return dict(value)
    return {"capacity": value}


def required_inputs(spec):
    """
    The input keys a spec refers to, i.e. its data contract.

    Any parameter given as a string starting with "@" is a named reference
    to a time series which travels separately from the spec; this collects
    them so a caller knows what has to be supplied.

    Parameters
    ----------
    spec: dict, required
        As returned by `build_spec`.

    Returns
    -------
    sorted list of keys, without the "@".
    """
    keys = set()
    for params in spec.get("components", {}).values():
        for value in params.values():
            if isinstance(value, str) and value.startswith("@"):
                keys.add(value[1:])
    return sorted(keys)


#: buses of the template. One heat bus carries both space heating and hot
#: water: a deliberate simplification, so the two are not distinguishable by
#: temperature level here (see docs/energysystem.md).
ELEC_BUS = "elec"
HEAT_BUS = "heat"
COOL_BUS = "cool"

#: interest rate applied to equipment given as an investment decision
DEFAULT_WACC = 0.06

#: registered equipment builders, filled by @equipment below
EQUIPMENT_BUILDERS = {}


def equipment(name, supplies_heat=False):
    """
    Registers an equipment builder under its key in the sheet.

    Parameters
    ----------
    name: str, required
        Key used in `BuildingSystemParameters.equipment`.
    supplies_heat: bool, optional (default: False)
        Whether the equipment can serve the heat bus. A building whose sheet
        contains none is not electrically heated, and the template gives it
        a non-electric heat source instead of an infeasible model.
    """

    def register(func):
        if name in EQUIPMENT_BUILDERS:
            raise ValueError("Duplicate equipment builder '{}'".format(name))
        func.supplies_heat = supplies_heat
        EQUIPMENT_BUILDERS[name] = func
        return func

    return register


class BuildingSystemParameters(object):
    """
    What one building has, as plain serializable data.

    Parameters
    ----------
    equipment: dict, optional
        Equipment key -> parameters. A key which is absent or maps to None
        is not built. Every key must be registered via `@equipment`; an
        unknown one is an error rather than a silently ignored entry.
    tariff: dict, optional
        "import"/"export" electricity prices and the "heat" price used when
        the building is not electrically heated. Each is a number or a
        "@key" reference, like any other spec value.
    zone: dict, optional
        Overrides of the zone parameters (max_load, initial_T_m). The
        zone's physics and comfort band come from the building
        configuration, via `tsib.envelope.zone_parameters`.
    meta: dict, optional
        Descriptive only, never built and never part of the building's
        cache key: the grid connection point ("bus_id"), the intended time
        step ("freq"), provenance such as the "kwp" behind a PV profile.
    """

    def __init__(self, equipment=None, tariff=None, zone=None, meta=None):
        self.equipment = {
            key: value for key, value in dict(equipment or {}).items()
            if value is not None
        }
        self.tariff = dict(tariff or {})
        self.zone = dict(zone or {})
        self.meta = dict(meta or {})

        unknown = sorted(set(self.equipment) - set(EQUIPMENT_BUILDERS))
        if unknown:
            raise ValueError(
                "Unknown equipment {} - registered: {}".format(
                    unknown, sorted(EQUIPMENT_BUILDERS)
                )
            )

    # --- serialization ------------------------------------------------

    def to_dict(self):
        return {
            "equipment": copy.deepcopy(self.equipment),
            "tariff": copy.deepcopy(self.tariff),
            "zone": copy.deepcopy(self.zone),
            "meta": copy.deepcopy(self.meta),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            equipment=data.get("equipment"),
            tariff=data.get("tariff"),
            zone=data.get("zone"),
            meta=data.get("meta"),
        )

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    # --- convenience ---------------------------------------------------

    def build_spec(self, zone_params=None):
        return build_spec(self, zone_params)

    def required_inputs(self):
        """Input keys this building's system will demand."""
        return required_inputs(build_spec(self))

    def check_index(self, index):
        """
        Verifies a time index against the sheet's intended resolution.

        `meta["freq"]` records what the parameters were meant for, so that
        mixing an hourly building into a 15 minute study is caught rather
        than silently resampled.
        """
        freq = self.meta.get("freq")
        if freq is None:
            return True
        actual = getattr(index, "freqstr", None) or getattr(index, "freq", None)
        if actual is not None and str(actual) != str(freq):
            raise ValueError(
                "Parameters are given for '{}' but the time index is "
                "'{}'".format(freq, actual)
            )
        return True

    def __repr__(self):
        return "BuildingSystemParameters({})".format(
            ", ".join(sorted(self.equipment)) or "no equipment"
        )


# ---------------------------------------------------------------------
# equipment builders
# ---------------------------------------------------------------------


def _reject_unknown(name, params):
    if params:
        raise ValueError(
            "Unknown parameter(s) {} for equipment '{}'".format(
                sorted(params), name
            )
        )


@equipment("heat_pump", supplies_heat=True)
def _heat_pump(name, params, spec):
    """Electric heat generator on the shared heat bus."""
    params = dict(params)
    capacity = capacity_params(params.pop("capacity_kw", None))
    cop = params.pop("cop", "@cop")
    wacc = params.pop("wacc", DEFAULT_WACC)
    _reject_unknown(name, params)
    spec.add_component(
        name, "heat_pump", bus_in=ELEC_BUS, bus_out=HEAT_BUS, cop=cop,
        wacc=wacc, **capacity
    )


@equipment("pv")
def _pv(name, params, spec):
    """
    Roof PV, given either as an absolute generation profile [kW] or as a
    peak power [kWp] against a specific yield profile [kW/kWp].

    The absolute profile is the form exchanged with other models, because
    then no second engine recomputes the yield from the same kWp and
    arrives somewhere else. Given a profile, "kwp" is provenance only and
    does not scale it a second time.
    """
    params = dict(params)
    profile = params.pop("profile", None)
    kwp = params.pop("kwp", None)
    specific_yield = params.pop("specific_yield", "@pv_yield")
    wacc = params.pop("wacc", DEFAULT_WACC)
    curtailable = params.pop("curtailable", True)
    _reject_unknown(name, params)

    if profile is not None:
        yield_profile, capacity = profile, capacity_params(1.0)
    elif kwp is not None:
        yield_profile, capacity = specific_yield, capacity_params(kwp)
    else:
        raise ValueError(
            "Equipment 'pv' needs either a 'profile' [kW] or a 'kwp'"
        )

    spec.add_component(
        name, "pv", bus=ELEC_BUS, specific_yield=yield_profile,
        curtailable=curtailable, wacc=wacc, **capacity
    )


@equipment("battery")
def _battery(name, params, spec):
    """Electrical storage behind the meter."""
    spec.add_component(
        name, "battery", bus_in=ELEC_BUS, bus_out=ELEC_BUS,
        **_storage_params(name, params, "capacity_kwh", "power_kw")
    )


@equipment("buffer")
def _buffer(name, params, spec):
    """
    Hot water buffer on the heat bus. It serves both space heating and hot
    water, because the template does not separate them.
    """
    params = dict(params)
    standby = params.pop("standby_loss_kW", None)
    built = _storage_params(name, params, "capacity_kwh", "power_kw")
    if standby is not None:
        built["standby_loss_kW"] = standby
    spec.add_component(
        name, "thermal_storage", bus_in=HEAT_BUS, bus_out=HEAT_BUS, **built
    )


def _storage_params(name, params, capacity_key, power_key):
    """
    Shared translation of a storage entry onto the `_storage` factory.

    `roundtrip_efficiency` is accepted because that is how storage is
    usually specified outside tsib; it is split evenly over the two
    directions, which is the convention that makes the round trip come out
    right.
    """
    params = dict(params)
    built = capacity_params(params.pop(capacity_key, None))

    power = params.pop(power_key, None)
    if power is not None:
        built["charge_power_limit"] = power
        built["discharge_power_limit"] = power

    roundtrip = params.pop("roundtrip_efficiency", None)
    if roundtrip is not None:
        one_way = float(roundtrip) ** 0.5
        built["charge_efficiency"] = one_way
        built["discharge_efficiency"] = one_way

    for key in ("charge_efficiency", "discharge_efficiency",
                "self_discharge_per_h", "soc_min", "soc_max",
                "initial_soc", "balanced", "wacc"):
        if key in params:
            built[key] = params.pop(key)

    built.setdefault("wacc", DEFAULT_WACC)
    _reject_unknown(name, params)
    return built


# ---------------------------------------------------------------------
# the template
# ---------------------------------------------------------------------


def build_spec(params, zone_params=None):
    """
    Composes the tsib building template for one equipment sheet.

    Every building shares this network and differs only in which equipment
    is present and how it is sized:

        grid ─── elec ──┬── household_load
                        ├── pv, battery          (if present)
                        └── heat_pump            (if present)
                                │
                    heat ───────┼── buffer       (if present)
                                ├── dhw_load
                                └── thermalzone ─── cool

    Parameters
    ----------
    params: BuildingSystemParameters or dict, required
    zone_params: dict, optional
        The building's 5R1C parameters, i.e. `tsib.envelope.zone_parameters`
        or `Building.zone_parameters()`. Without them the zone component
        names its inputs but carries no physics: such a spec is complete
        enough to ask what it needs (`required_inputs`), not to be built.

    Returns
    -------
    dict - the spec: version, buses, components. Plain data throughout, so
    it can be stored as JSON and handed to an energy system model.
    """
    if not isinstance(params, BuildingSystemParameters):
        params = BuildingSystemParameters.from_dict(params)

    spec = SpecBuilder()
    spec.add_bus(ELEC_BUS, carrier="electricity")
    spec.add_bus(HEAT_BUS, carrier="heat")
    spec.add_bus(COOL_BUS, carrier="cool")

    grid = {"bus": ELEC_BUS,
            "import_price": params.tariff.get("import", "@elecPrice")}
    export = params.tariff.get("export")
    if export is not None:
        grid["export_price"] = export
    spec.add_component("grid", "grid", **grid)

    spec.add_component("household_load", "demand", bus=ELEC_BUS,
                       profile="@elecLoad")
    spec.add_component("dhw_load", "demand", bus=HEAT_BUS,
                       profile="@hotWaterLoad")
    # The comfort ceiling is a hard bound by default, so the zone needs a way
    # to shed heat or a summer with any solar gain is infeasible. A sheet may
    # override this with can_cool=False plus comfort_ub_penalty, which lets
    # the zone drift above the band and pay per Kelvin-hour instead.
    spec.add_component("cool_supply", "source", bus=COOL_BUS,
                       price=DEFAULT_COOL_COST)

    electrically_heated = False
    for name in sorted(params.equipment):
        builder = EQUIPMENT_BUILDERS[name]
        builder(name, params.equipment[name], spec)
        electrically_heated |= builder.supplies_heat

    if not electrically_heated:
        # a gas or district heated building still belongs in the study: it
        # contributes household load, PV and battery to the grid, and its
        # heat comes from outside the electricity system
        spec.add_component(
            "heat_supply", "source", bus=HEAT_BUS,
            price=params.tariff.get("heat", DEFAULT_HEAT_COST)
        )

    # the zone's ten scalars are written into the spec, its five time
    # series are referenced by name like every other profile
    zone = {"heat_bus": HEAT_BUS, "cool_bus": COOL_BUS}
    zone.update({key: "@" + key for key in ZONE_SERIES})
    if zone_params is not None:
        zone.update({key: value for key, value in zone_params.items()
                     if key not in ZONE_SERIES})
    zone.update(params.zone)
    spec.add_component("thermalzone", "zone5r1c", **zone)

    return spec.to_dict()
