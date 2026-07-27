# -*- coding: utf-8 -*-
"""
Energy system optimization for tsib, built on oemof-solph.

tsib contributes two things to solph: the 5R1C thermal zone - the one piece
of building physics solph has no equivalent for - and a building block kit
that assembles parameterized buildings into ready-to-solve energy systems.

The design rule behind the zone being a component rather than a precomputed
load profile: its temperature states are free decision variables inside a
comfort band, so the building's thermal mass stays available as a
dispatchable flexibility resource in the same solve as storage, PV and
prices.

    import tsib
    from tsib.optimization import presets, build_system, solve, node_results

    spec = presets.heat_load_only()
    es, nodes = build_system(spec, bdg.cfg)
    model, _ = solve(es)
    results = node_results(model, nodes)["thermalzone"]
"""

from . import presets
from .config import ThermalZoneConfig, calc_surface_irradiance
from .control import ComfortControl
from .envelope import ENVELOPE_ELEMENTS, load_control_options, load_envelope_options
from .investment import (
    ContinuousInvestment,
    DiscreteOptionInvestment,
    InvestmentOption,
    annuity_factor,
)
from .registry import COMPONENT_FACTORIES, build_component, factory
from .results import flow_series, node_results, objective_value, solve
from .solverutils import detect_solver, manageSolverOpts
from .spec import SystemSpec, build_system, resolve_profile
from .zone5r1c import ThermalZone5R1C, ThermalZone5R1CBlock, zone_results

__all__ = [
    "presets",
    "SystemSpec",
    "build_system",
    "resolve_profile",
    "solve",
    "node_results",
    "flow_series",
    "objective_value",
    "ThermalZone5R1C",
    "ThermalZone5R1CBlock",
    "ThermalZoneConfig",
    "zone_results",
    "calc_surface_irradiance",
    "COMPONENT_FACTORIES",
    "build_component",
    "factory",
    "ComfortControl",
    "ENVELOPE_ELEMENTS",
    "load_envelope_options",
    "load_control_options",
    "InvestmentOption",
    "ContinuousInvestment",
    "DiscreteOptionInvestment",
    "annuity_factor",
    "detect_solver",
    "manageSolverOpts",
]
