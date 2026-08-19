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

from . import parameterization, presets
from .base import TsibBlock, TsibComponent, assert_constraint_groups
from .config import ThermalZoneConfig, calc_surface_irradiance
from .envelope import ENVELOPE_ELEMENTS, existing_envelope
from .investment import (
    ContinuousInvestment,
    InvestmentOption,
    annuity_factor,
)
from .registry import COMPONENT_FACTORIES, build_component, factory
from .results import flow_series, node_results, objective_value, solve
from .solverutils import detect_solver, manageSolverOpts
from .parameterization import (
    EQUIPMENT_BUILDERS,
    BuildingSystemParameters,
    build_spec,
    equipment,
)
from .spec import SystemSpec, build_system, required_inputs, resolve_profile
from .zone5r1c import (
    ThermalZone5R1C,
    ThermalZone5R1CBlock,
    comfort_bounds,
    zone_results,
)

__all__ = [
    "presets",
    "parameterization",
    "SystemSpec",
    "build_system",
    "resolve_profile",
    "required_inputs",
    "BuildingSystemParameters",
    "build_spec",
    "equipment",
    "EQUIPMENT_BUILDERS",
    "solve",
    "node_results",
    "flow_series",
    "objective_value",
    "ThermalZone5R1C",
    "ThermalZone5R1CBlock",
    "TsibComponent",
    "TsibBlock",
    "assert_constraint_groups",
    "ThermalZoneConfig",
    "zone_results",
    "comfort_bounds",
    "calc_surface_irradiance",
    "COMPONENT_FACTORIES",
    "build_component",
    "factory",
    "ENVELOPE_ELEMENTS",
    "existing_envelope",
    "InvestmentOption",
    "ContinuousInvestment",
    "annuity_factor",
    "detect_solver",
    "manageSolverOpts",
]
