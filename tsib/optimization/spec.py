# -*- coding: utf-8 -*-
"""
Declarative energy system specification and the builder which turns it into
an oemof-solph EnergySystem.

A `SystemSpec` is deliberately a plain, serializable description of the
*topology plus scalar parameters* - never of the time series. Profiles are
referenced by name (`"@elecLoad"`) and resolved against the building
configuration at build time. That boundary is what keeps a spec small
enough to store, diff and vary across thousands of parameterized buildings,
instead of dragging 8760-element arrays through the description.

There is exactly one build path: the preset functions in `presets.py` return
`SystemSpec` objects, so a preset is nothing but a named spec.
"""

import copy
import json

import numpy as np
import pandas as pd
from oemof import solph

from .base import assert_constraint_groups
from .registry import build_component

#: Prefix marking a value which is resolved from the building configuration.
PROFILE_PREFIX = "@"


class SystemSpec(object):
    """
    Topology and scalar parameters of an energy system.

    Parameters
    ----------
    buses: dict, required
        Bus name -> dict, currently only {"carrier": str} which is
        documentation for the reader; solph buses are carrier agnostic.
    components: dict, required
        Component name -> dict with a "type" key naming a factory in
        `registry.COMPONENT_FACTORIES` plus that factory's parameters.
        Bus-valued parameters hold bus *names*, profile-valued parameters
        either a number, a sequence, or a "@key" reference into the
        building configuration.
    """

    def __init__(self, buses=None, components=None):
        self.buses = dict(buses or {})
        self.components = dict(components or {})

    # --- authoring helpers ------------------------------------------

    def add_bus(self, name, carrier=None):
        if name in self.buses:
            raise ValueError("Duplicate bus '{}'".format(name))
        self.buses[name] = {"carrier": carrier}
        return name

    def add_component(self, name, type, **params):
        if name in self.components:
            raise ValueError("Duplicate component '{}'".format(name))
        self.components[name] = dict(params, type=type)
        return name

    def copy(self):
        return SystemSpec(copy.deepcopy(self.buses), copy.deepcopy(self.components))

    # --- serialization ------------------------------------------------

    def to_dict(self):
        return {"buses": copy.deepcopy(self.buses),
                "components": copy.deepcopy(self.components)}

    @classmethod
    def from_dict(cls, data):
        return cls(buses=data.get("buses"), components=data.get("components"))

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return "SystemSpec({} buses, {} components)".format(
            len(self.buses), len(self.components)
        )


def resolve_profile(value, cfg, n_steps, name=""):
    """
    Resolves a spec value into a number or a float array of length n_steps.

    A string starting with "@" is looked up in the building configuration;
    anything else is passed through (scalars stay scalars so solph can
    broadcast them itself).
    """
    if isinstance(value, str):
        if not value.startswith(PROFILE_PREFIX):
            raise ValueError(
                "String parameter '{}' for {} must be a configuration "
                "reference starting with '{}'".format(value, name, PROFILE_PREFIX)
            )
        key = value[len(PROFILE_PREFIX):]
        if key not in cfg or cfg[key] is None:
            raise KeyError(
                "Profile reference '{}' of {} is not in the building "
                "configuration".format(value, name)
            )
        value = cfg[key]

    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.values
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=float).ravel()
        if len(values) != n_steps:
            raise ValueError(
                "Profile of {} has {} values, expected {}".format(
                    name, len(values), n_steps
                )
            )
        return values
    return value


def build_system(spec, cfg, timeindex=None):
    """
    Builds a solph EnergySystem from a spec and a building configuration.

    Parameters
    ----------
    spec: SystemSpec or dict, required
    cfg: dict or ThermalZoneConfig, required
        Resolved building configuration; source of all profile references
        and of the 5R1C parameterization.
    timeindex: pandas.DatetimeIndex, optional
        Defaults to the weather index of the configuration.

    Returns
    -------
    (solph.EnergySystem, dict) - the system and a mapping of component name
    to the created solph node(s); components which expand into several nodes
    (a grid connection becomes a Source and a Sink) map to a list.
    """
    if isinstance(spec, dict):
        spec = SystemSpec.from_dict(spec)
    config = getattr(cfg, "cfg", cfg)

    if timeindex is None:
        timeindex = config["weather"].index
    n_steps = len(timeindex)

    # infer_last_interval=True is required: solph 0.6 defaults to False,
    # which would turn 8760 time stamps into 8759 intervals and silently
    # drop the last hour of the year.
    es = solph.EnergySystem(timeindex=timeindex, infer_last_interval=True)

    # step size in hours; investment costing is scaled by the horizon in
    # hours rather than in steps, so it stays correct below hourly resolution
    step_size_h = float(es.timeincrement[0])

    buses = {}
    for name in spec.buses:
        buses[name] = solph.Bus(label=name)
    es.add(*buses.values())

    nodes = {}
    for name, params in spec.components.items():
        params = dict(params)
        ctype = params.pop("type", None)
        if ctype is None:
            raise ValueError("Component '{}' has no 'type'".format(name))
        created = build_component(
            ctype,
            name,
            params,
            buses=buses,
            cfg=cfg,
            n_steps=n_steps,
            step_size_h=step_size_h,
        )
        es.add(*created)
        nodes[name] = created[0] if len(created) == 1 else created

    assert_constraint_groups(nodes)

    return es, nodes
