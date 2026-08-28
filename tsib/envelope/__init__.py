# -*- coding: utf-8 -*-
"""
Building physics of the thermal zone: the envelope as it is built, the
gains acting on it, and the comfort band it is held to.

This is what tsib knows about a building's thermal behaviour, expressed
without any simulation or optimization framework. `zone_parameters(cfg)`
condenses it into the ten scalars and five time series a 5R1C zone needs,
which is the contract handed to whatever model actually computes the heat
load - an external energy system model today, tsib's own forward model
once it exists.

    from tsib.envelope import zone_parameters

    params = zone_parameters(bdg.cfg)
    params["C_m"], params["H"]["Walls"], params["comfort_lb"]
"""

from .comfort import comfort_bounds
from .config import ThermalZoneConfig, calc_surface_irradiance
from .elements import ENVELOPE_ELEMENTS, existing_envelope
from .gains import (
    DESIGN_ADJUST,
    DESIGN_T_INDOOR,
    SOLAR_ELEMENTS,
    ZONE_SERIES,
    ZoneGains,
    zone_parameters,
)

__all__ = [
    "comfort_bounds",
    "ThermalZoneConfig",
    "calc_surface_irradiance",
    "ENVELOPE_ELEMENTS",
    "existing_envelope",
    "ZoneGains",
    "zone_parameters",
    "ZONE_SERIES",
    "SOLAR_ELEMENTS",
    "DESIGN_ADJUST",
    "DESIGN_T_INDOOR",
]
